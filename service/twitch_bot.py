# twitch_bot.py
import asyncio
import json
import os
import threading
import time
from pathlib import Path
import sys
from typing import Optional

import requests
import traceback
from twitchio.ext import commands

def _data_root() -> Path:
    """Return a stable per-user data directory for the app.
    Windows: %LOCALAPPDATA%/autoaffili
    macOS: ~/Library/Application Support/autoaffili
    Linux: ~/.config/autoaffili (respects XDG_CONFIG_HOME)
    """
    try:
        if os.name == "nt":
            base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(base) / "autoaffili"
        elif sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "autoaffili"
        else:
            xdg = os.getenv("XDG_CONFIG_HOME")
            cfg = Path(xdg) if xdg else (Path.home() / ".config")
            return cfg / "autoaffili"
    except Exception:
        return Path.cwd() / "autoaffili"

APP_DIR = _data_root()
try:
    APP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

TOKEN_FILE = APP_DIR / "twitch_tokens.json"

# ---- simple file logger (useful when packaged, no console) ----
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "chat.log"
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    pass

_log_lock = threading.Lock()

def _log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} {msg}\n"
    with _log_lock:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

# Ensure the log file is created as soon as the module loads
_log("module loaded")

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
DEFAULT_CLIENT_ID = "b7sci9y497dfx11fox4ck33ryhdmqe"
TOKEN_REFRESH_GRACE = 120  # seconds before expiry to trigger refresh
REFRESH_CHECK_INTERVAL = 300  # seconds between background refresh checks


_token_cache = {
    "token": None,
    "expires_at": None,
    "refresh_token": None,
    "raw": None,
}
_channel_cache = {"channel": None}
_token_lock = threading.Lock()
_refresh_thread = None

bot = None
_bot_thread = None
_bot_ready = threading.Event()

# De-duplication of recent messages to avoid Twitch duplicate suppression
_dedupe_lock = threading.Lock()
_last_msg_text: Optional[str] = None
_last_msg_ts: float = 0.0
_DEDUP_SECONDS = float(os.getenv("TWITCH_DEDUP_SECONDS", "90"))


def _format_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = str(token)
    if not token.startswith("oauth:"):
        token = f"oauth:{token}"
    return token


def _read_token_file() -> Optional[dict]:
    if not TOKEN_FILE.exists():
        return None
    try:
        return json.loads(TOKEN_FILE.read_text())
    except Exception as exc:
        print("[twitch_bot] failed to read token file:", exc)
        return None


def _write_token_file(data: dict) -> None:
    try:
        TOKEN_FILE.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        print("[twitch_bot] failed to write token file:", exc)


def _stamp_token_metadata(payload: dict) -> dict:
    stamped = dict(payload or {})
    now = int(time.time())
    stamped["obtained_at"] = now
    expires_in = stamped.get("expires_in")
    if isinstance(expires_in, (int, float)):
        stamped["expires_at"] = int(now + float(expires_in))
    else:
        stamped.pop("expires_at", None)
    return stamped


def _calculate_expires_at(payload: dict) -> Optional[float]:
    expires_at = payload.get("expires_at")
    if isinstance(expires_at, (int, float)):
        return float(expires_at)
    expires_in = payload.get("expires_in")
    obtained_at = payload.get("obtained_at")
    if isinstance(expires_in, (int, float)) and isinstance(obtained_at, (int, float)):
        return float(obtained_at) + float(expires_in)
    return None


def _should_refresh(payload: dict) -> bool:
    expires_at = _calculate_expires_at(payload)
    if not expires_at:
        # Legacy token files may lack timestamps; refresh once to stamp metadata.
        return True
    return time.time() >= expires_at - TOKEN_REFRESH_GRACE

def _refresh_token_via_api(payload: dict) -> Optional[dict]:
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        print("[twitch_bot] cannot refresh token: refresh_token missing")
        return None

    client_id = os.getenv("TWITCH_CLIENT_ID", DEFAULT_CLIENT_ID)
    if not client_id or client_id == "TWITCH_CLIENT_ID_PLACEHOLDER":
        print("[twitch_bot] cannot refresh token: TWITCH_CLIENT_ID is not configured")
        return None

    data = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post(TOKEN_URL, data=data, timeout=10)
    except Exception as exc:
        print("[twitch_bot] refresh request failed:", exc)
        return None

    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        print("[twitch_bot] refresh request returned non-200:", body)
        return None

    try:
        new_payload = resp.json()
    except Exception as exc:
        print("[twitch_bot] failed to parse refresh response:", exc)
        return None

    if not isinstance(new_payload, dict):
        return None

    new_payload.setdefault("refresh_token", refresh_token)
    if payload.get("channel_login"):
        new_payload.setdefault("channel_login", payload.get("channel_login"))
    new_payload = _stamp_token_metadata(new_payload)
    _write_token_file(new_payload)
    print("[twitch_bot] refreshed Twitch access token automatically.")
    return new_payload


def _maybe_refresh_token(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    if not _should_refresh(payload):
        return payload
    refreshed = _refresh_token_via_api(payload)
    return refreshed or payload


def _load_access_token_from_file(force_reload: bool = False) -> Optional[str]:
    with _token_lock:
        data = None
        if not force_reload:
            data = _token_cache.get("raw")
        if data is None:
            data = _read_token_file()
        if data is None:
            _token_cache.update({"raw": None, "token": None, "expires_at": None})
            return None

        data = _maybe_refresh_token(data)
        if data is None:
            _token_cache.update({"raw": None, "token": None, "expires_at": None})
            return None

        _token_cache["raw"] = data
        access_token = data.get("access_token")
        if not access_token:
            _token_cache.update({"token": None, "expires_at": None})
            return None

        formatted = _format_token(access_token)
        _token_cache["token"] = formatted
        _token_cache["expires_at"] = _calculate_expires_at(data)
        _token_cache["refresh_token"] = data.get("refresh_token")
        if data.get("channel_login"):
            _channel_cache["channel"] = data["channel_login"]

        if bot is not None and formatted:
            try:
                bot._http.token = formatted.split(":", 1)[-1]
            except Exception:
                pass

        return formatted


def _get_token(force: bool = False) -> Optional[str]:
    token_env = os.getenv("TWITCH_ACCESS_TOKEN")
    if token_env:
        return _format_token(token_env)

    with _token_lock:
        cached = _token_cache.get("token")
        expires_at = _token_cache.get("expires_at")
    if not force and cached and expires_at and time.time() < expires_at - TOKEN_REFRESH_GRACE:
        return cached

    return _load_access_token_from_file(force_reload=force)


def _load_channel_login() -> Optional[str]:
    channel = os.getenv("TWITCH_CHANNEL")
    if channel:
        return channel

    _get_token()
    with _token_lock:
        cached = _channel_cache.get("channel")
        if cached:
            return cached
        raw = _token_cache.get("raw")
    if raw and raw.get("channel_login"):
        with _token_lock:
            _channel_cache["channel"] = raw["channel_login"]
        return raw["channel_login"]

    data = _read_token_file()
    if data and data.get("channel_login"):
        with _token_lock:
            _token_cache["raw"] = data
            _channel_cache["channel"] = data["channel_login"]
        return data["channel_login"]

    return None


def _refresh_monitor_loop() -> None:
    while True:
        time.sleep(REFRESH_CHECK_INTERVAL)
        try:
            _get_token(force=True)
        except Exception as exc:
            print("[twitch_bot] refresh monitor error:", exc)


def _ensure_refresh_monitor_started() -> None:
    global _refresh_thread
    if _refresh_thread and _refresh_thread.is_alive():
        return
    _refresh_thread = threading.Thread(target=_refresh_monitor_loop, daemon=True)
    _refresh_thread.start()


def _bot_thread_main(token: str, channel: str) -> None:
    global bot
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            bot_local = commands.Bot(token=token, prefix="!", initial_channels=[channel])
        except Exception as exc:
            _log(f"failed to construct bot: {repr(exc)}")
            return
        bot = bot_local
        _log("bot constructed")
        try:
            bot_local.run()
        except Exception as exc:
            _log(f"bot.run() error: {repr(exc)}")
    finally:
        _bot_ready.clear()


def _ensure_bot_started() -> None:
    global bot, _bot_thread
    token = _get_token()
    channel = _load_channel_login()
    if not token or not channel:
        print("[twitch_bot] missing token or channel; cannot start bot")
        try:
            _log("missing token or channel; cannot start bot")
        except Exception:
            pass
        return

    if bot is None or _bot_thread is None or not _bot_thread.is_alive():
        # Start dedicated bot thread which owns its asyncio loop and constructs the bot
        _bot_ready.clear()
        try:
            _bot_thread = threading.Thread(target=_bot_thread_main, args=(token, channel), daemon=True)
            _bot_thread.start()
            _log("bot thread started")
        except Exception as exc:
            print("[twitch_bot] failed to start bot thread:", repr(exc))
            try:
                _log(f"failed to start bot thread: {repr(exc)}")
            except Exception:
                pass
            return

        # best-effort wait for bot.loop appearing
        for _ in range(50):
            if getattr(bot, "loop", None) is not None:
                _bot_ready.set()
                break
            time.sleep(0.1)
        print("[twitch_bot] bot started")
        _log("bot started")

    _ensure_refresh_monitor_started()


def send_link_to_chat(short_link, keywords):
    _ensure_bot_started()
    # Wait briefly for bot.loop to be ready
    if bot is None or getattr(bot, "loop", None) is None:
        last_warn = 0.0
        deadline = time.time() + 6.0
        while time.time() < deadline and (bot is None or getattr(bot, "loop", None) is None):
            if time.time() - last_warn > 1.5:
                try:
                    _log("waiting for bot loop...")
                except Exception:
                    pass
                last_warn = time.time()
            time.sleep(0.2)
    if bot is None or getattr(bot, "loop", None) is None:
        print("[twitch_bot] bot not available, cannot send")
        try:
            _log("bot not available, cannot send")
        except Exception:
            pass
        return

    channel = _load_channel_login()
    if not channel:
        print("[twitch_bot] missing channel; cannot send message")
        return

    # Message format: "<keywords>? <link>"
    message = f"{keywords}? {short_link}"

    # De-dup: skip if identical message sent within the recent window
    now = time.time()
    with _dedupe_lock:
        global _last_msg_text, _last_msg_ts
        if _last_msg_text == message and (now - _last_msg_ts) < _DEDUP_SECONDS:
            wait_left = int(_DEDUP_SECONDS - (now - _last_msg_ts))
            note = f"skip duplicate message (last sent {int(now - _last_msg_ts)}s ago, wait ~{wait_left}s)"
            print(f"[twitch_bot] {note}")
            _log(note)
            return
        _last_msg_text = message
        _last_msg_ts = now

    async def _send():
        import time as _t
        deadline = _t.time() + 12.0
        last_warn = 0.0
        while _t.time() < deadline:
            try:
                chan = bot.get_channel(channel)
                if chan:
                    await chan.send(message)
                    print(f"[twitch_bot] sent to #{channel}: {message}")
                    _log(f"sent to #{channel}: {message}")
                    return
                for c in getattr(bot, 'connected_channels', []) or []:
                    if getattr(c, 'name', None) == channel:
                        await c.send(message)
                        print(f"[twitch_bot] sent to #{channel}: {message}")
                        _log(f"sent to #{channel}: {message}")
                        return
                if _t.time() - last_warn > 3.0:
                    print("[twitch_bot] waiting for channel connection...")
                    _log("waiting for channel connection...")
                    last_warn = _t.time()
                await asyncio.sleep(0.5)
            except Exception as exc:
                print("[twitch_bot] send error:", exc)
                _log(f"send error: {exc!r}")
                await asyncio.sleep(0.5)
        print("[twitch_bot] give up sending: channel not connected in time")
        _log("give up sending: channel not connected in time")

    try:
        print(f"[twitch_bot] scheduling send to #{channel}...")
        _log(f"scheduling send to #{channel}: {message}")
        import concurrent.futures

        done: "concurrent.futures.Future[bool]" = concurrent.futures.Future()

        async def _runner():
            try:
                await _send()
                if not done.done():
                    done.set_result(True)
            except Exception as e:  # noqa: BLE001
                print("[twitch_bot] send error (runner):", repr(e))
                try:
                    traceback.print_exc()
                except Exception:  # noqa: BLE001
                    pass
                if not done.done():
                    done.set_exception(e)

        # Schedule on TwitchIO's event loop thread and wait for completion here
        bl = getattr(bot, "loop", None)
        if bl is None:
            raise RuntimeError("TwitchIO bot loop not ready")
        bl.call_soon_threadsafe(lambda: asyncio.create_task(_runner()))
        done.result(timeout=20)
    except Exception as exc:  # noqa: BLE001
        print("[twitch_bot] send error (thread-safe):", repr(exc))
        _log(f"send error (thread-safe): {repr(exc)}")
        try:
            traceback.print_exc()
        except Exception:  # noqa: BLE001
            pass


