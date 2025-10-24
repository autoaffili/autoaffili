"""Twitch OAuth companion tool for Autoaffili (Device Code flow)."""

from __future__ import annotations

import argparse
import json
import os
import time
import webbrowser
import threading
from pathlib import Path
import sys
from typing import Optional

import requests

DEVICE_CODE_URL = "https://id.twitch.tv/oauth2/device"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
SCOPES = ["chat:read", "chat:edit"]
DEFAULT_CLIENT_ID = "b7sci9y497dfx11fox4ck33ryhdmqe"


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


def _stamp_token_payload(payload: dict, *, force: bool = False) -> dict:
    stamped = dict(payload or {})
    if not force and 'obtained_at' in stamped and 'expires_at' in stamped:
        return stamped
    now = int(time.time())
    stamped['obtained_at'] = now
    expires_in = stamped.get('expires_in')
    if isinstance(expires_in, (int, float)):
        stamped['expires_at'] = int(now + float(expires_in))
    else:
        stamped.pop('expires_at', None)
    return stamped


def _calculate_expires_at(payload: dict) -> Optional[float]:
    expires_at = payload.get('expires_at')
    if isinstance(expires_at, (int, float)):
        return float(expires_at)
    expires_in = payload.get('expires_in')
    obtained_at = payload.get('obtained_at')
    if isinstance(expires_in, (int, float)) and isinstance(obtained_at, (int, float)):
        return float(obtained_at) + float(expires_in)
    return None


def _token_is_near_expiry(payload: dict, grace: float = 120.0) -> bool:
    expires_at = _calculate_expires_at(payload)
    if not expires_at:
        return False
    return time.time() >= expires_at - grace


def _verify_access_token(payload: dict, client_id: str) -> bool:
    access_token = payload.get("access_token")
    if not access_token:
        return False
    try:
        login = fetch_user_login(access_token, client_id)
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (400, 401, 403):
            return False
        print("[twitch_oauth_tool] warning: token verification error:", exc)
        return True
    except Exception as exc:
        print("[twitch_oauth_tool] warning: token verification exception:", exc)
        return True
    if login:
        payload.setdefault("channel_login", login)
    return True


def _attempt_refresh(client_id: str, payload: dict) -> Optional[dict]:
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        return None
    try:
        new_payload = refresh_access_token(client_id, refresh_token)
    except Exception as exc:
        print("[twitch_oauth_tool] refresh attempt failed:", exc)
        return None
    new_payload.setdefault("refresh_token", refresh_token)
    if payload.get("channel_login"):
        new_payload.setdefault("channel_login", payload.get("channel_login"))
    new_payload = _stamp_token_payload(new_payload, force=True)
    save_tokens(new_payload)
    return new_payload


def request_device_code(client_id: str) -> dict:
    data = {
        "client_id": client_id,
        "scope": " ".join(SCOPES),
    }
    response = requests.post(DEVICE_CODE_URL, data=data, timeout=10)
    if not response.ok:
        print("[twitch_oauth_tool] device code request failed:", response.text)
        response.raise_for_status()
    payload = response.json()
    required_keys = {"device_code", "user_code", "verification_uri"}
    if not required_keys.issubset(payload):
        raise RuntimeError("Device code response missing required keys")
    return payload


def poll_device_authorization(client_id: str, device_code: str, interval: int, expires_in: int, status_callback=None) -> dict:
    poll_interval = max(1, int(interval or 0))
    deadline = time.time() + max(1, int(expires_in or 0))
    status_cooldown = 0.0

    while True:
        if time.time() >= deadline:
            raise RuntimeError("Device code expired before authorization was granted")

        data = {
            "client_id": client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        response = requests.post(TOKEN_URL, data=data, timeout=10)

        try:
            body = response.json()
        except Exception:
            body = {}

        if response.ok:
            return body

        raw_error = ""
        error = ""
        if isinstance(body, dict):
            raw_error = body.get("error") or body.get("message") or body.get("error_description") or ""
            error = str(raw_error).lower()
        else:
            raw_error = response.text

        if error == "authorization_pending":
            remaining = max(0, int(deadline - time.time()))
            if time.time() >= status_cooldown:
                msg = f"waiting for approval... ~{remaining}s remaining"
                if status_callback:
                    status_callback(msg)
                else:
                    print(f"[twitch_oauth_tool] {msg}")
                status_cooldown = time.time() + max(5, poll_interval)
            time.sleep(poll_interval)
            continue
        if error == "slow_down":
            poll_interval += 1
            msg = f"slow_down received, polling every {poll_interval}s"
            if status_callback:
                status_callback(msg)
            else:
                print(f"[twitch_oauth_tool] {msg}")
            time.sleep(poll_interval)
            continue
        if error in {"authorization_declined", "access_denied"}:
            raise RuntimeError("Authorization was declined in the Twitch browser flow")
        if error == "expired_token":
            raise RuntimeError("Device code expired before authorization was granted")

        failure_msg = raw_error or response.status_code
        if status_callback:
            status_callback(f"token polling failed: {failure_msg}")
        else:
            print("[twitch_oauth_tool] token polling failed:", body or response.text)
        response.raise_for_status()


def refresh_access_token(client_id: str, refresh_token: str) -> dict:
    data = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    response = requests.post(TOKEN_URL, data=data, timeout=10)
    if not response.ok:
        print("[twitch_oauth_tool] token refresh failed:", response.text)
        response.raise_for_status()
    return response.json()


def fetch_user_login(access_token: str, client_id: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Client-ID": client_id,
    }
    resp = requests.get("https://api.twitch.tv/helix/users", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if data:
        return data[0].get("login")
    return None


def save_tokens(payload: dict) -> Path:
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    TOKEN_FILE.write_text(json.dumps(payload, indent=2))
    return TOKEN_FILE


def load_tokens() -> dict:
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"Token file not found: {TOKEN_FILE}")
    return json.loads(TOKEN_FILE.read_text())


def _notify_backend_auth(channel_login: str, obtained_at: int | None = None, twitch_user_id: str | None = None) -> None:
    """
    Notify the backend that a user has authenticated.
    BACKEND_AUTH_NOTIFY can override the default endpoint.
    """
    backend_url = os.getenv(
        "BACKEND_AUTH_NOTIFY",
        "https://autoaffili-backend-802674334607.us-east4.run.app/auth_notify",
    )
    payload = {
        "channel_login": channel_login,
        "obtained_at": obtained_at,
        "twitch_user_id": twitch_user_id,
    }
    try:
        requests.post(backend_url, json=payload, timeout=5)
        print(f"[twitch_oauth_tool] Notified backend at {backend_url}")
    except Exception as e:
        print(f"[twitch_oauth_tool] Warning: failed to notify backend: {e}")


def run_interactive_flow(host: str, port: int) -> None:
    del host, port
    client_id = os.getenv("TWITCH_CLIENT_ID", DEFAULT_CLIENT_ID)
    if not client_id or client_id == "TWITCH_CLIENT_ID_PLACEHOLDER":
        raise RuntimeError("TWITCH_CLIENT_ID must be set to your registered Twitch app client ID")

    device_payload = request_device_code(client_id)
    user_code = device_payload["user_code"]
    verification_uri = device_payload["verification_uri"]
    verification_complete = device_payload.get("verification_uri_complete")
    expires_in = int(device_payload.get("expires_in", 0))
    interval = int(device_payload.get("interval", 5))

    expires_msg = f" (expires in {expires_in // 60}m {expires_in % 60}s)" if expires_in else ""
    if verification_complete:
        print(f"1. Open: {verification_complete}{expires_msg}")
    else:
        print(f"1. Open: {verification_uri}{expires_msg}")
    print(f"2. Enter the code: {user_code}")
    print("3. Approve Autoaffili to post chat messages.")

    if verification_complete:
        try:
            webbrowser.open(verification_complete, new=1)
            print("[twitch_oauth_tool] Attempted to open the Twitch verification page in your default browser.")
        except Exception as exc:
            print("[twitch_oauth_tool] Could not open browser automatically:", exc)

    print("Waiting for Twitch approval...")
    token_payload = poll_device_authorization(
        client_id,
        device_payload["device_code"],
        interval,
        expires_in,
    )
    access_token = token_payload.get("access_token")
    channel_login = None

    if access_token:
        try:
            channel_login = fetch_user_login(access_token, client_id)
        except Exception as exc:
            print("[twitch_oauth_tool] warning: failed to fetch channel login:", exc)
    if channel_login:
        token_payload["channel_login"] = channel_login

    token_payload = _stamp_token_payload(token_payload, force=True)
    save_tokens(token_payload)

    # >>> NEW: notify backend of successful auth (after we have channel_login and obtained_at)
    if channel_login:
        _notify_backend_auth(channel_login, token_payload.get("obtained_at"), token_payload.get("user_id"))

    print("Authorization complete! Access token obtained for Autoaffili service:")
    print(access_token)
    if channel_login:
        print(f"Detected Twitch channel login: {channel_login}")
    expires = token_payload.get("expires_in")
    if expires:
        print(f"Token expires in {expires} seconds. Tokens saved to {TOKEN_FILE}")
    else:
        print(f"Tokens saved to {TOKEN_FILE}")
    return token_payload


def run_device_flow_gui(client_id: str) -> dict:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        raise RuntimeError("tkinter is not available") from exc

    device_payload = request_device_code(client_id)
    user_code = device_payload["user_code"]
    verification_uri = device_payload["verification_uri"]
    verification_complete = device_payload.get("verification_uri_complete") or verification_uri
    expires_in = int(device_payload.get("expires_in", 0))
    interval = int(device_payload.get("interval", 5))

    root = tk.Tk()
    root.title("Autoaffili - Twitch Authorization")
    root.resizable(False, False)

    instructions = (
        "Authorize Autoaffili to post Twitch chat messages."
    )

    tk.Label(root, text=instructions, justify="left", wraplength=360).pack(padx=16, pady=(16, 8))

    code_frame = tk.Frame(root)
    code_frame.pack(padx=16, pady=4, fill="x")
    tk.Label(code_frame, text="Code:", font=("Segoe UI", 10, "bold"), width=6).pack(side="left")
    code_entry = tk.Entry(code_frame, font=("Consolas", 14), justify="center")
    code_entry.insert(0, user_code)
    code_entry.configure(state="readonly")
    code_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))

    status_var = tk.StringVar(value="Waiting for approval...")
    status_label = tk.Label(root, textvariable=status_var, wraplength=360, fg="#444")
    status_label.pack(padx=16, pady=(8, 12))

    result: dict = {}

    def open_link() -> None:
        try:
            webbrowser.open(verification_complete, new=1)
        except Exception as exc:
            messagebox.showwarning("Autoaffili", f"Unable to open browser automatically: {exc}")

    def update_status(message: str) -> None:
        status_var.set(message)

    def on_success(payload: dict) -> None:
        result["payload"] = payload
        status_var.set("Authorization complete!")
        messagebox.showinfo("Autoaffili", "Twitch authorization succeeded.")
        root.destroy()

    def on_error(err: Exception) -> None:
        status_var.set(str(err))
        messagebox.showerror("Autoaffili", f"Authorization failed: {err}")
        root.destroy()

    def on_cancel() -> None:
        result["error"] = RuntimeError("Authorization cancelled")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    tk.Button(root, text="Open Twitch", command=open_link).pack(padx=16, pady=(0, 16))

    def status_callback(message: str) -> None:
        root.after(0, lambda: update_status(message))

    def poll_thread() -> None:
        try:
            payload = poll_device_authorization(
                client_id,
                device_payload["device_code"],
                interval,
                expires_in,
                status_callback=status_callback,
            )
        except Exception as exc:
            result["error"] = exc
            root.after(0, lambda: on_error(exc))
        else:
            root.after(0, lambda: on_success(payload))

    threading.Thread(target=poll_thread, daemon=True).start()
    root.after(250, open_link)
    root.mainloop()

    if "payload" in result:
        return result["payload"]
    raise result.get("error", RuntimeError("Authorization failed"))


def ensure_tokens(run_gui: bool = True) -> dict:
    client_id = os.getenv("TWITCH_CLIENT_ID", DEFAULT_CLIENT_ID)
    if not client_id or client_id == "TWITCH_CLIENT_ID_PLACEHOLDER":
        raise RuntimeError("TWITCH_CLIENT_ID must be set to your registered Twitch app client ID")

    host = os.getenv("TWITCH_REDIRECT_HOST", "localhost")
    port = int(os.getenv("TWITCH_REDIRECT_PORT", "8766"))

    def obtain_via_device() -> dict:
        if run_gui:
            try:
                payload = run_device_flow_gui(client_id)
                # Enrich with channel login for downstream bot convenience
                at = payload.get("access_token") if isinstance(payload, dict) else None
                if at:
                    try:
                        ch = fetch_user_login(at, client_id)
                        if ch:
                            payload["channel_login"] = ch
                    except Exception:
                        pass
                payload = _stamp_token_payload(payload, force=True)
                save_tokens(payload)
                # >>> NEW: notify backend for GUI/device flow as well
                if payload.get("channel_login"):
                    _notify_backend_auth(payload["channel_login"], payload.get("obtained_at"), payload.get("user_id"))
                return payload
            except Exception as exc:
                print("[twitch_oauth_tool] GUI authorization unavailable:", exc)
        payload = run_interactive_flow(host, port)
        if payload is None:
            payload = load_tokens()
        payload = _stamp_token_payload(payload, force=True)
        save_tokens(payload)
        # >>> NEW: notify backend when obtained via interactive flow or fallback
        if isinstance(payload, dict) and payload.get("channel_login"):
            _notify_backend_auth(payload["channel_login"], payload.get("obtained_at"), payload.get("user_id"))
        return payload

    try:
        data = load_tokens()
    except FileNotFoundError:
        return obtain_via_device()

    if not isinstance(data, dict):
        return obtain_via_device()

    refreshed = False
    if _token_is_near_expiry(data):
        refreshed_payload = _attempt_refresh(client_id, data)
        if refreshed_payload:
            data = refreshed_payload
            refreshed = True
        else:
            return obtain_via_device()

    if not _verify_access_token(data, client_id):
        refreshed_payload = _attempt_refresh(client_id, data)
        if refreshed_payload:
            data = refreshed_payload
            refreshed = True
        else:
            return obtain_via_device()

    if refreshed:
        return data

    # Ensure metadata exists without overwriting valid timestamps
    stamped = _stamp_token_payload(data)
    if stamped is not data:
        save_tokens(stamped)
        return stamped
    return data


def run_refresh_flow() -> None:
    client_id = os.getenv("TWITCH_CLIENT_ID", DEFAULT_CLIENT_ID)
    if not client_id or client_id == "TWITCH_CLIENT_ID_PLACEHOLDER":
        raise RuntimeError("TWITCH_CLIENT_ID must be set to refresh tokens")

    data = load_tokens()
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Stored token file does not contain a refresh_token")

    print("Refreshing Twitch access token...")
    token_payload = refresh_access_token(client_id, refresh_token)
    token_payload.setdefault("refresh_token", refresh_token)
    if data.get("channel_login"):
        token_payload.setdefault("channel_login", data.get("channel_login"))

    token_payload = _stamp_token_payload(token_payload, force=True)
    save_tokens(token_payload)
    print("Access token refreshed. New access token:")
    print(token_payload.get("access_token"))
    new_expires = token_payload.get("expires_in")
    if new_expires:
        print(f"Token expires in {new_expires} seconds. Tokens saved to {TOKEN_FILE}")
    else:
        print(f"Tokens saved to {TOKEN_FILE}")
    return token_payload


def main():
    parser = argparse.ArgumentParser(description="Twitch OAuth companion for Autoaffili (Device Code)")
    parser.add_argument("--refresh", action="store_true", help="refresh using stored refresh token")
    args = parser.parse_args()

    host = os.getenv("TWITCH_REDIRECT_HOST", "localhost")
    port = int(os.getenv("TWITCH_REDIRECT_PORT", "8766"))

    if args.refresh:
        run_refresh_flow()
    else:
        run_interactive_flow(host, port)


if __name__ == "__main__":
    main()
