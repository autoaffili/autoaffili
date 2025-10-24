# link_generator.py
"""
Build an Amazon search URL from extracted keywords, shorten it with Geniuslink, and send
the short link to Twitch chat with the channel-specific sid parameter.
"""

import os
import time
import urllib.parse
from typing import Optional

import requests

# Amazon search configuration (no affiliate tag so Geniuslink can localize)
AMAZON_DOMAIN = os.getenv("AMAZON_DOMAIN", "www.amazon.com").strip() or "www.amazon.com"
# Hardcoded global throttle: one link per 20 minutes
LINK_THROTTLE_SECONDS = 180
_last_link_ts = 0.0

def _resolve_channel_login() -> Optional[str]:
    """Best-effort resolve the Twitch channel login for sid/ascsubtag tracking.
    Order: TWITCH_CHANNEL env -> twitch_bot token cache -> None
    """
    # First prefer explicit env override
    channel = (os.getenv("TWITCH_CHANNEL") or "").strip()
    if channel:
        return channel
    # Fallback: ask twitch_bot (lazy import to avoid heavy deps at module import)
    try:
        from twitch_bot import _load_channel_login as _load_ch
        channel = (_load_ch() or "").strip()
        if channel:
            return channel
    except Exception:
        pass
    return None


def build_amazon_search(keywords: str) -> str:
    """Return an Amazon US search URL suitable for Geniuslink localization."""
    if not keywords:
        return ""
    # Normalize whitespace and url-encode
    q = " ".join(keywords.split())
    query = urllib.parse.quote_plus(q)
    return f"https://{AMAZON_DOMAIN}/s?k={query}"

def handle_keywords(keywords, original_transcript=None):
    """
    Build the Amazon search URL, shorten it with Geniuslink, and send it to Twitch chat.
    `keywords` may be a string or a list/tuple of strings (we pick the first).
    """
    # Normalize keywords input (support list/tuple)
    if isinstance(keywords, (list, tuple)):
        keywords = keywords[0] if keywords else ""
    if keywords is None:
        keywords = ""
    keywords = str(keywords).strip()

    if not keywords:
        print("[link_generator] no keywords provided, skipping link generation")
        return

    # Global throttle: allow at most one link every LINK_THROTTLE_SECONDS
    global _last_link_ts
    now = time.time()
    if LINK_THROTTLE_SECONDS > 0 and (_last_link_ts and (now - _last_link_ts) < LINK_THROTTLE_SECONDS):
        wait_left = int(LINK_THROTTLE_SECONDS - (now - _last_link_ts))
        print(f"[link_generator] throttled: next link allowed in ~{wait_left}s")
        return
    # Resolve channel for sid tracking and build the long URL
    channel = _resolve_channel_login()
    long_url = build_amazon_search(keywords)

    # Shorten via backend (no fallback)
    try:
        base = os.getenv(
            "BACKEND_BASE_URL",
            "https://autoaffili-backend-802674334607.us-east4.run.app",
        ).rstrip("/")
        payload = {"url": long_url, "sid": channel or None}
        resp = requests.post(f"{base}/gl/shorten", json=payload, timeout=10)
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
        if resp.status_code != 200 or not isinstance(data, dict) or not data.get("short_url"):
            msg = data or resp.text
            print(f"[link_generator] backend shorten failed: {resp.status_code} {msg}")
            return
        short_link = str(data["short_url"]).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[link_generator] shorten error: {exc}")
        return

    # Debug print (local visibility)
    print(f"[link_generator] Generated Geniuslink for '{keywords}': {short_link}")

    # Send to Twitch chat
    try:
        from twitch_bot import send_link_to_chat
        send_link_to_chat(short_link, keywords)
        _last_link_ts = time.time()
    except Exception as e:
        # Don't crash the service if Twitch fails; log the error for debugging.
        print("[link_generator] Error sending link to Twitch:", e)
