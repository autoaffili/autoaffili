# service/service_launcher.py

import os
# Load .env from common locations so packaged EXE picks up settings, not just repo runs
def _load_envs():
    # 1) service/.env (repo/dev)
    try:
        load_dotenv()
    except Exception:
        pass
    # 2) %LOCALAPPDATA%/autoaffili/.env (per-user install)
    try:
        la = os.getenv("LOCALAPPDATA")
        if la:
            base = os.path.join(la, "autoaffili")
            for fname in (".env", "autoaffili.env"):
                p = os.path.join(base, fname)
                if os.path.exists(p):
                    load_dotenv(p, override=True)
    except Exception:
        pass
    # 3) %PROGRAMDATA%/autoaffili/.env (per-machine install)
    try:
        pd = os.getenv("PROGRAMDATA")
        if pd:
            base = os.path.join(pd, "autoaffili")
            for fname in (".env", "autoaffili.env"):
                p = os.path.join(base, fname)
                if os.path.exists(p):
                    load_dotenv(p, override=True)
    except Exception:
        pass

_load_envs()


import time

from twitch_oauth_tool import ensure_tokens
import whisper_worker as _ww


_AUTO_AUTHORIZE = os.getenv("TWITCH_AUTO_AUTHORIZE", "1") != "0"
_USE_GUI_AUTH = os.getenv("TWITCH_GUI_AUTH", "1") != "0"


def start_all():
    if _AUTO_AUTHORIZE:
        ensure_tokens(run_gui=_USE_GUI_AUTH)
    # start whisper worker (recording/transcription)
    _ww.start_worker()
    # twitch bot will start lazily when send_link_to_chat is called


if __name__ == "__main__":
    start_all()
    # keep the main thread alive
    while True:
        time.sleep(1)

