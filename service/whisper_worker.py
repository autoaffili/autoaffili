import os
import time
import threading
import queue
import socket
import struct
import math
import numpy as np
import json
import csv
import re
import glob
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

# Heavy imports are loaded lazily inside ensure_model_loaded to avoid startup failures
torch = None
AutoProcessor = None
AutoModelForSpeechSeq2Seq = None
pipeline = None

# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "48000"))
CHUNK_SECS = float(os.getenv("AUDIO_CHUNK_SECONDS", "10.0"))
OVERLAP_SECS = float(os.getenv("AUDIO_CHUNK_OVERLAP_SECS", "0.27"))
CHUNK_FRAMES = max(1, int(SAMPLE_RATE * CHUNK_SECS))
WHISPER_SAMPLE_RATE = 16000

# Backend selection and model config
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "faster").strip().lower()
MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", os.getenv("WHISPER_MODEL", "Systran/faster-whisper-medium.en")).strip()
FW_CONDITION_ON_PREV = os.getenv("FW_CONDITION_ON_PREV", "0") != "0"
FW_VAD_FILTER = os.getenv("FW_VAD_FILTER", "1") != "0"
ASR_STRIDE_SECS = float(os.getenv("ASR_STRIDE_SECS", "0.8"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en").strip() or "en"
VAD_AGGR = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0..3 where 3 is most aggressive
TRANSCRIBE_WORKERS = int(os.getenv("TRANSCRIBE_WORKERS", "1"))
# Drop any transcript whose ASR runtime exceeds this many seconds (0 disables)
# Hard-coded to 5.0s per request.
TRANSCRIBE_MAX_SECONDS = 5.0
RESET_DIAGNOSTICS_ON_START = os.getenv("RESET_DIAGNOSTICS_ON_START", "1") == "1"
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "1") == "1"
# Debugging controls for ASR
DEBUG_ASR = os.getenv("DEBUG_ASR", "1") == "1"
DEBUG_ASR_EMPTY_ONLY = os.getenv("DEBUG_ASR_EMPTY_ONLY", "1") == "1"
DEBUG_ASR_MAX_BYTES = int(os.getenv("DEBUG_ASR_MAX_BYTES", "16384"))
# When enabled, write ASR debug JSON for every chunk (not just empty/error)
DEBUG_ASR_ALL_CHUNKS = os.getenv("DEBUG_ASR_ALL_CHUNKS", "0") == "1"
# Simple AGC to boost very quiet inputs before ASR
AGC_ENABLE = os.getenv("AGC_ENABLE", "1") == "1"
AGC_TARGET_RMS = float(os.getenv("AGC_TARGET_RMS", "0.05"))
AGC_MAX_GAIN = float(os.getenv("AGC_MAX_GAIN", "15.0"))
EDGE_TRIM_ENABLE = os.getenv("EDGE_TRIM_ENABLE", "1") == "1"
EDGE_TRIM_VAD_AGGR = int(os.getenv("EDGE_TRIM_VAD_AGGR", str(VAD_AGGR)))

SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.004"))
SAVE_SILENCE_WAVS = False
SAVE_REPEAT_WAVS = False

AUDIO_BRIDGE_HOST = os.getenv("AUDIO_BRIDGE_HOST", "127.0.0.1")
# Separate ports for desktop and mic streams (mix indices 0 and 1 by default)
AUDIO_BRIDGE_PORT_DESKTOP = int(os.getenv("AUDIO_BRIDGE_PORT_DESKTOP", os.getenv("AUDIO_BRIDGE_PORT", "8765")))
AUDIO_BRIDGE_PORT_MIC = int(os.getenv("AUDIO_BRIDGE_PORT_MIC", "8766"))

# Diagnostics storage
BASE_DIR = os.path.dirname(__file__)
_LOCALAPP = os.getenv("LOCALAPPDATA") or ""
# Prefer a user-writable app base under LocalAppData
APP_BASE = os.path.join(_LOCALAPP, "autoaffili") if _LOCALAPP else BASE_DIR
DIAG_DIR = os.path.join(APP_BASE, "logs")
SUSPECT_WAV_DIR = os.path.join(DIAG_DIR, "suspect_wavs")
DIAG_CSV = os.path.join(DIAG_DIR, "diagnostics.csv")
os.makedirs(DIAG_DIR, exist_ok=True)
os.makedirs(SUSPECT_WAV_DIR, exist_ok=True)
DIAG_ASR_DIR = os.path.join(DIAG_DIR, "asr_debug")
try:
    os.makedirs(DIAG_ASR_DIR, exist_ok=True)
except Exception:
    pass

# Status file (for quick troubleshooting from installer build)
STATUS_PATH = os.path.join(APP_BASE, "status.json")

CSV_HEADER = [
    "seq",
    "ts",
    "q_before",
    "q_after",
    "blocks_collected",
    "chunk_secs",
    "transcribe_time",
    "n_segments",
    "segments_preview",
    "segment_lengths",
    "process_start_ts",
    "process_elapsed",
    "note",
    "mix_rms",
    "backend_call",
    "backend_reason",
    "backend_latency_ms",
    "token_estimate",
    "source",
    "transcript",
    "device",
    "model",
]

# -----------------------------
# Globals
# -----------------------------
audio_queue_desktop = queue.Queue(maxsize=64)
audio_queue_mic = queue.Queue(maxsize=64)
_tail_desktop = np.zeros(0, dtype=np.float32)
_tail_mic = np.zeros(0, dtype=np.float32)
_tail_lock = threading.Lock()

_asr_pipeline = None  # transformers pipeline
_asr_backend = None   # 'faster' or 'transformers'
_asr_model_faster = None  # faster_whisper.WhisperModel
_pool: Optional[ThreadPoolExecutor] = None
_seq_lock = threading.Lock()
_last_error_str: Optional[str] = None
_status_cache: Dict[str, object] = {}
_asr_lock = threading.Lock()
_asr_device_str: Optional[str] = None
_asr_model_name: Optional[str] = None
_asr_generate_kwargs: Optional[dict] = None
# Track bridge connectivity for idle-exit behavior
_bridge_state = {"desktop": False, "mic": False}
_bridge_lock = threading.Lock()
_last_any_connected_ts = 0.0

# Auto-exit if OBS/plugin disconnects and stays idle for a while
EXIT_WHEN_IDLE = os.getenv("EXIT_WHEN_IDLE", "0") == "1"
IDLE_EXIT_SECONDS = float(os.getenv("IDLE_EXIT_SECONDS", "10"))
_service_start_ts = time.time()

# -----------------------------
# Helpers
# -----------------------------
def _reset_diagnostics():
    if not RESET_DIAGNOSTICS_ON_START:
        return
    for p in glob.glob(os.path.join(SUSPECT_WAV_DIR, "*.wav")):
        try:
            os.remove(p)
        except Exception:
            pass
    for p in glob.glob(os.path.join(DIAG_ASR_DIR, "*.json")):
        try:
            os.remove(p)
        except Exception:
            pass
    try:
        with open(DIAG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
    except Exception as e:
        print("[whisper_worker][diag] failed to reset diagnostics:", e)

    try:
        open(os.path.join(DIAG_DIR, "errors.log"), "w", encoding="utf-8").close()
    except Exception:
        pass

_REPEATED_WORD_RE = re.compile(r"\\b(\\w+)(?:\\s+\\1){6,}", re.IGNORECASE)

# Common filler/hallucination phrases we want to suppress if emitted repeatedly
 


    

def _set_bridge_connected(name: str, connected: bool, port: int) -> None:
    """Update in-process bridge state and status.json."""
    global _last_any_connected_ts
    try:
        with _bridge_lock:
            _bridge_state[name] = bool(connected)
            if _bridge_state.get("desktop") or _bridge_state.get("mic"):
                _last_any_connected_ts = time.time()
    except Exception:
        pass
    # Keep status.json in sync for external diagnostics
    try:
        _write_status({
            "bridges": {name: {"listening": True, "connected": bool(connected), "port": port}}
        })
    except Exception:
        pass

def _idle_exit_monitor():
    """Terminate the helper when both bridges are disconnected for a while."""
    if not EXIT_WHEN_IDLE:
        return
    # Give some grace period on startup
    while True:
        try:
            time.sleep(1.0)
            since_start = time.time() - _service_start_ts
            if since_start < 15.0:
                continue
            with _bridge_lock:
                d = _bridge_state.get("desktop", False)
                m = _bridge_state.get("mic", False)
            if not d and not m:
                # no active connections
                if _last_any_connected_ts == 0.0:
                    idle_for = since_start
                else:
                    idle_for = time.time() - _last_any_connected_ts
                if idle_for >= IDLE_EXIT_SECONDS:
                    try:
                        print(f"[whisper_worker] idle for {idle_for:.1f}s with no OBS connections; exiting")
                    except Exception:
                        pass
                    os._exit(0)
        except Exception:
            # Never crash the monitor; keep looping
            pass

# -----------------------------
# OBS-driven shutdown monitor (Windows)
# -----------------------------
def _shutdown_event_monitor():
    """On Windows, wait on a named event set by the OBS plugin at exit, then exit immediately.
    This allows removing idle-exit entirely while still cleaning up with OBS lifecycle.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        import ctypes.wintypes as wt
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Create or open a manual-reset, initially non-signaled event
        CreateEventW = kernel32.CreateEventW
        CreateEventW.argtypes = [wt.LPVOID, wt.BOOL, wt.BOOL, wt.LPCWSTR]
        CreateEventW.restype = wt.HANDLE
        SetEvent = kernel32.SetEvent
        SetEvent.argtypes = [wt.HANDLE]
        SetEvent.restype = wt.BOOL
        WaitForSingleObject = kernel32.WaitForSingleObject
        WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        WaitForSingleObject.restype = wt.DWORD
        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wt.HANDLE]
        CloseHandle.restype = wt.BOOL

        INFINITE = 0xFFFFFFFF
        # Local\ prefix scopes to the current session
        name = "Local\\autoaffili_shutdown"
        h = CreateEventW(None, True, False, name)
        if not h:
            return
        try:
            # Block until OBS plugin signals us
            WaitForSingleObject(h, INFINITE)
        finally:
            try:
                CloseHandle(h)
            except Exception:
                pass
        os._exit(0)
    except Exception:
        # Don't let platform issues crash the service
        return


def _heartbeat_loop():
    """Append periodic heartbeat rows to diagnostics.csv with session uptime."""
    interval = 120  # seconds
    while True:
        try:
            time.sleep(interval)
            ts = time.time()
            uptime = int(ts - _service_start_ts)
            row = [""] * len(CSV_HEADER)
            row[1] = ts
            row[12] = f"heartbeat (uptime={uptime}s)"
            row[18] = "heartbeat"
            row[19] = f"uptime={uptime}s"
            with open(DIAG_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as exc:
            try:
                print("[whisper_worker][diag] heartbeat write failed:", exc)
            except Exception:
                pass


def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _apply_agc(x: np.ndarray, target_rms: float, max_gain: float) -> np.ndarray:
    """RMS-based AGC: scales x to target_rms, limited by max_gain.
    Input/output: mono float32 in [-1, 1]."""
    if not x.size or target_rms <= 0.0:
        return x
    cur = _rms(x)
    if cur <= 1e-8:
        return x
    gain = max(0.0, target_rms / cur)
    if max_gain > 0.0:
        gain = min(gain, max_gain)
    if abs(gain - 1.0) < 1e-3:
        return x
    y = x * gain
    np.clip(y, -1.0, 1.0, out=y)
    return y

def _resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data.astype(np.float32, copy=False)
    orig_len = data.shape[0]
    if orig_len == 0:
        return data.astype(np.float32, copy=False)
    new_len = max(1, int(round(orig_len * target_sr / float(orig_sr))))
    x_old = np.linspace(0.0, 1.0, orig_len, endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32, copy=False)


def _write_status(update: Dict[str, object]) -> None:
    try:
        # Merge shallow updates with existing
        current: Dict[str, object] = {}
        if os.path.exists(STATUS_PATH):
            try:
                with open(STATUS_PATH, "r", encoding="utf-8") as f:
                    current = json.load(f) or {}
            except Exception:
                # Fall back to in-process cache if file read fails
                current = dict(_status_cache) if isinstance(_status_cache, dict) else {}
        else:
            # Seed with cache if available
            current = dict(_status_cache) if isinstance(_status_cache, dict) else {}
        # Special handling to merge nested 'bridges'
        if "bridges" in update:
            bridges_update = update.get("bridges") or {}
            if not isinstance(bridges_update, dict):
                bridges_update = {}
            bridges_current = current.get("bridges") or {}
            if not isinstance(bridges_current, dict):
                bridges_current = {}
            bridges_current.update(bridges_update)
            current["bridges"] = bridges_current
        # Merge remaining keys shallowly
        for k, v in update.items():
            if k == "bridges":
                continue
            current[k] = v
        current["last_update"] = time.time()
        os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        # Update process-local cache
        try:
            _status_cache.clear()
            _status_cache.update(current)
        except Exception:
            pass
    except Exception:
        pass

# Edge-trim helper using WebRTC VAD (trim leading/trailing non-voiced regions)
def _vad_edge_trim(wav: np.ndarray, sr: int, aggressiveness: int = 2) -> Tuple[np.ndarray, float, float]:
    """Return (trimmed_audio, offset_seconds, trimmed_seconds).
    If no voiced frames found, returns (empty, 0.0, 0.0).
    wav: mono float32 in [-1,1] at sample rate sr.
    """
    try:
        import webrtcvad
    except Exception:
        # If VAD missing, no trimming
        return wav, 0.0, float(len(wav) / max(1, sr))

    vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
    # Build 30ms frames for VAD
    frame_len = int(0.03 * sr)
    if len(wav) < frame_len:
        return wav, 0.0, float(len(wav) / max(1, sr))
    int16 = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
    voiced_flags: List[bool] = []
    for i in range(0, len(int16) - frame_len + 1, frame_len):
        frame = int16[i:i + frame_len].tobytes()
        is_speech = False
        try:
            is_speech = vad.is_speech(frame, sr)
        except Exception:
            pass
        voiced_flags.append(is_speech)
    if not voiced_flags:
        return wav, 0.0, float(len(wav) / max(1, sr))
    # Find first and last voiced frames
    try:
        first = next(idx for idx, v in enumerate(voiced_flags) if v)
        last = len(voiced_flags) - 1 - next(idx for idx, v in enumerate(reversed(voiced_flags)) if v)
    except StopIteration:
        # No voiced frames at all
        return np.zeros(0, dtype=np.float32), 0.0, 0.0
    start_sample = first * frame_len
    end_sample = min(len(wav), (last + 1) * frame_len)
    trimmed = wav[start_sample:end_sample].copy()
    offset_sec = float(start_sample) / float(sr)
    dur_sec = float(len(trimmed)) / float(sr)
    return trimmed, offset_sec, dur_sec

# -----------------------------
# Audio bridge (OBS -> service)
# -----------------------------
def _recv_exact(sock: socket.socket, size: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < size:
        try:
            chunk = sock.recv(size - len(buf))
        except Exception:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def _json_preview(obj, max_bytes: int = 16384) -> str:
    try:
        txt = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        try:
            txt = str(obj)
        except Exception:
            txt = "<unserializable>"
    if len(txt) > max_bytes:
        return txt[:max_bytes] + "...<truncated>"
    return txt

def _write_asr_debug(seq: int, source: str, info: Dict[str, object], raw_obj) -> None:
    if not DEBUG_ASR:
        return
    try:
        payload = dict(info or {})
        payload.setdefault("source", source)
        payload.setdefault("seq", seq)
        payload.setdefault("model", _asr_model_name or MODEL_NAME)
        payload.setdefault("device", _asr_device_str or "unknown")
        payload.setdefault("generate_kwargs", _asr_generate_kwargs or {})
        payload["raw_preview"] = _json_preview(raw_obj, DEBUG_ASR_MAX_BYTES)
        name = f"{int(time.time()*1000)}_seq{seq:05d}_{source}.json"
        path = os.path.join(DIAG_ASR_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _bridge_loop(target_queue: "queue.Queue[np.ndarray]", port: int, name: str):
    while True:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((AUDIO_BRIDGE_HOST, port))
        except Exception as e:
            print(f"[whisper_worker] bridge({name}) bind failed: {e}")
            try:
                with open(os.path.join(DIAG_DIR, "errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"[bridge:{name}] bind failed on {AUDIO_BRIDGE_HOST}:{port}: {e}\n")
            except Exception:
                pass
            srv.close()
            time.sleep(1)
            continue
        srv.listen(1)
        print(f"[whisper_worker] audio bridge({name}) listening on {AUDIO_BRIDGE_HOST}:{port}")
        _set_bridge_connected(name, False, port)
        try:
            conn, addr = srv.accept()
            print(f"[whisper_worker] bridge({name}) accepted from {addr}")
            _set_bridge_connected(name, True, port)
        except Exception as e:
            print(f"[whisper_worker] bridge({name}) accept failed: {e}")
            try:
                with open(os.path.join(DIAG_DIR, "errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"[bridge:{name}] accept failed: {e}\n")
            except Exception:
                pass
            srv.close()
            time.sleep(1)
            continue
        print(f"[whisper_worker] audio bridge({name}) connected from {addr}")
        _set_bridge_connected(name, True, port)
        with conn:
            while True:
                header = _recv_exact(conn, 12)
                if not header:
                    break
                frames, channels, sample_rate = struct.unpack('<III', header)
                if channels <= 0 or channels > 8 or sample_rate < 8000 or sample_rate > 192000:
                    msg = f"[bridge:{name}] invalid header frames={frames} channels={channels} rate={sample_rate}"
                    print("[whisper_worker]", msg)
                    try:
                        with open(os.path.join(DIAG_DIR, "errors.log"), "a", encoding="utf-8") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass
                    break
                if frames == 0 or channels == 0:
                    continue
                payload = _recv_exact(conn, frames * channels * 4)
                if not payload:
                    break
                arr = np.frombuffer(payload, dtype=np.float32)
                try:
                    arr = arr.reshape(frames, channels)
                except ValueError:
                    continue
                mono = arr.mean(axis=1)
                if sample_rate != SAMPLE_RATE:
                    try:
                        mono = _resample_audio(mono, sample_rate, SAMPLE_RATE)
                    except Exception as e:
                        print(f"[whisper_worker] ({name}) resample failed from {sample_rate} to {SAMPLE_RATE}: {e}")
                        continue
                try:
                    target_queue.put_nowait(mono.astype(np.float32, copy=False))
                except queue.Full:
                    try:
                        target_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        target_queue.put_nowait(mono.astype(np.float32, copy=False))
                    except Exception:
                        pass
        print(f"[whisper_worker] audio bridge({name}) disconnected, awaiting reconnection")
        _set_bridge_connected(name, False, port)
        srv.close()
        time.sleep(0.5)

# -----------------------------
# Model loader
# -----------------------------
def ensure_model_loaded():
    global _asr_pipeline, torch, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, _last_error_str, _asr_device_str, _asr_model_name, _asr_generate_kwargs, _asr_backend, _asr_model_faster
    if _asr_pipeline is not None or _asr_model_faster is not None:
        return
    _write_status({"model_ready": False})
    # Lazy imports and model prep with robust diagnostics
    try:
        if WHISPER_BACKEND == 'faster':
            # Load Faster-Whisper (CTranslate2)
            import importlib
            fw = importlib.import_module('faster_whisper')
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
            default_threads = _auto_ct2_threads()
            env_threads = os.getenv("CT2_CPU_THREADS")
            try:
                cpu_threads = int(env_threads) if env_threads else default_threads
            except Exception:
                cpu_threads = default_threads
            if cpu_threads <= 0:
                cpu_threads = default_threads
            try:
                beam_size = max(1, int(os.getenv("CT2_BEAM_SIZE", "1")))
            except Exception:
                beam_size = 1
            print(f"[whisper_worker] Loading Faster-Whisper model '{MODEL_NAME}' on device=cpu compute_type={compute_type} threads={cpu_threads} ...")
            model = fw.WhisperModel(MODEL_NAME, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
            _asr_model_faster = model
            _asr_backend = 'faster'
            _asr_device_str = 'cpu'
            _asr_model_name = MODEL_NAME
            _asr_generate_kwargs = {"beam_size": beam_size, "compute_type": compute_type}
            print("[whisper_worker] Model ready (Faster-Whisper).")
        else:
            # Transformers fallback (CPU)
            if torch is None:
                import importlib
                try:
                    torch = importlib.import_module('torch')
                except Exception as e:
                    raise RuntimeError(f"Failed to import torch: {e}")
                try:
                    _tfm = importlib.import_module('transformers')
                except Exception as e:
                    raise RuntimeError(f"Failed to import transformers: {e}")
                AutoProcessor = _tfm.AutoProcessor
                AutoModelForSpeechSeq2Seq = _tfm.AutoModelForSpeechSeq2Seq
                pipeline = _tfm.pipeline

            device = torch.device("cpu")
            print(f"[whisper_worker] Loading transformers model '{MODEL_NAME}' on device={device} ...")
            try:
                processor = AutoProcessor.from_pretrained(MODEL_NAME)
            except Exception as e:
                raise RuntimeError(f"Failed to load AutoProcessor.from_pretrained({MODEL_NAME}): {e}")
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)
            except Exception as e:
                raise RuntimeError(f"Failed to load AutoModelForSpeechSeq2Seq.from_pretrained({MODEL_NAME}): {e}")
            try:
                model.to(device)
                if hasattr(model, 'eval'):
                    model.eval()
            except Exception:
                pass
            name_l = (MODEL_NAME or "").lower()
            english_only = name_l.endswith('.en') or any(s in name_l for s in ['tiny.en', 'base.en', 'small.en', 'medium.en'])
            gen_kwargs = {}
            if not english_only:
                gen_kwargs = {"task": "transcribe"}
                if WHISPER_LANGUAGE:
                    gen_kwargs["language"] = WHISPER_LANGUAGE
            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "chunk_length_s": max(1.0, CHUNK_SECS),
                "stride_length_s": max(0.0, ASR_STRIDE_SECS),
                "return_timestamps": True,
                "device": -1,
            }
            if gen_kwargs:
                pipeline_kwargs["generate_kwargs"] = gen_kwargs
            _asr_pipeline = pipeline(**pipeline_kwargs)
            _asr_backend = 'transformers'
            _asr_device_str = str(device)
            _asr_model_name = MODEL_NAME
            _asr_generate_kwargs = dict(gen_kwargs) if gen_kwargs else {}
            print("[whisper_worker] Model ready.")
        _last_error_str = None
        _write_status({"model_ready": True, "last_error": None})
    except Exception as e:
        # Persist error to logs and status so installer can surface it
        err_msg = f"model_init_error: {e}"
        _last_error_str = err_msg
        try:
            with open(os.path.join(DIAG_DIR, "errors.log"), "a", encoding="utf-8") as f:
                f.write("[whisper_worker] " + err_msg + "\n")
                f.write(traceback.format_exc() + "\n")
        except Exception:
            pass
        _write_status({"model_ready": False, "last_error": err_msg})
        # Leave _asr_pipeline as None; caller will emit 'model_error' note
        return


def _write_bridge_info():
    try:
        # Keep globals writable if we sync from plugin.ini
        global AUDIO_BRIDGE_PORT_DESKTOP, AUDIO_BRIDGE_PORT_MIC
        info = {
            "host": AUDIO_BRIDGE_HOST,
            "desktop_port": AUDIO_BRIDGE_PORT_DESKTOP,
            "mic_port": AUDIO_BRIDGE_PORT_MIC,
        }
        # Write both a JSON and INI-style helper for the plugin
        local = os.getenv("LOCALAPPDATA")
        base = os.path.join(local, "autoaffili") if local else APP_BASE
        os.makedirs(base, exist_ok=True)
        # JSON file
        with open(os.path.join(base, "bridge.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        # plugin.ini with ports for convenience
        ini_path = os.path.join(base, "plugin.ini")
        if not os.path.exists(ini_path):
            with open(ini_path, "w", encoding="utf-8") as f:
                f.write(f"desktop_port={AUDIO_BRIDGE_PORT_DESKTOP}\n")
                f.write(f"mic_port={AUDIO_BRIDGE_PORT_MIC}\n")
                f.write("desktop_mix=0\nmic_mix=1\n")
        else:
            # Keep plugin.ini ports in sync if user edited them
            try:
                txt = open(ini_path, "r", encoding="utf-8").read()
                lines = [x.strip() for x in txt.splitlines() if x.strip()]
                ini_ports = {}
                for ln in lines:
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        ini_ports[k.strip()] = v.strip()
                dp = int(ini_ports.get("desktop_port", AUDIO_BRIDGE_PORT_DESKTOP))
                mp = int(ini_ports.get("mic_port", AUDIO_BRIDGE_PORT_MIC))
                if dp != AUDIO_BRIDGE_PORT_DESKTOP or mp != AUDIO_BRIDGE_PORT_MIC:
                    AUDIO_BRIDGE_PORT_DESKTOP = dp
                    AUDIO_BRIDGE_PORT_MIC = mp
            except Exception:
                pass
    except Exception as e:
        print("[whisper_worker] failed to write bridge info:", e)

# -----------------------------
# Background processing helper
# -----------------------------
def _bg_process_wrapper(text_local: str, diag: Dict):
    process_start = time.time()
    metrics = None
    try:
        from context_filter import process_transcript_multi as _pt_multi
        # approximate timing window for this chunk
        start_ts = float(diag.get("ts") or time.time()) - float(diag.get("chunk_secs") or 0.0)
        end_ts = float(diag.get("ts") or time.time())
        src = str(diag.get("source") or "unknown")
        metrics = _pt_multi(src, text_local, start_ts, end_ts, diag)
    except Exception as e:
        print("[whisper_worker] process_transcript error (diag):", e)
    process_elapsed = time.time() - process_start

    if metrics and isinstance(metrics, dict):
        diag.update(metrics)

    diag["process_start_ts"] = process_start
    diag["process_elapsed"] = process_elapsed

    try:
        with open(DIAG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                diag.get("seq"),
                diag.get("ts"),
                diag.get("q_before"),
                diag.get("q_after"),
                diag.get("blocks_collected"),
                diag.get("chunk_secs"),
                diag.get("transcribe_time"),
                diag.get("n_segments"),
                diag.get("segments_preview"),
                diag.get("segment_lengths"),
                diag.get("process_start_ts"),
                diag.get("process_elapsed"),
                diag.get("note"),
                diag.get("mix_rms"),
                diag.get("backend_call"),
                diag.get("backend_reason"),
                diag.get("backend_latency_ms"),
                diag.get("token_estimate"),
                diag.get("source"),
                diag.get("transcript"),
                diag.get("device"),
                diag.get("model"),
            ])
    except Exception as e:
        print("[whisper_worker][diag] failed to write diagnostics row:", e)

# -----------------------------
# Transcription worker
# -----------------------------
def _transcribe_and_dispatch(source: str, seq: int, arr: np.ndarray, coll_meta: Dict, tail_in: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict]:
    global _asr_pipeline

    frames_len_seconds = float(arr.shape[0]) / SAMPLE_RATE
    chunk_rms = _rms(arr)
    trim_offset_sec = 0.0
    trimmed_sec = frames_len_seconds

    seg_texts: List[str] = []
    seg_lens: List[int] = []
    n_segments = 0
    seg_preview = ""
    trans_time = 0.0
    timed_out = False

    # If the chunk appears silent/very quiet, skip transcription to avoid hallucinations
    if chunk_rms < SILENCE_RMS_THRESHOLD:
        note = "pre_silence_skip"
        text_out = ""
        diag = {
            "seq": seq,
            "ts": time.time(),
            "q_before": coll_meta.get("q_before"),
            "q_after": coll_meta.get("q_after"),
            "blocks_collected": coll_meta.get("blocks_collected"),
            "chunk_secs": round(frames_len_seconds, 3),
            "transcribe_time": 0.0,
            "n_segments": 0,
            "segments_preview": "",
            "segment_lengths": "",
            "note": note,
            "mix_rms": round(chunk_rms, 5),
            "backend_call": False,
            "backend_reason": "",
            "backend_latency_ms": None,
            "token_estimate": None,
        }
        # Add source + transcript for diagnostics
        diag["source"] = source
        diag["transcript"] = text_out
        # Include device/model for extended diagnostics
        try:
            diag["device"] = _asr_device_str or str(torch.device("cpu"))
        except Exception:
            diag["device"] = "cpu"
        diag["model"] = _asr_model_name or MODEL_NAME
        if DEBUG_ASR and DEBUG_ASR_ALL_CHUNKS:
            _write_asr_debug(seq, source, {
                "reason": note,
                "frames_len_seconds": frames_len_seconds,
                "rms": chunk_rms,
                "rms_threshold": SILENCE_RMS_THRESHOLD,
            }, {})
        try:
            threading.Thread(target=_bg_process_wrapper, args=(text_out, diag), daemon=True).start()
        except Exception as e:
            print("[whisper_worker][diag] failed to start bg process thread:", e)
        return tail_in if tail_in is not None else np.zeros(0, dtype=np.float32), diag

    try:
        t0 = time.time()
        tmp = arr.astype(np.float32)
        tmp = _resample_audio(tmp, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        trimmed_sec = float(len(tmp)) / WHISPER_SAMPLE_RATE if tmp.size else 0.0

        if EDGE_TRIM_ENABLE and tmp.size:
            trimmed_arr, trim_offset_sec, trimmed_len_sec = _vad_edge_trim(tmp, WHISPER_SAMPLE_RATE, EDGE_TRIM_VAD_AGGR)
            if trimmed_arr.size:
                tmp = trimmed_arr
                trimmed_sec = trimmed_len_sec
            else:
                tmp = np.zeros(0, dtype=np.float32)
                trimmed_sec = 0.0

        rms_before_agc = _rms(tmp)

        if tmp.size == 0:
            note = "trimmed_silence"
            text_out = ""
            diag = {
                "seq": seq,
                "ts": time.time(),
                "q_before": coll_meta.get("q_before"),
                "q_after": coll_meta.get("q_after"),
                "blocks_collected": coll_meta.get("blocks_collected"),
                "chunk_secs": round(frames_len_seconds, 3),
                "transcribe_time": 0.0,
                "n_segments": 0,
                "segments_preview": "",
                "segment_lengths": "",
                "note": note,
                "mix_rms": round(chunk_rms, 5),
                "backend_call": False,
                "backend_reason": "",
                "backend_latency_ms": None,
                "token_estimate": None,
                "trim_offset_secs": trim_offset_sec,
                "trimmed_secs": trimmed_sec,
            }
            diag["source"] = source
            diag["transcript"] = text_out
            try:
                diag["device"] = _asr_device_str or str(torch.device("cpu"))
            except Exception:
                diag["device"] = "cpu"
            diag["model"] = _asr_model_name or MODEL_NAME
            if DEBUG_ASR and DEBUG_ASR_ALL_CHUNKS:
                _write_asr_debug(seq, source, {
                    "reason": note,
                    "frames_len_seconds": frames_len_seconds,
                    "trim_offset_secs": trim_offset_sec,
                    "trimmed_secs": trimmed_sec,
                    "rms": chunk_rms,
                }, {})
            try:
                threading.Thread(target=_bg_process_wrapper, args=(text_out, diag), daemon=True).start()
            except Exception as e:
                print("[whisper_worker][diag] failed to start bg process thread:", e)
            return tail_in if tail_in is not None else np.zeros(0, dtype=np.float32), diag

        # If model not ready yet, emit a lightweight diag row and return
        if _asr_pipeline is None and _asr_model_faster is None:
            note = "model_error" if (_last_error_str or "") else "model_loading"
            text_out = ""
            diag = {
                "seq": seq,
                "ts": time.time(),
                "q_before": coll_meta.get("q_before"),
                "q_after": coll_meta.get("q_after"),
                "blocks_collected": coll_meta.get("blocks_collected"),
                "chunk_secs": round(frames_len_seconds, 3),
                "transcribe_time": 0.0,
                "n_segments": 0,
                "segments_preview": "",
                "segment_lengths": "",
                "note": note,
                "mix_rms": round(chunk_rms, 5),
                "backend_call": False,
                "backend_reason": (_last_error_str or ""),
                "backend_latency_ms": None,
                "token_estimate": None,
                "trim_offset_secs": trim_offset_sec,
                "trimmed_secs": trimmed_sec,
            }
            diag["source"] = source
            diag["transcript"] = text_out
            try:
                diag_local = dict(diag)
                diag_local["source"] = source
                threading.Thread(target=_bg_process_wrapper, args=(text_out, diag_local), daemon=True).start()
            except Exception:
                pass
            next_tail = arr[-int(max(0.0, OVERLAP_SECS) * SAMPLE_RATE):].copy() if OVERLAP_SECS > 0 else np.zeros(0, dtype=np.float32)
            return next_tail, diag

        # Optional AGC to compensate for very quiet inputs
        if AGC_ENABLE:
            tmp = _apply_agc(tmp, AGC_TARGET_RMS, AGC_MAX_GAIN)
        rms_after_agc = _rms(tmp)
        # Run ASR (backend-specific) under lock
        texts = []
        with _asr_lock:
            if _asr_backend == 'faster' and _asr_model_faster is not None:
                # Faster-Whisper expects float32 mono at 16 kHz
                beam_size = int((_asr_generate_kwargs or {}).get('beam_size', 1))
                language = WHISPER_LANGUAGE or 'en'
                segments, info = _asr_model_faster.transcribe(
                    tmp,
                    language=language,
                    task='transcribe',
                    beam_size=beam_size,
                    vad_filter=FW_VAD_FILTER,
                    condition_on_previous_text=FW_CONDITION_ON_PREV,
                )
                seg_list = list(segments)
                for seg in seg_list:
                    t = (getattr(seg, 'text', '') or '').strip()
                    if t:
                        texts.append(t)
            else:
                # Transformers pipeline output
                res = _asr_pipeline(tmp)
                chunks = res.get("chunks") if isinstance(res, dict) else None
                if isinstance(chunks, list) and chunks:
                    for ch in chunks:
                        try:
                            t = (ch.get("text") or "").strip()
                            if t:
                                texts.append(t)
                        except Exception:
                            continue
                else:
                    try:
                        t = (res.get("text") or "").strip() if isinstance(res, dict) else str(res).strip()
                    except Exception:
                        t = ""
                    if t:
                        texts.append(t)
        trans_time = time.time() - t0
        # Enforce transcription timeout policy
        try:
            if TRANSCRIBE_MAX_SECONDS > 0.0 and trans_time > TRANSCRIBE_MAX_SECONDS:
                timed_out = True
                # Drop this transcript: clear any text collected
                seg_texts.clear()
                seg_lens.clear()
                n_segments = 0
                seg_preview = ""
        except Exception:
            pass
        for tx in texts:
            clean = tx[:200].replace("\n", " ")
            seg_texts.append(clean)
            seg_lens.append(len(clean))
        n_segments = len(seg_texts)
        seg_preview = " || ".join(seg_texts[:3])
    except Exception as e:
        err_msg = f"[whisper_worker] Transcription error: {e}"
        print(err_msg)
        try:
            with open(os.path.join(DIAG_DIR, "errors.log"), "a", encoding="utf-8") as f:
                f.write(err_msg + "\n")
                f.write(traceback.format_exc() + "\n")
        except Exception:
            pass
        if DEBUG_ASR:
            _write_asr_debug(seq, source, {
                "error": str(e),
                "chunk_secs": float(len(arr)) / float(SAMPLE_RATE),
                "rms": chunk_rms,
            }, "<exception>")

    print(
        f"[whisper_worker] transcribe_time={trans_time:.2f}s, chunk_secs={frames_len_seconds:.2f}s, "
        f"blocks={coll_meta.get('blocks_collected')}, n_segments={n_segments}, rms={chunk_rms:.5f}"
    )

    note = "asr_error" if 'err_msg' in locals() else ""
    joined_preview = " ".join(seg_texts) if seg_texts else ""
    if n_segments == 0:
        if timed_out:
            note = "timeout_drop"
        elif chunk_rms < SILENCE_RMS_THRESHOLD:
            note = "silence"
        else:
            note = "empty_transcript_non_silent"

        if DEBUG_ASR and (not DEBUG_ASR_EMPTY_ONLY or True):
            # Dump raw ASR result summary when the model produced no text
            _write_asr_debug(seq, source, {
                "reason": note,
                "frames_len_seconds": frames_len_seconds,
                "rms_before_agc": rms_before_agc,
                "rms_after_agc": locals().get('rms_after_agc', rms_before_agc),
                "q_before": coll_meta.get("q_before"),
                "q_after": coll_meta.get("q_after"),
                "blocks_collected": coll_meta.get("blocks_collected"),
                "queue_delay": float(max(0.0, t0 - float(coll_meta.get('submit_ts', t0)))),
                "transcribe_time": trans_time if 'trans_time' in locals() else None,
                "trim_offset_secs": trim_offset_sec,
                "trimmed_secs": trimmed_sec,
            }, locals().get('res', {}))
    else:
        if _REPEATED_WORD_RE.search(joined_preview):
            note = "user_repetition"
    text_out = " ".join(seg_texts).strip()

    diag = {
        "seq": seq,
        "ts": time.time(),
        "q_before": coll_meta.get("q_before"),
        "q_after": coll_meta.get("q_after"),
        "blocks_collected": coll_meta.get("blocks_collected"),
        "chunk_secs": round(frames_len_seconds, 3),
        "transcribe_time": round(trans_time or 0.0, 3),
        "n_segments": n_segments,
        "segments_preview": seg_preview,
        "segment_lengths": "|".join(map(str, seg_lens)) if seg_lens else "",
        "note": note,
        "mix_rms": round(chunk_rms, 5),
        "backend_call": False,
        "backend_reason": "",
        "backend_latency_ms": None,
        "token_estimate": None,
        # Edge-trim diagnostics (approx seconds relative to original chunk)
        "trim_offset_secs": trim_offset_sec,
        "trimmed_secs": trimmed_sec,
        # Queue timing
        "queue_delay": float(max(0.0, t0 - float(coll_meta.get('submit_ts', t0)))),
    }
    # Include source + transcript text for diagnostics
    diag["source"] = source
    diag["transcript"] = text_out
    # Include device/model for extended diagnostics
    try:
        diag["device"] = _asr_device_str or str(torch.device("cpu"))
    except Exception:
        diag["device"] = "cpu"
    diag["model"] = _asr_model_name or MODEL_NAME

    if DEBUG_ASR and DEBUG_ASR_ALL_CHUNKS:
        _write_asr_debug(seq, source, {
            "reason": note or "ok",
            "frames_len_seconds": frames_len_seconds,
            "rms": chunk_rms,
            "n_segments": n_segments,
        }, locals().get('res', {}))

    try:
        diag_local = dict(diag)
        diag_local["source"] = source
        # If timed out, pass empty text to avoid backend calls based on this chunk
        text_for_bg = (f"[{source}] {text_out}" if (text_out and not timed_out) else "")
        threading.Thread(target=_bg_process_wrapper, args=(text_for_bg, diag_local), daemon=True).start()
    except Exception as e:
        print("[whisper_worker][diag] failed to start bg process thread:", e)

    # Prepare next tail (overlap)
    overlap_frames = int(max(0.0, OVERLAP_SECS) * SAMPLE_RATE)
    next_tail = arr[-overlap_frames:].copy() if overlap_frames > 0 and arr.size >= overlap_frames else np.zeros(0, dtype=np.float32)

    return next_tail, diag

# -----------------------------
# Recorder loop (chunk aggregator)
# -----------------------------
def recorder_loop(name: str, q_in: "queue.Queue[np.ndarray]", tail_key: str):
    seq = 1
    target_frames = int(CHUNK_SECS * SAMPLE_RATE)

    global _tail_desktop, _tail_mic

    while True:
        frames_collected: List[np.ndarray] = []
        frames_accum = 0
        q_before = q_in.qsize()

        while frames_accum < target_frames:
            try:
                block = q_in.get(timeout=1.0)
            except queue.Empty:
                if frames_accum == 0:
                    continue
                else:
                    break
            frames_collected.append(block)
            frames_accum += block.shape[0]

        if not frames_collected:
            time.sleep(0.01)
            continue

        arr = np.concatenate(frames_collected, axis=0)
        q_after = q_in.qsize()

        coll_meta = {
            "q_before": q_before,
            "q_after": q_after,
            "blocks_collected": len(frames_collected),
        }

        # Get current tail and prepend for overlap
        with _tail_lock:
            tail = _tail_desktop if tail_key == "desktop" else _tail_mic
        if tail.size > 0:
            arr_proc = np.concatenate([tail, arr], axis=0)
        else:
            arr_proc = arr

        try:
            # Submit ASR job but do not block the recorder loop waiting for it.
            # This prevents a stuck decode from halting subsequent chunks/rows.
            submit = True
            if name == "desktop":
                submit = os.getenv("TRANSCRIBE_DESKTOP", "1") == "1"
            elif name == "mic":
                submit = os.getenv("TRANSCRIBE_MIC", "1") == "1"
            if submit:
                coll_meta["submit_ts"] = time.time()
                _pool.submit(_transcribe_and_dispatch, name, seq, arr_proc.copy(), coll_meta, tail)
            # Compute next tail immediately and update, independent of ASR completion
            overlap_frames = int(max(0.0, OVERLAP_SECS) * SAMPLE_RATE)
            next_tail = arr[-overlap_frames:].copy() if overlap_frames > 0 and arr.size >= overlap_frames else np.zeros(0, dtype=np.float32)
            with _tail_lock:
                if tail_key == "desktop":
                    _tail_desktop = next_tail
                else:
                    _tail_mic = next_tail
        except Exception as e:
            print(f"[whisper_worker] failed to queue ASR job ({name}):", e)

        seq += 1

# -----------------------------
# Bootstrap
# -----------------------------
def start_worker():
    global _pool
    _reset_diagnostics()
    # Try to sync ports with plugin.ini if it exists so plugin and service match
    try:
        base = APP_BASE
        ini_path = os.path.join(base, "plugin.ini")
        if os.path.exists(ini_path):
            txt = open(ini_path, "r", encoding="utf-8").read()
            lines = [x.strip() for x in txt.splitlines() if x.strip()]
            ini_ports = {}
            for ln in lines:
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    ini_ports[k.strip()] = v.strip()
            dp = int(ini_ports.get("desktop_port", AUDIO_BRIDGE_PORT_DESKTOP))
            mp = int(ini_ports.get("mic_port", AUDIO_BRIDGE_PORT_MIC))
            if dp != AUDIO_BRIDGE_PORT_DESKTOP or mp != AUDIO_BRIDGE_PORT_MIC:
                print(f"[whisper_worker] syncing ports from plugin.ini desktop={dp} mic={mp}")
                globals()["AUDIO_BRIDGE_PORT_DESKTOP"] = dp
                globals()["AUDIO_BRIDGE_PORT_MIC"] = mp
    except Exception as e:
        print("[whisper_worker] failed to sync ports from plugin.ini:", e)
    # Initialize status snapshot for installer diagnostics
    try:
        _write_status({
            "bridges": {
                "desktop": {"listening": False, "connected": False, "port": AUDIO_BRIDGE_PORT_DESKTOP},
                "mic": {"listening": False, "connected": False, "port": AUDIO_BRIDGE_PORT_MIC},
            },
            "model_ready": False,
            "last_error": None,
        })
    except Exception:
        pass
    # Start bridges first so OBS can connect while model loads
    threading.Thread(target=_bridge_loop, args=(audio_queue_desktop, AUDIO_BRIDGE_PORT_DESKTOP, "desktop"), daemon=True).start()
    threading.Thread(target=_bridge_loop, args=(audio_queue_mic, AUDIO_BRIDGE_PORT_MIC, "mic"), daemon=True).start()
    # Start OBS-driven shutdown monitor in the background (Windows)
    threading.Thread(target=_shutdown_event_monitor, daemon=True).start()
    # Optional idle-exit (disabled by default); can be enabled via EXIT_WHEN_IDLE=1
    threading.Thread(target=_idle_exit_monitor, daemon=True).start()
    # Session heartbeat for diagnostics
    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    _pool = ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS)
    threading.Thread(target=recorder_loop, args=("desktop", audio_queue_desktop, "desktop"), daemon=True).start()
    threading.Thread(target=recorder_loop, args=("mic", audio_queue_mic, "mic"), daemon=True).start()
    _write_bridge_info()
    print(f"[whisper_worker] recorder+pool running (workers={TRANSCRIBE_WORKERS}, chunk_secs={CHUNK_SECS}, overlap_secs={OVERLAP_SECS})")
    # Load model in the background to avoid blocking bridge acceptance
    threading.Thread(target=ensure_model_loaded, daemon=True).start()

if __name__ == "__main__":
    start_worker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass






def _auto_ct2_threads() -> int:
    """Derive a sensible default for CT2 CPU threads based on logical cores."""
    logical = os.cpu_count() or 4
    # Target roughly the number of physical cores (~50% of logical threads)
    default = max(1, math.ceil(logical * 0.5))
    # Clamp to a safe universal range
    default = max(4, min(12, default))
    return default





