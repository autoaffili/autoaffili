# context_filter.py
import os
import time
import re
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

import requests

BACKEND_BASE = os.getenv(
    "BACKEND_BASE_URL",
    "https://autoaffili-backend-802674334607.us-east4.run.app",
).rstrip("/")

MERGED_CONTEXT_CHUNKS = max(1, int(os.getenv("MERGED_CONTEXT_CHUNKS", "2")))

"""Heuristics removed: always call backend in multi-source path."""

RECENT_TRANSCRIPTS: deque = deque(maxlen=max(1, MERGED_CONTEXT_CHUNKS))
# Per-source rolling buffers for multi-source processing
RECENT_BY_SOURCE: Dict[str, deque] = {
    "desktop": deque(maxlen=max(1, MERGED_CONTEXT_CHUNKS)),
    "mic": deque(maxlen=max(1, MERGED_CONTEXT_CHUNKS)),
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Note: legacy local heuristics removed as requested.


def _dedupe_keywords(kws: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for kw in kws:
        cleaned = str(kw).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


def _select_keyword(kws: List[str]) -> Optional[str]:
    return kws[0] if kws else None

def _canonicalize_keyword(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or None


def call_backend_extract(text: str) -> Tuple[bool, List[str], float, int]:
    """
    Call the backend /extract_multi endpoint and return:
        - is_product_mention (bool)
        - keywords (list of str)
        - latency in milliseconds (float)
        - token usage estimate (int)
    """
    payload_tokens_est = _estimate_tokens(text) + 64  # prompt + completion budget
    started = time.perf_counter()
    latency_ms = None
    try:
        resp = requests.post(f"{BACKEND_BASE}/extract_multi", json={"segments": [{"source": "mic", "text": text, "start": None, "end": None}]}, timeout=12)
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.status_code == 200:
            j = resp.json()
            is_prod = bool(j.get("is_product_mention", False))
            kws = j.get("keywords", [])
            if isinstance(kws, str):
                kws = [part.strip() for part in kws.split(",") if part.strip()]
            elif isinstance(kws, (list, tuple)):
                kws = [str(k).strip() for k in kws if str(k).strip()]
            else:
                kws = []
            usage = j.get("usage") if isinstance(j, dict) else None
            token_usage = None
            if isinstance(usage, dict):
                token_usage = usage.get("total_tokens") or usage.get("prompt_tokens")
            estimated_tokens = token_usage or payload_tokens_est
            return is_prod, _dedupe_keywords(kws), latency_ms, int(estimated_tokens)
        print(f"[context_filter] backend returned non-200: {resp.status_code} {resp.text}")
    except Exception as e:
        latency_ms = (time.perf_counter() - started) * 1000
        print("[context_filter] backend extract error:", e)
    return False, [], latency_ms, payload_tokens_est


def call_backend_extract_multi(segments: List[Dict[str, object]]) -> Tuple[bool, List[str], float, int]:
    """
    POST /extract_multi with structured segments.
    segments: [{source, text, start, end}]
    """
    payload_tokens_est = _estimate_tokens("\n".join(str(s.get("text", "")) for s in segments)) + 64
    started = time.perf_counter()
    latency_ms = None
    try:
        resp = requests.post(f"{BACKEND_BASE}/extract_multi", json={"segments": segments}, timeout=12)
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.status_code == 200:
            j = resp.json()
            is_prod = bool(j.get("is_product_mention", False))
            kws = j.get("keywords", [])
            if isinstance(kws, str):
                kws = [part.strip() for part in kws.split(",") if part.strip()]
            elif isinstance(kws, (list, tuple)):
                kws = [str(k).strip() for k in kws if str(k).strip()]
            else:
                kws = []
            usage = j.get("usage") if isinstance(j, dict) else None
            token_usage = None
            if isinstance(usage, dict):
                token_usage = usage.get("total_tokens") or usage.get("prompt_tokens")
            estimated_tokens = token_usage or payload_tokens_est
            return is_prod, _dedupe_keywords(kws), latency_ms, int(estimated_tokens)
        print(f"[context_filter] backend returned non-200 (multi): {resp.status_code} {resp.text}")
    except Exception as e:
        latency_ms = (time.perf_counter() - started) * 1000
        print("[context_filter] backend extract_multi error:", e)
    return False, [], latency_ms, payload_tokens_est


def process_transcript(text: str, diag: Optional[Dict] = None) -> Dict[str, object]:
    """
    Merge recent transcripts for context and always defer relevance to the backend model.
    Returns metrics for diagnostics.
    """
    text = (text or "").strip()
    metrics = {
        "backend_call": False,
        "backend_reason": "",
        "backend_latency_ms": None,
        "token_estimate": 0,
    }
    if not text:
        metrics["backend_reason"] = "empty_transcript"
        return metrics

    RECENT_TRANSCRIPTS.append(text)
    merged = " ".join(RECENT_TRANSCRIPTS)
    metrics["token_estimate"] = _estimate_tokens(merged)
    metrics["backend_reason"] = "auto"

    ok, kws, latency_ms, token_estimate = call_backend_extract(merged)
    metrics["backend_call"] = True
    metrics["backend_latency_ms"] = round(latency_ms, 2) if latency_ms is not None else None
    metrics["token_estimate"] = token_estimate

    if ok and kws:
        keyword = _select_keyword(kws)
        if keyword:
            try:
                from link_generator import handle_keywords
                handle_keywords(keyword, merged)
                print(f"[context_filter] product mention -> '{keyword}'")
            except Exception as e:
                print("[context_filter] error calling link_generator:", e)
        return metrics

    print("[context_filter] no product mention detected")
    return metrics


def process_transcript_multi(source: str, text: str, start_ts: float, end_ts: float, diag: Optional[Dict] = None) -> Dict[str, object]:
    """
    Keep per-source rolling context and call backend with structured segments
    from both sources for richer context. Only the primary source (mic) triggers
    the backend call; other sources simply update the rolling window.
    """
    text = (text or "").strip()
    metrics = {
        "backend_call": False,
        "backend_reason": "",
        "backend_latency_ms": None,
        "token_estimate": 0,
    }
    src = (source or "").strip().lower()
    # Respect upstream timeout/drop signal to skip backend entirely for this chunk
    try:
        if isinstance(diag, dict):
            note = str(diag.get("note", "")).strip().lower()
            if note == "timeout_drop":
                metrics["backend_reason"] = "timeout_drop"
                metrics["decision"] = "skip"
                return metrics
    except Exception:
        pass
    PRIMARY_SOURCE = "mic"
    # Unconditional mic backend call: allow empty text when src is mic; otherwise skip
    if not text and src != PRIMARY_SOURCE:
        metrics["backend_reason"] = "empty_transcript"
        metrics["decision"] = "skip"
        return metrics
    if src not in RECENT_BY_SOURCE:
        RECENT_BY_SOURCE[src] = deque(maxlen=max(1, MERGED_CONTEXT_CHUNKS))

    RECENT_BY_SOURCE[src].append({
        "source": src,
        "text": text,
        "start": float(start_ts or 0.0),
        "end": float(end_ts or 0.0),
    })

    # Merge recent segments across both sources
    segs: List[Dict[str, object]] = []
    for sname, dq in RECENT_BY_SOURCE.items():
        for item in dq:
            segs.append(dict(item))
    # Sort by start time for readability
    try:
        segs.sort(key=lambda s: s.get("start", 0.0))
    except Exception:
        pass

    metrics["token_estimate"] = _estimate_tokens("\n".join(s.get("text", "") for s in segs))
    if src != PRIMARY_SOURCE:
        metrics["backend_reason"] = "skip_source"
        metrics["backend_call"] = False
        metrics["decision"] = "skip"
        return metrics

    metrics["backend_reason"] = "multi_auto"

    ok, kws, latency_ms, token_estimate = call_backend_extract_multi(segs)
    metrics["backend_call"] = True
    metrics["backend_latency_ms"] = round(latency_ms, 2) if latency_ms is not None else None
    metrics["token_estimate"] = token_estimate
    # Simple decision: trust backend is_product_mention; if keyword present, send (throttle still applies)
    chosen_kw: Optional[str] = _canonicalize_keyword(_select_keyword(kws))
    if ok and chosen_kw:
        try:
            from link_generator import handle_keywords
            handle_keywords(chosen_kw, "\n".join(s.get("text", "") for s in segs))
            print(f"[context_filter] product mention -> '{chosen_kw}' (multi)")
        except Exception as e:
            print("[context_filter] error calling link_generator (multi):", e)
        metrics.update({
            "keyword": chosen_kw,
            "decision": "sent",
        })
        return metrics

    reason = "no_product" if not ok else "no_keyword"
    print(f"[context_filter] no product mention detected (multi) or gated: reason={reason} kw={chosen_kw!r}")
    metrics.update({
        "keyword": chosen_kw,
        "decision": "blocked",
        "backend_reason": "multi_auto",
    })
    return metrics


