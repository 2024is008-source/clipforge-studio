"""
OpenShorts-inspired 5-Gemini-API-Key Pool & Automatic Failover Rotation Manager.
Supports up to 5 Gemini Free Tier API keys (GEMINI_API_KEY_1 .. GEMINI_API_KEY_5)
with round-robin job/chapter assignment, automatic 429/quota failover rotation,
and per-key health diagnostics (persisted in key-health.json).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import WORK_DIR

T = TypeVar("T")

MAX_GEMINI_KEYS = 5
KEYS_STORE_PATH = Path(WORK_DIR) / "gemini_keys_pool.json"
KEY_HEALTH_PATH = Path(WORK_DIR) / "key-health.json"

_job_counter = itertools.count()
_pool_lock = threading.Lock()


def mask_api_key(key: str) -> str:
    """Return a safe masked representation of an API key (never log raw keys)."""
    k = (key or "").strip()
    if not k:
        return "Not set"
    if len(k) <= 10:
        return "****"
    return f"{k[:6]}…{k[-4:]}"


def _read_saved_keys() -> list[str]:
    try:
        if KEYS_STORE_PATH.is_file():
            data = json.loads(KEYS_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("keys"), list):
                return [str(x).strip() for x in data["keys"][:MAX_GEMINI_KEYS]]
    except Exception:
        pass
    return [""] * MAX_GEMINI_KEYS


def save_key_slots(slots: list[str], model: str | None = None, parallel_workers: int | None = None) -> dict[str, Any]:
    """Persist up to 5 Gemini API key slots to local worker storage and update process env."""
    normalized = [(slots[i].strip() if i < len(slots) and isinstance(slots[i], str) else "") for i in range(MAX_GEMINI_KEYS)]
    payload: dict[str, Any] = {"keys": normalized, "updated_at": time.time()}
    if model:
        payload["model"] = model.strip()
        os.environ["GEMINI_VIDEO_MODEL"] = model.strip()
    if parallel_workers is not None:
        w = max(1, min(MAX_GEMINI_KEYS, int(parallel_workers)))
        payload["parallel_workers"] = w
        os.environ["GEMINI_ANALYSIS_WORKERS"] = str(w)

    with _pool_lock:
        KEYS_STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Sync environment variables GEMINI_API_KEY_1..5 and primary GEMINI_API_KEY
        active = [k for k in normalized if k]
        for i in range(MAX_GEMINI_KEYS):
            env_name = f"GEMINI_API_KEY_{i + 1}"
            if normalized[i]:
                os.environ[env_name] = normalized[i]
            else:
                os.environ.pop(env_name, None)
        if active:
            os.environ["GEMINI_API_KEY"] = active[0]
            os.environ["GEMINI_KEY_POOL"] = "1"
    return get_pool_status()


def numbered_names(env: dict[str, str] | os._Environ[str]) -> list[str]:
    return sorted(
        (n for n in env if re.fullmatch(r"GEMINI_API_KEY_[1-9][0-9]?", n)),
        key=lambda n: int(n.rsplit("_", 1)[1]),
    )


def get_five_key_slots(env: dict[str, str] | None = None) -> list[str]:
    """Return the exact 5 slots [slot1, slot2, slot3, slot4, slot5] combining saved pool and env."""
    source_env = os.environ if env is None else env
    saved = _read_saved_keys()
    slots: list[str] = []
    for i in range(MAX_GEMINI_KEYS):
        env_val = source_env.get(f"GEMINI_API_KEY_{i + 1}", "").strip()
        if i == 0 and not env_val:
            env_val = source_env.get("GEMINI_API_KEY", "").strip()
        val = saved[i] if (i < len(saved) and saved[i]) else env_val
        slots.append(val)
    return slots


def configured(env: dict[str, str] | None = None) -> list[str]:
    """Return deduplicated list of active Gemini API keys (up to 5 keys)."""
    source_env = os.environ if env is None else env
    slots = get_five_key_slots(source_env)
    names = ["GEMINI_API_KEY", *numbered_names(source_env)]
    for name in names:
        v = source_env.get(name, "").strip()
        if v and v not in slots and len(slots) < MAX_GEMINI_KEYS:
            slots.append(v)
    unique: list[str] = []
    for k in slots:
        if k and k not in unique:
            unique.append(k)
    return unique[:MAX_GEMINI_KEYS]


def assign_job(env: dict[str, str] | None = None) -> dict[str, str]:
    """Round-robin primary key per job/chapter, retaining remaining keys as ordered failover."""
    target = dict(os.environ if env is None else env)
    keys = configured(target)
    if not keys:
        return target
    with _pool_lock:
        index = next(_job_counter) % len(keys)
    rotated = keys[index:] + keys[:index]
    for name in numbered_names(target):
        target.pop(name, None)
    target.update({f"GEMINI_API_KEY_{i + 1}": key for i, key in enumerate(rotated)})
    target["GEMINI_API_KEY"] = rotated[0]
    return target


def rotated_keys_for_task(offset: int | None = None) -> list[str]:
    """Return all configured keys rotated by a round-robin offset (or explicit chapter offset)."""
    keys = configured()
    if not keys:
        return []
    if offset is None:
        with _pool_lock:
            offset = next(_job_counter)
    idx = offset % len(keys)
    return keys[idx:] + keys[:idx]


def is_key_retryable(exc: BaseException | str) -> bool:
    """
    True when rotating to the next Gemini API key in the 5-key pool is safe and effective.
    Matches OpenShorts' _gemini_is_key_retryable: rotates on 429/quota/auth errors,
    never on content policy blocks.
    """
    msg = str(exc).upper()
    return any(
        tok in msg
        for tok in (
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE LIMIT",
            "QUOTA",
            "401",
            "403",
            "API_KEY",
            "API KEY",
            "PERMISSION_DENIED",
        )
    )


def _load_health() -> dict[str, Any]:
    try:
        if KEY_HEALTH_PATH.is_file():
            data = json.loads(KEY_HEALTH_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def record_key_outcome(
    key: str,
    ok: bool,
    latency_ms: float = 0.0,
    error_msg: str | None = None,
) -> None:
    """Record per-key health metrics without ever storing the raw secret key on disk."""
    masked = mask_api_key(key)
    with _pool_lock:
        health = _load_health()
        entry = health.get(masked, {"requests": 0, "successes": 0, "rate_limits": 0})
        entry["requests"] = int(entry.get("requests", 0)) + 1
        if ok:
            entry["successes"] = int(entry.get("successes", 0)) + 1
            entry["status"] = "healthy"
            entry["last_error"] = None
            entry["cooldown_until"] = 0.0
        else:
            err_upper = (error_msg or "").upper()
            if "429" in err_upper or "RESOURCE_EXHAUSTED" in err_upper or "QUOTA" in err_upper:
                entry["rate_limits"] = int(entry.get("rate_limits", 0)) + 1
                entry["status"] = "rate_limited"
                entry["cooldown_until"] = time.time() + 60.0
            else:
                entry["status"] = "error"
            entry["last_error"] = (error_msg or "Request failed")[:160]
        entry["latency_ms"] = round(latency_ms, 1)
        entry["updated_at"] = time.time()
        health[masked] = entry
        try:
            KEY_HEALTH_PATH.write_text(json.dumps(health, indent=2), encoding="utf-8")
        except Exception:
            pass


def get_pool_status() -> dict[str, Any]:
    """Return status of all 5 Gemini API key slots for the Settings UI."""
    slots = get_five_key_slots()
    health = _load_health()
    now = time.time()
    slot_infos = []
    configured_count = 0
    healthy_count = 0
    for idx, raw_key in enumerate(slots):
        has_key = bool(raw_key and len(raw_key.strip()) > 10)
        masked = mask_api_key(raw_key) if has_key else "Not set"
        h = health.get(masked, {}) if has_key else {}
        status = h.get("status", "ready" if has_key else "empty")
        cooldown_until = float(h.get("cooldown_until", 0.0) or 0.0)
        if status == "rate_limited" and now >= cooldown_until:
            status = "ready"
        if has_key:
            configured_count += 1
            if status in ("healthy", "ready"):
                healthy_count += 1
        slot_infos.append(
            {
                "slot": idx + 1,
                "env_name": f"GEMINI_API_KEY_{idx + 1}",
                "configured": has_key,
                "masked": masked,
                "status": status,
                "requests": int(h.get("requests", 0)),
                "successes": int(h.get("successes", 0)),
                "rate_limits": int(h.get("rate_limits", 0)),
                "latency_ms": h.get("latency_ms"),
                "last_error": h.get("last_error"),
            }
        )

    saved_meta = {}
    try:
        if KEYS_STORE_PATH.is_file():
            saved_meta = json.loads(KEYS_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass

    model = (
        saved_meta.get("model")
        or os.environ.get("GEMINI_VIDEO_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-2.5-flash"
    )
    workers = int(
        saved_meta.get("parallel_workers")
        or os.environ.get("GEMINI_ANALYSIS_WORKERS")
        or max(1, min(configured_count, MAX_GEMINI_KEYS))
    )
    return {
        "max_keys": MAX_GEMINI_KEYS,
        "configured_count": configured_count,
        "healthy_count": healthy_count,
        "model": model,
        "parallel_workers": max(1, min(MAX_GEMINI_KEYS, workers)),
        "effective_rpm_multiplier": max(1, configured_count),
        "slots": slot_infos,
    }


def execute_with_key_rotation(
    call_fn: Callable[[str, int], T],
    chapter_offset: int | None = None,
    max_attempts_per_key: int = 2,
) -> T:
    """
    Execute a Gemini operation across the 5-key pool.
    - Starts at the round-robin or chapter-assigned key slot.
    - Retries transient 503/network errors on the same key before rotating.
    - Immediately rotates to the next key in the pool on 429 / quota / auth errors.
    """
    keys = rotated_keys_for_task(chapter_offset)
    if not keys:
        raise RuntimeError(
            "No Gemini API key configured. Add up to 5 keys (GEMINI_API_KEY_1..5) in Settings → AI & Processing."
        )

    # Prefer keys not currently in a 60s 429 cooldown
    health = _load_health()
    now = time.time()
    ready_keys = [
        k for k in keys if float(health.get(mask_api_key(k), {}).get("cooldown_until", 0.0) or 0.0) <= now
    ]
    ordered_keys = ready_keys + [k for k in keys if k not in ready_keys]

    last_error: Exception | None = None
    for key_idx, api_key in enumerate(ordered_keys):
        for attempt in range(1, max_attempts_per_key + 1):
            t0 = time.monotonic()
            try:
                result = call_fn(api_key, key_idx)
                record_key_outcome(api_key, ok=True, latency_ms=(time.monotonic() - t0) * 1000.0)
                return result
            except Exception as exc:
                last_error = exc
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                record_key_outcome(api_key, ok=False, latency_ms=elapsed_ms, error_msg=str(exc))
                if is_key_retryable(exc):
                    # Rotate immediately to the next key in the 5-key pool
                    break
                if attempt < max_attempts_per_key:
                    time.sleep(1.0 * attempt)
                    continue
    raise last_error or RuntimeError("All configured Gemini API keys failed.")
