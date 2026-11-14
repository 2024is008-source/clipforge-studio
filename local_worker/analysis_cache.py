"""Process-local finalized analysis LRU; callers serialize access with ANALYSIS_LOCK.

No analysis/model imports, inference, downloads, or persistent result files. Code and
package updates invalidate the cache by restarting the worker. Keep the explicit
settings/checkpoint lists in sync with the analysis pipeline when adding options.
"""
from __future__ import annotations

import json
import os
import stat
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from . import config

MAX_ENTRIES = 8
MAX_BYTES = 32 * 1024 * 1024

# openshorts_engine, tuning, layout_decision and detection_service settings only.
# In particular, exclude tokens, API keys, and YOLO_OFFLINE/YOLO_AUTOINSTALL (which
# the model loader mutates). SAM2/Whisper/render settings do not affect this pass.
ANALYSIS_ENV_NAMES = (
    "DETECT_STRIDE", "JUMP_CONFIRM_FRAMES", "SCENE_CUT_RESET", "SPLIT_TIGHTNESS",
    "CAM_DEADBAND_X", "CAM_DEADBAND_Y", "CAM_TAU_X", "CAM_TAU_Y",
    "CAM_MAX_SPEED_X", "CAM_MAX_SPEED_Y", "CUT_THR_MIN", "CUT_THR_MAX",
    "CUT_THR_SIGMAS", "PUNCH_IN_ZOOM", "REFRAME_SAMPLE_FPS", "REFRAME_OBJECT_CLASSES",
    "HOLD_MIN_SEC", "HOLD_MAX_SEC", "SPLIT_MIN_HOLD_SEC",
    "LAYOUT_CONFIDENCE", "LAYOUT_VISIBILITY", "LAYOUT_MAX_PANEL_OVERLAP",
    "LAYOUT_DETECTION_SHARE", "LAYOUT_SWITCH_MARGIN",
    "YOLO_DEVICE", "YOLO_IMAGE_SIZE", "YOLO_CONFIDENCE",
    "FACE_MODEL_PATH", "YOLO_MODEL_PATH", "YOLO_POSE_MODEL_PATH", "CLIPFORGE_WORK_DIR",
)
_JOB_FIELDS = frozenset(("job_id", "source_id", "elapsed_sec", "cache_hit"))
_FACE_BACKENDS = frozenset(("mediapipe", "yunet", "haar_frontal_profile"))
_PERSON_BACKENDS = frozenset(("ready", "ready_cpu_fallback"))


def _file_identity(path: Path) -> tuple:
    """Stat, never read weights; retain missing paths so later availability misses."""
    try:
        path = path.resolve()
        info = path.stat()
        if stat.S_ISREG(info.st_mode):
            return (str(path), (info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns))
    except (OSError, ValueError, RuntimeError):
        pass
    return (str(path), None)


def _checkpoint_identities(settings: dict) -> tuple:
    # Face discovery falls back from the configured path to MODELS_DIR then CWD.
    # YOLO uses the same defaults unless explicitly configured; pose is opt-in.
    # Fingerprint all candidates, including missing and currently shadowed files.
    candidates = []
    face = settings["FACE_MODEL_PATH"]
    if face:
        candidates.append(Path(face))
    for name in ("YOLO_MODEL_PATH", "YOLO_POSE_MODEL_PATH"):
        configured = (settings[name] or "").strip()
        if configured:
            try:
                candidates.append(Path(configured).expanduser())
            except RuntimeError:
                candidates.append(Path(configured))
    for directory in (Path(config.MODELS_DIR), Path.cwd()):
        candidates.extend(directory / filename for filename in
                          ("face_detection_yunet_2023mar.onnx", "yolov8n.pt"))
    return tuple(_file_identity(path) for path in candidates)


def key_for(source_path: str | os.PathLike, start: float, end: float | None,
            framing_mode: str) -> tuple | None:
    """Return an immutable request fingerprint, or None for an unreadable source.

    Uses requested (not clamped) times, source path/size/nanosecond timestamps,
    allowlisted environment values and current local checkpoint identities. Runtime
    detector readiness is deliberately absent, so lazy initialization cannot miss.
    """
    source = _file_identity(Path(source_path))
    if source[1] is None:
        return None
    settings = {name: os.environ.get(name) for name in ANALYSIS_ENV_NAMES}
    return (source, start, end, framing_mode, tuple(settings.items()),
            _checkpoint_identities(settings))


def _is_finalized(result: dict) -> bool:
    if not isinstance(result, dict) or result.get("ok") is not True:
        return False
    if any(result.get(name) for name in ("error", "partial", "cancelled")):
        return False
    if result.get("finalized") is False:
        return False
    if result.get("status") not in (None, "success", "completed", "done", "ready", "ready_cpu_fallback"):
        return False
    detectors = result.get("detectors")
    if not isinstance(detectors, dict):
        return False
    face, person = detectors.get("face"), detectors.get("person")
    if not isinstance(face, str) or not isinstance(person, str):
        return False
    if face not in _FACE_BACKENDS or person not in _PERSON_BACKENDS:
        return False
    track, layouts = result.get("track"), result.get("layout_decisions")
    # finalize_layouts annotates every delivered point. Do not cache an earlier
    # tracking/smoothing pass, even if it has a nonempty track and ok=True.
    return (
        isinstance(track, list) and bool(track)
        and all(isinstance(point, dict) and {"t", "x", "y", "mode"} <= point.keys()
                and isinstance(point.get("layoutDecision"), dict)
                and {"scene", "reason"} <= point["layoutDecision"].keys() for point in track)
        and isinstance(layouts, list) and bool(layouts)
        and all(isinstance(layout, dict) and {"start", "end", "layout"} <= layout.keys()
                for layout in layouts)
    )


class AnalysisCache:
    """Deep-copy LRU bounded by entry count AND compact UTF-8 JSON payload bytes.

    Bounds may be reduced for tests, never increased beyond 8 entries / 32 MiB.
    No internal lock: main owns the get/analyze/put critical section. Failed,
    incomplete, non-JSON and oversized puts are no-ops (including replacements).
    """

    key_for = staticmethod(key_for)

    def __init__(self, *, max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES) -> None:
        if not 1 <= max_entries <= MAX_ENTRIES or not 1 <= max_bytes <= MAX_BYTES:
            raise ValueError("Cache limits must be positive and at most 8 entries / 32 MiB")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple, tuple[dict, int]] = OrderedDict()
        self._bytes = 0

    def get(self, key: tuple | None) -> dict | None:
        if key is None or key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return deepcopy(self._entries[key][0])

    def put(self, key: tuple | None, result: dict) -> None:
        if key is None or not _is_finalized(result):
            return
        payload = {name: value for name, value in result.items() if name not in _JOB_FIELDS}
        try:
            size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError, OverflowError, RecursionError):
            return
        if size > self._max_bytes:
            return
        # Retain the dict, not a JSON round-trip: preserve nested types/keys while
        # still accounting for the actual serialized byte size of the response.
        stored = deepcopy(payload)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        while self._entries and (len(self._entries) >= self._max_entries
                                 or self._bytes + size > self._max_bytes):
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self._bytes -= evicted_size
        self._entries[key] = (stored, size)
        self._bytes += size

    def clear(self) -> None:
        """Drop all entries and reset byte accounting (primarily for tests)."""
        self._entries.clear()
        self._bytes = 0


ANALYSIS_CACHE = AnalysisCache()
