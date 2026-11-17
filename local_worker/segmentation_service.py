"""Explicit, local-only SAM 2 video segmentation; never called by ordinary exports.

The HTTP adapter must resolve an authorized source_id before calling this module;
video_path is a trusted worker-local path, not an HTTP request field. Invalid
requests raise ValueError. Unavailable dependencies/resources and cancellation
return available=False with no frames. progress_cb receives fractions in [0, 1].

Segments are half-open [start, end). Sampling selects unique source frames nearest
a 3 Hz grid, restricted to the segment, and records their decoded source PTS.
This OpenCV adapter requires usable frame-count/FPS metadata and monotonic PTS;
it does not manufacture timestamps when the decoder cannot supply them. Prompts
snap to the nearest sampled timestamp (earlier on ties); collisions for the same
object are rejected. Tracking starts at each object's first supplied prompt and
loss is latched until another supplied prompt. No confidence is inferred.

Successful results describe bool masks at mask_width x mask_height, not source
resolution. bounds are normalized XYWH; area is mask pixels. mask_ref and
metadata_ref are relative to WORK_DIR. Read NPZ members using mask_key and
allow_pickle=False. References are ephemeral: the eight-entry cache can evict
older requests. One process owns the cache/model; the lock is process-local.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager

from .config import WORK_DIR

MAX_SEGMENT_SECONDS = 10.0
SAMPLE_FPS = 3.0
MAX_FRAMES = 32
MAX_SIDE = 512
MAX_OBJECTS = 8
MAX_PROMPTS = 64
MAX_POINTS_PER_PROMPT = 32
MAX_TOTAL_POINTS = 256
MAX_CACHE_ENTRIES = 8
_CACHE_VERSION = 1
_LOCK = threading.Lock()
_PREDICTOR = None
_PREDICTOR_KEY = None


class _Cancelled(Exception):
    pass


class _Unavailable(Exception):
    pass


def _check_cancel(is_cancelled):
    if is_cancelled is not None and is_cancelled():
        raise _Cancelled()


@contextmanager
def _operation_lock(is_cancelled):
    # A request queued behind inference must remain cancellable.
    while True:
        _check_cancel(is_cancelled)
        if _LOCK.acquire(timeout=0.1):
            break
    try:
        _check_cancel(is_cancelled)
        yield
    finally:
        _LOCK.release()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _coordinates(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} coordinates")
    result = [_number(v, name) for v in value]
    if any(v < 0 or v > 1 for v in result):
        raise ValueError(f"{name} coordinates must be normalized to [0, 1]")
    return result


def validate_request(start, end, prompts):
    start, end = _number(start, "start"), _number(end, "end")
    if start < 0 or not 0 < end - start <= MAX_SEGMENT_SECONDS:
        raise ValueError("segment must satisfy 0 <= start < end and last at most 10 seconds")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"provide 1..{MAX_PROMPTS} prompts")
    clean, objects, seen, total_points = [], set(), set(), 0
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("each prompt must be an object")
        object_id = prompt.get("object_id")
        if type(object_id) is not int or not 0 <= object_id <= 2**31 - 1:
            raise ValueError("object_id must be a nonnegative 32-bit integer")
        t = _number(prompt.get("t"), "prompt t")
        if not start <= t < end:
            raise ValueError("prompt t must lie in [start, end)")
        if (object_id, t) in seen:
            raise ValueError("duplicate object/keyframe prompt")
        seen.add((object_id, t))
        objects.add(object_id)
        if len(objects) > MAX_OBJECTS:
            raise ValueError(f"at most {MAX_OBJECTS} objects are allowed")
        item = {"object_id": object_id, "t": t}
        if "box" in prompt:
            if set(prompt) != {"object_id", "t", "box"}:
                raise ValueError("box prompts cannot include points, labels, or extra fields")
            x, y, w, h = _coordinates(prompt["box"], 4, "box")
            if w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
                raise ValueError("box must have positive area and fit inside the image")
            item["box"] = [x, y, w, h]
        else:
            if set(prompt) != {"object_id", "t", "points", "labels"}:
                raise ValueError("provide either a box or points and labels")
            points, labels = prompt["points"], prompt["labels"]
            if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS_PER_PROMPT:
                raise ValueError(f"provide 1..{MAX_POINTS_PER_PROMPT} points per prompt")
            if not isinstance(labels, list) or len(labels) != len(points):
                raise ValueError("point and label counts must match")
            if any(type(label) is not int or label not in (0, 1) for label in labels):
                raise ValueError("labels must be integer 0 or 1")
            item["points"] = [_coordinates(p, 2, "point") for p in points]
            item["labels"] = list(labels)
            total_points += len(points)
            if total_points > MAX_TOTAL_POINTS:
                raise ValueError(f"at most {MAX_TOTAL_POINTS} total points are allowed")
        clean.append(item)
    return start, end, sorted(clean, key=lambda p: (p["t"], p["object_id"]))


def _stat_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise OSError("not a regular file")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}


def _settings():
    checkpoint = os.environ.get("SAM2_MODEL_PATH", "").strip()
    config = os.environ.get("SAM2_CONFIG", "").strip()
    device = os.environ.get("SAM2_DEVICE", "auto").strip().lower()
    if not checkpoint or not config:
        raise _Unavailable("Set SAM2_MODEL_PATH to local weights and SAM2_CONFIG to a local Hydra config name")
    if device not in ("auto", "cpu", "cuda"):
        raise _Unavailable("SAM2_DEVICE must be auto, cpu, or cuda")
    if "://" in checkpoint or "://" in config or "\x00" in config:
        raise _Unavailable("SAM 2 requires local weights and a local Hydra config")
    try:
        identity = _stat_identity(checkpoint)
    except (OSError, ValueError):
        raise _Unavailable("SAM2_MODEL_PATH is not an accessible local checkpoint") from None
    if identity["size"] == 0:
        raise _Unavailable("SAM2_MODEL_PATH is an empty checkpoint")
    return config, identity, device


def _runtime(device):
    try:
        torch = importlib.import_module("torch")
        sam2 = importlib.import_module("sam2")
        builder = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    except (ImportError, OSError, AttributeError):
        raise _Unavailable("Original SAM 2 and PyTorch packages are unavailable; install them separately") from None
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise _Unavailable("SAM2_DEVICE=cuda requested but CUDA is unavailable")
    return torch, sam2, builder, device


def _config_identity(config, sam2):
    # Hydra usually resolves a package-relative name, not a working-directory path.
    roots = [Path(getattr(sam2, "__file__", "") or ".").parent, Path.cwd()]
    for root in roots:
        candidate = root / config
        for path in (candidate, candidate.with_suffix(".yaml")):
            try:
                return {"name": config, "file": _stat_identity(path)}
            except (OSError, ValueError):
                pass
    return {"name": config}


def _predictor(builder, config, checkpoint, device, model_key):
    global _PREDICTOR, _PREDICTOR_KEY
    if _PREDICTOR_KEY != model_key or _PREDICTOR is None:
        # Drop the old model before allocating another device/configuration.
        _PREDICTOR, _PREDICTOR_KEY = None, None
        try:
            model = builder(config_file=config, ckpt_path=checkpoint["path"], device=device)
        except Exception as exc:
            raise _Unavailable(f"SAM 2 could not load the local config/checkpoint ({type(exc).__name__})") from None
        _PREDICTOR, _PREDICTOR_KEY = model, model_key
    return _PREDICTOR


def _probe(cap, cv2, start, end):
    if not cap.isOpened():
        raise ValueError("source video cannot be opened")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if any(not math.isfinite(v) or v <= 0 for v in (width, height, fps, count)):
        raise ValueError("source video has unusable dimensions, frame count, or frame rate")
    width, height, count = int(width), int(height), int(count)
    if not width or not height or not count or end > count / fps:
        raise ValueError("segment is outside the source video duration")
    first = math.ceil(start * fps - 1e-9)
    last = min(count - 1, math.ceil(end * fps - 1e-9) - 1)
    if first > last:
        raise ValueError("segment contains no source frames")
    indices = []
    for i in range(min(MAX_FRAMES, math.ceil((end - start) * SAMPLE_FPS))):
        target = start + i / SAMPLE_FPS
        if target >= end:
            break
        index = max(first, min(last, math.floor(target * fps + 0.5)))
        if not indices or index != indices[-1]:
            indices.append(index)
    scale = min(1.0, MAX_SIDE / max(width, height))
    return width, height, max(1, round(width * scale)), max(1, round(height * scale)), indices


def _sample(cap, cv2, directory, dimensions, indices, start, end, cancel, report):
    width, height, mask_width, mask_height = dimensions
    timestamps = []
    for i, index in enumerate(indices):
        _check_cancel(cancel)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise _Unavailable("source video does not support frame seeking")
        ok, frame = cap.read()
        if not ok or frame is None or frame.shape[:2] != (height, width):
            raise _Unavailable("source frame decoding failed or dimensions changed")
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (not math.isfinite(t) or not start <= t < end
                or (timestamps and t <= timestamps[-1])):
            raise _Unavailable("decoder did not provide monotonic source timestamps inside the segment")
        if index > 0 and t == 0:
            raise _Unavailable("decoder did not provide source timestamps")
        timestamps.append(t)
        if frame.shape[:2] != (mask_height, mask_width):
            frame = cv2.resize(frame, (mask_width, mask_height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(directory / f"{i:05d}.jpg"), frame):
            raise _Unavailable("could not write sampled JPEG frames")
        report(0.1 + 0.2 * (i + 1) / len(indices))
    return timestamps


def _map_prompts(prompts, timestamps):
    grouped, mapped = {}, []
    seen = set()
    for prompt in prompts:
        index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - prompt["t"]))
        pair = (index, prompt["object_id"])
        if pair in seen:
            raise ValueError("multiple prompts for one object map to the same sampled frame")
        seen.add(pair)
        grouped.setdefault(index, []).append(prompt)
        mapped.append({"object_id": prompt["object_id"], "t": prompt["t"],
                       "frame_index": index, "sampled_t": timestamps[index]})
    return grouped, mapped


def _as_mask(logits, height, width, np):
    if logits is None:
        return None
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    array = np.asarray(logits)
    if array.shape == (1, height, width):
        array = array[0]
    if array.shape != (height, width) or not np.isfinite(array).all():
        return None
    mask = array > 0
    return mask if mask.any() else None


def _track(predictor, state, grouped, timestamps, object_ids, width, height,
           mask_ref, cancel, report, np):
    frames, masks, active, lost = [], {}, set(), set()

    def record(index, ids=(), logits=None):
        _check_cancel(cancel)
        positions = {int(oid): pos for pos, oid in enumerate(ids)}
        objects = []
        for oid in object_ids:
            item = {"object_id": oid, "status": "not_prompted", "bounds": None,
                    "area": 0, "area_fraction": 0.0, "mask_ref": None, "mask_key": None}
            mask = None
            if oid in active:
                if oid not in lost and oid in positions and logits is not None:
                    pos = positions[oid]
                    if pos < len(logits):
                        mask = _as_mask(logits[pos], height, width, np)
                if mask is None:
                    lost.add(oid)
                    item["status"] = "lost"
                else:
                    ys, xs = np.nonzero(mask)
                    x, y = int(xs.min()), int(ys.min())
                    w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
                    area = int(mask.sum())
                    member = f"f{index:03d}_o{oid}"
                    masks[member] = mask
                    item.update(status="tracked", bounds=[x / width, y / height, w / width, h / height],
                                area=area, area_fraction=area / (width * height),
                                mask_ref=mask_ref, mask_key=member)
            objects.append(item)
        frames.append({"frame_index": index, "t": timestamps[index], "objects": objects})
        report(0.3 + 0.6 * (index + 1) / len(timestamps))

    keyframes = sorted(grouped)
    for i in range(keyframes[0]):
        record(i)
    for n, index in enumerate(keyframes):
        for prompt in grouped[index]:
            _check_cancel(cancel)
            oid = prompt["object_id"]
            kwargs = {"inference_state": state, "frame_idx": index, "obj_id": oid,
                      "normalize_coords": True, "clear_old_points": True}
            if "box" in prompt:
                x, y, w, h = prompt["box"]
                kwargs["box"] = [x * width, y * height, (x + w) * width, (y + h) * height]
            else:
                kwargs["points"] = [[x * width, y * height] for x, y in prompt["points"]]
                kwargs["labels"] = prompt["labels"]
            predictor.add_new_points_or_box(**kwargs)
            active.add(oid)
            lost.discard(oid)
        stop = keyframes[n + 1] if n + 1 < len(keyframes) else len(timestamps)
        # SAM 2's max_frame_num_to_track is an inclusive distance, not a count.
        iterator = predictor.propagate_in_video(
            inference_state=state, start_frame_idx=index,
            max_frame_num_to_track=stop - index - 1, reverse=False)
        next_index = index
        try:
            for frame_index, ids, logits in iterator:
                _check_cancel(cancel)
                if (not isinstance(frame_index, (int, np.integer))
                        or not next_index <= frame_index < stop):
                    raise _Unavailable("SAM 2 returned an out-of-order or out-of-range frame")
                while next_index < frame_index:
                    record(next_index)
                    next_index += 1
                record(int(frame_index), ids, logits)
                next_index = int(frame_index) + 1
            while next_index < stop:
                record(next_index)
                next_index += 1
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    return frames, masks


def _cache_read(directory):
    try:
        with (directory / "metadata.json").open(encoding="utf-8") as stream:
            result = json.load(stream)
        if (result["available"] is True and result["cache_id"] == directory.name
                and (directory / "masks.npz").is_file()):
            os.utime(directory, None)
            return dict(result, cache_hit=True)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _prune_cache(root, keep):
    entries = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
               and re.fullmatch(r"[0-9a-f]{64}", p.name)]
    entries.sort(key=lambda p: (p.name == keep, p.stat().st_mtime_ns), reverse=True)
    for entry in entries[MAX_CACHE_ENTRIES:]:
        shutil.rmtree(entry)


def segment_video(video_path: str, start: float, end: float, prompts: list[dict], *,
                  is_cancelled=None, progress_cb=None) -> dict:
    """Segment an explicitly requested local source; see module contract above."""
    global _PREDICTOR, _PREDICTOR_KEY
    start, end, prompts = validate_request(start, end, prompts)
    if not isinstance(video_path, str) or not video_path or "://" in video_path:
        raise ValueError("video_path must be a trusted local source path")
    for callback in (is_cancelled, progress_cb):
        if callback is not None and not callable(callback):
            raise ValueError("callbacks must be callable")
    try:
        source = _stat_identity(video_path)
    except (OSError, ValueError):
        raise ValueError("source video is not an accessible local file") from None

    last_progress = 0.0

    def report(value):
        nonlocal last_progress
        last_progress = max(last_progress, min(1.0, max(0.0, float(value))))
        if progress_cb is not None:
            progress_cb(last_progress)
        # Completion follows atomic publication; cancellation then is too late.
        if last_progress < 1.0:
            _check_cancel(is_cancelled)

    try:
        with _operation_lock(is_cancelled):
            report(0.0)
            try:
                cv2 = importlib.import_module("cv2")
                np = importlib.import_module("numpy")
            except ImportError:
                raise _Unavailable("OpenCV and NumPy are required for local video sampling") from None
            cap = cv2.VideoCapture(source["path"])
            try:
                width, height, mw, mh, indices = _probe(cap, cv2, start, end)
                config, checkpoint, device_choice = _settings()
                torch, sam2, builder, device = _runtime(device_choice)
                config_id = _config_identity(config, sam2)
                model_key = json.dumps([config_id, checkpoint, device], sort_keys=True)
                identity = {"version": _CACHE_VERSION, "source": source, "model": model_key,
                            "start": start, "end": end, "prompts": prompts,
                            "sampling": [SAMPLE_FPS, MAX_FRAMES, MAX_SIDE]}
                cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                root = Path(WORK_DIR).resolve() / "segmentation"
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination = root / cache_id
                cached = _cache_read(destination)
                if cached is not None:
                    _check_cancel(is_cancelled)
                    _prune_cache(root, cache_id)
                    report(1.0)
                    return cached
                # Only complete directories with metadata and masks become visible.
                with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
                    temporary = Path(temporary)
                    frame_dir = temporary / "frames"
                    frame_dir.mkdir()
                    timestamps = _sample(cap, cv2, frame_dir, (width, height, mw, mh),
                                         indices, start, end, is_cancelled, report)
                    grouped, mapped = _map_prompts(prompts, timestamps)
                    _check_cancel(is_cancelled)
                    predictor = _predictor(builder, config, checkpoint, device, model_key)
                    state = None
                    try:
                        with torch.inference_mode():
                            _check_cancel(is_cancelled)
                            state = predictor.init_state(
                                video_path=str(frame_dir), offload_video_to_cpu=True,
                                offload_state_to_cpu=True)
                            mask_ref = f"segmentation/{cache_id}/masks.npz"
                            frames, masks = _track(
                                predictor, state, grouped, timestamps,
                                sorted({p["object_id"] for p in prompts}), mw, mh,
                                mask_ref, is_cancelled, report, np)
                    except _Cancelled:
                        _PREDICTOR, _PREDICTOR_KEY = None, None
                        raise
                    except Exception as exc:
                        _PREDICTOR, _PREDICTOR_KEY = None, None
                        raise _Unavailable(f"SAM 2 inference failed ({type(exc).__name__}): {str(exc)[:300]}") from None
                    finally:
                        if state is not None:
                            try:
                                predictor.reset_state(state)
                            except Exception as exc:
                                _PREDICTOR, _PREDICTOR_KEY = None, None
                                raise _Unavailable(f"SAM 2 state teardown failed ({type(exc).__name__})") from None
                            finally:
                                # SAM's reset_state does not release images/cached_features.
                                state.clear()
                    result = {"available": True, "cache_id": cache_id, "cache_hit": False,
                              "start": start, "end": end, "sample_fps": SAMPLE_FPS,
                              "original_width": width, "original_height": height,
                              "mask_width": mw, "mask_height": mh,
                              "sampled_timestamps": timestamps, "prompt_frames": mapped,
                              "mask_format": "npz-bool", "frames": frames,
                              "metadata_ref": f"segmentation/{cache_id}/metadata.json"}
                    product = temporary / "product"
                    product.mkdir()
                    _check_cancel(is_cancelled)
                    np.savez_compressed(product / "masks.npz", **masks)
                    with (product / "metadata.json").open("w", encoding="utf-8") as stream:
                        json.dump(result, stream, allow_nan=False, separators=(",", ":"))
                    # Do not publish a result against an identity changed mid-operation.
                    if (source != _stat_identity(source["path"])
                            or checkpoint != _stat_identity(checkpoint["path"])
                            or config_id != _config_identity(config, sam2)):
                        raise _Unavailable("source, checkpoint, or config changed during segmentation; retry")
                    report(0.98)
                    if destination.exists():
                        shutil.rmtree(destination)  # Incomplete/corrupt prior entry, under lock.
                    os.replace(product, destination)
                    _prune_cache(root, cache_id)
                    report(1.0)
                    return result
            finally:
                cap.release()
    except _Cancelled:
        return {"available": False, "reason": "Segmentation cancelled", "cancelled": True, "frames": []}
    except _Unavailable as exc:
        return {"available": False, "reason": str(exc), "frames": []}
    except (ImportError, OSError, RuntimeError, MemoryError) as exc:
        return {"available": False, "reason": f"Local SAM 2 segmentation failed ({type(exc).__name__}): {str(exc)[:300]}",
                "frames": []}
