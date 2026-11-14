"""Local-checkpoint YOLO inference; boxes and pose coordinates are normalized to [0, 1]."""
from __future__ import annotations

import hashlib
import math
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config

_CACHE_SIZE = 128
_MAX_DETECTIONS = 100


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _load_model(path: str, task: str) -> Any:
    # Also disable auto-install when another worker component imported YOLO first.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO
    from ultralytics.utils import checks

    checks.AUTOINSTALL = False
    return YOLO(path, task=task)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass
class _ModelState:
    identity: tuple
    model: Any = None
    error: str | None = None


class DetectionService:
    """Settings are read once from the environment; models load on their first request.

    status describes the last request: uninitialized, ready, ready_cpu_fallback,
    unavailable, or error. warning explains unavailable/error and CPU fallback states.
    The LRU stores successful results only, including empty detections, in memory.
    """

    def __init__(self) -> None:
        self.status = "uninitialized"
        self.warning: str | None = None
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}
        self._cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._paths = {
            "detect": os.environ.get("YOLO_MODEL_PATH", "").strip(),
            "pose": os.environ.get("YOLO_POSE_MODEL_PATH", "").strip(),
        }
        self._requested_device = os.environ.get("YOLO_DEVICE", "auto").strip().lower()
        self._device = "cpu"
        self._device_initialized = False
        self._device_warning: str | None = None
        self._cpu_fallback = False
        self._configuration_error: str | None = None
        try:
            self._image_size = int(os.environ.get("YOLO_IMAGE_SIZE", "640"))
            self._confidence = float(os.environ.get("YOLO_CONFIDENCE", "0.4"))
            if self._image_size <= 0:
                raise ValueError("YOLO_IMAGE_SIZE must be a positive integer")
            if not math.isfinite(self._confidence) or not 0 <= self._confidence <= 1:
                raise ValueError("YOLO_CONFIDENCE must be finite and between 0 and 1")
            if self._requested_device not in {"auto", "cpu", "cuda"}:
                raise ValueError("YOLO_DEVICE must be auto, cpu, or cuda")
        except ValueError as exc:
            self._configuration_error = f"Invalid YOLO configuration: {exc}"

    def _set_status(self, status: str, reason: str | None = None) -> None:
        self.status = status
        self.warning = "; ".join(filter(None, (self._device_warning, reason))) or None

    def _ready(self) -> None:
        self._set_status("ready_cpu_fallback" if self._cpu_fallback else "ready")

    def _initialize_device(self) -> None:
        if self._device_initialized:
            return
        self._device_initialized = True
        try:
            available = _cuda_available()
        except Exception as exc:
            available = False
            self._device_warning = f"CUDA availability check failed ({exc}); using CPU"
            self._cpu_fallback = self._requested_device != "cpu"
        if self._requested_device != "cpu" and available:
            self._device = "cuda"
        elif self._requested_device == "cuda" and not self._device_warning:
            self._device_warning = "CUDA requested but unavailable; using CPU"
            self._cpu_fallback = True

    def _checkpoint(self, task: str) -> tuple:
        configured = self._paths[task]
        if task == "pose" and not configured:
            raise ValueError("YOLO_POSE_MODEL_PATH is not configured; pose is unavailable")
        candidates = (
            [Path(configured).expanduser()]
            if configured
            else [Path(config.MODELS_DIR) / "yolov8n.pt", Path.cwd() / "yolov8n.pt"]
        )
        for candidate in candidates:
            try:
                path = candidate.resolve(strict=True)
                stat = path.stat()
                if path.suffix != ".pt" or not path.is_file() or stat.st_size <= 0:
                    continue
                return (
                    str(path), stat.st_dev, stat.st_ino, stat.st_size,
                    stat.st_mtime_ns, stat.st_ctime_ns,
                )
            except (OSError, ValueError, RuntimeError):
                continue
        setting = "YOLO_POSE_MODEL_PATH" if task == "pose" else "YOLO_MODEL_PATH"
        if configured:
            raise ValueError(f"{setting} must name an existing, nonempty local .pt checkpoint")
        raise ValueError("No cached yolov8n.pt in MODELS_DIR or CWD; downloads are disabled")

    def _get_model(self, task: str, identity: tuple) -> Any:
        state = self._models.get(task)
        if state is None or state.identity != identity:
            # Release an old checkpoint before loading its replacement.
            self._models.pop(task, None)
            state = _ModelState(identity)
            self._models[task] = state
            try:
                state.model = _load_model(identity[0], task)
                if getattr(state.model, "task", task) != task:
                    raise ValueError(f"Checkpoint does not provide the requested {task} task")
            except Exception as exc:
                state.model = None
                state.error = f"Cannot load {task} checkpoint: {type(exc).__name__}: {exc}"
        if state.error:
            self._set_status("unavailable", state.error)
        return state.model

    def _cache_key(self, frame: np.ndarray, task: str, identity: tuple) -> tuple:
        pixels = hashlib.blake2b(memoryview(frame), digest_size=20).digest()
        return (
            pixels, frame.shape, frame.dtype.str, task, identity, self._device,
            self._image_size, self._confidence, _MAX_DETECTIONS,
        )

    def _infer(self, model: Any, frame: np.ndarray, pose: bool) -> list[dict]:
        results = model.predict(
            source=frame, device=self._device, imgsz=self._image_size,
            conf=self._confidence, batch=1, max_det=_MAX_DETECTIONS,
            stream=False, verbose=False, save=False, show=False,
        )
        return self._normalize(results, model, frame.shape[1], frame.shape[0], pose)

    @staticmethod
    def _cuda_failure(exc: Exception) -> bool:
        reason = str(exc).lower()
        return isinstance(exc, (RuntimeError, MemoryError)) and any(
            marker in reason for marker in
            ("cuda", "cudnn", "cublas", "out of memory", "device-side", "gpu")
        )

    def _normalize(self, results: Any, model: Any, width: int, height: int, pose: bool) -> list[dict]:
        if results is None:
            raise ValueError("Model returned no results object")
        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                raise ValueError("Model result has no detection boxes")
            boxes = _numpy(result.boxes.xyxy).reshape(-1, 4)
            classes = _numpy(result.boxes.cls).reshape(-1)
            scores = _numpy(result.boxes.conf).reshape(-1)
            if not len(boxes) == len(classes) == len(scores):
                raise ValueError("Model returned inconsistent detection lengths")
            names = getattr(result, "names", None)
            if names is None:
                names = getattr(model, "names", {})
            points = None
            if pose and getattr(result, "keypoints", None) is not None:
                data = _numpy(result.keypoints.data)
                if data.ndim == 3 and data.shape[0] == len(boxes) and data.shape[2] == 3:
                    points = data
            for index, (box, cid, score) in enumerate(zip(boxes, classes, scores)):
                if not np.isfinite(box).all() or not np.isfinite([cid, score]).all():
                    continue
                if cid < 0 or cid != int(cid) or score < self._confidence:
                    continue
                x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(cid)
                label = str(class_id)
                if isinstance(names, dict):
                    label = str(names.get(class_id, names.get(str(class_id), label)))
                elif isinstance(names, (list, tuple)) and class_id < len(names):
                    label = str(names[class_id])
                detection = {
                    "box": [float(x1 / width), float(y1 / height),
                            float((x2 - x1) / width), float((y2 - y1) / height)],
                    "label": label,
                    "class_id": class_id,
                    "confidence": float(np.clip(score, 0, 1)),
                }
                if points is not None and np.isfinite(points[index]).all():
                    detection["keypoints"] = np.clip(
                        points[index] / [width, height, 1], 0, 1,
                    ).tolist()
                detections.append(detection)
                if len(detections) >= _MAX_DETECTIONS:
                    return detections
        return detections

    def predict(self, frame: np.ndarray, *, pose: bool = False) -> list[dict]:
        """Predict one HWC BGR frame. Failure returns [] with status/warning, never a substitute box."""
        with self._lock:
            if self._configuration_error:
                self._set_status("unavailable", self._configuration_error)
                return []
            if (
                not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or not all(frame.shape[:2])
                or frame.dtype.kind not in "uif"
            ):
                self._set_status("error", "Expected a nonempty numeric HWC BGR numpy frame")
                return []
            task = "pose" if pose else "detect"
            try:
                identity = self._checkpoint(task)
            except (OSError, ValueError) as exc:
                self._set_status("unavailable", str(exc))
                return []
            self._initialize_device()
            model = self._get_model(task, identity)
            if model is None:
                return []
            frame = np.ascontiguousarray(frame)
            key = self._cache_key(frame, task, identity)
            if key in self._cache:
                self._cache.move_to_end(key)
                self._ready()
                return deepcopy(self._cache[key])
            try:
                try:
                    detections = self._infer(model, frame, pose)
                except Exception as exc:
                    if self._device != "cuda" or not self._cuda_failure(exc):
                        raise
                    self._device = "cpu"
                    self._cpu_fallback = True
                    self._device_warning = f"CUDA inference failed ({type(exc).__name__}: {exc}); using CPU"
                    self._cache.clear()
                    for state in self._models.values():
                        if state.model is not None:
                            # Ultralytics .to() resets the device-bound predictor as well.
                            state.model.to("cpu")
                    detections = self._infer(model, frame, pose)
                    key = self._cache_key(frame, task, identity)
            except Exception as exc:
                self._set_status("error", f"Prediction failed: {type(exc).__name__}: {exc}")
                return []
            self._cache[key] = deepcopy(detections)
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
            self._ready()
            return detections


_SERVICE: DetectionService | None = None
_SERVICE_LOCK = threading.Lock()


def get_detection_service() -> DetectionService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = DetectionService()
        return _SERVICE
