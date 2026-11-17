"""
OpenShorts-Powered Object Detection, Active-Speaker Tracking, Reframe Engine v2 & Editing Director.
ClipForge Studio Native Reframe Engine by Minuja Ranthika Amarasekara (reframe_v2,
active_speaker.py, punch_in.py, editing_director.py, auto_edit.py, export_quality.py).

Key upgrades over basic center/face cropping:
1. Multi-Stage Detection Cascade on <=640px downscaled frames:
   - Stage 1: MediaPipe BlazeFace (model_selection=1), or OpenCV YuNet when MediaPipe is not
     installed, + IoU face deduplication (MAX_FACE_OVERLAP=0.30).
   - Stage 2: Portrait/Upper-Body Fallback via YOLOv8 ('person' class) when hats, helmets, or profiles hide faces.
   - Stage 3: Context-Preserving Object & Multi-Person Detection ('person', 'bicycle', 'sports ball',
     'skateboard', 'surfboard', 'tennis racket', 'laptop', 'tv') — prevents slicing off a second primary
     person or action subject by automatically upgrading the shot to GENERAL (BLUR or EDITORIAL).
2. SpeakerTracker with 30-frame switch cooldown, 3.0x sticky hysteresis, and 2D velocity identity matching
   (associate_faces) — a 1-frame blink or head turn never whips the camera to another face.
3. SmoothedCameraman ("Heavy Tripod") with 25% safe-zone deadband, jump_confirm_frames=3 outlier rejection
   (eliminates 69% of in-scene reversals), and instant begin_scene() reset on scene cuts.
4. split_layout + active_speaker: Verifies true two-shots (>=50% coexistence, >=20% horizontal separation,
   median left/right centres, independent_split IoU <= 0.15) and gates stacking on mouth-movement + 8kHz audio RMS.
5. Native FFmpeg sendcmd rendering (crop@c) + compact_keyframes + punch_in smoothstep beat zooms + EBU R128 mastering.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from dataclasses import dataclass
from .speaker_priority import (
    MouthMemory,
    PersonRegistry,
    TalkWatcher,
    apply_speaker_priority,
    attribute_talkers,
    coverage_of,
    head_box,
    main_lock_of,
    visible_by_window,
)
from .speaker_priority import WINDOW_SECONDS as SPEAK_WINDOW
from .layout_decision import foreground_humans
from .tuning import EditTuning, signature_of, tune

ASPECT_RATIO = 9.0 / 16.0
DELIVERY_MIN_WIDTH = 1080
DETECT_MAX_WIDTH = 640
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)
SCENE_CUT_RESET = os.environ.get("SCENE_CUT_RESET", "1") != "0"

# Split layout constants (from split_layout.py)
MIN_COEXISTENCE = 0.50
MIN_SEPARATION = 0.20
MIN_FACE_WIDTH = 0.045
SPLIT_TIGHTNESS = float(os.environ.get("SPLIT_TIGHTNESS", "0.8"))
MAX_FACE_OVERLAP = 0.30

# Camera pursuit, all overridable from the environment so a live run can be retuned without an edit.
# The per-sample creep this replaces moved the crop at crop_width * 0.005 * (30 / fps) per sample —
# about 3.8 px/sample on a 1920x1080 source, so on the ghost-hunter footage the crop sat more than a
# quarter crop-width off the subject for 15% of samples, and for up to 2.5 s at a stretch. Measured
# behind the new law: 2% of samples, 0.4 s at a stretch (local_worker/benchmark_performance.py).
# These are per-second rates, so the motion no longer depends on the sample stride.
CAM_DEADBAND_X = float(os.environ.get("CAM_DEADBAND_X", "0.08"))  # fraction of crop width tolerated
CAM_DEADBAND_Y = float(os.environ.get("CAM_DEADBAND_Y", "0.05"))
CAM_TAU_X = float(os.environ.get("CAM_TAU_X", "0.38"))  # seconds to close ~63% of the remaining error
CAM_TAU_Y = float(os.environ.get("CAM_TAU_Y", "0.55"))
CAM_MAX_SPEED_X = float(os.environ.get("CAM_MAX_SPEED_X", "0.55"))  # crop widths per second at unit error
CAM_MAX_SPEED_Y = float(os.environ.get("CAM_MAX_SPEED_Y", "0.45"))
# How long a shot or layout stays on screen before the edit may change it again comes from
# tuning.py, measured off this video's own cut rhythm — a single global number is what made fast-cut
# footage look like it was stuttering and long takes look like they were being punched.
# Bounds of the self-calibrated scene-cut threshold, in Bhattacharyya distance between the 16x16
# HSV histograms of two consecutive samples. See adaptive_cut_threshold.
CUT_THR_MIN = float(os.environ.get("CUT_THR_MIN", "0.16"))
CUT_THR_MAX = float(os.environ.get("CUT_THR_MAX", "0.55"))
CUT_THR_SIGMAS = float(os.environ.get("CUT_THR_SIGMAS", "3.0"))

# Active speaker: the talker-attribution window is owned by speaker_priority, and this is the same
# number the audio envelope is cut into, so the two clocks cannot drift apart.
WINDOW_SECONDS = SPEAK_WINDOW

# Punch-in constants (from punch_in.py)
MAX_ZOOM = float(os.environ.get("PUNCH_IN_ZOOM", "1.12"))
RISE_SECONDS = 0.25
HOLD_SECONDS = 1.30
FALL_SECONDS = 0.55
BEAT_WINDOW = 0.2
MIN_GAP_SECONDS = 12.0
BEAT_PROMINENCE = 0.55

_DETECT_LOCK = threading.Lock()
_mp_face_detector = None
_haar_cascade = None
_yunet_detector = None
# Which detectors actually loaded, so a silent weight miss is visible in the analyze response
# instead of quietly degrading framing to the gradient stub.
DETECTION_STATUS: dict[str, str] = {"face": "unloaded", "person": "unloaded"}


# The weights the face and person detectors need. Neither ships in a pip package, and without them
# the cascade below silently degrades to no detections at all, so the first analyze on a fresh
# machine caches them next to the Whisper models, the same way the transcription pass does.
MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
}
# A truncated download is worse than no download: it fails deep inside the model loader instead of
# here, and the error says nothing about the weight.
MODEL_MIN_BYTES = {"face_detection_yunet_2023mar.onnx": 200_000, "yolov8n.pt": 4_000_000}


def model_file(filename: str, env_var: str) -> str | None:
    """Locate a detection weight: explicit env path, then the worker's model cache, then the CWD."""
    from .config import MODELS_DIR

    for candidate in (os.environ.get(env_var), os.path.join(MODELS_DIR, filename), filename):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def fetch_model(filename: str, env_var: str) -> str | None:
    """Like `model_file`, but caches the weight on first use so a fresh install can detect at all.

    Returns None when the weight is absent and cannot be fetched, which the callers report through
    DETECTION_STATUS rather than treating as an empty frame.
    """
    existing = model_file(filename, env_var)
    if existing:
        return existing
    url = MODEL_URLS.get(filename)
    if not url:
        return None
    from .config import MODELS_DIR

    target = os.path.join(MODELS_DIR, filename)
    partial = f"{target}.partial"
    try:
        import urllib.request

        os.makedirs(MODELS_DIR, exist_ok=True, mode=0o700)
        request = urllib.request.Request(url, headers={"User-Agent": "clipforge-local-worker"})
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        if os.path.getsize(partial) < MODEL_MIN_BYTES.get(filename, 1024):
            return None
        os.replace(partial, target)
        return target
    except Exception:
        try:
            os.remove(partial)
        except OSError:
            pass
        return None


def finite(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def validate_source(width: float, height: float, aspect: float = ASPECT_RATIO) -> None:
    if (
        not all(math.isfinite(float(v)) for v in (width, height, aspect))
        or width < 2
        or height < 2
        or aspect <= 0
    ):
        raise ValueError("source dimensions must be at least 2 and aspect must be positive")


def crop_size(width: int, height: int, aspect: float = ASPECT_RATIO, tightness: float = 1.0) -> tuple[int, int]:
    """Largest even crop that fits inside (width, height), optionally tightened around a subject."""
    validate_source(width, height, aspect)
    h = min(height, width / aspect) * max(0.3, min(finite(tightness, 1.0), 1.0))
    w = min(width, h * aspect)
    return max(2, int(w) // 2 * 2), max(2, int(h) // 2 * 2)


def crop_position(
    width: int,
    height: int,
    crop_w: int,
    crop_h: int,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[int, int]:
    """Clamp finite source-pixel centres to even in-bounds offsets."""
    validate_source(width, height)
    cx_val = finite(cx, width / 2.0)
    cy_val = finite(cy, height / 2.0)
    x = max(0, min(int(round(cx_val - crop_w / 2.0)), width - crop_w))
    y = max(0, min(int(round(cy_val - crop_h / 2.0)), height - crop_h))
    return (x // 2) * 2, (y // 2) * 2


def subject_position(
    width: int,
    height: int,
    crop_w: int,
    crop_h: int,
    centre: tuple[float, ...] | list[float],
) -> tuple[int, int]:
    """Place a face in the upper third (0.34 anchor), retaining the whole head when possible."""
    cx = finite(centre[0], width / 2.0)
    cy = finite(centre[1], height / 2.0)
    anchor = 0.34 if len(centre) >= 4 else 0.42
    x, y = crop_position(width, height, crop_w, crop_h, cx, cy + crop_h * (0.5 - anchor))
    if len(centre) >= 4:
        face_h = max(0.0, finite(centre[3], 0.0))
        if face_h <= crop_h:
            low = max(0.0, cy + face_h / 2.0 - crop_h)
            high = min(float(height - crop_h), cy - face_h * 0.65)
            if low <= high:
                y = (int(max(low, min(y, high))) // 2) * 2
    return x, y


def needs_context(candidates: list[dict[str, Any]], width: int, height: int, aspect: float = ASPECT_RATIO) -> bool:
    """True when a tight 9:16 crop would discard another primary face or slice a wide group."""
    if not candidates:
        return False
    boxes = [c["box"] for c in candidates]
    largest = max(w * h for _, _, w, h in boxes)
    primary = [(x, y, w, h) for x, y, w, h in boxes if w >= width * 0.045 and w * h >= largest * 0.35]
    if not primary:
        return False
    cw, ch = crop_size(width, height, aspect)
    span_x = max(x + w for x, y, w, h in primary) - min(x for x, y, w, h in primary)
    span_y = max(y + h for x, y, w, h in primary) - min(y for x, y, w, h in primary)
    return span_x > cw * 0.95 or span_y > ch * 0.95


def _iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedupe_faces(candidates: list[dict[str, Any]], threshold: float = MAX_FACE_OVERLAP) -> list[dict[str, Any]]:
    """Drop duplicate detections of the same face, keeping the largest."""
    kept: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c["box"][2] * c["box"][3], reverse=True):
        if all(_iou(cand["box"], k["box"]) <= threshold for k in kept):
            kept.append(cand)
    return kept


def associate_faces(
    candidates: list[dict[str, Any]],
    known: list[dict[str, Any]],
    frame: int,
    width: int,
    next_id: int,
    ttl: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """One-to-one 2D velocity-predicted local identity matching (survives short occlusions/crossings)."""
    faces: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            x, y, w, h = map(float, candidate["box"])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                continue
            faces.append({"box": [x, y, w, h], "score": max(0.0, finite(candidate.get("score"), w * h))})
        except (KeyError, TypeError, ValueError):
            continue

    known = [k for k in known if 0 <= frame - k["last_frame"] <= ttl]
    pairs = []
    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        cx, cy = x + w / 2.0, y + h / 2.0
        for old in known:
            dt = max(1, frame - old["last_frame"])
            px = old["center"] + old.get("vx", 0.0) * min(dt, ttl * 0.4)
            py = old.get("cy", cy) + old.get("vy", 0.0) * min(dt, ttl * 0.4)
            distance = math.hypot(cx - px, cy - py)
            ratio = max(w * h, old.get("area", w * h)) / max(1.0, min(w * h, old.get("area", w * h)))
            if distance < width * 0.15 and ratio <= 3.0:
                pairs.append((distance / width + abs(math.log(ratio)) * 0.025, old["id"], i, old))

    assigned: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for _, pid, i, old in sorted(pairs, key=lambda p: p[:3]):
        if i not in assigned and pid not in used:
            assigned[i] = old
            used.add(pid)

    current: list[dict[str, Any]] = []
    updated = [k for k in known if k["id"] not in used]
    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        cx, cy = x + w / 2.0, y + h / 2.0
        old = assigned.get(i)
        pid = old["id"] if old else next_id
        if old is None:
            next_id += 1
        dt = max(1, frame - old["last_frame"]) if old else 1
        vx = (cx - old["center"]) / dt if old else 0.0
        vy = (cy - old.get("cy", cy)) / dt if old else 0.0
        updated.append(
            {
                "id": pid,
                "center": cx,
                "cy": cy,
                "area": w * h,
                "seen": (int(old.get("seen", 1)) + 1) if old else 1,
                "last_frame": frame,
                "vx": max(-width * 0.15, min(width * 0.15, vx)),
                "vy": max(-width * 0.15, min(width * 0.15, vy)),
            }
        )
        current.append({**face, "id": pid})
    return current, updated, next_id


class SpeakerTracker:
    """
    OpenShorts SpeakerTracker: tracks speakers over time with 3.0x sticky hysteresis
    and a 30-frame switch cooldown so temporary blinks/occlusions never whip the camera.
    """

    def __init__(self, stabilization_frames: int = 15, cooldown_frames: int = 30, fps: float = 30.0) -> None:
        self.max_missing_frames = max(1, round(fps))
        self.active_speaker_id: int | None = None
        self.speaker_scores: dict[int, float] = {}
        self.locked_counter = 0
        self.stabilization_threshold = stabilization_frames
        self.switch_cooldown = cooldown_frames
        self.last_switch_frame = -1000
        self.next_id = 0
        self.known_faces: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Forget previous shot's speaker at a scene cut."""
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.locked_counter = 0
        self.last_switch_frame = -1000
        self.known_faces = []

    def get_target(self, face_candidates: list[dict[str, Any]], frame_number: int, width: int) -> list[float] | None:
        current_candidates, self.known_faces, self.next_id = associate_faces(
            face_candidates,
            self.known_faces,
            frame_number,
            width,
            self.next_id,
            ttl=self.max_missing_frames,
        )
        alive = {k["id"] for k in self.known_faces}
        self.speaker_scores = {pid: score for pid, score in self.speaker_scores.items() if pid in alive}

        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand["id"]
            cx = cand["box"][0] + cand["box"][2] / 2.0
            centrality = 1.0 - min(1.0, abs(cx - width / 2.0) / max(1.0, width / 2.0))
            raw_score = cand["score"] / max(1.0, width * width * 0.05) * (0.55 + 0.45 * centrality)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0.0) + raw_score

        if not current_candidates:
            return None

        seen = {k["id"]: k.get("seen", 1) for k in self.known_faces}
        # Only guard the camera while the current subject is still on screen; once it has left,
        # the longest-lived newcomer takes over even if it has not matured yet.
        active_alive = any(c["id"] == self.active_speaker_id for c in current_candidates)
        best_candidate = None
        max_score = -1.0
        for cand in current_candidates:
            pid = cand["id"]
            if active_alive and pid != self.active_speaker_id:
                if seen.get(pid, 1) < self.stabilization_threshold:
                    continue
            total_score = self.speaker_scores.get(pid, 0.0)
            if pid == self.active_speaker_id:
                total_score *= 3.0  # Sticky hysteresis bonus
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate:
            target_id = best_candidate["id"]
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate["box"]

            # Hold current frame if within cooldown (even if active speaker blinked out on this frame)
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c["id"] == self.active_speaker_id), None)
                return old_cand["box"] if old_cand else None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate["box"]
        return None


class SmoothedCameraman:
    """
    OpenShorts SmoothedCameraman ("Heavy Tripod"):
    - 25% safe-zone deadband (camera stays stationary inside deadband).
    - jump_confirm_frames=3 outlier rejection (ignores single-frame detector jumps).
    - begin_scene() instant snap at scene cuts so there is zero headless-torso pan lag after a cut.
    """

    def __init__(
        self,
        output_width: int,
        output_height: int,
        video_width: int,
        video_height: int,
        aspect_ratio: float = ASPECT_RATIO,
        fps: float = 30.0,
    ) -> None:
        self.motion_time_scale = 30.0 / max(1.0, float(fps))
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio

        self.current_center_x = video_width / 2.0
        self.target_center_x = video_width / 2.0
        self.current_center_y = video_height / 2.0
        self.target_center_y = video_height / 2.0

        # The crop box and the exported track answer different questions. The crop centre can only
        # move when the crop is shorter than the source, which never happens on a landscape video:
        # a 9:16 crop spans the full height, so the centre is pinned at 0.5. Consumers (render.ts's
        # `sy = fy * vh - cropH * 0.36`, split panels, the blend default at line 1388) all read `y`
        # as where the head *is*, so aim that is actually followed has to be carried separately.
        self.current_anchor_y = 0.38 * video_height
        self.target_anchor_y = self.current_anchor_y
        self.subject_anchor_y = self.current_anchor_y / float(video_height)

        self.crop_width, self.crop_height = crop_size(video_width, video_height, aspect_ratio)
        self.deadband_x = CAM_DEADBAND_X
        self.deadband_y = CAM_DEADBAND_Y
        self.tau_x = CAM_TAU_X
        self.tau_y = CAM_TAU_Y
        self.max_speed_x = CAM_MAX_SPEED_X
        self.max_speed_y = CAM_MAX_SPEED_Y
        self.safe_zone_radius = self.crop_width * CAM_DEADBAND_X
        self.tracking_safe_zone_radius = self.safe_zone_radius
        self.sample_dt = 1.0 / max(1.0, float(fps))

        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target: float | None = None
        self._pending_count = 0
        self._snap_pending = False

    def apply_tuning(self, t: "EditTuning") -> None:
        """Take the pursuit law from this video's measured editing rhythm instead of the defaults."""
        self.deadband_x, self.deadband_y = t.deadband_x, t.deadband_y
        self.tau_x, self.tau_y = t.tau_x, t.tau_y
        self.max_speed_x, self.max_speed_y = t.max_speed_x, t.max_speed_y
        self.safe_zone_radius = self.crop_width * self.deadband_x
        self.tracking_safe_zone_radius = self.safe_zone_radius

    def set_sample_interval(self, dt: float) -> None:
        """How much source time elapses between update_target calls, so pursuit is per-second."""
        self.sample_dt = max(1e-3, float(dt))

    def _pursuit(self, error: float, span: float, tau: float, max_speed: float) -> float:
        """Exponential approach capped by a speed limit that grows with the error.

        Contractive, so it cannot overshoot, and independent of the sampling rate. Large errors
        (a subject that jumped across the shot) resolve in a few samples instead of seconds.
        """
        dt = self.sample_dt
        eased = abs(error) * (1.0 - math.exp(-dt / tau))
        limit = span * max_speed * dt * (1.0 + 3.0 * min(1.5, abs(error) / max(1.0, span)))
        return math.copysign(min(abs(error), eased, limit), error)

    def begin_scene(self) -> None:
        """Reset damping at a scene cut and snap to the first detected subject in the new shot."""
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True
        self.tracking_safe_zone_radius = self.safe_zone_radius
        self.current_center_x = self.target_center_x = self.video_width / 2.0
        self.current_center_y = self.target_center_y = self.video_height / 2.0

    def update_target(self, face_box: list[float] | tuple[float, ...] | None) -> None:
        if not face_box:
            return
        x, y, w, h = face_box
        if not all(math.isfinite(float(v)) for v in face_box) or w <= 0 or h <= 0:
            return
        new_center = max(0.0, min(float(x + w / 2.0), float(self.video_width)))
        _, top = subject_position(
            self.video_width,
            self.video_height,
            self.crop_width,
            self.crop_height,
            (new_center, y + h / 2.0, w, h),
        )
        new_y = top + self.crop_height / 2.0
        new_anchor = max(0.12, min(0.85, (y + h / 2.0) / float(self.video_height))) * float(self.video_height)
        self.tracking_safe_zone_radius = min(
            self.safe_zone_radius,
            max(0.0, self.crop_width / 2.0 - w / 2.0 - self.crop_width * 0.08),
        )

        if self._snap_pending:
            self._snap_pending = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center
            self.current_center_x = new_center
            self.current_center_y = self.target_center_y = new_y
            self.current_anchor_y = self.target_anchor_y = new_anchor
            return

        if abs(new_center - self.target_center_x) > self.safe_zone_radius:
            if self._pending_target is not None and abs(new_center - self._pending_target) <= self.safe_zone_radius:
                self._pending_count += 1
            else:
                self._pending_target = new_center
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = new_center
        self.target_center_y = new_y
        self.target_anchor_y = new_anchor

    def get_crop_box(self, force_snap: bool = False) -> tuple[int, int, int, int]:
        if force_snap:
            self.current_center_x = self.target_center_x
            self.current_center_y = self.target_center_y
            self.current_anchor_y = self.target_anchor_y
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.tracking_safe_zone_radius:
                direction = 1.0 if diff > 0 else -1.0
                self.current_center_x += self._pursuit(diff, self.crop_width, self.tau_x, self.max_speed_x)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1.0 and new_diff < 0) or (direction == -1.0 and new_diff > 0):
                    self.current_center_x = self.target_center_x

        half_crop = self.crop_width / 2.0
        self.current_center_x = max(half_crop, min(self.video_width - half_crop, self.current_center_x))

        diff_y = self.target_center_y - self.current_center_y
        if abs(diff_y) > self.crop_height * self.deadband_y:
            self.current_center_y += self._pursuit(diff_y, self.crop_height, self.tau_y, self.max_speed_y)

        # Same damping law as the crop's vertical aim, but the travel is measured in source heights
        # because the anchor is a position on the subject, not a limit the crop is allowed to move in.
        diff_anchor = self.target_anchor_y - self.current_anchor_y
        if abs(diff_anchor) > self.video_height * self.deadband_y:
            self.current_anchor_y += self._pursuit(
                diff_anchor, float(self.video_height), self.tau_y, self.max_speed_y
            )
        self.subject_anchor_y = max(0.0, min(1.0, self.current_anchor_y / float(self.video_height)))

        x1, y1 = crop_position(
            self.video_width,
            self.video_height,
            self.crop_width,
            self.crop_height,
            self.current_center_x,
            self.current_center_y,
        )
        return x1, y1, x1 + self.crop_width, y1 + self.crop_height


# --- Split Layout & Active Speaker Helpers (from split_layout.py & active_speaker.py) ---


def filter_edge_bystanders(candidates: list[dict[str, Any]], frame_w: int) -> list[dict[str, Any]]:
    """
    Rule 1 (Edge-Bystander Rejection):
    In a 1-person close-up inside a lineup (e.g. WA0029 0:01-0:02, WA0030 0:09-0:13), a neighbor's
    shoulder or half-face at the outer 12% margin (cx < 0.12 or cx > 0.88) must not be counted as
    a second primary subject when a central speaker dominates the frame.
    """
    if len(candidates) <= 1 or frame_w <= 0:
        return candidates
    largest_area = max((c["box"][2] * c["box"][3] for c in candidates), default=0.0)
    has_central_primary = any(
        0.12 * frame_w <= (c["box"][0] + c["box"][2] / 2.0) <= 0.88 * frame_w
        and (c["box"][2] * c["box"][3]) >= largest_area * 0.65
        for c in candidates
    )
    if not has_central_primary:
        return candidates
    kept: list[dict[str, Any]] = []
    for c in candidates:
        x, y, w, h = c["box"]
        cx = x + w / 2.0
        area = w * h
        in_edge_margin = cx < 0.12 * frame_w or cx > 0.88 * frame_w
        clipped_at_border = x <= 0.015 * frame_w or (x + w) >= 0.985 * frame_w
        if (in_edge_margin or clipped_at_border) and area < largest_area * 0.55:
            continue
        kept.append(c)
    return kept if kept else candidates


def _two_largest(candidates: list[dict[str, Any]], frame_w: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    deduped = filter_edge_bystanders(dedupe_faces(candidates), frame_w)
    big = [c for c in deduped if c["box"][2] >= MIN_FACE_WIDTH * frame_w]
    if len(big) < 2:
        return None
    largest = max(c["box"][2] * c["box"][3] for c in big)
    primary = [c for c in big if c["box"][2] * c["box"][3] >= largest * 0.35]
    if len(primary) < 2 or len(primary) > 2:
        return None
    primary.sort(key=lambda c: c["score"], reverse=True)
    left_c, right_c = sorted((primary[0], primary[1]), key=lambda c: c["box"][0])
    # Rule 2 (Middle-Subject Veto): reject 2-Up Split if a 3rd readable face sits between left and right
    left_cx = left_c["box"][0] + left_c["box"][2] / 2.0
    right_cx = right_c["box"][0] + right_c["box"][2] / 2.0
    for other in big:
        if other is left_c or other is right_c:
            continue
        ocx = other["box"][0] + other["box"][2] / 2.0
        if left_cx + 0.08 * frame_w < ocx < right_cx - 0.08 * frame_w:
            return None
    return (left_c, right_c)


def _centre(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, float(w), float(h)


def as_box(centre: tuple[float, ...] | list[float]) -> list[float]:
    cx, cy, w, h = centre
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def analyze_split_scene(
    frames: list[list[dict[str, Any]]],
    frame_w: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """
    OpenShorts split_layout.analyze_scene:
    Requires >= 50% coexistence of two primary faces separated by >= 20% width,
    computes median (cx, cy, w, h) for left and right speakers, and rejects walking/crossing shots.
    """
    if not frames:
        return None
    lefts, rights = [], []
    for candidates in frames:
        pair = _two_largest(candidates, frame_w)
        if not pair:
            continue
        left_c, right_c = _centre(pair[0]["box"]), _centre(pair[1]["box"])
        if (right_c[0] - left_c[0]) < MIN_SEPARATION * frame_w:
            continue
        lefts.append(left_c)
        rights.append(right_c)

    if len(lefts) < MIN_COEXISTENCE * len(frames):
        return None

    left = tuple(float(np.median([c[i] for c in lefts])) for i in range(4))
    right = tuple(float(np.median([c[i] for c in rights])) for i in range(4))
    for centres, median in ((lefts, left), (rights, right)):
        misses = sum(
            abs(c[0] - median[0]) > frame_w * 0.12 or abs(c[1] - median[1]) > frame_w * 0.10 for c in centres
        )
        if misses > max(1, len(centres) * 0.15):
            return None
    return left, right  # type: ignore[return-value]


def analyze_three_up_scene(
    frames: list[list[dict[str, Any]]],
    frame_w: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """
    Rule 3 (3-Up Split Grid for 3-Person Lineup / Desk Shots):
    When 3 distinct readable subjects coexist in >= 50% of a shot's samples with clean horizontal
    separation (each >= 18% frame width apart), return (left, mid, right) median centres.
    """
    if len(frames) < 3 or frame_w <= 0:
        return None
    lefts, mids, rights = [], [], []
    for candidates in frames:
        deduped = filter_edge_bystanders(dedupe_faces(candidates), frame_w)
        big = [c for c in deduped if c["box"][2] >= MIN_FACE_WIDTH * frame_w]
        if len(big) < 3:
            continue
        largest = max(c["box"][2] * c["box"][3] for c in big)
        primary = [c for c in big if c["box"][2] * c["box"][3] >= largest * 0.32]
        if len(primary) != 3:
            continue
        ordered = sorted(primary, key=lambda c: c["box"][0] + c["box"][2] / 2.0)
        lc, mc, rc = _centre(ordered[0]["box"]), _centre(ordered[1]["box"]), _centre(ordered[2]["box"])
        if (mc[0] - lc[0]) < 0.18 * frame_w or (rc[0] - mc[0]) < 0.18 * frame_w:
            continue
        lefts.append(lc)
        mids.append(mc)
        rights.append(rc)

    if len(lefts) < MIN_COEXISTENCE * len(frames):
        return None

    left = tuple(float(np.median([c[i] for c in lefts])) for i in range(4))
    mid = tuple(float(np.median([c[i] for c in mids])) for i in range(4))
    right = tuple(float(np.median([c[i] for c in rights])) for i in range(4))
    return left, mid, right  # type: ignore[return-value]


def split_geometry(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    centre: tuple[float, ...] | list[float],
) -> tuple[int, int, int, int, int]:
    half_h = (out_h // 2) - ((out_h // 2) % 2)
    aspect = out_w / float(half_h)
    tightness = 1.0 if min(orig_w, orig_h) < 480 else max(0.3, min(SPLIT_TIGHTNESS, 1.0))
    tight_h = orig_h * tightness
    cw, ch = crop_size(orig_w, int(min(orig_h, max(2.0, tight_h))), aspect)
    x, y = subject_position(orig_w, orig_h, cw, ch, centre)
    return cw, ch, x - (x % 2), y - (y % 2), half_h


def independent_split(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    centres: tuple[tuple[float, ...], tuple[float, ...]],
) -> bool:
    """Reject panels that repeat the same source region or overlap another primary face."""
    if len(centres) != 2 or any(len(c) < 4 for c in centres):
        return False
    regions = []
    for centre in centres:
        w, h, x, y, _ = split_geometry(orig_w, orig_h, out_w, out_h, centre)
        regions.append([x, y, w, h])
    if _iou(*regions) > 0.15:
        return False

    def overlap_fraction(region: list[int], box: list[float]) -> float:
        x, y, w, h = region
        a, b, c, d = box
        return max(0.0, min(x + w, a + c) - max(x, a)) * max(0.0, min(y + h, b + d) - max(y, b)) / max(1.0, c * d)

    for i, region in enumerate(regions):
        if overlap_fraction(region, as_box(centres[i])) < 0.9:
            return False
        if overlap_fraction(region, as_box(centres[1 - i])) > 0.1:
            return False
    return True


def split_filtergraph(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    left_centre: tuple[float, ...],
    right_centre: tuple[float, ...],
) -> str:
    top_w, top_h, top_x, top_y, half_h = split_geometry(orig_w, orig_h, out_w, out_h, left_centre)
    bot_w, bot_h, bot_x, bot_y, _ = split_geometry(orig_w, orig_h, out_w, out_h, right_centre)
    return (
        f"[0:v]split=2[ta][ba];"
        f"[ta]crop=w={top_w}:h={top_h}:x={top_x}:y={top_y},scale={out_w}:{half_h}[top];"
        f"[ba]crop=w={bot_w}:h={bot_h}:x={bot_x}:y={bot_y},scale={out_w}:{half_h}[bot];"
        f"[top][bot]vstack=inputs=2,scale={out_w}:{out_h},setsar=1[v]"
    )


def editorial_geometry(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    focus_x: float | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Full-height detail view on top + full-width context view on bottom (zero blurry bars!)."""
    validate_source(orig_w, orig_h, out_w / float(out_h))
    fg_h = int(round(out_w * orig_h / float(orig_w)))
    fg_h += fg_h % 2
    context_h = max(2, min(fg_h, (int(out_h * 0.45) // 2) * 2))
    detail_h = out_h - context_h
    cw, ch = crop_size(orig_w, orig_h, out_w / float(detail_h))
    x, y = crop_position(orig_w, orig_h, cw, ch, focus_x)
    return cw, ch, x, y, detail_h, context_h


def contain_background(out_w: int, out_h: int, source: str = "0:v", output: str = "v", prefix: str = "bed") -> str:
    w = max(2, round(out_w / 16) * 2)
    h = max(2, round(out_h / 16) * 2)
    sigma = max(1.0, 64.0 * w / 1080.0)
    soften = f"scale={w}:{h},gblur=sigma={sigma:.3f}:steps=2,scale={out_w}:{out_h},eq=brightness=-0.06:saturation=0.7"
    return (
        f"[{source}]split=2[{prefix}a][{prefix}b];"
        f"[{prefix}a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},setsar=1,{soften}[{prefix}bg];"
        f"[{prefix}b]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2,setsar=1[{prefix}fg];"
        f"[{prefix}bg][{prefix}fg]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1[{output}]"
    )


def editorial_filtergraph(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    focus_x: float | None = None,
) -> str:
    cw, ch, x, y, detail_h, context_h = editorial_geometry(orig_w, orig_h, out_w, out_h, focus_x)
    context = (
        f"[context]scale={out_w}:{context_h},setsar=1[bottom]"
        if abs(context_h - out_w * orig_h / orig_w) <= 2
        else contain_background(out_w, context_h, "context", "bottom", "contextbed")
    )
    return (
        f"[0:v]split=2[detail][context];"
        f"[detail]crop=w={cw}:h={ch}:x={x}:y={y},scale={out_w}:{detail_h}[top];"
        + context
        + ";[top][bottom]vstack=inputs=2,setsar=1[v]"
    )


# --- Audio envelope (the loudness the talker attribution and the punch-in beats are both read from) ---


def audio_envelope(video_path: str, start_s: float, duration_s: float, window_s: float = WINDOW_SECONDS) -> list[float]:
    try:
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{start_s:.4f}",
                "-t",
                f"{duration_s:.4f}",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "s16le",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    if not raw:
        return []
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    per_window = max(1, int(8000 * window_s))
    n = len(samples) // per_window
    if n < 1:
        return []
    windows = samples[: n * per_window].reshape(n, per_window)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    peak = rms.max()
    return (rms / peak).tolist() if peak > 0 else [0.0] * n


# --- Punch-In & Keyframe Compaction Helpers (from punch_in.py & framing_geometry.py) ---


def _ease_smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def emphasis_from_envelope(
    envelope: list[float],
    window: float = BEAT_WINDOW,
    min_gap: float = MIN_GAP_SECONDS,
    prominence: float = BEAT_PROMINENCE,
) -> list[float]:
    if not envelope:
        return []
    values = np.asarray(envelope, dtype=float)
    baseline = float(np.median(values))
    peak = float(values.max())
    if peak - baseline <= 1e-6:
        return []
    threshold = baseline + (peak - baseline) * prominence
    times: list[float] = []
    last = -1e9
    for i, v in enumerate(values):
        t = i * window
        if v >= threshold and t - last >= min_gap:
            times.append(round(t, 3))
            last = t
    return times


def zoom_curve(
    n_frames: int,
    fps: float,
    emphasis_times: list[float],
    max_zoom: float = MAX_ZOOM,
    start_offset: float = 0.0,
) -> list[float]:
    zooms = [1.0] * max(0, n_frames)
    if not zooms or max_zoom <= 1.0:
        return zooms
    span = RISE_SECONDS + HOLD_SECONDS + FALL_SECONDS
    for t in emphasis_times:
        local = t - start_offset
        if local <= -span or local >= n_frames / fps:
            continue
        for f in range(max(0, int(local * fps)), min(n_frames, int((local + span) * fps) + 1)):
            dt = f / fps - local
            if dt < 0:
                continue
            if dt < RISE_SECONDS:
                amount = _ease_smoothstep(dt / RISE_SECONDS)
            elif dt < RISE_SECONDS + HOLD_SECONDS:
                amount = 1.0
            elif dt < span:
                amount = 1.0 - _ease_smoothstep((dt - RISE_SECONDS - HOLD_SECONDS) / FALL_SECONDS)
            else:
                continue
            zooms[f] = max(zooms[f], 1.0 + (max_zoom - 1.0) * amount)
    return zooms


def crop_boxes(
    xs: list[int],
    zooms: list[float],
    crop_w: int,
    crop_h: int,
    orig_w: int,
    orig_h: int,
    ys: list[int] | None = None,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for i, x in enumerate(xs):
        z = zooms[i] if i < len(zooms) else 1.0
        z = 1.0 if min(orig_w, orig_h) < 480 else max(1.0, min(1.15, finite(z, 1.0)))
        w = int(crop_w / z)
        h = int(crop_h / z)
        w -= w % 2
        h -= h % 2
        w = max(2, min(w, orig_w))
        h = max(2, min(h, orig_h))
        cx = x + crop_w / 2.0
        base_y = ys[i] if (ys and i < len(ys)) else 0
        cy = base_y + crop_h / 2.0
        nx = max(0, min(int(round(cx - w / 2.0)), orig_w - w))
        ny = max(0, min(int(round(cy - h / 2.0)), orig_h - h))
        boxes.append((w, h, nx - (nx % 2), ny - (ny % 2)))
    return boxes


def compact_keyframes(
    boxes: list[tuple[int, int, int, int]],
    fps: float,
    tolerance: float = 2.0,
) -> list[dict[str, float]]:
    """Bounded-error linear source-pixel paths, reset independently per shot."""
    if not boxes:
        return []

    def key(index: int) -> dict[str, float]:
        w, h, x, y = boxes[index]
        return {"time": round(index / fps, 6), "x": float(x), "y": float(y), "width": float(w), "height": float(h)}

    if all(box == boxes[0] for box in boxes):
        return [key(0)]
    keep = {0, len(boxes) - 1}
    step = max(1, round(fps))
    for first in range(0, len(boxes) - 1, step):
        last = min(len(boxes) - 1, first + step)
        keep.update((first, last))
        pending = [(first, last)]
        while pending:
            a, b = pending.pop()
            error = 0.0
            worst = None
            for i in range(a + 1, b):
                p = (i - a) / float(b - a)
                distance = max(abs(boxes[i][axis] - (boxes[a][axis] * (1.0 - p) + boxes[b][axis] * p)) for axis in range(4))
                if distance > error:
                    error, worst = distance, i
            if worst is not None and error > tolerance:
                keep.add(worst)
                pending.extend(((a, worst), (worst, b)))
    return [key(i) for i in sorted(keep)]


def sample_keyframes(
    keys: list[dict[str, float]],
    fps: float,
    duration: float,
    width: int,
    height: int,
    interpolation: str = "linear",
) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    index = 0
    for frame in range(math.ceil(duration * fps - 1e-6)):
        t = frame / fps
        while index + 1 < len(keys) and keys[index + 1]["time"] <= t + 1e-6:
            index += 1
        a = keys[index]
        b = keys[min(index + 1, len(keys) - 1)]
        p = (
            max(0.0, min(1.0, (t - a["time"]) / (b["time"] - a["time"])))
            if b["time"] > a["time"] and interpolation == "linear"
            else 0.0
        )
        values = {name: a[name] * (1.0 - p) + b[name] * p for name in ("width", "height", "x", "y")}
        w = max(2, min((width // 2) * 2, int(round(values["width"] / 2.0)) * 2))
        h = max(2, min((height // 2) * 2, int(round(values["height"] / 2.0)) * 2))
        x, y = crop_position(width, height, w, h, values["x"] + w / 2.0, values["y"] + h / 2.0)
        result.append((w, h, x, y))
    return result


def sendcmd_lines(boxes: list[tuple[int, int, int, int]], fps: float, target: str = "crop@c") -> list[str]:
    """Deduplicated FFmpeg sendcmd lines for per-frame w/h/x/y."""
    lines: list[str] = []
    prev = None
    for i, box in enumerate(boxes):
        if box == prev:
            continue
        t = i / fps
        w, h, x, y = box
        pw, ph, px, py = prev if prev else (None, None, None, None)
        if w != pw:
            lines.append(f"{t:.4f} {target} w {w};")
        if h != ph:
            lines.append(f"{t:.4f} {target} h {h};")
        if x != px:
            lines.append(f"{t:.4f} {target} x {x};")
        if y != py:
            lines.append(f"{t:.4f} {target} y {y};")
        prev = box
    return lines


# --- Multi-Stage Object & Face Detectors (MediaPipe + YOLOv8 + OpenCV Fallback) ---


def _detection_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / float(DETECT_MAX_WIDTH)
    small = cv2.resize(frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)), interpolation=cv2.INTER_AREA)
    return small, scale


def detect_face_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
    """Faces via MediaPipe BlazeFace, else OpenCV YuNet, else Haar cascade — whichever is actually installed."""
    global _mp_face_detector, _haar_cascade, _yunet_detector
    import cv2

    height, width = frame.shape[:2]
    small, scale = _detection_frame(frame)

    with _DETECT_LOCK:
        if _mp_face_detector is None:
            try:
                import mediapipe as mp

                _mp_face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except Exception:
                _mp_face_detector = False

        if _mp_face_detector:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = _mp_face_detector.process(rgb)
            candidates = []
            for det in res.detections or []:
                bbox = det.location_data.relative_bounding_box
                x = int(max(0.0, bbox.xmin) * width)
                y = int(max(0.0, bbox.ymin) * height)
                w = int(min(1.0 - max(0.0, bbox.xmin), bbox.width) * width)
                h = int(min(1.0 - max(0.0, bbox.ymin), bbox.height) * height)
                if w > 0 and h > 0:
                    candidates.append({"box": [x, y, w, h], "score": float(w * h), "confidence": float(det.score[0])})
            DETECTION_STATUS["face"] = "mediapipe"
            return dedupe_faces(candidates)

        # YuNet ships inside opencv-python and holds up on profiles/hats where Haar returns nothing at all.
        if _yunet_detector is None:
            weight = fetch_model("face_detection_yunet_2023mar.onnx", "FACE_MODEL_PATH")
            try:
                _yunet_detector = cv2.FaceDetectorYN.create(weight, "", (640, 640), 0.6, 0.35, 5000) if weight else False
            except Exception:
                _yunet_detector = False

        if _yunet_detector:
            _yunet_detector.setInputSize((int(small.shape[1]), int(small.shape[0])))
            _, found = _yunet_detector.detect(small)
            candidates = []
            for row in found if found is not None else []:
                x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                if w > 0 and h > 0:
                    candidates.append(
                        {"box": [x * scale, y * scale, w * scale, h * scale], "score": float(w * h * scale * scale), "confidence": float(row[-1])}
                    )
            DETECTION_STATUS["face"] = "yunet"
            return dedupe_faces(candidates)

        # OpenCV Frontal + Left/Right Profile Haar Cascade fallback for 3/4 podcast angles
        if _haar_cascade is None:
            frontal_p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            profile_p = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
            _haar_cascade = {
                "frontal": cv2.CascadeClassifier(str(frontal_p)) if frontal_p.is_file() else None,
                "profile": cv2.CascadeClassifier(str(profile_p)) if profile_p.is_file() else None,
            }

        if isinstance(_haar_cascade, dict) and _haar_cascade.get("frontal") is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sw = int(small.shape[1])
            candidates = []
            frontal = _haar_cascade["frontal"]
            if frontal is not None and not frontal.empty():
                rects = frontal.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
                for rx, ry, rw, rh in rects if len(rects) else []:
                    x, y, w, h = int(rx * scale), int(ry * scale), int(rw * scale), int(rh * scale)
                    candidates.append({"box": [x, y, w, h], "score": float(w * h)})
            profile = _haar_cascade.get("profile")
            if profile is not None and not profile.empty():
                p_rects = profile.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
                for rx, ry, rw, rh in p_rects if len(p_rects) else []:
                    x, y, w, h = int(rx * scale), int(ry * scale), int(rw * scale), int(rh * scale)
                    candidates.append({"box": [x, y, w, h], "score": float(w * h * 0.92)})
                flipped = cv2.flip(gray, 1)
                pf_rects = profile.detectMultiScale(flipped, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
                for rx, ry, rw, rh in pf_rects if len(pf_rects) else []:
                    orig_rx = sw - int(rx) - int(rw)
                    x, y, w, h = int(orig_rx * scale), int(ry * scale), int(rw * scale), int(rh * scale)
                    candidates.append({"box": [x, y, w, h], "score": float(w * h * 0.92)})
            DETECTION_STATUS["face"] = "haar_frontal_profile"
            return dedupe_faces(candidates)

    DETECTION_STATUS["face"] = "none"
    return []


# What is worth aiming a camera at in narrative footage. Held, worn, handled, ridden or chased —
# the object a shot is *about* — so a phone held up to the lens, the backpack being argued over,
# the food in a challenge and the animal in the room all become anchors instead of blind spots.
# Furniture and fixtures are deliberately not in this set: a couch fills half a haunted living room
# and would hold the frame hostage while somebody is talking, so it only counts when it dominates.
OBJECT_CLASSES_PRIMARY = (
    "backpack,handbag,suitcase,umbrella,cell phone,remote,laptop,keyboard,mouse,tv,"
    "bottle,cup,wine glass,bowl,fork,knife,spoon,banana,apple,sandwich,orange,broccoli,carrot,"
    "hot dog,pizza,donut,cake,frisbee,sports ball,skateboard,surfboard,tennis racket,baseball bat,"
    "baseball glove,kite,skis,snowboard,bicycle,motorcycle,car,dog,cat,bird,horse,teddy bear,"
    "scissors,clock,hair drier,toothbrush,vase,tie"
)
OBJECT_CLASSES_CONTEXT = (
    "chair,couch,potted plant,bed,dining table,toilet,sink,refrigerator,microwave,oven,toaster,"
    "dishwasher,traffic light,fire hydrant,stop sign,parking meter,bench,airplane,bus,train,truck,"
    "boat,car,motorcycle,sheep,cow,elephant,bear,zebra,giraffe"
)
# A primary object is small and still meaningful (a phone in hand is ~0.7% of a 1080x1920 frame);
# a sofa only earns the camera when it is the scene.
OBJECT_MIN_AREA_PRIMARY = 0.006
OBJECT_MIN_AREA_CONTEXT = 0.05
OBJECT_MIN_CONF_CONTEXT = 0.55


def _object_class_ids(model: Any) -> tuple[dict[int, str], dict[int, str]]:
    """Map the COCO class ids to the primary and context object tiers, once per model."""
    names = getattr(model, "names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    primary = {v.strip().lower() for v in OBJECT_CLASSES_PRIMARY.split(",") if v.strip()}
    context = {v.strip().lower() for v in OBJECT_CLASSES_CONTEXT.split(",") if v.strip()}
    override = os.getenv("REFRAME_OBJECT_CLASSES")
    if override:
        primary = {v.strip().lower() for v in override.split(",") if v.strip()}
        context = set()
    by_name: dict[str, int] = {}
    for i, name in names.items():
        by_name.setdefault(str(name).strip().lower(), int(i))
    return (
        {by_name[n]: n for n in primary if n in by_name},
        {by_name[n]: n for n in context if n in by_name},
    )


def detect_framing_regions(frame: np.ndarray) -> list[dict[str, Any]]:
    from .detection_service import get_detection_service

    height, width = frame.shape[:2]
    small, _ = _detection_frame(frame)
    service = get_detection_service()
    detections = service.predict(small)
    DETECTION_STATUS["person"] = service.status
    primary = set(os.environ.get("REFRAME_OBJECT_CLASSES", OBJECT_CLASSES_PRIMARY).split(","))
    context = set() if os.environ.get("REFRAME_OBJECT_CLASSES") else set(OBJECT_CLASSES_CONTEXT.split(","))
    regions = []
    for detection in detections:
        label = detection["label"]
        x, y, w, h = detection["box"]
        confidence = detection["confidence"]
        if label == "person":
            if w < 0.045 or h < 0.25:
                continue
            if (x <= 0.01 or x + w >= 0.99) and w < 0.15:
                continue
        elif label in context:
            if w * h < OBJECT_MIN_AREA_CONTEXT or confidence < OBJECT_MIN_CONF_CONTEXT:
                continue
        elif label not in primary or w * h < OBJECT_MIN_AREA_PRIMARY or confidence < 0.45:
            continue
        regions.append({
            "box": [x * width, y * height, w * width, h * height],
            "label": label,
            "confidence": confidence,
            "class_id": detection["class_id"],
        })
    return regions[:32]


def detect_portrait_candidates(frame: np.ndarray, regions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    Fuse face detections with non-overlapping YOLOv8 'person' upper-body anchors.
    Critical for podcasts: if Speaker A is facing the camera (detected as a face) while Speaker B
    is in side-profile or occluded by a Shure SM7B boom mic (missed by face detector, caught by
    YOLOv8 'person'), returning `faces` early used to make Speaker B completely invisible!
    """
    faces = detect_face_candidates(frame)
    # When a close-up foreground face (>= 8% width) is centered in frame, drop distant tiny background
    # bokeh extras (< 3.5% width) so background faces do not masquerade as two-person dialogues.
    w_frame = float(frame.shape[1] or 1920)
    if faces and max(f["box"][2] for f in faces) >= 0.08 * w_frame:
        faces = [f for f in faces if f["box"][2] >= 0.035 * w_frame]
    try:
        people = [r for r in (detect_framing_regions(frame) if regions is None else regions) if r["label"] == "person"]
        largest = max((r["box"][2] * r["box"][3] for r in people), default=0.0)
        body_heads: list[dict[str, Any]] = []
        for person in people:
            x, y, w, h = person["box"]
            if w * h < largest * 0.30:
                continue
            # When clear faces are already detected, require stronger person confidence (>= 0.62) so
            # background wall sculptures / mannequins (~0.40-0.55) cannot veto solo Full 9:16 framing.
            if faces and float(person.get("confidence", 0.0)) < 0.62:
                continue
            head_cand = [x + w * 0.2, y, w * 0.6, h * 0.35]
            hx, hy, hw, hh = head_cand
            hcx, hcy = hx + hw / 2.0, hy + hh / 2.0
            overlaps_existing_face = any(
                _iou(head_cand, f["box"]) > 0.18
                or (
                    abs((f["box"][0] + f["box"][2] / 2.0) - hcx) < max(hw, f["box"][2]) * 0.65
                    and abs((f["box"][1] + f["box"][3] / 2.0) - hcy) < max(hh, f["box"][3]) * 0.75
                )
                for f in faces
            )
            if not overlaps_existing_face:
                body_heads.append(
                    {"box": head_cand, "full_box": person["box"], "score": float(w * h * 0.21), "confidence": person["confidence"], "body_anchor": True}
                )
        return dedupe_faces(faces + body_heads) if (faces or body_heads) else []
    except Exception:
        return faces


def detect_object_candidates(frame: np.ndarray, regions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Last-resort anchor for footage with nobody readable in it.

    Without this the tracker returns nothing for a ball, a board or a car chase and the camera parks
    dead centre for the whole shot. The boxes go through the same speaker tracker as faces, so the
    stickiness and cooldown that stop a blink from whipping the camera apply to objects too.
    """
    try:
        objects = [r for r in (detect_framing_regions(frame) if regions is None else regions) if r["label"] not in ("person", "saliency")]
        out: list[dict[str, Any]] = []
        for r in objects:
            x, y, w, h = r["box"]
            out.append({"box": [x, y, w, h], "score": float(w * h * r["confidence"]), "confidence": r["confidence"], "label": r["label"], "object_anchor": True})
        return out
    except Exception:
        return []


def smooth_trajectory_zero_phase(
    points: list[dict[str, Any]], sigma: float = 1.6
) -> list[dict[str, Any]]:
    """
    Apply a zero-phase Gaussian kernel inside each continuous camera shot (never across `cut=True`)
    for gimbal-smooth virtual camera pans with no lookahead lag.

    A template change is its own shot for this purpose: averaging a solo crop's aim with the two panel
    centres either side of a split hands the viewer a composition that never existed, and the camera
    spends the first second of every shot sliding out of the framing it was supposed to start in.
    """
    if len(points) <= 2:
        return points

    mode_of = lambda pt: pt.get("mode") or "fill"
    shots: list[list[dict[str, Any]]] = []
    for pt in points:
        if pt.get("cut") or not shots or mode_of(shots[-1][-1]) != mode_of(pt):
            shots.append([dict(pt)])
        else:
            shots[-1].append(dict(pt))

    radius = max(1, int(math.ceil(sigma * 2.5)))
    weights = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-radius, radius + 1)]

    smoothed_out: list[dict[str, Any]] = []
    for shot in shots:
        n = len(shot)
        if n <= 2:
            smoothed_out.extend(shot)
            continue
        for i in range(n):
            wx, wy, wx2, wy2, wsum = 0.0, 0.0, 0.0, 0.0, 0.0
            for k_idx, offset in enumerate(range(-radius, radius + 1)):
                j = min(n - 1, max(0, i + offset))
                w = weights[k_idx]
                p = shot[j]
                wx += float(p.get("x", 0.5)) * w
                wy += float(p.get("y", 0.38)) * w
                wx2 += float(p.get("x2", p.get("x", 0.5))) * w
                wy2 += float(p.get("y2", p.get("y", 0.38))) * w
                wsum += w
            item = dict(shot[i])
            item["x"] = round(wx / wsum, 4)
            item["y"] = round(wy / wsum, 4)
            item["x2"] = round(wx2 / wsum, 4)
            item["y2"] = round(wy2 / wsum, 4)
            smoothed_out.append(item)
    return smoothed_out


def limit_pursuit_rate(
    points: list[dict[str, Any]], max_speed_x: float, max_speed_y: float
) -> list[dict[str, Any]]:
    """Cap how fast the virtual camera may travel, one layout at a time.

    Smoothing is not a speed limit: a detector that puts a head somewhere else for one sample, or the
    speaking rule handing a crop to a different person, both survive a Gaussian as a shorter, faster
    step. The pan then reads as a jolt — a camera correcting itself — rather than a camera following a
    person.

    Each layout keeps its opening framing, because that is the frame the cut was made on; the rest of
    the run catches up at the pursuit law's own speed. Slowing down is the cost, and a run that lasts
    the hold has the time to pay it.
    """
    if len(points) <= 2 or max_speed_x <= 0 or max_speed_y <= 0:
        return points
    mode_of = lambda pt: pt.get("mode") or "fill"
    out = [dict(p) for p in points]
    begin = 0
    for i in range(1, len(out) + 1):
        if i == len(out) or out[i].get("cut") or mode_of(out[i]) != mode_of(out[i - 1]):
            for k in range(begin + 1, i):
                dt = max(1e-3, float(out[k]["t"]) - float(out[k - 1]["t"]))
                for key, limit in (
                    ("x", max_speed_x),
                    ("y", max_speed_y),
                    ("x2", max_speed_x),
                    ("y2", max_speed_y),
                ):
                    if key not in out[k]:
                        continue
                    here = float(out[k][key])
                    there = float(out[k - 1].get(key, here))
                    step = here - there
                    cap = limit * dt
                    if abs(step) > cap:
                        out[k][key] = round(there + math.copysign(cap, step), 4)
            begin = i
    return out


def _scene_spans(track_points: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Sample indices grouped into continuous camera shots, delimited by `cut=True` keyframes.

    OpenShorts' split_layout decides per scene, not per frame, so the coexistence vote and the
    median centres have to be computed inside each shot.
    """
    spans: list[tuple[int, int]] = []
    begin = 0
    for i, pt in enumerate(track_points):
        if i > begin and pt.get("cut"):
            spans.append((begin, i))
            begin = i
    if track_points:
        spans.append((begin, len(track_points)))
    return spans


def _normalise_centre(x: float, y: float) -> tuple[float, float]:
    return (
        round(max(0.05, min(0.95, x)), 4),
        round(max(0.08, min(0.92, y)), 4),
    )


def apply_scene_split_layout(
    track_points: list[dict[str, Any]],
    scene_candidates: list[list[dict[str, Any]]],
    orig_w: int,
    orig_h: int,
    out_w: int = 1080,
    out_h: int = 1920,
) -> list[dict[str, Any]]:
    """Stamp OpenShorts' per-scene 2-up decision onto the track points the browser renders.

    Without this, a shot only renders as a split on the frames where both faces survived
    detection; the frames in between fall back to a single-subject crop and the renderer's
    `x2 ?? 1 - fx` mirror puts the bottom panel on empty space.
    """
    scenes: list[dict[str, Any]] = []
    for begin, end in _scene_spans(track_points):
        pair = analyze_split_scene(scene_candidates[begin:end], orig_w)
        if pair is None:
            continue
        # Two panels only earn their place if each one frames a different body. Separation alone
        # passes when a single person leans across the shot, and the export then stacks two copies of
        # the same region, which the QA pass read as an accidental duplicate layer.
        if not independent_split(orig_w, orig_h, out_w, out_h, pair):
            continue
        left = _normalise_centre(pair[0][0] / orig_w, pair[0][1] / orig_h)
        right = _normalise_centre(pair[1][0] / orig_w, pair[1][1] / orig_h)
        scenes.append({"start": track_points[begin]["t"], "end": track_points[end - 1]["t"], "left": list(left), "right": list(right)})
        for pt in track_points[begin:end]:
            if pt["mode"] == "fit":
                continue
            pt["mode"], pt["x"], pt["y"], pt["x2"], pt["y2"] = "split", left[0], left[1], right[0], right[1]
    return scenes


def enforce_layout_hold(
    track_points: list[dict[str, Any]],
    min_hold_sec: float,
    split_min_hold_sec: float = 0.0,
) -> int:
    """Reject layout changes that do not survive their own minimum on screen.

    The per-frame two-shot test flips between "two faces survived detection" and "one did" on
    sub-second boundaries, so an export can cut to a split panel and back inside a second. A run
    shorter than the hold is folded into whichever neighbouring layout lasts longer, which keeps
    the layout on screen long enough to read as an intentional shot instead of a twitch.

    A split answers a different question than a crop does — it asserts that two people matter at the
    same moment — and a viewer needs longer to read that assertion than to read one face. So a
    two-panel run has its own, longer floor, and a shot that cannot hold the split that long is
    framed as a single subject instead of flickering between the two.
    """
    if not track_points or min_hold_sec <= 0:
        return 0

    def spans() -> list[list[int]]:
        out: list[list[int]] = []
        for i, pt in enumerate(track_points):
            if out and track_points[out[-1][1] - 1]["mode"] == pt["mode"]:
                out[-1][1] = i + 1
            else:
                out.append([i, i + 1])
        return out

    def span_len(begin: int, end: int) -> float:
        # A layout is on screen until the next keyframe changes it, so the run's last second belongs to
        # the run after it. Measuring to the run's own final keyframe instead called a one-sample twitch
        # a four-second shot and left it untouched.
        stop = track_points[end]["t"] if end < len(track_points) else track_points[-1]["t"]
        return stop - track_points[begin]["t"]

    def floor_for(mode: str) -> float:
        if mode == "split" and split_min_hold_sec > min_hold_sec:
            return split_min_hold_sec
        return min_hold_sec

    changed = 0
    protected_runs: set[int] = set()
    runs = spans()
    while len(runs) > 1:
        short = next(
            (
                k
                for k in range(len(runs))
                if runs[k][0] not in protected_runs
                and span_len(*runs[k]) < floor_for(track_points[runs[k][0]]["mode"])
            ),
            None,
        )
        if short is None:
            break
        begin, end = runs[short]
        # The first and last runs have only one neighbour to give them up to; an interior run joins
        # whichever side already holds the screen longer.
        if short == 0:
            keeper_run = runs[1]
        elif short == len(runs) - 1:
            keeper_run = runs[short - 1]
        else:
            prev_len = span_len(*runs[short - 1])
            next_len = span_len(*runs[short + 1])
            keeper_run = runs[short - 1] if prev_len >= next_len else runs[short + 1]
        keeper = track_points[keeper_run[0]]
        keep = keeper["mode"]
        if keep in ("fit", "editorial") and any(pt.get("cutProtected") for pt in track_points[begin:end]):
            protected_runs.add(begin)
            continue
        for pt in track_points[begin:end]:
            pt["mode"] = keep
            if keep == "split":
                # A folded-in frame has to adopt the pair it is joining; its own x2 is its own
                # subject, which would put both panels on the same person.
                pt["x"], pt["y"], pt["x2"], pt["y2"] = (
                    keeper["x"], keeper["y"], keeper.get("x2", keeper["x"]), keeper.get("y2", keeper["y"]),
                )
        changed += end - begin
        runs = spans()
    return changed


def adaptive_cut_threshold(dists: list[float]) -> float:
    """Put the shot boundary where this footage's own distances say it is.

    A fixed Bhattacharyya cut-off cannot work across sources: on a bright multi-camera show real cuts
    sit above 0.5, and on a night-vision segment the same hard cut between two rooms measures 0.2, so
    one threshold either stutters on handheld motion or never re-aims at all. Measured on the ghost
    hunter footage: 12 cuts found in a 25 s window that a scene-change scan of the same file shows
    had ~14, and 0 found in another 25 s window that really had ~6.

    So take the median as the bulk of "same shot" and stand three robust deviations above it, the way
    PySceneDetect's ContentDetector caps its threshold with the running spread of the metric. Clamped
    both ways: a clip with no variation at all should not call its own noise a cut, and a bright show
    should not need a bigger jump than 0.55 to be believed.
    """
    if len(dists) < 8:
        return CUT_THR_MAX
    med = statistics.median(dists)
    mad = statistics.median([abs(d - med) for d in dists]) * 1.4826
    return max(CUT_THR_MIN, min(CUT_THR_MAX, med + CUT_THR_SIGMAS * mad))


def _sampled_frames(cap, start_f: int, end_f: int, stride: int):
    """Yield every `stride`-th frame from `start_f` on, walking the decoder forward.

    One seek lands the decoder on the first sample; after that `grab()` steps over the frames the
    sample does not want without paying the BGR copy `read()` does. Seeking per sample measured
    ~45 ms/frame against ~2 ms for a grab on this footage, and the analysis runs the walk twice.
    """
    idx = start_f
    while idx < end_f:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        yield idx, frame
        idx += 1
        for _ in range(stride - 1):
            if idx >= end_f or not cap.grab():
                return
            idx += 1


def analyze_video_with_openshorts(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    framing_mode: str = "auto",
    progress_cb=None,
    is_cancelled=None,
) -> dict[str, Any]:
    """
    Run OpenShorts' scene-cut + SpeakerTracker + SmoothedCameraman + split_layout + active_speaker
    analysis on a video segment and return both:
    - Normalized `TrackPoint[]` for ClipForge's interactive browser canvas preview
    - Validated OpenShorts `EditPlan` shots (`TRACK`, `CENTER`, `BLUR`, `SPLIT`, `EDITORIAL`)
      and compact keyframes for native FFmpeg `sendcmd` export.
    """
    import cv2

    if not math.isfinite(start_sec) or start_sec < 0 or (end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec)):
        raise ValueError("Analysis needs a finite, positive source interval")
    if framing_mode not in ("auto", "fill", "split", "fit", "grid3", "editorial"):
        raise ValueError("Unknown framing mode")
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_dur = total_frames / fps if total_frames > 0 else 60.0
    end_s = min(total_dur, end_sec if end_sec is not None else total_dur)
    start_f = max(0, int(round(start_sec * fps)))
    end_f = max(start_f + 1, int(round(end_s * fps)))

    out_w, out_h = 1080, 1920
    cameraman = SmoothedCameraman(out_w, out_h, orig_w, orig_h, aspect_ratio=ASPECT_RATIO, fps=fps)
    tracker = SpeakerTracker(stabilization_frames=4, cooldown_frames=max(1, round(fps)), fps=fps)

    sample_fps = max(1.0, min(15.0, finite(os.environ.get("REFRAME_SAMPLE_FPS"), 7.5)))
    stride = max(1, int(round(fps / sample_fps)))
    sample_dt = stride / fps
    cameraman.set_sample_interval(sample_dt)
    sample_frames = list(range(start_f, end_f, stride))

    # --- Phase A: read this video's own editing rhythm before deciding anything about the camera ---
    # Only histograms and small-frame motion here, so the expensive detector pass can start from
    # numbers measured on the footage instead of from a constant someone typed in.
    hists: list[Any] = []
    motions: list[float] = []
    grays: list[float] = []
    sats: list[float] = []
    bottom_graphics: list[bool] = []
    prev_hist = None
    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    decoded_a = 0
    try:
        for f_idx, frame in _sampled_frames(cap, start_f, end_f, stride):
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("Video analysis cancelled")
            decoded_a += 1
            if progress_cb is not None and decoded_a % 10 == 0:
                progress_cb(0.35 * decoded_a / max(1, len(sample_frames)))
            small = cv2.resize(frame, (64, 36))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            motions.append(0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean()))
            grays.append(float(gray.mean()))
            sats.append(float(hsv[:, :, 1].mean()))
            # Measure centered lower-third text band edge stroke contrast rather than coarse 64x36 textures
            # so sunlit floors, clothing patterns, and letterbox borders do not false-trigger Blur+Fit.
            h_f, w_f = frame.shape[:2]
            sub_roi = cv2.cvtColor(
                cv2.resize(
                    frame[int(h_f * 0.78):int(h_f * 0.90), int(w_f * 0.22):int(w_f * 0.78)],
                    (128, 20),
                ),
                cv2.COLOR_BGR2GRAY,
            )
            gx_roi = np.abs(cv2.Sobel(sub_roi, cv2.CV_32F, 1, 0, ksize=3))
            p95, p50 = float(np.percentile(sub_roi, 95)), float(np.percentile(sub_roi, 50))
            bottom_graphics.append(
                bool(float(gx_roi.mean()) > 52.0 and float(sub_roi.max()) > 225.0 and (p95 - p50) > 80.0)
            )
            prev_gray = gray
            hists.append(hist)
        if not hists:
            raise ValueError("No video frames decoded in the requested interval")
    except BaseException:
        cap.release()
        raise
    if progress_cb is not None:
        progress_cb(0.35)

    # Where this footage's own edit boundary sits, measured from its histogram distances rather than
    # from a number that was right for the clip somebody tested with.
    cut_thr = adaptive_cut_threshold(
        [
            cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
            for a, b in zip(hists, hists[1:])
            if a is not None and b is not None
        ]
    )
    raw_gaps: list[float] = []
    last_change_f = sample_frames[0] if sample_frames else start_f
    prev_hist = None
    for f_idx, hist in zip(sample_frames, hists):
        if hist is None:
            continue
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > cut_thr:
            if (f_idx - last_change_f) >= int(fps * 0.45):
                raw_gaps.append((f_idx - last_change_f) / fps)
                last_change_f = f_idx
        prev_hist = hist

    black_frames = sum(1 for g in grays if g < 16.0)
    rhythm = signature_of(
        shot_gaps=raw_gaps,
        samples=len(sample_frames),
        face_hits=0,
        two_face_hits=0,
        person_hits=0,
        object_hits=0,
        face_widths=[],
        black_hits=black_frames,
        motion_sum=float(sum(motions)),
        travel=[],
        sample_dt=sample_dt,
        subjects_measured=False,
    )
    camera_tuning = tune(rhythm)
    cameraman.apply_tuning(camera_tuning)
    # A cut may only re-aim the camera once the previous shot has lasted about as long as this
    # video's own average shot; anything shorter is followed by a pan, which reads as an edit.
    min_cut_gap_f = max(int(fps * 0.45), int(round(fps * camera_tuning.cut_snap_min_sec)))
    track_points: list[dict[str, Any]] = []
    scene_candidates: list[list[dict[str, Any]]] = []
    face_hits = two_face_hits = person_hits = object_hits = 0
    face_widths: list[float] = []
    travel: list[float] = []
    prev_target_x: float | None = None
    prev_hist = None  # phase A left this on the last sample; the cut pass starts fresh
    last_cut_f = start_f  # the opening sample counts as a cut, so the hold applies from the first second
    cameraman.begin_scene()
    # Who is who, and whose mouth moves when: the tracker's ids are per shot, so a person needs an
    # identity of their own before the speaking rule can be applied to a whole clip.
    registry = PersonRegistry()
    mouths = MouthMemory()
    watcher = TalkWatcher()
    samples: list[dict[str, Any]] = []
    object_memory: dict[str, list[dict[str, Any]]] = {}
    next_object_id = 0

    try:
        total_steps = max(1, len(sample_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for step, (f_idx, frame) in enumerate(_sampled_frames(cap, start_f, end_f, stride)):
            if step >= len(hists):
                break
            hist = hists[step]
            if hist is None:
                continue

            t = f_idx / fps
            is_cut = (f_idx == start_f)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > cut_thr and (f_idx - last_cut_f) >= int(fps * 0.45):
                    # A new shot always drops the sticky speaker, so the camera aims at whoever is
                    # actually on screen in it. It only jumps there when the previous shot has been
                    # held long enough; otherwise the pursuit pans.
                    tracker.reset()
                    object_memory.clear()
                    if (f_idx - last_cut_f) >= min_cut_gap_f:
                        is_cut = True
                        last_cut_f = f_idx
                        cameraman.begin_scene()
            prev_hist = hist

            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("Video analysis cancelled")
            regions = detect_framing_regions(frame)
            portraits = detect_portrait_candidates(frame, regions)
            objects = detect_object_candidates(frame, regions)
            cands = portraits or objects
            target_box = tracker.get_target(cands, f_idx, orig_w)
            if target_box:
                cameraman.update_target(target_box)
                if prev_target_x is not None:
                    travel.append(abs(cameraman.target_center_x - prev_target_x) / cameraman.crop_width)
                prev_target_x = cameraman.target_center_x

            # The two anchor kinds never mix on one sample, but keep the split maths person-only: a
            # pair of chairs is not a two-shot.
            subject_cands = [c for c in cands if not c.get("object_anchor")]
            scene_candidates.append(subject_cands)
            faces = dedupe_faces(subject_cands)
            big = [c for c in faces if c["box"][2] >= MIN_FACE_WIDTH * orig_w]
            if any(c["box"][2] > 0 for c in subject_cands):
                face_hits += 1
            if len(big) >= 2:
                two_face_hits += 1
            if any(c.get("body_anchor") for c in subject_cands):
                person_hits += 1
            if any(c.get("object_anchor") for c in cands):
                object_hits += 1
            if big:
                face_widths.append(max(c["box"][2] for c in big) / float(orig_w))

            x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_cut)
            cx_norm = ((x1 + x2) / 2.0) / float(orig_w)
            cy_norm = cameraman.subject_anchor_y

            # Check if frame is a two-shot split candidate or wide context shot
            is_flashback = (sats[step] if step < len(sats) else 128.0) < 18.0
            has_bot_graphic = bool(bottom_graphics[step] if step < len(bottom_graphics) else False)
            mode = "fill"
            x2_norm, y2_norm = cx_norm, cy_norm
            pair = None if is_flashback else _two_largest(subject_cands, orig_w)
            if framing_mode == "split" or (
                framing_mode == "auto"
                and pair is not None
                and (pair[1]["box"][0] - pair[0]["box"][0]) >= MIN_SEPARATION * orig_w
            ):
                if pair is not None:
                    lc, rc = _centre(pair[0]["box"]), _centre(pair[1]["box"])
                    if independent_split(orig_w, orig_h, 1080, 1920, (lc, rc)):
                        cx_norm, cy_norm = lc[0] / orig_w, lc[1] / orig_h
                        x2_norm, y2_norm = rc[0] / orig_w, rc[1] / orig_h
                        mode = "split"
            elif framing_mode == "fit":
                mode = "fit"

            people: list[dict[str, Any]] = []
            if faces:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for cand in faces:
                    box = [float(v) for v in cand.get("full_box", cand["box"])]
                    kind = "person" if cand.get("body_anchor") else "face"
                    key = registry.observe(box, orig_w, orig_h, sample_dt, t, kind, not is_cut)
                    if key < 0:
                        continue
                    # Identity, the mouth patch and the framing all measure the same head box, so a
                    # person detected as a body one shot and as a face the next stays one person.
                    hbox = head_box(box, kind)
                    watcher.observe(
                        key, t, mouths.activity(key, hbox, gray_full, orig_w, orig_h, 1.0)
                    )
                    sil = refine_subject_silhouette(frame, hbox, kind)
                    people.append({
                        "key": key,
                        "box": box,
                        "kind": kind,
                        "confidence": cand.get("confidence", 0.5),
                        "yaw_bias": sil.head_yaw_bias,
                        "head_crown_y": sil.head_crown_y,
                        "occlusion": sil.occlusion_score,
                    })
            ambient_labels = set(OBJECT_CLASSES_CONTEXT.split(",")) | {
                "vase", "handbag", "tie", "backpack", "suitcase", "cup", "bottle", "wine glass", "bowl", "book",
            }
            for label in sorted({obj["label"] for obj in objects}):
                group = [obj for obj in objects if obj["label"] == label]
                identified, memory, next_object_id = associate_faces(
                    group, object_memory.get(label, []), f_idx, orig_w, next_object_id,
                    ttl=max(1, round(fps)),
                )
                object_memory[label] = memory
                for oc, identity in zip(group, identified):
                    people.append({"key": -1, "object_id": f"object:{identity['id']}",
                                   "box": [float(v) for v in oc["box"]], "kind": "object",
                                   "label": label, "confidence": oc["confidence"],
                                   "background_context": bool(faces) and (label in ambient_labels)})

            track_points.append(
                {
                    "t": round(t, 3),
                    "x": round(max(0.05, min(0.95, cx_norm)), 4),
                    "y": round(max(0.08, min(0.92, cy_norm)), 4),
                    "x2": round(max(0.05, min(0.95, x2_norm)), 4),
                    "y2": round(max(0.08, min(0.92, y2_norm)), 4),
                    "trackId": tracker.active_speaker_id or 1,
                    "mode": mode,
                    "cut": is_cut,
                    "subjectType": (
                        "object"
                        if any(c.get("object_anchor") for c in cands)
                        else "person" if any(c.get("body_anchor") for c in cands) else "face"
                    ),
                    "hasBottomGraphic": has_bot_graphic,
                    "isFlashback": is_flashback,
                }
            )
            samples.append({"t": round(t, 3), "people": people})
            if progress_cb is not None and step % 10 == 0:
                progress_cb(0.35 + 0.65 * (step + 1) / total_steps)
    finally:
        cap.release()
    if not samples:
        raise ValueError("No video frames decoded during tracking")
    if progress_cb is not None:
        progress_cb(1.0)

    full_rhythm = signature_of(
        shot_gaps=raw_gaps,
        samples=len(sample_frames),
        face_hits=face_hits,
        two_face_hits=two_face_hits,
        person_hits=person_hits,
        object_hits=object_hits,
        face_widths=face_widths,
        black_hits=black_frames,
        motion_sum=float(sum(motions)),
        travel=travel,
        sample_dt=sample_dt,
    )
    tuning = tune(full_rhythm)
    tuning.keep_measured_camera(camera_tuning)

    if tuning.split_allowed or framing_mode == "split":
        apply_scene_split_layout(track_points, scene_candidates, orig_w, orig_h)
    else:
        # This video never really stages two people side by side, so a 2-up panel would be pointed
        # at empty air for the whole shot. Keep every frame on one subject.
        for pt in track_points:
            if pt["mode"] == "split":
                pt["mode"], pt["x2"], pt["y2"] = "fill", pt["x"], pt["y"]
    layout_hold_folded = enforce_layout_hold(track_points, tuning.layout_hold_sec, tuning.split_min_hold_sec)

    # The punch-in envelope is cut at 0.2 s; the speaking rule reads 0.4 s windows, and the louder half
    # of a pair is what decides whether anybody had the floor.
    beat_env = audio_envelope(video_path, start_sec, max(0.5, end_s - start_sec), window_s=BEAT_WINDOW)
    beats = emphasis_from_envelope(beat_env)
    # punch_in works on the same segment-relative clock as the 8kHz envelope, at the sampling rate
    for pt, z in zip(track_points, zoom_curve(len(track_points), fps / stride, beats)):
        pt["zoom"] = round(1.0 + (z - 1.0) * tuning.punch_gain, 4)

    # Last word on the track: whoever holds the floor has to be in the frame, and a fold or a layout
    # gate that would have taken him out of it is overruled by this.
    loudness = [
        max(beat_env[i], beat_env[i + 1]) if i + 1 < len(beat_env) else beat_env[i]
        for i in range(0, len(beat_env), 2)
    ]
    talks = attribute_talkers(
        watcher,
        registry,
        loudness,
        math.ceil(max(0.5, end_s - start_sec) / WINDOW_SECONDS),
        visible_by_window(samples, watcher.start_t),
    )
    # The main speaker is the biggest floor-holder, and the floor is only counted by the attribution
    # above — asking the registry before it runs names the person who was simply on screen most.
    main_key = registry.main_speaker()
    # Classify human behaviors & situational framing rules (solo close-up cut protection,
    # subject+object union crop, triad panel anchors, gaze lead-room) before & after speaker priority
    behavior_pre = classify_human_behaviors(
        track_points, samples, talks, loudness, orig_w, orig_h, main_key, WINDOW_SECONDS
    )
    speakers = apply_speaker_priority(
        track_points, samples, main_key, talks, orig_w, orig_h,
        tuning.layout_hold_sec, tuning.split_min_hold_sec,
    )
    behavior_report = classify_human_behaviors(
        track_points, samples, talks, loudness, orig_w, orig_h, main_key, WINDOW_SECONDS
    )
    speakers.update(registry.report(main_key))
    speakers["behavior"] = behavior_report
    from .layout_decision import LayoutPolicy, anchor_solo_subjects, finalize_layouts

    layout_policy = LayoutPolicy.from_environment(tuning.layout_hold_sec)
    anchor_solo_subjects(track_points, samples, talks, main_key, orig_w, orig_h,
                         layout_policy, behavior_report["behavior_by_sample"])
    # What the viewer actually gets: the decided layouts, eased, and held to the pursuit law. The
    # guarantees are re-read off this track rather than off the decisions, because a camera that is
    # slowed down is a different camera from the one that was planned.
    delivered = limit_pursuit_rate(
        smooth_trajectory_zero_phase(track_points), tuning.max_speed_x, tuning.max_speed_y
    )
    layouts = finalize_layouts(delivered, samples, orig_w, orig_h, talks, main_key, framing_mode,
                               layout_policy, behaviors=behavior_report["behavior_by_sample"])
    if layouts:
        layouts[-1]["end"] = end_f / fps
    if delivered and len(delivered) == len(samples):
        speakers["coverage_delivered"] = coverage_of(
            delivered, samples, talks, orig_w, orig_h, main_key
        )
        speakers["main_lock_delivered"] = main_lock_of(
            delivered, samples, talks, main_key, float(samples[0]["t"]), orig_w, orig_h
        )

    return {
        "ok": True,
        "engine": "openshorts_reframe_v2",
        "width": orig_w,
        "height": orig_h,
        "fps": fps,
        "crop_width": cameraman.crop_width,
        "crop_height": cameraman.crop_height,
        "emphasis_beats": beats,
        "tuning": tuning.as_dict(),
        "scene_cut_threshold": round(cut_thr, 3),
        "shots_detected": len(raw_gaps),
        "layout_hold_sec": tuning.layout_hold_sec,
        "split_min_hold_sec": tuning.split_min_hold_sec,
        "layout_hold_folded": layout_hold_folded,
        "speakers": speakers,
        "detectors": dict(DETECTION_STATUS),
        "sample_fps": round(fps / stride, 3),
        "layout_decisions": layouts,
        "detections": [{"t": sample["t"], "subjects": [
            {"id": person["key"] if person["key"] >= 0 else person.get("object_id"),
             "kind": person["kind"], "label": person.get("label", person["kind"]),
             "box": [round(v / scale, 6) for v, scale in zip(person["box"], (orig_w, orig_h, orig_w, orig_h))],
             "confidence": person.get("confidence", 0.0)}
            for person in sample["people"]]} for sample in samples],
        "track": delivered,
    }


# ============================================================================
# YOLO + SAM 2 Hybrid Perception, One-Euro Filter (autoclip-main), and
# ML Human Behavior & Situational Decision Matrix
# ============================================================================



class OneEuroFilter:
    """Adaptive low-pass filter (1€ Filter) ported from autoclip-main/reframe/smoothing.py."""

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-4, dt))

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._x_prev = float(x)
            self._t_prev = float(t)
            return float(x)

        dt = float(t) - self._t_prev
        if dt <= 1e-6:
            return self._x_prev

        dx = (float(x) - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(t)
        return x_hat


def smooth_axis_one_euro(
    times: list[float],
    values: list[float],
    cuts: list[bool] | None = None,
    min_cutoff: float = 0.5,
    beta: float = 0.3,
    dead_zone: float = 0.012,
    max_velocity: float = 0.85,
) -> list[float]:
    """Smooth a 1D normalized trajectory with dead-zone + 1€ filter + velocity clamping, resetting at cuts."""
    if len(values) <= 1 or len(times) != len(values):
        return list(values)

    filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    out: list[float] = []
    anchor = float(values[0])
    prev_out = anchor
    prev_t = float(times[0])

    for i, (t, val) in enumerate(zip(times, values)):
        is_cut = bool(cuts[i]) if (cuts and i < len(cuts)) else (i == 0)
        if i == 0 or is_cut:
            filt.reset()
            anchor = float(val)
            raw = filt(t, anchor)
            prev_out = raw
            prev_t = float(t)
            out.append(round(raw, 4))
            continue

        if abs(float(val) - anchor) > dead_zone:
            anchor = float(val)

        filtered = filt(t, anchor)
        dt = max(1e-3, float(t) - prev_t)
        max_delta = max_velocity * dt
        delta = filtered - prev_out
        if abs(delta) > max_delta:
            filtered = prev_out + math.copysign(max_delta, delta)
            filt._x_prev = filtered

        prev_out = filtered
        prev_t = float(t)
        out.append(round(filtered, 4))
    return out


def build_single_pass_crop_expr(
    keyframes: list[tuple[float, int]],
    max_keyframes: int = 85,
) -> str:
    """Build a nested FFmpeg piecewise-linear expression `if(lt(t,t1),v0+slope*(t-t0),...)` (from autoclip-main)."""
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return str(int(keyframes[0][1]))

    pts = list(keyframes)
    thinned: list[tuple[float, int]] = [pts[0]]
    for i in range(1, len(pts) - 1):
        t_prev, v_prev = thinned[-1]
        t_cur, v_cur = pts[i]
        t_next, v_next = pts[i + 1]
        dt_total = t_next - t_prev
        if dt_total > 1e-6:
            expected = v_prev + (v_next - v_prev) * ((t_cur - t_prev) / dt_total)
            if abs(v_cur - expected) <= 1.5:
                continue
        thinned.append(pts[i])
    thinned.append(pts[-1])

    while len(thinned) > max_keyframes:
        best_idx = 1
        best_err = math.inf
        for i in range(1, len(thinned) - 1):
            t0, v0 = thinned[i - 1]
            t1, v1 = thinned[i]
            t2, v2 = thinned[i + 1]
            dt = max(1e-6, t2 - t0)
            interp = v0 + (v2 - v0) * ((t1 - t0) / dt)
            err = abs(v1 - interp)
            if err < best_err:
                best_err = err
                best_idx = i
        thinned.pop(best_idx)

    if len(thinned) == 1 or all(v == thinned[0][1] for _, v in thinned):
        return str(int(thinned[0][1]))

    def _nest(idx: int) -> str:
        t0, v0 = thinned[idx]
        t1, v1 = thinned[idx + 1]
        dt = t1 - t0
        if dt <= 1e-4 or v0 == v1:
            seg = f"{int(v0)}"
        else:
            slope = (v1 - v0) / dt
            seg = f"({int(v0)}+{slope:.2f}*(t-{t0:.3f}))"
        if idx == len(thinned) - 2:
            return f"if(lt(t,{t1:.3f}),{seg},{int(v1)})"
        return f"if(lt(t,{t1:.3f}),{seg},{_nest(idx + 1)})"

    return _nest(0)


@dataclass
class SubjectMaskMetrics:
    """Approximate subject metrics from the inexpensive contour analysis."""

    centroid_x: float
    centroid_y: float
    head_crown_y: float
    mask_area_ratio: float
    occlusion_score: float
    head_yaw_bias: float
    backend: str


def refine_subject_silhouette(
    frame_bgr: np.ndarray,
    box: list[float],
    kind: str = "face",
) -> SubjectMaskMetrics:
    """Cheap contour metrics; explicit video segmentation runs outside this per-person loop."""
    import cv2

    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = [float(v) for v in box]
    x0 = int(max(0, min(w_img - 2, round(x))))
    y0 = int(max(0, min(h_img - 2, round(y))))
    x1 = int(max(x0 + 2, min(w_img, round(x + w))))
    y1 = int(max(y0 + 2, min(h_img, round(y + h))))

    default_cx = (x0 + x1) / 2.0 / max(1, w_img)
    default_cy = (y0 + y1) / 2.0 / max(1, h_img)
    default_crown = y0 / max(1, h_img)

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
        return SubjectMaskMetrics(default_cx, default_cy, default_crown, 0.5, 0.0, 0.0, "bbox")

    small_h, small_w = min(64, roi.shape[0]), min(64, roi.shape[1])
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    luma = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = (cr >= 130) & (cr <= 178) & (cb >= 75) & (cb <= 135) & (luma >= 28)
    lower_half = luma[small_h // 2 :, :]
    dark_mic_ratio = float(np.mean(lower_half < 24)) if lower_half.size else 0.0

    if int(np.sum(skin_mask)) >= (small_h * small_w) // 10:
        fg_mask = skin_mask
    else:
        _, thresh = cv2.threshold(luma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_mask = thresh > 0

    ys, xs = np.where(fg_mask)
    if len(xs) < 8:
        return SubjectMaskMetrics(
            default_cx, default_cy, default_crown, 0.5, dark_mic_ratio, 0.0, "opencv_bbox"
        )

    rel_cx = float(np.mean(xs)) / max(1, small_w)
    rel_cy = float(np.mean(ys)) / max(1, small_h)
    rel_crown = float(np.min(ys)) / max(1, small_h)

    cx_norm = (x0 + rel_cx * (x1 - x0)) / max(1, w_img)
    cy_norm = (y0 + rel_cy * (y1 - y0)) / max(1, h_img)
    crown_norm = (y0 + rel_crown * (y1 - y0)) / max(1, h_img)
    area_ratio = float(len(xs)) / float(small_w * small_h)
    yaw_bias = max(-1.0, min(1.0, (rel_cx - 0.5) * 2.5))

    return SubjectMaskMetrics(
        centroid_x=round(cx_norm, 4),
        centroid_y=round(cy_norm, 4),
        head_crown_y=round(crown_norm, 4),
        mask_area_ratio=round(area_ratio, 3),
        occlusion_score=round(dark_mic_ratio, 3),
        head_yaw_bias=round(yaw_bias, 3),
        backend="opencv_silhouette",
    )


BEHAVIOR_SOLO_MONOLOGUE = "solo_monologue"
BEHAVIOR_EMPHATIC_GESTURE = "emphatic_gesture"
BEHAVIOR_OBJECT_DEMO = "object_demonstration"
BEHAVIOR_DYADIC_DIALOGUE = "dyadic_dialogue"
BEHAVIOR_TRIAD_PANEL = "triad_panel"
BEHAVIOR_GROUP_REACTION = "group_reaction"
BEHAVIOR_SILENT_CUTAWAY = "silent_listener_cutaway"
BEHAVIOR_PHYSICAL_ACTION = "physical_action"


@dataclass
class SituationalDecision:
    window_index: int
    behavior: str
    recommended_mode: str
    target_x: float
    target_y: float
    target_x2: float | None = None
    target_y2: float | None = None
    target_x3: float | None = None
    target_y3: float | None = None
    one_euro_beta: float = 0.30
    zoom_bias: float = 1.0
    cut_protected: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window_index,
            "behavior": self.behavior,
            "recommended_mode": self.recommended_mode,
            "target_x": round(self.target_x, 4),
            "target_y": round(self.target_y, 4),
            "one_euro_beta": round(self.one_euro_beta, 3),
            "zoom_bias": round(self.zoom_bias, 3),
            "cut_protected": self.cut_protected,
            "reason": self.reason,
        }


def classify_human_behaviors(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, Any],
    loudness: list[float],
    frame_w: int,
    frame_h: int,
    main_key: int | None = None,
    window_sec: float = 0.4,
) -> dict[str, Any]:
    """Analyze character behaviors across the clip and apply situational framing improvements."""
    if not track_points or len(track_points) != len(samples) or frame_w <= 0 or frame_h <= 0:
        return {"decisions": [], "summary": {}, "dominant_genre": "general"}

    start_t = float(samples[0].get("t", 0.0))
    decisions: list[SituationalDecision] = []
    behavior_counts: dict[str, int] = {}

    shot_spans: list[tuple[int, int]] = []
    s_begin = 0
    for idx, pt in enumerate(track_points):
        if idx > s_begin and pt.get("cut"):
            shot_spans.append((s_begin, idx))
            s_begin = idx
    shot_spans.append((s_begin, len(track_points)))

    shot_for_sample: dict[int, tuple[int, int]] = {}
    for b, e in shot_spans:
        for idx in range(b, e):
            shot_for_sample[idx] = (b, e)

    for i, (pt, s) in enumerate(zip(track_points, samples)):
        t = float(s.get("t", pt.get("t", 0.0)))
        w_idx = max(0, int((t - start_t) / max(1e-3, window_sec)))
        tw = talks.get(w_idx)
        loud = float(loudness[w_idx]) if w_idx < len(loudness) else 0.0

        all_entities = s.get("people", [])
        humans = foreground_humans(all_entities)
        objects = [p for p in all_entities if p.get("box") and p.get("kind") == "object"]

        shot_b, shot_e = shot_for_sample.get(i, (i, i + 1))
        shot_humans_counts = [
            len(foreground_humans(samples[k].get("people", [])))
            for k in range(shot_b, shot_e)
        ]
        median_shot_humans = (
            sorted(shot_humans_counts)[len(shot_humans_counts) // 2]
            if shot_humans_counts
            else len(humans)
        )

        talker_keys = set(tw.keys) if (tw is not None and getattr(tw, "talking", False)) else set()
        active_humans = [p for p in humans if int(p.get("key", -1)) in talker_keys]

        motion_vel = 0.0
        if i > shot_b:
            dt = max(1e-3, float(pt.get("t", 0.0)) - float(track_points[i - 1].get("t", 0.0)))
            motion_vel = abs(float(pt.get("x", 0.5)) - float(track_points[i - 1].get("x", 0.5))) / dt

        behavior = BEHAVIOR_SOLO_MONOLOGUE
        rec_mode = str(pt.get("mode") or "fill")
        tx = float(pt.get("x", 0.5))
        ty = float(pt.get("y", 0.38))
        tx2 = pt.get("x2")
        ty2 = pt.get("y2")
        tx3 = pt.get("x3")
        ty3 = pt.get("y3")
        beta = 0.30
        zoom_bias = 1.0
        cut_protected = False
        reason = "steady_framing"

        if median_shot_humans == 1 and len(humans) == 1:
            solo = humans[0]
            bx, by, bw, bh = [float(v) for v in solo["box"]]
            scx = (bx + bw / 2.0) / frame_w
            scy = (by + bh * 0.38) / frame_h
            yaw = float(solo.get("yaw_bias", 0.0))
            scx = max(0.08, min(0.92, scx + yaw * 0.018))
            if bw >= 0.085 * frame_w:
                cut_protected = True
                pt["cutProtected"] = True
                if rec_mode in ("fit", "editorial"):
                    rec_mode = "fill"
                    pt["mode"] = "fill"
                    tx, ty = scx, scy
                    pt["x"], pt["y"] = round(tx, 4), round(ty, 4)
                if not talker_keys:
                    behavior = BEHAVIOR_SILENT_CUTAWAY
                    reason = "solo_closeup_reaction_preserved_as_fill"
                elif motion_vel > 0.25 or loud > 0.72:
                    behavior = BEHAVIOR_EMPHATIC_GESTURE
                    beta = 0.45
                    zoom_bias = 1.05
                    reason = "emphatic_solo_speaker_with_lead_room"
                else:
                    behavior = BEHAVIOR_SOLO_MONOLOGUE
                    beta = 0.22
                    reason = "locked_solo_speaker_one_euro_stabilized"

        if rec_mode == "fill" and len(humans) >= 1 and objects:
            lead = active_humans[0] if active_humans else humans[0]
            lx, ly, lw, lh = [float(v) for v in lead["box"]]
            lcx = (lx + lw / 2.0) / frame_w
            lcy = (ly + lh * 0.38) / frame_h
            nearest_obj = min(
                objects,
                key=lambda o: abs(((o["box"][0] + o["box"][2] / 2.0) / frame_w) - lcx),
            )
            ox, oy, ow, oh = [float(v) for v in nearest_obj["box"]]
            ocx = (ox + ow / 2.0) / frame_w
            ocy = (oy + oh / 2.0) / frame_h
            if abs(ocx - lcx) <= 0.26:
                behavior = BEHAVIOR_OBJECT_DEMO
                tx = max(0.08, min(0.92, 0.72 * lcx + 0.28 * ocx))
                ty = max(0.10, min(0.88, 0.80 * lcy + 0.20 * ocy))
                pt["x"], pt["y"] = round(tx, 4), round(ty, 4)
                reason = "subject_plus_object_union_crop"

        if motion_vel >= 0.32 and rec_mode == "fill":
            behavior = BEHAVIOR_PHYSICAL_ACTION
            beta = 0.65
            reason = "high_velocity_action_pursuit"

        if len(humans) >= 2:
            if rec_mode == "split" and tx2 is not None:
                behavior = BEHAVIOR_DYADIC_DIALOGUE
                beta = 0.20
                reason = "dyadic_two_speaker_zero_overlap_split"
            elif len(humans) == 3 and len(active_humans) >= 2:
                ordered = sorted(humans, key=lambda p: p["box"][0] + p["box"][2] / 2.0)
                c0 = (ordered[0]["box"][0] + ordered[0]["box"][2] / 2.0) / frame_w
                c1 = (ordered[1]["box"][0] + ordered[1]["box"][2] / 2.0) / frame_w
                c2 = (ordered[2]["box"][0] + ordered[2]["box"][2] / 2.0) / frame_w
                if (c1 - c0) >= 0.16 and (c2 - c1) >= 0.16:
                    behavior = BEHAVIOR_TRIAD_PANEL
                    lead_h = active_humans[0]
                    others = [p for p in ordered if p is not lead_h][:2]
                    if len(others) == 2:
                        tx = (lead_h["box"][0] + lead_h["box"][2] / 2.0) / frame_w
                        ty = (lead_h["box"][1] + lead_h["box"][3] * 0.38) / frame_h
                        tx2 = (others[0]["box"][0] + others[0]["box"][2] / 2.0) / frame_w
                        ty2 = (others[0]["box"][1] + others[0]["box"][3] * 0.38) / frame_h
                        tx3 = (others[1]["box"][0] + others[1]["box"][2] / 2.0) / frame_w
                        ty3 = (others[1]["box"][1] + others[1]["box"][3] * 0.38) / frame_h
                        pt["x3"], pt["y3"] = round(tx3, 4), round(ty3, 4)
                        reason = "triad_panel_three_speaker_anchors"
            elif len(humans) >= 4 or rec_mode == "fit":
                behavior = BEHAVIOR_GROUP_REACTION
                reason = "group_panel_cinema_card_fit"

        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        decisions.append(
            SituationalDecision(
                window_index=w_idx,
                behavior=behavior,
                recommended_mode=rec_mode,
                target_x=tx,
                target_y=ty,
                target_x2=tx2,
                target_y2=ty2,
                target_x3=tx3,
                target_y3=ty3,
                one_euro_beta=beta,
                zoom_bias=zoom_bias,
                cut_protected=cut_protected,
                reason=reason,
            )
        )

    total = max(1, len(decisions))
    if (
        behavior_counts.get(BEHAVIOR_GROUP_REACTION, 0)
        + behavior_counts.get(BEHAVIOR_TRIAD_PANEL, 0)
        > 0.25 * total
    ):
        genre = "studio_panel_or_game_show"
    elif behavior_counts.get(BEHAVIOR_DYADIC_DIALOGUE, 0) > 0.25 * total:
        genre = "podcast_or_interview"
    elif behavior_counts.get(BEHAVIOR_PHYSICAL_ACTION, 0) > 0.15 * total:
        genre = "challenge_or_sports_action"
    elif behavior_counts.get(BEHAVIOR_OBJECT_DEMO, 0) > 0.15 * total:
        genre = "product_or_object_demonstration"
    else:
        genre = "creator_storytelling_or_monologue"

    return {
        "dominant_genre": genre,
        "behavior_counts": behavior_counts,
        "decisions": [d.as_dict() for d in decisions[:64]],
        # One audio window can straddle a cut; layout evidence must stay sample-aligned.
        "behavior_by_sample": {float(sample["t"]): d.behavior for sample, d in zip(samples, decisions)},
    }
