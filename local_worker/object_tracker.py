"""
OpenShorts-Powered Video Object & Speaker Tracking Module
(MediaPipe BlazeFace + YOLOv8 Person/Object Cascade + SpeakerTracker + SmoothedCameraman + split_layout + active_speaker).
"""
import importlib.util
import os
from typing import TypedDict

from .openshorts_engine import (
    OneEuroFilter,
    build_single_pass_crop_expr,
    classify_human_behaviors,
    refine_subject_silhouette,
    smooth_axis_one_euro,
)
from .openshorts_engine import (
    JUMP_CONFIRM_FRAMES,
    MIN_COEXISTENCE,
    MIN_SEPARATION,
    SmoothedCameraman,
    SpeakerTracker,
    analyze_split_scene,
    analyze_video_with_openshorts,
    apply_scene_split_layout,
    associate_faces,
    compact_keyframes,
    crop_position,
    crop_size,
    dedupe_faces,
    editorial_filtergraph,
    independent_split,
    needs_context,
    smooth_trajectory_zero_phase,
    split_filtergraph,
    split_geometry,
    subject_position,
)


class TrackCapabilities(TypedDict):
    ultralytics_yolo: bool
    opencv: bool
    mediapipe: bool
    active_backend: str
    supported_trackers: list[str]
    openshorts_features: list[str]
    detector_status: str
    detector_warning: str | None
    sam2_configured: bool
    sam2_installed: bool


def get_tracking_capabilities() -> TrackCapabilities:
    from .detection_service import get_detection_service

    detector = get_detection_service()
    has_yolo = importlib.util.find_spec("ultralytics") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None
    has_mp = importlib.util.find_spec("mediapipe") is not None
    if has_yolo and has_mp:
        backend = "openshorts_reframe_v2_yolo_mediapipe"
    elif has_cv2:
        backend = "openshorts_reframe_v2_opencv"
    else:
        backend = "openshorts_browser_reframe_v2"
    return {
        "ultralytics_yolo": has_yolo,
        "opencv": has_cv2,
        "mediapipe": has_mp,
        "active_backend": backend,
        "detector_status": detector.status,
        "detector_warning": detector.warning,
        "sam2_installed": importlib.util.find_spec("sam2") is not None,
        "sam2_configured": bool(os.environ.get("SAM2_CONFIG") and os.path.isfile(os.environ.get("SAM2_MODEL_PATH", ""))),
        "supported_trackers": [
            "openshorts_speaker_tracker_smoothed_cameraman",
            "split_layout_active_speaker_gate",
            "editorial_detail_context_stack",
            "yolo_mediapipe_face_person_cascade",
        ],
        "openshorts_features": [
            f"SpeakerTracker (cooldown_frames=30, 3.0x sticky hysteresis)",
            f"SmoothedCameraman (25% safe-zone, jump_confirm_frames={JUMP_CONFIRM_FRAMES}, cut snap)",
            f"split_layout (coexistence>={int(MIN_COEXISTENCE*100)}%, separation>={int(MIN_SEPARATION*100)}%, median centres)",
            "active_speaker (mouth region 20-80th percentile + 8kHz audio RMS gate)",
            "editing_director (TRACK, CENTER, SPLIT, EDITORIAL, BLUR + face-safe ASS captions)",
            "cached_yolo_detection (local checkpoint; actual status reported separately)",
            "opencv_contour_metrics (heuristic, not segmentation or pose)",
            "optional_sam2_video_segmentation (explicit prompted job; bounded 10-second segments)",
            "weighted_layout_safety (visibility, content, confidence, balance, relevance, resolution)",
            "one_euro_filter_and_single_pass_crop_expr (autoclip-main jitter-free rest & zero-lag action pan)",
            "human_behavior_situational_decision_matrix (8 behavioral states across all video genres)",
        ],
    }


__all__ = [
    "OneEuroFilter",
    "build_single_pass_crop_expr",
    "classify_human_behaviors",
    "refine_subject_silhouette",
    "smooth_axis_one_euro",
    "SmoothedCameraman",
    "SpeakerTracker",
    "analyze_split_scene",
    "analyze_video_with_openshorts",
    "apply_scene_split_layout",
    "associate_faces",
    "compact_keyframes",
    "crop_position",
    "crop_size",
    "dedupe_faces",
    "editorial_filtergraph",
    "get_tracking_capabilities",
    "independent_split",
    "needs_context",
    "smooth_trajectory_zero_phase",
    "split_filtergraph",
    "split_geometry",
    "subject_position",
]
