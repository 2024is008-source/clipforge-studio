"""Backend-native rendering for ClipForge.

One editing-decision structure (`decision.RenderDecision`) is produced by the browser and executed
here by FFmpeg, so preview and export cannot drift into two different editing systems.
"""

from .audio import build_audio_graph
from .decision import RenderDecision, decision_from_payload
from .framerate import choose_source_rate, rate_arg, requested_fps, resolve_output_fps
from .filters import (
    build_composite_graph,
    build_video_graph,
    caption_anchor_y,
    caption_band,
    crop_keyframes,
    effective_layout,
    panel_crop_chain,
    resolved_font_size,
    sendcmd_file_text,
    subject_panels,
    vignette_alpha_expression,
)
from .renderer import (
    EncoderUnavailable,
    RenderCancelled,
    RenderFailed,
    RenderRequest,
    render_clip,
    render_job_dir,
)
from .subtitles import write_ass
from .validator import ExpectedOutput, probe_source, validate_render
from .encoders import detect_encoder, encoder_video_args

__all__ = [
    "EncoderUnavailable",
    "ExpectedOutput",
    "RenderCancelled",
    "RenderDecision",
    "RenderFailed",
    "RenderRequest",
    "build_audio_graph",
    "build_composite_graph",
    "build_video_graph",
    "caption_anchor_y",
    "caption_band",
    "crop_keyframes",
    "decision_from_payload",
    "detect_encoder",
    "effective_layout",
    "encoder_video_args",
    "choose_source_rate",
    "rate_arg",
    "requested_fps",
    "resolve_output_fps",
    "panel_crop_chain",
    "render_clip",
    "render_job_dir",
    "sendcmd_file_text",
    "subject_panels",
    "validate_render",
    "write_ass",
]
