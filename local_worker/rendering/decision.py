"""The single editing-decision structure both the browser and the backend agree on.

The browser already holds everything in a `ClipJob` (trims, track, words, settings). This module is
the wire format for that job plus the media references the backend needs, and it is deliberately
strict: a decision that does not fully describe the render must fail here rather than produce a
silently different video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

VALID_LAYOUTS = ("auto", "fill", "split", "grid3", "editorial", "fit")
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30.0


@dataclass
class TrackPoint:
    """One measured framing decision in absolute source seconds, as the tracker emits it."""

    t: float
    x: float
    y: float
    mode: str = "fill"
    x2: float | None = None
    y2: float | None = None
    x3: float | None = None
    y3: float | None = None
    zoom: float = 1.0
    cut: bool = False
    has_bottom_graphic: bool = False
    subject_type: str = "face"
    tracking_confidence: float | None = None
    layout_decision: dict[str, Any] | None = None

    @staticmethod
    def from_payload(p: dict[str, Any]) -> "TrackPoint":
        return TrackPoint(
            t=float(p["t"]),
            x=float(p["x"]),
            y=float(p["y"]),
            mode=str(p.get("mode") or "fill"),
            x2=_opt_float(p.get("x2")),
            y2=_opt_float(p.get("y2")),
            x3=_opt_float(p.get("x3")),
            y3=_opt_float(p.get("y3")),
            zoom=float(p.get("zoom") or 1.0),
            cut=bool(p.get("cut")),
            has_bottom_graphic=bool(p.get("hasBottomGraphic")),
            subject_type=str(p.get("subjectType") or "face"),
            tracking_confidence=_opt_float(p.get("trackingConfidence")),
            layout_decision=p.get("layoutDecision") if isinstance(p.get("layoutDecision"), dict) else None,
        )


@dataclass
class Word:
    """One caption token with its source-time window, used for karaoke highlighting."""

    w: str
    start: float
    end: float


@dataclass
class AudioTrack:
    kind: str  # "original" | "music" | "voiceover"
    path: str
    volume: float = 1.0
    start: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck_under_voice: bool = False


@dataclass
class CaptionStyle:
    preset: str = "opus"
    highlight: str = "#22ff4e"
    secondary_highlight: str = "#ffd400"
    position_y: float = 0.78
    words_per_page: int = 3
    font_size: int | None = None
    show_emojis: bool = True
    show_hook: bool = True
    font: str = "Manrope"


@dataclass
class RenderDecision:
    """Everything needed to execute the edit; nothing the renderer should invent on its own."""

    source_path: str
    source_width: int
    source_height: int
    source_fps: float
    start: float
    end: float
    layout: str
    track: list[TrackPoint]
    words: list[Word]
    hook: str
    style: CaptionStyle
    # What the container's `r_frame_rate` says, kept beside the measured mean so a variable-rate
    # source can be recognised instead of silently undersampled.
    source_nominal_fps: float = 0.0
    zoom_punch: bool = True
    visual_effects: bool = True
    manual_x: float | None = None
    manual_y: float | None = None
    voice_enhance: bool = True
    target_fps: str = "29.97"
    audio: list[AudioTrack] = field(default_factory=list)
    brand_label: str | None = None
    # Clip-wide reservation, preserved by replace() through wordless layout segments.
    fit_caption_space: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        raise ValueError("Render coordinates must be finite")
    return out


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ValueError(f"Render decision is missing required field '{key}'")
    return payload[key]


def decision_from_payload(payload: dict[str, Any]) -> RenderDecision:
    """Validate the browser's payload into a decision, refusing an under-specified render."""
    source_path = str(_require(payload, "source_path"))
    start = float(_require(payload, "start"))
    end = float(_require(payload, "end"))
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Render decision requires finite 0 <= start < end")

    layout = str(payload.get("layout") or "auto")
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'; expected one of {', '.join(VALID_LAYOUTS)}")

    raw_track = payload.get("track") or []
    if not isinstance(raw_track, list):
        raise ValueError("'track' must be a list of measured framing points")
    track = [TrackPoint.from_payload(p) for p in raw_track]

    raw_words = payload.get("words") or []
    words = [Word(str(w.get("w") or ""), float(w.get("start", 0)), float(w.get("end", 0))) for w in raw_words]

    style_payload = payload.get("style") or {}
    style = CaptionStyle(
        preset=str(style_payload.get("preset") or "opus"),
        highlight=str(style_payload.get("highlight") or "#22ff4e"),
        secondary_highlight=str(style_payload.get("secondaryHighlight") or "#ffd400"),
        position_y=float(style_payload.get("positionY") if style_payload.get("positionY") is not None else 0.78),
        words_per_page=int(style_payload.get("wordsPerPage") or 3),
        font_size=_opt_float(style_payload.get("fontSize")),
        show_emojis=bool(style_payload.get("showEmojis", True)),
        show_hook=bool(style_payload.get("showHook", True)),
        font=str(style_payload.get("font") or "Manrope"),
    )

    audio = [
        AudioTrack(
            kind=str(a.get("kind") or "original"),
            path=str(a["path"]),
            volume=float(a.get("volume", 1.0)),
            start=float(a.get("start", 0.0)),
            fade_in=float(a.get("fade_in", 0.0)),
            fade_out=float(a.get("fade_out", 0.0)),
            duck_under_voice=bool(a.get("duck_under_voice")),
        )
        for a in (payload.get("audio") or [])
    ]

    return RenderDecision(
        source_path=source_path,
        source_width=int(_require(payload, "source_width")),
        source_height=int(_require(payload, "source_height")),
        source_fps=float(payload.get("source_fps") or 30.0),
        source_nominal_fps=_opt_float(payload.get("source_nominal_fps")) or 0.0,
        start=start,
        end=end,
        layout=layout,
        track=track,
        words=words,
        hook=str(payload.get("hook") or ""),
        style=style,
        fit_caption_space=style.preset != "none" and bool(words),
        zoom_punch=bool(payload.get("zoomPunch", True)),
        visual_effects=bool(payload.get("visualEffects", True)),
        manual_x=_opt_float(payload.get("manualX")),
        manual_y=_opt_float(payload.get("manualY")),
        voice_enhance=bool(payload.get("voiceEnhance", True)),
        target_fps=str(payload.get("fps") or "29.97"),
        audio=audio,
        brand_label=(str(payload["brandLabel"]) if payload.get("brandLabel") else None),
    )
