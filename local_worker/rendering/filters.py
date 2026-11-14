"""Crop, zoom, layout and grade as one FFmpeg filtergraph.

Parity rule: every number here reproduces what `src/lib/render.ts` paints on the canvas, because the
browser preview and this export must be the same edit. `crop_rect` is a direct port of
`drawCroppedSubject`, `word_punch_zoom` of `computeOpenShortsPunchZoom`, and `sample_track` of
`sampleTrack`. Where FFmpeg cannot match the canvas exactly (the eased layout transition, the radial
vignette) the approximation is stated at the call site rather than hidden.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .decision import OUTPUT_HEIGHT, OUTPUT_WIDTH, RenderDecision, TrackPoint
from .framerate import rate_arg

# src/lib/render.ts punch_in.py curve constants
PUNCH_RISE_SEC = 0.25
PUNCH_HOLD_SEC = 1.3
PUNCH_FALL_SEC = 0.55
PUNCH_SPAN_SEC = PUNCH_RISE_SEC + PUNCH_HOLD_SEC + PUNCH_FALL_SEC
PUNCH_MAX_ZOOM = 1.12
PUNCH_MIN_GAP_SEC = 8.0

# sampleTrack's cut lead: the incoming shot is shown this early so the cut lands on a frame edge.
CUT_SNAP_SEC = 0.02

# drawCroppedSubject: the subject sits at 0.36 of the crop height, and a panel that must clear the
# burned-in caption band only uses the top 84% of the source.
HEAD_ANCHOR = 0.36
SUBTITLE_SAFE_VH = 0.84
FIT_CAPTION_FOREGROUND_HEIGHT = int(OUTPUT_HEIGHT * 0.70)

GOLD_KEYWORD_RE = re.compile(
    r"(\$|\d|million|millions|billion|thousand|grand|everything|never|impossible|charity|winner)",
    re.IGNORECASE,
)

# CSS `contrast(1.05) saturate(1.14) brightness(1.02)` multiplies brightness; FFmpeg's `eq` adds it,
# which clipped a mid-grey frame to pure white (YAVG 143 -> 255). Working in planar RGB restores the
# multiplication, and the round trip through gbrp keeps `eq` from promoting tv-range 16-235 to 0-255.
GRADE_FILTER = "format=gbrp,eq=contrast=1.05:saturation=1.14,colorchannelmixer=rr=1.02:gg=1.02:bb=1.02,format=yuv420p"

# drawVignette's stops: no falloff until 62% of the way from the inner radius to the corner, then a
# linear ramp to rgba(0,0,0,0.36) that holds to the edge of the frame.
VIGNETTE_TRANSPARENT_STOP = 0.62
VIGNETTE_EDGE_ALPHA = 0.36
VIGNETTE_INNER_RADIUS = 0.3
VIGNETTE_CENTRE_Y = 0.48


def vignette_alpha_expression(width: int, height: int) -> str:
    """The canvas gradient as one `geq` alpha expression, so the falloff matches the preview.

    FFmpeg's own `vignette` filter was measured darkening a 1080x1920 corner crop to YAVG 0 against
    59 unvignetted, because its ellipse is inscribed in the frame rather than scaled to its diagonal.
    The canvas uses two circles with slightly different centres; one centre at their midpoint is the
    stated approximation.
    """
    r0 = VIGNETTE_INNER_RADIUS * height
    r1 = math.hypot(width / 2, height / 2)
    cut = r0 + VIGNETTE_TRANSPARENT_STOP * (r1 - r0)
    span = r1 - cut
    max_alpha = 255 * VIGNETTE_EDGE_ALPHA
    return (
        f"clip((sqrt(pow(X-{width / 2:.3f},2)+pow(Y-{VIGNETTE_CENTRE_Y * height:.3f},2))"
        f"-{cut:.3f})/{span:.3f}*{max_alpha:.3f},0,{max_alpha:.3f})"
    )


def caption_band(layout: str, font_size: int, height: int = OUTPUT_HEIGHT) -> tuple[int, int] | None:
    """The dark strip the preview paints behind captions in multi-panel layouts, or None.

    src/lib/render.ts:584-588 fills rgba(8,8,10,0.84) at H*0.5 with height fontSize*1.75 only for
    split/grid3, so the burn-in gets the same box rather than a per-line ASS border.
    """
    if layout not in ("split", "grid3"):
        return None
    band_h = int(round(font_size * 1.75)) // 2 * 2
    return int(round(height * 0.5 - band_h / 2)), band_h


def resolved_font_size(style) -> int:
    from .subtitles import PRESET_STYLE

    if style.font_size:
        return int(style.font_size)
    return PRESET_STYLE.get(style.preset, PRESET_STYLE["opus"])[0]


def caption_anchor_y(decision: RenderDecision, layout: str) -> float:
    """Where the caption block is centred, using the preview's own layout overrides.

    There is no face-aware lift in `src/lib/render.ts`: the band position depends only on the layout
    and on whether the source frame carries a burned-in lower third.
    """
    configured = max(0.0, min(1.0, decision.style.position_y))
    if layout in ("split", "grid3"):
        return 0.5
    if layout == "editorial":
        return 0.58
    if layout == "fit":
        return 0.83
    if any(p.has_bottom_graphic for p in decision.track):
        return min(configured, 0.65)
    return configured


def _cut_dip_filter(decision: RenderDecision | None) -> str:
    """Brief luminance dip across verified interior scene cuts, matching `drawCutDip` in render.ts."""
    if not decision or not decision.track or len(decision.track) < 2:
        return ""
    first_t = decision.track[0].t
    cuts = [
        round(p.t - decision.start, 4)
        for p in decision.track
        if p.cut and p.t > first_t + 0.001 and 0.05 < (p.t - decision.start) < decision.duration - 0.05
    ]
    if not cuts:
        return ""
    # Bound terms so extreme montage clips stay well within filter string limits.
    terms = "+".join(f"pow(max(0,1-abs(t-{ct:.4f})/0.15),2)" for ct in cuts[:24])
    return f"eq=brightness='-0.50*min(1,{terms})':eval=frame,"


def _edge_blur_filter(duration: float, lead: bool, trail: bool) -> str:
    """Blur-dip across a layout seam: out of the previous window, into the next, like the reference
    edits. Captions and the caption band are painted after this, so text stays sharp through the dip."""
    edge = min(0.16, duration / 4)
    if edge < 0.06 or not (lead or trail):
        return ""
    ramps = ((0.0, 0.35, 26), (0.35, 0.70, 14), (0.70, 1.0, 6))
    parts = []
    if lead:
        for a, b, sigma in ramps:
            parts.append(f"gblur=sigma={sigma}:enable='between(t,{a * edge:.3f},{b * edge:.3f})'")
    if trail:
        for a, b, sigma in ramps:
            parts.append(
                f"gblur=sigma={sigma}:enable='between(t,{duration - b * edge:.3f},{duration - a * edge:.3f})'"
            )
    return ",".join(parts) + ","


def _mono_pulse_filter(decision: RenderDecision | None, enabled: bool) -> str:
    """A desaturated beat on the emphasis punch, the reference's black-and-white insert."""
    if not decision or not enabled:
        return ""
    starts = [s - decision.start for s in word_punch_starts(decision.words)]
    terms = [s for s in starts if 0.0 <= s < decision.duration][:4]
    if not terms:
        return ""
    return ",".join(f"hue=s=0:enable='between(t,{s:.3f},{s + 0.6:.3f})'" for s in terms) + ","


def _grade_chain(
    label: str,
    vignette_index: int | None,
    band: tuple[int, int] | None,
    out: str = "[out]",
    decision: RenderDecision | None = None,
    edges: tuple[bool, bool] = (True, True),
    mono: bool = True,
) -> list[str]:
    """Grade -> vignette overlay -> seam blur / mono pulse / cut dip -> caption band, as the canvas paints."""
    effects = decision is None or decision.visual_effects
    tail = _edge_blur_filter(decision.duration if decision else 0.0, *edges) if effects else ""
    tail += _mono_pulse_filter(decision, mono and effects)
    tail += _cut_dip_filter(decision) if effects else ""
    if band is not None:
        y, h = band
        tail += f"drawbox=x=0:y={y}:w={OUTPUT_WIDTH}:h={h}:color=0x08080A@0.84:t=fill,"
    tail += "format=yuv420p"
    if not effects:
        # The vignette input stays declared but unconnected; FFmpeg ignores unmapped inputs.
        return [f"[{label}]{tail}{out}"]
    if vignette_index is None:
        return [f"[{label}]{GRADE_FILTER},{tail}{out}"]
    # The vignette is one rgba frame; overlay repeats its last frame for the whole window, so the
    # default `eof_action=repeat` is load-bearing and `shortest=1` must never be added here.
    return [
        f"[{label}]{GRADE_FILTER}[vg]",
        f"[vg][{vignette_index}:v]overlay=format=auto[vv]",
        f"[vv]{tail}{out}",
    ]


@dataclass
class Panel:
    """One rectangular region of the output canvas and where its framing comes from."""

    key: str  # "main" | "right" | "third"
    dest_w: int
    dest_h: int
    zoom_mul: float
    zoom_min: float
    exclude_bottom: bool
    crop_mode: str = "subject"  # "subject" | "centre" | "full"


@dataclass
class CropFrame:
    t: float
    w: float
    h: float
    x: float
    y: float


def smoothstep(x: float) -> float:
    v = max(0.0, min(1.0, x))
    return v * v * (3.0 - 2.0 * v)


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


def is_gold_keyword(word: str) -> bool:
    return bool(GOLD_KEYWORD_RE.search(word))


def word_punch_starts(words: list) -> list[float]:
    """Trigger times of the emphasis punch: same tokens, same 8s spacing, as the zoom curve."""
    starts: list[float] = []
    last_trigger = -100.0
    for w in words:
        start = _word_start(w)
        if start - last_trigger < PUNCH_MIN_GAP_SEC:
            continue
        text = _word_text(w)
        if text.endswith("!") or is_gold_keyword(text) or len(text) >= 9:
            last_trigger = start
            starts.append(start)
    return starts


def word_punch_zoom(words: list, t: float, punch_gain: float) -> float:
    """The canvas punch: an asymmetric smoothstep push on emphasised tokens, at most one per 8s."""
    max_zoom = 1.0
    for start in word_punch_starts(words):
        dt = t - start
        if 0.0 <= dt <= PUNCH_SPAN_SEC:
            if dt < PUNCH_RISE_SEC:
                amount = smoothstep(dt / PUNCH_RISE_SEC)
            elif dt < PUNCH_RISE_SEC + PUNCH_HOLD_SEC:
                amount = 1.0
            else:
                amount = 1.0 - smoothstep((dt - PUNCH_RISE_SEC - PUNCH_HOLD_SEC) / PUNCH_FALL_SEC)
            max_zoom = max(max_zoom, 1.0 + (PUNCH_MAX_ZOOM - 1.0) * amount)
    return 1.0 + (max_zoom - 1.0) * punch_gain


def _word_text(w) -> str:
    return str(getattr(w, "w", "") if not isinstance(w, dict) else w.get("w", ""))


def _word_start(w) -> float:
    if isinstance(w, dict):
        return float(w.get("s", w.get("start", 0.0)) or 0.0)
    return float(getattr(w, "start", getattr(w, "s", 0.0)) or 0.0)


def _point_field(p: TrackPoint, name: str) -> float | None:
    return getattr(p, name)


def sample_track(track: list[TrackPoint], t: float) -> TrackPoint:
    """The preview's own sampler: eased between measured points, hard-snapped across a cut.

    Port of `sampleTrack` in src/lib/reframe.ts, including the two rules that decide how the camera
    feels. Linear easing made the export a different camera from the preview, and interpolating
    across a scene cut blended two shots that were never meant to meet.
    """
    if not track:
        return TrackPoint(t=t, x=0.5, y=0.38, x2=0.5, y2=0.38, mode="fill")
    if t <= track[0].t:
        return track[0]
    if t >= track[-1].t:
        return track[-1]
    lo, hi = 0, len(track) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if track[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = track[lo], track[hi]
    if b.cut:
        # The last 20 ms of the outgoing shot already shows the incoming frame, as in the canvas.
        return b if t >= b.t - CUT_SNAP_SEC else a
    span = b.t - a.t
    k = 0.0 if span <= 0 else max(0.0, min(1.0, (t - a.t) / span))
    e = smoothstep(k)

    def lerp(name: str, default: float) -> float:
        va, vb = _point_field(a, name), _point_field(b, name)
        if va is None or vb is None:
            return default
        return va + (vb - va) * e

    def lerp_or(name: str, fallback: str) -> float:
        """A panel axis the tracker never measured follows its primary anchor, as `?? ` does."""
        va = _point_field(a, name)
        vb = _point_field(b, name)
        if va is None or vb is None:
            return lerp(fallback, 0.5)
        return va + (vb - va) * e

    def lerp_zoom() -> float:
        va = _point_field(a, "zoom")
        vb = _point_field(b, "zoom")
        from_a = 1.0 if va is None else va
        to_b = 1.0 if vb is None else vb
        return from_a + (to_b - from_a) * e

    return TrackPoint(
        t=t,
        x=lerp("x", 0.5),
        y=lerp("y", 0.38),
        x2=lerp_or("x2", "x"),
        y2=lerp_or("y2", "y"),
        x3=lerp_or("x3", "x"),
        y3=lerp_or("y3", "y"),
        zoom=lerp_zoom(),
        mode=a.mode,
        cut=a.cut,
        has_bottom_graphic=a.has_bottom_graphic or b.has_bottom_graphic,
        subject_type=a.subject_type,
        tracking_confidence=a.tracking_confidence,
        layout_decision=a.layout_decision,
    )


def crop_rect(
    fx: float,
    fy: float,
    zoom: float,
    source_w: int,
    source_h: int,
    dest_w: int,
    dest_h: int,
    exclude_bottom: bool,
    crop_mode: str = "subject",
) -> tuple[float, float, float, float]:
    """Port of drawCroppedSubject's source-rectangle maths, in source pixels."""
    vw, vh = float(source_w), float(source_h)
    target_aspect = dest_w / dest_h
    usable_vh = vh * SUBTITLE_SAFE_VH if exclude_bottom else vh
    z = max(1.0, zoom)
    crop_h = usable_vh / z
    crop_w = crop_h * target_aspect
    if crop_w > vw:
        crop_w = vw / z
        crop_h = crop_w / target_aspect
    sx = _clamp(fx * vw - crop_w / 2.0, 0.0, vw - crop_w)
    if crop_mode == "centre":
        sy = _clamp(fy * vh - crop_h / 2.0, 0.0, vh - crop_h)
    else:
        sy = _clamp(fy * vh - crop_h * HEAD_ANCHOR, 0.0, usable_vh - crop_h)
    return crop_w, crop_h, sx, sy


def panel_zoom(panel: Panel, base_zoom: float) -> float:
    return max(base_zoom * panel.zoom_mul, panel.zoom_min)


def _anchor_for(panel: Panel, p: TrackPoint, decision: RenderDecision) -> tuple[float, float]:
    if decision.manual_x is not None or decision.manual_y is not None:
        mx = decision.manual_x if decision.manual_x is not None else p.x
        my = decision.manual_y if decision.manual_y is not None else p.y
        if panel.key == "main":
            return mx, my
    if panel.key == "right":
        return p.x2 if p.x2 is not None else p.x, p.y2 if p.y2 is not None else p.y
    if panel.key == "third":
        return p.x3 if p.x3 is not None else p.x, p.y3 if p.y3 is not None else p.y
    return p.x, p.y


def layout_of(decision: RenderDecision, t: float) -> str:
    point = sample_track(decision.track, t)
    if decision.layout == "split" and (point.layout_decision or {}).get("reason") == "unsafe_manual_split":
        return "fit"
    if decision.layout != "auto":
        return decision.layout
    return point.mode or "fill"


def panels_for(decision: RenderDecision, layout: str) -> list[Panel]:
    """The output rectangles for a layout, matching the canvas composition for that layout."""
    w, h = OUTPUT_WIDTH, OUTPUT_HEIGHT
    if layout == "split":
        seam = h // 2
        return [
            Panel("main", w, seam, 1.32, 1.38, True),
            Panel("right", w, h - seam, 1.32, 1.38, True),
        ]
    if layout == "grid3":
        seam = h // 2
        half = w // 2
        return [
            Panel("main", w, seam, 1.35, 1.4, True),
            Panel("right", half, h - seam, 1.15, 1.2, True),
            Panel("third", w - half, h - seam, 1.15, 1.2, True),
        ]
    if layout == "editorial":
        full_width_h = (w * decision.source_height) / decision.source_width
        # Even for the same yuv420p reason as `fit`; the main panel is the remainder, so it stays even too.
        context_h = max(2, min((int(full_width_h) // 2) * 2, (int(h * 0.45) // 2) * 2))
        return [
            Panel("main", w, h - context_h, 1.0, 1.0, False),
            Panel("context", w, context_h, 1.0, 1.0, False, "full"),
        ]
    if layout == "fit":
        available_h = FIT_CAPTION_FOREGROUND_HEIGHT if decision.fit_caption_space else h
        scale = min(w / decision.source_width, available_h / decision.source_height)
        # Contain the whole source, rounding only the output size for yuv420p.
        fit_w = max(2, min(w, round(decision.source_width * scale / 2) * 2))
        fit_h = max(2, min(available_h, round(decision.source_height * scale / 2) * 2))
        return [Panel("main", fit_w, fit_h, 1.0, 1.0, False, "full")]
    return [Panel("main", w, h, 1.0, 1.0, False)]


def frame_times(decision: RenderDecision, fps: float) -> list[float]:
    n = max(1, int(round(decision.duration * fps)))
    return [i / fps for i in range(n)]


def _measured_neighbour(decision: RenderDecision, t: float, key: str) -> bool:
    """A panel may only be aimed at a subject the tracker actually measured.

    Mirrors the rule in `src/lib/render.ts`: an absent `x2` used to be filled by mirroring the one
    person of a single-subject shot into the second panel, so the same face appeared twice.
    """
    p = sample_track(decision.track, t)
    other = getattr(p, key)
    return other is not None and abs(other - p.x) >= 0.04


def effective_layout(decision: RenderDecision, layout: str, start: float, end: float, fps: float) -> str:
    """Downgrade a multi-panel layout to what the measurement supports over this window."""
    if layout not in ("split", "grid3"):
        return layout
    span = max(1, int(round((end - start) * fps)))
    votes = max(1, min(span, 12))
    times = [start + (end - start) * (i + 0.5) / votes for i in range(votes)]
    second = sum(1 for t in times if _measured_neighbour(decision, t, "x2")) >= votes // 2
    third = sum(1 for t in times if _measured_neighbour(decision, t, "x3")) >= votes // 2
    if layout == "split":
        return "split" if second else "fill"
    if second and third:
        return "grid3"
    return "split" if (second or third) else "fill"


def subject_panels(decision: RenderDecision, layout: str) -> list[Panel]:
    """The panels that need their own sendcmd-driven crop (a `full` backdrop does not)."""
    return [p for p in panels_for(decision, layout) if p.crop_mode != "full"]


def decoded_video_input_count(decision: RenderDecision, layout: str) -> int:
    """How many times the source must be opened for one layout.

    Each subject panel gets its own decoded input because sendcmd addresses filters by target name.
    `fit` splits a single decode; `editorial` adds a plain decode for its context strip.
    """
    base = len(subject_panels(decision, layout))
    if layout == "fit":
        return 1
    return base + (1 if layout == "editorial" else 0)


def crop_keyframes(decision: RenderDecision, panel: Panel, fps: float, punch_gain: float) -> list[CropFrame]:
    """The crop rectangle for every output frame, reduced only where the picture would not change.

    The old path sampled per frame and then thinned the list to a bounded-error approximation
    (1.5 source px of drift, 0.5 s between commands). FFmpeg holds a `crop` command until the next
    one, so an approximation is not a cheaper copy of the motion, it is a different motion: the
    framing sat still and then teleported, which is the stutter the preview never showed. Now a
    command is written whenever the rectangle FFmpeg would apply actually differs from the previous
    one, so the path is as dense as the pixels and not one line denser.
    """
    if panel.crop_mode == "full":
        return [CropFrame(0.0, decision.source_width, decision.source_height, 0.0, 0.0)]
    frames: list[CropFrame] = []
    last: tuple[int, int, int, int] | None = None
    for rel in frame_times(decision, fps):
        abs_t = decision.start + rel
        p = sample_track(decision.track, abs_t)
        zoom = (
            max(p.zoom, word_punch_zoom(decision.words, abs_t, punch_gain))
            if decision.zoom_punch
            else 1.0
        )
        zoom = panel_zoom(panel, zoom)
        fx, fy = _anchor_for(panel, p, decision)
        cw, ch, sx, sy = crop_rect(
            fx,
            fy,
            zoom,
            decision.source_width,
            decision.source_height,
            panel.dest_w,
            panel.dest_h,
            panel.exclude_bottom,
            panel.crop_mode,
        )
        rect = quantize_rect(cw, ch, sx, sy, decision.source_width, decision.source_height)
        if rect == last and frames:
            continue
        last = rect
        frames.append(CropFrame(rel, cw, ch, sx, sy))
    return frames


def quantize_rect(
    w: float, h: float, x: float, y: float, source_w: int, source_h: int
) -> tuple[int, int, int, int]:
    """The rectangle FFmpeg will really use: even-edged for yuv420p and inside the source."""
    qw = max(2, min(int(round(w)) // 2 * 2, source_w))
    qh = max(2, min(int(round(h)) // 2 * 2, source_h))
    qx = max(0, min(int(round(x)), source_w - qw))
    qy = max(0, min(int(round(y)), source_h - qh))
    return qw, qh, qx, qy


def sendcmd_file_text(frames: list[CropFrame], source_w: int, source_h: int) -> str:
    """Verified syntax for ffmpeg 9: one timed command per parameter, `;` between them.

    Every command carries its own time spec; `[ ... ]` grouping and un-timed continuation commands
    are both rejected by the parser, which was established against the installed binary.
    """
    lines: list[str] = []
    for f in frames:
        w, h, x, y = quantize_rect(f.w, f.h, f.x, f.y, source_w, source_h)
        t = f"{f.t:.4f}"
        lines.append(f"{t} crop w {w}; {t} crop h {h}; {t} crop x {x}; {t} crop y {y};")
    return "\n".join(lines) + "\n"


def _first_rect(frame: CropFrame, source_w: int, source_h: int) -> tuple[int, int, int, int]:
    """The crop's initial geometry, so the first frames are already framed before t=0 lands."""
    return quantize_rect(frame.w, frame.h, frame.x, frame.y, source_w, source_h)


def panel_crop_chain(
    panel: Panel,
    command_file: str,
    keyframes: list[CropFrame],
    decision: RenderDecision,
    fps: float,
) -> str:
    """The graph for one panel's own FFmpeg pass.

    A panel gets its own process because `sendcmd` matches commands by filter *name*: two `crop`
    filters in one graph cannot be given two different keyframe paths, so each crop-driven panel gets
    exactly one crop filter and one command file.

    `fps` runs before `sendcmd`, not after `crop`. Commands are timed against the frames that pass
    through, so resampling first is what makes command time n/fps land on output frame n; putting it
    after the crop addresses a 23.976 source on a 29.97 grid, and the crop is told to move at moments
    where no frame exists to receive the order.
    """
    if panel.crop_mode == "full":
        return (
            f"[0:v]fps={rate_arg(fps)},"
            f"scale={panel.dest_w}:{panel.dest_h}:flags=lanczos,setsar=1:1,format=yuv420p[v]"
        )
    w0, h0, x0, y0 = _first_rect(keyframes[0], decision.source_width, decision.source_height)
    return (
        f"[0:v]fps={rate_arg(fps)},sendcmd=f={command_file},"
        f"crop=w={w0}:h={h0}:x={x0}:y={y0},"
        f"scale={panel.dest_w}:{panel.dest_h}:flags=lanczos,setsar=1:1,format=yuv420p[v]"
    )


def build_composite_graph(
    decision: RenderDecision,
    layout: str,
    panel_count: int,
    fps: float,
    source_index: int,
    vignette_index: int | None = None,
    band: tuple[int, int] | None = None,
    edges: tuple[bool, bool] = (True, True),
    mono: bool = True,
) -> tuple[str, str]:
    """Compose, grade, and return (filter_complex, label).

    `fit` splits `source_index` into a blurred bed and full-source foreground, without panel inputs.
    Other layouts stack rendered panels; `editorial` uses `source_index` for its context strip.
    """
    label = "comp"
    if layout == "grid3" and panel_count >= 3:
        chains = ["[1:v][2:v]hstack[lower]", f"[0:v][lower]vstack[{label}]"]
    elif layout == "editorial":
        context = next(p for p in panels_for(decision, layout) if p.crop_mode == "full")
        chains = [
            f"[{source_index}:v]fps={rate_arg(fps)},scale={context.dest_w}:{context.dest_h},setsar=1:1,format=yuv420p[ctx]",
            f"[0:v][ctx]vstack[{label}]",
        ]
    elif layout == "fit":
        foreground = panels_for(decision, layout)[0]
        foreground_y = (
            f"'min((H-h)/2,{FIT_CAPTION_FOREGROUND_HEIGHT}-h)'"
            if decision.fit_caption_space else "(H-h)/2"
        )
        chains = [
            f"[{source_index}:v]fps={rate_arg(fps)},split=2[fit_bg][fit_fg]",
            f"[fit_bg]scale={OUTPUT_WIDTH}:2500:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
            f"gblur=sigma=28,eq=brightness=-0.05:saturation=1.08,setsar=1:1,format=yuv420p[bed]",
            f"[fit_fg]scale={foreground.dest_w}:{foreground.dest_h}:flags=lanczos,setsar=1:1,format=yuv420p[fg]",
            f"[bed][fg]overlay=(W-w)/2:{foreground_y}[{label}]",
        ]
    elif panel_count >= 2:
        chains = [f"[0:v][1:v]vstack[{label}]"]
    else:
        chains = [f"[0:v]null[{label}]"]
    chains += _grade_chain(label, vignette_index, band, decision=decision, edges=edges, mono=mono)
    return ";".join(chains), "[out]"


def build_video_graph(
    decision: RenderDecision,
    fps: float,
    command_files: list[str],
    keyframes: list[list[CropFrame]],
    layout: str | None = None,
    vignette_index: int | None = None,
    band: tuple[int, int] | None = None,
    edges: tuple[bool, bool] = (True, True),
    mono: bool = True,
) -> tuple[str, str]:
    """Single-pass graph for a one-panel layout: crop, scale to canvas, grade, vignette."""
    chosen = layout or layout_of(decision, decision.start)
    panel = subject_panels(decision, chosen)[0]
    w0, h0, x0, y0 = _first_rect(keyframes[0][0], decision.source_width, decision.source_height)
    chains = [
        f"[0:v]fps={rate_arg(fps)},sendcmd=f={command_files[0]},"
        f"crop=w={w0}:h={h0}:x={x0}:y={y0},"
        f"scale={panel.dest_w}:{panel.dest_h}:flags=lanczos,setsar=1:1[comp]"
    ]
    chains += _grade_chain("comp", vignette_index, band, decision=decision, edges=edges, mono=mono)
    return ";".join(chains), "[out]"


def autoclip_crop_expression(
    frames: list[CropFrame], source_w: int, source_h: int, max_keyframes: int = 85
) -> tuple[str, str, str, str]:
    """Compile CropFrame list into single-pass piecewise-linear FFmpeg crop expressions (from autoclip-main).

    Returns (w_expr, h_expr, x_expr, y_expr) suitable for `crop=w='...':h='...':x='...':y='...'`.
    """
    from ..openshorts_engine import build_single_pass_crop_expr

    if not frames:
        return str(source_w), str(source_h), "0", "0"
    quantized = [(f.t, *quantize_rect(f.w, f.h, f.x, f.y, source_w, source_h)) for f in frames]
    w_expr = build_single_pass_crop_expr([(t, w) for t, w, _, _, _ in quantized], max_keyframes)
    h_expr = build_single_pass_crop_expr([(t, h) for t, _, h, _, _ in quantized], max_keyframes)
    x_expr = build_single_pass_crop_expr([(t, x) for t, _, _, x, _ in quantized], max_keyframes)
    y_expr = build_single_pass_crop_expr([(t, y) for t, _, _, _, y in quantized], max_keyframes)
    return w_expr, h_expr, x_expr, y_expr
