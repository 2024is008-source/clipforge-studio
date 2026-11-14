"""Execute a RenderDecision with FFmpeg: no canvas, no MediaRecorder, no real-time playback.

A clip is split where the measured layout changes. A `fill` window renders in one FFmpeg pass; a
multi-panel window renders one pass per panel and a final composite pass, because `sendcmd` addresses
filters by name and two `crop` filters in one graph cannot follow two keyframe paths. Caption timing,
crop geometry, grade and audio all come from the same decision the preview draws.

Progress is FFmpeg's own `-progress pipe:1`, weighted by the pixels each pass produces, so the
percentage is frames actually encoded rather than a timer.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

from .audio import build_audio_graph, extra_audio_inputs
from .decision import OUTPUT_HEIGHT, OUTPUT_WIDTH, RenderDecision, Word
from .encoders import EncoderChoice, detect_encoder, encoder_video_args
from .framerate import rate_arg
from .filters import (
    CropFrame,
    Panel,
    build_composite_graph,
    build_video_graph,
    caption_anchor_y,
    caption_band,
    crop_keyframes,
    effective_layout,
    layout_of,
    panel_crop_chain,
    resolved_font_size,
    sendcmd_file_text,
    subject_panels,
    vignette_alpha_expression,
)
from .subtitles import HOOK_DURATION, paginate, write_ass

MIN_SEGMENT_SEC = 0.5
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
CONTAINER_ARGS = ["-movflags", "+faststart", "-shortest", "-max_interleave_delta", "0"]
# Visually transparent and cheap to decode, but not the delivery encoder: an intermediate only has to
# survive one more encode, so it trades file size for speed. B-frames are off because the composite
# pass decodes these files back and an out-of-order SEI was measured crashing that decode. Threads are
# capped because x264 reserves per-thread buffers and the failure this machine actually hits is the
# system commit limit (`x264 [error]: malloc of size 7186688 failed`), not the encoder.
INTERMEDIATE_ARGS = [
    "-c:v", "libx264", "-qp", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-bf", "0", "-threads", "4",
]
FONT_SUFFIXES = (".ttf", ".otf", ".woff2")
DEFAULT_MAX_RUNTIME_SEC = 3600.0


class RenderFailed(RuntimeError):
    """FFmpeg refused, crashed, or produced nothing usable."""


class RenderCancelled(RuntimeError):
    """The user cancelled; the caller keeps the job dir and the process is already dead."""


class EncoderUnavailable(RenderFailed):
    """The requested encoder could not be opened this time; a CPU retry can still deliver the file."""


@dataclass
class RenderRequest:
    decision: RenderDecision
    job_dir: str
    fps: float = 30.0
    punch_gain: float = 1.0
    encoder: EncoderChoice | None = None
    font_dir: str | None = None
    max_runtime_sec: float = DEFAULT_MAX_RUNTIME_SEC
    # Trusted completed sequence from overlays.resolve_overlay, never a browser path.
    overlay_dir: str | None = None


@dataclass
class Segment:
    layout: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Pass:
    """One FFmpeg invocation: a panel crop, a window composite, or the final concatenation."""

    args: list[str]
    duration: float
    weight: float
    label: str
    output: str
    verify_frames: bool = False
    expected_frames: int | None = None


@dataclass
class RenderResult:
    output_path: str
    duration: float
    fps: float
    encoder: str
    encoder_kind: str
    segments: list[Segment]
    passes: list[str]
    caption_events: int
    font_dir: str | None
    elapsed_sec: float
    source_has_audio: bool
    timings: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "output_path": self.output_path,
            "duration": round(self.duration, 3),
            "fps": self.fps,
            "encoder": self.encoder,
            "encoder_kind": self.encoder_kind,
            "segments": [
                {"layout": s.layout, "start": round(s.start, 3), "end": round(s.end, 3)} for s in self.segments
            ],
            "passes": self.passes,
            "caption_events": self.caption_events,
            "font_dir": self.font_dir,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "source_has_audio": self.source_has_audio,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


def render_job_dir(job_id: str) -> str:
    from ..config import JOBS_DIR

    path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(path, exist_ok=True, mode=0o700)
    return path


def ffprobe_json(path: str) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderFailed(f"ffprobe could not read {path}: {exc}") from exc
    if out.returncode != 0:
        raise RenderFailed(f"ffprobe rejected {path}: {out.stderr.decode('utf-8', 'replace').strip()}")
    return json.loads(out.stdout.decode("utf-8", "replace") or "{}")


def frame_count(path: str) -> int:
    """Frames the container says it holds; 0 when the file cannot be read at all."""
    try:
        info = ffprobe_json(path)
    except RenderFailed:
        return 0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            try:
                return int(stream.get("nb_frames") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def source_has_audio(path: str) -> bool:
    """Ask the container instead of assuming: a silent source must not fail QC for missing audio."""
    return any(s.get("codec_type") == "audio" for s in ffprobe_json(path).get("streams", []))


def layout_segments(decision: RenderDecision, fps: float) -> list[Segment]:
    """Preserve finalized layout runs; only smooth and remeasure legacy tracks."""
    frames = max(1, int(round(decision.duration * fps)))
    raw: list[Segment] = []
    for i in range(frames):
        rel = i / fps
        layout = layout_of(decision, decision.start + rel)
        if raw and raw[-1].layout == layout:
            raw[-1].end = (i + 1) / fps
        else:
            raw.append(Segment(layout, rel, (i + 1) / fps))

    if decision.track and all(isinstance(p.layout_decision, dict) for p in decision.track):
        return raw

    merged: list[Segment] = []
    for seg in raw:
        if merged and seg.duration < MIN_SEGMENT_SEC:
            merged[-1].end = seg.end
        else:
            merged.append(Segment(seg.layout, seg.start, seg.end))

    measured: list[Segment] = []
    for seg in merged:
        layout = effective_layout(
            decision, seg.layout, decision.start + seg.start, decision.start + seg.end, fps
        )
        if measured and measured[-1].layout == layout:
            measured[-1].end = seg.end
        else:
            measured.append(Segment(layout, seg.start, seg.end))
    return measured


def segment_decision(decision: RenderDecision, segment: Segment) -> RenderDecision:
    """The same decision inside one layout window; times stay absolute source seconds."""
    start = decision.start + segment.start
    end = decision.start + segment.end
    words = [
        Word(w.w, max(start, w.start), min(end, w.end))
        for w in decision.words
        if w.end > start and w.start < end
    ]
    return replace(decision, start=start, end=end, words=words,
                   hook=decision.hook if segment.start < HOOK_DURATION else "")


def caption_event_count(sub: RenderDecision) -> int:
    events = len(paginate(sub.words, sub.style.words_per_page, sub.start))
    if sub.style.show_hook and sub.hook.strip():
        events += 1
    return events


def _prepare_font_dir(job_dir: str, font_dir: str | None) -> str | None:
    """Copy caption fonts beside the render so the ASS path needs no Windows escaping."""
    if not font_dir or not os.path.isdir(font_dir):
        return None
    names = [n for n in os.listdir(font_dir) if n.lower().endswith(FONT_SUFFIXES)]
    if not names:
        return None
    target = os.path.join(job_dir, "fonts")
    os.makedirs(target, exist_ok=True)
    for name in names:
        dst = os.path.join(target, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(font_dir, name), dst)
    return "fonts"


def _write_captions(job_dir: str, sub: RenderDecision, index: int, font_rel: str | None,
                    layout: str, hook_duration: float) -> str:
    name = f"captions_s{index:03d}.ass"
    write_ass(sub, os.path.join(job_dir, name), caption_anchor_y(sub, layout), hook_duration=hook_duration)
    return f"ass=filename={name}" + (f":fontsdir={font_rel}" if font_rel else "")


def _write_keyframes(job_dir: str, name: str, frames: list[CropFrame], sub: RenderDecision) -> str:
    with open(os.path.join(job_dir, name), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(sendcmd_file_text(frames, sub.source_width, sub.source_height))
    return name


def _base_args() -> list[str]:
    return ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning", "-nostats", "-progress", "pipe:1"]


def _extra_audio_paths(sub: RenderDecision) -> list[str]:
    return [track.path for track in extra_audio_inputs(sub)]


def _ensure_vignette(job_dir: str) -> str:
    """Paint the canvas vignette once per job as an alpha frame the graph can overlay.

    `geq` costs a few hundred milliseconds here and nothing during the encode, which is the opposite
    of running it per frame. The PNG is one frame, and overlay repeats its last frame for the window.
    """
    path = os.path.join(job_dir, "vignette.png")
    if os.path.isfile(path):
        return path
    expr = vignette_alpha_expression(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=black@0.0:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-vf",
        f"format=rgba,geq=r=0:g=0:b=0:a='{expr}'",
        "-frames:v",
        "1",
        path,
    ]
    out = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    if out.returncode != 0 or not os.path.isfile(path):
        raise RenderFailed(f"could not build the vignette frame: {out.stderr.decode('utf-8', 'replace').strip()}")
    return path


def plan_passes(
    request: RenderRequest,
    segment: Segment,
    index: int,
    has_audio: bool,
    font_rel: str | None,
    vignette_path: str,
    edges: tuple[bool, bool] = (True, True),
) -> tuple[list[Pass], str, int]:
    """The FFmpeg invocations for one layout window, in execution order, plus its output file."""
    job_dir = request.job_dir
    fps = request.fps
    mono = request.punch_gain > 0
    sub = segment_decision(request.decision, segment)
    if request.overlay_dir is not None:
        # Keep words for punch timing; the canvas already painted all text.
        sub = replace(sub, hook="", brand_label=None, style=replace(sub.style, show_hook=False))
    layout = layout_of(sub, sub.start)
    panels: list[Panel] = subject_panels(sub, layout)
    canvas = OUTPUT_WIDTH * OUTPUT_HEIGHT
    band = None if request.overlay_dir is not None else caption_band(layout, resolved_font_size(sub.style))
    passes: list[Pass] = []

    events = 0 if request.overlay_dir is not None else caption_event_count(sub)
    ass = _write_captions(job_dir, sub, index, font_rel, layout,
                          HOOK_DURATION - segment.start) if events else ""
    window_out = os.path.join(job_dir, f"segment_{index:03d}.mp4")

    if layout == "fit" or (layout == "fill" and len(panels) == 1):
        audio_paths = _extra_audio_paths(sub)
        if layout == "fit":
            graph, label = build_composite_graph(sub, layout, 0, fps, 0, len(audio_paths) + 1, band, edges, mono)
        else:
            frames = crop_keyframes(sub, panels[0], fps, request.punch_gain)
            cmd = _write_keyframes(job_dir, f"crop_s{index:03d}_p0.cmd", frames, sub)
            graph, label = build_video_graph(sub, fps, [cmd], [frames], layout, len(audio_paths) + 1, band, edges, mono)
        video_graph = f"{graph};{label}{ass}[vout]" if ass else graph.replace("[out]", "[vout]")
        args = _base_args()
        args += ["-ss", f"{sub.start:.4f}", "-t", f"{sub.duration:.4f}", "-i", sub.source_path]
        for path in audio_paths:
            args += ["-i", path]
        args += ["-i", vignette_path]
        fragment_graph, tail = _final_tail(sub, has_audio, 0, 1, video_graph)
        args += ["-filter_complex", fragment_graph, *tail]
        args += encoder_video_args(request.encoder, rate_arg(request.fps))
        args += CONTAINER_ARGS + [window_out]
        passes.append(
            Pass(args, sub.duration, sub.duration, f"render {layout} window {index + 1}", window_out)
        )
        return passes, window_out, events

    panel_files: list[str] = []
    for i, panel in enumerate(panels):
        frames = crop_keyframes(sub, panel, fps, request.punch_gain)
        cmd = _write_keyframes(job_dir, f"crop_s{index:03d}_p{i}.cmd", frames, sub)
        graph = panel_crop_chain(panel, cmd, frames, sub, fps)
        out = os.path.join(job_dir, f"panel_s{index:03d}_p{i}.mp4")
        args = _base_args()
        args += ["-ss", f"{sub.start:.4f}", "-t", f"{sub.duration:.4f}", "-i", sub.source_path]
        args += ["-filter_complex", graph, "-map", "[v]", "-an"]
        args += INTERMEDIATE_ARGS + ["-r", rate_arg(fps), out]
        weight = sub.duration * (panel.dest_w * panel.dest_h) / canvas
        passes.append(
            Pass(args, sub.duration, weight, f"{panel.key} crop window {index + 1}", out, True)
        )
        panel_files.append(out)

    needs_source = layout in ("fit", "editorial")
    source_input = len(panel_files)
    audio_paths = _extra_audio_paths(sub)
    # One seeked decode serves the blurred bed / context strip, the original audio, and the extras.
    decodes_source = needs_source or has_audio or bool(audio_paths)
    first_extra_input = source_input + 1
    args = _base_args()
    next_input = 0
    for path in panel_files:
        args += ["-i", path]
        next_input += 1
    if decodes_source:
        args += ["-ss", f"{sub.start:.4f}", "-t", f"{sub.duration:.4f}", "-i", sub.source_path]
        next_input += 1
    for path in audio_paths:
        args += ["-i", path]
        next_input += 1
    args += ["-i", vignette_path]
    composite_graph, comp_label = build_composite_graph(
        sub, layout, len(panel_files), fps, source_input, next_input, band, edges, mono
    )
    video_graph = (
        f"{composite_graph};{comp_label}{ass}[vout]"
        if ass
        else composite_graph.replace("[out]", "[vout]")
    )
    fragment_graph, tail = _final_tail(sub, has_audio, source_input, first_extra_input, video_graph)
    args += ["-filter_complex", fragment_graph, *tail]
    args += encoder_video_args(request.encoder, rate_arg(request.fps))
    args += CONTAINER_ARGS + [window_out]
    passes.append(
        Pass(args, sub.duration, sub.duration, f"composite {layout} window {index + 1}", window_out)
    )
    return passes, window_out, events


def _final_tail(
    sub: RenderDecision, has_audio: bool, audio_index: int, first_extra_index: int, video_graph: str
) -> tuple[str, list[str]]:
    """Splice the audio graph into the pass's filter_complex and return its map/codec args."""
    if not has_audio:
        return video_graph, ["-map", "[vout]", "-an"]
    indices: dict[int, int] = {}
    slot = 0
    for position, track in enumerate(sub.audio):
        if track.kind == "original":
            continue
        indices[position] = first_extra_index + slot
        slot += 1
    fragment, outa = build_audio_graph(sub, audio_index, indices)
    return f"{video_graph};{fragment}", ["-map", "[vout]", "-map", outa, *AUDIO_ARGS]


def run_render(
    request: RenderRequest,
    on_progress=None,
    is_cancelled=None,
) -> RenderResult:
    """Render a whole decision, pinning uploaded assets until every pass has stopped."""
    if request.overlay_dir is None:
        return _run_render(request, on_progress, is_cancelled)
    # Lazy import keeps direct, non-overlay renderer usage independent of FastAPI.
    from .overlays import reference_overlay

    with reference_overlay(request.overlay_dir, request.fps, request.decision.duration):
        try:
            return _run_render(request, on_progress, is_cancelled)
        finally:
            # Preserve uploads for retry, but remove any partial delivery file.
            pending = os.path.join(request.job_dir, "out_canvas_overlay.mp4")
            if os.path.isfile(pending):
                os.remove(pending)


def _run_render(
    request: RenderRequest,
    on_progress=None,
    is_cancelled=None,
) -> RenderResult:
    """Plan and execute the same passes for native and canvas-caption exports."""
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled("Render cancelled by request.")
    if not shutil.which("ffmpeg"):
        raise RenderFailed("FFmpeg is not installed or not on PATH")
    decision = request.decision
    if not os.path.isfile(decision.source_path):
        raise RenderFailed(f"Source not found: {decision.source_path}")
    if decision.duration <= 0:
        raise RenderFailed("Render decision has no positive duration to execute")

    encoder = request.encoder or detect_encoder()
    if encoder is None:
        raise RenderFailed("No working video encoder was measured on this machine")
    request = replace(request, encoder=encoder)

    job_dir = request.job_dir
    os.makedirs(job_dir, exist_ok=True)
    has_audio = source_has_audio(decision.source_path)
    font_rel = _prepare_font_dir(job_dir, request.font_dir) if request.overlay_dir is None else None
    vignette_path = _ensure_vignette(job_dir)
    segments = layout_segments(decision, request.fps)
    # Per-window audio restarts loudnorm, the compressor and any fades at every seam, which dips the
    # level on each layout cut; with seams, windows go out silent and the concat mixes audio once.
    window_audio = has_audio and len(segments) == 1

    passes: list[Pass] = []
    window_outputs: list[str] = []
    caption_events = 0
    for index, segment in enumerate(segments):
        segment_passes, window_out, events = plan_passes(
            request, segment, index, window_audio, font_rel, vignette_path,
            (index > 0, index < len(segments) - 1),
        )
        passes += segment_passes
        window_outputs.append(window_out)
        caption_events += events

    if len(window_outputs) > 1:
        passes.append(_concat_pass(request, window_outputs, has_audio))
    if request.overlay_dir is not None:
        passes.append(_overlay_pass(request, passes[-1].output, has_audio))
    total_weight = sum(p.weight for p in passes) or 1.0

    started = time.time()
    timings: dict[str, float] = {}

    def reporter(done_weight: float, p: Pass):
        def report(fraction: float) -> None:
            if on_progress is None:
                return
            overall = (done_weight + fraction * p.weight) / total_weight
            on_progress(max(0.0, min(1.0, overall)), p.label)

        return report

    done_weight = 0.0
    for p in passes:
        began = time.time()
        _run(p.args, job_dir, p.duration, reporter(done_weight, p), is_cancelled, started, request.max_runtime_sec)
        if not os.path.isfile(p.output):
            raise RenderFailed(f"FFmpeg produced no file for {p.label}")
        if p.expected_frames is not None:
            got = frame_count(p.output)
            if got != p.expected_frames:
                raise RenderFailed(f"{p.label} produced {got} frames, expected exactly {p.expected_frames}")
        if p.verify_frames:
            want = int(round(p.duration * request.fps))
            got = frame_count(p.output)
            # One frame either way is the last partial frame being dropped; anything larger means the
            # intermediate is truncated, which would decode as garbage one pass later.
            if abs(got - want) > 1:
                # A half-written intermediate decodes as garbage one pass later, where the error no
                # longer names the file that caused it, so it is caught and re-encoded here.
                _run(
                    p.args, job_dir, p.duration, reporter(done_weight, p), is_cancelled, started,
                    request.max_runtime_sec,
                )
                got = frame_count(p.output)
                if abs(got - want) > 1:
                    raise RenderFailed(
                        f"{p.label} produced {got} frames, expected {want}; the intermediate is unusable"
                    )
        timings[p.label] = time.time() - began
        done_weight += p.weight

    final = passes[-1].output
    output_path = os.path.join(job_dir, "out_backend.mp4")
    if os.path.abspath(final) != os.path.abspath(output_path):
        os.replace(final, output_path)

    return RenderResult(
        output_path=output_path,
        duration=decision.duration,
        fps=request.fps,
        encoder=encoder.name,
        encoder_kind=encoder.kind,
        segments=segments,
        passes=[p.label for p in passes],
        caption_events=caption_events,
        font_dir=os.path.join(job_dir, font_rel) if font_rel else None,
        elapsed_sec=time.time() - started,
        source_has_audio=has_audio,
        timings=timings,
    )


def _overlay_pass(request: RenderRequest, assembled: str, has_audio: bool) -> Pass:
    """Burn full-timeline canvas PNGs once, after all native layout windows are joined."""
    from .overlays import overlay_frame_count

    count = overlay_frame_count(request.fps, request.decision.duration)
    fps = rate_arg(request.fps)
    final = os.path.join(request.job_dir, "out_canvas_overlay.mp4")
    if request.overlay_dir is None or os.path.abspath(assembled) == os.path.abspath(final):
        raise RenderFailed("Overlay pass requires a sequence and a distinct output path")
    args = _base_args()
    args += ["-i", assembled, "-f", "image2", "-framerate", fps, "-start_number", "0",
             "-i", os.path.join(request.overlay_dir, "%06d.png")]
    # Pad the native rounded frame count to the canvas's full-timeline ceil count.
    graph = (
        f"[0:v]fps={fps},tpad=stop_mode=clone:stop_duration={1 / request.fps:.12f},"
        f"trim=end_frame={count},setpts=PTS-STARTPTS[base];"
        "[1:v]format=rgba,setpts=PTS-STARTPTS[canvas];"
        "[base][canvas]overlay=0:0:eof_action=endall:shortest=1:format=auto:alpha=straight,"
        "format=yuv420p[vout]"
    )
    args += ["-filter_complex", graph, "-map", "[vout]"]
    # Preserve the already-normalized AAC without another lossy encode.
    args += ["-map", "0:a:0", "-c:a", "copy"] if has_audio else ["-an"]
    args += encoder_video_args(request.encoder, fps)
    args += ["-frames:v", str(count), "-t", f"{count / request.fps:.12f}"]
    # Do not truncate the PNG sequence to a slightly shorter AAC stream.
    args += [arg for arg in CONTAINER_ARGS if arg != "-shortest"] + [final]
    return Pass(args, request.decision.duration, request.decision.duration,
                "composite canvas overlays", final, expected_frames=count)


def _concat_pass(request: RenderRequest, outputs: list[str], has_audio: bool) -> Pass:
    """Join silent layout windows with one re-encode and mix the whole clip's audio in one chain."""
    decision = request.decision
    job_dir = request.job_dir
    final = os.path.join(job_dir, "out_assembled.mp4" if request.overlay_dir is not None else "out_backend.mp4")
    args = _base_args()
    for path in outputs:
        args += ["-i", path]
    count = len(outputs)
    video_graph = "".join(f"[{i}:v]" for i in range(count)) + f"concat=n={count}:v=1:a=0[vout]"
    if has_audio:
        args += ["-ss", f"{decision.start:.4f}", "-t", f"{decision.duration:.4f}", "-i", decision.source_path]
        for path in _extra_audio_paths(decision):
            args += ["-i", path]
    graph, tail = _final_tail(decision, has_audio, count, count + 1, video_graph)
    args += ["-filter_complex", graph, *tail]
    args += encoder_video_args(request.encoder, rate_arg(request.fps))
    args += CONTAINER_ARGS + [final]
    return Pass(args, decision.duration, decision.duration, "concatenate layout windows", final)


SOFTWARE_ENCODERS = ("libx264", "libx264rgb", "libx265")
# Only lines that talk about the accelerator count, so a filtergraph typo is never mistaken for a
# driver problem and retried pointlessly on CPU.
_ACCELERATOR_TOKENS = (
    "nvenc", "cuda", "mfx", "qsv", "amf", "videotoolbox", "d3d11", "vaapi", "vulkan",
    "device", "session", "opening encoder",
)
_SESSION_FAILURES = (
    "out of memory", "busy", "already_mapped", "no encode device", "no cuda-capable device",
    "cuda_error", "could not open cuda", "couldn't open cuda", "cannot load", "not supported",
    "unsupported", "failed to create", "create failed", "failed to initialize", "error initializing",
    "no device", "device creation failed", "error while opening encoder", "cannot open", "not available",
)


def _hardware_session_failed(args: list[str], lines: list[str]) -> bool:
    """True when a hardware encoder refused to open, which a software retry can still deliver."""
    if "-c:v" not in args:
        return False
    encoder = args[args.index("-c:v") + 1]
    if encoder in SOFTWARE_ENCODERS:
        return False
    for line in lines:
        lowered = line.lower()
        if encoder not in lowered and not any(token in lowered for token in _ACCELERATOR_TOKENS):
            continue
        if any(marker in lowered for marker in _SESSION_FAILURES):
            return True
    return False


def _run(
    args: list[str],
    cwd: str,
    duration: float,
    on_fraction,
    is_cancelled,
    started_wall: float,
    max_runtime_sec: float,
) -> None:
    """Stream FFmpeg's counters while cancellation remains responsive during silent passes."""
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled("Render cancelled by request.")
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise RenderFailed(f"Could not start FFmpeg: {exc}") from exc

    def stop(kill: bool) -> None:
        if proc.poll() is None:
            (proc.kill if kill else proc.terminate)()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    progress_lines: queue.Queue[str | None] = queue.Queue()
    errors: deque[str] = deque(maxlen=200)

    def read_progress() -> None:
        try:
            for line in proc.stdout:
                progress_lines.put(line)
        finally:
            progress_lines.put(None)

    def read_errors() -> None:
        for line in proc.stderr:
            errors.append(line)

    readers = [threading.Thread(target=read_progress), threading.Thread(target=read_errors)]
    for reader in readers:
        reader.start()
    stdout_done = False
    try:
        while True:
            if time.time() - started_wall > max_runtime_sec:
                stop(True)
                raise RenderFailed(f"Render exceeded its {max_runtime_sec:.0f}s budget and was killed.")
            if is_cancelled is not None and is_cancelled():
                stop(False)
                raise RenderCancelled("Render cancelled by request.")
            if stdout_done:
                try:
                    proc.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            try:
                line = progress_lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                stdout_done = True
                continue
            line = line.strip()
            if line.startswith(("out_time_us=", "out_time_ms=")) and duration > 0:
                key, _, value = line.partition("=")
                try:
                    encoded = int(value) / 1_000_000.0
                except ValueError:
                    continue
                # FFmpeg's out_time_ms is normally expressed in microseconds.
                if key == "out_time_ms" and encoded > duration * 100:
                    encoded /= 1000.0
                on_fraction(max(0.0, min(1.0, encoded / duration)))
    finally:
        stop(True)
        for reader in readers:
            reader.join()
        proc.stdout.close()
        proc.stderr.close()
    stderr_text = "".join(errors)
    code = proc.returncode
    if code != 0:
        lines = [line for line in stderr_text.strip().splitlines() if line.strip()]
        # The last lines are the generic "encoder could not be opened" wrap-up; the reason is usually
        # the first line that mentions one, so both ends of the log are reported.
        markers = ("error", "invalid", "no such", "cannot", "failed", "not found")
        head = [line for line in lines if any(m in line.lower() for m in markers)][:4]
        tail = " | ".join([*head, *lines[-4:]])
        failure = f"FFmpeg exited {code}: {tail}"
        lowered = stderr_text.lower()
        if _hardware_session_failed(args, lines):
            # A browser tab encoding at the same time can take the last NVENC session, an Intel iGPU
            # can be handed to another app, and an AMD driver can refuse a second AMF context. All of
            # them are retryable on the software encoder rather than a broken filtergraph.
            raise EncoderUnavailable(failure + " (the hardware encoder could not open a session)")
        if "no space left" in lowered or "errno 112" in lowered or "too small" in lowered:
            raise RenderFailed(
                f"{failure}\nThe disk that holds the render folder is full. Free space on it, or "
                "point the worker at another drive with CLIPFORGE_WORK_DIR, and retry the export."
            )
        if "malloc of size" in lowered or "cannot allocate memory" in lowered:
            raise RenderFailed(
                f"{failure}\nThe machine ran out of memory mid-pass. Close other memory-heavy apps "
                "(including the browser's live preview) and retry the export."
            )
        raise RenderFailed(failure)
    on_fraction(1.0)


render_clip = run_render
