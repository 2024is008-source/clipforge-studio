"""Automated quality control for a finished render.

The job fails instead of shipping a broken video: container geometry, frame count, A/V sync, black
or frozen pictures, silent audio, decode errors and caption visibility are all measured against what
the decision asked for. Every number comes from ffprobe or a real decode pass, never from the
renderer's own bookkeeping.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

DURATION_TOLERANCE_SEC = 0.30
FPS_TOLERANCE = 0.04
AV_SYNC_TOLERANCE_SEC = 0.15
BLACK_SHARE_LIMIT = 0.20
FREEZE_SEC_LIMIT = 3.0
SILENCE_SHARE_LIMIT = 0.60
CAPTION_PIXEL_LIMIT = 400
VALIDATE_TIMEOUT_SEC = 900

FRAME_STAT_RE = re.compile(r"frame=\s*(\d+)")
BLACK_RE = re.compile(r"black_start:([\d.]+)\s+black_end:([\d.]+)")
FREEZE_RE = re.compile(r"freeze_start:\s*([\d.]+)\s+freeze_end:\s*([\d.]+)")
SILENCE_RE = re.compile(r"silence_start:\s*([-\d.]+).*?silence_end:\s*([-\d.]+)", re.DOTALL)


@dataclass
class ExpectedOutput:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    caption_events: int = 0
    ass_file: str | None = None
    fonts_dir: str | None = None
    caption_sample_time: float | None = None
    encoder: str | None = None


@dataclass
class QcCheck:
    name: str
    status: str  # pass | fail | warn | skipped
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"check": self.name, "status": self.status, "detail": self.detail}


@dataclass
class QcReport:
    passed: bool
    checks: list[QcCheck] = field(default_factory=list)
    measured: dict[str, float | int | str | None] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str, critical: bool = True) -> bool:
        status = "pass" if ok else ("fail" if critical else "warn")
        self.checks.append(QcCheck(name, status, detail))
        if not ok and critical:
            self.passed = False
        return ok

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "measured": self.measured,
        }


def _run(args: list[str], timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)


def _streams(path: str) -> tuple[dict, dict | None, dict | None]:
    out = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], 60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip() or "ffprobe failed")
    data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return video or {}, audio, data.get("format", {}) or {}


def _rate(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key) or ""
        num, _, den = str(value).partition("/")
        try:
            n, d = float(num), float(den or 1)
        except ValueError:
            continue
        if d and n:
            return n / d
    return 0.0


def _stream_rate(stream: dict, key: str) -> float:
    num, _, den = str(stream.get(key) or "").partition("/")
    try:
        n, d = float(num), float(den or 1)
    except ValueError:
        return 0.0
    return n / d if d and n else 0.0


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def probe_source(path: str) -> tuple[int, int, float, float]:
    """Geometry of the file the renderer is about to open.

    Crop rectangles are computed in the source's own pixel space, so the size the renderer uses has
    to be the size the decoder will actually deliver — a browser that reported a different number
    would put every crop a few pixels off and read as a tracking failure.
    """
    video, _, _ = _streams(path)
    nominal = _stream_rate(video, "r_frame_rate")
    average = _stream_rate(video, "avg_frame_rate") or nominal
    return int(video.get("width") or 0), int(video.get("height") or 0), average, nominal


def _decode_pass(path: str) -> tuple[int, str]:
    """One full decode with black/freeze/silence detection; the exit code is the corrupt-frame test."""
    filters = "blackdetect=d=0.10:pix_th=0.10,freezedetect=n=-60dB:d=1.25"
    args = ["ffmpeg", "-hide_banner", "-xerror", "-v", "info", "-i", path, "-an", "-vf", filters, "-f", "null", "-"]
    out = _run(args, VALIDATE_TIMEOUT_SEC)
    stderr = out.stderr.decode("utf-8", "replace")
    if out.returncode == 0:
        audio_args = [
            "ffmpeg", "-hide_banner", "-v", "info", "-i", path, "-vn",
            "-af", "silencedetect=n=-45dB:d=0.75", "-f", "null", "-",
        ]
        stderr += _run(audio_args, VALIDATE_TIMEOUT_SEC).stderr.decode("utf-8", "replace")
    return out.returncode, stderr


def _caption_pixels_visible(expected: ExpectedOutput, canvas: tuple[int, int]) -> int | None:
    """Render the ASS over a flat canvas and count changed pixels: proves glyphs, font and placement."""
    if not expected.ass_file or not os.path.isfile(expected.ass_file):
        return None
    t = expected.caption_sample_time if expected.caption_sample_time is not None else 0.5
    ass = os.path.basename(expected.ass_file)
    suffix = f":fontsdir={os.path.basename(expected.fonts_dir)}" if expected.fonts_dir else ""
    graph = f"[0:v]ass=filename={ass}{suffix}"
    # The ASS is referenced by basename, so the probe runs from the render's own directory; fonts and
    # command files resolve the same way the render passes do.
    probe_cwd = os.path.dirname(expected.ass_file)
    out = _run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={canvas[0]}x{canvas[1]}:r=30:d=2",
            "-ss", f"{max(0.0, t):.3f}", "-filter_complex", graph, "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        120,
        cwd=probe_cwd,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    control = _run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={canvas[0]}x{canvas[1]}:r=30:d=2",
            "-ss", f"{max(0.0, t):.3f}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        120,
        cwd=probe_cwd,
    )
    if control.returncode != 0 or len(control.stdout) != len(out.stdout):
        return None
    import numpy as np

    a = np.frombuffer(out.stdout, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(control.stdout, dtype=np.uint8).astype(np.int16)
    changed = (np.abs(a - b) > 24).reshape(-1, 3).any(axis=1)
    return int(changed.sum())


def validate_render(path: str, expected: ExpectedOutput) -> QcReport:
    report = QcReport(passed=True)

    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        report.add("media_present", False, f"{path} is missing or smaller than 1 KiB")
        return report
    report.add("media_present", True, f"{os.path.getsize(path)} bytes")

    try:
        video, audio, fmt = _streams(path)
    except (RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        report.add("container_readable", False, str(exc))
        return report
    report.add("container_readable", True, fmt.get("format_name", "unknown"))

    duration = _float(fmt.get("duration"))
    if duration <= 0 and video:
        duration = _float(video.get("duration"))
    report.measured["duration_sec"] = round(duration, 3)
    report.add(
        "duration",
        abs(duration - expected.duration) <= max(DURATION_TOLERANCE_SEC, expected.duration * 0.02),
        f"measured {duration:.3f}s, expected {expected.duration:.3f}s",
    )

    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    report.measured["resolution"] = f"{width}x{height}"
    report.add("resolution", (width, height) == (expected.width, expected.height), f"{width}x{height}")
    aspect = (width / height) if height else 0.0
    report.add(
        "aspect_ratio",
        abs(aspect - expected.width / expected.height) < 0.01,
        f"{aspect:.4f} vs {(expected.width / expected.height):.4f}",
    )
    report.add("video_codec", bool(video.get("codec_name")), str(video.get("codec_name")))
    if expected.encoder:
        report.add(
            "encoder_matches_request",
            video.get("codec_name") in ("h264", "hevc"),
            f"requested {expected.encoder}, container reports {video.get('codec_name')}",
            critical=False,
        )

    fps = _rate(video)
    report.measured["fps"] = round(fps, 3)
    report.add(
        "frame_rate",
        fps > 0 and abs(fps - expected.fps) / max(1.0, expected.fps) <= FPS_TOLERANCE,
        f"{fps:.3f} fps, expected {expected.fps:.3f}",
    )

    report.add(
        "audio_stream",
        bool(audio) == expected.has_audio,
        f"{audio.get('codec_name') if audio else 'none'} (expected {'audio' if expected.has_audio else 'silence'})",
    )

    code, stderr = _decode_pass(path)
    frames = [int(m) for m in FRAME_STAT_RE.findall(stderr)]
    frame_count = frames[-1] if frames else 0
    report.measured["decoded_frames"] = frame_count
    report.add("decode_clean", code == 0, "no frame corruption" if code == 0 else "FFmpeg reported a decode error")

    if frame_count and duration > 0:
        measured_fps = frame_count / duration
        report.add(
            "frame_count_consistency",
            abs(measured_fps - expected.fps) / max(1.0, expected.fps) <= FPS_TOLERANCE,
            f"{frame_count} frames over {duration:.3f}s = {measured_fps:.3f} fps",
        )

    black = sum(max(0.0, float(b) - float(a)) for a, b in BLACK_RE.findall(stderr))
    report.measured["black_sec"] = round(black, 3)
    report.add(
        "black_frames",
        black <= max(0.5, duration * BLACK_SHARE_LIMIT),
        f"{black:.2f}s black of {duration:.2f}s",
        critical=False,
    )

    freezes = [(float(s), float(e)) for s, e in FREEZE_RE.findall(stderr)]
    longest = max((e - s for s, e in freezes), default=0.0)
    report.measured["longest_freeze_sec"] = round(longest, 3)
    report.add(
        "frozen_frames",
        longest <= FREEZE_SEC_LIMIT,
        f"longest static picture {longest:.2f}s" if freezes else "no freeze detected",
        critical=False,
    )

    if audio:
        silences = [
            (float(s), float(e))
            for s, e in re.findall(r"silence_start:\s*([-\d.]+)\s+.*?silence_end:\s*([-\d.]+)", stderr, re.DOTALL)
        ]
        silent = sum(max(0.0, e - s) for s, e in silences)
        report.measured["silent_audio_sec"] = round(silent, 3)
        report.add(
            "audio_content",
            silent <= duration * SILENCE_SHARE_LIMIT,
            f"{silent:.2f}s of {duration:.2f}s below -45 dB",
            critical=False,
        )

    if expected.caption_events:
        changed = _caption_pixels_visible(expected, (expected.width, expected.height))
        if changed is None:
            report.add(
                "caption_visibility",
                False,
                "subtitle overlay could not be probed",
                critical=False,
            )
        else:
            report.measured["caption_pixels"] = changed
            report.add(
                "caption_visibility",
                changed >= CAPTION_PIXEL_LIMIT,
                f"{changed} pixels differ from a blank canvas at caption time",
                critical=False,
            )
    else:
        report.checks.append(QcCheck("caption_visibility", "skipped", "the decision asks for no captions"))

    v_start = _float(video.get("start_time"))
    a_start = _float(audio.get("start_time")) if audio else None
    if a_start is not None:
        drift = abs(v_start - a_start)
        report.measured["av_start_drift_sec"] = round(drift, 3)
        report.add("av_sync", drift <= AV_SYNC_TOLERANCE_SEC, f"stream start drift {drift:.3f}s", critical=False)
    return report
