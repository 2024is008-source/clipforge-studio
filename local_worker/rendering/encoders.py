"""Encoder selection by measured capability, not by what the encoder list claims.

An encoder appearing in `ffmpeg -encoders` only proves the binary was built with it. NVENC still
needs a matching driver, QSV needs a supported Intel iGPU, AMF needs an AMD one, so each candidate
has to survive a real micro-encode at the delivery resolution before it is trusted with an export.

Every candidate is held to one rate control (see `BITRATE_ARGS`), so a laptop on integrated graphics
or plain CPU cores delivers the same picture as a desktop on NVENC — only the time it takes differs.
Candidates are tried best-first and each one carries progressively simpler variants, because an older
driver may reject a knob the current one accepts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

NULL_OUTPUT = ["-f", "null", "-"]

# The delivery frame. Probing at thumbnail size would pass on GPUs that then refuse 1080x1920.
DELIVERY_WIDTH = 1080
DELIVERY_HEIGHT = 1920
PROBE_TIMEOUT_SEC = 45
# A corruption floor, not a quality target: every healthy profile measured above 0.999 on this
# pattern, so anything under 0.90 is a broken driver rather than a lossy-but-valid encode.
MIN_PROBE_SSIM = 0.90

# One rate control for every encoder, hardware or software. Measured on a real 1080x1920 ClipForge
# master: x264 at this target scored VMAF 97.91 against the same input where Intel QSV scored 97.34,
# while letting x264 pick its own bitrate from a CRF dropped it to 95.56. Sharing the target is what
# keeps a CPU-only laptop's output equal to a desktop's.
BITRATE_ARGS = ["-b:v", "16M", "-maxrate", "18M", "-bufsize", "32M"]

ENV_ENCODER = "CLIPFORGE_VIDEO_ENCODER"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    vendor: str
    kind: str
    integrated: bool
    variants: tuple[tuple[str, ...], ...]


# H.264 High profile everywhere: HEVC would change the delivery codec, and every GPU that can encode
# HEVC can also encode H.264, so a second HEVC entry buys nothing and costs player compatibility.
# VAAPI is deliberately absent — it needs an hwupload filtergraph the renderer does not build, and
# Linux machines without NVENC get the quality-matched software path instead.
SPECS: tuple[EncoderSpec, ...] = (
    EncoderSpec(
        "h264_nvenc", "NVIDIA", "gpu", False,
        (
            ("-preset", "p5", "-profile:v", "high"),
            ("-preset", "slow", "-profile:v", "high"),
            ("-profile:v", "high"),
        ),
    ),
    EncoderSpec(
        "h264_qsv", "Intel", "gpu", True,
        (
            ("-preset", "slower", "-profile:v", "high"),
            ("-preset", "medium", "-profile:v", "high"),
            ("-profile:v", "high"),
        ),
    ),
    EncoderSpec(
        "h264_amf", "AMD", "gpu", True,
        (
            ("-quality", "high_quality", "-rc", "vbr_peak", "-profile:v", "high"),
            ("-quality", "quality", "-rc", "vbr_peak", "-profile:v", "high"),
            ("-quality", "balanced", "-rc", "vbr_peak", "-profile:v", "high"),
            ("-rc", "vbr_peak", "-profile:v", "high"),
        ),
    ),
    EncoderSpec(
        "h264_videotoolbox", "Apple", "gpu", True,
        (
            ("-profile:v", "high", "-realtime", "0"),
            ("-profile:v", "high"),
        ),
    ),
    EncoderSpec(
        "libx264", "Software", "cpu", False,
        (
            ("-preset", "medium", "-profile:v", "high"),
            ("-preset", "fast", "-profile:v", "high"),
            ("-profile:v", "high"),
        ),
    ),
)


@dataclass
class EncoderChoice:
    name: str
    kind: str
    quality_args: list[str]
    note: str = ""
    vendor: str = "Software"
    integrated: bool = False

    @property
    def label(self) -> str:
        if self.kind == "cpu":
            return "CPU (x264)"
        return f"{self.vendor} {'integrated' if self.integrated else 'GPU'} ({self.name})"

    def video_args(self) -> list[str]:
        return ["-c:v", self.name, *self.quality_args, *BITRATE_ARGS]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "vendor": self.vendor,
            "integrated": self.integrated,
            "label": self.label,
            "note": self.note,
            "args": self.video_args(),
        }


_cache: dict[str, EncoderChoice | None] = {}


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def _probe_available() -> set[str]:
    if not shutil.which("ffmpeg"):
        return set()
    try:
        out = _run(["ffmpeg", "-hide_banner", "-encoders"], 20).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V") and "." in parts[0]:
            names.add(parts[1])
    return names


def _encode_works(candidate: EncoderChoice) -> bool:
    """A short synthetic encode through the real driver path, with the exact production arguments.

    Exit status alone is not enough: a driver can accept every option and still emit corrupted
    pictures. The encode is scored against the pattern it was given, which rejects that class of
    failure. It cannot judge content-dependent rate control, so the profiles below stay conservative.
    """
    source = f"testsrc=size={DELIVERY_WIDTH}x{DELIVERY_HEIGHT}:rate=30:duration=0.4"
    with tempfile.TemporaryDirectory(prefix="clipforge-encoder-") as folder:
        clip = os.path.join(folder, "probe.mp4")
        try:
            encoded = _run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi", "-i", source,
                    "-an", *candidate.video_args(),
                    "-pix_fmt", "yuv420p", "-threads", "1", clip,
                ],
                PROBE_TIMEOUT_SEC,
            )
            if encoded.returncode != 0 or not os.path.exists(clip) or os.path.getsize(clip) == 0:
                return False
            scored = _run(
                [
                    "ffmpeg", "-hide_banner", "-v", "info", "-i", clip,
                    "-f", "lavfi", "-i", source,
                    "-lavfi", "[0:v][1:v]ssim", "-threads", "1", *NULL_OUTPUT,
                ],
                PROBE_TIMEOUT_SEC,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    match = re.search(r"All:([0-9.]+)", scored.stderr.decode("utf-8", "replace"))
    if match is None:
        return False
    return float(match.group(1)) >= MIN_PROBE_SSIM


def _from_spec(spec: EncoderSpec, quality: tuple[str, ...], note: str = "") -> EncoderChoice:
    return EncoderChoice(
        spec.name, spec.kind, list(quality), note=note,
        vendor=spec.vendor, integrated=spec.integrated,
    )


def requested_encoder() -> str | None:
    """An explicit override, so a GPU machine can be pinned to the software path for comparison."""
    value = (os.environ.get(ENV_ENCODER) or "").strip()
    if not value or value.lower() == "auto":
        return None
    return value


def detect_encoder(force: str | None = None) -> EncoderChoice | None:
    """First encoder that actually encodes on this machine, cached for the process lifetime."""
    force = force or requested_encoder()
    key = force or "auto"
    if key in _cache:
        return _cache[key]
    if not shutil.which("ffmpeg"):
        _cache[key] = None
        return None
    available = _probe_available()
    chosen: EncoderChoice | None = None
    for spec in SPECS:
        if force and spec.name != force:
            continue
        if spec.name not in available:
            continue
        for index, quality in enumerate(spec.variants):
            candidate = _from_spec(spec, quality, note="" if index == 0 else "compatibility profile")
            if _encode_works(candidate):
                chosen = candidate
                break
        if chosen is not None:
            break
    _cache[key] = chosen
    return chosen


def cpu_encoder() -> EncoderChoice:
    """The software encoder used when no accelerator exists, or a GPU one runs out of capacity."""
    for spec in SPECS:
        if spec.kind == "cpu":
            return _from_spec(spec, spec.variants[0], note="software fallback")
    raise RuntimeError("no software encoder is defined")


def encoder_video_args(choice: EncoderChoice | None, fps_arg: str) -> list[str]:
    if choice is None:
        choice = cpu_encoder()
    return [*choice.video_args(), "-pix_fmt", "yuv420p", "-r", fps_arg]


_DETECT = object()


def encoder_capabilities(choice=_DETECT) -> dict:
    """What this machine can actually do, for the health endpoint and the studio's hardware panel."""
    chosen = detect_encoder() if choice is _DETECT else choice
    software = cpu_encoder()
    if chosen is None:
        tier = "unavailable"
    elif chosen.kind == "cpu":
        tier = "cpu"
    elif chosen.integrated:
        tier = "integrated"
    else:
        tier = "discrete"
    return {
        "tier": tier,
        "encoder": chosen.describe() if chosen else None,
        "software_fallback": software.describe(),
        "delivery": {
            "width": DELIVERY_WIDTH,
            "height": DELIVERY_HEIGHT,
            "pixel_format": "yuv420p",
            "codec": "h264",
            "profile": "high",
            "bitrate": BITRATE_ARGS[1],
            "max_bitrate": BITRATE_ARGS[3],
        },
        "quality_locked": True,
    }