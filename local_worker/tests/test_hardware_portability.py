"""Every PC must reach the same delivery quality, whichever encoder it actually has.

The profiles here were chosen by measurement on a real 1080x1920 ClipForge master, not by reading
FFmpeg docs. Against the same input: Intel QSV VBR scored VMAF 97.34, x264 at the same 16M target
97.91, x264 left to pick its own bitrate from a CRF only 95.56, and an `-extbrc 1 -mbbrc 1` QSV
variant that FFmpeg accepted without complaint scored 53.17 while overshooting the ceiling. These
tests keep those results from silently regressing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_worker import main
from local_worker.rendering import encoders
from local_worker.rendering.encoders import (
    BITRATE_ARGS,
    MIN_PROBE_SSIM,
    SPECS,
    EncoderChoice,
    cpu_encoder,
    detect_encoder,
    encoder_capabilities,
    encoder_video_args,
    requested_encoder,
)

HAS_FFMPEG = bool(shutil.which("ffmpeg"))


class ProfileTests(unittest.TestCase):
    def test_every_profile_holds_the_same_delivery_ceiling(self):
        for spec in SPECS:
            for index, quality in enumerate(spec.variants):
                with self.subTest(encoder=spec.name, variant=index):
                    args = encoders._from_spec(spec, quality).video_args()
                    self.assertEqual(args[0], "-c:v")
                    self.assertEqual(args[1], spec.name)
                    self.assertIn("-profile:v", args)
                    self.assertEqual(args[args.index("-profile:v") + 1], "high")
                    # Identical rate control on every path is what makes the outputs comparable.
                    self.assertEqual(args[-len(BITRATE_ARGS):], BITRATE_ARGS)
                    self.assertNotIn("-crf", args)

    def test_software_encoder_keeps_the_explicit_bitrate_target(self):
        # Letting x264 choose its own bitrate from a CRF measured 95.56 VMAF on a real master where
        # the shared 16M target measured 97.91 — a silent quality drop on exactly the machines that
        # have no GPU. The software path must not trade quality for a smaller file.
        args = cpu_encoder().video_args()
        self.assertEqual(args[args.index("-b:v") + 1], "16M")
        self.assertNotIn("-crf", args)
        self.assertNotIn("-qp", args)

    def test_delivery_is_always_h264(self):
        self.assertTrue(all(spec.name.startswith("h264_") or spec.name == "libx264" for spec in SPECS))

    def test_measured_broken_qsv_rate_control_is_not_offered(self):
        qsv = next(spec for spec in SPECS if spec.name == "h264_qsv")
        for quality in qsv.variants:
            self.assertNotIn("-extbrc", quality)
            self.assertNotIn("-mbbrc", quality)
            # Mixing ICQ with a bitrate ceiling dropped the same clip to VMAF 92.69.
            self.assertNotIn("-global_quality", quality)

    def test_software_encoder_is_the_last_resort_and_always_defined(self):
        self.assertEqual(SPECS[-1].kind, "cpu")
        self.assertEqual(cpu_encoder().name, "libx264")
        self.assertEqual(cpu_encoder().note, "software fallback")

    def test_no_encoder_still_produces_valid_software_arguments(self):
        args = encoder_video_args(None, "30/1")
        self.assertEqual(args[:2], ["-c:v", "libx264"])
        self.assertIn("yuv420p", args)
        self.assertEqual(args[-2:], ["-r", "30/1"])


class CapabilityTests(unittest.TestCase):
    def test_tier_follows_the_measured_encoder(self):
        cases = (
            (EncoderChoice("h264_nvenc", "gpu", [], vendor="NVIDIA"), "discrete"),
            (EncoderChoice("h264_qsv", "gpu", [], vendor="Intel", integrated=True), "integrated"),
            (EncoderChoice("h264_amf", "gpu", [], vendor="AMD", integrated=True), "integrated"),
            (cpu_encoder(), "cpu"),
            (None, "unavailable"),
        )
        for choice, tier in cases:
            with self.subTest(tier=tier):
                caps = encoder_capabilities(choice)
                self.assertEqual(caps["tier"], tier)
                self.assertTrue(caps["quality_locked"])
                self.assertEqual(caps["software_fallback"]["name"], "libx264")
                self.assertEqual(caps["delivery"]["width"], 1080)
                self.assertEqual(caps["delivery"]["height"], 1920)

    def test_labels_name_the_vendor_so_the_ui_never_says_generic_gpu(self):
        self.assertEqual(cpu_encoder().label, "CPU (x264)")
        self.assertEqual(
            EncoderChoice("h264_qsv", "gpu", [], vendor="Intel", integrated=True).label,
            "Intel integrated (h264_qsv)",
        )

    def test_env_override_pins_the_path_and_auto_means_probe(self):
        for value, expected in (("libx264", "libx264"), ("auto", None), ("", None), ("  ", None)):
            with self.subTest(value=value):
                with patch.dict(os.environ, {encoders.ENV_ENCODER: value}):
                    self.assertEqual(requested_encoder(), expected)

    def test_forced_encoder_that_this_machine_lacks_returns_none(self):
        encoders._cache.clear()
        try:
            self.assertIsNone(detect_encoder("a_codec_that_does_not_exist"))
        finally:
            encoders._cache.clear()

    def test_worker_and_renderer_share_one_probe(self):
        self.assertEqual(main.VIDEO_ENCODER, main.VIDEO_ENCODER_CHOICE.name if main.VIDEO_ENCODER_CHOICE else None)
        summary = main.accelerator_summary(main.ENGINE.get_hardware_info())
        self.assertIn(summary["tier"], {"discrete", "integrated", "cpu", "unavailable"})
        self.assertTrue(summary["quality_locked"])
        self.assertGreaterEqual(summary["cpu_threads"], 1)

    def test_software_mastering_gets_more_wall_clock_not_less_quality(self):
        self.assertGreater(main._master_budget_seconds(60, "cpu"), main._master_budget_seconds(60, "gpu"))
        self.assertGreaterEqual(main._master_budget_seconds(0, "cpu"), 600)


class ProbeGateTests(unittest.TestCase):
    """A zero exit status is not proof of a usable picture."""

    def _fake_run(self, ssim: float | None, writes_file: bool = True):
        def run(args, timeout):
            if any("ssim" in arg for arg in args):
                text = "" if ssim is None else f"[Parsed_ssim_0] SSIM All:{ssim} (20.0)\n"
                return subprocess.CompletedProcess(args, 0, b"", text.encode())
            if writes_file:
                Path(args[-1]).write_bytes(b"not really a video, but non-empty")
            return subprocess.CompletedProcess(args, 0, b"", b"")
        return run

    def test_corrupted_output_is_rejected_even_when_ffmpeg_exits_zero(self):
        candidate = cpu_encoder()
        with patch.object(encoders, "_run", self._fake_run(MIN_PROBE_SSIM - 0.01)):
            self.assertFalse(encoders._encode_works(candidate))

    def test_clean_output_is_accepted(self):
        with patch.object(encoders, "_run", self._fake_run(0.999)):
            self.assertTrue(encoders._encode_works(cpu_encoder()))

    def test_missing_score_or_empty_file_is_rejected(self):
        for ssim, writes in ((None, True), (0.999, False)):
            with self.subTest(ssim=ssim, writes=writes):
                with patch.object(encoders, "_run", self._fake_run(ssim, writes)):
                    self.assertFalse(encoders._encode_works(cpu_encoder()))

    def test_probe_runs_at_delivery_resolution(self):
        seen: list[list[str]] = []

        def run(args, timeout):
            seen.append(args)
            return self._fake_run(0.999)(args, timeout)

        with patch.object(encoders, "_run", run):
            encoders._encode_works(cpu_encoder())
        self.assertTrue(any(f"size={encoders.DELIVERY_WIDTH}x{encoders.DELIVERY_HEIGHT}" in arg for arg in seen[0]))


@unittest.skipUnless(HAS_FFMPEG, "local FFmpeg required")
class MeasuredParityTests(unittest.TestCase):
    def test_hardware_and_software_paths_both_stay_above_the_quality_floor(self):
        import re

        hardware = detect_encoder()
        candidates = [cpu_encoder()]
        if hardware and hardware.kind == "gpu":
            candidates.insert(0, hardware)
        source = "testsrc2=size=1080x1920:rate=30:duration=1"
        scores = {}
        with tempfile.TemporaryDirectory() as folder:
            for choice in candidates:
                clip = os.path.join(folder, f"{choice.name}.mp4")
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source,
                     "-vf", "noise=alls=6:allf=t", "-an", *choice.video_args(),
                     "-pix_fmt", "yuv420p", "-r", "30", clip],
                    check=True, capture_output=True, timeout=300,
                )
                scored = subprocess.run(
                    ["ffmpeg", "-v", "info", "-i", clip, "-f", "lavfi", "-i", source,
                     "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=300,
                )
                match = re.search(r"All:([0-9.]+)", scored.stderr)
                self.assertIsNotNone(match, scored.stderr[-500:])
                scores[choice.name] = float(match.group(1))
        for name, score in scores.items():
            with self.subTest(encoder=name):
                self.assertGreaterEqual(score, 0.90)


if __name__ == "__main__":
    unittest.main()