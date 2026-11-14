"""Layout seams must not dip the audio: the clip's sound is mixed once, not per layout window."""

from __future__ import annotations

import array
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from local_worker.rendering import renderer
from local_worker.rendering.decision import AudioTrack, decision_from_payload
from local_worker.rendering.encoders import cpu_encoder
from local_worker.rendering.renderer import RenderRequest, Segment


def seam_decision(source: str, end: float = 4.0):
    return decision_from_payload({
        "source_path": source,
        "source_width": 1280,
        "source_height": 720,
        "source_fps": 30.0,
        "start": 0.0,
        "end": end,
        "layout": "auto",
        "track": [
            {"t": 0.0, "x": 0.5, "y": 0.4, "mode": "fill"},
            {"t": 1.99, "x": 0.5, "y": 0.4, "mode": "fill"},
            {"t": 2.0, "x": 0.5, "y": 0.4, "mode": "fit", "cut": True},
            {"t": end, "x": 0.5, "y": 0.4, "mode": "fit"},
        ],
        "words": [],
        "style": {"preset": "none", "showHook": False},
        "zoomPunch": False,
    })


class ConcatAudioPlanTest(unittest.TestCase):
    def test_concat_mixes_full_clip_audio_once(self):
        with tempfile.TemporaryDirectory() as job_dir:
            decision = seam_decision(os.path.join(job_dir, "source.mp4"))
            music = os.path.join(job_dir, "music.wav")
            decision.audio = [AudioTrack("original", ""), AudioTrack("music", music, fade_out=1.0)]
            request = RenderRequest(decision, job_dir, fps=30, encoder=cpu_encoder())
            planned = renderer._concat_pass(request, ["a.mp4", "b.mp4"], True)
            args = planned.args
            inputs = [args[i + 1] for i, arg in enumerate(args) if arg == "-i"]
            self.assertEqual(inputs, ["a.mp4", "b.mp4", decision.source_path, music])
            self.assertEqual(args[args.index("-ss") + 1], "0.0000")
            self.assertEqual(args[args.index("-t") + 1], "4.0000")
            graph = args[args.index("-filter_complex") + 1]
            self.assertIn("[0:v][1:v]concat=n=2:v=1:a=0[vout]", graph)
            self.assertIn("[2:a]aresample", graph)
            self.assertIn("[3:a]aresample", graph)
            self.assertEqual(graph.count("loudnorm"), 1)
            # A music fade belongs to the clip's end, not to the end of each window.
            self.assertIn("afade=t=out:st=3.000", graph)

    def test_windows_render_silent_when_the_clip_has_seams(self):
        with tempfile.TemporaryDirectory() as job_dir:
            source = os.path.join(job_dir, "source.mp4")
            open(source, "wb").close()
            request = RenderRequest(seam_decision(source), job_dir, fps=30, encoder=cpu_encoder())
            with patch.object(renderer.shutil, "which", return_value="ffmpeg"), \
                 patch.object(renderer, "source_has_audio", return_value=True), \
                 patch.object(renderer, "_ensure_vignette", return_value=os.path.join(job_dir, "v.png")), \
                 patch.object(renderer, "layout_segments",
                              return_value=[Segment("fill", 0, 2), Segment("fit", 2, 4)]), \
                 patch.object(renderer, "plan_passes", wraps=renderer.plan_passes) as plan, \
                 patch.object(renderer, "_run", side_effect=RuntimeError("planned only")):
                with self.assertRaises(RuntimeError):
                    renderer.run_render(request)
            self.assertEqual(plan.call_count, 2)
            self.assertEqual([call.args[3] for call in plan.call_args_list], [False, False])


def _windowed_rms(path: str, window: float = 0.05) -> list[float]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    samples = array.array("h", raw)
    step = int(48000 * window)
    out = []
    for i in range(0, len(samples) - step + 1, step):
        chunk = samples[i:i + step]
        out.append((sum(s * s for s in chunk) / step) ** 0.5)
    return out


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the seam audio render")
class SeamAudioRenderTest(unittest.TestCase):
    def test_steady_tone_keeps_its_level_across_the_layout_seam(self):
        with tempfile.TemporaryDirectory() as job_dir:
            source = os.path.join(job_dir, "source.mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=4",
                 "-f", "lavfi", "-i", "sine=f=440:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", source],
                check=True,
            )
            request = RenderRequest(seam_decision(source), job_dir, fps=30, encoder=cpu_encoder())
            with patch.object(renderer, "layout_segments",
                              return_value=[Segment("fill", 0, 2), Segment("fit", 2, 4)]):
                result = renderer.run_render(request)
            self.assertEqual(len(result.segments), 2)
            # Per-window AAC left a brief near-silent gap at each seam; only short windows see it.
            rms = _windowed_rms(result.output_path, 0.005)
            body = rms[100:-100]
            steady = sorted(body)[len(body) // 2]
            seam = rms[370:430]  # 1.85s .. 2.15s
            self.assertGreater(steady, 0)
            self.assertGreater(min(seam) / steady, 0.8, f"seam dipped to {min(seam) / steady:.2f}")


if __name__ == "__main__":
    unittest.main()
