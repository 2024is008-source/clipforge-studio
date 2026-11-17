"""The export must move the way the preview moves.

These lock the constants that src/lib/render.ts and src/lib/subtitles.ts calibrate against, so a tune
in one pipeline cannot silently drift away from the other.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from local_worker.rendering.decision import decision_from_payload
from local_worker.rendering.filters import (
    PUNCH_FALL_SEC,
    PUNCH_HOLD_SEC,
    PUNCH_RISE_SEC,
    PUNCH_SPAN_SEC,
    PUNCH_MAX_ZOOM,
    _grade_chain,
    word_punch_zoom,
)
from local_worker.rendering.subtitles import ENTRANCE_FADE_MS, write_ass


def decision(words: list[dict], style: dict) -> object:
    return decision_from_payload(
        {
            "source_path": "source.mp4",
            "source_width": 1920,
            "source_height": 1080,
            "source_fps": 30.0,
            "start": 0.0,
            "end": 6.0,
            "layout": "fill",
            "track": [{"t": 0.0, "x": 0.5, "y": 0.38}],
            "words": words,
            "hook": "",
            "style": style,
        }
    )


class TransitionsAreCleanTest(unittest.TestCase):
    """An edit changes the shot, not the exposure: nothing may darken or smear a frame at a seam.

    These three effects each had a name that suggested polish — a dip to black on cuts, a defocus
    ladder across layout seams, a radial vignette — and each one read as a transition instead.
    """

    def test_the_dip_blur_and_vignette_are_gone_from_the_export_module(self):
        from local_worker.rendering import filters

        for name in (
            "_cut_dip_filter", "CUT_DIP_SEC", "_edge_blur_filter", "_edge_blur_ramps",
            "EDGE_BLUR_SEC", "EDGE_BLUR_SIGMAS", "vignette_alpha_expression", "VIGNETTE_EDGE_ALPHA",
        ):
            self.assertFalse(hasattr(filters, name), name)

    def test_the_grade_chain_adds_nothing_at_a_seam(self):
        graph = ";".join(_grade_chain("[c]", None, decision=decision([], {})))
        for token in ("gblur", "overlay", "eq=brightness"):
            self.assertNotIn(token, graph)

    def test_the_preview_dropped_the_same_three(self):
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "..", "src", "lib", "render.ts")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for name in ("drawCutDip", "drawVignette", "vignetteFill", "seamDipBlur", "EDGE_BLUR_SIGMAS"):
            self.assertNotIn(name, text)


class PunchCurveTest(unittest.TestCase):
    WORDS = [{"w": "MILLION!", "start": 2.0, "end": 2.4}]

    def test_the_release_is_longer_than_the_push(self):
        """A 0.55 s fall after a 0.25 s rise read as a snap-back rather than a settle."""
        self.assertGreaterEqual(PUNCH_FALL_SEC, 0.7)
        self.assertGreater(PUNCH_FALL_SEC, PUNCH_RISE_SEC)
        self.assertAlmostEqual(
            PUNCH_SPAN_SEC, PUNCH_RISE_SEC + PUNCH_HOLD_SEC + PUNCH_FALL_SEC, places=12
        )

    def test_zoom_reaches_full_at_the_top_of_the_rise_and_rests_after_the_span(self):
        self.assertAlmostEqual(
            word_punch_zoom(self.WORDS, 2.0 + PUNCH_RISE_SEC, 1.0), PUNCH_MAX_ZOOM, places=9
        )
        self.assertAlmostEqual(word_punch_zoom(self.WORDS, 2.0 + PUNCH_SPAN_SEC + 0.01, 1.0), 1.0)

    def test_no_frame_jumps_more_than_a_ninth_of_the_zoom(self):
        """Bounded per-frame delta is what separates a glide from a step at 30fps."""
        samples = [word_punch_zoom(self.WORDS, 2.0 + i / 30.0, 1.0) for i in range(40)]
        worst = max(abs(b - a) for a, b in zip(samples, samples[1:]))
        self.assertLess(worst, (PUNCH_MAX_ZOOM - 1.0) / 2.5)


class EntranceFadeTest(unittest.TestCase):
    WORDS = [{"w": "million", "start": 0.2, "end": 0.6}]

    def ass(self, preset: str) -> str:
        dec = decision(self.WORDS, {"preset": preset, "showHook": False})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.ass")
            write_ass(dec, path)
            with open(path, "r", encoding="utf-8-sig") as handle:
                return handle.read()

    def test_fades_are_per_preset_not_a_single_constant(self):
        self.assertEqual(ENTRANCE_FADE_MS["reveal"], 360)
        self.assertEqual(ENTRANCE_FADE_MS["bounce"], 0)
        self.assertEqual(ENTRANCE_FADE_MS["typewriter"], 0)
        self.assertIn(r"\fad(360,0)", self.ass("reveal"))
        self.assertIn(r"\fad(90,0)", self.ass("opus"))
        self.assertIn(r"\fad(0,0)", self.ass("bounce"))

    def test_a_preset_outside_the_table_still_gets_a_fade(self):
        self.assertIn(r"\fad(80,0)", self.ass("not_a_preset"))

    def test_the_emoji_is_tweened_to_the_preview_overshoot(self):
        dec = decision([{"w": "million", "start": 0.2, "end": 0.6}], {"preset": "opus"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.ass")
            write_ass(dec, path)
            with open(path, "r", encoding="utf-8-sig") as handle:
                content = handle.read()
        self.assertIn(r"\t(0,180,\fscx115\fscy115)", content)
        self.assertIn(r"\t(180,340,\fscx100\fscy100)", content)
        # Overshoot only: a flat \fscx120 would be the old static approximation.
        self.assertNotIn(r"{\fscx120\fscy120}", content)


if __name__ == "__main__":
    unittest.main()
