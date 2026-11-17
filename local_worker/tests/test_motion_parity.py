"""The export must move the way the preview moves.

These lock the constants that src/lib/render.ts and src/lib/subtitles.ts calibrate against, so a tune
in one pipeline cannot silently drift away from the other.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest

from local_worker.rendering.decision import decision_from_payload
from local_worker.rendering.filters import (
    BED_BLUR_SIGMA,
    BED_SATURATION,
    PUNCH_FALL_SEC,
    PUNCH_HOLD_SEC,
    PUNCH_RISE_SEC,
    PUNCH_SPAN_SEC,
    PUNCH_MAX_ZOOM,
    _grade_chain,
    build_composite_graph,
    word_punch_zoom,
)
from local_worker.rendering.subtitles import (
    ENTRANCE_FADE_MS,
    HOOK_DURATION,
    HOOK_FONT_SIZE,
    HOOK_LINE_PITCH,
    HOOK_MAX_WIDTH,
    HOOK_PAD_X,
    HOOK_PAD_Y,
    HOOK_RADIUS,
    HOOK_TOP,
    write_ass,
)


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


class HookCardParityTest(unittest.TestCase):
    """The hook is a centred white card near the top; canvas preview and ASS fallback share its box."""

    HOOK = "Eating a city's food for a million dollars?"

    def ts_hook_constants(self) -> dict[str, float]:
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "..", "src", "lib", "subtitles.ts")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        names = (
            "HOOK_DURATION", "HOOK_FONT_SIZE", "HOOK_LINE_PITCH", "HOOK_PAD_X",
            "HOOK_PAD_Y", "HOOK_TOP", "HOOK_RADIUS", "HOOK_MAX_WIDTH",
        )
        return {
            name: float(re.search(rf"export const {name} = ([\d.]+);", text).group(1))
            for name in names
        }

    def ass(self) -> str:
        dec = decision_from_payload(
            {
                "source_path": "source.mp4",
                "source_width": 1920,
                "source_height": 1080,
                "source_fps": 30.0,
                "start": 0.0,
                "end": 6.0,
                "layout": "fill",
                "track": [{"t": 0.0, "x": 0.5, "y": 0.38}],
                "words": [],
                "hook": self.HOOK,
                "style": {"preset": "opus", "showHook": True},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.ass")
            write_ass(dec, path)
            with open(path, "r", encoding="utf-8-sig") as handle:
                return handle.read()

    def test_the_preview_constants_are_the_export_constants(self):
        ts = self.ts_hook_constants()
        for name, value in (
            ("HOOK_DURATION", HOOK_DURATION), ("HOOK_FONT_SIZE", HOOK_FONT_SIZE),
            ("HOOK_LINE_PITCH", HOOK_LINE_PITCH), ("HOOK_PAD_X", HOOK_PAD_X),
            ("HOOK_PAD_Y", HOOK_PAD_Y), ("HOOK_TOP", HOOK_TOP),
            ("HOOK_RADIUS", HOOK_RADIUS), ("HOOK_MAX_WIDTH", HOOK_MAX_WIDTH),
        ):
            self.assertEqual(ts[name], value, name)

    def test_the_ass_card_is_top_centred_with_the_preview_wrap_width(self):
        style = next(line for line in self.ass().splitlines() if line.startswith("Style: Hook,"))
        fields = style.split(": ", 1)[1].split(",")
        self.assertEqual(fields[2], str(HOOK_FONT_SIZE))
        self.assertEqual(fields[15], "3")  # opaque box
        self.assertEqual(fields[16], str(HOOK_PAD_Y))
        self.assertEqual(fields[18], "8")  # top centre
        margin = (1080 - (HOOK_MAX_WIDTH - 2 * HOOK_PAD_X)) // 2
        self.assertEqual(fields[19], str(margin))
        self.assertEqual(fields[20], str(margin))
        self.assertEqual(fields[21], str(HOOK_TOP + HOOK_PAD_Y))

    def test_the_hook_keeps_sentence_case_in_both_pipelines(self):
        content = self.ass()
        dialogue = next(line for line in content.splitlines() if ",Hook," in line)
        self.assertIn("Eating a city's food", dialogue)
        self.assertNotIn("EATING", dialogue)


class ContextBedParityTest(unittest.TestCase):
    """The fit bed is the shot defocused at neutral exposure, identical in canvas and export graph."""

    def ts_bed_constants(self) -> dict[str, float]:
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "..", "src", "lib", "render.ts")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        return {
            name: float(re.search(rf"export const {name} = ([\d.]+);", text).group(1))
            for name in ("BED_BLUR_SIGMA", "BED_SATURATION")
        }

    def test_the_preview_constants_are_the_export_constants(self):
        ts = self.ts_bed_constants()
        self.assertEqual(ts["BED_BLUR_SIGMA"], BED_BLUR_SIGMA)
        self.assertEqual(ts["BED_SATURATION"], BED_SATURATION)

    def test_the_export_bed_is_a_quarter_res_blur_at_neutral_exposure(self):
        dec = decision([], {"preset": "opus"})
        graph, _ = build_composite_graph(dec, "fit", 0, 30.0, 0)
        bed = next(chain for chain in graph.split(";") if chain.endswith("[bed]"))
        self.assertIn(f"scale={1080 // 4}:{1920 // 4},gblur=sigma={BED_BLUR_SIGMA}", bed)
        self.assertIn(f"eq=saturation={BED_SATURATION}", bed)
        self.assertNotIn("eq=brightness", bed)
        self.assertNotIn("lutyuv", bed)

    def test_the_preview_blurs_and_keeps_chroma_without_darkening_or_washing(self):
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, "..", "..", "src", "lib", "render.ts")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("blur(${BED_BLUR_SIGMA}px) saturate(${BED_SATURATION})", text)
        self.assertNotIn("brightness(0.9)", text)
        self.assertNotIn("BED_WASH_ALPHA", text)


if __name__ == "__main__":
    unittest.main()
