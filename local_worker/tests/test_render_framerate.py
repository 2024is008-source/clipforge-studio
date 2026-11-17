"""Regression tests for the frame-rate policy: the output is cut at the footage's own cadence.

Two failures these guard against both read as stutter on slow pans. Forcing every export to 29.97
turns 23.976 film into an uneven 4-4-5-4 duplication pattern, and giving FFmpeg a decimal 23.976
drifts the timestamps against the 24000/1001 grid the filters work on.
"""

from __future__ import annotations

import unittest

from local_worker.rendering.decision import TrackPoint, decision_from_payload
from local_worker.rendering.filters import Panel, crop_keyframes, panel_crop_chain
from local_worker.rendering.framerate import (
    STANDARD_RATES,
    choose_source_rate,
    is_variable_rate,
    rate_arg,
    requested_fps,
    resolve_output_fps,
    snap_to_standard,
)

PANEL = Panel("main", 1080, 1920, 1.0, 1.0, False)


class RequestedRateTest(unittest.TestCase):
    def test_the_studio_default_survives_junk(self):
        self.assertAlmostEqual(requested_fps("29.97"), 29.97002997002997, places=9)
        self.assertAlmostEqual(requested_fps("nonsense"), 29.97002997002997, places=9)
        self.assertAlmostEqual(requested_fps(""), 29.97002997002997, places=9)
        self.assertAlmostEqual(requested_fps(0), 29.97002997002997, places=9)
        self.assertAlmostEqual(requested_fps(120), 29.97002997002997, places=9)
        self.assertAlmostEqual(requested_fps(25), 25.0, places=9)


class SnapTest(unittest.TestCase):
    def test_a_genuine_rate_is_not_snapped_to_its_drop_frame_neighbour(self):
        """Ordered scanning used to turn 30 into 29.97 and 60 into 59.94 - both are 0.03 away."""
        self.assertAlmostEqual(snap_to_standard(30.0), 30.0, places=9)
        self.assertAlmostEqual(snap_to_standard(60.0), 60.0, places=9)
        self.assertAlmostEqual(snap_to_standard(24.0), 24.0, places=9)
        self.assertAlmostEqual(snap_to_standard(59.94005994), 59.94005994005994, places=9)

    def test_measured_rates_come_back_as_the_standard_the_footage_was_shot_at(self):
        self.assertAlmostEqual(snap_to_standard(23.975163005985255), 23.976023976023978, places=9)
        self.assertAlmostEqual(snap_to_standard(29.9698), 29.97002997002997, places=9)

    def test_an_odd_rate_stays_odd(self):
        self.assertAlmostEqual(snap_to_standard(31.0), 31.0, places=9)
        self.assertAlmostEqual(snap_to_standard(15.0), 15.0, places=9)


class RateArgTest(unittest.TestCase):
    def test_every_standard_rate_gets_its_exact_rational(self):
        for fps, arg in STANDARD_RATES:
            self.assertEqual(rate_arg(fps), arg)

    def test_a_non_standard_rate_falls_back_to_a_decimal(self):
        self.assertEqual(rate_arg(27.5), "27.500000")


class VariableRateTest(unittest.TestCase):
    def test_one_cadence_read_two_ways_is_not_variable(self):
        self.assertFalse(is_variable_rate(29.97002997002997, 30.0))
        self.assertFalse(is_variable_rate(23.976023976023978, 23.975163005985255))
        self.assertFalse(is_variable_rate(30.0, 30.0))

    def test_a_real_range_is_variable(self):
        self.assertTrue(is_variable_rate(60.0, 30.0))

    def test_a_variable_source_is_sampled_at_its_fastest_instant(self):
        """Duplicating frames loses nothing; dropping them loses a moment of the user's clip."""
        self.assertAlmostEqual(choose_source_rate(23.975, 30.0), 30.0, places=9)
        self.assertAlmostEqual(choose_source_rate(29.970, 29.970), 29.970, places=3)
        self.assertAlmostEqual(choose_source_rate(0.0, 25.0), 25.0, places=9)


class ResolveOutputRateTest(unittest.TestCase):
    def test_film_is_not_pushed_to_the_export_default(self):
        resolved = resolve_output_fps(23.975163005985255, "29.97")
        self.assertAlmostEqual(resolved, 23.976023976023978, places=9)
        self.assertEqual(rate_arg(resolved), "24000/1001")

    def test_a_request_that_matches_the_source_keeps_the_source(self):
        self.assertAlmostEqual(resolve_output_fps(29.97002997002997, "30"), 29.97002997002997, places=9)

    def test_a_clean_whole_multiple_is_allowed(self):
        self.assertAlmostEqual(resolve_output_fps(29.97002997002997, "59.94"), 59.94005994005994, places=9)
        self.assertAlmostEqual(resolve_output_fps(59.94005994005994, "29.97"), 29.97002997002997, places=9)

    def test_an_uneven_conversion_stays_at_the_source_cadence(self):
        """24 to 29.97 is a 4-4-5-4 pattern, so the request is refused rather than juddered through."""
        self.assertAlmostEqual(resolve_output_fps(24.0, "29.97"), 24.0, places=9)
        self.assertAlmostEqual(resolve_output_fps(25.0, "29.97"), 25.0, places=9)

    def test_an_unmeasurable_source_falls_back_to_the_request(self):
        self.assertAlmostEqual(resolve_output_fps(0.0, "29.97"), 29.97002997002997, places=9)
        self.assertAlmostEqual(resolve_output_fps(120.0, "29.97"), 29.97002997002997, places=9)


class GraphOrderTest(unittest.TestCase):
    """The resample has to happen before the timed crops, or the commands address the wrong frames."""

    def decision(self) -> object:
        return decision_from_payload({
            "source_path": "source.mp4",
            "source_width": 1920,
            "source_height": 1080,
            "source_fps": 23.975163005985255,
            "start": 0.0,
            "end": 2.0,
            "layout": "fill",
            "track": [
                {"t": 0.0, "x": 0.3, "y": 0.4},
                {"t": 1.0, "x": 0.5, "y": 0.4},
                {"t": 2.0, "x": 0.7, "y": 0.4},
            ],
            "words": [],
            "hook": "",
            "style": {},
        })

    def test_fps_conversion_precedes_the_crop(self):
        fps = resolve_output_fps(23.975163005985255, "29.97")
        decision = self.decision()
        keyframes = crop_keyframes(decision, PANEL, fps, 1.0)
        graph = panel_crop_chain(PANEL, "c.cmd", keyframes, decision, fps)
        self.assertLess(graph.index("fps=24000/1001"), graph.index("crop=w="))
        self.assertLess(graph.index("fps=24000/1001"), graph.index("sendcmd"))
        self.assertNotIn("29.97", graph)

    def test_command_times_land_on_output_frames(self):
        fps = resolve_output_fps(23.975163005985255, "29.97")
        decision = self.decision()
        keyframes = crop_keyframes(decision, PANEL, fps, 1.0)
        self.assertGreater(len(keyframes), 1)
        for frame in keyframes:
            slot = frame.t * fps
            self.assertLess(abs(slot - round(slot)), 1e-6, msg=f"{frame.t} is not a frame slot")


if __name__ == "__main__":
    unittest.main()
