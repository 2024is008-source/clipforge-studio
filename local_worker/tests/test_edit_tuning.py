"""Unit tests for the per-video editing-style calibration (tuning) and the layout hold."""
import os
import random
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker.openshorts_engine import adaptive_cut_threshold, enforce_layout_hold
from local_worker.tuning import Signature, tune


class TestPerVideoTuning(unittest.TestCase):
    def test_the_hold_and_the_camera_come_from_the_clips_own_rhythm(self):
        rapid = tune(
            Signature(
                shot_gaps=[1.0, 1.2, 0.9, 1.1],
                samples=200,
                face_samples=190,
                two_face_share=0.30,
                median_face_width=0.16,
                motion_energy=90.0,
                subject_travel=0.05,
            )
        )
        self.assertEqual(rapid.style, "rapid_multicam")
        # The measured 0.84 s sits under the hold floor: a template that changes every second reads
        # as a twitch, so rapid footage still gets the calmest legal hold.
        self.assertAlmostEqual(rapid.layout_hold_sec, 3.0, places=3)
        self.assertAlmostEqual(rapid.split_min_hold_sec, 3.0, places=3)
        self.assertAlmostEqual(rapid.cut_snap_min_sec, 0.63, places=3)
        self.assertAlmostEqual(rapid.tau_x, 0.30, places=3)

        long_take = tune(
            Signature(
                shot_gaps=[6.0, 7.5],
                samples=200,
                face_samples=180,
                two_face_share=0.05,
                median_face_width=0.30,
                motion_energy=12.0,
                subject_travel=0.03,
            )
        )
        self.assertEqual(long_take.style, "long_take")
        self.assertAlmostEqual(long_take.layout_hold_sec, 5.0, places=3)
        self.assertAlmostEqual(long_take.tau_x, 0.46, places=3)
        self.assertGreater(long_take.deadband_x, rapid.deadband_x)
        self.assertFalse(long_take.split_allowed)

    def test_a_window_with_no_cuts_is_one_take_not_dialogue(self):
        locked = tune(
            Signature(
                shot_gaps=[],
                samples=200,
                face_samples=200,
                median_face_width=0.34,
                motion_energy=4.0,
                subject_travel=0.01,
            )
        )
        self.assertEqual(locked.style, "static_single_shot")
        self.assertAlmostEqual(locked.layout_hold_sec, 5.0, places=3)
        self.assertAlmostEqual(locked.max_speed_x, 0.35, places=3)

    def test_the_pre_pass_cannot_declare_action_it_did_not_detect(self):
        # No detector has run, so face_samples is 0 by construction rather than by observation.
        pre_pass = tune(
            Signature(
                shot_gaps=[3.0, 4.0],
                samples=200,
                subjects_measured=False,
                motion_energy=40.0,
            )
        )
        measured = tune(
            Signature(
                shot_gaps=[3.0, 4.0],
                samples=200,
                face_samples=0,
                motion_energy=40.0,
            )
        )
        self.assertEqual(pre_pass.style, "dialogue_cuts")
        self.assertEqual(measured.style, "action_object")

    def test_night_footage_gets_a_wider_tolerance_and_a_softer_punch(self):
        dark = tune(
            Signature(
                shot_gaps=[2.2, 2.6, 1.9],
                samples=200,
                face_samples=60,
                black_share=0.45,
                two_face_share=0.10,
                median_face_width=0.09,
                motion_energy=25.0,
            )
        )
        self.assertAlmostEqual(dark.deadband_x, 0.10, places=3)
        self.assertAlmostEqual(dark.tau_x, 0.494, places=3)
        self.assertAlmostEqual(dark.max_speed_y, 0.35, places=3)
        self.assertAlmostEqual(dark.punch_gain, 0.6, places=3)
        self.assertFalse(dark.split_allowed)


class TestAdaptiveSceneCutThreshold(unittest.TestCase):
    def setUp(self):
        random.seed(7)

    def test_it_finds_real_cuts_on_bright_and_on_dark_footage(self):
        bright = [random.gauss(0.16, 0.05) for _ in range(188)] + [0.52, 0.60, 0.70, 0.49, 0.55, 0.61, 0.58, 0.66]
        dark = [random.gauss(0.06, 0.02) for _ in range(188)] + [0.20, 0.24, 0.19, 0.22, 0.26, 0.21, 0.23, 0.25]
        for dists in (bright, dark):
            threshold = adaptive_cut_threshold(dists)
            self.assertLessEqual(sum(1 for d in dists if d > threshold), 10)
            self.assertGreaterEqual(sum(1 for d in dists[-8:] if d > threshold), 6)

    def test_handheld_motion_and_a_locked_camera_do_not_produce_cuts(self):
        handheld = [random.gauss(0.30, 0.04) for _ in range(196)]
        self.assertLess(sum(1 for d in handheld if d > adaptive_cut_threshold(handheld)), 4)
        locked = [random.gauss(0.05, 0.01) for _ in range(196)]
        self.assertEqual(sum(1 for d in locked if d > adaptive_cut_threshold(locked)), 0)

    def test_a_window_too_short_to_measure_keeps_the_strict_bound(self):
        self.assertGreaterEqual(adaptive_cut_threshold([0.3, 0.1]), 0.5)


class TestLayoutHold(unittest.TestCase):
    @staticmethod
    def _track(modes_and_times):
        return [{"t": t, "mode": m, "x": 0.5, "y": 0.4, "x2": 0.5, "y2": 0.4} for m, t in modes_and_times]

    def test_a_split_that_cannot_survive_the_hold_is_folded_away(self):
        track = self._track(
            [("fill", 0.0), ("fill", 1.0), ("split", 2.0), ("split", 2.5), ("fill", 3.0), ("fill", 5.0)]
        )
        self.assertEqual(enforce_layout_hold(track, 2.0), 2)
        self.assertEqual({pt["mode"] for pt in track}, {"fill"})

    def test_folding_into_a_split_carries_the_pair_geometry(self):
        track = self._track(
            [("split", 0.0), ("fill", 4.0), ("fill", 4.5), ("split", 5.0), ("split", 12.0)]
        )
        for pt in track:
            if pt["mode"] == "split":
                pt["x"], pt["y"], pt["x2"], pt["y2"] = 0.3, 0.4, 0.8, 0.6
        enforce_layout_hold(track, 2.0)
        self.assertEqual(track[2]["mode"], "split")
        self.assertEqual(track[2]["x2"], 0.8)

    def test_a_split_has_to_hold_the_screen_for_three_seconds(self):
        track = self._track(
            [("fill", 0.0), ("fill", 4.0), ("split", 5.0), ("split", 6.8), ("fill", 7.0), ("fill", 12.0)]
        )
        # 1.8 s clears the general hold, so only the split's own floor can object to it.
        self.assertEqual(enforce_layout_hold(track, 1.5), 0)
        self.assertEqual(enforce_layout_hold(track, 1.5, 3.0), 2)
        self.assertEqual({pt["mode"] for pt in track}, {"fill"})

    def test_a_short_fill_inside_a_split_does_not_end_the_split(self):
        track = self._track(
            [("split", 0.0), ("split", 4.0), ("fill", 5.0), ("fill", 5.5), ("split", 6.0), ("split", 12.0)]
        )
        for pt in track:
            pt["x2"], pt["y2"] = (0.8, 0.6) if pt["mode"] == "split" else (pt["x"], pt["y"])
        enforce_layout_hold(track, 1.5, 3.0)
        self.assertEqual({pt["mode"] for pt in track}, {"split"})
        self.assertEqual(track[2]["x2"], 0.8)

    def test_a_short_split_at_the_start_folds_into_the_shot_that_follows(self):
        track = self._track([("split", 0.0), ("split", 1.0), ("fill", 2.0), ("fill", 10.0)])
        enforce_layout_hold(track, 1.5, 3.0)
        self.assertEqual({pt["mode"] for pt in track}, {"fill"})

    def test_no_hold_leaves_the_track_alone(self):
        track = self._track([("fill", 0.0), ("split", 1.0), ("fill", 2.0)])
        self.assertEqual(enforce_layout_hold(track, 0.0), 0)
        self.assertEqual(track[1]["mode"], "split")


if __name__ == "__main__":
    unittest.main()
