"""Where the edits are, and which of them are not edits at all.

The reframe engine reads its footage on a 7.5 Hz sample grid. That is fast enough to find a shot and far
too coarse to say which frame it starts on, and it cannot see at all whether an excursion in the colour
histogram comes back. Both of those decide what the viewer sees: the first is the difference between a
camera that cuts with the edit and one that holds the outgoing composition for another tenth of a second,
and the second is a lightning flash buying the camera a re-aim and costing the tracker its speaker
identities.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from local_worker.clipforge_engine import (
    CUT_THR_MIN,
    SPLIT_SHOT_MAX_FACE_WIDTH,
    SPLIT_SHOT_MIN_SEC,
    SPLIT_SHOT_TWO_FACE_SHARE,
    _histogram_of_hsv,
    enforce_shot_split_evidence,
    refine_cut_frames,
    shot_boundaries,
    shot_face_evidence,
)

FPS = 30.0
STRIDE = 4  # what the engine walks at the default REFRAME_SAMPLE_FPS=7.5 on 30 fps footage
THR = CUT_THR_MIN  # 0.16: the lowest threshold the adaptive pass will ever settle on
SEC = STRIDE / FPS


def _hist(slot: int) -> np.ndarray:
    """A normalised 16x16 histogram living in one bin, the way the engine normalises its own."""
    hist = np.zeros((16, 16), dtype=np.float32)
    hist[slot % 16, (slot * 5) % 16] = 1.0
    cv2.normalize(hist, hist)
    return hist


def _scene(slots: list[int]) -> list[np.ndarray]:
    return [_hist(slot) for slot in slots]


def _dense(frames: list[np.ndarray]) -> list[np.ndarray]:
    """One signature per frame, from the same yardstick the analysis walk compares samples by."""
    return [_histogram_of_hsv(cv2, cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) for frame in frames]


def _at(seconds: list[float]) -> list[int]:
    return [int(round(t * FPS)) for t in seconds]


class TransientExcursionTest(unittest.TestCase):
    def test_a_sustained_change_of_scene_opens_a_shot(self):
        boundaries, refused = shot_boundaries(
            cv2, _scene([0, 0, 0, 1, 1, 1]), _at([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]), FPS, THR
        )
        self.assertEqual(boundaries, [3])
        self.assertEqual(refused, 0)

    def test_a_one_sample_flash_is_not_a_cut(self):
        boundaries, refused = shot_boundaries(
            cv2, _scene([0, 0, 1, 0, 0, 0]), _at([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]), FPS, THR
        )
        self.assertEqual(boundaries, [])
        self.assertEqual(refused, 1)

    def test_a_flash_holding_two_samples_is_still_not_a_cut(self):
        """And the flash *ending* is not a second cut either: one strobe must not cost two re-aims."""
        boundaries, refused = shot_boundaries(
            cv2, _scene([0, 0, 1, 2, 0, 0, 0]), _at([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), FPS, THR
        )
        self.assertEqual(boundaries, [])
        self.assertEqual(refused, 1)

    def test_the_shot_after_a_refused_flash_is_still_found(self):
        boundaries, refused = shot_boundaries(
            cv2, _scene([0, 0, 1, 0, 0, 2, 2]), _at([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5]), FPS, THR
        )
        self.assertEqual(refused, 1)
        self.assertEqual(boundaries, [5])

    def test_an_excursion_at_the_end_of_the_clip_is_believed(self):
        # Nothing follows it to prove it came back, and guessing wrong here loses a real edit.
        boundaries, refused = shot_boundaries(
            cv2, _scene([0, 0, 0, 1]), _at([0.0, 0.5, 1.0, 1.5]), FPS, THR
        )
        self.assertEqual(boundaries, [3])
        self.assertEqual(refused, 0)

    def test_an_edit_shorter_than_the_minimum_shot_never_opens_a_shot(self):
        boundaries, _ = shot_boundaries(
            cv2, _scene([0, 0, 1, 2]), _at([0.0, 0.5, 1.0, 1.3]), FPS, THR
        )
        self.assertEqual(boundaries, [2])

    def test_the_opening_sample_is_never_reported_as_a_cut(self):
        self.assertEqual(shot_boundaries(cv2, _scene([0, 0, 0]), _at([0.0, 0.5, 1.0]), FPS, THR), ([], 0))

    def test_a_sample_the_walk_could_not_decode_is_skipped_not_compared(self):
        hists = [_hist(0), _hist(0), None, _hist(1), _hist(1)]
        boundaries, refused = shot_boundaries(
            cv2, hists, _at([0.0, 0.5, 1.0, 2.0, 2.5]), FPS, THR
        )
        self.assertEqual(boundaries, [3])
        self.assertEqual(refused, 0)


class FrameAccurateBoundaryTest(unittest.TestCase):
    """The sample grid says which eighth of a second; the decoded frames say which one."""

    def _plates(self, length: int, switch_at: int) -> list[np.ndarray]:
        """`switch_at` frames of one colour, the rest of another."""
        out = []
        for f in range(length):
            plate = np.zeros((36, 64, 3), dtype=np.uint8)
            plate[:, :] = (40, 200, 210) if f < switch_at else (110, 60, 210)
            out.append(cv2.cvtColor(plate, cv2.COLOR_HSV2BGR))
        return out

    def _grid(self, samples: int = 6) -> list[int]:
        return [i * STRIDE for i in range(samples)]

    def test_the_edit_lands_on_the_frame_it_happens_on(self):
        for offset in (1, 2, 3):
            with self.subTest(offset=offset):
                switch_at = STRIDE + offset
                frames = self._plates(6 * STRIDE, switch_at)
                sample_frames = self._grid()
                dense = _dense(frames)
                hists = [dense[i] for i in sample_frames]
                boundaries, refused = shot_boundaries(
                    cv2, hists, sample_frames, FPS, THR, min_shot_sec=0.0
                )
                self.assertEqual((boundaries, refused), ([2], 0))
                self.assertEqual(
                    refine_cut_frames(cv2, dense, boundaries, sample_frames, THR, 0), [switch_at]
                )

    def test_a_walk_that_started_inside_the_file_still_counts_frames_from_its_own_start(self):
        offset = 1_000
        switch_at = offset + STRIDE + 2
        frames = self._plates(6 * STRIDE, switch_at - offset)
        sample_frames = [offset + i for i in self._grid()]
        dense = _dense(frames)
        self.assertEqual(
            refine_cut_frames(cv2, dense, [2], sample_frames, THR, offset), [switch_at]
        )

    def test_a_boundary_the_frames_deny_stays_on_the_sample(self):
        # A spike the footage does not contain (a hiccup in the walk, not an edit) must not invent a
        # frame for it: the grid says sample 2, and the frames between the samples say nothing moved.
        frames = self._plates(6 * STRIDE, 6 * STRIDE)
        sample_frames = self._grid()
        dense = _dense(frames)
        hists = [dense[i] for i in sample_frames]
        hists[2] = _hist(9)
        self.assertEqual(refine_cut_frames(cv2, dense, [2], sample_frames, THR, 0), [sample_frames[2]])

    def test_a_cut_beyond_the_frames_the_walk_finished_is_left_where_it_was(self):
        # The dense run stops one frame after the anchor: nothing here can say where the edit landed,
        # and the bounds check has to answer that by keeping the sample rather than by raising.
        frames = self._plates(6 * STRIDE, 6 * STRIDE)
        sample_frames = self._grid()
        dense = _dense(frames)
        self.assertEqual(
            refine_cut_frames(cv2, dense[: STRIDE + 1], [2], sample_frames, THR, 0), [sample_frames[2]]
        )

    def test_no_boundaries_needs_no_frames(self):
        self.assertEqual(refine_cut_frames(cv2, [], [], [], THR, 0), [])


def _face(centre_x: float, width: float, orig_w: int = 1920) -> dict:
    w = width * orig_w
    return {"box": [centre_x * orig_w - w / 2.0, 300.0, w, w * 1.6]}


def _cands(count: int, width: float = 0.12) -> list[dict]:
    return [_face(0.28 + 0.44 * i, width) for i in range(count)]


def _shot(candidates: list[list[dict]], mode: str, cut: bool = False) -> list[dict]:
    """One shot's track points, all stamped with the layout the scene pass voted for."""
    return [
        {
            "t": round(i * SEC, 3),
            "x": 0.4, "y": 0.3,
            "x2": 0.72 if mode == "split" else 0.4, "y2": 0.3,
            "mode": mode,
            "cut": bool(cut and i == 0),
        }
        for i in range(len(candidates))
    ]


class PerShotSplitEvidenceTest(unittest.TestCase):
    """A 2-up is a property of the shot, not of the clip it happens to sit inside."""

    TWO = 20  # samples, about 2.6 s: long enough to read as an intentional shot

    def test_a_two_hander_inside_a_monologue_keeps_its_panels(self):
        monologue = [_cands(1) for _ in range(60)]
        two_shot = [_cands(2) for _ in range(self.TWO)]
        candidates = monologue + two_shot
        points = _shot(monologue, "split") + _shot(two_shot, "split", cut=True)
        evidence = shot_face_evidence(points, candidates, 1920)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["two_face_share"], 0.0)
        self.assertEqual(evidence[1]["two_face_share"], 1.0)
        self.assertEqual(enforce_shot_split_evidence(points, evidence), len(monologue))
        self.assertEqual({p["mode"] for p in points[:len(monologue)]}, {"fill"})
        self.assertEqual({p["mode"] for p in points[len(monologue):]}, {"split"})

    def test_the_folded_panel_points_at_the_subject_it_kept(self):
        candidates = [_cands(1) for _ in range(30)]
        points = _shot(candidates, "split")
        enforce_shot_split_evidence(points, shot_face_evidence(points, candidates, 1920))
        self.assertTrue(all(p["x"] == p["x2"] and p["y"] == p["y2"] for p in points))

    def test_a_shot_that_only_blinks_a_second_face_folds_to_one_subject(self):
        candidates = [_cands(2) if i in (4, 5, 6) else _cands(1) for i in range(30)]
        points = _shot(candidates, "split")
        evidence = shot_face_evidence(points, candidates, 1920)
        self.assertLess(evidence[0]["two_face_share"], SPLIT_SHOT_TWO_FACE_SHARE)
        self.assertEqual(enforce_shot_split_evidence(points, evidence), 30)
        self.assertEqual({p["mode"] for p in points}, {"fill"})

    def test_two_faces_too_big_for_a_pair_of_panels_fold(self):
        candidates = [_cands(2, width=SPLIT_SHOT_MAX_FACE_WIDTH + 0.2) for _ in range(30)]
        points = _shot(candidates, "split")
        evidence = shot_face_evidence(points, candidates, 1920)
        self.assertGreater(evidence[0]["median_face_width"], SPLIT_SHOT_MAX_FACE_WIDTH)
        self.assertGreaterEqual(evidence[0]["two_face_share"], SPLIT_SHOT_TWO_FACE_SHARE)
        self.assertEqual(enforce_shot_split_evidence(points, evidence), 30)

    def test_a_two_shot_shorter_than_its_own_minimum_fold(self):
        candidates = [_cands(2) for _ in range(6)]
        points = _shot(candidates, "split")
        evidence = shot_face_evidence(points, candidates, 1920)
        self.assertLess(evidence[0]["seconds"], SPLIT_SHOT_MIN_SEC)
        self.assertEqual(enforce_shot_split_evidence(points, evidence), 6)

    def test_a_sustained_two_shot_is_left_alone(self):
        candidates = [_cands(2) for _ in range(self.TWO)]
        points = _shot(candidates, "split")
        self.assertEqual(enforce_shot_split_evidence(points, shot_face_evidence(points, candidates, 1920)), 0)


if __name__ == "__main__":
    unittest.main()
