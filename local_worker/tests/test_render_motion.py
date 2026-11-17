"""Regression tests for the crop path: the export must move the frame the way the preview does.

The stutter this covers came from two places - an error-bounded keyframe thinning that made FFmpeg
hold one rectangle and then teleport, and a sampler that eased linearly and interpolated across
scene cuts. Both were silent: the video was watchable, the camera just never glided.
"""

from __future__ import annotations

import unittest

from local_worker.rendering.decision import TrackPoint, decision_from_payload
from local_worker.rendering.filters import (
    CUT_SNAP_SEC,
    Panel,
    crop_keyframes,
    quantize_rect,
    sample_track,
    sendcmd_file_text,
    smoothstep,
)

FPS = 30.0
PANEL = Panel("main", 1080, 1920, 1.0, 1.0, False)


def track_of(points: list[tuple[float, float, float]], cut_at: set[float] | None = None) -> list[TrackPoint]:
    # A cut is flagged on the first keyframe of the incoming shot, which is how the tracker writes it.
    cut_at = cut_at or set()
    return [
        TrackPoint(t=t, x=x, y=y, mode="fill", zoom=1.0, cut=t in cut_at) for t, x, y in points
    ]


def decision_of(track: list[TrackPoint], start: float = 0.0, end: float = 4.0, **overrides) -> object:
    payload = {
        "source_path": "source.mp4",
        "source_width": 1920,
        "source_height": 1080,
        "source_fps": 30.0,
        "start": start,
        "end": end,
        "layout": "fill",
        "track": [{"t": p.t, "x": p.x, "y": p.y, "cut": p.cut} for p in track],
        "words": [],
        "hook": "",
        "style": {},
        "zoomPunch": False,
    }
    # The cut flag marks the first keyframe of the incoming shot, as the tracker writes it.
    payload.update(overrides)
    return decision_from_payload(payload)


class SampleTrackTest(unittest.TestCase):
    def test_rejected_manual_split_keeps_provenance_between_samples(self):
        from local_worker.rendering.filters import layout_of
        points = [{"t": t, "x": .3, "y": .4, "x2": .7, "y2": .4, "mode": "fit",
                   "trackingConfidence": .3,
                   "layoutDecision": {"scene": 0, "reason": "unsafe_manual_split"}}
                  for t in (0, 1)]
        decision = decision_of([], layout="split")
        decision.track = [TrackPoint.from_payload(point) for point in points]
        self.assertEqual(layout_of(decision, .5), "fit")
        self.assertEqual(sample_track(decision.track, .5).tracking_confidence, .3)

    def setUp(self) -> None:
        self.track = track_of([(0.0, 0.30, 0.40), (1.0, 0.70, 0.40), (2.0, 0.30, 0.40)])

    def test_eases_between_points_like_the_canvas(self):
        mid = sample_track(self.track, 0.5)
        self.assertAlmostEqual(mid.x, 0.30 + 0.40 * smoothstep(0.5), places=12)
        self.assertAlmostEqual(mid.x, 0.5, places=12)  # smoothstep(0.5) is exactly one half

    def test_velocity_is_zero_at_each_measured_point(self):
        """The canvas settles into each anchor instead of cornering through it at speed."""
        before = sample_track(self.track, 1.0 - 1 / 30)
        at = sample_track(self.track, 1.0)
        after = sample_track(self.track, 1.0 + 1 / 30)
        linear_step = 0.40 / 30  # what plain lerp would move in one frame
        self.assertLess(abs(after.x - at.x), linear_step / 3)
        self.assertLess(abs(at.x - before.x), linear_step / 3)

    def test_holds_the_incoming_shot_across_a_cut(self):
        track = track_of([(0.0, 0.2, 0.4), (1.0, 0.8, 0.4)], cut_at={1.0})
        self.assertAlmostEqual(sample_track(track, 0.97).x, 0.2, places=12)
        self.assertAlmostEqual(sample_track(track, 0.99).x, 0.8, places=12)
        self.assertAlmostEqual(CUT_SNAP_SEC, 0.02, places=12)

    def test_never_blends_two_shots_at_a_cut(self):
        track = track_of([(0.0, 0.2, 0.4), (1.0, 0.8, 0.4)], cut_at={1.0})
        values = [sample_track(track, t / 30).x for t in range(31)]
        self.assertNotIn(True, [0.2 < v < 0.8 for v in values])

    def test_missing_secondary_anchor_follows_the_primary(self):
        track = [TrackPoint(t=0.0, x=0.2, y=0.4), TrackPoint(t=1.0, x=0.6, y=0.4)]
        point = sample_track(track, 0.5)
        self.assertAlmostEqual(point.x2, point.x, places=12)
        self.assertAlmostEqual(point.y2, point.y, places=12)


class CropKeyframesTest(unittest.TestCase):
    # The tracker's own density: ~7.5 anchors a second, spanning the whole 4 s window.
    DRIFT = [(i / 7.5, 0.30 + i * 0.01, 0.40) for i in range(31)]

    def test_a_steady_drift_gets_a_command_every_frame_it_moves(self):
        """The old thinning kept a point only every 0.5 s, so the framing sat and then jumped."""
        frames = crop_keyframes(decision_of(track_of(self.DRIFT)), PANEL, FPS, 1.0)
        self.assertGreater(len(frames), 100)
        gaps = [b.t - a.t for a, b in zip(frames, frames[1:])]
        self.assertLessEqual(max(gaps), 2 / FPS + 1e-9)

    def test_consecutive_frames_move_by_a_fraction_of_a_crop(self):
        frames = crop_keyframes(decision_of(track_of(self.DRIFT)), PANEL, FPS, 1.0)
        steps = [abs(b.x - a.x) for a, b in zip(frames, frames[1:])]
        self.assertLess(max(steps), 0.02 * 1920)

    def test_a_stationary_subject_does_not_write_one_command_per_frame(self):
        """Density is bounded by the pixels, so a locked shot stays a small file."""
        track = track_of([(0.0, 0.5, 0.4), (4.0, 0.5, 0.4)])
        frames = crop_keyframes(decision_of(track), PANEL, FPS, 1.0)
        self.assertEqual(len(frames), 1)

    def test_command_times_are_relative_to_the_window(self):
        track = track_of([(10.0, 0.4, 0.4), (14.0, 0.6, 0.4)])
        frames = crop_keyframes(decision_of(track, start=10.0, end=14.0), PANEL, FPS, 1.0)
        self.assertAlmostEqual(frames[0].t, 0.0, places=9)
        self.assertLess(frames[-1].t, 4.0)

    def test_a_cut_appears_as_one_command_and_not_a_path(self):
        track = track_of([(0.0, 0.2, 0.4), (2.0, 0.8, 0.4), (4.0, 0.8, 0.4)], cut_at={2.0})
        frames = crop_keyframes(decision_of(track), PANEL, FPS, 1.0)
        crossing = [f for f in frames if 1.5 < f.t < 2.5]
        self.assertEqual(len(crossing), 1)
        self.assertGreater(crossing[0].x, frames[0].x)

    def test_quantization_keeps_the_rectangle_inside_the_source_and_even(self):
        w, h, x, y = quantize_rect(1080.6, 1920.7, -12.4, 1080.0, 1920, 1080)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x + w, 1920)
        self.assertLessEqual(y + h, 1080)

    def test_sendcmd_file_carries_a_time_on_every_command(self):
        track = track_of([(0.0, 0.4, 0.4), (4.0, 0.6, 0.4)])
        frames = crop_keyframes(decision_of(track), PANEL, FPS, 1.0)
        lines = sendcmd_file_text(frames, 1920, 1080).strip().splitlines()
        self.assertEqual(len(lines), len(frames))
        for line in lines:
            head = line.split(" ", 1)[0]
            float(head)
            self.assertEqual(line.count("crop"), 4)


if __name__ == "__main__":
    unittest.main()
