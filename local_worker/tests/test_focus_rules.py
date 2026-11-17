"""The gimbal focus rules: who owns the camera's attention, and how the attention travels.

Locks the rule ladder (active talker > presumed host > reaction hold > object > tracker), the
smoothstep handoff glide, gaze lead-room and the calm-shot breathing drift, so a tune in one
place cannot quietly turn the camera back into a servo that chases the biggest face.
"""
import os
import re
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker import clipforge_engine as engine
from local_worker.focus_rules import (
    BREATH_AMPLITUDE,
    BREATH_RAMP_SEC,
    FOCUS_HANDOFF_SEC,
    GAZE_LEAD_ROOM,
    OBJECT_DEMO_BEHAVIOR,
    REACTION_HOLD_SEC,
    RULE_ACTIVE_TALKER,
    RULE_OBJECT_SUBJECT,
    RULE_PRESUMED_MAIN,
    RULE_REACTION_HOLD,
    RULE_SHARED_FLOOR,
    RULE_TRACKER_HOLD,
    apply_gimbal_breathing,
    direct_speaker_focus,
    focus_aim,
    focus_subject,
)
from local_worker.speaker_priority import TalkWindow, WINDOW_SECONDS

W, H = 1920, 1080


def face(cx: float, cy: float = 0.35, size: float = 200.0) -> list[float]:
    return [cx * W - size / 2.0, cy * H - size / 2.0, size, size]


def person(key: int, cx: float, cy: float = 0.35, size: float = 200.0, **extra) -> dict:
    return {"key": key, "box": face(cx, cy, size), "kind": "face", **extra}


def obj(cx: float, cy: float = 0.6) -> dict:
    return {"key": -1, "object_id": "object:0", "box": face(cx, cy, 240.0),
            "kind": "object", "label": "laptop"}


def sample(t: float, people: list[dict]) -> dict:
    return {"t": t, "people": people}


def point(t: float, cx: float, cy: float = 0.35, mode: str = "fill", **extra) -> dict:
    return {"t": t, "x": cx, "y": cy, "mode": mode, "zoom": 1.0, **extra}


def talking(*keys) -> TalkWindow:
    return TalkWindow(frozenset(keys), len(keys) > 1, True)


def silent(loud=False, ambiguous=False) -> TalkWindow:
    return TalkWindow(frozenset(), False, loud, ambiguous=ambiguous)


def windows(verdicts: list[TalkWindow]) -> dict[int, TalkWindow]:
    return dict(enumerate(verdicts))


class RuleLadderTest(unittest.TestCase):
    def test_the_active_talker_owns_the_frame_even_when_the_host_is_bigger(self):
        s = sample(0.0, [person(0, 0.5, size=260.0), person(1, 0.25)])
        subject, rule = focus_subject(s, talking(1), 0, frozenset(), -1e9, 0.0)
        self.assertEqual(rule, RULE_ACTIVE_TALKER)
        self.assertEqual(subject["key"], 1)

    def test_a_shared_floor_is_a_group_shot_not_a_single_focus(self):
        s = sample(0.0, [person(0, 0.3), person(1, 0.7)])
        subject, rule = focus_subject(s, talking(0, 1), 0, frozenset(), -1e9, 0.0)
        self.assertIsNone(subject)
        self.assertEqual(rule, RULE_SHARED_FLOOR)

    def test_loud_ambiguous_audio_presumes_the_host_when_he_is_on_screen(self):
        s = sample(0.0, [person(0, 0.3), person(1, 0.7)])
        subject, rule = focus_subject(s, silent(loud=True, ambiguous=True), 0, frozenset(), -1e9, 0.0)
        self.assertEqual(rule, RULE_PRESUMED_MAIN)
        self.assertEqual(subject["key"], 0)

    def test_an_offscreen_voice_hands_the_frame_to_the_host(self):
        s = sample(0.0, [person(0, 0.5)])
        subject, rule = focus_subject(s, talking(9), 0, frozenset(), -1e9, 0.0)
        self.assertEqual(rule, RULE_PRESUMED_MAIN)
        self.assertEqual(subject["key"], 0)

    def test_the_reaction_hold_keeps_the_last_talker_through_a_pause(self):
        s = sample(1.2, [person(0, 0.7), person(1, 0.25)])
        subject, rule = focus_subject(s, silent(), None, frozenset({1}), 0.8, 1.2)
        self.assertEqual(rule, RULE_REACTION_HOLD)
        self.assertEqual(subject["key"], 1)

    def test_the_reaction_hold_expires_and_hands_back_to_the_tracker(self):
        s = sample(2.0, [person(1, 0.25)])
        subject, rule = focus_subject(s, silent(), None, frozenset({1}), 0.8, 2.0)
        self.assertIsNone(subject)
        self.assertEqual(rule, RULE_TRACKER_HOLD)

    def test_a_new_talker_overrides_the_reaction_hold_immediately(self):
        s = sample(1.0, [person(0, 0.7), person(1, 0.25)])
        subject, rule = focus_subject(s, talking(0), None, frozenset({1}), 0.8, 1.0)
        self.assertEqual(rule, RULE_ACTIVE_TALKER)
        self.assertEqual(subject["key"], 0)

    def test_the_reaction_hold_needs_the_last_talker_still_on_screen(self):
        s = sample(1.0, [person(0, 0.7)])
        subject, rule = focus_subject(s, silent(), None, frozenset({1}), 0.8, 1.0)
        self.assertIsNone(subject)
        self.assertEqual(rule, RULE_TRACKER_HOLD)

    def test_an_object_is_the_subject_only_when_no_human_is_in_shot(self):
        only_thing = sample(0.0, [obj(0.5)])
        subject, rule = focus_subject(only_thing, silent(), None, frozenset(), -1e9, 0.0)
        self.assertIsNone(subject)
        self.assertEqual(rule, RULE_OBJECT_SUBJECT)
        with_human = sample(0.0, [person(0, 0.3), obj(0.7)])
        subject, rule = focus_subject(with_human, silent(), None, frozenset(), -1e9, 0.0)
        self.assertIsNone(subject)
        self.assertEqual(rule, RULE_TRACKER_HOLD)

    def test_the_hold_is_two_audio_windows(self):
        self.assertAlmostEqual(REACTION_HOLD_SEC, 2 * WINDOW_SECONDS, places=12)


class HandoffGlideTest(unittest.TestCase):
    """A floor change inside a shot is a deliberate pan: eased at both ends, never a teleport."""

    def setup(self, cut_at: float | None = None):
        people = [person(0, 0.25), person(1, 0.75)]
        times = [round(i * 0.1, 3) for i in range(31)]
        samples = [sample(t, people) for t in times]
        points = [point(t, 0.25, cut=(t == cut_at)) for t in times]
        # A holds the floor for 2 s (windows 0-4), then B takes it.
        talks = windows([talking(0)] * 5 + [talking(1)] * 11)
        return points, samples, talks

    def run_director(self, points, samples, talks, **kw):
        return direct_speaker_focus(points, samples, talks, 0, W, H, **kw)

    def test_the_glide_starts_where_the_camera_is_and_lands_on_the_new_talker(self):
        points, samples, talks = self.setup()
        report = self.run_director(points, samples, talks)
        self.assertEqual(report["handoffs"], 1)
        self.assertEqual(points[20]["x"], 0.25)  # the handoff opens on the outgoing framing
        self.assertAlmostEqual(points[-1]["x"], 0.75, delta=0.01)
        self.assertAlmostEqual(points[-1]["y"], 0.35, delta=0.01)

    def test_the_glide_eases_in_and_out_with_bounded_steps(self):
        points, samples, talks = self.setup()
        self.run_director(points, samples, talks)
        xs = [p["x"] for p in points[20:27]]
        deltas = [b - a for a, b in zip(xs, xs[1:])]
        self.assertTrue(all(d >= -1e-9 for d in deltas), "the pan must not reverse")
        self.assertLess(deltas[0], deltas[2], "ease-in: the pan accelerates out of the hold")
        self.assertLess(deltas[-1], deltas[2], "ease-out: the pan settles onto the new talker")
        worst = max(abs(b["x"] - a["x"]) for a, b in zip(points, points[1:]))
        self.assertLess(worst, 0.5 * 1.5 / FOCUS_HANDOFF_SEC * 0.1 + 0.01)
        self.assertGreater(worst, 0.0)

    def test_a_cut_snaps_because_the_edit_moved_the_frame_not_the_camera(self):
        points, samples, talks = self.setup(cut_at=2.0)
        report = self.run_director(points, samples, talks)
        self.assertEqual(report["handoffs"], 0)
        self.assertAlmostEqual(points[20]["x"], 0.75, delta=0.01)

    def test_a_framing_that_already_shows_the_talker_is_left_alone(self):
        people = [person(0, 0.25)]
        times = [round(i * 0.1, 3) for i in range(11)]
        samples = [sample(t, people) for t in times]
        points = [point(t, 0.25) for t in times]
        report = self.run_director(points, samples, windows([talking(0)] * 3))
        self.assertEqual(report["handoffs"], 0)
        self.assertTrue(all(p["x"] == 0.25 for p in points))
        self.assertEqual(report["rules"][RULE_ACTIVE_TALKER], 11)

    def test_the_object_demonstration_union_crop_is_not_pulled_back_to_the_person(self):
        points, samples, talks = self.setup()
        for p in points:
            p["x"] = 0.4  # a union aim between the talker and the thing they are showing
        behaviors = {s["t"]: OBJECT_DEMO_BEHAVIOR for s in samples}
        self.run_director(points, samples, talks, behaviors=behaviors)
        self.assertTrue(all(p["x"] == 0.4 for p in points))

    def test_split_panels_are_left_to_the_layout_pass(self):
        points, samples, talks = self.setup()
        for p in points:
            p.update(mode="split", x2=0.75, y2=0.35)
        self.run_director(points, samples, talks)
        self.assertTrue(all(p["x"] == 0.25 and p["x2"] == 0.75 for p in points))

    def test_every_sample_reports_the_rule_that_named_its_subject(self):
        points, samples, talks = self.setup()
        report = self.run_director(points, samples, talks)
        self.assertEqual(sum(report["rules"].values()), len(points))
        self.assertTrue(all(p.get("focusRule") for p in points))


class ReactionHoldTest(unittest.TestCase):
    def test_silence_stays_with_the_last_talker_then_hands_back(self):
        people = [person(0, 0.75), person(1, 0.25)]
        times = [round(i * 0.1, 3) for i in range(21)]
        samples = [sample(t, people) for t in times]
        # While A talks the camera is on him; at the pause the tracker swings to the bigger
        # listener, and the reaction hold is what pulls attention back to A.
        points = [point(t, 0.25 if t < 0.8 else 0.75) for t in times]
        talks = windows([talking(1)] * 2 + [silent()] * 4)
        report = direct_speaker_focus(points, samples, talks, None, W, H)
        # A's last window ends at 0.8 s; the hold covers samples up to 1.6 s.
        self.assertEqual(report["rules"].get(RULE_REACTION_HOLD), 9)
        self.assertEqual(report["rules"].get(RULE_TRACKER_HOLD), 4)
        self.assertLess(points[16]["x"], points[8]["x"], "the camera glides back toward the last talker")
        self.assertAlmostEqual(points[16]["x"], 0.25, delta=0.06)
        self.assertEqual(points[17]["x"], 0.75, "once the hold expires the tracker's aim stands again")


class GazeLeadRoomTest(unittest.TestCase):
    def test_the_aim_leads_into_the_gaze_by_the_solo_law(self):
        p = person(0, 0.5, yaw_bias=1.0)
        x, _ = focus_aim(p, W, H)
        self.assertAlmostEqual(x, 0.5 + GAZE_LEAD_ROOM, places=6)
        p["yaw_bias"] = -1.0
        x, _ = focus_aim(p, W, H)
        self.assertAlmostEqual(x, 0.5 - GAZE_LEAD_ROOM, places=6)

    def test_the_lead_room_matches_the_solo_closeup_constant(self):
        source = os.path.join(os.path.dirname(__file__), "..", "clipforge_engine.py")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        lead = float(re.search(r"scx \+ yaw \* ([\d.]+)", text).group(1))
        self.assertEqual(lead, GAZE_LEAD_ROOM)

    def test_the_union_crop_behavior_name_matches_the_engine(self):
        self.assertEqual(OBJECT_DEMO_BEHAVIOR, engine.BEHAVIOR_OBJECT_DEMO)


class BreathingDriftTest(unittest.TestCase):
    def track(self, style_points=None):
        times = [round(i / 7.5, 3) for i in range(76)]  # ~10 s at the analysis sample rate
        return style_points or [point(t, 0.5) for t in times]

    def test_a_calm_held_shot_drifts_within_a_few_pixels_and_never_jumps(self):
        points = self.track()
        out = apply_gimbal_breathing(points, "static_single_shot")
        dx = [p["x"] - 0.5 for p in out]
        self.assertTrue(any(abs(d) > 1e-4 for d in dx), "the drift must actually move")
        self.assertLessEqual(max(abs(d) for d in dx), BREATH_AMPLITUDE + 1e-4)
        worst_speed = max(abs(b["x"] - a["x"]) / (b["t"] - a["t"]) for a, b in zip(out, out[1:]))
        self.assertLess(worst_speed, 0.01, "drift is two orders below the pursuit cap")

    def test_the_drift_is_zero_at_a_cut_and_ramps_in(self):
        points = self.track()
        out = apply_gimbal_breathing(points, "long_take")
        self.assertEqual(out[0]["x"], 0.5)
        self.assertEqual(out[0]["y"], 0.35)
        early = max(abs(p["x"] - 0.5) for p in out[:4])
        self.assertLess(early, BREATH_AMPLITUDE / 2)
        cut = self.track()
        cut[38]["cut"] = True
        out = apply_gimbal_breathing(cut, "long_take")
        self.assertEqual(out[38]["x"], 0.5, "a cut opens without drift")
        self.assertLess(abs(out[39]["x"] - 0.5), BREATH_AMPLITUDE / 2)

    def test_fast_cut_footage_and_non_fill_layouts_do_not_drift(self):
        points = self.track()
        self.assertIs(apply_gimbal_breathing(points, "rapid_multicam"), points)
        self.assertIs(apply_gimbal_breathing(points, "dialogue_cuts"), points)
        mixed = self.track()
        for p in mixed[::2]:
            p["mode"] = "split"
            p["x2"], p["y2"] = 0.7, 0.35
        out = apply_gimbal_breathing(mixed, "static_single_shot")
        for a, b in zip(mixed, out):
            if a["mode"] == "split":
                self.assertEqual((a["x"], a["y"]), (b["x"], b["y"]))

    def test_the_drift_stays_inside_the_track_bounds(self):
        points = self.track([point(round(i / 7.5, 3), 0.95, 0.92) for i in range(76)])
        out = apply_gimbal_breathing(points, "static_single_shot")
        self.assertTrue(all(0.05 <= p["x"] <= 0.95 and 0.08 <= p["y"] <= 0.92 for p in out))


if __name__ == "__main__":
    unittest.main()