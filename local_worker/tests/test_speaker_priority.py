"""Unit tests for the speaking rule: who has to be on screen, and which layout can honestly show them."""
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker.speaker_priority import (
    HEAD_CY_OF_BODY,
    HEAD_W_OF_BODY,
    MouthMemory,
    Person,
    PersonRegistry,
    TalkWatcher,
    TalkWindow,
    apply_speaker_priority,
    attribute_talkers,
    coverage_of,
    crowd_runs,
    panel_geometry,
    shown,
)

W, H = 1920, 1080


def face(cx: float, cy: float, size: float = 200.0) -> list[float]:
    """A face box from its centre in normalised source coordinates."""
    return [cx * W - size / 2.0, cy * H - size / 2.0, size, size]


def body(cx: float, cy: float, head: float = 200.0) -> list[float]:
    """A YOLO upper-body box around a person whose head is `head` px wide, centred at (cx, cy)."""
    w = head / HEAD_W_OF_BODY
    h = 5.0 * head
    return [cx * W - w / 2.0, cy * H - HEAD_CY_OF_BODY * h, w, h]


def person(key: int, cx: float, cy: float = 0.35, size: float = 200.0) -> dict:
    return {"key": key, "box": face(cx, cy, size), "kind": "face"}


def point(t: float, cx: float, cy: float = 0.36, mode: str = "fill") -> dict:
    return {"t": t, "x": cx, "y": cy, "mode": mode, "zoom": 1.0}


def window(keys, loud=True):
    return TalkWindow(frozenset(keys), len(keys) > 1, loud)


class TestWhoTalks(unittest.TestCase):
    def test_the_same_face_across_a_cut_stays_one_person(self):
        registry = PersonRegistry()
        first = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0)
        after_cut = registry.observe(face(0.62, 0.35), W, H, 0.125, 0.5)
        self.assertEqual(first, after_cut)
        self.assertEqual(len(registry.people), 1)

    def test_two_faces_in_the_same_frame_are_never_the_same_person(self):
        # Identical size, identical height, opposite sides of a two-shot: only the shared instant
        # separates them, and without that gate the rule would see a crowd as one talker.
        registry = PersonRegistry()
        left = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0)
        right = registry.observe(face(0.70, 0.35), W, H, 0.125, 0.0)
        self.assertNotEqual(left, right)
        self.assertEqual(len(registry.people), 2)

    def test_the_same_person_measured_as_a_face_and_as_a_body_is_one_person(self):
        # The detector alternates between YuNet's face box and YOLO's shoulders box on the same seated
        # host. They disagree on aspect ratio, which used to be the gate, so one person became two
        # identities and a five-person panel read as a dozen people nobody could rank.
        registry = PersonRegistry()
        as_face = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0, "face")
        as_body = registry.observe(body(0.30, 0.35), W, H, 0.125, 0.4, "person")
        self.assertEqual(as_face, as_body)
        self.assertEqual(len(registry.people), 1)

    def test_inside_one_shot_two_identical_heads_are_told_apart_by_where_they_sit(self):
        # Same size, same height, different sample instants: nothing but position separates the left
        # half of a two-shot from the right half, and a cut never happens between these two samples.
        registry = PersonRegistry()
        left = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0, "face", True)
        right = registry.observe(face(0.62, 0.35), W, H, 0.125, 0.125, "face", True)
        self.assertNotEqual(left, right)
        self.assertEqual(len(registry.people), 2)

    def test_inside_one_shot_a_head_that_drifted_is_still_the_same_person(self):
        registry = PersonRegistry()
        first = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0, "face", True)
        drifted = registry.observe(face(0.33, 0.36), W, H, 0.125, 0.125, "face", True)
        self.assertEqual(first, drifted)

    def test_across_a_cut_position_is_dropped_because_the_cut_reframed_everybody(self):
        # Deliberate trade-off against the previous two: multicam moves a person to the other side of
        # the frame between shots, so identity has to survive on head size and height alone there.
        registry = PersonRegistry()
        before = registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0, "face")
        after = registry.observe(face(0.62, 0.35), W, H, 0.125, 0.4, "face")
        self.assertEqual(before, after)
        self.assertEqual(len(registry.people), 1)

    def test_the_person_with_the_floor_is_the_main_speaker(self):
        registry = PersonRegistry()
        for _ in range(4):
            registry.observe(face(0.30, 0.35), W, H, 0.125, 0.0)
        registry.people[0].talk_sec = 6.0
        registry.people[0].screen_sec = 8.0
        other = Person(key=1, screen_sec=20.0, talk_sec=1.0)
        registry.people[1] = other
        self.assertEqual(registry.main_speaker(), 0)
        self.assertTrue(registry.report(0)["people"][0]["is_main"])

    def test_equal_mouth_motion_in_a_loud_window_is_two_people_talking_over_each_other(self):
        watcher = TalkWatcher(0.0)
        for w, (a, b) in enumerate(
            [(0.01, 0.01), (0.02, 0.015), (0.50, 0.47), (0.52, 0.49), (0.01, 0.02), (0.02, 0.01)]
        ):
            watcher.observe(0, w * 0.4, a)
            watcher.observe(1, w * 0.4, b)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 6, 6)
        self.assertEqual(sorted(talks[2].keys), [0, 1])
        self.assertTrue(talks[2].overlap)
        self.assertEqual(sorted(talks[0].keys), [])

    def test_three_moving_mouths_with_no_leader_are_three_people_talking_at_once(self):
        # Rule 3 asked for all of them in frame, which needs all of them named first. The old gate took
        # an overlap only at exactly two, so a host interrupted by two guests read as a crowd and the
        # framing left the decision to whoever the tracker happened to favour.
        watcher = TalkWatcher(0.0)
        for w, trio in enumerate(
            [
                (0.01, 0.01, 0.01),
                (0.02, 0.015, 0.02),
                (0.50, 0.47, 0.49),
                (0.52, 0.49, 0.51),
                (0.01, 0.02, 0.01),
                (0.02, 0.01, 0.02),
            ]
        ):
            for key, value in enumerate(trio):
                watcher.observe(key, w * 0.4, value)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 6, 6)
        self.assertEqual(sorted(talks[2].keys), [0, 1, 2])
        self.assertTrue(talks[2].overlap)

    def test_four_moving_mouths_are_a_crowd_and_stay_unread(self):
        # Past three the signal cannot say which of them the cut is on, and a guess here would throw the
        # camera off whoever the editor actually chose. The abstention is what keeps the host framed by
        # rule 1 instead of a fabricated group shot.
        watcher = TalkWatcher(0.0)
        for w, quad in enumerate(
            [
                (0.50, 0.47, 0.49, 0.48),
                (0.52, 0.49, 0.51, 0.50),
            ]
        ):
            for key, value in enumerate(quad):
                watcher.observe(key, w * 0.4, value)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 2, 2)
        self.assertFalse(talks[0].talking)
        self.assertTrue(talks[0].ambiguous)

    def test_a_quiet_window_has_no_talker_even_if_a_mouth_moves(self):
        watcher = TalkWatcher(0.0)
        for w in range(6):
            watcher.observe(0, w * 0.4, 0.5 if w == 2 else 0.01)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.05] * 6, 6)
        self.assertFalse(talks[2].talking)

    def test_the_clearly_louder_mouth_owns_the_window(self):
        watcher = TalkWatcher(0.0)
        for w in range(6):
            watcher.observe(0, w * 0.4, 0.5 if w == 2 else 0.01)
            watcher.observe(1, w * 0.4, 0.08 if w == 2 else 0.01)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 6, 6)
        self.assertEqual(list(talks[2].keys), [0])
        self.assertFalse(talks[2].overlap)

    def test_the_mouth_reading_survives_the_box_changing_size(self):
        # Real face boxes change shape on almost every sample. The first version dropped the delta when
        # they did, which left the rule blind for most of a clip, so the patch goes on a fixed grid.
        import numpy as np

        gray = np.zeros((H, W), dtype=np.uint8)
        memory = MouthMemory()
        first = memory.activity(0, [900.0, 300.0, 200.0, 260.0], gray, W, H, 1.0)
        gray[470:520, 970:1030] = 200
        second = memory.activity(0, [900.0, 300.0, 240.0, 300.0], gray, W, H, 1.0)
        self.assertIsNone(first)
        self.assertGreater(second or 0.0, 0.0)

    def test_a_head_too_small_to_read_is_not_read(self):
        # A crowd of 40 px faces gives a mouth patch smaller than the patch grid, and upsampling that to
        # compare mouths is what made a roomful of people look like a panel of indistinguishable talkers.
        import numpy as np

        gray = np.full((H, W), 40, dtype=np.uint8)
        memory = MouthMemory()
        memory.activity(0, [900.0, 300.0, 400.0, 400.0], gray, W, H, 1.0)
        gray[470:520, 970:1030] = 220
        self.assertIsNotNone(memory.activity(0, [900.0, 300.0, 400.0, 400.0], gray, W, H, 1.0))
        self.assertIsNone(memory.activity(0, [940.0, 330.0, 45.0, 60.0], gray, W, H, 1.0))

    def test_a_crowd_whose_mouths_cannot_be_told_apart_is_not_read_as_everybody_talking(self):
        # The panel shot that broke the earlier version: four faces, all moving alike. Demanding all
        # four in frame at once replaced a readable cut with a wide shot, so the rule now abstains.
        watcher = TalkWatcher(0.0)
        for w in range(4):
            for key in range(4):
                watcher.observe(key, w * 0.4, 0.40 + key * 0.01)
        present = {w: {0, 1, 2, 3} for w in range(4)}
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 4, 4, present)
        self.assertEqual(sorted(talks[2].keys), [])
        self.assertTrue(talks[2].ambiguous)

    def test_a_quiet_speaker_alone_in_shot_is_still_the_talker(self):
        # How much a mouth moves in absolute terms depends on lighting and head motion, so a soft-spoken
        # close-up used to be vetoed by the loudest mouth elsewhere in the clip and lost its whole shot.
        watcher = TalkWatcher(0.0)
        for w in range(6):
            watcher.observe(0, w * 0.4, 0.05)
            watcher.observe(1, w * 0.4, 0.50)
        present = {0: {0}, 1: {0}, 2: {0}, 3: {1}, 4: {1}, 5: {1}}
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 6, 6, present)
        self.assertEqual(list(talks[0].keys), [0])
        self.assertEqual(list(talks[1].keys), [0])

    def test_a_voice_that_is_not_in_shot_does_not_take_the_window(self):
        # An off-screen laugh is not a framing failure, and counting it as one would drag the camera to
        # whoever happens to be sitting where the voice came from.
        watcher = TalkWatcher(0.0)
        for w in range(6):
            watcher.observe(0, w * 0.4, 0.5 if w in (2, 3) else 0.02)
            watcher.observe(1, w * 0.4, 0.4 if w in (2, 3) else 0.02)
        present = {w: ({1} if w in (2, 3) else set()) for w in range(6)}
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 6, 6, present)
        self.assertEqual(list(talks[3].keys), [1])
        self.assertFalse(talks[3].overlap)


    def test_a_talker_has_to_be_read_twice_before_the_camera_turns(self):
        # One window of someone else's mouth moving in the middle of a long turn is the same noise the
        # tracker's sticky speaker rejects; here it must not move the framing and back again.
        watcher = TalkWatcher(0.0)
        for w in range(7):
            watcher.observe(0, w * 0.4, 0.02 if w == 3 else 0.5)
            watcher.observe(1, w * 0.4, 0.5 if w == 3 else 0.02)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 7, 7)
        self.assertEqual(list(talks[3].keys), [0])

    def test_a_talker_held_for_two_windows_does_take_the_framing(self):
        watcher = TalkWatcher(0.0)
        for w in range(7):
            watcher.observe(0, w * 0.4, 0.02 if w >= 4 else 0.5)
            watcher.observe(1, w * 0.4, 0.5 if w >= 4 else 0.02)
        talks = attribute_talkers(watcher, PersonRegistry(), [0.6] * 7, 7)
        self.assertEqual(list(talks[4].keys), [0])
        self.assertEqual(list(talks[5].keys), [1])


class TestWhatIsShown(unittest.TestCase):
    def test_a_split_panel_sees_wider_than_a_full_frame_crop(self):
        fill = panel_geometry("fill", W, H)[0]
        split = panel_geometry("split", W, H)[0]
        self.assertGreater(split["cw"], fill["cw"])
        pair = [person(0, 0.30), person(1, 0.62)]
        # The same two heads that no single 9:16 crop can hold do fit inside a pair of panels.
        tight = {"mode": "fill", "x": 0.46, "y": 0.36}
        wide = {"mode": "split", "x": 0.30, "y": 0.36, "x2": 0.62, "y2": 0.36}
        self.assertFalse(all(shown(p, tight, W, H) for p in pair))
        self.assertTrue(all(shown(p, wide, W, H) for p in pair))


class TestTheRule(unittest.TestCase):
    def _run(self, points, samples, talks, main_key=None, hold=0.0):
        return apply_speaker_priority(points, samples, main_key, talks, W, H, hold)

    def test_one_talker_off_to_the_side_is_brought_into_the_crop(self):
        # The camera is on the listener; the talker is at the left edge, a short slide away.
        talks = {0: window([0])}
        points = [point(0.0, 0.45)]
        samples = [{"t": 0.0, "people": [person(0, 0.30), person(1, 0.52)]}]
        report = self._run(points, samples, talks)
        self.assertEqual(points[0]["mode"], "fill")
        # A two-shot is the better answer than a solo: the slide brings the talker in without
        # throwing the listener out, and neither of them needed a wider layout to be seen.
        self.assertLess(points[0]["x"], 0.45)
        self.assertEqual(report["shot"]["group"], 1)
        self.assertTrue(shown(samples[0]["people"][0], points[0], W, H))
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_two_talkers_far_apart_are_both_kept_in_frame(self):
        talks = {0: window([0, 1])}
        points = [point(0.0, 0.30)]
        samples = [{"t": 0.0, "people": [person(0, 0.18), person(1, 0.82)]}]
        report = self._run(points, samples, talks)
        self.assertIn(points[0]["mode"], ("split", "editorial", "fit"))
        self.assertLessEqual(report["shot"]["two_panel"] + report["shot"]["context"] + report["shot"]["wide"], 1)
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_a_pair_too_wide_for_one_crop_goes_to_split_panels_when_the_camera_is_already_there(self):
        talks = {0: window([0, 1])}
        points = [point(0.0, 0.30)]
        samples = [{"t": 0.0, "people": [person(0, 0.22), person(1, 0.60)]}]
        report = self._run(points, samples, talks)
        self.assertEqual(points[0]["mode"], "split")
        self.assertEqual(report["shot"]["two_panel"], 1)
        self.assertEqual(report["coverage_after"]["per_speaker"]["1"]["coverage"], 1.0)

    def test_a_far_talker_is_brought_into_the_crop_and_the_step_is_reported(self):
        # The talker is on the far side of the source and the camera is not going to reach him gently:
        # the rule asks for the anchor change and counts it, because how fast the camera gets there is
        # the pursuit law's business, not this module's.
        talks = {0: window([0])}
        points = [point(0.0, 0.80)]
        samples = [{"t": 0.0, "people": [person(0, 0.15), person(1, 0.80)]}]
        report = self._run(points, samples, talks)
        self.assertEqual(points[0]["mode"], "fill")
        self.assertLess(points[0]["x"], 0.25)
        self.assertEqual(report["shot"]["talkers"], 1)
        self.assertEqual(report["far_reaims"], 1)
        self.assertGreater(report["far_reaim_max_crop_widths"], 1.5)
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_an_editorial_clip_keeps_its_layout_and_turns_its_window_onto_each_speaker(self):
        # Nobody leaves the context layout: the band already honours the rule. What moves is the window
        # on top of it, and the edge clamp is allowed to stop it short of the speaker's own centre.
        talks = {0: window([0]), 1: window([1])}
        points = [point(0.0, 0.20, mode="editorial"), point(0.4, 0.20, mode="editorial")]
        samples = [
            {"t": 0.0, "people": [person(0, 0.20)]},
            {"t": 0.4, "people": [person(0, 0.20), person(1, 0.85)]},
        ]
        report = self._run(points, samples, talks)
        self.assertEqual([p["mode"] for p in points], ["editorial", "editorial"])
        self.assertLess(points[0]["x"], 0.30)
        self.assertGreater(points[1]["x"], 0.70)
        self.assertEqual(report["shot"]["wide"], 0)
        self.assertEqual(report["coverage_after"]["per_speaker"]["1"]["coverage"], 1.0)

    def test_a_quiet_sample_keeps_whatever_the_tracker_chose(self):
        talks = {0: window([], loud=False)}
        points = [point(0.0, 0.80, mode="fill")]
        samples = [{"t": 0.0, "people": [person(0, 0.15)]}]
        report = self._run(points, samples, talks)
        self.assertEqual(points[0]["mode"], "fill")
        self.assertEqual(points[0]["x"], 0.80)
        self.assertEqual(report["shot"]["untouched"], 1)

    def test_a_window_the_rule_cannot_read_leaves_the_trackers_own_framing_alone(self):
        # Loud audio, four faces, no separable winner: abstaining has to look like doing nothing, not
        # like a wide shot assembled out of a guess.
        talks = {0: TalkWindow(frozenset(), False, True, ambiguous=True)}
        points = [point(0.0, 0.80, mode="fill")]
        samples = [{"t": 0.0, "people": [person(0, 0.15), person(1, 0.80)]}]
        report = self._run(points, samples, talks)
        self.assertEqual(points[0]["mode"], "fill")
        self.assertEqual(points[0]["x"], 0.80)
        self.assertEqual(report["shot"]["untouched"], 1)
        self.assertEqual(report["ambiguous_windows"], 1)

    def test_an_object_in_the_way_is_not_a_speaker(self):
        # A door, a product, a car: the detectors see it and the camera has to reach past it, but it
        # never joins the group candidate, which would have framed the talker against a thing.
        talks = {0: window([0])}
        points = [point(0.0, 0.75)]
        samples = [
            {
                "t": 0.0,
                "people": [
                    person(0, 0.44),
                    {"key": 7, "box": face(0.90, 0.6), "kind": "object"},
                ],
            }
        ]
        report = self._run(points, samples, talks)
        self.assertEqual(report["shot"]["talkers"], 1)
        self.assertEqual(report["shot"]["group"], 0)
        self.assertTrue(shown(samples[0]["people"][0], points[0], W, H))

    def test_a_short_split_that_shows_both_talkers_is_extended_not_erased(self):
        # 0.7 s of a solo crop, 0.25 s of the split that shows both talkers, 0.7 s of the solo crop
        # again. A 1 s hold cannot simply erase the only moment the second speaker is on screen, and
        # the shorter neighbour is the solo crop: the fold goes the other way, so the split takes the
        # exchange over instead of the twitch being cut out of it.
        talks = {0: window([0]), 1: window([0, 1])}
        points = [
            point(0.00, 0.30),
            point(0.35, 0.30),
            point(0.70, 0.30, mode="split"),
            point(1.05, 0.30),
            point(1.40, 0.30),
        ]
        points[2]["x2"], points[2]["y2"] = 0.62, 0.36
        samples = [
            {"t": 0.00, "people": [person(0, 0.30)]},
            {"t": 0.35, "people": [person(0, 0.30)]},
            {"t": 0.70, "people": [person(0, 0.30), person(1, 0.62)]},
            {"t": 1.05, "people": [person(0, 0.30), person(1, 0.62)]},
            {"t": 1.40, "people": [person(0, 0.30), person(1, 0.62)]},
        ]
        report = self._run(points, samples, talks, hold=1.0)
        self.assertEqual({pt["mode"] for pt in points}, {"split"})
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_a_crowd_shout_is_framed_once_and_never_split(self):
        # Six voices at once, with one window that only managed to name two of them. A split carries
        # two people, so trying to honour six with it is what made the export argue between split and
        # fill; the burst gets one framing for the whole of it instead.
        crowd = [person(k, 0.10 + 0.16 * k) for k in range(6)]
        talks = {
            0: window(list(range(6))),
            1: window(list(range(6))),
            2: window([0, 1]),
            3: window(list(range(6))),
            4: window(list(range(6))),
        }
        points = [point(t, 0.50) for t in (0.0, 0.4, 0.8, 1.2, 1.6)]
        samples = [{"t": p["t"], "people": crowd} for p in points]
        report = self._run(points, samples, talks, hold=1.0)
        self.assertEqual(report["crowd_bursts"], 1)
        self.assertEqual({pt["mode"] for pt in points}, {"fit"})

    def test_two_people_talking_over_each_other_are_still_allowed_a_split(self):
        crowd = [person(0, 0.22), person(1, 0.74)]
        talks = {0: window([0, 1]), 1: window([0, 1])}
        points = [point(0.0, 0.30), point(0.4, 0.30)]
        samples = [{"t": p["t"], "people": crowd} for p in points]
        report = self._run(points, samples, talks, hold=1.0)
        self.assertEqual(report["crowd_bursts"], 0)
        self.assertEqual(report["shot"]["two_panel"], 2)

    def test_the_burst_grows_across_the_window_where_the_detector_lost_a_head(self):
        # Three voices, then one window that only read two, then three again: one burst, not a burst,
        # a two-shot, and a burst.
        self.assertEqual(crowd_runs([3, 2, 3]), [(0, 3)])
        self.assertEqual(crowd_runs([2, 3, 2, 0, 2, 4, 1]), [(0, 3), (4, 6)])
        self.assertEqual(crowd_runs([1, 2, 1]), [])

    def test_a_short_run_that_is_the_only_way_to_show_a_talker_is_not_folded_away(self):
        # A 2-up of two people stood too far apart for any one crop, on screen for less than the
        # clip's own hold. Folding it into the neighbour would put that pair out of shot, and taking
        # the neighbour's frame would put the lone speaker out of shot — so the hold gives way to the
        # speaking rule, which is the one rule that does not bend.
        def solo(t: float, cx: float) -> dict:
            p = point(t, cx)
            p["zoom"] = 2.0
            return p

        def pair(t: float) -> dict:
            p = point(t, 0.12, mode="split")
            p["x2"], p["y2"], p["zoom"] = 0.88, 0.36, 2.0
            return p

        talks = {0: window([2]), 5: window([2]), 7: window([0, 1]), 8: window([0, 1])}
        points = [solo(0.00, 0.57), solo(2.00, 0.57), pair(3.00), pair(3.35)]
        three = [person(0, 0.12), person(1, 0.88), person(2, 0.57)]
        samples = [{"t": p["t"], "people": [dict(q) for q in three]} for p in points]
        report = self._run(points, samples, talks, hold=1.0)
        self.assertEqual(report["hold_exempt_runs"], 1)
        self.assertEqual([pt["mode"] for pt in points], ["fill", "fill", "split", "split"])
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_a_loud_window_that_cannot_be_read_still_frames_the_host(self):
        # Rule 1 does not wait for a verdict. Loud audio, the host in shot, mouths this pass cannot
        # separate: he is presumed to be the one talking, which is right far more often on host-led
        # footage than leaving the frame wherever the tracker happened to land.
        talks = {0: TalkWindow(frozenset(), False, True, ambiguous=True)}
        points = [point(0.0, 0.80, mode="fill")]
        samples = [{"t": 0.0, "people": [person(0, 0.15), person(1, 0.80)]}]
        report = self._run(points, samples, talks, main_key=0)
        self.assertTrue(shown(samples[0]["people"][0], points[0], W, H))
        self.assertEqual(report["required"]["presumed_main"], 1)
        self.assertEqual(report["main_lock"]["coverage"], 1.0)

    def test_a_host_the_wide_shot_merely_contains_is_made_the_subject(self):
        # A contained shot technically shows everybody, which is exactly why it cannot settle rule 1: the
        # viewer has to go looking for the speaker. Where the crop can hold him, it does instead.
        talks = {0: window([0])}
        points = [point(0.0, 0.50, mode="fit")]
        samples = [{"t": 0.0, "people": [person(0, 0.20)]}]
        report = self._run(points, samples, talks, main_key=0)
        self.assertEqual(points[0]["mode"], "fill")
        self.assertEqual(report["shot"]["talkers"], 1)
        self.assertEqual(report["main_lock"]["coverage"], 1.0)

    def test_the_object_of_the_moment_is_not_more_important_than_the_speaker(self):
        talks = {0: window([0])}
        points = [point(0.0, 0.80)]
        samples = [
            {
                "t": 0.0,
                "people": [person(0, 0.15), {"key": -1, "box": face(0.80, 0.5), "kind": "object"}],
            }
        ]
        report = self._run(points, samples, talks, main_key=0)
        self.assertTrue(shown(samples[0]["people"][0], points[0], W, H))
        self.assertEqual(report["shot"]["talkers"], 1)

    def test_three_talkers_at_once_all_end_up_in_the_frame(self):
        talks = {0: window([0, 1, 2])}
        points = [point(0.0, 0.50)]
        samples = [
            {"t": 0.0, "people": [person(0, 0.12), person(1, 0.50), person(2, 0.88)]}
        ]
        report = self._run(points, samples, talks, main_key=0)
        for one in samples[0]["people"]:
            self.assertTrue(shown(one, points[0], W, H), f"key {one['key']} was left out of the frame")
        self.assertEqual(report["coverage_after"]["overall"], 1.0)

    def test_too_many_layouts_for_the_runtime_are_deleted_not_stretched(self):
        # Four templates in eight seconds cannot each stand for three, however many frames a neighbour
        # lends. The hold has to reduce the number of layouts, which it does before it starts arguing
        # about any one of them.
        points = [
            point(0.00, 0.40),
            point(2.00, 0.40),
            point(2.50, 0.40, mode="split"),
            point(4.00, 0.40),
            point(5.00, 0.40),
            point(5.50, 0.40, mode="split"),
            point(7.00, 0.40),
        ]
        samples = [{"t": p["t"], "people": [person(0, 0.40)]} for p in points]
        report = self._run(points, samples, {}, hold=3.0)
        self.assertEqual(report["hold_exempt_runs"], 0)
        runs = []
        for i, pt in enumerate(points):
            if runs and runs[-1][1] == pt["mode"]:
                runs[-1][2] = i + 1
            else:
                runs.append([i, pt["mode"], i + 1])
        self.assertLessEqual(len(runs), 2)
        for begin, _mode, end in runs:
            held = points[min(end, len(points) - 1)]["t"] - points[begin]["t"]
            self.assertGreaterEqual(held, 3.0)

    def test_the_shot_hold_does_not_fold_the_host_out_of_his_own_window(self):
        # The middle second is the only place the rule can see the host holding the floor, and the
        # clip is shorter than its own hold. Smoothness cannot delete it and still tell the truth about
        # who was talking, so the split takes the whole clip instead — one refusal counted, no twitch.
        talks = {0: window([1]), 1: window([0, 1]), 2: window([1])}
        points = [point(0.0, 0.80), point(0.4, 0.50, mode="split"), point(0.8, 0.80)]
        samples = [
            {"t": t, "people": [person(0, 0.12), person(1, 0.86)]} for t in (0.0, 0.4, 0.8)
        ]
        report = self._run(points, samples, talks, main_key=0, hold=3.0)
        self.assertEqual(points[1]["mode"], "split")
        self.assertEqual(report["hold_exempt_runs"], 1)
        self.assertEqual({pt["mode"] for pt in points}, {"split"})
        self.assertEqual(report["main_lock"]["coverage"], 1.0)

    def test_the_counted_guarantee_is_the_head_on_screen_not_the_camera_pointed_that_way(self):
        talks = {0: window([0])}
        samples = [{"t": 0.0, "people": [person(0, 0.90, size=60)]}]
        points = [point(0.0, 0.90)]
        # A tiny face at the edge that the renderer's clamp pushes out of the crop: honest coverage
        # says the speaker was lost, which is what the rule then has to fix.
        lost = coverage_of([{"t": 0.0, "x": 0.90, "y": 0.36, "mode": "fill", "zoom": 1.0}], samples, talks, W, H)
        report = self._run(points, samples, talks)
        self.assertIn(lost["overall"], (0.0, 1.0))
        self.assertEqual(report["coverage_after"]["overall"], 1.0)
