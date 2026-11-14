import copy
import unittest

from local_worker.layout_decision import LayoutPolicy, WEIGHTS, anchor_solo_subjects, finalize_layouts, foreground_humans
from local_worker.speaker_priority import TalkWindow


class LayoutDecisionTests(unittest.TestCase):
    def decide(self, people, *, mode="auto", points=None, talks=None, width=1920, height=1080, behaviors=None):
        points = points or [{"t": i * .2, "x": .5, "y": .32, "x2": .5, "y2": .32, "cut": i == 0}
                            for i in range(10)]
        samples = [{"t": p["t"], "people": copy.deepcopy(people)} for p in points]
        report = finalize_layouts(points, samples, width, height, talks or {}, None, mode, behaviors=behaviors)
        return points, report

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_centered_confident_face_can_fill(self):
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99}])
        self.assertEqual(points[0]["mode"], "fill")
        self.assertTrue(report[0]["candidates"]["fill"]["eligible"])

    def test_uncertain_and_empty_detections_preserve_source(self):
        for people in ([], [{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .2}]):
            points, report = self.decide(people)
            self.assertEqual({p["mode"] for p in points}, {"fit"})
            self.assertIn("low_tracking_confidence", report[0]["reason"])

    def test_important_object_retained_with_face(self):
        people = [{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99},
                  {"key": -1, "kind": "object", "box": [80, 500, 300, 300], "confidence": .95}]
        points, report = self.decide(people)
        self.assertEqual(points[0]["mode"], "fit")
        self.assertIn("required_content_outside_crop", report[0]["reason"])

    def test_manual_split_never_duplicates_one_subject(self):
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99}], mode="split")
        self.assertEqual(points[0]["mode"], "fit")
        self.assertEqual(report[0]["reason"], "unsafe_manual_split")

    def test_two_distinct_subjects_use_two_distinct_panels(self):
        points = [{"t": i * .2, "x": .2, "y": .3, "x2": .8, "y2": .3, "cut": i == 0} for i in range(10)]
        people = [{"key": 0, "kind": "face", "box": [320, 230, 120, 160], "confidence": .99},
                  {"key": 1, "kind": "face", "box": [1480, 230, 120, 160], "confidence": .99}]
        points, report = self.decide(people, points=points)
        self.assertEqual(points[0]["mode"], "split")
        self.assertTrue(report[0]["candidates"]["split"]["eligible"])

    def test_source_graphics_force_preservation_for_group_shots(self):
        points = [{"t": i * .2, "x": .5, "y": .32, "hasBottomGraphic": True, "cut": i == 0} for i in range(10)]
        people = [{"key": 0, "kind": "face", "box": [320, 230, 120, 160], "confidence": .99},
                  {"key": 1, "kind": "face", "box": [1480, 230, 120, 160], "confidence": .99}]
        points, report = self.decide(people, points=points)
        self.assertEqual(points[0]["mode"], "fit")
        self.assertIn("preserve_source_graphics", report[0]["reason"])

    def test_solo_speaker_closeup_trades_lower_third(self):
        points = [{"t": i * .2, "x": .5, "y": .32, "hasBottomGraphic": True, "cut": i == 0} for i in range(10)]
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99}], points=points)
        self.assertEqual(points[0]["mode"], "fill")
        self.assertTrue(report[0]["candidates"]["fill"]["eligible"])

    def test_manual_fill_and_fit_survive_other_passes(self):
        for mode in ("fill", "fit"):
            points, _ = self.decide([], mode=mode)
            self.assertEqual({p["mode"] for p in points}, {mode})

    def test_missing_active_speaker_is_not_permission_to_crop(self):
        talks = {i: TalkWindow(frozenset({99}), False, True) for i in range(5)}
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99}], talks=talks)
        self.assertEqual(points[0]["mode"], "fit")
        self.assertIn("speaker_not_observed", report[0]["reason"])

    def test_safety_overrides_hold_at_next_scene(self):
        points = [{"t": 0, "x": .5, "y": .32, "cut": True},
                  {"t": .2, "x": .5, "y": .32, "cut": True}]
        samples = [{"t": 0, "people": [{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .99}]},
                   {"t": .2, "people": []}]
        report = finalize_layouts(points, samples, 1920, 1080, {}, None, policy=LayoutPolicy(hold_seconds=5))
        self.assertEqual([r["layout"] for r in report], ["fill", "fit"])

    def test_mismatched_evidence_rejected(self):
        with self.assertRaises(ValueError):
            finalize_layouts([{"t": 0}], [], 1920, 1080, {}, None)
        self.assertEqual(finalize_layouts([], [], 1920, 1080, {}, None), [])

    def test_solo_person_medium_shot_chooses_fill_over_fit(self):
        # A medium-closeup talking head caught as `kind="person"` with wide shoulders (700px) must
        # still frame Full 9:16 (`fill`) around their head zone instead of falling back to blur+fit.
        points, report = self.decide([
            {"key": 0, "kind": "person", "box": [610, 120, 700, 920], "confidence": 0.92},
            {"key": -1, "kind": "object", "box": [100, 200, 180, 260], "label": "vase", "confidence": 0.88, "background_context": True},
        ])
        self.assertEqual(points[0]["mode"], "fill")
        self.assertTrue(report[0]["candidates"]["fill"]["eligible"])

    def test_mismatched_attribution_solo_monologue_still_closeup(self):
        talks = {i: TalkWindow(frozenset({99}), False, True) for i in range(5)}
        behaviors = {i * .2: "solo_monologue" for i in range(10)}
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 140, 180], "confidence": .99}],
                                     talks=talks, behaviors=behaviors)
        self.assertEqual(points[0]["mode"], "fill")

    def test_mismatched_attribution_listener_cutaway_preserves_source(self):
        talks = {i: TalkWindow(frozenset({99}), False, True) for i in range(5)}
        behaviors = {i * .2: "silent_listener_cutaway" for i in range(10)}
        points, report = self.decide([{"key": 0, "kind": "face", "box": [900, 270, 140, 180], "confidence": .99}],
                                     talks=talks, behaviors=behaviors)
        self.assertEqual(points[0]["mode"], "fit")
        self.assertIn("speaker_not_observed", report[0]["reason"])

    def test_background_bystander_does_not_block_closeup(self):
        points, report = self.decide([
            {"key": 0, "kind": "face", "box": [900, 270, 142, 180], "confidence": .91},
            {"key": 1, "kind": "face", "box": [120, 620, 57, 70], "confidence": .8},
        ])
        self.assertEqual(points[0]["mode"], "fill")

    def test_duplicate_detection_of_one_head_still_closeup(self):
        talks = {i: TalkWindow(frozenset({99}), False, True) for i in range(5)}
        behaviors = {i * .2: "solo_monologue" for i in range(10)}
        points, report = self.decide([
            {"key": 2, "kind": "face", "box": [910, 202, 110, 183], "confidence": .84},
            {"key": 11, "kind": "face", "box": [925, 212, 120, 180], "confidence": .61},
        ], talks=talks, behaviors=behaviors)
        self.assertEqual(points[0]["mode"], "fill")

    def test_two_heads_a_part_are_not_merged_into_one_subject(self):
        talks = {i: TalkWindow(frozenset({99}), False, True) for i in range(5)}
        behaviors = {i * .2: "solo_monologue" for i in range(10)}
        points, report = self.decide([
            {"key": 2, "kind": "face", "box": [500, 202, 110, 183], "confidence": .84},
            {"key": 11, "kind": "face", "box": [1006, 220, 135, 180], "confidence": .61},
        ], talks=talks, behaviors=behaviors)
        self.assertEqual(points[0]["mode"], "fit")

    def test_adjacent_distinct_heads_are_not_deduplicated(self):
        humans = [
            {"key": 0, "kind": "face", "box": [800, 200, 120, 180], "confidence": .93},
            {"key": 1, "kind": "face", "box": [920, 200, 120, 180], "confidence": .92},
        ]
        self.assertEqual(len(foreground_humans(humans)), 2)

    def test_overlap_and_ambiguous_floor_do_not_enable_solo_fallback(self):
        for talk in (TalkWindow(frozenset({0, 99}), True, True),
                     TalkWindow(frozenset({99}), False, True, ambiguous=True)):
            talks = {i: talk for i in range(5)}
            behaviors = {i * .2: "solo_monologue" for i in range(10)}
            points, _ = self.decide([
                {"key": 0, "kind": "face", "box": [900, 270, 140, 180], "confidence": .99},
            ], talks=talks, behaviors=behaviors)
            self.assertEqual(points[0]["mode"], "fit")

    def test_visible_smaller_talker_cannot_be_replaced_by_larger_listener(self):
        talks = {i: TalkWindow(frozenset({1}), False, True) for i in range(5)}
        behaviors = {i * .2: "solo_monologue" for i in range(10)}
        points, _ = self.decide([
            {"key": 0, "kind": "face", "box": [900, 270, 220, 260], "confidence": .99},
            {"key": 1, "kind": "face", "box": [200, 270, 130, 180], "confidence": .93},
        ], talks=talks, behaviors=behaviors)
        self.assertEqual(points[0]["mode"], "fit")

    def test_static_background_heads_do_not_block_confessional_closeup(self):
        points = [{"t": i * .2, "x": .5, "y": .32, "hasBottomGraphic": True,
                   "cut": i == 0} for i in range(12)]
        samples = [{"t": point["t"], "people": [
            {"key": 0, "kind": "face", "box": [850 + i * 6, 220, 200, 250], "confidence": .93},
            {"key": 1, "kind": "face", "box": [250, 220, 140, 180], "confidence": .72},
        ]} for i, point in enumerate(points)]
        original = copy.deepcopy(samples)
        report = finalize_layouts(points, samples, 1920, 1080, {}, None)
        self.assertEqual(report[0]["layout"], "fill")
        self.assertEqual(samples, original)

    def test_equal_size_still_listener_preserves_group(self):
        points = [{"t": i * .2, "x": .5, "y": .32, "hasBottomGraphic": True,
                   "cut": i == 0} for i in range(12)]
        samples = [{"t": point["t"], "people": [
            {"key": 0, "kind": "face", "box": [850 + i * 6, 220, 200, 250], "confidence": .93},
            {"key": 1, "kind": "face", "box": [250, 220, 200, 250], "confidence": .93},
        ]} for i, point in enumerate(points)]
        report = finalize_layouts(points, samples, 1920, 1080, {}, None)
        self.assertEqual(report[0]["layout"], "fit")

    def test_group_reactions_preserved_when_one_person_holds_floor(self):
        talks = {i: TalkWindow(frozenset({1}), False, True) for i in range(5)}
        points, _ = self.decide([
            {"key": i, "kind": "face", "box": [200 + i * 650, 270, 140, 180], "confidence": .93}
            for i in range(3)
        ], talks=talks)
        self.assertEqual(points[0]["mode"], "fit")

    def test_finalization_preserves_delivered_camera_anchors(self):
        points = [{"t": i * .2, "x": .48, "y": .36, "cut": i == 0} for i in range(10)]
        points, _ = self.decide([
            {"key": 0, "kind": "face", "box": [900, 270, 140, 180], "confidence": .99},
        ], points=points)
        self.assertEqual({(p["x"], p["y"]) for p in points}, {(.48, .36)})

    def test_solo_anchors_interpolate_detection_gap_without_crossing_cut(self):
        points = [{"t": i * .2, "x": .2, "y": .32, "cut": i in (0, 5)} for i in range(10)]
        samples = [{"t": point["t"], "people": [] if i == 2 else [
            {"key": 0, "kind": "face", "box": [850 if i < 5 else 1300, 220, 200, 250],
             "confidence": .93},
        ]} for i, point in enumerate(points)]
        anchor_solo_subjects(points, samples, {}, None, 1920, 1080, LayoutPolicy(), {})
        self.assertAlmostEqual(points[2]["x"], 950 / 1920)
        self.assertAlmostEqual(points[4]["x"], 950 / 1920)
        self.assertAlmostEqual(points[5]["x"], 1400 / 1920)

    def test_wide_establishing_shot_stays_fit(self):
        points, report = self.decide([{"key": 0, "kind": "person", "box": [860, 300, 200, 500], "confidence": .85}])
        self.assertEqual(points[0]["mode"], "fit")

    def test_scene_sized_vehicle_is_context_not_subject(self):
        track = [{"t": i * .2, "x": 770 / 1920, "y": .32, "cut": i == 0} for i in range(10)]
        points, report = self.decide([
            {"key": 0, "kind": "face", "box": [700, 270, 140, 180], "confidence": .9},
            {"key": -1, "kind": "object", "box": [960, 200, 900, 800], "label": "car", "confidence": .7},
        ], points=track)
        self.assertEqual(points[0]["mode"], "fill")

    def test_recovers_to_fill_immediately_across_hard_cut_from_fit(self):
        points = [{"t": 0.0, "x": .5, "y": .32, "cut": True},
                  {"t": 0.5, "x": .5, "y": .32, "cut": True}]
        samples = [
            {"t": 0.0, "people": []},
            {"t": 0.5, "people": [{"key": 0, "kind": "face", "box": [900, 270, 120, 180], "confidence": .95}]},
        ]
        report = finalize_layouts(points, samples, 1920, 1080, {}, None, policy=LayoutPolicy(hold_seconds=3.0))
        self.assertEqual([r["layout"] for r in report], ["fit", "fill"])


if __name__ == "__main__":
    unittest.main()
