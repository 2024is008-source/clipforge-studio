"""Unit tests for OpenShorts Reframe Engine v2 (SpeakerTracker, SmoothedCameraman, split_layout, active_speaker) and 5-Key Gemini Rotation Pool."""
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker.gemini_pool import (
    assign_job,
    configured,
    execute_with_key_rotation,
    is_key_retryable,
    save_key_slots,
)
from local_worker.openshorts_engine import (
    SmoothedCameraman,
    SpeakerTracker,
    analyze_split_scene,
    analyze_three_up_scene,
    filter_edge_bystanders,
    apply_scene_split_layout,
    compact_keyframes,
    independent_split,
    limit_pursuit_rate,
    model_file,
    sendcmd_lines,
    smooth_trajectory_zero_phase,
    split_filtergraph,
    zoom_curve,
)


class TestOpenShortsReframeAndGeminiPool(unittest.TestCase):
    def test_speaker_tracker_hysteresis_and_cooldown_hold(self):
        tracker = SpeakerTracker(cooldown_frames=30, fps=30.0)
        # Frame 0: lock onto speaker A at x=300
        box_a = {"box": [300, 200, 180, 220], "score": 180 * 220}
        box_b = {"box": [1200, 200, 180, 220], "score": 180 * 220}
        t0 = tracker.get_target([box_a], 0, 1920)
        self.assertEqual(t0, [300.0, 200.0, 180.0, 220.0])

        # Frame 5 (inside 30-frame cooldown): speaker A blinks out, only speaker B visible
        # Must return None (hold current frame) instead of whipping to speaker B!
        t5 = tracker.get_target([box_b], 5, 1920)
        self.assertIsNone(t5)

        # Frame 35 (past 30-frame cooldown): speaker B takes over cleanly
        t35 = tracker.get_target([box_b], 35, 1920)
        self.assertEqual(t35, [1200.0, 200.0, 180.0, 220.0])

        # Scene cut reset clears old speaker lock immediately
        tracker.reset()
        self.assertIsNone(tracker.active_speaker_id)

    def test_stabilization_holds_the_subject_against_a_passing_detection(self):
        tracker = SpeakerTracker(stabilization_frames=4, cooldown_frames=30, fps=30.0)
        box_a = {"box": [300, 200, 400, 400], "score": 400 * 400}
        box_b = {"box": [1200, 200, 400, 400], "score": 400 * 400}
        for frame in range(0, 60, 10):
            self.assertEqual(tracker.get_target([box_a], frame, 1920), [300.0, 200.0, 400.0, 400.0])

        # B appears for a single sample while A is still on screen: the camera must not whip to it.
        self.assertEqual(tracker.get_target([box_a, box_b], 65, 1920), [300.0, 200.0, 400.0, 400.0])

        # Once A has left for good, B takes over even though it never matured.
        self.assertEqual(tracker.get_target([box_b], 300, 1920), [1200.0, 200.0, 400.0, 400.0])

    def test_model_file_resolves_env_override_then_reports_a_missing_weight(self):
        """A weight the worker cannot find must resolve to None, not silently enable the saliency stub."""
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            weight = os.path.join(d, "yolov8n.pt")
            with open(weight, "wb") as fh:
                fh.write(b"not-a-model")
            with patch.dict(os.environ, {"YOLO_MODEL_PATH": weight}):
                self.assertEqual(model_file("yolov8n.pt", "YOLO_MODEL_PATH"), weight)
            with patch.dict(os.environ, {"YOLO_MODEL_PATH": os.path.join(d, "absent.pt")}):
                self.assertIsNone(model_file("definitely-absent.pt", "YOLO_MODEL_PATH"))

    def test_smoothed_cameraman_jump_confirmation_and_scene_cut_snap(self):
        cam = SmoothedCameraman(1080, 1920, 1920, 1080, fps=30.0)
        cam.begin_scene()
        # First detection after begin_scene snaps immediately
        cam.update_target([300, 200, 160, 200])
        self.assertAlmostEqual(cam.current_center_x, 380.0)

        # Single-frame outlier jump to x=1500 is ignored (jump_confirm_frames=3)
        cam.update_target([1420, 200, 160, 200])
        self.assertAlmostEqual(cam.target_center_x, 380.0)

        # Three consecutive consistent detections confirm the move
        cam.update_target([1418, 200, 160, 200])
        cam.update_target([1420, 200, 160, 200])
        self.assertAlmostEqual(cam.target_center_x, 1500.0)

    def test_split_layout_median_two_shot_and_independence(self):
        left = {"box": [320, 240, 160, 200], "score": 32000}
        right = {"box": [1280, 250, 160, 200], "score": 32000}
        frames = [[left, right] for _ in range(8)]
        pair = analyze_split_scene(frames, 1920)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertTrue(independent_split(1920, 1080, 1080, 1920, pair))
        fg = split_filtergraph(1920, 1080, 1080, 1920, pair[0], pair[1])
        self.assertIn("vstack=inputs=2", fg)

        # Shot-reverse-shot alternating single faces must return None
        alt_frames = [[left], [right], [left], [right], [left], [right]]
        self.assertIsNone(analyze_split_scene(alt_frames, 1920))

    def test_scene_split_layout_stamps_every_frame_of_the_shot(self):
        left = {"box": [320, 240, 160, 200], "score": 32000}
        right = {"box": [1280, 250, 160, 200], "score": 32000}
        # Frames 1, 3, 4 only see one face: without the scene-level vote they would render
        # as a single-subject crop with a mirrored bottom panel inside the same 2-shot.
        modes = ["split", "fill", "split", "fill", "fill", "split"]
        pts = [
            {
                "t": round(i * 0.2, 3),
                "x": 0.5,
                "y": 0.5,
                "x2": 0.5,
                "y2": 0.5,
                "trackId": 1,
                "mode": modes[i],
                "cut": i == 0,
            }
            for i in range(6)
        ]
        cands = [[left, right], [left], [left, right], [right], [left], [left, right]]

        scenes = apply_scene_split_layout(pts, cands, 1920, 1080)

        self.assertEqual(len(scenes), 1)
        self.assertEqual({p["mode"] for p in pts}, {"split"})
        for p in pts:
            self.assertAlmostEqual(p["x"], round(400 / 1920, 4), places=4)
            self.assertAlmostEqual(p["y"], round(340 / 1080, 4), places=4)
            self.assertAlmostEqual(p["x2"], round(1360 / 1920, 4), places=4)
            self.assertAlmostEqual(p["y2"], round(350 / 1080, 4), places=4)

        # A single-person shot stays untouched
        solo = [{"t": 0.0, "x": 0.3, "y": 0.4, "x2": 0.3, "y2": 0.4, "trackId": 1, "mode": "fill", "cut": True}]
        self.assertEqual(apply_scene_split_layout(solo, [[left]], 1920, 1080), [])
        self.assertEqual(solo[0]["mode"], "fill")

    def test_scene_split_layout_refuses_panels_that_frame_the_same_body(self):
        # One person leaning across the shot clears the 20% centre-separation test, but the two
        # panels would crop overlapping regions of the same frame.
        a = {"box": [640, 200, 400, 500], "score": 200000}
        b = {"box": [960, 210, 400, 500], "score": 200000}
        pts = [
            {
                "t": round(i * 0.2, 3),
                "x": 0.5,
                "y": 0.5,
                "x2": 0.5,
                "y2": 0.5,
                "trackId": 1,
                "mode": "fill",
                "cut": i == 0,
            }
            for i in range(6)
        ]
        cands = [[a, b]] * 6
        self.assertEqual(apply_scene_split_layout(pts, cands, 1920, 1080), [])
        self.assertEqual({p["mode"] for p in pts}, {"fill"})

    def test_beat_zoom_curve_holds_then_decays(self):
        zooms = zoom_curve(24, 7.5, [1.0], max_zoom=1.12)
        self.assertEqual(len(zooms), 24)
        self.assertEqual(zooms[0], 1.0)
        self.assertAlmostEqual(max(zooms), 1.12, places=4)
        self.assertLess(zooms[-1], 1.05)

    def test_compact_keyframes_and_sendcmd(self):
        boxes = [(608, 1080, 400 + i * 2, 0) for i in range(30)]
        keys = compact_keyframes(boxes, 30.0)
        self.assertGreaterEqual(len(keys), 2)
        cmds = sendcmd_lines(boxes, 30.0)
        self.assertTrue(any("crop@c x" in c for c in cmds))

    def test_the_camera_never_travels_faster_than_the_pursuit_law_allows(self):
        # A detector that puts the head somewhere else for one sample is a step, not a pan. Smoothing
        # shortens the step; only the rate limit stops the frame from lurching after it.
        track = [{"t": i / 30.0, "x": 0.5, "y": 0.4, "mode": "fill"} for i in range(30)]
        for i in range(10, 30):
            track[i]["x"] = 0.9
        out = limit_pursuit_rate(track, 0.7, 0.45)
        self.assertEqual(out[0]["x"], 0.5)  # the shot opens where the cut was made
        steps = [out[i + 1]["x"] - out[i]["x"] for i in range(len(out) - 1)]
        self.assertLessEqual(max(abs(s) for s in steps), 0.7 / 30.0 + 1e-4)
        self.assertAlmostEqual(out[-1]["x"], 0.9, places=2)

    def test_a_template_change_is_not_smoothed_into_the_shot_before_it(self):
        # Averaging a solo crop with the two panel centres either side of a split invents a framing
        # that was never decided, and the shot spends its first second sliding out of the composition
        # the cut landed on.
        track = [
            {"t": 0.0, "x": 0.20, "y": 0.4, "x2": 0.20, "y2": 0.4, "mode": "fill"},
            {"t": 1.0, "x": 0.20, "y": 0.4, "x2": 0.20, "y2": 0.4, "mode": "fill"},
            {"t": 2.0, "x": 0.80, "y": 0.4, "x2": 0.95, "y2": 0.4, "mode": "split"},
            {"t": 3.0, "x": 0.80, "y": 0.4, "x2": 0.95, "y2": 0.4, "mode": "split"},
        ]
        out = smooth_trajectory_zero_phase(track, sigma=1.6)
        self.assertEqual(out[2]["x"], 0.80)
        self.assertEqual(out[1]["x"], 0.20)

    def test_five_key_gemini_pool_rotation_and_failover(self):
        status = save_key_slots(
            [
                "AIzaSyTestKeySlot1_abcdefghijk",
                "AIzaSyTestKeySlot2_abcdefghijk",
                "AIzaSyTestKeySlot3_abcdefghijk",
                "AIzaSyTestKeySlot4_abcdefghijk",
                "AIzaSyTestKeySlot5_abcdefghijk",
            ],
            model="gemini-2.5-flash",
            parallel_workers=5,
        )
        self.assertEqual(status["configured_count"], 5)
        self.assertEqual(status["effective_rpm_multiplier"], 5)
        self.assertEqual(len(configured()), 5)

        env1 = assign_job({})
        env2 = assign_job({})
        self.assertNotEqual(env1["GEMINI_API_KEY"], env2["GEMINI_API_KEY"])
        self.assertTrue(is_key_retryable("HTTP 429 RESOURCE_EXHAUSTED: Rate limit reached"))

        # Verify automatic failover when first 2 keys raise 429
        tried_slots = []

        def flaky_call(api_key: str, idx: int) -> str:
            tried_slots.append(idx)
            if len(tried_slots) <= 2:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded on this key")
            return f"success_on_slot_{idx}"

        out = execute_with_key_rotation(flaky_call)
        self.assertEqual(out, "success_on_slot_2")
        self.assertEqual(len(tried_slots), 3)



    def test_rule1_edge_bystander_rejection_prevents_clone_split_on_lineup_closeups(self):
        # Primary speaker in center (cx = 920, 260x300) + neighbor's partial face peeking at right edge (cx = 1810, 110x140)
        center_speaker = {"box": [790, 220, 260, 300], "score": 78000}
        edge_bystander = {"box": [1755, 250, 110, 140], "score": 15400}
        filtered = filter_edge_bystanders([center_speaker, edge_bystander], 1920)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["box"], center_speaker["box"])
        # Must NOT trigger a 2-up split on a single-speaker lineup close-up
        self.assertIsNone(analyze_split_scene([[center_speaker, edge_bystander]] * 6, 1920))

    def test_rule2_middle_subject_veto_prevents_duplicate_panelists_at_desk(self):
        # 3 panelists seated at desk: left (x=260), middle (x=880), right (x=1480)
        p_left = {"box": [180, 240, 170, 210], "score": 35700}
        p_mid = {"box": [800, 240, 165, 205], "score": 33825}
        p_right = {"box": [1400, 240, 170, 210], "score": 35700}
        # 2-Up Split must be vetoed because middle panelist sits between left and right
        self.assertIsNone(analyze_split_scene([[p_left, p_mid, p_right]] * 6, 1920))
        # Instead, 3-Up Grid (analyze_three_up_scene) detects all 3 cleanly
        three_up = analyze_three_up_scene([[p_left, p_mid, p_right]] * 6, 1920)
        self.assertIsNotNone(three_up)
        assert three_up is not None
        self.assertAlmostEqual(three_up[0][0], 265.0)
        self.assertAlmostEqual(three_up[1][0], 882.5)
        self.assertAlmostEqual(three_up[2][0], 1485.0)




class TestYoloSam2BehaviorAndBackendUpgrade(unittest.TestCase):
    def test_one_euro_suppresses_stationary_jitter_and_tracks_fast_motion(self):
        from local_worker.openshorts_engine import OneEuroFilter
        filt = OneEuroFilter(min_cutoff=0.5, beta=0.3)
        noisy = [0.50, 0.504, 0.496, 0.503, 0.497, 0.502]
        smoothed = [filt(i * 0.1, x) for i, x in enumerate(noisy)]
        self.assertLess(max(smoothed) - min(smoothed), (max(noisy) - min(noisy)) * 0.5)

    def test_smooth_axis_one_euro_resets_cleanly_at_scene_cuts(self):
        from local_worker.openshorts_engine import smooth_axis_one_euro
        times = [0.0, 0.13, 0.26, 0.39, 0.52, 0.65]
        vals = [0.22, 0.225, 0.218, 0.78, 0.782, 0.779]
        cuts = [True, False, False, True, False, False]
        out = smooth_axis_one_euro(times, vals, cuts, dead_zone=0.015)
        self.assertAlmostEqual(out[0], 0.22, places=3)
        self.assertAlmostEqual(out[2], 0.22, places=3)
        self.assertAlmostEqual(out[3], 0.78, places=3)

    def test_single_pass_crop_expression_compiles_piecewise_if_chain(self):
        from local_worker.rendering.filters import CropFrame, autoclip_crop_expression
        frames = [
            CropFrame(0.0, 608.0, 1080.0, 200.0, 0.0),
            CropFrame(1.0, 608.0, 1080.0, 400.0, 0.0),
            CropFrame(2.0, 608.0, 1080.0, 400.0, 0.0),
        ]
        w_expr, h_expr, x_expr, y_expr = autoclip_crop_expression(frames, 1920, 1080)
        self.assertEqual(w_expr, "608")
        self.assertEqual(h_expr, "1080")
        self.assertEqual(y_expr, "0")
        self.assertIn("if(lt(t,", x_expr)

    def test_silhouette_refiner_extracts_centroid_head_crown_and_mic_occlusion(self):
        import numpy as np
        from local_worker.openshorts_engine import refine_subject_silhouette
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        frame[260:420, 920:1040] = (130, 160, 215)
        frame[420:500, 860:960] = (8, 8, 8)
        metrics = refine_subject_silhouette(frame, [840.0, 240.0, 240.0, 280.0], kind="face")
        self.assertGreater(metrics.centroid_x, 840.0 / 1920.0)
        self.assertLess(metrics.head_crown_y, metrics.centroid_y)
        self.assertGreater(metrics.occlusion_score, 0.15)
        self.assertGreater(metrics.head_yaw_bias, 0.0)

    def test_situational_decision_matrix_handles_object_demo_solo_cut_and_triad(self):
        from local_worker.openshorts_engine import (
            BEHAVIOR_OBJECT_DEMO,
            BEHAVIOR_SILENT_CUTAWAY,
            BEHAVIOR_TRIAD_PANEL,
            classify_human_behaviors,
        )
        from local_worker.speaker_priority import TalkWindow
        W, H = 1920, 1080
        track_points = [
            {"t": 0.0, "x": 0.45, "y": 0.38, "mode": "fill", "cut": True},
            {"t": 0.4, "x": 0.50, "y": 0.38, "mode": "fit", "cut": True},
            {"t": 0.8, "x": 0.50, "y": 0.38, "mode": "fill", "cut": True},
        ]
        samples = [
            {
                "t": 0.0,
                "people": [
                    {"key": 0, "box": [760.0, 240.0, 220.0, 260.0], "kind": "face"},
                    {"key": -1, "box": [1080.0, 360.0, 240.0, 300.0], "kind": "object"},
                ],
            },
            {
                "t": 0.4,
                "people": [
                    {"key": 1, "box": [820.0, 200.0, 280.0, 320.0], "kind": "face", "yaw_bias": 0.4},
                ],
            },
            {
                "t": 0.8,
                "people": [
                    {"key": 0, "box": [220.0, 260.0, 180.0, 220.0], "kind": "face"},
                    {"key": 1, "box": [860.0, 260.0, 180.0, 220.0], "kind": "face"},
                    {"key": 2, "box": [1480.0, 260.0, 180.0, 220.0], "kind": "face"},
                ],
            },
        ]
        talks = {
            0: TalkWindow(frozenset([0]), False, True),
            1: TalkWindow(frozenset(), False, False),
            2: TalkWindow(frozenset([0, 1, 2]), True, True),
        }
        report = classify_human_behaviors(
            track_points, samples, talks, [0.7, 0.1, 0.8], W, H, main_key=0
        )
        decisions = report["decisions"]
        self.assertEqual(decisions[0]["behavior"], BEHAVIOR_OBJECT_DEMO)
        self.assertGreater(track_points[0]["x"], 0.47)
        self.assertEqual(decisions[1]["behavior"], BEHAVIOR_SILENT_CUTAWAY)
        self.assertEqual(track_points[1]["mode"], "fill")
        self.assertTrue(track_points[1].get("cutProtected"))
        self.assertEqual(decisions[2]["behavior"], BEHAVIOR_TRIAD_PANEL)
        self.assertIsNotNone(track_points[2].get("x3"))

    def test_behavior_evidence_covers_windows_beyond_report_preview(self):
        from local_worker.openshorts_engine import classify_human_behaviors
        from local_worker.speaker_priority import TalkWindow
        n = 100
        track_points = [{"t": i * 0.5 + 0.25, "x": 0.5, "y": 0.38, "mode": "fill", "cut": i == 0}
                        for i in range(n)]
        samples = [{"t": i * 0.5 + 0.25,
                    "people": [{"key": 0, "box": [860.0, 240.0, 200.0, 240.0], "kind": "face"}]}
                   for i in range(n)]
        talks = {i: TalkWindow(frozenset([0]), False, True) for i in range(n)}
        report = classify_human_behaviors(track_points, samples, talks, [0.5] * n, 1920, 1080,
                                          main_key=0, window_sec=0.5)
        self.assertEqual(len(report["decisions"]), 64)
        self.assertEqual(sorted(report["behavior_by_sample"]), [point["t"] for point in track_points])

    def test_behavior_evidence_does_not_cross_cuts_inside_one_audio_window(self):
        from local_worker.openshorts_engine import (
            BEHAVIOR_GROUP_REACTION, BEHAVIOR_SILENT_CUTAWAY, classify_human_behaviors,
        )
        points = [{"t": t, "x": .5, "y": .38, "mode": "fit", "cut": True} for t in (0.0, .2)]
        face = {"key": 0, "box": [800, 200, 200, 250], "kind": "face"}
        group = [{"key": i, "box": [200 + i * 500, 200, 200, 250], "kind": "face"}
                 for i in range(3)]
        samples = [{"t": 0.0, "people": [face]}, {"t": .2, "people": group}]
        report = classify_human_behaviors(points, samples, {}, [0.0], 1920, 1080)
        self.assertEqual(report["behavior_by_sample"], {
            0.0: BEHAVIOR_SILENT_CUTAWAY, .2: BEHAVIOR_GROUP_REACTION,
        })

    def test_bystander_head_does_not_turn_solo_shot_into_group(self):
        from local_worker.openshorts_engine import (
            BEHAVIOR_GROUP_REACTION,
            BEHAVIOR_SOLO_MONOLOGUE,
            classify_human_behaviors,
        )
        from local_worker.speaker_priority import TalkWindow
        track_points = [{"t": 0.0, "x": 0.5, "y": 0.38, "mode": "fit", "cut": True}]
        samples = [{"t": 0.0, "people": [
            {"key": 2, "box": [800.0, 240.0, 300.0, 400.0], "kind": "face"},
            {"key": 8, "box": [1500.0, 300.0, 90.0, 110.0], "kind": "face"},
        ]}]
        talks = {0: TalkWindow(frozenset([5]), False, True)}
        report = classify_human_behaviors(track_points, samples, talks, [0.1], 1920, 1080, main_key=2)
        self.assertEqual(report["decisions"][0]["behavior"], BEHAVIOR_SOLO_MONOLOGUE)
        self.assertNotEqual(report["decisions"][0]["behavior"], BEHAVIOR_GROUP_REACTION)
        self.assertEqual(report["behavior_by_sample"][0.0], BEHAVIOR_SOLO_MONOLOGUE)

    def test_five_person_desk_never_auto_selects_editorial_duplicate_stack(self):
        from local_worker.speaker_priority import candidates_for
        W, H = 1920, 1080
        people = [
            {"key": k, "box": [cx * W - 90, 300, 180, 220], "kind": "face"}
            for k, cx in enumerate([0.18, 0.34, 0.50, 0.66, 0.82])
        ]
        talkers = [people[0], people[4]]
        cands = candidates_for(talkers, people, people[0], 0, W, H, 1.0, allow_editorial=False)
        modes = [c["mode"] for c in cands]
        self.assertNotIn("editorial", modes)
        self.assertNotIn("split", modes)
        self.assertIn("fit", modes)

    def test_solo_closeup_between_hard_cuts_never_folds_into_fit(self):
        from local_worker.openshorts_engine import enforce_layout_hold
        from local_worker.speaker_priority import TalkWindow, apply_speaker_priority
        W, H = 1920, 1080
        points = [
            {"t": 0.0, "x": 0.5, "y": 0.38, "mode": "fit", "cut": True, "zoom": 1.0},
            {"t": 1.2, "x": 0.5, "y": 0.38, "mode": "fit", "cut": False, "zoom": 1.0},
            {"t": 2.0, "x": 0.48, "y": 0.36, "mode": "fill", "cut": True, "cutProtected": True, "zoom": 1.0},
            {"t": 3.0, "x": 0.5, "y": 0.38, "mode": "fit", "cut": True, "zoom": 1.0},
            {"t": 5.0, "x": 0.5, "y": 0.38, "mode": "fit", "cut": False, "zoom": 1.0},
        ]
        enforce_layout_hold(points, min_hold_sec=2.0)
        self.assertEqual(points[2]["mode"], "fill")
        crowd = [{"key": k, "box": [0.15 * (k + 1) * W - 60, 300, 120, 150], "kind": "face"} for k in range(5)]
        solo = [{"key": 0, "box": [0.48 * W - 140, 220, 280, 320], "kind": "face"}]
        samples = [
            {"t": 0.0, "people": crowd},
            {"t": 1.2, "people": crowd},
            {"t": 2.0, "people": solo},
            {"t": 3.0, "people": crowd},
            {"t": 5.0, "people": crowd},
        ]
        talks = {w: TalkWindow(frozenset([0]), False, True) for w in range(15)}
        apply_speaker_priority(points, samples, 0, talks, W, H, min_hold_sec=2.0)
        self.assertEqual(points[2]["mode"], "fill")

    def test_clean_hook_title_keeps_a_leading_contraction_whole(self):
        from local_worker.rendering.subtitles import clean_hook_title

        self.assertEqual(
            clean_hook_title("that's crazy I think you have to do it"),
            "that's crazy I think you have to do",
        )

    def test_clean_hook_title_repairs_broken_school_backpack_fragment(self):
        import tempfile
        from local_worker.rendering.decision import decision_from_payload
        from local_worker.rendering.subtitles import clean_hook_title, write_ass
        cleaned = clean_hook_title("is a school backpack Yeah hers is a")
        self.assertEqual(cleaned.lower(), "hers is a school backpack")
        dec = decision_from_payload({
            "source_path": "source.mp4",
            "source_width": 1920,
            "source_height": 1080,
            "source_fps": 29.97,
            "start": 0.0,
            "end": 10.0,
            "layout": "fill",
            "track": [{"t": 0.0, "x": 0.5, "y": 0.38}],
            "words": [{"w": "Hers", "start": 0.2, "end": 0.5}],
            "hook": "is a school backpack Yeah hers is a",
            "style": {"showHook": True},
        })
        with tempfile.TemporaryDirectory() as tmp:
            ass_path = os.path.join(tmp, "test.ass")
            write_ass(dec, ass_path)
            with open(ass_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            self.assertIn("HERS IS A SCHOOL BACKPACK", content)
            self.assertNotIn("IS A SCHOOL BACKPACK YEAH HERS IS A", content)


class SampledFramesWalk(unittest.TestCase):
    """The analysis walks the decoder instead of seeking per sample, so the walk itself is the contract."""

    class _Cap:
        def __init__(self, n: int):
            self.n, self.pos, self.reads, self.grabs = n, 0, 0, 0

        def read(self):
            self.reads += 1
            if self.pos >= self.n:
                return False, None
            self.pos += 1
            return True, self.pos - 1

        def grab(self):
            self.grabs += 1
            if self.pos >= self.n:
                return False
            self.pos += 1
            return True

    def test_yields_every_stride_th_frame_and_only_reads_those(self):
        from local_worker.openshorts_engine import _sampled_frames

        cap = self._Cap(30)
        got = list(_sampled_frames(cap, 0, 10, 3))
        self.assertEqual([i for i, _ in got], [0, 3, 6, 9])
        self.assertEqual(cap.reads, 4)
        self.assertEqual(cap.grabs, 6)

    def test_starts_mid_file_and_stops_when_the_stream_runs_out(self):
        from local_worker.openshorts_engine import _sampled_frames

        # The fake holds eight frames from wherever the caller seeked to, so the walk keeps the
        # caller's sample grid and simply ends when the decoder does.
        cap = self._Cap(8)
        self.assertEqual([i for i, _ in _sampled_frames(cap, 5, 20, 2)], [5, 7, 9, 11])


if __name__ == "__main__":
    unittest.main()
