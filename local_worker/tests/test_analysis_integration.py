"""Offline analysis integration: synthetic decoder, mocked inference and audio only.

The candidate adapters, sampling, tracking, speaker rules, smoothing and final layout
policy run together. No video file, model runtime, weights or FFmpeg is required.
"""
import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker import layout_decision
from local_worker import openshorts_engine as engine

WIDTH, HEIGHT, FPS = 640, 360, 20.0
VIDEO_PATH = str(Path(__file__).resolve().parent / "synthetic-not-on-disk.mp4")


def face(box=(288, 72, 64, 54), confidence=0.93):
    return {"box": list(box), "score": float(box[2] * box[3]), "confidence": confidence}


def region(box, label="person", confidence=0.87, class_id=0):
    return {"box": list(box), "label": label, "class_id": class_id, "confidence": confidence}


class SyntheticCapture:
    """Seekable metadata + lazy, small frames; log both walks without storing media."""

    def __init__(self):
        self.opened = True
        self.position = 0
        self.pass_number = 0
        self.total_frames = 80
        self.decode_limits = {}
        self.empty_read_succeeds = False
        self.seeks = []
        self.reads = []
        self.grabs = []
        self.release_count = 0
        self.frame = np.full((HEIGHT, WIDTH, 3), (45, 100, 160), dtype=np.uint8)

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: FPS,
            cv2.CAP_PROP_FRAME_WIDTH: self.frame.shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self.frame.shape[0],
            cv2.CAP_PROP_FRAME_COUNT: self.total_frames,
        }[prop]

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(f"Unexpected decoder property: {prop}")
        self.seeks.append((prop, value))
        self.position = int(value)
        self.pass_number += 1
        return True

    def read(self):
        self.reads.append((self.pass_number, self.position))
        limit = self.decode_limits.get(self.pass_number, self.total_frames)
        if self.position >= limit:
            return self.empty_read_succeeds, None
        frame = self.frame.copy()
        frame[0, 0, 0] = self.position % 256
        self.position += 1
        return True, frame

    def grab(self):
        self.grabs.append((self.pass_number, self.position))
        limit = self.decode_limits.get(self.pass_number, self.total_frames)
        if self.position >= limit:
            return False
        self.position += 1
        return True

    def release(self):
        self.release_count += 1
        self.opened = False


class TestAnalysisIntegration(unittest.TestCase):
    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        self.cap = SyntheticCapture()
        self.capture = self.patch(patch.object(cv2, "VideoCapture", return_value=self.cap))
        # Block optional model imports even if a future change bypasses the mocked boundary.
        self.patch(patch.dict(sys.modules, {"torch": None, "ultralytics": None, "mediapipe": None}))
        self.download = self.patch(patch.object(
            engine, "fetch_model", side_effect=AssertionError("Tests must not fetch weights")
        ))
        self.patch(patch.dict(os.environ, {
            "REFRAME_SAMPLE_FPS": "10",
            "LAYOUT_CONFIDENCE": "0.45",
            "LAYOUT_VISIBILITY": "0.98",
            "LAYOUT_MAX_PANEL_OVERLAP": "0.15",
            "LAYOUT_DETECTION_SHARE": "0.65",
            "LAYOUT_SWITCH_MARGIN": "0.08",
        }))
        self.patch(patch.dict(engine.DETECTION_STATUS, {"face": "mock", "person": "mock"}, clear=True))
        self.faces = [face()]
        self.regions = []
        self.region_batches = []

        def infer_regions(frame):
            # A region inference must happen in phase B, not in the histogram pass.
            self.assertEqual(self.cap.pass_number, 2)
            batch = copy.deepcopy(self.regions)
            self.region_batches.append(batch)
            return batch

        self.inference = self.patch(patch.object(engine, "detect_framing_regions", side_effect=infer_regions))
        self.face_detector = self.patch(patch.object(
            engine, "detect_face_candidates", side_effect=lambda frame: copy.deepcopy(self.faces)
        ))
        self.portraits = self.patch(patch.object(
            engine, "detect_portrait_candidates", wraps=engine.detect_portrait_candidates
        ))
        self.objects = self.patch(patch.object(
            engine, "detect_object_candidates", wraps=engine.detect_object_candidates
        ))
        self.audio = self.patch(patch.object(engine, "audio_envelope", return_value=[0.0] * 10))
        self.progress = Mock()
        actual_finalize = layout_decision.finalize_layouts

        def finalize(points, samples, *args, **kwargs):
            self.before_finalization = copy.deepcopy(points)
            self.layout_samples = copy.deepcopy(samples)
            self.finalized_track = points
            self.layout_report = actual_finalize(points, samples, *args, **kwargs)
            return self.layout_report

        self.finalize = self.patch(patch.object(layout_decision, "finalize_layouts", side_effect=finalize))

    def tearDown(self):
        self.download.assert_not_called()

    def analyze(self, **kwargs):
        options = {"start_sec": 1.0, "end_sec": 3.0, "progress_cb": self.progress}
        options.update(kwargs)
        return engine.analyze_video_with_openshorts(VIDEO_PATH, **options)

    def assert_shared_inference(self, count):
        self.assertEqual(self.inference.call_count, count)
        self.assertEqual(self.portraits.call_count, count)
        self.assertEqual(self.objects.call_count, count)
        self.assertEqual(self.face_detector.call_count, count)
        for i, (portrait_call, object_call) in enumerate(zip(
            self.portraits.call_args_list, self.objects.call_args_list
        )):
            self.assertIs(portrait_call.args[1], self.region_batches[i])
            self.assertIs(object_call.args[1], self.region_batches[i])
            self.assertIs(portrait_call.args[0], object_call.args[0])

    def assert_normalized(self, result):
        # Reject NaN/Infinity and numpy-only values in the actual API response.
        json.dumps(result, allow_nan=False)
        self.assertTrue(result["ok"])
        self.assertTrue(result["track"])
        self.assertEqual([p["t"] for p in result["track"]], [d["t"] for d in result["detections"]])
        for point in result["track"]:
            for axis in ("x", "y", "x2", "y2", "x3", "y3"):
                if axis in point:
                    self.assertTrue(math.isfinite(point[axis]))
                    self.assertGreaterEqual(point[axis], 0.0)
                    self.assertLessEqual(point[axis], 1.0)
        for detection in result["detections"]:
            for subject in detection["subjects"]:
                self.assertEqual(len(subject["box"]), 4)
                self.assertTrue(all(0 <= v <= 1 for v in subject["box"]))
                self.assertLessEqual(subject["box"][0] + subject["box"][2], 1.000001)
                self.assertLessEqual(subject["box"][1] + subject["box"][3], 1.000001)
                self.assertTrue(0 <= subject["confidence"] <= 1)

    def test_two_sampled_walks_share_one_region_inference_per_phase_b_sample(self):
        self.regions = [
            region((256, 60, 128, 240)),  # Same person as the face: must not duplicate it.
            region((32, 180, 144, 90), "laptop", 0.87, 63),
        ]
        result = self.analyze()
        self.capture.assert_called_once_with(VIDEO_PATH, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        self.assertEqual(self.cap.seeks, [(cv2.CAP_PROP_POS_FRAMES, 20)] * 2)
        expected = [(phase, i) for phase in (1, 2) for i in range(20, 60, 2)]
        self.assertEqual(self.cap.reads, expected)
        self.assertEqual(self.cap.grabs, [(phase, i + 1) for phase, i in expected])
        self.assertEqual(self.cap.release_count, 1)
        self.assert_shared_inference(20)
        self.assertEqual([int(c.args[0][0, 0, 0]) for c in self.inference.call_args_list], list(range(20, 60, 2)))
        for detection in result["detections"]:
            self.assertEqual([s["label"] for s in detection["subjects"]], ["face", "laptop"])
        self.assertEqual([p["t"] for p in result["track"]], [round(1 + i / 10, 3) for i in range(20)])
        self.assertEqual((result["width"], result["height"], result["fps"], result["sample_fps"]), (WIDTH, HEIGHT, FPS, 10.0))
        self.audio.assert_called_once_with(VIDEO_PATH, 1.0, 2.0, window_s=engine.BEAT_WINDOW)
        progress = [c.args[0] for c in self.progress.call_args_list]
        self.assertEqual(progress, sorted(progress))
        self.assertIn(0.35, progress)
        self.assertEqual(progress[-1], 1.0)
        self.assertTrue(all(0 <= p <= 1 for p in progress))
        self.assert_normalized(result)

    def test_face_confidence_and_wide_object_context_reach_final_policy_and_response(self):
        self.regions = [region((32, 180, 144, 90), "laptop", 0.87, 63)]
        result = self.analyze()
        self.finalize.assert_called_once()
        self.assertIs(result["track"], self.finalized_track)
        self.assertIs(result["layout_decisions"], self.layout_report)
        self.assertEqual({p["mode"] for p in self.before_finalization}, {"fill"})
        self.assertEqual({p["mode"] for p in result["track"]}, {"fit"})
        self.assertIn("required_content_outside_crop", result["layout_decisions"][0]["reason"])
        for sample, detection in zip(self.layout_samples, result["detections"]):
            human, obj = sample["people"]
            self.assertEqual((human["kind"], human["box"], human["confidence"]), ("face", self.faces[0]["box"], 0.93))
            self.assertEqual((obj["kind"], obj["label"], obj["box"], obj["confidence"]), ("object", "laptop", self.regions[0]["box"], 0.87))
            human_out, object_out = detection["subjects"]
            self.assertEqual(human_out["box"], [0.45, 0.2, 0.1, 0.15])
            self.assertEqual(human_out["confidence"], 0.93)
            self.assertIsInstance(human_out["id"], int)
            self.assertGreaterEqual(human_out["id"], 0)
            self.assertEqual(object_out, {"id": "object:0", "kind": "object", "label": "laptop", "box": [0.05, 0.5, 0.225, 0.25], "confidence": 0.87})
        self.assert_normalized(result)

    def test_empty_precomputed_regions_are_reused_and_no_detections_preserve_source(self):
        self.faces = []
        result = self.analyze()
        self.assert_shared_inference(20)
        self.assertTrue(all(not d["subjects"] for d in result["detections"]))
        self.assertEqual({p["mode"] for p in result["track"]}, {"fit"})
        self.assertIn("low_tracking_confidence", result["layout_decisions"][0]["reason"])
        self.assert_normalized(result)

    def test_body_fallback_keeps_full_box_and_confidence_beside_a_visible_face(self):
        self.regions = [region((460, 50, 140, 270), confidence=0.76)]
        result = self.analyze()
        self.assert_shared_inference(20)
        for detection in result["detections"]:
            subjects = {s["kind"]: s for s in detection["subjects"]}
            self.assertEqual(set(subjects), {"face", "person"})
            self.assertEqual(subjects["person"]["box"], [0.71875, 0.138889, 0.21875, 0.75])
            self.assertEqual(subjects["person"]["confidence"], 0.76)
            self.assertNotEqual(subjects["face"]["id"], subjects["person"]["id"])
        self.assert_normalized(result)

    def test_object_only_fallback_does_not_invent_a_speaker(self):
        self.faces = []
        self.regions = [region((280, 150, 80, 80), "sports ball", 0.91, 32)]
        result = self.analyze()
        self.assert_shared_inference(20)
        self.assertEqual({p["subjectType"] for p in result["track"]}, {"object"})
        for detection in result["detections"]:
            self.assertEqual(len(detection["subjects"]), 1)
            self.assertEqual(detection["subjects"][0]["kind"], "object")
            self.assertEqual(detection["subjects"][0]["id"], "object:0")
        self.assert_normalized(result)

    def test_objects_keep_distinct_ids_without_becoming_speakers(self):
        self.regions = [region((40, 150, 80, 80), "laptop", .91, 63),
                        region((480, 150, 80, 80), "laptop", .92, 63)]
        result = self.analyze()
        identities = [{s["id"] for s in d["subjects"] if s["kind"] == "object"}
                      for d in result["detections"]]
        self.assertTrue(all(ids == {"object:0", "object:1"} for ids in identities))
        self.assertEqual(result["layout_decisions"][-1]["end"], 3.0)

    def test_low_face_confidence_is_not_replaced_by_tracking_score(self):
        self.faces = [face(confidence=0.2)]
        result = self.analyze()
        self.assertEqual({p["mode"] for p in result["track"]}, {"fit"})
        self.assertIn("low_tracking_confidence", result["layout_decisions"][0]["reason"])
        self.assertTrue(all(d["subjects"][0]["confidence"] == 0.2 for d in result["detections"]))

    def test_manual_fit_overrules_real_solo_closeup_behavior_after_smoothing(self):
        result = self.analyze(framing_mode="fit")
        self.assertEqual({p["mode"] for p in self.before_finalization}, {"fill"})
        self.assertEqual({p["mode"] for p in result["track"]}, {"fit"})
        self.assertEqual({p["zoom"] for p in result["track"]}, {1.0})
        self.assertEqual(result["layout_decisions"][0]["reason"], "manual_override")
        self.assert_normalized(result)

    def test_manual_fill_is_authoritative_even_without_detections(self):
        self.faces = []
        result = self.analyze(framing_mode="fill")
        self.assertEqual({p["mode"] for p in result["track"]}, {"fill"})
        self.assertEqual(result["layout_decisions"][0]["reason"], "manual_override")

    def test_manual_editorial_survives_solo_behavior(self):
        result = self.analyze(framing_mode="editorial")
        self.assertEqual({p["mode"] for p in result["track"]}, {"editorial"})
        self.assertEqual(result["layout_decisions"][0]["reason"], "manual_override")
        self.assert_normalized(result)

    def test_manual_split_cannot_duplicate_a_solo_subject(self):
        result = self.analyze(framing_mode="split")
        self.assertEqual({p["mode"] for p in result["track"]}, {"fit"})
        self.assertEqual(result["layout_decisions"][0]["reason"], "unsafe_manual_split")

    def test_manual_split_keeps_two_distinct_visible_subjects(self):
        # Below 480 px high, the legacy panel geometry disables tightening and
        # cannot produce nonoverlapping panels, even with well-separated faces.
        self.cap.frame = np.full((540, 960, 3), (45, 100, 160), dtype=np.uint8)
        self.faces = [face((160, 115, 60, 80)), face((740, 115, 60, 80))]
        result = self.analyze(framing_mode="split")
        self.assertEqual({p["mode"] for p in result["track"]}, {"split"})
        self.assertTrue(result["layout_decisions"][0]["candidates"]["split"]["eligible"])
        self.assertTrue(all(p["x"] < p["x2"] for p in result["track"]))
        self.assert_normalized(result)

    def test_cancellation_in_phase_a_releases_capture_and_skips_inference(self):
        cancelled = Mock(side_effect=[False, True])
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            self.analyze(is_cancelled=cancelled)
        self.assertEqual(self.cap.release_count, 1)
        self.assertEqual(self.cap.pass_number, 1)
        self.inference.assert_not_called()
        self.audio.assert_not_called()
        self.finalize.assert_not_called()
        self.assertNotIn(call(1.0), self.progress.call_args_list)

    def test_cancellation_in_phase_b_releases_capture_before_postprocessing(self):
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            self.analyze(is_cancelled=lambda: self.inference.call_count >= 2)
        self.assertEqual(self.cap.release_count, 1)
        self.assert_shared_inference(2)
        self.audio.assert_not_called()
        self.finalize.assert_not_called()
        self.assertNotIn(call(1.0), self.progress.call_args_list)

    def test_no_decoded_frames_releases_capture_and_reports_error(self):
        self.cap.decode_limits[1] = 0
        with self.assertRaisesRegex(ValueError, "No video frames decoded"):
            self.analyze()
        self.assertEqual(self.cap.release_count, 1)
        self.inference.assert_not_called()
        self.audio.assert_not_called()
        self.finalize.assert_not_called()
        self.assertNotIn(call(1.0), self.progress.call_args_list)

    def test_successful_read_with_no_frame_is_also_a_decode_failure(self):
        self.cap.decode_limits[1] = 0
        self.cap.empty_read_succeeds = True
        with self.assertRaisesRegex(ValueError, "No video frames decoded"):
            self.analyze()
        self.assertEqual(self.cap.release_count, 1)
        self.inference.assert_not_called()

    def test_no_frames_on_second_walk_is_not_a_successful_empty_analysis(self):
        self.cap.decode_limits[2] = 0
        try:
            with self.assertRaisesRegex(ValueError, "No video frames decoded"):
                self.analyze()
        finally:
            self.assertEqual(self.cap.release_count, 1)
        self.inference.assert_not_called()
        self.assertNotIn(call(1.0), self.progress.call_args_list)

    def test_inference_exception_releases_capture_and_propagates(self):
        self.inference.side_effect = RuntimeError("synthetic inference failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            self.analyze()
        self.assertEqual(self.cap.release_count, 1)
        self.audio.assert_not_called()
        self.finalize.assert_not_called()
        self.assertNotIn(call(1.0), self.progress.call_args_list)

    def test_unopenable_capture_is_released_without_reading(self):
        self.cap.opened = False
        with self.assertRaisesRegex(ValueError, "Cannot open video"):
            self.analyze()
        self.assertEqual(self.cap.release_count, 1)
        self.assertEqual(self.cap.reads, [])
        self.inference.assert_not_called()

    def test_invalid_interval_or_mode_is_rejected_before_opening_decoder(self):
        for options in (
            {"start_sec": -1.0}, {"start_sec": float("nan")},
            {"end_sec": float("inf")}, {"end_sec": 1.0},
            {"end_sec": 0.5}, {"framing_mode": "unknown"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.analyze(**options)
        self.capture.assert_not_called()
        self.inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
