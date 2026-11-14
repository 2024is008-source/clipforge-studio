"""Offline cache tests: synthetic results and tiny local files, no model/decoder imports."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker import analysis_cache as ac


def finalized_result():
    return {
        "ok": True,
        "engine": "openshorts_reframe_v2",
        "width": 640, "height": 360, "fps": 30.0,
        "detectors": {"face": "yunet", "person": "ready"},
        "track": [{"t": 0.0, "x": 0.5, "y": 0.4, "mode": "fill", "zoom": 1.0,
                   "layoutDecision": {"scene": 0, "reason": "highest_safe_score"}}],
        "layout_decisions": [{"start": 0.0, "end": 2.0, "layout": "fill",
                              "candidates": {"fill": {"eligible": True}}}],
        "detections": [{"t": 0.0, "subjects": []}],
        "speakers": {"coverage_delivered": {"share": 1.0}},
    }


def serialized_size(result):
    payload = {key: value for key, value in result.items()
               if key not in ("job_id", "source_id", "elapsed_sec")}
    return len(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


class TestAnalysisCache(unittest.TestCase):
    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"synthetic-source-not-video")
        self.patch(patch.dict(os.environ, {}, clear=True))
        self.patch(patch.object(ac.config, "MODELS_DIR", str(self.models)))
        self.patch(patch.object(ac.Path, "cwd", return_value=self.cwd))
        # A cache operation must never require an optional model or video runtime.
        self.patch(patch.dict(sys.modules, {
            "torch": None, "ultralytics": None, "mediapipe": None, "cv2": None,
        }))
        self.cache = ac.AnalysisCache()
        self.result = finalized_result()
        ac.ANALYSIS_CACHE.clear()
        self.addCleanup(ac.ANALYSIS_CACHE.clear)

    def key(self, **overrides):
        options = {"source_path": self.source, "start": 0.0, "end": 2.0, "framing_mode": "auto"}
        options.update(overrides)
        return self.cache.key_for(**options)

    def assert_bounds(self, cache):
        self.assertLessEqual(len(cache._entries), cache._max_entries)
        self.assertLessEqual(cache._bytes, cache._max_bytes)
        self.assertEqual(cache._bytes, sum(size for _, size in cache._entries.values()))

    def test_repeated_request_skips_mock_analysis_and_singleton_api_works(self):
        analyze = Mock(return_value=self.result)

        def execute():
            key = ac.key_for(self.source, 0.0, 2.0, "auto")
            result = ac.ANALYSIS_CACHE.get(key)
            if result is None:
                result = analyze()
                ac.ANALYSIS_CACHE.put(key, result)
            return result

        self.assertEqual(execute(), execute())
        analyze.assert_called_once_with()
        self.assertEqual(self.key(), ac.key_for(str(self.source), 0, 2, "auto"))

    def test_put_and_get_deep_copy_nested_data_and_preserve_key_types(self):
        self.result["speakers"]["people"] = {1: {"windows": [0, 1]}}
        expected = copy.deepcopy(self.result)
        key = self.key()
        self.cache.put(key, self.result)
        self.result["track"][0]["layoutDecision"]["reason"] = "mutated input"
        self.result["speakers"]["people"][1]["windows"].append(2)
        self.result["track"].append({})
        first = self.cache.get(key)
        self.assertEqual(first, expected)
        first["layout_decisions"][0]["candidates"]["fill"]["eligible"] = False
        first["speakers"]["people"][1]["windows"].clear()
        first["job_id"] = "new-job"
        self.assertEqual(self.cache.get(key), expected)

    def test_job_metadata_is_not_retained_or_counted(self):
        self.result.update(job_id="old-job", source_id="old-source", elapsed_sec=90.0)
        expected = finalized_result()
        self.cache.put(self.key(), self.result)
        self.assertEqual(self.cache.get(self.key()), expected)
        self.assertEqual(self.cache._bytes, serialized_size(expected))
        self.assertEqual(self.result["job_id"], "old-job")

    def test_source_resolved_path_size_and_mtime_invalidate(self):
        key = self.key()
        self.cache.put(key, self.result)
        alias = self.root / "models" / ".." / self.source.name
        self.assertEqual(key, self.key(source_path=alias))
        other = self.root / "other.mp4"
        other.write_bytes(self.source.read_bytes())
        self.assertNotEqual(key, self.key(source_path=other))
        old = self.source.stat()
        self.source.write_bytes(b"changed-size")
        os.utime(self.source, ns=(old.st_atime_ns, old.st_mtime_ns))
        size_key = self.key()
        self.assertNotEqual(key, size_key)
        self.assertIsNone(self.cache.get(size_key))
        os.utime(self.source, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(size_key, self.key())

    def test_missing_or_unstatable_source_bypasses_cache(self):
        for source in (self.root / "absent.mp4", self.models):
            with self.subTest(source=source):
                self.assertIsNone(self.key(source_path=source))
        with patch.object(ac.Path, "stat", side_effect=PermissionError("synthetic denial")):
            self.assertIsNone(self.key())
        self.cache.put(None, self.result)
        self.assertIsNone(self.cache.get(None))
        self.assertEqual(self.cache._bytes, 0)

    def test_requested_range_and_all_layout_modes_are_separate(self):
        key = self.key()
        self.cache.put(key, self.result)
        for options in ({"start": 0.1}, {"end": 3.0}, {"end": None},
                        *({"framing_mode": mode} for mode in
                          ("fill", "fit", "split", "grid3", "editorial"))):
            with self.subTest(options=options):
                changed = self.key(**options)
                self.assertNotEqual(key, changed)
                self.assertIsNone(self.cache.get(changed))

    def test_every_analysis_environment_setting_invalidates(self):
        # Intentionally independent of the module's allowlist: catch omissions.
        names = (
            "DETECT_STRIDE", "JUMP_CONFIRM_FRAMES", "SCENE_CUT_RESET", "SPLIT_TIGHTNESS",
            "CAM_DEADBAND_X", "CAM_DEADBAND_Y", "CAM_TAU_X", "CAM_TAU_Y",
            "CAM_MAX_SPEED_X", "CAM_MAX_SPEED_Y", "CUT_THR_MIN", "CUT_THR_MAX", "CUT_THR_SIGMAS",
            "PUNCH_IN_ZOOM", "REFRAME_SAMPLE_FPS", "REFRAME_OBJECT_CLASSES",
            "HOLD_MIN_SEC", "HOLD_MAX_SEC", "SPLIT_MIN_HOLD_SEC", "LAYOUT_CONFIDENCE",
            "LAYOUT_VISIBILITY", "LAYOUT_MAX_PANEL_OVERLAP", "LAYOUT_DETECTION_SHARE",
            "LAYOUT_SWITCH_MARGIN", "YOLO_DEVICE", "YOLO_IMAGE_SIZE", "YOLO_CONFIDENCE",
            "FACE_MODEL_PATH", "YOLO_MODEL_PATH", "YOLO_POSE_MODEL_PATH", "CLIPFORGE_WORK_DIR",
        )
        key = self.key()
        self.cache.put(key, self.result)
        for name in names:
            with self.subTest(name=name), patch.dict(os.environ, {name: "changed"}):
                self.assertNotEqual(key, self.key())
                self.assertIsNone(self.cache.get(self.key()))
        self.assertEqual(key, self.key())

    def test_secrets_unrelated_environment_and_runtime_readiness_do_not_invalidate(self):
        engine = SimpleNamespace(DETECTION_STATUS={"face": "unloaded", "person": "unloaded"})
        service = SimpleNamespace(status="uninitialized", _device="cpu")
        with patch.dict(sys.modules, {
            "local_worker.openshorts_engine": engine,
            "local_worker.detection_service": SimpleNamespace(_SERVICE=service),
        }):
            key = self.key()
            self.cache.put(key, self.result)
            engine.DETECTION_STATUS.update(face="yunet", person="ready")
            service.status, service._device = "ready", "cuda"
            with patch.dict(os.environ, {
                "CLIPFORGE_HELPER_TOKEN": "synthetic-secret", "CLIPFORGE_API_KEY": "synthetic-secret",
                "GEMINI_API_KEY": "synthetic-secret", "CLIPFORGE_WORKER_PORT": "9999",
                "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true", "SAM2_DEVICE": "cuda",
            }):
                self.assertEqual(key, self.key())
                self.assertNotIn("synthetic-secret", repr(self.key()))
                self.assertEqual(self.cache.get(self.key()), self.result)

    def assert_checkpoint_changes(self, path):
        missing = self.key()
        self.cache.put(missing, self.result)
        path.write_bytes(b"fake-weight-v1")
        available = self.key()
        self.assertNotEqual(missing, available)
        self.assertIsNone(self.cache.get(available))
        self.cache.put(available, self.result)
        old = path.stat()
        path.write_bytes(b"fake-weight-v2")  # Same size, different timestamp.
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
        modified = self.key()
        self.assertNotEqual(available, modified)
        self.assertIsNone(self.cache.get(modified))
        path.unlink()
        self.assertNotEqual(modified, self.key())

    def test_all_configured_checkpoints_track_missing_available_modified_deleted(self):
        for name, suffix in (("FACE_MODEL_PATH", ".onnx"), ("YOLO_MODEL_PATH", ".pt"),
                             ("YOLO_POSE_MODEL_PATH", ".pt")):
            path = self.root / (name + suffix)
            with self.subTest(name=name), patch.dict(os.environ, {name: str(path)}):
                self.assert_checkpoint_changes(path)

    def test_default_yunet_and_yolo_in_models_and_cwd_track_changes(self):
        for directory in (self.models, self.cwd):
            for filename in ("face_detection_yunet_2023mar.onnx", "yolov8n.pt"):
                with self.subTest(directory=directory, filename=filename):
                    self.assert_checkpoint_changes(directory / filename)

    def test_configured_yolo_expands_user_and_trims_whitespace(self):
        path = self.root / "custom.pt"
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": "  ~/custom.pt  "}), \
                patch.object(ac.Path, "expanduser", return_value=path) as expand:
            self.assert_checkpoint_changes(path)
            self.assertGreater(expand.call_count, 0)

    def test_default_cache_directory_identity_is_in_key(self):
        key = self.key()
        with patch.object(ac.config, "MODELS_DIR", str(self.root / "other-cache")):
            self.assertNotEqual(key, self.key())
        with patch.object(ac.Path, "cwd", return_value=self.root):
            self.assertNotEqual(key, self.key())

    def test_default_eight_entry_bound_and_get_promotes_lru(self):
        for index in range(8):
            self.cache.put((index,), self.result)
        self.assertEqual(len(self.cache._entries), 8)
        self.assertEqual(self.cache.get((0,)), self.result)
        self.cache.get((999,))  # Misses must not change recency.
        self.cache.put((8,), self.result)
        self.assertIsNone(self.cache.get((1,)))
        for index in (0, 2, 3, 4, 5, 6, 7, 8):
            self.assertIsNotNone(self.cache.get((index,)))
        self.assert_bounds(self.cache)

    def test_byte_limit_uses_utf8_bytes_and_evicts_lru_below_entry_limit(self):
        self.result["note"] = "\u4e2d\u6587" * 20
        size = serialized_size(self.result)
        self.assertGreater(size, len(json.dumps(self.result, ensure_ascii=False, separators=(",", ":"))))
        cache = ac.AnalysisCache(max_bytes=2 * size)
        cache.put((0,), self.result)
        cache.put((1,), self.result)
        self.assertEqual(cache._bytes, 2 * size)  # The exact boundary is allowed.
        cache.get((0,))
        cache.put((2,), self.result)
        self.assertIsNone(cache.get((1,)))
        self.assertIsNotNone(cache.get((0,)))
        self.assertIsNotNone(cache.get((2,)))
        self.assertEqual(len(cache._entries), 2)
        self.assert_bounds(cache)

    def test_larger_entry_can_evict_multiple_older_entries(self):
        size = serialized_size(self.result)
        cache = ac.AnalysisCache(max_bytes=3 * size)
        for index in range(3):
            cache.put((index,), self.result)
        larger = copy.deepcopy(self.result)
        larger["padding"] = "x" * size
        cache.put((3,), larger)
        self.assertEqual(list(cache._entries), [(3,)])
        self.assert_bounds(cache)

    def test_oversize_rejection_preserves_entries_and_replacement(self):
        size = serialized_size(self.result)
        cache = ac.AnalysisCache(max_bytes=size)
        cache.put((0,), self.result)
        oversized = copy.deepcopy(self.result)
        oversized["extra"] = "x"
        cache.put((1,), oversized)
        cache.put((0,), oversized)
        self.assertIsNone(cache.get((1,)))
        self.assertEqual(cache.get((0,)), self.result)
        self.assertEqual(cache._bytes, size)
        self.assert_bounds(cache)

    def test_replacement_updates_size_and_recency_without_double_counting(self):
        size = serialized_size(self.result)
        cache = ac.AnalysisCache(max_entries=2, max_bytes=3 * size)
        cache.put((0,), self.result)
        cache.put((1,), self.result)
        larger = copy.deepcopy(self.result)
        larger["extra"] = "x" * 20
        cache.put((0,), larger)
        self.assertEqual(cache._bytes, size + serialized_size(larger))
        cache.put((2,), self.result)
        self.assertIsNone(cache.get((1,)))
        self.assertEqual(cache.get((0,)), larger)
        cache.put((0,), self.result)
        self.assertEqual(cache._bytes, 2 * size)
        self.assert_bounds(cache)

    def test_clear_resets_entries_and_byte_accounting(self):
        cache = ac.AnalysisCache(max_bytes=serialized_size(self.result))
        cache.put((0,), self.result)
        cache.clear()
        cache.clear()
        self.assertIsNone(cache.get((0,)))
        self.assertEqual(cache._bytes, 0)
        cache.put((1,), self.result)
        self.assertEqual(cache.get((1,)), self.result)
        self.assert_bounds(cache)

    def test_limits_cannot_exceed_eight_entries_or_32_mib(self):
        self.assertEqual(self.cache._max_entries, 8)
        self.assertEqual(self.cache._max_bytes, 32 * 1024 * 1024)
        for options in ({"max_entries": 0}, {"max_entries": 9}, {"max_bytes": 0},
                        {"max_bytes": 32 * 1024 * 1024 + 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ac.AnalysisCache(**options)

    def test_failures_empty_tracks_and_partial_results_are_not_cached(self):
        cases = [None, {}, {"ok": True}, {"track": self.result["track"]}]
        for change in (
            {"ok": False}, {"ok": "true"}, {"track": []}, {"track": None}, {"track": [{}]},
            {"error": "inference failed"}, {"partial": True}, {"cancelled": True},
            {"finalized": False}, {"status": "unavailable"}, {"status": "error"},
            {"status": "running"}, {"layout_decisions": []}, {"layout_decisions": [{}]},
            {"detectors": None}, {"detectors": {}},
        ):
            cases.append(dict(self.result, **change))
        before_finalization = copy.deepcopy(self.result)
        del before_finalization["track"][0]["layoutDecision"]
        cases.append(before_finalization)
        for bad in cases:
            with self.subTest(result=bad):
                self.cache.put(self.key(), bad)
                self.assertIsNone(self.cache.get(self.key()))
                self.assertEqual(self.cache._bytes, 0)

    def test_unavailable_or_error_detectors_are_excluded_despite_ok_and_tracks(self):
        for detector in ("face", "person"):
            for status in ("uninitialized", "unloaded", "unavailable", "error", "none", "", None, [], {}):
                with self.subTest(detector=detector, status=status):
                    bad = copy.deepcopy(self.result)
                    bad["detectors"][detector] = status
                    self.cache.put(self.key(), bad)
                    self.assertIsNone(self.cache.get(self.key()))

    def test_successful_fallbacks_and_empty_detections_are_cacheable(self):
        for face in ("mediapipe", "yunet", "haar_frontal_profile"):
            for person in ("ready", "ready_cpu_fallback"):
                with self.subTest(face=face, person=person):
                    self.result["detectors"] = {"face": face, "person": person}
                    self.result["track"][0]["mode"] = "fit"
                    self.cache.put(self.key(), self.result)
                    self.assertEqual(self.cache.get(self.key()), self.result)

    def test_non_json_and_nonfinite_results_are_rejected_without_losing_good_entry(self):
        key = self.key()
        self.cache.put(key, self.result)
        cyclic = []
        cyclic.append(cyclic)
        for value in (object(), {1, 2}, float("nan"), float("inf"), cyclic):
            with self.subTest(value_type=type(value)):
                bad = dict(self.result, invalid=value)
                self.cache.put(("new",), bad)
                self.cache.put(key, bad)
                self.assertIsNone(self.cache.get(("new",)))
                self.assertEqual(self.cache.get(key), self.result)
                self.assertEqual(self.cache._bytes, serialized_size(self.result))


if __name__ == "__main__":
    unittest.main()
