"""Offline detector tests: all model loading and CUDA probing are mocked."""
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker import detection_service as ds


def result(boxes=(), classes=None, scores=None, names=None, points=None):
    count = len(boxes)
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            cls=np.asarray(classes if classes is not None else [0] * count, dtype=float),
            conf=np.asarray(scores if scores is not None else [0.9] * count, dtype=float),
        ),
        names={0: "person"} if names is None else names,
        keypoints=None if points is None else SimpleNamespace(data=np.asarray(points, dtype=float)),
        orig_shape=(640, 640),
    )


def model(task="detect", results=None):
    instance = Mock()
    instance.task = task
    instance.names = {0: "person"}
    instance.predict.return_value = [result([[20, 10, 100, 50]])] if results is None else results
    instance.to.return_value = instance
    return instance


class TestDetectionService(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        self.weight = self.root / "detect.pt"
        self.weight.write_bytes(b"mock-checkpoint")
        self.pose_weight = self.root / "pose.pt"
        self.pose_weight.write_bytes(b"mock-pose-checkpoint")
        env = {
            "YOLO_MODEL_PATH": str(self.weight), "YOLO_POSE_MODEL_PATH": "",
            "YOLO_DEVICE": "auto", "YOLO_IMAGE_SIZE": "640", "YOLO_CONFIDENCE": "0.4",
        }
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, env).start()
        patch.object(ds.config, "MODELS_DIR", str(self.models)).start()
        patch.object(ds.config, "WORK_DIR", str(self.root)).start()
        patch.object(ds.Path, "cwd", return_value=self.cwd).start()
        patch.object(ds, "_SERVICE", None).start()
        self.detector = model()
        self.load = patch.object(ds, "_load_model", return_value=self.detector).start()
        self.cuda = patch.object(ds, "_cuda_available", return_value=False).start()
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.service = ds.DetectionService()

    def test_singleton_is_lazy_and_thread_safe(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            services = list(pool.map(lambda _: ds.get_detection_service(), range(24)))
        self.assertTrue(all(item is services[0] for item in services))
        self.assertEqual(services[0].status, "uninitialized")
        self.assertIsNone(services[0].warning)
        self.load.assert_not_called()
        self.cuda.assert_not_called()

    def test_defaults_normalize_using_original_frame_and_reuse_model(self):
        detections = self.service.predict(self.frame)
        self.assertEqual(detections, [{
            "box": [0.1, 0.1, 0.4, 0.4], "class_id": 0,
            "label": "person", "confidence": 0.9,
        }])
        self.service.predict(self.frame + 1)
        self.load.assert_called_once_with(str(self.weight.resolve()), "detect")
        self.cuda.assert_called_once_with()
        self.assertEqual(self.detector.predict.call_count, 2)
        kwargs = self.detector.predict.call_args.kwargs
        self.assertEqual(kwargs["batch"], 1)
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["conf"], 0.4)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertNotIn("half", kwargs)
        self.assertFalse(kwargs["save"])
        self.assertEqual(self.service.status, "ready")
        self.assertIsNone(self.service.warning)

    def test_cache_hits_for_equal_pixels_and_isolated_return_values(self):
        first = self.service.predict(self.frame)
        first[0]["box"][0] = 0.99
        first.append({"label": "changed"})
        second = self.service.predict(self.frame.copy())
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["box"][0], 0.1)
        second[0]["label"] = "changed"
        self.assertEqual(self.service.predict(self.frame)[0]["label"], "person")
        self.detector.predict.assert_called_once()

    def test_cache_tracks_pixels_dimensions_dtype_and_noncontiguous_frames(self):
        self.service.predict(self.frame)
        self.service.predict(np.asfortranarray(self.frame))
        self.assertEqual(self.detector.predict.call_count, 1)
        self.frame[0, 0, 0] = 1
        self.service.predict(self.frame)
        self.service.predict(self.frame.reshape(200, 100, 3))
        self.service.predict(self.frame.astype(np.float32))
        self.assertEqual(self.detector.predict.call_count, 4)

    def test_lru_cache_is_bounded_and_retains_recent_hits(self):
        with patch.object(ds, "_CACHE_SIZE", 2):
            self.service.predict(self.frame)
            self.service.predict(self.frame + 1)
            self.service.predict(self.frame)
            self.service.predict(self.frame + 2)
            self.service.predict(self.frame)
            self.assertEqual(self.detector.predict.call_count, 3)
            self.service.predict(self.frame + 1)
            self.assertEqual(self.detector.predict.call_count, 4)
            self.assertEqual(len(self.service._cache), 2)
        self.load.assert_called_once()

    def test_both_successful_empty_result_forms_are_cached(self):
        for results in ([result()], []):
            with self.subTest(results=results):
                service = ds.DetectionService()
                self.detector.predict.reset_mock()
                self.detector.predict.return_value = results
                self.assertEqual(service.predict(self.frame), [])
                self.assertEqual(service.predict(self.frame.copy()), [])
                self.detector.predict.assert_called_once()
                self.assertEqual(service.status, "ready")
                self.assertIsNone(service.warning)

    def test_prediction_failures_are_not_cached_and_recovery_clears_warning(self):
        self.detector.predict.side_effect = [ValueError("bad inference"), [result()]]
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "error")
        self.assertIn("bad inference", self.service.warning)
        self.assertEqual(len(self.service._cache), 0)
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "ready")
        self.assertIsNone(self.service.warning)
        self.assertEqual(self.detector.predict.call_count, 2)
        self.load.assert_called_once()

    def test_malformed_results_are_failures_not_cached_empty_detections(self):
        malformed = result([[0, 0, 10, 10]], classes=[])
        for results in (None, [SimpleNamespace(boxes=None)], [malformed]):
            with self.subTest(results=results):
                self.detector.predict.return_value = results
                self.assertEqual(self.service.predict(self.frame), [])
                self.assertEqual(self.service.status, "error")
                self.assertTrue(self.service.warning)
                self.assertEqual(len(self.service._cache), 0)
        self.assertEqual(self.detector.predict.call_count, 3)

    def test_weights_stat_identity_invalidates_cache_and_reloads_once(self):
        self.service.predict(self.frame)
        original = self.weight.stat()
        os.utime(self.weight, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        self.service.predict(self.frame)
        self.service.predict(self.frame)
        self.assertEqual(self.load.call_count, 2)
        self.assertEqual(self.detector.predict.call_count, 2)
        self.weight.write_bytes(b"replacement checkpoint with different size")
        self.service.predict(self.frame)
        self.assertEqual(self.load.call_count, 3)

    def test_settings_are_in_cache_identity_and_passed_to_predict(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "cpu", "YOLO_IMAGE_SIZE": "320", "YOLO_CONFIDENCE": "0.65"}):
            configured = ds.DetectionService()
        configured.predict(self.frame)
        kwargs = self.detector.predict.call_args.kwargs
        self.assertEqual((kwargs["device"], kwargs["imgsz"], kwargs["conf"]), ("cpu", 320, 0.65))
        self.service.predict(self.frame)
        self.assertNotEqual(next(iter(configured._cache)), next(iter(self.service._cache)))

    def test_invalid_settings_report_unavailable_without_loading(self):
        for settings in (
            {"YOLO_DEVICE": "remote"}, {"YOLO_IMAGE_SIZE": "zero"}, {"YOLO_IMAGE_SIZE": "0"},
            {"YOLO_CONFIDENCE": "nan"}, {"YOLO_CONFIDENCE": "inf"}, {"YOLO_CONFIDENCE": "1.1"},
            {"YOLO_CONFIDENCE": "-0.1"},
        ):
            with self.subTest(settings=settings), patch.dict(os.environ, settings):
                service = ds.DetectionService()
                self.assertEqual(service.predict(self.frame), [])
                self.assertEqual(service.status, "unavailable")
                self.assertIn("configuration", service.warning)
        self.load.assert_not_called()
        self.cuda.assert_not_called()

    def test_invalid_checkpoint_paths_never_reach_loader_or_fallback(self):
        (self.models / "yolov8n.pt").write_bytes(b"cached")
        empty = self.root / "empty.pt"
        empty.touch()
        yaml = self.root / "model.yaml"
        yaml.write_text("not a checkpoint", encoding="utf-8")
        for path in (str(self.root / "missing.pt"), str(empty), str(yaml), str(self.root), "https://example.com/model.pt"):
            with self.subTest(path=path), patch.dict(os.environ, {"YOLO_MODEL_PATH": path}):
                service = ds.DetectionService()
                self.assertEqual(service.predict(self.frame), [])
                self.assertEqual(service.status, "unavailable")
                self.assertIn("local .pt", service.warning)
        self.load.assert_not_called()
        self.cuda.assert_not_called()

    def test_corrupt_checkpoint_load_failure_is_remembered_until_file_changes(self):
        self.load.side_effect = RuntimeError("invalid checkpoint")
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.predict(self.frame + 1), [])
        self.load.assert_called_once()
        self.assertEqual(self.service.status, "unavailable")
        self.assertIn("invalid checkpoint", self.service.warning)
        self.assertEqual(len(self.service._cache), 0)
        self.load.side_effect = None
        self.weight.write_bytes(b"replacement valid mocked checkpoint")
        self.assertTrue(self.service.predict(self.frame))
        self.assertEqual(self.load.call_count, 2)
        self.assertEqual(self.service.status, "ready")
        self.assertIsNone(self.service.warning)

    def test_missing_runtime_is_reported_not_installed(self):
        self.load.side_effect = ImportError("ultralytics is unavailable")
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "unavailable")
        self.assertIn("ultralytics is unavailable", self.service.warning)
        self.detector.predict.assert_not_called()

    def test_fallback_only_uses_existing_models_dir_then_cwd_checkpoint(self):
        cached = self.models / "yolov8n.pt"
        cwd_cached = self.cwd / "yolov8n.pt"
        cached.write_bytes(b"models cache")
        cwd_cached.write_bytes(b"cwd cache")
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": ""}):
            service = ds.DetectionService()
            service.predict(self.frame)
            self.load.assert_called_with(str(cached.resolve()), "detect")
            cached.unlink()
            service.predict(self.frame)
            self.load.assert_called_with(str(cwd_cached.resolve()), "detect")
            cwd_cached.unlink()
            self.assertEqual(service.predict(self.frame), [])
            self.assertEqual(service.status, "unavailable")
            self.assertIn("downloads are disabled", service.warning)
        self.assertEqual(self.load.call_count, 2)

    def test_deleted_checkpoint_cannot_serve_stale_cache(self):
        self.service.predict(self.frame)
        self.weight.unlink()
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "unavailable")
        self.assertTrue(self.service.warning)
        self.detector.predict.assert_called_once()

    def test_pose_is_explicit_separate_reused_and_normalized(self):
        pose_model = model("pose", [result([[0, 0, 200, 100]], points=[[[100, 50, 0.8], [-10, 200, 1.2]]])])
        self.load.side_effect = [self.detector, pose_model]
        with patch.dict(os.environ, {"YOLO_POSE_MODEL_PATH": str(self.pose_weight)}):
            service = ds.DetectionService()
        ordinary = service.predict(self.frame)
        self.assertNotIn("keypoints", ordinary[0])
        self.load.assert_called_once()
        pose_output = service.predict(self.frame, pose=True)
        self.assertEqual(pose_output[0]["keypoints"], [[0.5, 0.5, 0.8], [0.0, 1.0, 1.0]])
        self.assertEqual(pose_output[0]["box"], [0.0, 0.0, 1.0, 1.0])
        pose_output[0]["keypoints"][0][0] = 0.99
        self.assertEqual(service.predict(self.frame, pose=True)[0]["keypoints"][0][0], 0.5)
        service.predict(self.frame + 1, pose=True)
        service.predict(self.frame + 1)
        self.assertEqual(self.load.call_count, 2)
        self.load.assert_called_with(str(self.pose_weight.resolve()), "pose")
        self.assertEqual(self.detector.predict.call_count, 2)
        self.assertEqual(pose_model.predict.call_count, 2)
        self.cuda.assert_called_once()

    def test_missing_pose_never_substitutes_detection_and_does_not_poison_detection(self):
        self.assertEqual(self.service.predict(self.frame, pose=True), [])
        self.assertEqual(self.service.status, "unavailable")
        self.assertIn("YOLO_POSE_MODEL_PATH", self.service.warning)
        self.load.assert_not_called()
        self.assertTrue(self.service.predict(self.frame))
        self.assertEqual(self.service.status, "ready")
        self.assertIsNone(self.service.warning)
        self.assertEqual(self.service.predict(self.frame, pose=True), [])
        self.assertTrue(self.service.predict(self.frame))
        self.assertIsNone(self.service.warning)

    def test_wrong_checkpoint_task_is_unavailable(self):
        self.detector.task = "classify"
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "unavailable")
        self.assertIn("requested detect task", self.service.warning)
        self.detector.predict.assert_not_called()

    def test_normalization_clamps_finite_boxes_and_supports_names_list(self):
        self.detector.predict.return_value = [result(
            [[-10, -20, 250, 110], [20, 10, 100, 50], [np.nan, 0, 20, 30],
             [0, 0, np.inf, 20], [80, 10, 20, 40], [250, 0, 300, 50],
             [10, 10, 10, 30], [0, 0, 20, 20], [0, 0, 20, 20], [0, 0, 20, 20]],
            classes=[1, 7, 0, 0, 0, 0, 0, np.nan, 0.5, 0],
            scores=[1.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, np.inf],
            names=["person", "dog"],
        )]
        output = self.service.predict(self.frame)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0], {"box": [0.0, 0.0, 1.0, 1.0], "class_id": 1, "label": "dog", "confidence": 1.0})
        self.assertEqual(output[1]["label"], "7")
        self.assertEqual(self.service.status, "ready")
        self.assertTrue(all(0 <= value <= 1 for item in output for value in item["box"]))

    def test_tensor_like_results_are_moved_to_cpu_without_importing_torch(self):
        output = result([[0, 0, 100, 50]], names={"0": "subject"})
        for attribute in ("xyxy", "cls", "conf"):
            array = getattr(output.boxes, attribute)
            tensor = Mock()
            tensor.detach.return_value = tensor
            tensor.cpu.return_value = tensor
            tensor.numpy.return_value = array
            setattr(output.boxes, attribute, tensor)
        self.detector.predict.return_value = [output]
        self.assertEqual(self.service.predict(self.frame)[0]["label"], "subject")
        output.boxes.xyxy.cpu.assert_called_once()

    def test_invalid_pose_points_are_omitted_without_changing_detection(self):
        pose_model = model("pose", [result([[0, 0, 100, 50]], points=[[[np.nan, 10, 0.8]]])])
        self.load.return_value = pose_model
        with patch.dict(os.environ, {"YOLO_POSE_MODEL_PATH": str(self.pose_weight)}):
            output = ds.DetectionService().predict(self.frame, pose=True)
        self.assertEqual(len(output), 1)
        self.assertNotIn("keypoints", output[0])

    def test_detection_count_is_bounded(self):
        self.detector.predict.return_value = [result([[0, 0, 20, 20]] * (ds._MAX_DETECTIONS + 20))]
        output = self.service.predict(self.frame)
        self.assertEqual(len(output), ds._MAX_DETECTIONS)
        self.assertEqual(self.detector.predict.call_args.kwargs["max_det"], ds._MAX_DETECTIONS)

    def test_cuda_failure_retries_cpu_once_and_preserves_reason_on_cache_hit(self):
        self.cuda.return_value = True
        self.detector.predict.side_effect = [RuntimeError("CUDA out of memory"), [result()], [result()]]
        self.assertEqual(self.service.predict(self.frame), [])
        devices = [call.kwargs["device"] for call in self.detector.predict.call_args_list]
        self.assertEqual(devices, ["cuda", "cpu"])
        self.detector.to.assert_called_once_with("cpu")
        self.assertEqual(self.service.status, "ready_cpu_fallback")
        self.assertIn("CUDA out of memory", self.service.warning)
        self.service.predict(self.frame)
        self.assertEqual(self.detector.predict.call_count, 2)
        self.assertIn("using CPU", self.service.warning)
        self.service.predict(self.frame + 1)
        self.assertEqual(self.detector.predict.call_args.kwargs["device"], "cpu")
        self.cuda.assert_called_once()
        self.load.assert_called_once()

    def test_cuda_error_during_tensor_transfer_also_retries_on_cpu(self):
        self.cuda.return_value = True
        output = result([[0, 0, 100, 50]])
        tensor = Mock()
        tensor.detach.return_value = tensor
        tensor.cpu.side_effect = RuntimeError("CUDA device-side assert triggered")
        output.boxes.xyxy = tensor
        self.detector.predict.side_effect = [[output], [result()]]
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.service.status, "ready_cpu_fallback")
        self.assertIn("device-side assert", self.service.warning)
        self.assertEqual(self.detector.predict.call_count, 2)
        self.detector.to.assert_called_once_with("cpu")

    def test_cpu_retry_failure_is_reported_and_not_cached(self):
        self.cuda.return_value = True
        self.detector.predict.side_effect = [RuntimeError("cuDNN execution failed"), RuntimeError("CPU also failed"), [result()]]
        self.assertEqual(self.service.predict(self.frame), [])
        self.assertEqual(self.detector.predict.call_count, 2)
        self.assertEqual(self.service.status, "error")
        self.assertIn("cuDNN execution failed", self.service.warning)
        self.assertIn("CPU also failed", self.service.warning)
        self.assertEqual(len(self.service._cache), 0)
        self.service.predict(self.frame)
        self.assertEqual(self.detector.predict.call_count, 3)
        self.assertEqual(self.service.status, "ready_cpu_fallback")
        self.assertNotIn("CPU also failed", self.service.warning)

    def test_cuda_fallback_moves_both_loaded_tasks_off_gpu(self):
        self.cuda.return_value = True
        pose_model = model("pose", [result()])
        self.load.side_effect = [self.detector, pose_model]
        with patch.dict(os.environ, {"YOLO_POSE_MODEL_PATH": str(self.pose_weight)}):
            service = ds.DetectionService()
        service.predict(self.frame)
        service.predict(self.frame, pose=True)
        self.detector.predict.side_effect = [RuntimeError("CUDA kernel failure"), [result()]]
        service.predict(self.frame + 1)
        self.detector.to.assert_called_once_with("cpu")
        pose_model.to.assert_called_once_with("cpu")
        service.predict(self.frame, pose=True)
        self.assertEqual(pose_model.predict.call_args.kwargs["device"], "cpu")
        self.assertEqual(self.load.call_count, 2)

    def test_cpu_or_non_cuda_errors_are_not_retried(self):
        for cuda, error in ((False, RuntimeError("CUDA out of memory")), (True, RuntimeError("invalid tensor shape"))):
            with self.subTest(cuda=cuda):
                self.cuda.return_value = cuda
                self.detector.predict.reset_mock()
                self.detector.predict.side_effect = error
                service = ds.DetectionService()
                self.assertEqual(service.predict(self.frame), [])
                self.detector.predict.assert_called_once()
                self.assertEqual(service.status, "error")
                self.assertIn(str(error), service.warning)
        self.detector.to.assert_not_called()

    def test_explicit_cuda_unavailable_reports_cpu_fallback(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "cuda"}):
            service = ds.DetectionService()
        service.predict(self.frame)
        service.predict(self.frame + 1)
        self.assertEqual(service.status, "ready_cpu_fallback")
        self.assertIn("CUDA requested but unavailable", service.warning)
        self.assertEqual(self.detector.predict.call_args.kwargs["device"], "cpu")
        self.cuda.assert_called_once()

    def test_explicit_cpu_stays_cpu_even_when_cuda_exists(self):
        self.cuda.return_value = True
        with patch.dict(os.environ, {"YOLO_DEVICE": "cpu"}):
            service = ds.DetectionService()
        service.predict(self.frame)
        self.assertEqual(self.detector.predict.call_args.kwargs["device"], "cpu")
        self.assertEqual(service.status, "ready")
        self.assertIsNone(service.warning)

    def test_cuda_probe_failure_is_reported_and_only_probed_once(self):
        self.cuda.side_effect = RuntimeError("driver unavailable")
        self.service.predict(self.frame)
        self.service.predict(self.frame + 1)
        self.cuda.assert_called_once()
        self.assertEqual(self.service.status, "ready_cpu_fallback")
        self.assertIn("driver unavailable", self.service.warning)

    def test_invalid_frames_do_not_load_models(self):
        for frame in ("frame.jpg", [], np.zeros((0, 20, 3)), np.zeros((20, 20)), np.zeros((20, 20, 4)), np.empty((2, 2, 3), dtype=object)):
            with self.subTest(frame_type=type(frame)):
                self.assertEqual(self.service.predict(frame), [])
                self.assertEqual(self.service.status, "error")
                self.assertIn("numpy frame", self.service.warning)
        self.load.assert_not_called()
        self.cuda.assert_not_called()

    def test_concurrent_requests_share_one_load_and_serialized_inference(self):
        barrier = threading.Barrier(8)
        active = 0
        maximum_active = 0
        count_lock = threading.Lock()

        def infer(**kwargs):
            nonlocal active, maximum_active
            self.assertTrue(self.service._lock.locked())
            with count_lock:
                active += 1
                maximum_active = max(active, maximum_active)
            output = [result([[0, 0, 20, 20]])]
            with count_lock:
                active -= 1
            return output

        def predict(index):
            barrier.wait(timeout=10)
            return self.service.predict(self.frame + index)

        self.detector.predict.side_effect = infer
        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(predict, range(8)))
        self.assertTrue(all(outputs))
        self.assertEqual(maximum_active, 1)
        self.load.assert_called_once()
        self.cuda.assert_called_once()
        self.assertEqual(self.detector.predict.call_count, 8)


if __name__ == "__main__":
    unittest.main()
