"""SAM 2 service contracts, using synthetic captures and a torch-free predictor."""
import copy
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_worker import segmentation_service as service


def box_prompt(t=0.0, object_id=1):
    return {"object_id": object_id, "t": t, "box": [0.1, 0.2, 0.3, 0.4]}


class SyntheticCapture:
    def __init__(self, width=1000, height=500, fps=12.0, count=240):
        self.width, self.height, self.fps, self.count = width, height, fps, count
        self.index = 0
        self.released = False
        self.read_indices = []
        self.pts = None
        self.fail_read = False
        self.fail_seek = False
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, prop):
        values = {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                  cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count}
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts(self.index) if self.pts else 1000.0 * self.index / self.fps
        return values[prop]

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.index = int(value)
        return not self.fail_seek

    def read(self):
        self.read_indices.append(self.index)
        if self.fail_read or self.index >= self.count:
            return False, None
        return True, np.full((self.height, self.width, 3), self.index % 256, dtype=np.uint8)

    def release(self):
        self.released = True


class FakePredictor:
    def __init__(self):
        self.states = []
        self.frame_dirs = []
        self.prompts = []
        self.propagations = []
        self.events = []
        self.reset_count = 0
        self.closed_count = 0
        self.behavior = {}
        self.skip_frames = set()
        self.raise_frame = None
        self.init_error = None
        self.prompt_error = None
        self.reset_error = None
        self.init_hook = None
        self.lock_checks = []

    def init_state(self, video_path, offload_video_to_cpu, offload_state_to_cpu):
        self.lock_checks.append(service._LOCK.locked())
        assert offload_video_to_cpu is True
        assert offload_state_to_cpu is True
        directory = Path(video_path)
        self.frame_dirs.append(directory)
        names = sorted(directory.iterdir())
        assert 1 <= len(names) <= service.MAX_FRAMES
        assert [p.name for p in names] == [f"{i:05d}.jpg" for i in range(len(names))]
        image = cv2.imread(str(names[0]))
        h, w = image.shape[:2]
        assert max(h, w) <= service.MAX_SIDE
        if self.init_error:
            raise self.init_error
        state = {"height": h, "width": w, "num_frames": len(names), "ids": [],
                 "images": object(), "cached_features": object()}
        assert all(previous == {} for previous in self.states), "overlapping predictor operations"
        self.states.append(state)
        if self.init_hook:
            self.init_hook()
        return state

    def add_new_points_or_box(self, **kwargs):
        self.lock_checks.append(service._LOCK.locked())
        if self.prompt_error:
            raise self.prompt_error
        state = kwargs["inference_state"]
        oid = kwargs["obj_id"]
        if oid not in state["ids"]:
            state["ids"].append(oid)
        self.prompts.append({k: copy.deepcopy(v) for k, v in kwargs.items() if k != "inference_state"})
        self.events.append(("prompt", kwargs["frame_idx"], oid))
        return kwargs["frame_idx"], state["ids"], None

    def propagate_in_video(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse):
        self.lock_checks.append(service._LOCK.locked())
        assert reverse is False
        stop = start_frame_idx + max_frame_num_to_track + 1
        assert 0 <= start_frame_idx < stop <= inference_state["num_frames"]
        self.propagations.append((start_frame_idx, max_frame_num_to_track))
        try:
            for index in range(start_frame_idx, stop):
                self.events.append(("frame", index))
                if index == self.raise_frame:
                    raise RuntimeError("synthetic inference failure")
                if index in self.skip_frames:
                    continue
                ids, logits = [], []
                # Reverse IDs so tests cannot accidentally assume positional object IDs.
                for oid in reversed(inference_state["ids"]):
                    mode = self.behavior.get((index, oid), "tracked")
                    if mode == "missing":
                        continue
                    ids.append(oid)
                    if mode == "none":
                        logits.append(None)
                        continue
                    shape = (1, inference_state["height"], inference_state["width"])
                    mask = np.full(shape, -2.0, dtype=np.float32)
                    if mode == "tracked":
                        mask[:, 2:5, 4 + oid:8 + oid] = 1.0
                    elif mode == "nan":
                        mask[:, 0, 0] = np.nan
                    elif mode == "bad_shape":
                        mask = np.ones((2, 2), dtype=np.float32)
                    logits.append(mask)
                yield index, ids, logits
        finally:
            self.closed_count += 1

    def reset_state(self, state):
        self.lock_checks.append(service._LOCK.locked())
        self.reset_count += 1
        if self.reset_error:
            raise self.reset_error
        state["ids"].clear()  # Service must also release images and cached_features.


class TestSegmentationService(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"synthetic source identity; capture is mocked")
        self.checkpoint = self.root / "weights.pt"
        self.checkpoint.write_bytes(b"test checkpoint identity, not real weights")
        self.config = self.root / "local_sam2.yaml"
        self.config.write_text("test-config", encoding="utf-8")
        self.predictor = FakePredictor()
        self.builder = Mock(return_value=self.predictor)
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=Mock(return_value=False))
        torch.inference_mode = Mock(side_effect=nullcontext)
        self.torch = torch
        sam2 = types.ModuleType("sam2")
        sam2.__file__ = str(self.root / "__init__.py")
        builder_module = types.ModuleType("sam2.build_sam")
        builder_module.build_sam2_video_predictor = self.builder
        self.captures = []
        self.capture_factory = SyntheticCapture

        def open_capture(_path):
            capture = self.capture_factory()
            self.captures.append(capture)
            return capture

        patches = [
            patch.dict(sys.modules, {"torch": torch, "sam2": sam2, "sam2.build_sam": builder_module}),
            patch.dict(os.environ, {"SAM2_MODEL_PATH": str(self.checkpoint),
                                    "SAM2_CONFIG": self.config.name, "SAM2_DEVICE": "auto"}),
            patch.object(service, "WORK_DIR", str(self.root)),
            patch.object(service, "_PREDICTOR", None),
            patch.object(service, "_PREDICTOR_KEY", None),
            patch.object(cv2, "VideoCapture", side_effect=open_capture),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, start=0.0, end=2.0, prompts=None, **kwargs):
        return service.segment_video(str(self.source), start, end,
                                     [box_prompt(start)] if prompts is None else prompts, **kwargs)

    def objects(self, result, oid=1):
        return [next(obj for obj in frame["objects"] if obj["object_id"] == oid)
                for frame in result["frames"]]

    def assert_clean(self):
        self.assertTrue(all(capture.released for capture in self.captures))
        self.assertTrue(all(not directory.exists() for directory in self.predictor.frame_dirs))
        self.assertTrue(all(state == {} for state in self.predictor.states))
        root = self.root / "segmentation"
        if root.exists():
            self.assertFalse(list(root.glob(".pending-*")))

    def assert_unavailable(self, result):
        self.assertFalse(result["available"], result)
        self.assertTrue(result["reason"])
        self.assertEqual(result["frames"], [])
        self.assert_clean()

    def test_import_does_not_import_torch_sam_or_cv2(self):
        script = ("import sys; from local_worker import segmentation_service; "
                  "assert 'torch' not in sys.modules; assert 'sam2' not in sys.modules; "
                  "assert 'cv2' not in sys.modules")
        environment = dict(os.environ, CLIPFORGE_WORK_DIR=str(self.root / "import-only"))
        result = subprocess.run([sys.executable, "-c", script], cwd=str(REPO_ROOT),
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_api_pixel_box_timestamps_metadata_and_compressed_masks(self):
        progress = []
        result = self.call(start=2.0, end=4.0, progress_cb=progress.append)
        self.assertTrue(result["available"], result)
        self.builder.assert_called_once_with(config_file=self.config.name,
                                             ckpt_path=str(self.checkpoint), device="cpu")
        self.assertEqual((result["original_width"], result["original_height"]), (1000, 500))
        self.assertEqual((result["mask_width"], result["mask_height"]), (512, 256))
        expected_times = [i / 12.0 for i in (24, 28, 32, 36, 40, 44)]
        np.testing.assert_allclose(result["sampled_timestamps"], expected_times)
        self.assertEqual([f["t"] for f in result["frames"]], result["sampled_timestamps"])
        prompt = self.predictor.prompts[0]
        np.testing.assert_allclose(prompt["box"], [51.2, 51.2, 204.8, 153.6])
        self.assertTrue(prompt["normalize_coords"])
        self.assertTrue(prompt["clear_old_points"])
        self.assertEqual(self.predictor.propagations, [(0, 5)])
        item = self.objects(result)[0]
        self.assertEqual(item["status"], "tracked")
        self.assertEqual(item["area"], 12)
        np.testing.assert_allclose(item["bounds"], [5 / 512, 2 / 256, 4 / 512, 3 / 256])
        self.assertAlmostEqual(item["area_fraction"], 12 / (512 * 256))
        self.assertNotIn("confidence", item)
        self.assertFalse(Path(item["mask_ref"]).is_absolute())
        with np.load(self.root / item["mask_ref"], allow_pickle=False) as archive:
            mask = archive[item["mask_key"]]
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(mask.shape, (256, 512))
            self.assertEqual(int(mask.sum()), 12)
        metadata = json.loads((self.root / result["metadata_ref"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata, result)
        self.assertEqual(progress[0], 0.0)
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(progress, sorted(progress))
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in progress))
        self.assertTrue(all(self.predictor.lock_checks))
        self.assertEqual(self.predictor.reset_count, 1)
        self.assert_clean()

    def test_points_use_pixels_and_original_labels(self):
        result = self.call(prompts=[{"object_id": 9, "t": 0, "points": [[0, 1], [0.5, 0.25]],
                                     "labels": [0, 1]}])
        self.assertTrue(result["available"], result)
        prompt = self.predictor.prompts[0]
        self.assertEqual(prompt["points"], [[0.0, 256.0], [256.0, 64.0]])
        self.assertEqual(prompt["labels"], [0, 1])
        self.assertNotIn("box", prompt)

    def test_ten_second_bound_sample_rate_and_resize(self):
        result = self.call(end=10)
        self.assertTrue(result["available"], result)
        self.assertEqual(len(result["frames"]), 30)
        self.assertLessEqual(len(result["frames"]), service.MAX_FRAMES)
        self.assertEqual(self.captures[0].read_indices, list(range(0, 120, 4)))
        self.assertEqual(self.predictor.propagations, [(0, 29)])
        self.assertLess(max(result["sampled_timestamps"]), 10)
        self.assert_clean()

    def test_small_portrait_video_is_not_upscaled(self):
        self.capture_factory = lambda: SyntheticCapture(width=40, height=80)
        result = self.call()
        self.assertEqual((result["mask_width"], result["mask_height"]), (40, 80))

    def test_low_fps_deduplicates_samples_and_short_interval_is_inclusive_distance_zero(self):
        self.capture_factory = lambda: SyntheticCapture(fps=1, count=20)
        result = self.call(end=0.1)
        self.assertTrue(result["available"], result)
        self.assertEqual(result["sampled_timestamps"], [0.0])
        self.assertEqual(self.predictor.propagations, [(0, 0)])

    def test_records_decoded_pts_not_requested_grid_time(self):
        def factory():
            cap = SyntheticCapture(fps=10)
            cap.pts = lambda index: index * 100.0 + 5.0
            return cap
        self.capture_factory = factory
        result = self.call(end=1)
        self.assertTrue(result["available"], result)
        np.testing.assert_allclose(result["sampled_timestamps"], [0.005, 0.305, 0.705])
        self.assertEqual(result["prompt_frames"][0]["sampled_t"], 0.005)

    def test_lost_stays_lost_until_supplied_later_prompt(self):
        self.predictor.behavior[(1, 1)] = "empty"
        result = self.call(prompts=[box_prompt(0), box_prompt(1)])
        self.assertEqual([o["status"] for o in self.objects(result)],
                         ["tracked", "lost", "lost", "tracked", "tracked", "tracked"])
        self.assertEqual(self.predictor.propagations, [(0, 2), (3, 2)])
        self.assertLess(self.predictor.events.index(("frame", 2)),
                        self.predictor.events.index(("prompt", 3, 1)))
        for item in self.objects(result)[1:3]:
            self.assertIsNone(item["bounds"])
            self.assertIsNone(item["mask_ref"])
            self.assertIsNone(item["mask_key"])
            self.assertEqual(item["area"], 0)
        self.assertEqual(len(self.predictor.prompts), 2)

    def test_missing_nonfinite_wrong_shape_and_none_masks_are_lost_not_boxes(self):
        for mode in ("missing", "nan", "bad_shape", "none", "empty"):
            with self.subTest(mode=mode):
                self.predictor.behavior[(0, 1)] = mode
                prompt = box_prompt()
                prompt["box"][0] += ("missing", "nan", "bad_shape", "none", "empty").index(mode) / 100
                result = self.call(prompts=[prompt])
                self.assertTrue(result["available"], result)
                self.assertTrue(all(o["status"] == "lost" for o in self.objects(result)))
                self.assertTrue(all(o["mask_ref"] is None for o in self.objects(result)))
                with np.load(self.root / "segmentation" / result["cache_id"] / "masks.npz",
                             allow_pickle=False) as archive:
                    self.assertEqual(archive.files, [])

    def test_missing_propagated_frame_reports_loss_and_preserves_full_timeline(self):
        self.predictor.skip_frames = {1, 5}
        result = self.call()
        self.assertEqual([f["frame_index"] for f in result["frames"]], list(range(6)))
        self.assertEqual([o["status"] for o in self.objects(result)], ["tracked"] + ["lost"] * 5)

    def test_objects_begin_only_at_own_prompt_and_output_id_order_is_respected(self):
        result = self.call(prompts=[box_prompt(0, 1), box_prompt(1, 2)])
        self.assertEqual([o["status"] for o in self.objects(result, 2)],
                         ["not_prompted"] * 3 + ["tracked"] * 3)
        self.assertAlmostEqual(self.objects(result, 1)[3]["bounds"][0], 5 / 512)
        self.assertAlmostEqual(self.objects(result, 2)[3]["bounds"][0], 6 / 512)
        self.assertEqual(self.predictor.prompts[1]["frame_idx"], 3)

    def test_segment_before_first_prompt_is_not_backfilled(self):
        result = self.call(prompts=[box_prompt(1)])
        self.assertEqual([o["status"] for o in self.objects(result)],
                         ["not_prompted"] * 3 + ["tracked"] * 3)
        self.assertEqual(self.predictor.propagations, [(3, 2)])

    def test_prompt_snapping_and_collisions(self):
        result = self.call(prompts=[box_prompt(0.51)])
        self.assertEqual(result["prompt_frames"][0]["frame_index"], 2)
        calls = self.builder.call_count
        with self.assertRaisesRegex(ValueError, "same sampled frame"):
            self.call(prompts=[box_prompt(0.01), box_prompt(0.02)])
        self.assertEqual(self.builder.call_count, calls)
        self.assert_clean()

    def test_invalid_segment_values_and_source_duration(self):
        for start, end in [(-1, 1), (1, 1), (2, 1), (0, 10.00001), (0, math.nan),
                           (math.inf, 1), (True, 1), ("0", 1), (0, math.inf), (19, 21)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.call(start=start, end=end)
        self.builder.assert_not_called()
        self.assert_clean()

    def test_invalid_prompt_shapes_ranges_nans_and_counts(self):
        valid = box_prompt()
        invalid = [None, {}, [], [None], [valid] * 65,
                   [dict(valid, object_id=True)], [dict(valid, object_id=-1)],
                   [dict(valid, object_id=1.0)], [dict(valid, object_id=2**31)],
                   [dict(valid, t=math.nan)], [dict(valid, t=math.inf)],
                   [dict(valid, t=-0.1)], [dict(valid, t=2)],
                   [dict(valid, box=[0, 0, 1])], [dict(valid, box=[0, 0, 0, 1])],
                   [dict(valid, box=[0.8, 0, 0.3, 0.1])],
                   [dict(valid, box=[math.nan, 0, 0.1, 0.1])],
                   [dict(valid, box=[True, 0, 0.1, 0.1])],
                   [dict(valid, box=[0, -0.1, 0.1, 0.1])],
                   [dict(valid, points=[[0, 0]], labels=[1])],
                   [dict(valid, unexpected="path")], [valid, valid],
                   [{"object_id": 1, "t": 0}],
                   [box_prompt(0, oid) for oid in range(9)]]
        base = {"object_id": 1, "t": 0, "points": [[0.5, 0.5]], "labels": [1]}
        for changes in ({"points": []}, {"labels": []}, {"labels": [2]}, {"labels": [True]},
                        {"points": [[0.1]]}, {"points": [[math.inf, 0]]},
                        {"points": [[1.1, 0]]}, {"points": [[0, 0]] * 33, "labels": [1] * 33}):
            invalid.append([dict(base, **changes)])
        invalid.append([dict(base, t=i / 10, points=[[0, 0]] * 32, labels=[1] * 32)
                        for i in range(9)])
        for prompts in invalid:
            with self.subTest(prompts=repr(prompts)[:120]), self.assertRaises(ValueError):
                service.segment_video(str(self.source), 0, 2, prompts)
        self.builder.assert_not_called()
        self.assertEqual(self.captures, [])

    def test_rejects_remote_missing_source_paths_and_noncallable_callbacks(self):
        for path in (None, "", "https://example.com/video.mp4", str(self.root / "missing.mp4"), str(self.root)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                service.segment_video(path, 0, 1, [box_prompt()])
        with self.assertRaises(ValueError):
            self.call(progress_cb=3)
        with self.assertRaises(ValueError):
            self.call(is_cancelled=True)

    def test_missing_weights_configuration_package_and_cuda_are_unavailable(self):
        cases = [{"SAM2_MODEL_PATH": ""}, {"SAM2_CONFIG": ""},
                 {"SAM2_MODEL_PATH": str(self.root / "not-here.pt")},
                 {"SAM2_MODEL_PATH": "https://example.com/model.pt"},
                 {"SAM2_CONFIG": "https://example.com/config.yaml"},
                 {"SAM2_DEVICE": "mps"}, {"SAM2_DEVICE": "cuda"}]
        for environment in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                self.assert_unavailable(self.call())
        with patch.dict(sys.modules, {"sam2.build_sam": None}):
            self.assert_unavailable(self.call())
        self.checkpoint.write_bytes(b"")
        self.assert_unavailable(self.call())
        self.builder.assert_not_called()

    def test_invalid_input_is_rejected_even_when_sam_is_not_configured(self):
        with patch.dict(os.environ, {"SAM2_MODEL_PATH": ""}):
            with self.assertRaises(ValueError):
                self.call(start=19, end=21)

    def test_model_loading_errors_are_graceful_and_do_not_publish(self):
        for error in (ValueError("bad config"), MemoryError("out of RAM"), RuntimeError("out of VRAM")):
            with self.subTest(error=error):
                self.builder.side_effect = error
                self.assert_unavailable(self.call())
                self.assertEqual(list((self.root / "segmentation").iterdir()), [])
                self.assertIsNone(service._PREDICTOR)

    def test_cache_hit_skips_sampling_and_inference(self):
        first = self.call()
        second = self.call()
        self.assertEqual(first["cache_id"], second["cache_id"])
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.builder.assert_called_once()
        self.assertEqual(len(self.predictor.states), 1)
        self.assertEqual(self.captures[1].read_indices, [])
        self.assert_clean()

    def test_predictor_reused_but_cache_key_tracks_source_checkpoint_config_and_request(self):
        first = self.call()
        other_request = self.call(start=2, end=4)
        self.assertNotEqual(first["cache_id"], other_request["cache_id"])
        self.assertEqual(self.builder.call_count, 1)
        self.source.write_bytes(b"changed source stat")
        changed_source = self.call()
        self.assertNotEqual(first["cache_id"], changed_source["cache_id"])
        self.assertEqual(self.builder.call_count, 1)
        self.checkpoint.write_bytes(b"changed checkpoint stat")
        changed_weights = self.call()
        self.assertNotEqual(changed_source["cache_id"], changed_weights["cache_id"])
        self.assertEqual(self.builder.call_count, 2)
        self.config.write_text("changed config stat", encoding="utf-8")
        changed_config = self.call()
        self.assertNotEqual(changed_weights["cache_id"], changed_config["cache_id"])
        self.assertEqual(self.builder.call_count, 3)
        self.torch.cuda.is_available.return_value = True
        changed_device = self.call()
        self.assertNotEqual(changed_config["cache_id"], changed_device["cache_id"])
        self.assertEqual(self.builder.call_count, 4)
        self.assertEqual(self.builder.call_args.kwargs["device"], "cuda")

    def test_cache_is_bounded_to_eight_complete_entries(self):
        ids = []
        for i in range(10):
            prompt = box_prompt()
            prompt["box"][0] = i / 20
            result = self.call(end=1, prompts=[prompt])
            self.assertTrue(result["available"], result)
            ids.append(result["cache_id"])
        directories = list((self.root / "segmentation").iterdir())
        self.assertEqual(len(directories), 8)
        self.assertEqual({p.name for p in directories}, set(ids[-8:]))
        self.assertTrue(all((p / "metadata.json").is_file() and (p / "masks.npz").is_file()
                            for p in directories))
        self.builder.assert_called_once()
        self.assert_clean()

    def test_missing_mask_archive_is_rebuilt_atomically(self):
        first = self.call()
        archive = self.root / "segmentation" / first["cache_id"] / "masks.npz"
        archive.unlink()
        second = self.call()
        self.assertFalse(second["cache_hit"])
        self.assertTrue(archive.is_file())
        self.assertEqual(first["cache_id"], second["cache_id"])
        self.assertEqual(self.predictor.reset_count, 2)

    def test_cancellation_before_work_and_while_waiting_for_lock(self):
        self.assertTrue(self.call(is_cancelled=lambda: True)["cancelled"])
        self.assertEqual(self.captures, [])
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls > 1

        with service._LOCK:
            result = self.call(is_cancelled=cancelled)
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.captures, [])

    def test_cancellation_during_sampling_cleans_temporary_frames(self):
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = value >= 0.15

        result = self.call(progress_cb=progress, is_cancelled=lambda: cancelled)
        self.assertTrue(result["cancelled"])
        self.builder.assert_not_called()
        self.assert_unavailable(result)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_cancellation_during_propagation_closes_generator_and_clears_state(self):
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = value >= 0.4

        result = self.call(progress_cb=progress, is_cancelled=lambda: cancelled)
        self.assertTrue(result["cancelled"])
        self.assert_unavailable(result)
        self.assertEqual(self.predictor.reset_count, 1)
        self.assertEqual(self.predictor.closed_count, 1)
        self.assertIsNone(service._PREDICTOR)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_exception_during_propagation_closes_generator_and_releases_state(self):
        self.predictor.raise_frame = 1
        self.assert_unavailable(self.call())
        self.assertEqual(self.predictor.reset_count, 1)
        self.assertEqual(self.predictor.closed_count, 1)
        self.assertIsNone(service._PREDICTOR)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_init_prompt_and_reset_failures_cleanup(self):
        self.predictor.init_error = MemoryError("init failed")
        self.assert_unavailable(self.call())
        self.assertEqual(self.predictor.reset_count, 0)
        self.predictor.init_error = None
        self.predictor.prompt_error = RuntimeError("prompt failed")
        self.assert_unavailable(self.call())
        self.assertEqual(self.predictor.reset_count, 1)
        self.predictor.prompt_error = None
        self.predictor.reset_error = RuntimeError("reset failed")
        self.assert_unavailable(self.call())
        self.assertIsNone(service._PREDICTOR)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_archive_failure_leaves_no_published_entry(self):
        with patch.object(np, "savez_compressed", side_effect=OSError("disk full")):
            self.assert_unavailable(self.call())
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_cancellation_immediately_before_publish_removes_staged_products(self):
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = value >= 0.98

        result = self.call(progress_cb=progress, is_cancelled=lambda: cancelled)
        self.assertTrue(result["cancelled"])
        self.assert_unavailable(result)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])
        self.assertEqual(self.predictor.reset_count, 1)

    def test_completion_is_commit_point_for_cancellation(self):
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = value == 1.0

        result = self.call(progress_cb=progress, is_cancelled=lambda: cancelled)
        self.assertTrue(result["available"], result)
        self.assertTrue((self.root / result["metadata_ref"]).is_file())
        self.assert_clean()

    def test_tensor_style_logits_use_cpu_conversion_and_strict_zero_threshold(self):
        values = np.array([[0, 0.1], [-0.1, 2]], dtype=np.float32)
        tensor = Mock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = values[None, :, :]
        mask = service._as_mask(tensor, 2, 2, np)
        np.testing.assert_array_equal(mask, [[False, True], [False, True]])
        tensor.detach.assert_called_once()
        tensor.detach.return_value.cpu.assert_called_once()
        tensor.detach.return_value.cpu.return_value.numpy.assert_called_once()

    def test_out_of_range_predictor_frame_is_unavailable_and_generator_closed(self):
        closed = []

        def invalid_propagation(**kwargs):
            try:
                yield 32, [], None
            finally:
                closed.append(True)

        with patch.object(self.predictor, "propagate_in_video", side_effect=invalid_propagation):
            self.assert_unavailable(self.call())
        self.assertEqual(closed, [True])
        self.assertEqual(self.predictor.reset_count, 1)
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_changed_source_during_operation_is_not_published(self):
        def progress(value):
            if value > 0.5:
                self.source.write_bytes(b"modified while inference was running")
        result = self.call(progress_cb=progress)
        self.assert_unavailable(result)
        self.assertIn("changed during", result["reason"])
        self.assertEqual(list((self.root / "segmentation").iterdir()), [])

    def test_decoder_failure_and_untrustworthy_timestamps_are_not_fabricated(self):
        for mode in ("read", "seek", "nan_pts", "zero_pts", "outside_pts"):
            def factory():
                cap = SyntheticCapture()
                cap.fail_read = mode == "read"
                cap.fail_seek = mode == "seek"
                if mode == "nan_pts":
                    cap.pts = lambda index: math.nan
                elif mode == "zero_pts":
                    cap.pts = lambda index: 0
                elif mode == "outside_pts":
                    cap.pts = lambda index: 30000
                return cap
            self.capture_factory = factory
            with self.subTest(mode=mode):
                self.assert_unavailable(self.call())
        self.builder.assert_not_called()

    def test_invalid_video_metadata_is_rejected(self):
        for changes in ({"fps": 0}, {"fps": math.nan}, {"width": 0}, {"count": math.inf}):
            self.capture_factory = lambda: SyntheticCapture(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call()
        self.assert_clean()

    def test_operations_are_serialized_and_singleton_is_reused_across_threads(self):
        entered = threading.Event()
        release = threading.Event()
        results, errors = [], []

        def hook():
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test timed out waiting for release")
        self.predictor.init_hook = hook

        def run(start):
            try:
                results.append(self.call(start=start, end=start + 1))
            except Exception as exc:
                errors.append(exc)
        first = threading.Thread(target=run, args=(0,))
        second = threading.Thread(target=run, args=(1,))
        first.start()
        try:
            self.assertTrue(entered.wait(timeout=10))
            second.start()
        finally:
            release.set()
            first.join(timeout=10)
            if second.ident is not None:
                second.join(timeout=10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["available"] for result in results), results)
        self.builder.assert_called_once()
        self.assertTrue(all(self.predictor.lock_checks))
        self.assert_clean()

    def test_synthetic_local_video_with_real_opencv_capture(self):
        path = self.root / "synthetic.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer unavailable")
        try:
            for i in range(24):
                writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
        finally:
            writer.release()
        # Restore the native class saved before setUp patched VideoCapture.
        with patch.object(cv2, "VideoCapture", REAL_VIDEO_CAPTURE):
            result = service.segment_video(str(path), 0.0, 1.0, [box_prompt()])
        self.assertTrue(result["available"], result)
        np.testing.assert_allclose(result["sampled_timestamps"], [0, 1 / 3, 2 / 3])
        self.assertEqual((result["original_width"], result["original_height"]), (64, 48))
        self.assert_clean()


REAL_VIDEO_CAPTURE = cv2.VideoCapture

if __name__ == "__main__":
    unittest.main()
