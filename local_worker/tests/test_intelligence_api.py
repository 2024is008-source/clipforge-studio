import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from local_worker import main


class Jobs:
    def __init__(self, root):
        self.root = root
        self.jobs = {}

    def create_job(self, kind, params):
        jid = f"job{len(self.jobs)}"
        job = dict(id=jid, type=kind, params=params, status="queued", progress=0,
                   cancelled=False, result=None, error=None, job_dir=self.root)
        self.jobs[jid] = job
        return job

    def update_job(self, jid, **values):
        self.jobs[jid].update(values)
        if values.get("status") == "completed" and self.jobs[jid]["result"] is None:
            raise AssertionError("Completed jobs must publish their result atomically")

    def get_job(self, jid):
        return self.jobs.get(jid)

    def is_cancelled(self, jid):
        return self.jobs[jid]["cancelled"]

    def cancel_job(self, jid):
        self.jobs[jid].update(cancelled=True, status="cancelled")
        return True


class InlineThread:
    def __init__(self, target, args=(), **kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class IntelligenceApiTests(unittest.TestCase):
    def setUp(self):
        main.ANALYSIS_CACHE.clear()
        self.addCleanup(main.ANALYSIS_CACHE.clear)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name, "source.mp4").write_bytes(b"synthetic source")
        self.jobs = Jobs(self.temp.name)
        for patcher in (
            patch.object(main, "SOURCES_DIR", self.temp.name),
            patch.object(main, "JOB_MANAGER", self.jobs),
            patch.object(main, "is_readable_media_file", return_value=True),
            patch.object(main, "threading", SimpleNamespace(Thread=InlineThread)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def test_source_ids_reject_traversal_and_aliases(self):
        for value in ("../source", "source/", "source.mp4", "a" * 65, {}, 1, None):
            self.assertIsNone(main._safe_source_id(value))
        self.assertEqual(main._safe_source_id("source-123_A"), "source-123_A")

    def test_analysis_queues_and_reports_own_result(self):
        output = {"ok": True, "track": [{"t": 1, "x": .5, "y": .3, "mode": "fit"}]}
        with patch.object(main, "analyze_video_with_openshorts", return_value=output) as analyze:
            response = self.client.post("/v1/openshorts/analyze", data={
                "source_id": "source", "start": 1, "end": 2, "queue": "true", "framing_mode": "fill"})
        self.assertEqual(response.status_code, 200)
        jid = response.json()["job_id"]
        state = self.client.get(f"/v1/openshorts/analyze/jobs/{jid}").json()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["result"]["track"], output["track"])
        self.assertEqual(analyze.call_args.kwargs["framing_mode"], "fill")
        self.assertTrue(callable(analyze.call_args.kwargs["is_cancelled"]))

    def test_repeated_analysis_reuses_final_result_with_new_job_identity(self):
        output = {"ok": True, "detectors": {"face": "yunet", "person": "ready"},
                  "track": [{"t": 1, "x": .5, "y": .4, "mode": "fit",
                             "layoutDecision": {"scene": 0, "reason": "uncertain"}}],
                  "layout_decisions": [{"start": 1, "end": 2, "layout": "fit"}]}
        with patch.object(main, "analyze_video_with_openshorts", return_value=output) as analyze:
            results = [self.client.post("/v1/openshorts/analyze", data={
                "source_id": "source", "start": 1, "end": 2}).json() for _ in range(2)]
            self.assertEqual(analyze.call_count, 1)
            self.client.post("/v1/openshorts/analyze", data={
                "source_id": "source", "start": 1, "end": 2, "framing_mode": "fit"})
            self.assertEqual(analyze.call_count, 2)
        self.assertFalse(results[0]["cache_hit"])
        self.assertTrue(results[1]["cache_hit"])
        self.assertNotEqual(results[0]["job_id"], results[1]["job_id"])
        self.assertEqual(results[0]["track"], results[1]["track"])

    def test_analysis_bad_interval_and_mode_rejected_before_job(self):
        for data in ({"start": "nan", "end": 2}, {"start": 3, "end": 2},
                     {"start": 0, "end": 1, "framing_mode": "unknown"}):
            response = self.client.post("/v1/openshorts/analyze", data={"source_id": "source", **data})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(self.jobs.jobs)

    def test_analysis_failure_is_not_success(self):
        with patch.object(main, "analyze_video_with_openshorts", side_effect=ValueError("No frames")):
            response = self.client.post("/v1/openshorts/analyze", data={
                "source_id": "source", "end": 2, "queue": "true"})
        state = self.client.get(f'/v1/openshorts/analyze/jobs/{response.json()["job_id"]}').json()
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["error"], "No frames")

    def test_analysis_cancellation_stays_cancelled(self):
        with patch.object(main, "analyze_video_with_openshorts", side_effect=InterruptedError()):
            response = self.client.post("/v1/openshorts/analyze", data={
                "source_id": "source", "end": 2, "queue": "true"})
        state = self.client.get(f'/v1/openshorts/analyze/jobs/{response.json()["job_id"]}').json()
        self.assertEqual(state["status"], "cancelled")

    def test_segmentation_unavailable_is_explicit(self):
        output = {"available": False, "reason": "No local checkpoint", "frames": []}
        with patch("local_worker.segmentation_service.segment_video", return_value=output) as segment:
            response = self.client.post("/v1/segmentation/jobs", json={
                "sourceId": "source", "start": 0, "end": 2,
                "prompts": [{"object_id": 1, "t": 0, "box": [.1, .1, .3, .3]}]})
        self.assertEqual(response.status_code, 200)
        state = self.client.get(f'/v1/segmentation/jobs/{response.json()["job_id"]}').json()
        self.assertEqual(state["result"], output)
        self.assertEqual(segment.call_args.args[0], str(Path(self.temp.name, "source.mp4")))

    def test_segmentation_invalid_prompt_and_missing_source_rejected(self):
        body = {"sourceId": "source", "start": 0, "end": 11, "prompts": []}
        self.assertEqual(self.client.post("/v1/segmentation/jobs", json=body).status_code, 400)
        body["sourceId"] = "missing"
        self.assertEqual(self.client.post("/v1/segmentation/jobs", json=body).status_code, 404)
        body["sourceId"] = "../source"
        self.assertEqual(self.client.post("/v1/segmentation/jobs", json=body).status_code, 400)

    def test_cancel_cannot_target_other_job_type(self):
        job = self.jobs.create_job("render", {})
        for endpoint in ("segmentation", "openshorts/analyze"):
            response = self.client.post(f'/v1/{endpoint}/jobs/{job["id"]}/cancel')
            self.assertEqual(response.status_code, 404)
        self.assertFalse(job["cancelled"])

    def test_failed_uploads_leave_cached_source_and_no_partial_files(self):
        source = Path(self.temp.name, "source.mp4")
        before = source.read_bytes()
        for endpoint, data in (("/v1/sources/source", {}),
                               ("/v1/openshorts/analyze", {"source_id": "source", "end": 1})):
            for content, expected in ((b"", 400), (b"oversize", 413)):
                with self.subTest(endpoint=endpoint, content=content), patch.object(main, "MAX_UPLOAD_BYTES", 3):
                    response = self.client.post(endpoint, data=data,
                                                files={"file": ("test.mp4", content, "video/mp4")})
                self.assertEqual(response.status_code, expected)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual([p.name for p in Path(self.temp.name).iterdir()], ["source.mp4"])

    def test_render_info_preserves_source_cadence(self):
        for fps, expected in ((24, 24), (24000 / 1001, 24000 / 1001), (60, 30)):
            with self.subTest(fps=fps), patch.object(main, "probe_source", return_value=(1920, 1080, fps, fps)):
                response = self.client.get("/v1/sources/source/render-info?fps=30")
            self.assertEqual(response.json()["fps"], expected)

    def test_render_info_rejects_missing_and_invalid_sources(self):
        self.assertEqual(self.client.get("/v1/sources/missing/render-info").status_code, 409)
        self.assertEqual(self.client.get("/v1/sources/source.mp4/render-info").status_code, 400)
        with patch.object(main, "probe_source", side_effect=ValueError("No video")):
            self.assertEqual(self.client.get("/v1/sources/source/render-info").status_code, 422)

    def test_render_overlay_is_pinned_at_resolved_cadence_and_released_on_failure(self):
        lifecycle = []

        @contextmanager
        def pin(path, fps, duration):
            lifecycle.append(("pin", path, fps, duration))
            try:
                yield path
            finally:
                lifecycle.append("release")

        def render(*args):
            self.assertEqual(args[4], "trusted-overlay")
            self.assertEqual(len(lifecycle), 1)
            raise RuntimeError("Synthetic encoder failure")

        fps = 24000 / 1001
        with patch.object(main, "probe_source", return_value=(1920, 1080, fps, fps)), \
             patch.object(main, "detect_encoder", return_value=SimpleNamespace(name="cpu")), \
             patch.object(main, "resolve_overlay", return_value="trusted-overlay") as resolve, \
             patch.object(main, "reference_overlay", side_effect=pin), \
             patch.object(main, "_execute_decision_render", side_effect=render):
            response = self.client.post("/v1/render/decisions", json={
                "sourceId": "source", "overlayId": "overlay-id",
                "decision": {"start": 1, "end": 3, "layout": "auto", "fps": "30"}})
        self.assertEqual(response.status_code, 200)
        resolve.assert_called_once_with("overlay-id", fps, 2)
        self.assertEqual(lifecycle, [("pin", "trusted-overlay", fps, 2), "release"])
        self.assertEqual(self.jobs.get_job(response.json()["job_id"])["status"], "error")

    def test_render_rejects_invalid_overlay_before_creating_job(self):
        with patch.object(main, "probe_source", return_value=(1920, 1080, 24, 24)), \
             patch.object(main, "detect_encoder", return_value=SimpleNamespace(name="cpu")), \
             patch.object(main, "resolve_overlay", side_effect=ValueError("fps mismatch")):
            response = self.client.post("/v1/render/decisions", json={
                "sourceId": "source", "overlayId": "overlay-id",
                "decision": {"start": 1, "end": 3, "layout": "auto"}})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.jobs.jobs)

    def test_overlay_router_and_delete_preflight_are_registered(self):
        response = self.client.post("/v1/render/overlays", json={
            "fps": 30, "frameCount": 0, "width": 1080, "height": 1920})
        self.assertEqual(response.status_code, 400)
        response = self.client.options("/v1/render/overlays/example", headers={
            "Origin": "http://127.0.0.1:8080", "Access-Control-Request-Method": "DELETE"})
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
