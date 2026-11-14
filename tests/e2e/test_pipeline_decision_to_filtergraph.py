"""End-to-end pipeline verification from transcript + tracking points to multi-pass FFmpeg filtergraphs."""
import os
import tempfile
import unittest
from local_worker.rendering.decision import decision_from_payload
from local_worker.rendering.encoders import cpu_encoder
from local_worker.rendering.renderer import RenderRequest, Segment, plan_passes


class TestEndToEndDecisionToFiltergraph(unittest.TestCase):
    def test_multi_layout_decision_plans_valid_9x16_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "input.mp4")
            open(source, "wb").close()
            decision = decision_from_payload({
                "source_path": source,
                "source_width": 1920,
                "source_height": 1080,
                "source_fps": 30.0,
                "start": 0.0,
                "end": 6.0,
                "layout": "auto",
                "hook": "Watch this moment",
                "track": [
                    {"t": 0.0, "x": 0.45, "y": 0.34, "mode": "fill", "cut": True},
                    {"t": 3.0, "x": 0.30, "y": 0.35, "x2": 0.72, "y2": 0.36, "mode": "split", "cut": True},
                ],
                "words": [
                    {"w": "Watch", "start": 0.2, "end": 0.6},
                    {"w": "this", "start": 0.6, "end": 1.0},
                    {"w": "moment!", "start": 1.0, "end": 1.6},
                ],
                "style": {"preset": "opus", "showHook": True, "wordsPerPage": 3},
                "zoomPunch": True,
            })
            req = RenderRequest(decision=decision, job_dir=tmp, fps=30.0, encoder=cpu_encoder())
            fill_passes, fill_out, fill_events = plan_passes(
                req, Segment("fill", 0.0, 3.0), 0, False, None, "vignette.png"
            )
            split_passes, split_out, _ = plan_passes(
                req, Segment("split", 3.0, 6.0), 1, False, None, "vignette.png"
            )
            self.assertEqual(len(fill_passes), 1)
            self.assertGreaterEqual(fill_events, 2)
            self.assertTrue(fill_out.endswith("segment_000.mp4"))
            # 2-Up Split renders 2 panel crops + 1 composite pass = 3 passes
            self.assertEqual(len(split_passes), 3)
            self.assertTrue(split_out.endswith("segment_001.mp4"))


if __name__ == "__main__":
    unittest.main()
