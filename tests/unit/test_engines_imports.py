"""Verify domain engine packages import cleanly and expose core contracts."""
import sys
import unittest
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestEngineModuleBoundaries(unittest.TestCase):
    def test_engine_packages_expose_public_apis(self):
        import engines.video as video
        import engines.vision as vision
        import engines.audio as audio
        import engines.tracking as tracking
        import engines.rendering as rendering

        self.assertTrue(hasattr(video, "layout_decision"))
        self.assertTrue(hasattr(video, "tuning"))
        self.assertTrue(hasattr(vision, "detection_service"))
        self.assertTrue(hasattr(vision, "segmentation_service"))
        self.assertTrue(hasattr(audio, "whisper_engine"))
        self.assertTrue(hasattr(audio, "speaker_priority"))
        self.assertTrue(callable(tracking.analyze_video_with_openshorts))
        self.assertTrue(callable(rendering.renderer.run_render))
        self.assertTrue(callable(rendering.encoders.detect_encoder))


if __name__ == "__main__":
    unittest.main()
