"""The Effects switch must strip grade, vignette, seam blur, mono pulse and cut dips from the export."""

from __future__ import annotations

import unittest

from local_worker.rendering.decision import decision_from_payload
from local_worker.rendering.filters import GRADE_FILTER, _grade_chain


def decision(effects: bool):
    return decision_from_payload({
        "source_path": "source.mp4",
        "source_width": 1920,
        "source_height": 1080,
        "start": 0.0,
        "end": 4.0,
        "layout": "fill",
        "track": [
            {"t": 0.0, "x": 0.5, "y": 0.4},
            {"t": 2.0, "x": 0.5, "y": 0.4, "cut": True},
        ],
        "words": [{"w": "incredible!", "start": 1.0, "end": 1.5}],
        "style": {},
        "visualEffects": effects,
    })


class EffectsToggleTest(unittest.TestCase):
    def test_defaults_on(self):
        payload = {"source_path": "s.mp4", "source_width": 2, "source_height": 2, "start": 0, "end": 1}
        self.assertTrue(decision_from_payload(payload).visual_effects)

    def test_on_keeps_full_chain(self):
        graph = ";".join(_grade_chain("comp", 1, None, decision=decision(True)))
        self.assertIn(GRADE_FILTER, graph)
        self.assertIn("[1:v]overlay", graph)
        self.assertIn("gblur", graph)
        self.assertIn("hue=s=0", graph)

    def test_off_strips_every_effect(self):
        chains = _grade_chain("comp", 1, (800, 100), decision=decision(False))
        graph = ";".join(chains)
        self.assertEqual(len(chains), 1)
        for token in (GRADE_FILTER, "overlay", "gblur", "hue=s=0", "eq=brightness"):
            self.assertNotIn(token, graph)
        self.assertIn("drawbox", graph)
        self.assertTrue(graph.startswith("[comp]") and graph.endswith("format=yuv420p[out]"))


if __name__ == "__main__":
    unittest.main()
