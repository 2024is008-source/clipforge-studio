from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from local_worker.rendering.decision import OUTPUT_HEIGHT, OUTPUT_WIDTH, decision_from_payload
from local_worker.rendering.encoders import cpu_encoder
from local_worker.rendering.filters import (
    GRADE_FILTER,
    CropFrame,
    build_composite_graph,
    caption_anchor_y,
    crop_keyframes,
    decoded_video_input_count,
    panel_crop_chain,
    panels_for,
    subject_panels,
)
from local_worker.rendering.renderer import RenderRequest, Segment, plan_passes, segment_decision

SOURCES = [
    ("landscape", 1920, 1080, 1080, 608),
    ("portrait", 1080, 1920, 1080, 1920),
    ("tall portrait", 1080, 2400, 864, 1920),
    ("ultrawide", 3440, 1440, 1080, 452),
    ("square", 1080, 1080, 1080, 1080),
    ("odd dimensions", 1919, 1079, 1080, 608),
]

CAPTION_SOURCES = [
    ("landscape", 1920, 1080, 1080, 608),
    ("portrait", 1080, 1920, 756, 1344),
    ("tall portrait", 1080, 2400, 604, 1344),
    ("ultrawide", 3440, 1440, 1080, 452),
    ("square", 1080, 1080, 1080, 1080),
    ("odd dimensions", 1919, 1079, 1080, 608),
]


def fit_decision(width=1920, height=1080, *, preset="opus", with_words=True, position_y=0.78):
    return decision_from_payload({
        "source_path": os.path.abspath("source.mp4"),
        "source_width": width,
        "source_height": height,
        "source_fps": 30.0,
        "start": 12.5,
        "end": 14.5,
        "layout": "fit",
        "track": [
            {"t": 12.5, "x": 0.05, "y": 0.1, "zoom": 1.8, "mode": "fit"},
            {"t": 14.5, "x": 0.8, "y": 0.9, "zoom": 1.4, "mode": "fit"},
        ],
        "words": [{"w": "Everything!", "start": 12.5, "end": 13.5}] if with_words else [],
        "style": {"preset": preset, "showHook": False, "showEmojis": False, "positionY": position_y},
        "zoomPunch": True,
        "manualX": 0.95,
        "manualY": 0.9,
    })


class FitGeometryTest(unittest.TestCase):
    def test_contains_full_source_with_even_bounded_dimensions(self):
        for name, width, height, expected_w, expected_h in SOURCES:
            with self.subTest(source=name):
                decision = fit_decision(width, height, with_words=False)
                panel, = panels_for(decision, "fit")
                self.assertEqual((panel.dest_w, panel.dest_h), (expected_w, expected_h))
                self.assertEqual(panel.crop_mode, "full")
                self.assertEqual((panel.zoom_mul, panel.zoom_min, panel.exclude_bottom), (1, 1, False))
                self.assertEqual(panel.dest_w % 2, 0)
                self.assertEqual(panel.dest_h % 2, 0)
                self.assertTrue(0 < panel.dest_w <= OUTPUT_WIDTH)
                self.assertTrue(0 < panel.dest_h <= OUTPUT_HEIGHT)
                scale = min(OUTPUT_WIDTH / width, OUTPUT_HEIGHT / height)
                self.assertLessEqual(abs(panel.dest_w - width * scale), 1)
                self.assertLessEqual(abs(panel.dest_h - height * scale), 1)
                self.assertAlmostEqual(
                    panel.dest_w / panel.dest_h, width / height,
                    delta=(1 + width / height) / panel.dest_h,
                )
                self.assertEqual(subject_panels(decision, "fit"), [])
                self.assertEqual(decoded_video_input_count(decision, "fit"), 1)

    def test_contains_full_source_with_caption_safe_bounds(self):
        for name, width, height, expected_w, expected_h in CAPTION_SOURCES:
            with self.subTest(source=name):
                decision = fit_decision(width, height, with_words=True)
                panel, = panels_for(decision, "fit")
                self.assertEqual((panel.dest_w, panel.dest_h), (expected_w, expected_h))
                self.assertEqual(panel.crop_mode, "full")
                self.assertEqual((panel.zoom_mul, panel.zoom_min, panel.exclude_bottom), (1, 1, False))
                self.assertEqual(panel.dest_w % 2, 0)
                self.assertEqual(panel.dest_h % 2, 0)
                self.assertTrue(0 < panel.dest_w <= OUTPUT_WIDTH)
                self.assertTrue(0 < panel.dest_h <= OUTPUT_HEIGHT)
                self.assertAlmostEqual(
                    panel.dest_w / panel.dest_h, width / height,
                    delta=(1 + width / height) / panel.dest_h,
                )

    def test_keyframes_ignore_tracking_manual_offsets_and_punch(self):
        for name, width, height, _, _ in SOURCES:
            with self.subTest(source=name):
                decision = fit_decision(width, height)
                panel, = panels_for(decision, "fit")
                frames = crop_keyframes(decision, panel, 30.0, 2.0)
                self.assertEqual(frames, [CropFrame(0, width, height, 0, 0)])
                neutral = replace(decision, track=[], zoom_punch=False, manual_x=None, manual_y=None)
                self.assertEqual(frames, crop_keyframes(neutral, panel, 30.0, 1.0))
                graph = panel_crop_chain(panel, "unused.cmd", frames, decision, 30.0)
                self.assertNotIn("crop=", graph)
                self.assertNotIn("sendcmd", graph)
                self.assertIn(f"scale={panel.dest_w}:{panel.dest_h}:flags=lanczos", graph)

    def test_composite_splits_the_source_and_centers_the_uncropped_foreground(self):
        for name, width, height, expected_w, expected_h in SOURCES:
            with self.subTest(source=name):
                graph, label = build_composite_graph(
                    fit_decision(width, height, with_words=False), "fit", 0, 30000 / 1001, 0,
                    vignette_index=1, band=(800, 100),
                )
                self.assertEqual(label, "[out]")
                self.assertIn("[0:v]fps=30000/1001,split=2[fit_bg][fit_fg]", graph)
                foreground = next(chain for chain in graph.split(";") if chain.startswith("[fit_fg]"))
                self.assertEqual(
                    foreground,
                    f"[fit_fg]scale={expected_w}:{expected_h}:flags=lanczos,setsar=1:1,format=yuv420p[fg]",
                )
                self.assertEqual(graph.count("crop="), 1)
                self.assertIn("gblur=sigma=28", graph)
                self.assertIn("[bed][fg]overlay=(W-w)/2:(H-h)/2[comp]", graph)
                self.assertIn(f"[comp]{GRADE_FILTER}[vg]", graph)
                self.assertIn("[vg][1:v]overlay=format=auto[vv]", graph)
                self.assertIn("drawbox=x=0:y=800:w=1080:h=100:", graph)
                # Seam blur dips before the caption band so text stays sharp through the transition.
                self.assertLess(graph.index("gblur=sigma=26"), graph.index("drawbox=x=0:y=800"))
                self.assertNotIn("sendcmd", graph)
                self.assertNotIn("zoompan", graph)

    def test_composite_with_captions_positions_above_caption_band(self):
        for name, width, height, expected_w, expected_h in CAPTION_SOURCES:
            with self.subTest(source=name):
                graph, label = build_composite_graph(
                    fit_decision(width, height, with_words=True), "fit", 0, 30000 / 1001, 0,
                    vignette_index=1, band=(800, 100),
                )
                self.assertEqual(label, "[out]")
                self.assertIn("[0:v]fps=30000/1001,split=2[fit_bg][fit_fg]", graph)
                foreground = next(chain for chain in graph.split(";") if chain.startswith("[fit_fg]"))
                self.assertEqual(
                    foreground,
                    f"[fit_fg]scale={expected_w}:{expected_h}:flags=lanczos,setsar=1:1,format=yuv420p[fg]",
                )
                self.assertIn("[bed][fg]overlay=(W-w)/2:'min((H-h)/2,1344-h)'[comp]", graph)

    def test_aspect_ratios_and_odd_dimensions_extended(self):
        """Extended aspect ratio test: 16:9, 4:3, 9:16, 1:1, 32:9 ultrawide, tall portrait, odd dims."""
        cases = [
            ("16:9 HD", 1280, 720, 1080, 608),
            ("16:9 4K", 3840, 2160, 1080, 608),
            ("4:3 classic", 1440, 1080, 1080, 810),
            ("4:3 SD", 640, 480, 1080, 810),
            ("9:16 mobile", 720, 1280, 1080, 1920),
            ("1:1 square small", 600, 600, 1080, 1080),
            ("32:9 extreme superultrawide", 5120, 1440, 1080, 304),
            ("9:21 ultra tall portrait", 1080, 2520, 822, 1920),
            ("odd 853x481", 853, 481, 1080, 610),
            ("odd 1001x1001", 1001, 1001, 1080, 1080),
        ]
        for label, w, h, exp_w, exp_h in cases:
            with self.subTest(ratio=label):
                dec = fit_decision(w, h, with_words=False)
                panel, = panels_for(dec, "fit")
                self.assertEqual((panel.dest_w, panel.dest_h), (exp_w, exp_h))
                self.assertEqual(panel.dest_w % 2, 0)
                self.assertEqual(panel.dest_h % 2, 0)
                self.assertTrue(0 < panel.dest_w <= OUTPUT_WIDTH)
                self.assertTrue(0 < panel.dest_h <= OUTPUT_HEIGHT)
                # Verify aspect ratio preservation
                self.assertAlmostEqual(panel.dest_w / panel.dest_h, w / h, delta=0.02)

    def test_different_caption_positions_and_anchors(self):
        """Verify caption anchor Y and band calculation across positions."""
        for pos_y in [0.2, 0.5, 0.78, 0.83, 0.9]:
            dec = fit_decision(1920, 1080, with_words=True, position_y=pos_y)
            anchor = caption_anchor_y(dec, "fit")
            self.assertEqual(anchor, 0.83)  # fit layout default anchor
            split_anchor = caption_anchor_y(dec, "split")
            self.assertEqual(split_anchor, 0.5)
            editorial_anchor = caption_anchor_y(dec, "editorial")
            self.assertEqual(editorial_anchor, 0.58)

    def test_public_composite_helper_uses_the_requested_full_source_input(self):
        graph, _ = build_composite_graph(fit_decision(), "fit", 1, 30, 3)
        self.assertIn("[3:v]fps=30/1,split=2", graph)
        self.assertNotIn("[0:v]", graph)
        self.assertIn("[comp]" + GRADE_FILTER, graph)


class FitPlanTest(unittest.TestCase):
    def test_hook_uses_clip_clock_across_layout_segments(self):
        decision = fit_decision()
        decision = replace(decision, end=decision.start + 8, words=[], hook="KISS THE RING",
                           style=replace(decision.style, show_hook=True))
        with tempfile.TemporaryDirectory() as job_dir:
            request = RenderRequest(decision, job_dir, fps=30, encoder=cpu_encoder())
            vignette = os.path.join(job_dir, "vignette.png")
            for index, segment in enumerate((Segment("fit", 0, 3), Segment("fit", 3, 5),
                                             Segment("fit", 5, 8))):
                _, _, events = plan_passes(request, segment, index, False, None, vignette)
                path = os.path.join(job_dir, f"captions_s{index:03d}.ass")
                if index == 2:
                    self.assertEqual(events, 0)
                    self.assertFalse(os.path.exists(path))
                else:
                    with open(path, encoding="utf-8-sig") as handle:
                        text = handle.read()
                    end = "0:00:03.00" if index == 0 else "0:00:00.60"
                    self.assertIn(f"Dialogue: 1,0:00:00.00,{end},Hook", text)
                    self.assertIn(r"\fad(160,220)", text)
                    self.assertNotIn("\f", text)

    def test_fit_is_one_final_pass_with_audio_and_captions_but_no_panel_encode(self):
        for layout in ("fit", "auto"):
            for has_audio in (False, True):
                with self.subTest(layout=layout, has_audio=has_audio), tempfile.TemporaryDirectory() as job_dir:
                    decision = replace(fit_decision(), layout=layout)
                    request = RenderRequest(decision, job_dir, fps=30, encoder=cpu_encoder())
                    vignette = os.path.join(job_dir, "vignette.png")
                    with patch("local_worker.rendering.renderer.crop_keyframes") as crops, \
                         patch("local_worker.rendering.renderer._write_keyframes") as commands, \
                         patch("local_worker.rendering.renderer.panel_crop_chain") as panels:
                        passes, output, events = plan_passes(
                            request, Segment("fit", 0, 2), 0, has_audio, None, vignette,
                        )
                    crops.assert_not_called()
                    commands.assert_not_called()
                    panels.assert_not_called()
                    self.assertEqual(len(passes), 1)
                    planned, = passes
                    self.assertEqual(planned.label, "render fit window 1")
                    self.assertEqual(planned.output, output)
                    self.assertEqual(output, os.path.join(job_dir, "segment_000.mp4"))
                    self.assertFalse(planned.verify_frames)
                    self.assertEqual((planned.duration, planned.weight), (2, 2))
                    self.assertEqual(events, 1)
                    self.assertEqual(os.listdir(job_dir), ["captions_s000.ass"])
                    inputs = [planned.args[i + 1] for i, arg in enumerate(planned.args) if arg == "-i"]
                    self.assertEqual(inputs, [decision.source_path, vignette])
                    self.assertEqual(planned.args[planned.args.index("-ss") + 1], "12.5000")
                    self.assertEqual(planned.args[planned.args.index("-t") + 1], "2.0000")
                    self.assertNotIn("-qp", planned.args)
                    graph = planned.args[planned.args.index("-filter_complex") + 1]
                    self.assertIn("[0:v]fps=30/1,split=2", graph)
                    self.assertIn("[vg][1:v]overlay=format=auto[vv]", graph)
                    self.assertIn("[out]ass=filename=captions_s000.ass[vout]", graph)
                    if has_audio:
                        self.assertIn("[0:a]aresample=48000:async=1", graph)
                        self.assertIn("loudnorm=I=-16", graph)
                        self.assertIn("[outa]", planned.args)
                    else:
                        self.assertIn("-an", planned.args)
                        self.assertNotIn("[0:a]", graph)

    def test_fit_without_captions_writes_no_intermediates_and_preserves_extra_audio_indices(self):
        from local_worker.rendering.decision import AudioTrack

        with tempfile.TemporaryDirectory() as job_dir:
            decision = replace(fit_decision(), words=[], audio=[
                AudioTrack("original", ""),
                AudioTrack("music", os.path.join(job_dir, "music.wav")),
            ])
            request = RenderRequest(decision, job_dir, fps=30, encoder=cpu_encoder())
            vignette = os.path.join(job_dir, "vignette.png")
            passes, _, events = plan_passes(request, Segment("fit", 0, 2), 0, True, None, vignette)
            self.assertEqual(len(passes), 1)
            self.assertEqual(events, 0)
            self.assertEqual(os.listdir(job_dir), [])
            args = passes[0].args
            inputs = [args[i + 1] for i, arg in enumerate(args) if arg == "-i"]
            self.assertEqual(inputs, [decision.source_path, decision.audio[1].path, vignette])
            graph = args[args.index("-filter_complex") + 1]
            self.assertIn("[1:a]aresample=48000:async=1", graph)
            self.assertIn("[vg][2:v]overlay=format=auto[vv]", graph)
            self.assertIn("format=yuv420p[vout]", graph)
            self.assertNotIn("ass=filename", graph)


if __name__ == "__main__":
    unittest.main()
