"""Unit tests for OpenShorts architecture upgrades in ClipForge Studio."""

from __future__ import annotations

import copy
import unittest

from local_worker.clip_selection import (
    build_transcript_windows,
    dedupe_overlapping,
    score_batches,
    snap_clip_to_words,
    trim_to_best,
)
from local_worker.clipforge_engine import (
    apply_scene_split_layout,
    delivery_size,
    select_expr,
    source_already_fits,
)
from local_worker.hook_placement import (
    BOTTOM_CAPTIONS,
    SEAM_CAPTIONS,
    caption_bands,
    face_cost,
    head_box,
    place_hook_banner,
)
from local_worker.layout_decision import LayoutPolicy, anchor_solo_subjects, finalize_layouts
from local_worker.rendering.subtitles import clean_hook_title
from local_worker.speaker_priority import (
    TalkWindow,
    is_conversation,
    normalise_activity,
    speaker_cut_trajectory,
)


class TestOpenShortsUpgrades(unittest.TestCase):
    def test_clean_hook_title_fixes_both_uploaded_video_hooks(self):
        self.assertEqual(
            clean_hook_title("are many different ways of communication It could"),
            "Many different ways of communication",
        )
        self.assertEqual(
            clean_hook_title("To my left are 5 people All 5"),
            "To my left are 5 people",
        )
        self.assertEqual(
            clean_hook_title("is a school backpack Yeah hers is a"),
            "Hers is a school backpack",
        )

    def test_hook_placement_avoids_faces_and_split_seam(self):
        # Default placement when no faces overlap
        res_default = place_hook_banner(faces=[], layout_modes="fill")
        self.assertEqual(res_default.top_px, 145)
        self.assertEqual(res_default.scale, 1.0)

        # Face directly in top banner region (y=0.08..0.22) pushes hook downward away from head_box
        high_face = [(0.38, 0.10, 0.62, 0.24)]
        res_avoid = place_hook_banner(faces=high_face, layout_modes="fill")
        self.assertGreater(res_avoid.top_px, 260)
        self.assertLessEqual(res_avoid.face_overlap, 0.08)

        # Split layout forbids the 0.42..0.58 seam caption band
        bands = caption_bands("split")
        self.assertIn(SEAM_CAPTIONS, bands)
        self.assertIn(BOTTOM_CAPTIONS, caption_bands("fill"))
        hb = head_box((0.4, 0.2, 0.6, 0.4))
        self.assertLess(hb[1], 0.2)
        self.assertGreater(face_cost((0.3, 0.1, 0.7, 0.35), [(0.4, 0.2, 0.6, 0.4)]), 0.1)

    def test_clip_selection_windowing_dedup_and_word_snapping(self):
        segs = [{"start": i * 20.0, "end": (i + 1) * 20.0, "text": f"seg {i}"} for i in range(8)]
        windows = build_transcript_windows(segs, window_seconds=90.0, overlap_seconds=30.0)
        self.assertGreaterEqual(len(windows), 2)

        batches = score_batches(7, max_batch_size=3)
        self.assertEqual(sum(b - a for a, b in batches), 7)

        shorts = [
            {"start": 10.0, "end": 40.0, "predicted_score": 82},
            {"start": 15.0, "end": 45.0, "predicted_score": 95},  # overlaps >= 25% with first, higher score wins
            {"start": 60.0, "end": 90.0, "predicted_score": 88},
        ]
        deduped = dedupe_overlapping(shorts, ratio=0.25)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["predicted_score"], 95)
        best = trim_to_best(shorts, max_clips=1)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["predicted_score"], 95)

        words = [
            {"w": "previously.", "s": 9.2, "e": 9.8},
            {"w": "Here", "s": 10.3, "e": 10.6},
            {"w": "is", "s": 10.6, "e": 10.8},
            {"w": "the", "s": 10.8, "e": 11.0},
            {"w": "story.", "s": 25.0, "e": 25.6},
            {"w": "Next", "s": 26.2, "e": 26.5},
        ]
        snapped_s, snapped_e = snap_clip_to_words(10.0, 25.8, words, min_duration=10.0, max_duration=60.0)
        self.assertGreaterEqual(snapped_s, 9.8)
        self.assertLessEqual(snapped_s, 10.3)
        self.assertGreaterEqual(snapped_e, 25.6)
        self.assertLessEqual(snapped_e, 26.2)

    def test_active_speaker_conversation_gate_cancels_false_split_on_silent_listener(self):
        # 10 samples (2.0s = 5 windows): Speaker 0 talks in 4 windows (80%), Listener 1 has 1 window blip (10% of samples)
        talks = {
            0: TalkWindow(frozenset({0}), False, True),
            1: TalkWindow(frozenset({0}), False, True),
            2: TalkWindow(frozenset({0}), False, True),
            3: TalkWindow(frozenset({0}), False, True),
            4: TalkWindow(frozenset({1}), False, True),
        }
        self.assertFalse(is_conversation(talks, 0, 5, min_share=0.25))
        self.assertTrue(
            is_conversation(
                {0: TalkWindow(frozenset({0}), False, True), 1: TalkWindow(frozenset({1}), False, True)},
                0,
                2,
                min_share=0.20,
            )
        )

        # In layout_decision: Speaker 1 is on the right (x=1450) talking solo while Listener 0 stands silent on the left (x=300)
        talks_solo_right = {i: TalkWindow(frozenset({1}), False, True) for i in range(5)}
        points = [{"t": i * 0.2, "x": 0.2, "y": 0.32, "x2": 0.8, "y2": 0.32, "mode": "split", "cut": i == 0} for i in range(10)]
        people = [
            {"key": 0, "kind": "face", "box": [300, 250, 140, 180], "confidence": 0.95},
            {"key": 1, "kind": "face", "box": [1450, 250, 140, 180], "confidence": 0.95},
        ]
        samples = [{"t": p["t"], "people": copy.deepcopy(people)} for p in points]
        policy = LayoutPolicy()
        anchor_solo_subjects(points, samples, talks_solo_right, 1, 1920, 1080, policy, {})
        report = finalize_layouts(points, samples, 1920, 1080, talks_solo_right, 1, "auto", policy)
        self.assertEqual(report[0]["layout"], "fill")
        self.assertAlmostEqual(points[0]["x"], (1450 + 70) / 1920, places=2)

    def test_reframe_v2_helpers_and_speaker_cut_trajectory(self):
        self.assertTrue(source_already_fits(1080, 1920))
        self.assertFalse(source_already_fits(1920, 1080))
        self.assertEqual(delivery_size(608, 1080), (1080, 1920))
        self.assertEqual(delivery_size(1200, 2134), (1200, 2134))
        expr = select_expr(0, 100, stride=4, cut_frames=[25])
        self.assertIn("eq(n,25)", expr)

        norm = normalise_activity({0: {0: 0.01, 1: 0.01, 2: 0.20, 3: 0.25, 4: 0.01}})
        self.assertEqual(norm[0][0], 0.0)
        self.assertGreater(norm[0][3], 0.9)

        samples = [
            {"t": i * 0.4, "people": [
                {"key": 0, "kind": "face", "box": [300, 250, 140, 180]},
                {"key": 1, "kind": "face", "box": [1450, 250, 140, 180]},
            ]}
            for i in range(6)
        ]
        talks = {
            0: TalkWindow(frozenset({0}), False, True),
            1: TalkWindow(frozenset({0}), False, True),
            2: TalkWindow(frozenset({1}), False, True),  # 1 window only -> held on 0
            3: TalkWindow(frozenset({1}), False, True),
            4: TalkWindow(frozenset({1}), False, True),  # 3rd window -> switches to 1
            5: TalkWindow(frozenset({1}), False, True),
        }
        traj = speaker_cut_trajectory(samples, talks, 0.0, 1920, 1080, min_hold_windows=3)
        self.assertLess(traj[0][0], 0.3)
        self.assertLess(traj[2][0], 0.3)
        self.assertGreater(traj[4][0], 0.7)



    def test_youtube_url_normalization_handles_youtu_be_si_links_and_shorts(self):
        from local_worker.youtube_downloader import normalize_youtube_url

        ok, canonical, vid = normalize_youtube_url("https://youtu.be/VuoxBFGZGF4?si=0sH6Bnj0Qjqf6w6N")
        self.assertTrue(ok)
        self.assertEqual(vid, "VuoxBFGZGF4")
        self.assertEqual(canonical, "https://www.youtube.com/watch?v=VuoxBFGZGF4")

        ok2, canonical2, vid2 = normalize_youtube_url("youtu.be/VuoxBFGZGF4?si=abc123&t=15")
        self.assertTrue(ok2)
        self.assertEqual(vid2, "VuoxBFGZGF4")
        self.assertEqual(canonical2, "https://www.youtube.com/watch?v=VuoxBFGZGF4")

        ok3, canonical3, vid3 = normalize_youtube_url("https://www.youtube.com/shorts/VuoxBFGZGF4?feature=share")
        self.assertTrue(ok3)
        self.assertEqual(vid3, "VuoxBFGZGF4")
        self.assertEqual(canonical3, "https://www.youtube.com/watch?v=VuoxBFGZGF4")

        bad_ok, _, _ = normalize_youtube_url("https://example.com/watch?v=VuoxBFGZGF4")
        self.assertFalse(bad_ok)



    def test_youtube_download_progress_parsing_handles_templates_bytes_and_fragments(self):
        from local_worker.youtube_downloader import compute_hook_progress, parse_progress_line

        # Standard [download] line
        p1 = parse_progress_line("[download]  42.5% of 120.00MiB at 12.00MiB/s ETA 00:05", 0.0, stream_stage=0)
        self.assertAlmostEqual(p1, 0.425 * 0.88, places=3)

        # Structured --progress-template line when _percent_str is NA but bytes are known
        p2 = parse_progress_line("progress:NA|52428800|104857600|104857600", p1, stream_stage=0)
        self.assertAlmostEqual(p2, 0.50 * 0.88, places=3)

        # Second stream (audio) scales smoothly from 88% to 98% without resetting to 0%
        p3 = parse_progress_line("progress:50.0%|1000|2000|2000", 0.88, stream_stage=1)
        self.assertAlmostEqual(p3, 0.93, places=3)

        # Hook dict progress calculation
        h1 = compute_hook_progress({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100}, 0.0, 0)
        self.assertAlmostEqual(h1, 0.22, places=3)


if __name__ == "__main__":
    unittest.main()
