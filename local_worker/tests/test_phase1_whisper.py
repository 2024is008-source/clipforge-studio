"""Automated Unit & Integration Tests for Phase 1: Local Whisper Transcription Engine."""
import os
import sys
import tempfile
import unittest
import wave

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from local_worker.audio_processor import (
    compute_chunk_boundaries,
    extract_normalized_wav,
    probe_media_duration,
    slice_wav_chunks,
)
from local_worker.config import SUPPORTED_LANGUAGES, WHISPER_MODELS
from local_worker.job_manager import JOB_MANAGER, JobManager
from local_worker.speaker_processing import assign_speakers_to_words, check_diarization_available
from local_worker.transcript_validator import merge_and_validate_chunk_words
from local_worker.transcription_jobs import run_transcription_job
from local_worker.whisper_engine import ENGINE


class TestPhase1LocalWhisper(unittest.TestCase):
    def test_models_and_languages_configured(self):
        self.assertEqual(set(WHISPER_MODELS.keys()), {"tiny", "base", "small", "medium", "large-v3"})
        lang_codes = {l["code"] for l in SUPPORTED_LANGUAGES}
        self.assertIn("en", lang_codes)
        self.assertIn("si", lang_codes)
        self.assertIn("auto", lang_codes)

    def test_hardware_detection_and_cpu_fallback(self):
        hw = ENGINE.get_hardware_info()
        self.assertIn("cuda_available", hw)
        self.assertTrue(hw["cpu_fallback_available"])
        self.assertIn(hw["recommended_model"], WHISPER_MODELS)
        self.assertIn(hw["recommended_compute_type"], ("int8_float16", "int8"))

    def test_silence_aligned_chunk_boundaries(self):
        # 800s video with a natural silence at 355.0s and 710.0s
        silences = [120.0, 355.0, 710.0]
        bounds = compute_chunk_boundaries(800.0, silences, target_chunk_sec=360.0, overlap_sec=1.5)
        self.assertEqual(len(bounds), 3)
        # First chunk snaps to silence at 355.0s instead of hard-cutting at 360.0s
        self.assertAlmostEqual(bounds[0][0], 0.0)
        self.assertAlmostEqual(bounds[0][1], 355.0)
        self.assertAlmostEqual(bounds[0][2], 0.0)
        # Second chunk starts at 355.0 - 1.5 = 353.5s (overlap preserved) and snaps to 710.0s
        self.assertAlmostEqual(bounds[1][0], 353.5)
        self.assertAlmostEqual(bounds[1][1], 710.0)
        self.assertAlmostEqual(bounds[1][2], 1.5)

    def test_word_timestamp_validation_and_overlap_deduplication(self):
        # Chunk 0: 0..10s, Chunk 1: offset 8.5s with 1.5s overlap
        chunk0_words = [
            {"word": "Today", "start": 1.25, "end": 1.58},
            {"word": "we", "start": 8.60, "end": 8.90},
            {"word": "build", "start": 9.00, "end": 9.45},
        ]
        chunk1_words = [
            # Duplicate words in the 8.5..10.0s overlap window
            {"word": "we", "start": 0.12, "end": 0.41},
            {"word": "build", "start": 0.52, "end": 0.96},
            # New words after overlap
            {"word": "clips", "start": 1.60, "end": 2.10},
            # Word with missing timestamp -> must be flagged, not silently faked
            {"word": "fast", "start": None, "end": None},
        ]
        report = merge_and_validate_chunk_words(
            [(0.0, 0.0, chunk0_words), (8.5, 1.5, chunk1_words)],
            source_duration=15.0,
        )
        self.assertTrue(report["valid"])
        self.assertEqual([w["w"] for w in report["words"]], ["Today", "we", "build", "clips", "fast"])
        self.assertEqual(report["duplicates_removed"], 2)
        self.assertEqual(report["missing_timing_count"], 1)
        self.assertEqual(report["words"][0]["source"], "whisper")
        self.assertIsNone(report["words"][0]["speaker_id"])
        self.assertTrue(report["words"][-1]["timing_warning"])
        # Verify monotonic non-negative timestamps within source_duration
        for i, w in enumerate(report["words"]):
            self.assertGreaterEqual(w["s"], 0.0)
            self.assertGreater(w["e"], w["s"])
            self.assertLessEqual(w["e"], 15.0)
            if i > 0:
                self.assertGreaterEqual(w["s"], report["words"][i - 1]["s"])

    def test_speaker_processing_independence_and_manual_rename(self):
        diar = check_diarization_available()
        self.assertIn("available", diar)
        words = [
            {"w": "Hello", "s": 0.5, "e": 0.9, "speaker_id": None, "source": "whisper", "timing_warning": False},
            {"w": "Welcome", "s": 2.1, "e": 2.6, "speaker_id": None, "source": "whisper", "timing_warning": False},
        ]
        # Without diarization segments, speaker_id stays None
        no_diar = assign_speakers_to_words(words)
        self.assertIsNone(no_diar[0]["speaker_id"])
        # With diarization segments + manual alias ("Speaker 1" -> "Host")
        with_diar = assign_speakers_to_words(
            words,
            diarization_segments=[{"start": 0.0, "end": 1.5, "speaker": "Speaker 1"}],
            speaker_labels={"Speaker 1": "Host"},
        )
        self.assertEqual(with_diar[0]["speaker_id"], "Host")
        self.assertIsNone(with_diar[1]["speaker_id"])

    def test_real_ffmpeg_audio_extraction_and_cached_transcription_job(self):
        sample_video = "/working_dir/c_e272aece355daf24/user_uploaded_files/ttl=63d/user_uploaded/Eating A Store For $1,000,000 Challenge!(1).mp4"
        if not os.path.exists(sample_video):
            self.skipTest("Sample video not present")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_out = os.path.join(tmpdir, "extracted.wav")
            dur = extract_normalized_wav(sample_video, wav_out)
            self.assertGreater(dur, 15.0)
            with wave.open(wav_out, "rb") as wf:
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)

            # Test full job lifecycle + caching using deterministic mock transcribe callable
            call_count = 0

            def mock_whisper_callable(wav_p, **kwargs):
                nonlocal call_count
                call_count += 1
                return {
                    "text": "If these four contestants eat everything inside",
                    "words": [
                        {"word": "If", "start": 0.10, "end": 0.28},
                        {"word": "these", "start": 0.28, "end": 0.52},
                        {"word": "four", "start": 0.52, "end": 0.78},
                        {"word": "contestants", "start": 0.78, "end": 1.35},
                    ],
                    "detected_language": "en",
                    "device": "cpu",
                    "compute_type": "int8",
                }

            job1 = JOB_MANAGER.create_job("transcription", {"model": "small"})
            # Copy sample_video into job1 dir so cleanup doesn't delete original upload
            temp_media1 = os.path.join(job1["job_dir"], "input.mp4")
            with open(sample_video, "rb") as src_f, open(temp_media1, "wb") as dst_f:
                dst_f.write(src_f.read())

            res1 = run_transcription_job(
                job1["id"],
                temp_media1,
                model_name="small",
                force_retranscribe=True,
                custom_transcribe_fn=mock_whisper_callable,
            )
            self.assertEqual(call_count, 1)
            self.assertEqual(len(res1["words"]), 4)
            self.assertEqual(res1["words"][0]["w"], "If")
            self.assertEqual(res1["words"][0]["source"], "whisper")

            # Second job on identical file & settings must hit cache without calling Whisper again
            job2 = JOB_MANAGER.create_job("transcription", {"model": "small"})
            temp_media2 = os.path.join(job2["job_dir"], "input.mp4")
            with open(sample_video, "rb") as src_f, open(temp_media2, "wb") as dst_f:
                dst_f.write(src_f.read())

            res2 = run_transcription_job(
                job2["id"],
                temp_media2,
                model_name="small",
                force_retranscribe=False,
                custom_transcribe_fn=mock_whisper_callable,
            )
            self.assertEqual(call_count, 1, "Whisper must not re-run when cached transcript exists!")
            self.assertEqual(len(res2["words"]), 4)



    def test_rule6_and_9_merges_contractions_comma_numbers_and_repairs_homophones(self):
        raw_words = [
            {"word": "I", "start": 0.1, "end": 0.2},
            {"word": "will", "start": 0.2, "end": 0.4},
            {"word": "see", "start": 0.4, "end": 0.6},
            {"word": "y", "start": 0.6, "end": 0.7},
            {"word": "'all", "start": 0.7, "end": 0.9},
            {"word": "for", "start": 1.0, "end": 1.2},
            {"word": "$", "start": 1.2, "end": 1.3},
            {"word": "6", "start": 1.3, "end": 1.4},
            {"word": ",614", "start": 1.4, "end": 1.8},
            {"word": "and", "start": 1.9, "end": 2.0},
            {"word": "2", "start": 2.0, "end": 2.1},
            {"word": ".4", "start": 2.1, "end": 2.3},
            {"word": "million", "start": 2.3, "end": 2.7},
            {"word": "with", "start": 2.8, "end": 3.0},
            {"word": "105", "start": 3.0, "end": 3.3},
            {"word": "looks.", "start": 3.3, "end": 3.7},
        ]
        report = merge_and_validate_chunk_words([(0.0, 0.0, raw_words)], source_duration=10.0)
        tokens = [w["w"] for w in report["words"]]
        self.assertEqual(
            tokens,
            ["I", "will", "see", "y'all", "for", "$6,614", "and", "2.4", "million", "with", "105", "loaves."],
        )


if __name__ == "__main__":
    unittest.main()
