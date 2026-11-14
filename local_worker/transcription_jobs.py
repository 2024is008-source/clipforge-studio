"""End-to-end execution of local faster-whisper transcription jobs with silence chunking, offset restoration, and caching."""
import hashlib
import os
import shutil
from typing import Any, Callable
from .audio_processor import extract_normalized_wav, slice_wav_chunks
from .config import DEFAULT_CHUNK_SECONDS, DEFAULT_OVERLAP_SECONDS
from .job_manager import JOB_MANAGER, JobManager
from .speaker_processing import assign_speakers_to_words
from .transcript_validator import merge_and_validate_chunk_words
from .whisper_engine import ENGINE


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_transcription_job(
    jid: str,
    source_media_path: str,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "auto",
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    initial_prompt: str | None = None,
    force_retranscribe: bool = False,
    custom_transcribe_fn: Callable[..., dict] | None = None,
) -> dict[str, Any]:
    """
    Execute a full local Whisper transcription pipeline:
    1. Compute SHA-256 & check local transcript cache (unless force_retranscribe=True).
    2. Extract 16kHz mono PCM WAV via FFmpeg.
    3. Split long audio at natural silence boundaries with overlap.
    4. Transcribe each chunk via faster-whisper.
    5. Restore source timestamps, deduplicate overlap words, and validate.
    6. Clean up temporary WAV slices while persisting the validated transcript.
    """
    job = JOB_MANAGER.get_job(jid)
    if not job:
        raise ValueError(f"Unknown job {jid}")
    job_dir = job["job_dir"]

    try:
        file_sha = compute_file_sha256(source_media_path)
        cache_key = JobManager.compute_cache_key(file_sha, model_name, language, word_timestamps, vad_filter)

        if not force_retranscribe:
            cached = JOB_MANAGER.get_cached(cache_key)
            if cached:
                JOB_MANAGER.update_job(
                    jid,
                    status="completed",
                    progress=1.0,
                    result={**cached, "cached": True},
                    log="Loaded verified transcript from local cache (skipped redundant Whisper run).",
                )
                return cached

        if JOB_MANAGER.is_cancelled(jid):
            JOB_MANAGER.update_job(jid, status="cancelled", log="Job cancelled before audio extraction.")
            return {}

        JOB_MANAGER.update_job(jid, status="extracting_audio", progress=0.05, log="Extracting 16 kHz mono PCM WAV with FFmpeg…")
        wav_path = os.path.join(job_dir, "normalized_16k.wav")
        source_duration = extract_normalized_wav(source_media_path, wav_path)

        chunks = slice_wav_chunks(wav_path, job_dir, target_chunk_sec=chunk_seconds, overlap_sec=overlap_seconds)
        JOB_MANAGER.update_job(
            jid,
            status="transcribing",
            progress=0.12,
            log=f"Audio duration {source_duration:.1f}s split into {len(chunks)} silence-aligned chunk(s).",
        )

        transcribe_callable = custom_transcribe_fn or ENGINE.transcribe_file
        chunk_word_lists: list[tuple[float, float, list[dict]]] = []
        full_texts: list[str] = []
        detected_lang = language if language != "auto" else "en"
        active_device = "cpu"
        active_compute = "int8"
        fallback_notice = None

        for idx, ch_spec in enumerate(chunks):
            if JOB_MANAGER.is_cancelled(jid):
                JOB_MANAGER.update_job(jid, status="cancelled", log=f"Job cancelled at chunk {idx + 1}/{len(chunks)}.")
                return {}

            JOB_MANAGER.update_job(
                jid,
                progress=round(0.12 + 0.78 * (idx / max(1, len(chunks))), 3),
                log=f"Transcribing chunk {idx + 1}/{len(chunks)} (offset {ch_spec['offset_sec']:.1f}s, {ch_spec['duration_sec']:.1f}s)…",
            )
            res = transcribe_callable(
                ch_spec["path"],
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=None if language == "auto" else language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
                should_cancel=lambda: JOB_MANAGER.is_cancelled(jid),
            )
            chunk_word_lists.append((ch_spec["offset_sec"], ch_spec["overlap_sec"], res.get("words", [])))
            if res.get("text"):
                full_texts.append(res["text"])
            if idx == 0:
                detected_lang = res.get("detected_language") or detected_lang
                active_device = res.get("device") or active_device
                active_compute = res.get("compute_type") or active_compute
                fallback_notice = res.get("fallback_notice")

        validation = merge_and_validate_chunk_words(chunk_word_lists, source_duration)
        words_with_speakers = assign_speakers_to_words(validation["words"])

        result_payload = {
            "text": " ".join(full_texts).strip(),
            "words": words_with_speakers,
            "duration": source_duration,
            "language": detected_lang,
            "model": model_name,
            "device": active_device,
            "compute_type": active_compute,
            "timing_source": "whisper",
            "missing_timing_count": validation["missing_timing_count"],
            "duplicates_removed": validation["duplicates_removed"],
            "warnings": validation["warnings"] + ([fallback_notice] if fallback_notice else []),
            "cached": False,
            "file_sha256": file_sha,
        }

        JOB_MANAGER.save_cache(cache_key, result_payload)
        JOB_MANAGER.update_job(
            jid,
            status="completed",
            progress=1.0,
            result=result_payload,
            log=f"Transcription complete: {len(words_with_speakers)} validated words on {active_device} ({active_compute}).",
        )
        return result_payload
    except InterruptedError:
        JOB_MANAGER.update_job(jid, status="cancelled", log="Transcription interrupted by cancellation.")
        return {}
    except Exception as e:
        JOB_MANAGER.update_job(jid, status="error", error=str(e), log=f"Error: {e}")
        raise
    finally:
        # Remove temporary audio files while preserving job metadata
        for fname in os.listdir(job_dir):
            if fname.endswith((".wav", ".mp4", ".webm", ".mov", ".mkv", ".bin")):
                try:
                    os.remove(os.path.join(job_dir, fname))
                except OSError:
                    pass
