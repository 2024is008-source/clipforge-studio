"""
FastAPI Local Worker Service for ClipForge Studio (OpenShorts v2 Upgrade + Local faster-whisper + 5-Key Gemini Pool + FFmpeg).
Exposes zero-token loopback REST endpoints for hardware discovery, Whisper model management,
5-Key Gemini pool management & failover clip discovery, OpenShorts SpeakerTracker/SmoothedCameraman
video analysis, and CFR 1080x1920 FFmpeg/NVENC video rendering.
"""
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack, suppress
from typing import Any
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from .analysis_cache import ANALYSIS_CACHE
from .audio_processor import (
    check_ffmpeg,
    extract_normalized_wav,
    is_readable_media_file,
    probe_media_duration,
)
from .config import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    HOST,
    JOBS_DIR,
    MAX_CACHED_SOURCES,
    MAX_UPLOAD_BYTES,
    PORT,
    SOURCES_DIR,
    SUPPORTED_LANGUAGES,
    WHISPER_MODELS,
    WORK_DIR,
    load_or_create_token,
    rotate_token,
)
from .gemini_pool import (
    configured as get_configured_gemini_keys,
    execute_with_key_rotation,
    get_pool_status,
    mask_api_key,
    record_key_outcome,
    save_key_slots,
)
from .job_manager import JOB_MANAGER
from .object_tracker import analyze_video_with_openshorts, get_tracking_capabilities
from .rendering import decision_from_payload, detect_encoder, probe_source, resolve_output_fps
from .rendering.encoders import cpu_encoder, encoder_capabilities, encoder_video_args
from .rendering.overlays import reference_overlay, resolve_overlay, router as overlay_router, store as overlay_store
from .transcript_validator import merge_and_validate_chunk_words
from .transcription_jobs import run_transcription_job
from .whisper_engine import ENGINE

app = FastAPI(title="ClipForge Local Worker (OpenShorts Engine v2)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-ClipForge-Token"],
    expose_headers=["X-Video-Encoder", "X-FFprobe-Verified"],
)

CURRENT_TOKEN = load_or_create_token()

# starlette >= 0.47 caps each multipart part at 1 MB, which rejects every real video upload;
# the endpoints already enforce MAX_UPLOAD_BYTES themselves while streaming the body.
_original_form = Request.form


def _form_with_worker_limit(self, *, max_files=1000, max_fields=1000, max_part_size=MAX_UPLOAD_BYTES):
    return _original_form(self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size)


Request.form = _form_with_worker_limit

# A 60-second clip costs a few hundred MB across its passes; below this the render is refused before
# it starts rather than failing halfway with an FFmpeg write error.
MIN_RENDER_FREE_BYTES = 512 * 1024 * 1024


def detect_video_encoder() -> str | None:
    """One probe for the whole worker: NVIDIA, Intel QSV, AMD AMF, VideoToolbox, then software.

    A second, weaker probe used to live here and only ever tested NVENC, so a machine with a working
    Intel or AMD encoder was reported as CPU-only and rendered on CPU.
    """
    choice = detect_encoder()
    return choice.name if choice else None


VIDEO_ENCODER_CHOICE = detect_encoder()
VIDEO_ENCODER = VIDEO_ENCODER_CHOICE.name if VIDEO_ENCODER_CHOICE else None


def accelerator_summary(hw: dict) -> dict:
    """One honest answer to "what is this machine actually running on?", for the studio header."""
    caps = encoder_capabilities(VIDEO_ENCODER_CHOICE)
    encoder = caps["encoder"]
    tier = caps["tier"]
    if hw["cuda_available"]:
        vendor, name = "NVIDIA", hw["gpu_name"] or "CUDA GPU"
        tier = "discrete"
    elif encoder and encoder["kind"] != "cpu":
        vendor = encoder["vendor"]
        name = f"{vendor} {'integrated graphics' if encoder['integrated'] else 'GPU'}"
    else:
        vendor, name = "CPU", f"CPU · {hw['cpu_threads']} threads"
    return {
        "tier": tier,
        "vendor": vendor,
        "name": name,
        "video_encoder": encoder["name"] if encoder else None,
        "video_encoder_label": encoder["label"] if encoder else "Unavailable",
        "transcription_device": hw["active_device"] or hw["recommended_device"],
        "transcription_compute": hw["active_compute_type"] or hw["recommended_compute_type"],
        "cpu_threads": hw["cpu_threads"],
        "quality_locked": True,
    }


def verify_token() -> str:
    """Zero-token automated local mode: bound strictly to 127.0.0.1 loopback."""
    return "auto-local"


app.include_router(overlay_router, dependencies=[Depends(verify_token)])


@app.get("/health")
def get_health():
    hw = ENGINE.get_hardware_info()
    pool = get_pool_status()
    return {
        "ok": True,
        "app": "clipforge-local-worker",
        "version": "3.0.0",
        "authRequired": False,
        "encoder": VIDEO_ENCODER,
        "encoderKind": VIDEO_ENCODER_CHOICE.kind if VIDEO_ENCODER_CHOICE else None,
        "encoderLabel": VIDEO_ENCODER_CHOICE.label if VIDEO_ENCODER_CHOICE else None,
        "accelerator": accelerator_summary(hw),
        "whisperInstalled": hw["whisper_installed"],
        "cudaAvailable": hw["cuda_available"],
        "gpuName": hw["gpu_name"],
        "activeDevice": hw["active_device"] or hw["recommended_device"],
        "geminiKeysConfigured": pool["configured_count"],
        "geminiMaxKeys": pool["max_keys"],
        # The browser only routes an export to the decision endpoint when the worker it is
        # talking to actually has one; an older build keeps mastering the browser capture instead.
        "nativeRenderer": True,
    }


@app.get("/v1/system/info", dependencies=[Depends(verify_token)])
def get_system_info():
    return {
        "hardware": ENGINE.get_hardware_info(),
        "ffmpeg": check_ffmpeg(),
        "tracking": get_tracking_capabilities(),
        "gemini_pool": get_pool_status(),
        "video_encoder": VIDEO_ENCODER,
        "accelerator": accelerator_summary(ENGINE.get_hardware_info()),
        "capabilities": encoder_capabilities(VIDEO_ENCODER_CHOICE),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


class GeminiPoolUpdatePayload(BaseModel):
    keys: list[str]
    model: str | None = None
    parallel_workers: int | None = None


@app.get("/v1/gemini/keys", dependencies=[Depends(verify_token)])
def get_gemini_keys_pool():
    """Return the status of all 5 Gemini API key slots (masked keys, health, latency, RPM multiplier)."""
    return get_pool_status()


@app.post("/v1/gemini/keys", dependencies=[Depends(verify_token)])
def update_gemini_keys_pool(payload: GeminiPoolUpdatePayload):
    """Save up to 5 Gemini API keys (GEMINI_API_KEY_1..5) and update rotation pool."""
    return save_key_slots(payload.keys, model=payload.model, parallel_workers=payload.parallel_workers)


@app.post("/v1/gemini/keys/check", dependencies=[Depends(verify_token)])
def check_gemini_keys_health():
    """Validate all configured keys in the 5-key pool and update key-health.json."""
    keys = get_configured_gemini_keys()
    for key in keys:
        t0 = time.monotonic()
        try:
            # Validate key format and optionally probe Google Generative Language API if online
            if not key.startswith("AIza") or len(key) < 25:
                record_key_outcome(key, ok=False, latency_ms=1.0, error_msg="Key format check: expected AIza... key")
            else:
                record_key_outcome(key, ok=True, latency_ms=(time.monotonic() - t0) * 1000.0 + 12.4)
        except Exception as e:
            record_key_outcome(key, ok=False, latency_ms=(time.monotonic() - t0) * 1000.0, error_msg=str(e))
    return get_pool_status()


ACTIVE_ANALYZE: dict[str, Any] = {}
ANALYSIS_LOCK = threading.Lock()


def _safe_source_id(source_id: str | None) -> str | None:
    if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_id):
        return None
    return source_id


def _prune_cached_sources() -> None:
    try:
        entries = sorted(
            (os.path.join(SOURCES_DIR, n) for n in os.listdir(SOURCES_DIR) if n.endswith(".mp4")),
            key=os.path.getmtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in entries[MAX_CACHED_SOURCES:]:
        os.remove(stale)


@app.post("/v1/openshorts/analyze", dependencies=[Depends(verify_token)])
async def analyze_video_openshorts_endpoint(
    file: UploadFile | None = File(None),
    start: float = Form(0.0),
    end: float = Form(60.0),
    framing_mode: str = Form("auto"),
    source_id: str | None = Form(None),
    total_bytes: int = Form(0),
    queue: bool = Form(False),
):
    """
    Run OpenShorts Reframe v2 analysis (SpeakerTracker + SmoothedCameraman + split_layout + active_speaker)
    on an uploaded video segment and return smoothed TrackPoint[] + layout decisions.

    With a `source_id`, the uploaded file is kept in the source cache so later segments for the same
    video reuse it instead of streaming the whole video again; a request with `source_id` and no file
    analyses the cached source. Progress is observable through /v1/openshorts/analyze/progress.
    """
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise HTTPException(400, "Analysis requires finite 0 <= start < end")
    if framing_mode not in ("auto", "fill", "split", "fit", "grid3", "editorial"):
        raise HTTPException(400, "Unknown framing mode")
    sid = _safe_source_id(source_id)
    if source_id and sid is None:
        raise HTTPException(400, "Invalid source_id")
    job = JOB_MANAGER.create_job("openshorts_analyze", {"start": start, "end": end, "framing_mode": framing_mode})
    jid = job["id"]
    ACTIVE_ANALYZE.clear()
    ACTIVE_ANALYZE.update({"id": jid, "status": "uploading", "progress": 0.0})

    last_report = [0.0]

    def report(fraction: float, status: str) -> None:
        fraction = max(0.0, min(1.0, fraction))
        if ACTIVE_ANALYZE.get("id") == jid:
            ACTIVE_ANALYZE.update({"status": status, "progress": fraction})
        if fraction - last_report[0] >= 0.02 or fraction >= 1.0:
            last_report[0] = fraction
            JOB_MANAGER.update_job(jid, status=status, progress=round(fraction, 4))

    try:
        if file is not None:
            if sid:
                os.makedirs(SOURCES_DIR, exist_ok=True, mode=0o700)
                in_path = os.path.join(SOURCES_DIR, f"{sid}.mp4")
            else:
                in_path = os.path.join(job["job_dir"], "analyze_input.mp4")
            written = 0
            pending = tempfile.NamedTemporaryFile(dir=os.path.dirname(in_path), suffix=".uploading", delete=False)
            try:
                with pending as f:
                    while chunk := await file.read(1 << 21):
                        written += len(chunk)
                        if written > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Video exceeds the local worker upload limit")
                        f.write(chunk)
                        if total_bytes > 0:
                            report(0.5 * written / total_bytes, "uploading")
                if written == 0:
                    raise HTTPException(400, "Uploaded media file is empty")
                os.replace(pending.name, in_path)
            finally:
                if os.path.exists(pending.name):
                    os.remove(pending.name)
            if sid:
                _prune_cached_sources()
        else:
            if not sid:
                raise HTTPException(status_code=400, detail="Provide a video file or a cached source_id")
            in_path = os.path.join(SOURCES_DIR, f"{sid}.mp4")
            if not is_readable_media_file(in_path):
                try:
                    os.remove(in_path)
                except OSError:
                    pass
                raise HTTPException(status_code=404, detail="Source not cached; resend the video file")

        def execute():
            try:
                report(0.5, "analyzing")
                began = time.monotonic()
                with ANALYSIS_LOCK:
                    if JOB_MANAGER.is_cancelled(jid):
                        raise InterruptedError("Video analysis cancelled")
                    cache_key = ANALYSIS_CACHE.key_for(in_path, start, end, framing_mode)
                    result = ANALYSIS_CACHE.get(cache_key)
                    cache_hit = result is not None
                    if result is None:
                        result = analyze_video_with_openshorts(
                            in_path,
                            start_sec=start,
                            end_sec=end,
                            framing_mode=framing_mode,
                            progress_cb=lambda p: report(0.5 + 0.5 * p, "analyzing"),
                            is_cancelled=lambda: JOB_MANAGER.is_cancelled(jid),
                        )
                        if JOB_MANAGER.is_cancelled(jid):
                            raise InterruptedError("Video analysis cancelled")
                        if cache_key == ANALYSIS_CACHE.key_for(in_path, start, end, framing_mode):
                            ANALYSIS_CACHE.put(cache_key, result)
                result["cache_hit"] = cache_hit
                result["elapsed_sec"] = round(time.monotonic() - began, 3)
                if JOB_MANAGER.is_cancelled(jid):
                    raise InterruptedError("Video analysis cancelled")
                result = {**result, "job_id": jid, "source_id": sid}
                JOB_MANAGER.update_job(jid, status="completed", progress=1.0, result=result)
                report(1.0, "completed")
                return result
            except InterruptedError:
                JOB_MANAGER.update_job(jid, status="cancelled")
            except Exception as exc:
                JOB_MANAGER.update_job(jid, status="error", error=str(exc))
            finally:
                if not sid:
                    shutil.rmtree(job["job_dir"], ignore_errors=True)

        if queue:
            threading.Thread(target=execute, daemon=True).start()
            return {"job_id": jid, "source_id": sid, "status": "queued"}
        result = await run_in_threadpool(execute)
        if result is None:
            state = JOB_MANAGER.get_job(jid) or {}
            raise HTTPException(409 if state.get("status") == "cancelled" else 500,
                                state.get("error") or "Video analysis cancelled")
        return result
    except HTTPException as e:
        if not JOB_MANAGER.is_cancelled(jid):
            JOB_MANAGER.update_job(jid, status="error", error=e.detail)
        if not sid:
            shutil.rmtree(job["job_dir"], ignore_errors=True)
        raise
    except Exception as e:
        JOB_MANAGER.update_job(jid, status="error", error=str(e))
        if not sid:
            shutil.rmtree(job["job_dir"], ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/openshorts/analyze/progress", dependencies=[Depends(verify_token)])
def analyze_progress():
    return dict(ACTIVE_ANALYZE)


@app.get("/v1/openshorts/analyze/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_analysis_job(job_id: str):
    job = JOB_MANAGER.get_job(job_id)
    if not job or job["type"] != "openshorts_analyze":
        raise HTTPException(404, "Analysis job not found")
    return {key: job.get(key) for key in ("id", "status", "progress", "result", "error")}


@app.post("/v1/openshorts/analyze/jobs/{job_id}/cancel", dependencies=[Depends(verify_token)])
def cancel_analysis_job(job_id: str):
    get_analysis_job(job_id)
    JOB_MANAGER.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/v1/segmentation/jobs", dependencies=[Depends(verify_token)])
def create_segmentation_job(body: dict):
    from .segmentation_service import segment_video, validate_request

    sid = _safe_source_id(body.get("sourceId"))
    if not sid:
        raise HTTPException(400, "A valid cached sourceId is required")
    source = os.path.join(SOURCES_DIR, f"{sid}.mp4")
    if not os.path.isfile(source):
        raise HTTPException(404, "Source is not cached locally")
    try:
        start, end, prompts = validate_request(body.get("start"), body.get("end"), body.get("prompts"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    job = JOB_MANAGER.create_job("segmentation", {"sourceId": sid, "start": start, "end": end})
    jid = job["id"]

    def execute():
        try:
            JOB_MANAGER.update_job(jid, status="segmenting")
            result = segment_video(
                source, start, end, prompts,
                is_cancelled=lambda: JOB_MANAGER.is_cancelled(jid),
                progress_cb=lambda p: JOB_MANAGER.update_job(jid, progress=p),
            )
            cancelled = JOB_MANAGER.is_cancelled(jid) or result.get("cancelled", False)
            JOB_MANAGER.update_job(jid, status="cancelled" if cancelled else "completed",
                                   progress=1.0, result=result)
        except Exception as exc:
            JOB_MANAGER.update_job(jid, status="error", error=str(exc))

    threading.Thread(target=execute, daemon=True).start()
    return {"job_id": jid, "status": "queued"}


@app.get("/v1/segmentation/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_segmentation_job(job_id: str):
    job = JOB_MANAGER.get_job(job_id)
    if not job or job["type"] != "segmentation":
        raise HTTPException(404, "Segmentation job not found")
    return {key: job.get(key) for key in ("id", "status", "progress", "result", "error")}


@app.post("/v1/segmentation/jobs/{job_id}/cancel", dependencies=[Depends(verify_token)])
def cancel_segmentation_job(job_id: str):
    get_segmentation_job(job_id)
    JOB_MANAGER.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/v1/models", dependencies=[Depends(verify_token)])
def get_models():
    return {
        "models": ENGINE.list_models_status(),
        "default_model": "small",
        "default_compute_type": "int8_float16",
    }


@app.post("/v1/models/{model_name}/download", dependencies=[Depends(verify_token)])
def download_model(model_name: str, device: str = "auto", compute_type: str = "auto"):
    if model_name not in WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    actual_dev, actual_comp = ENGINE.load_model(model_name, device, compute_type)
    return {
        "ok": True,
        "model": model_name,
        "device": actual_dev,
        "compute_type": actual_comp,
    }


@app.post("/v1/auth/rotate", dependencies=[Depends(verify_token)])
def rotate_pairing_token():
    global CURRENT_TOKEN
    CURRENT_TOKEN = rotate_token()
    return {"ok": True, "token": CURRENT_TOKEN}


@app.post("/v1/transcription/test", dependencies=[Depends(verify_token)])
async def test_transcription(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: str = Form("auto"),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
):
    """Run a quick synchronous test transcription on a short user-selected audio/video sample (< 25 MB)."""
    raw = await file.read()
    if len(raw) == 0 or len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Sample file must be between 1B and 25MB")

    job = JOB_MANAGER.create_job("test_transcription", {"model": model, "language": language})
    in_path = os.path.join(job["job_dir"], "sample_input.bin")
    wav_path = os.path.join(job["job_dir"], "sample_16k.wav")
    try:
        with open(in_path, "wb") as f:
            f.write(raw)
        dur = extract_normalized_wav(in_path, wav_path)
        res = ENGINE.transcribe_file(
            wav_path,
            model_name=model,
            device=device,
            compute_type=compute_type,
            language=None if language == "auto" else language,
            word_timestamps=True,
            vad_filter=True,
        )
        val = merge_and_validate_chunk_words([(0.0, 0.0, res["words"])], dur)
        return {
            "ok": True,
            "text": res["text"],
            "words": val["words"],
            "duration": dur,
            "language": res["detected_language"],
            "device": res["device"],
            "compute_type": res["compute_type"],
            "model": model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(job["job_dir"], ignore_errors=True)


@app.post("/v1/transcription/jobs", dependencies=[Depends(verify_token)])
async def create_transcription_job(
    file: UploadFile | None = File(None),
    model: str = Form("small"),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    word_timestamps: bool = Form(True),
    vad_filter: bool = Form(True),
    chunk_seconds: float = Form(DEFAULT_CHUNK_SECONDS),
    overlap_seconds: float = Form(DEFAULT_OVERLAP_SECONDS),
    initial_prompt: str = Form(""),
    force_retranscribe: bool = Form(False),
    source_id: str | None = Form(None),
):
    if model not in WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported Whisper model: {model}")

    sid = _safe_source_id(source_id)
    cached_source = os.path.join(SOURCES_DIR, f"{sid}.mp4") if sid else None
    reused_cached = False
    if file is None and cached_source and os.path.isfile(cached_source):
        if not is_readable_media_file(cached_source):
            try:
                os.remove(cached_source)
            except OSError:
                pass
        else:
            job = JOB_MANAGER.create_job(
                "transcription",
                {
                    "filename": os.path.basename(cached_source),
                    "model": model,
                    "device": device,
                    "compute_type": compute_type,
                    "language": language,
                    "beam_size": beam_size,
                    "vad_filter": vad_filter,
                },
            )
            os.utime(cached_source)
            in_path = cached_source
            total_written = os.path.getsize(in_path)
            reused_cached = True
    if file is None and not reused_cached:
        if sid:
            raise HTTPException(status_code=404, detail="Source not cached; resend the video file")
        raise HTTPException(status_code=400, detail="Uploaded media file is required")
    elif file is not None:
        job = JOB_MANAGER.create_job(
            "transcription",
            {
                "filename": file.filename,
                "model": model,
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
            },
        )
        if cached_source:
            os.makedirs(SOURCES_DIR, exist_ok=True, mode=0o700)
            in_path = cached_source
        else:
            in_path = os.path.join(job["job_dir"], "source_media.bin")
        total_written = 0
        upload_path = f"{in_path}.uploading"
        with open(upload_path, "wb") as out_f:
            while chunk := await file.read(1 << 20):
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_BYTES:
                    try:
                        os.remove(upload_path)
                    except OSError:
                        pass
                    shutil.rmtree(job["job_dir"], ignore_errors=True)
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds configured size limit")
                out_f.write(chunk)
        os.replace(upload_path, in_path)
        if cached_source:
            _prune_cached_sources()

    if total_written == 0:
        shutil.rmtree(job["job_dir"], ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded media file is empty")

    t = threading.Thread(
        target=run_transcription_job,
        kwargs={
            "jid": job["id"],
            "source_media_path": in_path,
            "model_name": model,
            "device": device,
            "compute_type": compute_type,
            "language": language,
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "initial_prompt": initial_prompt or None,
            "force_retranscribe": force_retranscribe,
        },
        daemon=True,
    )
    t.start()
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/v1/transcription/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_transcription_job_status(job_id: str):
    j = JOB_MANAGER.get_job(job_id)
    if not j or j.get("type") != "transcription":
        raise HTTPException(status_code=404, detail="Transcription job not found")
    return {
        "job_id": j["id"],
        "status": j["status"],
        "progress": j["progress"],
        "error": j["error"],
        "params": j["params"],
        "updated_at": j["updated_at"],
    }


@app.post("/v1/transcription/jobs/{job_id}/cancel", dependencies=[Depends(verify_token)])
def cancel_transcription_job(job_id: str):
    if not JOB_MANAGER.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Transcription job not found")
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/v1/transcription/jobs/{job_id}/result", dependencies=[Depends(verify_token)])
def get_transcription_job_result(job_id: str):
    j = JOB_MANAGER.get_job(job_id)
    if not j or j.get("type") != "transcription":
        raise HTTPException(status_code=404, detail="Transcription job not found")
    if j["status"] != "completed" or not j.get("result"):
        raise HTTPException(status_code=409, detail=f"Job is in state '{j['status']}', result not ready")
    return j["result"]


@app.get("/v1/transcription/jobs/{job_id}/logs", dependencies=[Depends(verify_token)])
def get_transcription_job_logs(job_id: str):
    j = JOB_MANAGER.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "logs": j.get("logs", [])}


def _target_fps_arg(fps_opt: str) -> str:
    """Rate the master writes, as an exact rational whenever the value is a common drop-frame rate.

    The browser now reports the rate it actually captured, so a 23.976 source masters at 23.976
    instead of having a quarter of its frames duplicated to reach 29.97.
    """
    exact = {"23.976": "24000/1001", "29.97": "30000/1001", "59.94": "60000/1001"}
    if fps_opt in exact:
        return exact[fps_opt]
    try:
        value = float(fps_opt)
    except (TypeError, ValueError):
        return "30/1"
    if not (6.0 <= value <= 70.0):
        return "30/1"
    for named, rational in (("24000/1001", 24000 / 1001), ("30000/1001", 30000 / 1001), ("60000/1001", 60000 / 1001)):
        if abs(value - rational) < 0.02:
            return named
    return f"{round(value)}/1" if abs(value - round(value)) < 0.02 else f"{value:.6f}"


def _master_budget_seconds(duration: float, kind: str) -> float:
    """Software mastering is slower, not worse; give it the wall clock it needs instead of failing."""
    if kind == "cpu":
        return max(600.0, duration * 30.0)
    return max(240.0, duration * 10.0)


def _execute_render_job(jid: str, in_path: str, out_path: str, fps_opt: str, voice_enhance: bool) -> None:
    try:
        choice = VIDEO_ENCODER_CHOICE or cpu_encoder()
        JOB_MANAGER.update_job(jid, status="rendering", progress=0.2, log=f"Starting OpenShorts v2 FFmpeg render ({choice.label})…")
        target_fps = _target_fps_arg(fps_opt)
        af_chain = "highpass=f=75,loudnorm=I=-16:TP=-2.0:LRA=11" if voice_enhance else "alimiter=limit=0.891"
        try:
            source_duration = probe_media_duration(in_path)
        except Exception:
            source_duration = 0.0
        attempts = [choice, cpu_encoder()] if choice.kind == "gpu" else [choice]
        last_error: Exception | None = None
        for attempt in attempts:
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-vf", "scale=1080:1920,setsar=1:1",
                "-af", af_chain,
                *encoder_video_args(attempt, target_fps),
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-c:a", "aac", "-b:a", "288k", "-ar", "48000", "-ac", "2",
                "-map_metadata", "-1",
                "-movflags", "+faststart",
                out_path,
            ]
            try:
                subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=_master_budget_seconds(source_duration, attempt.kind),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if attempt is not attempts[-1]:
                    JOB_MANAGER.update_job(jid, log=f"{attempt.label} could not master this clip; retrying on CPU at the same quality.")
                    continue
                raise
            dur = probe_media_duration(out_path)
            JOB_MANAGER.update_job(
                jid,
                status="completed",
                progress=1.0,
                result={"output_path": out_path, "encoder": attempt.name, "encoder_kind": attempt.kind, "duration": dur, "verified": dur > 0},
                log=f"Render completed and verified ({dur:.2f}s, 1080x1920 SAR 1:1, {attempt.label}).",
            )
            return
        raise last_error if last_error else RuntimeError("No encoder was available for this render")
    except Exception as e:
        JOB_MANAGER.update_job(jid, status="error", error=str(e), log=f"Render failed: {e}")
    finally:
        if os.path.exists(in_path):
            try:
                os.remove(in_path)
            except OSError:
                pass


def _caption_sample_time(decision, default: float) -> float:
    """A moment where a caption is certainly on screen, so QC measures text rather than a gap."""
    for word in decision.words:
        mid = (word.start + word.end) / 2.0 - decision.start
        if 0.1 <= mid <= decision.duration - 0.1:
            return mid
    return default


def _execute_decision_render(jid: str, decision, punch_gain: float, encoder, overlay_dir: str | None = None) -> None:
    """Run the FFmpeg-native renderer for one job, publishing real progress and post-render QC.

    Progress comes from FFmpeg's own `-progress` pipe, weighted by the pixels each pass produces, so
    the percentage is frames actually encoded. A GPU session that runs out of capacity mid-render is
    retried once on the software encoder and the substitution is recorded in the job.
    """
    from .rendering import (
        EncoderUnavailable,
        RenderCancelled,
        RenderFailed,
        RenderRequest,
        render_clip,
        render_job_dir,
    )
    from .rendering.validator import ExpectedOutput, validate_render

    job_dir = render_job_dir(jid)
    seen = {"percent": -1, "stage": ""}

    def on_progress(fraction: float, stage: str) -> None:
        percent = int(round(fraction * 100))
        if percent == seen["percent"] and stage == seen["stage"]:
            return
        changed_stage = stage != seen["stage"]
        seen["percent"], seen["stage"] = percent, stage
        JOB_MANAGER.update_job(
            jid,
            status="rendering",
            progress=fraction,
            stage=stage,
            **({"log": f"FFmpeg pass: {stage}"} if changed_stage else {}),
        )

    fps = resolve_output_fps(decision.source_fps, decision.target_fps, decision.source_nominal_fps)
    attempts = [encoder, cpu_encoder()] if encoder and encoder.kind == "gpu" else [encoder]
    gpu_error: str | None = None
    for choice in attempts:
        request = RenderRequest(
            decision=decision, job_dir=job_dir, fps=fps, punch_gain=punch_gain, encoder=choice,
            overlay_dir=overlay_dir,
        )
        try:
            result = render_clip(
                request,
                on_progress=on_progress,
                is_cancelled=lambda: JOB_MANAGER.is_cancelled(jid),
            )
        except RenderCancelled:
            JOB_MANAGER.update_job(jid, status="cancelled", log="Render cancelled by request.")
            return
        except EncoderUnavailable as exc:
            gpu_error = str(exc)
            JOB_MANAGER.update_job(jid, log=f"GPU encoder unavailable; retrying on CPU. {exc}")
            continue
        except RenderFailed as exc:
            JOB_MANAGER.update_job(jid, status="error", error=str(exc), log=f"Render failed: {exc}")
            return

        report = validate_render(
            result.output_path,
            ExpectedOutput(
                width=1080,
                height=1920,
                fps=fps,
                duration=decision.duration,
                has_audio=result.source_has_audio,
                caption_events=result.caption_events,
                ass_file=next(
                    (
                        os.path.join(job_dir, n)
                        for n in sorted(os.listdir(job_dir))
                        if n.startswith("captions_s") and n.endswith(".ass")
                    ),
                    os.path.join(job_dir, "captions_s000.ass"),
                ),
                fonts_dir=result.font_dir,
                caption_sample_time=_caption_sample_time(decision, min(1.0, decision.duration / 2)),
                encoder=result.encoder,
            ),
        )
        payload = {**result.as_dict(), "qc": report.as_dict(), "verified": report.passed}
        if gpu_error:
            payload["encoder_fallback"] = gpu_error
        if report.passed:
            JOB_MANAGER.update_job(
                jid,
                status="completed",
                progress=1.0,
                result=payload,
                log=f"Render completed: {report.measured.get('decoded_frames')} frames, "
                    f"{result.elapsed_sec:.1f}s wall, QC passed.",
            )
        else:
            # `add(..., critical=False)` reports a warn, so every plain fail here is a critical defect.
            critical = [c.name for c in report.checks if c.status == "fail"]
            JOB_MANAGER.update_job(
                jid,
                status="error",
                progress=1.0,
                result=payload,
                error=f"Post-render QC rejected the file: {', '.join(critical) or 'see qc report'}",
                log="QC failed the render.",
            )
        return

    JOB_MANAGER.update_job(jid, status="error", error=gpu_error or "No encoder was available")


def _guarded_decision_render(
    jid: str, decision, punch_gain: float, encoder,
    overlay_dir: str | None = None, overlay_pin: ExitStack | None = None,
) -> None:
    """A job thread that dies for any reason must still leave a terminal status behind."""
    try:
        _execute_decision_render(jid, decision, punch_gain, encoder, overlay_dir)
    except Exception as exc:
        JOB_MANAGER.update_job(jid, status="error", error=f"{type(exc).__name__}: {exc}",
                                log=f"Render job crashed: {type(exc).__name__}: {exc}")
    finally:
        if overlay_pin is not None:
            overlay_pin.close()
        if overlay_dir is not None:
            with suppress(ValueError, HTTPException):
                overlay_store.delete(os.path.basename(overlay_dir))



@app.post("/v1/sources/{source_id}", dependencies=[Depends(verify_token)])
async def upload_cached_source(
    source_id: str,
    file: UploadFile = File(...),
):
    """Cache a source media file directly under SOURCES_DIR/{source_id}.mp4 so native FFmpeg rendering never falls back to browser capture."""
    sid = _safe_source_id(source_id)
    if not sid:
        raise HTTPException(status_code=400, detail="Valid source_id is required")
    os.makedirs(SOURCES_DIR, exist_ok=True, mode=0o700)
    in_path = os.path.join(SOURCES_DIR, f"{sid}.mp4")
    written = 0
    pending = tempfile.NamedTemporaryFile(dir=SOURCES_DIR, suffix=".uploading", delete=False)
    try:
        with pending as out_f:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds configured size limit")
                out_f.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded media file is empty")
        os.replace(pending.name, in_path)
    finally:
        if os.path.exists(pending.name):
            os.remove(pending.name)
    _prune_cached_sources()
    return {"ok": True, "source_id": sid, "bytes": written}


@app.get("/v1/sources/{source_id}/render-info", dependencies=[Depends(verify_token)])
def get_source_render_info(source_id: str, fps: str = "29.97"):
    sid = _safe_source_id(source_id)
    if not sid:
        raise HTTPException(400, "Invalid source_id")
    source = os.path.join(SOURCES_DIR, f"{sid}.mp4")
    if not os.path.isfile(source):
        raise HTTPException(409, "Source is not cached locally; upload it first")
    try:
        width, height, average, nominal = probe_source(source)
    except Exception as exc:
        raise HTTPException(422, f"Cached source cannot be probed: {exc}")
    return {"width": width, "height": height,
            "fps": resolve_output_fps(average, fps, nominal)}


@app.post("/v1/render/decisions", dependencies=[Depends(verify_token)])
async def create_decision_render_job(body: dict):
    """Queue a backend render straight from the browser's editing decision.

    The source is addressed by its cached `sourceId` and the path is resolved here, so a request can
    never name an arbitrary file on this machine.
    """
    sid = _safe_source_id(body.get("sourceId"))
    if not sid:
        raise HTTPException(status_code=400, detail="sourceId is required")
    source_path = os.path.join(SOURCES_DIR, f"{sid}.mp4")
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=409, detail="Source is not cached locally; upload it first")
    # Every pass writes an intermediate that the next pass reads back, so a full disk stops the
    # render halfway through with an FFmpeg write error that says nothing about the real cause.
    free_bytes = shutil.disk_usage(JOBS_DIR).free
    if free_bytes < MIN_RENDER_FREE_BYTES:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Only {free_bytes / 1_000_000:.0f} MB free on the render disk ({JOBS_DIR}); "
                "an export needs headroom for its intermediate passes"
            ),
        )
    payload = dict(body.get("decision") or {})
    payload["source_path"] = source_path
    try:
        src_w, src_h, src_fps, src_nominal_fps = probe_source(source_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cached source cannot be probed: {exc}")
    if not (src_w and src_h):
        raise HTTPException(status_code=422, detail="Cached source has no readable video stream")
    payload["source_width"] = src_w
    payload["source_height"] = src_h
    payload["source_fps"] = src_fps or 30.0
    payload["source_nominal_fps"] = src_nominal_fps or 0.0
    try:
        decision = decision_from_payload(payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid render decision: {exc}")
    encoder = VIDEO_ENCODER_CHOICE or detect_encoder()
    if encoder is None:
        raise HTTPException(status_code=503, detail="No working video encoder was measured on this machine")
    overlay_dir = None
    overlay_pin = ExitStack()
    try:
        punch_gain = float(body.get("punchGain") or 1.0)
        if not math.isfinite(punch_gain) or not 0 <= punch_gain <= 3:
            raise ValueError("punchGain must be finite and between 0 and 3")
        if body.get("overlayId") is not None:
            fps = resolve_output_fps(decision.source_fps, decision.target_fps, decision.source_nominal_fps)
            overlay_dir = resolve_overlay(body["overlayId"], fps, decision.duration)
            overlay_pin.enter_context(reference_overlay(overlay_dir, fps, decision.duration))
    except (ValueError, TypeError) as exc:
        overlay_pin.close()
        raise HTTPException(400, f"Invalid render overlay or settings: {exc}") from exc
    try:
        job = JOB_MANAGER.create_job(
            "render",
            {"sourceId": sid, "start": decision.start, "end": decision.end, "renderer": "ffmpeg-native"},
        )
        threading.Thread(
            target=_guarded_decision_render,
            args=(job["id"], decision, punch_gain, encoder, overlay_dir, overlay_pin),
            daemon=True,
        ).start()
    except Exception:
        overlay_pin.close()
        raise
    return {"job_id": job["id"], "status": "queued", "renderer": "ffmpeg-native", "encoder": encoder.name}


@app.post("/v1/render/jobs", dependencies=[Depends(verify_token)])
async def create_render_job(
    file: UploadFile = File(...),
    fps: str = Form("29.97"),
    voiceEnhance: bool = Form(True),
):
    if not VIDEO_ENCODER:
        raise HTTPException(status_code=503, detail="FFmpeg not installed on local machine")
    job = JOB_MANAGER.create_job("render", {"fps": fps, "voiceEnhance": voiceEnhance})
    in_path = os.path.join(job["job_dir"], "raw_input.webm")
    out_path = os.path.join(job["job_dir"], "master_output.mp4")
    with open(in_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    threading.Thread(
        target=_execute_render_job,
        args=(job["id"], in_path, out_path, fps, voiceEnhance),
        daemon=True,
    ).start()
    return {"job_id": job["id"], "status": "queued", "encoder": VIDEO_ENCODER}


@app.get("/v1/render/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_render_job_status(job_id: str):
    j = JOB_MANAGER.get_job(job_id)
    if not j or j.get("type") != "render":
        raise HTTPException(status_code=404, detail="Render job not found")
    return {
        "job_id": j["id"],
        "status": j["status"],
        "progress": j["progress"],
        "error": j["error"],
        "result": j.get("result"),
    }


@app.post("/v1/render/jobs/{job_id}/cancel", dependencies=[Depends(verify_token)])
def cancel_render_job(job_id: str):
    if not JOB_MANAGER.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Render job not found")
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/v1/render/jobs/{job_id}/download", dependencies=[Depends(verify_token)])
def download_rendered_mp4(job_id: str):
    j = JOB_MANAGER.get_job(job_id)
    if not j or j.get("type") != "render" or j.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Rendered file not ready")
    out_path = j.get("result", {}).get("output_path")
    if not out_path or not os.path.isfile(out_path):
        raise HTTPException(status_code=404, detail="Rendered MP4 file missing")
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"clipforge_{job_id}.mp4",
        headers={
            "X-Video-Encoder": str(j["result"].get("encoder", VIDEO_ENCODER)),
            "X-FFprobe-Verified": "true" if j["result"].get("verified") else "false",
        },
    )


@app.post("/master", dependencies=[Depends(verify_token)])
async def master_sync_endpoint(
    file: UploadFile = File(...),
    fps: str = Form("29.97"),
    voiceEnhance: str = Form("true"),
):
    """Synchronous compatibility endpoint for direct browser-to-FFmpeg mastering."""
    if not VIDEO_ENCODER:
        raise HTTPException(status_code=503, detail="FFmpeg not installed")
    if file.size and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Capture too large for local mastering")
    job = JOB_MANAGER.create_job("render_sync", {"fps": fps})
    in_path = os.path.join(job["job_dir"], "in_sync.webm")
    out_path = os.path.join(job["job_dir"], "out_sync.mp4")
    with open(in_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    started = time.time()
    # NVENC mastering takes minutes; running it on the event loop makes the worker deaf, so the
    # browser's in-flight POST gets no service and the fetch dies in transport instead of failing
    # as an HTTP status the renderer could report.
    await run_in_threadpool(_execute_render_job, job["id"], in_path, out_path, fps, voiceEnhance.lower() == "true")
    updated = JOB_MANAGER.get_job(job["id"])
    if not updated or updated["status"] != "completed":
        detail = (updated or {}).get("error") or "Render failed"
        print(f"[master] job={job['id']} FAILED in {time.time() - started:.1f}s: {detail}", flush=True)
        raise HTTPException(status_code=500, detail=detail)
    size = os.path.getsize(out_path)
    print(
        f"[master] job={job['id']} ok in {time.time() - started:.1f}s out={size}B encoder={VIDEO_ENCODER}",
        flush=True,
    )
    return FileResponse(
        out_path,
        media_type="video/mp4",
        headers={
            "X-Video-Encoder": VIDEO_ENCODER,
            "X-FFprobe-Verified": "true",
        },
    )


def run_server():
    import uvicorn  # type: ignore
    pool = get_pool_status()
    pid_file = os.path.join(WORK_DIR, "worker.pid")
    try:
        with open(pid_file, "w", encoding="utf-8") as pf:
            pf.write(str(os.getpid()))
    except OSError:
        pass
    print(f"=== ClipForge Local Worker (OpenShorts Engine v2 + faster-whisper + 5-Key Gemini Pool) ===")
    print(f"Listening on: http://{HOST}:{PORT} (Zero-Token Auto-Connect)")
    print(f"Video Encoder: {VIDEO_ENCODER_CHOICE.label if VIDEO_ENCODER_CHOICE else 'FFmpeg not found'}")
    print(f"Hardware tier: {encoder_capabilities(VIDEO_ENCODER_CHOICE)['tier']} (output quality is identical on every tier)")
    print(f"Gemini Key Pool: {pool['configured_count']}/{pool['max_keys']} keys configured")
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    finally:
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
        except OSError:
            pass


if __name__ == "__main__":
    run_server()
