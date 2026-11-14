"""Persistent background job manager & transcript cache for Local Whisper and FFmpeg Render jobs."""
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from typing import Any
from .config import JOBS_DIR, JOB_TTL_SECONDS

STATE_FILE = os.path.join(JOBS_DIR, "jobs_state.json")
CACHE_FILE = os.path.join(JOBS_DIR, "transcript_cache.json")


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.transcript_cache: dict[str, dict[str, Any]] = {}
        self._load_state()

    def _load_state(self) -> None:
        with self._lock:
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        for jid, j in loaded.items():
                            # Mark any interrupted in-progress jobs from prior crash/restart as recoverable
                            if j.get("status") in ("queued", "uploading", "analyzing", "segmenting", "extracting_audio", "transcribing", "rendering"):
                                j["status"] = "interrupted_restart"
                                j["error"] = "Worker restarted while job was running; ready to retry."
                            self.jobs[jid] = j
                except Exception:
                    self.jobs = {}
            if os.path.exists(CACHE_FILE):
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        self.transcript_cache = json.load(f)
                except Exception:
                    self.transcript_cache = {}

    def _save_state(self) -> None:
        try:
            serializable = {
                jid: {k: v for k, v in j.items() if k != "cancel_flag"}
                for jid, j in self.jobs.items()
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
        except Exception:
            pass

    def save_cache(self, cache_key: str, result_payload: dict[str, Any]) -> None:
        with self._lock:
            self.transcript_cache[cache_key] = {
                "cached_at": time.time(),
                "result": result_payload,
            }
            try:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.transcript_cache, f)
            except Exception:
                pass

    def get_cached(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self.transcript_cache.get(cache_key)
            return entry.get("result") if entry else None

    @staticmethod
    def compute_cache_key(file_sha256: str, model: str, language: str, word_timestamps: bool, vad_filter: bool) -> str:
        raw = f"{file_sha256}:{model}:{language or 'auto'}:{int(word_timestamps)}:{int(vad_filter)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def create_job(self, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
        self.cleanup_expired()
        jid = uuid.uuid4().hex[:14]
        job_dir = os.path.join(JOBS_DIR, jid)
        os.makedirs(job_dir, exist_ok=True, mode=0o700)
        job = {
            "id": jid,
            "type": job_type,
            "status": "queued",
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "params": params,
            "job_dir": job_dir,
            "logs": [f"Job {jid} created ({job_type})"],
            "result": None,
            "error": None,
            "cancelled": False,
        }
        with self._lock:
            self.jobs[jid] = job
            self._save_state()
        return job

    def update_job(self, jid: str, **patch: Any) -> None:
        with self._lock:
            j = self.jobs.get(jid)
            if not j:
                return
            log_msg = patch.pop("log", None)
            if log_msg:
                j.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] {log_msg}")
            j.update(patch)
            j["updated_at"] = time.time()
            self._save_state()

    def cancel_job(self, jid: str) -> bool:
        with self._lock:
            j = self.jobs.get(jid)
            if not j:
                return False
            j["cancelled"] = True
            if j["status"] in ("queued", "uploading", "analyzing", "segmenting", "extracting_audio", "transcribing", "rendering"):
                j["status"] = "cancelled"
            j.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] Cancellation requested.")
            self._save_state()
            return True

    def is_cancelled(self, jid: str) -> bool:
        with self._lock:
            j = self.jobs.get(jid)
            return bool(j and j.get("cancelled"))

    def get_job(self, jid: str) -> dict[str, Any] | None:
        with self._lock:
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                jid for jid, j in self.jobs.items()
                if now - j.get("updated_at", now) > JOB_TTL_SECONDS
                and j.get("status") in ("completed", "error", "cancelled", "interrupted_restart")
            ]
            for jid in expired:
                j = self.jobs.pop(jid, None)
                if j and j.get("job_dir") and os.path.isdir(j["job_dir"]):
                    shutil.rmtree(j["job_dir"], ignore_errors=True)
            if expired:
                self._save_state()


JOB_MANAGER = JobManager()
