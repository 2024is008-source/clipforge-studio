"""Configuration, security settings, and hardware profile definitions for the ClipForge Local Worker."""
import os
import secrets
import tempfile
from typing import TypedDict

HOST = os.environ.get("CLIPFORGE_WORKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLIPFORGE_WORKER_PORT", "8765"))

default_worker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "worker_data"))
WORK_DIR = os.environ.get("CLIPFORGE_WORK_DIR") or default_worker_dir
JOBS_DIR = os.path.join(WORK_DIR, "jobs")
MODELS_DIR = os.path.join(WORK_DIR, "models_cache")
SOURCES_DIR = os.path.join(WORK_DIR, "sources")
for d in (WORK_DIR, JOBS_DIR, MODELS_DIR, SOURCES_DIR):
    os.makedirs(d, exist_ok=True, mode=0o700)

TOKEN_FILE = os.path.join(WORK_DIR, ".pairing_token")


def load_or_create_token() -> str:
    if os.environ.get("CLIPFORGE_HELPER_TOKEN"):
        return os.environ["CLIPFORGE_HELPER_TOKEN"].strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            tok = f.read().strip()
            if tok:
                return tok
    tok = secrets.token_urlsafe(24)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    return tok


def rotate_token() -> str:
    tok = secrets.token_urlsafe(24)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    return tok


ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
}
if os.environ.get("CLIPFORGE_ALLOWED_ORIGIN"):
    for o in os.environ["CLIPFORGE_ALLOWED_ORIGIN"].split(","):
        if o.strip():
            ALLOWED_ORIGINS.add(o.strip().rstrip("/"))

MAX_UPLOAD_BYTES = int(os.environ.get("CLIPFORGE_MAX_UPLOAD_MB", "2048")) * 1024 * 1024  # 2 GB default for long videos
MAX_CACHED_SOURCES = int(os.environ.get("CLIPFORGE_MAX_CACHED_SOURCES", "2"))
JOB_TTL_SECONDS = 3600  # 1 hour automatic temp cleanup
DEFAULT_CHUNK_SECONDS = 360  # 6-minute default chunks (within 5-10 min range)
DEFAULT_OVERLAP_SECONDS = 1.5


class WhisperModelSpec(TypedDict):
    id: str
    label: str
    download_size_mb: int
    min_vram_mb: int
    description: str


WHISPER_MODELS: dict[str, WhisperModelSpec] = {
    "tiny": {
        "id": "tiny",
        "label": "Tiny",
        "download_size_mb": 75,
        "min_vram_mb": 1000,
        "description": "Fastest, lower accuracy (~75 MB).",
    },
    "base": {
        "id": "base",
        "label": "Base",
        "download_size_mb": 145,
        "min_vram_mb": 1200,
        "description": "Fast, good for simple content (~145 MB).",
    },
    "small": {
        "id": "small",
        "label": "Small (Recommended)",
        "download_size_mb": 484,
        "min_vram_mb": 2200,
        "description": "Recommended default for NVIDIA RTX 2050 (int8_float16) and modern CPUs (~484 MB).",
    },
    "medium": {
        "id": "medium",
        "label": "Medium",
        "download_size_mb": 1530,
        "min_vram_mb": 4800,
        "description": "Better quality with higher resource requirements (~1.53 GB).",
    },
    "large-v3": {
        "id": "large-v3",
        "label": "Large-v3",
        "download_size_mb": 3090,
        "min_vram_mb": 6500,
        "description": "Highest accuracy, suitable for higher-VRAM hardware (~3.09 GB).",
    },
}

SUPPORTED_LANGUAGES = [
    {"code": "auto", "name": "Auto-detect language"},
    {"code": "en", "name": "English"},
    {"code": "si", "name": "Sinhala (සිංහල)"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "hi", "name": "Hindi"},
    {"code": "ta", "name": "Tamil"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "zh", "name": "Chinese"},
    {"code": "ar", "name": "Arabic"},
    {"code": "pt", "name": "Portuguese"},
]
