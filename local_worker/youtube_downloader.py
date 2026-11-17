"""Resilient YouTube video downloader for ClipForge Studio (`local_worker`).

Handles:
- Canonical URL normalization for `youtu.be/<id>?si=...`, `youtube.com/watch?v=<id>`,
  `youtube.com/shorts/<id>`, `youtube.com/live/<id>`, `youtube.com/embed/<id>`, `m.youtube.com`,
  and `music.youtube.com`.
- Automatic `yt-dlp` installation/upgrade if missing from the Python environment.
- Real-time download percentage tracking via both native `yt_dlp.YoutubeDL` `progress_hooks`
  (covering `downloaded_bytes / total_bytes`, `total_bytes_estimate`, and DASH/HLS
  `fragment_index / fragment_count`) and CLI `--progress --no-quiet --newline` output.
- Multi-client fallback cascade (`default,-android_sdkless` -> `tv,web_safari,ios` -> browser cookies)
  to bypass YouTube HTTP 403 Forbidden / `android_sdkless` / bot-check failures.
- H.264 + AAC `.mp4` stream selection and FFmpeg remux verification so downloaded videos play
  natively in both browser `<video>` elements and the local FFmpeg render pipeline.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import WORK_DIR

DOWNLOADS_DIR = os.path.join(WORK_DIR, "youtube_downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True, mode=0o700)

ALLOWED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS_RE = re.compile(r"(?:\[download\]|progress:)\s*([\d.]+)%")
FRAG_PROGRESS_RE = re.compile(r"frag\s+(\d+)/(\d+)", re.IGNORECASE)

JOB_TTL_SECONDS = 60 * 60  # 1 hour retention
MAX_CONCURRENT_DOWNLOADS = 2
DOWNLOAD_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)

YOUTUBE_JOBS: dict[str, dict[str, Any]] = {}
YOUTUBE_JOBS_LOCK = threading.Lock()
_YTDLP_UPGRADED_ONCE = False


def normalize_youtube_url(raw_url: str) -> tuple[bool, str, str | None]:
    """Validate and normalize any YouTube URL (including `youtu.be/<id>?si=...`) to a canonical watch URL.

    Returns `(is_valid, canonical_url_or_error, video_id)`.
    """
    text = (raw_url or "").strip()
    if not text:
        return False, "Please enter a YouTube URL", None

    if not re.match(r"^https?://", text, re.IGNORECASE):
        if text.startswith(("youtu.be/", "www.youtube.com/", "youtube.com/", "m.youtube.com/")):
            text = "https://" + text
        else:
            return False, "Only http:// and https:// YouTube URLs are supported", None

    try:
        parsed = urlparse(text)
    except Exception as exc:
        return False, f"Invalid URL: {exc}", None

    if parsed.scheme.lower() not in ("http", "https"):
        return False, "Only http:// and https:// YouTube URLs are supported", None

    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in ALLOWED_YOUTUBE_HOSTS:
        return (
            False,
            f"Unsupported host '{host}'. Please provide a youtube.com or youtu.be link.",
            None,
        )

    video_id: str | None = None
    path_parts = [p for p in (parsed.path or "").split("/") if p]

    if host in ("youtu.be", "www.youtu.be"):
        if path_parts:
            candidate = path_parts[0]
            if VIDEO_ID_RE.match(candidate):
                video_id = candidate
    else:
        qs = parse_qs(parsed.query or "")
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0].strip()
            if VIDEO_ID_RE.match(candidate):
                video_id = candidate
        if not video_id and len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live", "v"):
            candidate = path_parts[1].strip()
            if VIDEO_ID_RE.match(candidate):
                video_id = candidate

    if not video_id:
        return False, "Could not extract a valid 11-character YouTube video ID from the URL", None

    canonical = f"https://www.youtube.com/watch?v={video_id}"
    return True, canonical, video_id


def parse_progress_line(line: str, current_progress: float = 0.0, stream_stage: int = 0) -> float | None:
    """Extract normalized `0.0..0.99` progress from a `yt-dlp` output line.

    Handles standard `[download]  42.5%`, custom `progress:42.5%|downloaded|total`, and
    DASH/HLS fragment counts (`frag 12/40`). Scales video stream (stage 0) across `0.00..0.88`
    and audio stream (stage >= 1) across `0.88..0.98` so progress never resets to 0% mid-download.
    """
    clean = ANSI_ESCAPE_RE.sub("", line or "").strip()
    if not clean:
        return None

    raw_ratio: float | None = None
    if "progress:" in clean:
        payload = clean.split("progress:", 1)[-1]
        parts = [p.strip() for p in payload.split("|")]
        if parts:
            m_pct = re.search(r"([\d.]+)%", parts[0])
            if m_pct:
                try:
                    raw_ratio = float(m_pct.group(1)) / 100.0
                except ValueError:
                    pass
            if raw_ratio is None and len(parts) >= 3:
                try:
                    dl_b = float(parts[1])
                    tot_b = float(parts[2]) if parts[2] not in ("NA", "None", "", "0") else float(parts[3] if len(parts) > 3 and parts[3] not in ("NA", "None", "") else 0)
                    if tot_b > 0 and dl_b >= 0:
                        raw_ratio = dl_b / tot_b
                except (ValueError, IndexError):
                    pass

    if raw_ratio is None:
        m = PROGRESS_RE.search(clean)
        if m:
            try:
                raw_ratio = float(m.group(1)) / 100.0
            except ValueError:
                pass

    if raw_ratio is None:
        mf = FRAG_PROGRESS_RE.search(clean)
        if mf:
            try:
                f_idx = float(mf.group(1))
                f_tot = float(mf.group(2))
                if f_tot > 0:
                    raw_ratio = f_idx / f_tot
            except ValueError:
                pass

    if raw_ratio is None:
        return None

    clamped = max(0.0, min(1.0, raw_ratio))
    if stream_stage <= 0:
        mapped = clamped * 0.88
    else:
        mapped = 0.88 + clamped * 0.10
    return round(min(0.99, max(current_progress, mapped)), 4)


def compute_hook_progress(d: dict[str, Any], current_progress: float = 0.0, stream_stage: int = 0) -> float:
    """Compute monotonic `0.0..0.99` progress from a `yt_dlp` Python `progress_hooks` dict."""
    status = d.get("status")
    if status == "finished":
        target = 0.88 if stream_stage <= 0 else 0.98
        return round(max(current_progress, target), 4)
    if status != "downloading":
        return current_progress

    dl = float(d.get("downloaded_bytes") or 0.0)
    tot = float(d.get("total_bytes") or d.get("total_bytes_estimate") or 0.0)
    raw_ratio: float | None = None
    if tot > 0 and dl >= 0:
        raw_ratio = dl / tot
    else:
        frag_idx = d.get("fragment_index")
        frag_cnt = d.get("fragment_count")
        if frag_idx is not None and frag_cnt:
            try:
                raw_ratio = float(frag_idx) / max(1.0, float(frag_cnt))
            except (TypeError, ValueError):
                pass
    if raw_ratio is None:
        pct_str = ANSI_ESCAPE_RE.sub("", str(d.get("_percent_str") or ""))
        m = re.search(r"([\d.]+)%", pct_str)
        if m:
            try:
                raw_ratio = float(m.group(1)) / 100.0
            except ValueError:
                pass
    if raw_ratio is None:
        return current_progress

    clamped = max(0.0, min(1.0, raw_ratio))
    mapped = clamped * 0.88 if stream_stage <= 0 else 0.88 + clamped * 0.10
    return round(min(0.99, max(current_progress, mapped)), 4)


def cleanup_expired_downloads() -> None:
    """Remove expired download jobs and temporary files older than JOB_TTL_SECONDS."""
    now = time.time()
    with YOUTUBE_JOBS_LOCK:
        expired = [
            jid
            for jid, job in YOUTUBE_JOBS.items()
            if now - float(job.get("created_at", now)) > JOB_TTL_SECONDS
        ]
        for jid in expired:
            job = YOUTUBE_JOBS.pop(jid, None)
            if job and job.get("file") and os.path.exists(job["file"]):
                try:
                    os.remove(job["file"])
                except OSError:
                    pass


def ensure_yt_dlp_installed(force_upgrade: bool = False) -> bool:
    """Ensure `yt_dlp` Python module or `yt-dlp` CLI is available; auto-install via pip if missing."""
    global _YTDLP_UPGRADED_ONCE
    if not force_upgrade:
        try:
            import yt_dlp  # noqa: F401
            return True
        except ImportError:
            if shutil.which("yt-dlp"):
                return True
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        _YTDLP_UPGRADED_ONCE = True
    except Exception:
        pass
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return bool(shutil.which("yt-dlp"))


def _base_cmd() -> list[str]:
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        binary = shutil.which("yt-dlp")
        if binary:
            return [binary]
        return [sys.executable, "-m", "yt_dlp"]


def _resolve_downloaded_file(jid: str, reported_path: str | None) -> str | None:
    """Locate the finished video file for `jid` and ensure it is inside `DOWNLOADS_DIR`."""
    abs_dir = os.path.abspath(DOWNLOADS_DIR)
    if reported_path:
        cand = os.path.abspath(reported_path)
        if cand.startswith(abs_dir) and os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand

    exact_mp4 = os.path.join(abs_dir, f"{jid}.mp4")
    if os.path.isfile(exact_mp4) and os.path.getsize(exact_mp4) > 0:
        return exact_mp4

    matches = sorted(
        glob.glob(os.path.join(abs_dir, f"{jid}.*")),
        key=lambda p: os.path.getsize(p) if os.path.isfile(p) else 0,
        reverse=True,
    )
    for m in matches:
        if m.endswith((".part", ".ytdl", ".temp")):
            continue
        if os.path.isfile(m) and os.path.getsize(m) > 0:
            if not m.endswith(".mp4") and shutil.which("ffmpeg"):
                remuxed = os.path.join(abs_dir, f"{jid}.mp4")
                res = subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", m, "-c", "copy", "-movflags", "+faststart", remuxed,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if res.returncode == 0 and os.path.isfile(remuxed) and os.path.getsize(remuxed) > 0:
                    try:
                        os.remove(m)
                    except OSError:
                        pass
                    return remuxed
            return m
    return None


def _run_single_ytdlp_attempt(
    jid: str,
    url: str,
    quality: int,
    extra_args: list[str],
    job: dict[str, Any],
) -> tuple[bool, str | None, str]:
    """Run one `yt-dlp` attempt with `--progress --no-quiet` so `--print` never silences progress."""
    out_tmpl = os.path.join(DOWNLOADS_DIR, f"{jid}.%(ext)s")
    fmt = (
        f"bv*[height<={quality}][vcodec^=avc1]+ba[acodec^=mp4a]/"
        f"bv*[height<={quality}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<={quality}][ext=mp4]/"
        f"bv*[height<={quality}]+ba/"
        f"b[height<={quality}]/b"
    )
    # CRITICAL: `--print` in yt-dlp implicitly enables `--quiet` unless `--progress --no-quiet`
    # are explicitly passed! We pass `--progress`, `--no-quiet`, `--newline`, and a structured
    # `--progress-template` so percentage updates stream on every chunk.
    cmd = [
        *_base_cmd(),
        "--no-playlist",
        "--no-warnings",
        "--progress",
        "--no-quiet",
        "--newline",
        "--progress-template",
        "download:progress:%(progress._percent_str)s|%(progress.downloaded_bytes)s|%(progress.total_bytes)s|%(progress.total_bytes_estimate)s",
        "--force-ipv4",
        "--geo-bypass",
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "--print",
        "before_dl:title:%(title)s",
        "--print",
        "after_move:filepath:%(filepath)s",
        *extra_args,
        "-o",
        out_tmpl,
        url,
    ]
    reported_file: str | None = None
    last_err = ""
    stream_stage = 0
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[download] Destination:"):
                if float(job.get("progress", 0.0)) >= 0.50:
                    stream_stage = 1
            elif line.startswith("[Merger]"):
                job["progress"] = max(float(job.get("progress", 0.0)), 0.99)
            else:
                new_p = parse_progress_line(line, float(job.get("progress", 0.0)), stream_stage)
                if new_p is not None:
                    job["progress"] = new_p
            if line.startswith("title:"):
                title_val = line[len("title:") :].strip()
                if title_val and title_val != "NA":
                    job["title"] = title_val
            elif line.startswith("filepath:"):
                fp = line[len("filepath:") :].strip()
                if fp and os.path.isfile(fp):
                    reported_file = fp
            elif os.path.isfile(line) and os.path.abspath(line).startswith(os.path.abspath(DOWNLOADS_DIR)):
                reported_file = line
            elif "ERROR:" in line:
                last_err = line.split("ERROR:", 1)[-1].strip()
            job["log"] = line[-240:]
        proc.wait(timeout=600)
        resolved = _resolve_downloaded_file(jid, reported_file)
        if proc.returncode == 0 and resolved:
            return True, resolved, ""
        return False, None, last_err or job.get("log") or f"yt-dlp exited with code {proc.returncode}"
    except Exception as exc:
        return False, None, str(exc)


def _execute_youtube_download(jid: str, canonical_url: str, quality: int) -> None:
    if not DOWNLOAD_SEMAPHORE.acquire(timeout=10):
        with YOUTUBE_JOBS_LOCK:
            if jid in YOUTUBE_JOBS:
                YOUTUBE_JOBS[jid]["status"] = "error"
                YOUTUBE_JOBS[jid]["error"] = "Too many concurrent downloads in progress; please wait a moment."
        return
    try:
        job = YOUTUBE_JOBS.get(jid)
        if not job:
            return

        if not ensure_yt_dlp_installed(force_upgrade=False):
            job["status"] = "error"
            job["error"] = "yt-dlp is not installed. Run: pip install -U yt-dlp"
            return

        strategies: list[list[str]] = [
            ["--extractor-args", "youtube:player_client=default,-android_sdkless"],
            ["--extractor-args", "youtube:player_client=tv,web_safari,ios"],
            ["--extractor-args", "youtube:player_client=ios,mweb"],
        ]
        for browser in ("edge", "chrome", "firefox", "brave"):
            strategies.append(
                [
                    "--cookies-from-browser",
                    browser,
                    "--extractor-args",
                    "youtube:player_client=default,-android_sdkless",
                ]
            )

        last_error = "Download failed"
        for idx, extra_args in enumerate(strategies):
            for part_file in glob.glob(os.path.join(DOWNLOADS_DIR, f"{jid}.*")):
                try:
                    os.remove(part_file)
                except OSError:
                    pass

            ok, filepath, err_msg = _run_single_ytdlp_attempt(
                jid, canonical_url, quality, extra_args, job
            )
            if ok and filepath:
                job["file"] = filepath
                job["progress"] = 1.0
                job["status"] = "done"
                return
            if err_msg and "cookie" not in err_msg.lower() and "keyring" not in err_msg.lower():
                last_error = err_msg

            if idx == 1 and not _YTDLP_UPGRADED_ONCE:
                ensure_yt_dlp_installed(force_upgrade=True)
                ok_retry, fp_retry, err_retry = _run_single_ytdlp_attempt(
                    jid,
                    canonical_url,
                    quality,
                    ["--extractor-args", "youtube:player_client=default,-android_sdkless"],
                    job,
                )
                if ok_retry and fp_retry:
                    job["file"] = fp_retry
                    job["progress"] = 1.0
                    job["status"] = "done"
                    return
                if err_retry:
                    last_error = err_retry

        job["status"] = "error"
        job["error"] = f"YouTube download failed: {last_error}"
    except Exception as exc:
        with YOUTUBE_JOBS_LOCK:
            if jid in YOUTUBE_JOBS:
                YOUTUBE_JOBS[jid]["status"] = "error"
                YOUTUBE_JOBS[jid]["error"] = f"Download error: {exc}"
    finally:
        DOWNLOAD_SEMAPHORE.release()


def start_youtube_download(url: str, quality: int = 1080, rights_confirmed: bool = True) -> dict[str, Any]:
    """Validate URL, enqueue a background YouTube download job, and return `{'id': jid, ...}`."""
    cleanup_expired_downloads()
    if not rights_confirmed:
        return {"error": "Please confirm you have rights/authorization to import this video.", "status_code": 400}

    ok, canonical_or_err, video_id = normalize_youtube_url(url)
    if not ok:
        return {"error": canonical_or_err, "status_code": 400}

    clamped_quality = min(2160, max(360, int(quality or 1080)))
    jid = uuid.uuid4().hex[:12]

    with YOUTUBE_JOBS_LOCK:
        active_count = sum(1 for j in YOUTUBE_JOBS.values() if j.get("status") == "downloading")
        if active_count >= MAX_CONCURRENT_DOWNLOADS:
            return {"error": "Maximum concurrent YouTube downloads reached; please wait.", "status_code": 429}
        YOUTUBE_JOBS[jid] = {
            "id": jid,
            "video_id": video_id,
            "url": canonical_or_err,
            "quality": clamped_quality,
            "status": "downloading",
            "progress": 0.0,
            "title": "",
            "created_at": time.time(),
        }

    threading.Thread(
        target=_execute_youtube_download,
        args=(jid, canonical_or_err, clamped_quality),
        daemon=True,
    ).start()
    return {"id": jid, "video_id": video_id, "canonical_url": canonical_or_err, "status_code": 200}


def get_youtube_download_status(jid: str) -> dict[str, Any] | None:
    cleanup_expired_downloads()
    clean_id = re.sub(r"[^a-f0-9]", "", (jid or "").lower())
    with YOUTUBE_JOBS_LOCK:
        job = YOUTUBE_JOBS.get(clean_id)
        if not job:
            return None
        return {k: v for k, v in job.items() if k not in ("file", "log")}


def get_youtube_download_file(jid: str) -> str | None:
    cleanup_expired_downloads()
    clean_id = re.sub(r"[^a-f0-9]", "", (jid or "").lower())
    with YOUTUBE_JOBS_LOCK:
        job = YOUTUBE_JOBS.get(clean_id)
        if not job or job.get("status") != "done" or not job.get("file"):
            return None
        fpath = os.path.abspath(str(job["file"]))
        if not fpath.startswith(os.path.abspath(DOWNLOADS_DIR)) or not os.path.isfile(fpath):
            return None
        return fpath
