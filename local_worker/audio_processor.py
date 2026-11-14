"""FFmpeg audio extraction (16kHz mono 16-bit PCM WAV) and silence-aware long-audio chunking."""
import json
import os
import re
import shutil
import subprocess
import wave
from typing import TypedDict


class AudioChunkSpec(TypedDict):
    index: int
    path: str
    offset_sec: float
    duration_sec: float
    overlap_sec: float


def check_ffmpeg() -> dict[str, bool]:
    return {
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
    }


def probe_media_duration(media_path: str) -> float:
    """Return accurate duration in seconds via ffprobe, or wave header fallback."""
    if shutil.which("ffprobe"):
        try:
            res = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    media_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=True,
            )
            data = json.loads(res.stdout.decode("utf-8", errors="replace"))
            dur = float(data.get("format", {}).get("duration", 0.0))
            if dur > 0:
                return dur
        except Exception:
            pass
    if media_path.lower().endswith(".wav") and os.path.exists(media_path):
        try:
            with wave.open(media_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            pass
    return 0.0


def is_readable_media_file(media_path: str) -> bool:
    """Return whether FFprobe can parse the media container."""
    if not os.path.isfile(media_path) or os.path.getsize(media_path) == 0:
        return False
    if not shutil.which("ffprobe"):
        return True
    try:
        subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=format_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                media_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def extract_normalized_wav(input_media_path: str, output_wav_path: str) -> float:
    """
    Extract normalized 16 kHz, mono, 16-bit PCM WAV using FFmpeg without loading audio into browser memory.
    Returns extracted WAV duration in seconds.
    """
    if not os.path.exists(input_media_path):
        raise FileNotFoundError(f"Source file not found: {input_media_path}")
    if os.path.getsize(input_media_path) == 0:
        raise ValueError("Source media file is empty (0 bytes)")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is not installed or not on PATH")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_media_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if res.returncode != 0 or not os.path.exists(output_wav_path):
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg audio extraction failed: {err or 'invalid or silent media stream'}")

    dur = probe_media_duration(output_wav_path)
    if dur <= 0:
        raise ValueError("Extracted audio has zero duration")
    return dur


def detect_silence_midpoints(wav_path: str, noise_db: int = -35, min_silence_sec: float = 0.35) -> list[float]:
    """Use FFmpeg silencedetect filter to locate natural silence midpoints in seconds."""
    if not shutil.which("ffmpeg") or not os.path.exists(wav_path):
        return []
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", wav_path,
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        stderr_txt = res.stderr.decode("utf-8", errors="replace")
        starts: list[float] = []
        midpoints: list[float] = []
        for line in stderr_txt.splitlines():
            m_start = re.search(r"silence_start:\s*([\d.]+)", line)
            if m_start:
                starts.append(float(m_start.group(1)))
            m_end = re.search(r"silence_end:\s*([\d.]+)", line)
            if m_end and starts:
                s_val = starts.pop(0)
                e_val = float(m_end.group(1))
                midpoints.append((s_val + e_val) / 2.0)
        return sorted(midpoints)
    except Exception:
        return []


def compute_chunk_boundaries(
    total_duration: float,
    silence_points: list[float],
    target_chunk_sec: float = 360.0,
    overlap_sec: float = 1.5,
    search_window_sec: float = 25.0,
) -> list[tuple[float, float, float]]:
    """
    Compute (start_sec, end_sec, overlap_with_prev_sec) for long audio.
    Snaps chunk end points to the nearest natural silence within +/- search_window_sec
    so sentences are never split unnecessarily.
    """
    if total_duration <= 0:
        return []
    if total_duration <= target_chunk_sec * 1.15:
        return [(0.0, total_duration, 0.0)]

    chunks: list[tuple[float, float, float]] = []
    cursor = 0.0
    while cursor < total_duration:
        ideal_end = cursor + target_chunk_sec
        if ideal_end >= total_duration - 5.0:
            chunks.append((cursor, total_duration, 0.0 if cursor == 0.0 else overlap_sec))
            break

        # Find closest silence midpoint near ideal_end
        candidates = [
            s for s in silence_points
            if (ideal_end - search_window_sec) <= s <= min(total_duration - 2.0, ideal_end + search_window_sec)
            and s > cursor + 30.0
        ]
        if candidates:
            chosen_end = min(candidates, key=lambda s: abs(s - ideal_end))
        else:
            chosen_end = ideal_end

        chunks.append((cursor, chosen_end, 0.0 if cursor == 0.0 else overlap_sec))
        cursor = max(cursor + 10.0, chosen_end - overlap_sec)

    return chunks


def slice_wav_chunks(
    wav_path: str,
    out_dir: str,
    target_chunk_sec: float = 360.0,
    overlap_sec: float = 1.5,
) -> list[AudioChunkSpec]:
    """Slice a 16kHz mono PCM WAV into silence-aligned chunks on disk."""
    total_dur = probe_media_duration(wav_path)
    if total_dur <= 0:
        raise ValueError("Cannot chunk WAV with 0s duration")

    silences = detect_silence_midpoints(wav_path)
    bounds = compute_chunk_boundaries(total_dur, silences, target_chunk_sec, overlap_sec)
    if len(bounds) == 1:
        return [
            {
                "index": 0,
                "path": wav_path,
                "offset_sec": 0.0,
                "duration_sec": total_dur,
                "overlap_sec": 0.0,
            }
        ]

    specs: list[AudioChunkSpec] = []
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        total_frames = wf.getnframes()

        for idx, (s_sec, e_sec, ov_sec) in enumerate(bounds):
            start_frame = max(0, min(total_frames, int(round(s_sec * rate))))
            end_frame = max(start_frame, min(total_frames, int(round(e_sec * rate))))
            wf.setpos(start_frame)
            raw_frames = wf.readframes(end_frame - start_frame)

            chunk_path = os.path.join(out_dir, f"chunk_{idx:04d}.wav")
            with wave.open(chunk_path, "wb") as out_wf:
                out_wf.setnchannels(nchannels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(rate)
                out_wf.writeframes(raw_frames)

            specs.append(
                {
                    "index": idx,
                    "path": chunk_path,
                    "offset_sec": s_sec,
                    "duration_sec": max(0.0, e_sec - s_sec),
                    "overlap_sec": ov_sec,
                }
            )
    return specs
