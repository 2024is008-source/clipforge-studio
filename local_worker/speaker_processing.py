"""
Optional local speaker diarization & manual speaker assignment module.
Separates Whisper transcription from speaker diarization so transcription never fails when diarization is unavailable.
"""
import importlib.util
from typing import TypedDict
from .transcript_validator import ClipForgeWord


class DiarizationStatus(TypedDict):
    available: bool
    backend: str | None
    reason: str


def check_diarization_available() -> DiarizationStatus:
    """Check if an optional local diarization library (e.g. pyannote.audio) is installed and configured."""
    try:
        spec = importlib.util.find_spec("pyannote.audio")
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return {
            "available": False,
            "backend": None,
            "reason": "pyannote.audio not installed (optional). speaker_id defaults to null and supports manual speaker naming.",
        }
    return {
        "available": True,
        "backend": "pyannote.audio",
        "reason": "Local pyannote.audio backend detected.",
    }


def assign_speakers_to_words(
    words: list[ClipForgeWord],
    diarization_segments: list[dict] | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> list[ClipForgeWord]:
    """
    Assign speaker_id to words if local diarization segments (`[{start, end, speaker}]`) are provided,
    otherwise keep `speaker_id` as None (or apply user-defined manual speaker aliases).
    """
    labels = speaker_labels or {}
    out: list[ClipForgeWord] = []
    for w in words:
        spk: str | None = w.get("speaker_id")
        if diarization_segments:
            mid = (w["s"] + w["e"]) / 2.0
            for seg in diarization_segments:
                if float(seg.get("start", 0.0)) <= mid <= float(seg.get("end", 0.0)):
                    spk = str(seg.get("speaker", "Speaker 1"))
                    break
        if spk and spk in labels:
            spk = labels[spk]
        copy_w: ClipForgeWord = dict(w)  # type: ignore[assignment]
        copy_w["speaker_id"] = spk
        out.append(copy_w)
    return out
