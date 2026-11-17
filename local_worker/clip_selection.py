"""Transcript windowing, global viral score ranking, overlap deduplication, and word-boundary snapping.

Ported from OpenShorts `clip_selection.py` (`https://github.com/mutonby/openshorts/blob/main/clip_selection.py`):
- `build_transcript_windows`: splits long transcripts into 90s windows with 30s overlap snapped to segment boundaries.
- `score_batches`: divides candidate clips into balanced LLM scoring batches.
- `trim_to_best`: globally ranks candidate shorts by `predicted_score` before restoring chronological order.
- `dedupe_overlapping`: suppresses candidate clips that overlap an higher-scoring clip by `>= 25%` of the shorter clip.
- `snap_clip_to_words`: snaps clip start/end timestamps to clean sentence-opening and sentence-closing Whisper word boundaries.
"""

from __future__ import annotations

import math
import re
from typing import Any

WINDOW_SECONDS = 90.0
OVERLAP_SECONDS = 30.0
MIN_TAIL_SECONDS = 25.0

SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*$")
CLAUSE_END_RE = re.compile(r"[,;:—–-][\"'”’)]*$")
CONTINUATION_START_RE = re.compile(
    r"^(?:and|but|or|so|because|which|that|who|whom|whose|where|when|while|although|though|if|unless|until|since)\b",
    re.IGNORECASE,
)


def build_transcript_windows(
    segments: list[dict[str, Any]],
    window_seconds: float = WINDOW_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[list[dict[str, Any]]]:
    """Slice transcript segments into overlapping windows (`90s` window, `30s` overlap) without splitting segments."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: float(s.get("start", s.get("s", 0.0))))
    total_start = float(ordered[0].get("start", ordered[0].get("s", 0.0)))
    total_end = float(ordered[-1].get("end", ordered[-1].get("e", total_start)))
    if total_end - total_start <= window_seconds:
        return [ordered]

    step = max(10.0, window_seconds - overlap_seconds)
    windows: list[list[dict[str, Any]]] = []
    cursor = total_start
    while cursor < total_end:
        win_end = cursor + window_seconds
        bucket = [
            seg
            for seg in ordered
            if float(seg.get("end", seg.get("e", 0.0))) > cursor
            and float(seg.get("start", seg.get("s", 0.0))) < win_end
        ]
        if bucket:
            if (
                windows
                and float(bucket[-1].get("end", bucket[-1].get("e", 0.0)))
                - float(bucket[0].get("start", bucket[0].get("s", 0.0)))
                < MIN_TAIL_SECONDS
            ):
                # Merge a short trailing tail into the previous window
                seen_ids = {id(s) for s in windows[-1]}
                for seg in bucket:
                    if id(seg) not in seen_ids:
                        windows[-1].append(seg)
                break
            windows.append(bucket)
        if win_end >= total_end:
            break
        cursor += step
    return windows


def score_batches(count: int, max_batch_size: int = 3) -> list[tuple[int, int]]:
    """Split `count` candidates into balanced `(start_idx, end_idx)` batches for comparative scoring."""
    if count <= 0:
        return []
    max_batch_size = max(1, int(max_batch_size))
    num_batches = math.ceil(count / max_batch_size)
    base = count // num_batches
    rem = count % num_batches
    batches: list[tuple[int, int]] = []
    cursor = 0
    for i in range(num_batches):
        size = base + (1 if i < rem else 0)
        batches.append((cursor, cursor + size))
        cursor += size
    return batches


def _score_of(clip: dict[str, Any]) -> float:
    for key in ("predicted_score", "viral_score", "score"):
        val = clip.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _span_of(clip: dict[str, Any]) -> tuple[float, float]:
    start = float(clip.get("start", clip.get("start_seconds", clip.get("s", 0.0))))
    end = float(clip.get("end", clip.get("end_seconds", clip.get("e", start))))
    return start, max(start, end)


def dedupe_overlapping(
    shorts: list[dict[str, Any]],
    ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Drop candidate shorts whose overlap with a higher-scoring short is `>= ratio` of the shorter clip."""
    if len(shorts) <= 1:
        return list(shorts)
    ranked = sorted(
        enumerate(shorts),
        key=lambda item: (-_score_of(item[1]), _span_of(item[1])[0], item[0]),
    )
    kept_with_idx: list[tuple[int, dict[str, Any]]] = []
    for idx, cand in ranked:
        s0, e0 = _span_of(cand)
        dur0 = max(1e-3, e0 - s0)
        duplicate = False
        for _, existing in kept_with_idx:
            s1, e1 = _span_of(existing)
            dur1 = max(1e-3, e1 - s1)
            ov = max(0.0, min(e0, e1) - max(s0, s1))
            if ov / min(dur0, dur1) >= ratio:
                duplicate = True
                break
        if not duplicate:
            kept_with_idx.append((idx, cand))
    kept_with_idx.sort(key=lambda item: (_span_of(item[1])[0], item[0]))
    return [c for _, c in kept_with_idx]


def trim_to_best(
    shorts: list[dict[str, Any]],
    max_clips: int,
    overlap_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Deduplicate overlapping candidates, keep the top `max_clips` by score, and restore chronological order."""
    if max_clips <= 0 or not shorts:
        return []
    deduped = dedupe_overlapping(shorts, ratio=overlap_ratio)
    ranked = sorted(
        enumerate(deduped),
        key=lambda item: (-_score_of(item[1]), _span_of(item[1])[0], item[0]),
    )[:max_clips]
    ranked.sort(key=lambda item: (_span_of(item[1])[0], item[0]))
    return [c for _, c in ranked]


def _word_tuple(w: Any) -> tuple[str, float, float]:
    if isinstance(w, dict):
        text = str(w.get("w", w.get("word", w.get("text", "")))).strip()
        start = float(w.get("s", w.get("start", 0.0)))
        end = float(w.get("e", w.get("end", start)))
        return text, start, max(start, end)
    text = str(getattr(w, "w", getattr(w, "word", getattr(w, "text", "")))).strip()
    start = float(getattr(w, "s", getattr(w, "start", 0.0)))
    end = float(getattr(w, "e", getattr(w, "end", start)))
    return text, start, max(start, end)


def snap_clip_to_words(
    start: float,
    end: float,
    words: list[Any],
    min_duration: float = 5.0,
    max_duration: float = 90.0,
    search_window: float = 1.5,
    max_lead: float = 0.35,
    max_tail: float = 0.45,
) -> tuple[float, float]:
    """Snap `[start, end]` to clean Whisper word boundaries (`clip_selection.py` algorithm).

    - Finds the best sentence-opening word within `±search_window` of `start` (preferring a word
      immediately following terminal punctuation `[.!?]` or a `>= 0.30s` pause, and avoiding
      conjunction continuation words).
    - Finds the best sentence-closing word within `±search_window` of `end` (preferring terminal
      punctuation `[.!?]` or a pause before the next word).
    - Adds a natural `<= max_lead` pre-roll and `<= max_tail` post-roll clamped so adjacent words outside
      the clip are never bled into the audio track.
    """
    if not words or end <= start:
        return round(start, 3), round(end, 3)

    parsed = [_word_tuple(w) for w in words]
    parsed = [p for p in parsed if p[0]]
    if not parsed:
        return round(start, 3), round(end, 3)

    n = len(parsed)

    # 1. Pick start word index near `start`
    start_candidates: list[tuple[float, int]] = []
    for i, (text, ws, _we) in enumerate(parsed):
        dist = abs(ws - start)
        if dist <= search_window:
            prev_text = parsed[i - 1][0] if i > 0 else "."
            gap_before = (ws - parsed[i - 1][2]) if i > 0 else 1.0
            bonus = 0.0
            if i == 0 or SENTENCE_END_RE.search(prev_text):
                bonus -= 0.65
            elif gap_before >= 0.30:
                bonus -= 0.40
            elif CLAUSE_END_RE.search(prev_text):
                bonus -= 0.20
            if CONTINUATION_START_RE.match(text):
                bonus += 0.35
            start_candidates.append((dist + bonus, i))

    if start_candidates:
        start_idx = min(start_candidates, key=lambda item: item[0])[1]
    else:
        start_idx = min(range(n), key=lambda i: abs(parsed[i][1] - start))

    # 2. Pick end word index near `end`
    end_candidates: list[tuple[float, int]] = []
    for i, (text, _ws, we) in enumerate(parsed):
        if i < start_idx:
            continue
        dist = abs(we - end)
        if dist <= search_window:
            gap_after = (parsed[i + 1][1] - we) if i + 1 < n else 1.0
            bonus = 0.0
            if i == n - 1 or SENTENCE_END_RE.search(text):
                bonus -= 0.65
            elif gap_after >= 0.30:
                bonus -= 0.40
            elif CLAUSE_END_RE.search(text):
                bonus -= 0.20
            end_candidates.append((dist + bonus, i))

    if end_candidates:
        end_idx = min(end_candidates, key=lambda item: item[0])[1]
    else:
        end_idx = min(range(start_idx, n), key=lambda i: abs(parsed[i][2] - end))

    # Enforce min_duration and max_duration while staying on word boundaries
    while end_idx + 1 < n and (parsed[end_idx][2] - parsed[start_idx][1]) < min_duration:
        end_idx += 1
    while end_idx > start_idx and (parsed[end_idx][2] - parsed[start_idx][1]) > max_duration:
        end_idx -= 1

    first_s = parsed[start_idx][1]
    last_e = parsed[end_idx][2]

    # Compute safe pre-roll and post-roll that never cross into neighboring words
    if start_idx > 0:
        prev_e = parsed[start_idx - 1][2]
        lead = min(max_lead, max(0.0, (first_s - prev_e) * 0.5))
    else:
        lead = min(max_lead, first_s)

    if end_idx + 1 < n:
        next_s = parsed[end_idx + 1][1]
        tail = min(max_tail, max(0.0, (next_s - last_e) * 0.5))
    else:
        tail = max_tail

    snapped_start = max(0.0, first_s - lead)
    snapped_end = max(snapped_start + 0.1, last_e + tail)
    return round(snapped_start, 3), round(snapped_end, 3)
