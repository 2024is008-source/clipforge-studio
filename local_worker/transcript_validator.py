"""
Validates faster-whisper word timestamps, merges sub-word continuation tokens (contractions,
comma/decimal numbers, currency prefixes, and size tokens), applies Contextual ASR homophone repair,
removes chunk-overlap duplicates, and produces ClipForge-compatible Word objects.
"""
from __future__ import annotations

import re
from typing import TypedDict


class ClipForgeWord(TypedDict):
    w: str
    s: float
    e: float
    speaker_id: str | None
    source: str
    timing_warning: bool


class ValidationReport(TypedDict):
    words: list[ClipForgeWord]
    total_words: int
    missing_timing_count: int
    duplicates_removed: int
    valid: bool
    warnings: list[str]


_APOSTROPHE_CONTINUATION_RE = re.compile(
    r"^['’](all|s|t|re|ve|ll|d|m|em|cause|clock|tis|twas)\b", re.IGNORECASE
)
_NUMBER_COMMA_DECIMAL_CONT_RE = re.compile(r"^[,.\u066B]\d+")
_CURRENCY_ONLY_RE = re.compile(r"^[$€£¥₹]$")
_LEADING_DIGITS_RE = re.compile(r"^\d[\d,.]*%?[,!?.]*$")
_TRAILING_NUMBER_PREFIX_RE = re.compile(r"^\$?\d+[,.]$")
_SHORT_INT_RE = re.compile(r"^\$?\d{1,3}$")
_COMMA_THOUSANDS_RE = re.compile(r"^,\d{3}\b")
_SIZE_SUFFIX_RE = re.compile(r"^-?(xl|xxl|xxxl|lbs|kg|km|mph|fps)%?[,!?.]*$", re.IGNORECASE)


def normalize_token(word_str: str) -> str:
    return "".join(ch.lower() for ch in word_str.strip() if ch.isalnum())


def should_merge_continuation(prev_text: str, curr_text: str, gap_sec: float = 0.0) -> bool:
    """
    Determine whether `curr_text` is a sub-word continuation of `prev_text` that Whisper split:
    - Contractions starting with an apostrophe: "y" + "'all" -> "y'all", "don" + "'t" -> "don't"
    - Previous token ending with an apostrophe: "y'" + "all" -> "y'all"
    - Comma/decimal number continuations: "6" + ",614" -> "6,614", "2" + ".4" -> "2.4"
    - Standalone currency symbol before digits: "$" + "6,614" -> "$6,614"
    - Number + clothing/weight size token: "4" + "XL" -> "4XL"
    """
    if gap_sec > 0.45:
        return False
    p = prev_text.strip()
    c = curr_text.strip()
    if not p or not c:
        return False
    if _APOSTROPHE_CONTINUATION_RE.match(c):
        return True
    if p.endswith(("'", "’")) and re.match(r"^[a-zA-Z]+[,!?.]*$", c):
        return True
    if _NUMBER_COMMA_DECIMAL_CONT_RE.match(c) and re.search(r"\d$", p):
        return True
    if _CURRENCY_ONLY_RE.match(p) and _LEADING_DIGITS_RE.match(c):
        return True
    if _TRAILING_NUMBER_PREFIX_RE.match(p) and _LEADING_DIGITS_RE.match(c):
        return True
    if _SHORT_INT_RE.match(p) and _COMMA_THOUSANDS_RE.match(c):
        return True
    if re.match(r"^\d{1,2}$", p) and re.match(r"^-?(xl|xxl|xxxl)[,!?.]*$", c, re.IGNORECASE):
        return True
    return False


# Multi-word contextual ASR homophone repairs observed in challenge & studio panel videos
_HOMOPHONE_PHRASE_FIXES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("105", "looks"), ("105", "loaves")),
    (("cans", "the", "pair"), ("cans", "of", "pears")),
    (("cans", "the", "pears"), ("cans", "of", "pears")),
    (("new", "hour"), ("new", "aisle")),
    (("jars", "of", "season"), ("jars", "of", "seasoning")),
    (("good", "fucking", "god"), ("good", "fucking", "point")),
    (("good", "f***ing", "god"), ("good", "f***ing", "point")),
]


def repair_contextual_homophones(words: list[ClipForgeWord]) -> list[ClipForgeWord]:
    """Repair multi-token ASR homophone errors in-place while preserving exact word timestamps."""
    n = len(words)
    if n < 2:
        return words
    for i in range(n):
        for pattern, replacement in _HOMOPHONE_PHRASE_FIXES:
            k = len(pattern)
            if i + k > n:
                continue
            window = [normalize_token(words[i + j]["w"]) for j in range(k)]
            if tuple(window) == tuple(normalize_token(p) for p in pattern):
                for j, rep in enumerate(replacement):
                    orig = words[i + j]["w"]
                    # Preserve trailing punctuation from original token
                    punct_match = re.search(r"([,!?.]+)$", orig)
                    suffix = punct_match.group(1) if punct_match else ""
                    if orig.isupper():
                        rep_cased = rep.upper()
                    elif orig[:1].isupper():
                        rep_cased = rep.capitalize()
                    else:
                        rep_cased = rep
                    words[i + j]["w"] = f"{rep_cased}{suffix}"
    return words


def merge_and_validate_chunk_words(
    chunk_word_lists: list[tuple[float, float, list[dict]]],
    source_duration: float,
) -> ValidationReport:
    """
    Merge words across chunks:
    - Each item in `chunk_word_lists` is `(chunk_offset_sec, overlap_sec, raw_words)`.
    - Restores timestamps to original source timeline (`chunk_offset_sec + w.start`).
    - Merges sub-word continuations (apostrophe contractions like "y'all", comma numbers like "6,614",
      decimals like "2.4", and currency symbols like "$6,614") so numbers and contractions never split.
    - Deduplicates words in the overlap zone between chunk N-1 and chunk N.
    - Enforces: `s >= 0`, `e > s`, `e <= source_duration`, preserved word order, and flags any missing timings.
    """
    merged: list[ClipForgeWord] = []
    missing_timing_count = 0
    duplicates_removed = 0
    warnings: list[str] = []
    max_dur = max(0.1, float(source_duration))

    for chunk_idx, (offset_sec, overlap_sec, raw_words) in enumerate(chunk_word_lists):
        overlap_boundary = offset_sec + overlap_sec if chunk_idx > 0 else -1.0

        for rw in raw_words:
            text = str(rw.get("word") if "word" in rw else rw.get("w", "")).strip()
            if not text:
                continue

            raw_s = rw.get("start") if "start" in rw else rw.get("s")
            raw_e = rw.get("end") if "end" in rw else rw.get("e")
            timing_missing = raw_s is None or raw_e is None

            if timing_missing:
                missing_timing_count += 1
                prev_end = merged[-1]["e"] if merged else max(0.0, offset_sec)
                s_val = min(max_dur - 0.05, max(0.0, prev_end))
                e_val = min(max_dur, s_val + 0.16)
            else:
                s_val = float(raw_s) + offset_sec
                e_val = float(raw_e) + offset_sec

            s_val = round(max(0.0, min(max_dur - 0.02, s_val)), 3)
            if e_val <= s_val:
                e_val = round(min(max_dur, s_val + 0.08), 3)
            else:
                e_val = round(min(max_dur, e_val), 3)

            # Check if this token is a sub-word continuation of the immediately preceding word
            if merged and should_merge_continuation(
                merged[-1]["w"], text, max(0.0, s_val - merged[-1]["e"])
            ):
                clean_c = text.lstrip("-") if re.match(r"^\d", merged[-1]["w"]) and text.lower().startswith("-xl") else text
                merged[-1]["w"] = f"{merged[-1]['w']}{clean_c}"
                merged[-1]["e"] = round(max(merged[-1]["e"], e_val), 3)
                merged[-1]["timing_warning"] = merged[-1]["timing_warning"] or timing_missing
                continue

            # Enforce monotonic non-decreasing start order
            if merged and s_val < merged[-1]["s"]:
                if chunk_idx > 0 and s_val <= overlap_boundary + 0.35:
                    norm_new = normalize_token(text)
                    is_dup = any(
                        normalize_token(prev["w"]) == norm_new and abs(prev["s"] - s_val) <= 0.65
                        for prev in merged[-8:]
                    )
                    if is_dup:
                        duplicates_removed += 1
                        continue
                s_val = merged[-1]["s"]
                if e_val <= s_val:
                    e_val = round(min(max_dur, s_val + 0.06), 3)

            # Also check overlap duplicates even when s_val >= merged[-1]["s"]
            if chunk_idx > 0 and s_val <= overlap_boundary + 0.35 and merged:
                norm_new = normalize_token(text)
                is_dup = any(
                    normalize_token(prev["w"]) == norm_new and abs(prev["s"] - s_val) <= 0.45
                    for prev in merged[-6:]
                )
                if is_dup:
                    duplicates_removed += 1
                    continue

            # Reject accidental identical zero-gap duplicate token from decoding glitch
            if (
                merged
                and normalize_token(merged[-1]["w"]) == normalize_token(text)
                and abs(merged[-1]["s"] - s_val) < 0.015
                and abs(merged[-1]["e"] - e_val) < 0.015
            ):
                duplicates_removed += 1
                continue

            merged.append(
                {
                    "w": text,
                    "s": s_val,
                    "e": e_val,
                    "speaker_id": rw.get("speaker_id"),
                    "source": "whisper_fallback" if timing_missing else "whisper",
                    "timing_warning": timing_missing,
                }
            )

    repair_contextual_homophones(merged)

    if missing_timing_count > 0:
        warnings.append(
            f"{missing_timing_count} word(s) lacked native Whisper timestamps and used bounded fallback timing."
        )

    return {
        "words": merged,
        "total_words": len(merged),
        "missing_timing_count": missing_timing_count,
        "duplicates_removed": duplicates_removed,
        "valid": len(merged) > 0,
        "warnings": warnings,
    }
