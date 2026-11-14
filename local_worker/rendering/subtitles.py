"""Caption pages, karaoke highlighting and styling as an ASS file libass can burn.

The browser paints word-by-word motion with `wordMotion`; FFmpeg cannot reproduce a spring, so the
active word arrives with a karaoke colour sweep plus a short fade instead. Text, segmentation,
timing, colour and placement all carry over unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .decision import OUTPUT_HEIGHT, OUTPUT_WIDTH, RenderDecision, Word
from .filters import is_gold_keyword

MAX_INTRA_PAGE_GAP_SEC = 0.65

# src/lib/subtitles.ts EMOJI_MAP, kept in the same order so the first match wins identically.
EMOJI_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(million|millions|billion|thousand|dollar|dollars|money|grand|cash|rich|charity|price|prize|paid|pay|\$\d+)\b", re.I), "💰"),
    (re.compile(r"\b(win|won|winner|champion|trophy|contestant|contestants|compete|first|victory)\b", re.I), "🏆"),
    (re.compile(r"\b(store|grocery|supermarket|aisle|cart|shop|shopping|bought|buy)\b", re.I), "🛒"),
    (re.compile(r"\b(weight|lbs|pounds|scale|lose|lost|gym|treadmill|mile|miles|run|running|workout|fitness|exercise)\b", re.I), "🏋️"),
    (re.compile(r"\b(shirt|xl|2xl|3xl|4xl|clothes|outfit|transformation)\b", re.I), "👕"),
    (re.compile(r"\b(ghost|ghosts|haunted|spooky|paranormal|demon|spirit|hunter)\b", re.I), "👻"),
    (re.compile(r"\b(trash|expired|rotten|spoiled|waste|garbage)\b", re.I), "🗑️"),
    (re.compile(r"\b(game|gaming|console|controller|arcade|player)\b", re.I), "🎮"),
    (re.compile(r"\b(crushing|sick|insane|fire|crazy|amazing|best|boom|unbelievable)\b", re.I), "🔥"),
    (re.compile(r"\b(scared|fear|omg|never|impossible|shock|wait|secret|risk|risking)\b", re.I), "😱"),
    (re.compile(r"\b(eat|eating|food|calories|hungry|taste|meal|feast)\b", re.I), "🍔"),
]

# preset -> (font size, outline, shadow, bold, primary colour when idle)
PRESET_STYLE = {
    "opus": (92, 7, 4, -1, "&H00FFFFFF"),
    "beast": (96, 8, 3, -1, "&H00FFFFFF"),
    "karaoke": (84, 5, 2, 0, "&H00E0E0E0"),
    "bounce": (90, 6, 4, -1, "&H00FFFFFF"),
    "typewriter": (80, 4, 0, 0, "&H00FFFFFF"),
    "glow": (88, 3, 10, -1, "&H00FFFFFF"),
    "reveal": (86, 6, 3, -1, "&H00FFFFFF"),
    "minimal": (76, 0, 6, 0, "&H00F2F2F2"),
}


@dataclass
class CaptionPage:
    start: float
    end: float
    words: list[Word]
    emoji: str | None = None


def page_emoji(words: list[Word]) -> tuple[str | None, float]:
    for w in words:
        for rx, emoji in EMOJI_MAP:
            if rx.search(w.w):
                return emoji, w.start
    return None, 0.0


def paginate(words: list[Word], count: int, clip_start: float) -> list[CaptionPage]:
    """Group tokens into caption pages, breaking early on a pause the viewer reads as a new line."""
    if not words:
        return []
    size = max(1, int(count))
    pages: list[CaptionPage] = []
    current: list[Word] = []
    for w in words:
        if current and w.start - current[-1].end > MAX_INTRA_PAGE_GAP_SEC:
            pages.append(_page(current, clip_start))
            current = []
        current.append(w)
        if len(current) >= size:
            pages.append(_page(current, clip_start))
            current = []
    if current:
        pages.append(_page(current, clip_start))
    return [p for p in pages if p.end > p.start]


def _page(words: list[Word], clip_start: float) -> CaptionPage:
    emoji, _ = page_emoji(words)
    return CaptionPage(
        start=max(clip_start, words[0].start) - clip_start,
        end=max(clip_start, words[-1].end) - clip_start,
        words=words,
        emoji=emoji,
    )


def _hex_to_ass(value: str) -> str:
    text = (value or "").lstrip("#")
    if len(text) != 6:
        return "&H00FFFFFF"
    r, g, b = text[0:2], text[2:4], text[4:6]
    return f"&H00{b}{g}{r}".upper()


def _stamp(seconds: float) -> str:
    total = max(0.0, seconds)
    cs = int(round(total * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"



LEADING_FRAGMENT_RE = re.compile(
    r"^(?:is\s+a|are\s+a|was\s+a|were\s+a|and|but|so|or|yeah|um|uh|like|well|because|that|which)\s+",
    re.IGNORECASE,
)
TRAILING_DANGLING_RE = re.compile(
    r"\s+\b(?:a|an|the|is|are|was|were|to|of|in|on|for|with|and|or|but|yeah|uh|um|so|like|that)\s*$",
    re.IGNORECASE,
)


def clean_hook_title(raw_hook: str, words: list[Word] | None = None) -> str:
    """Repair mid-sentence lowercase fragments and dangling articles in Hook Pill titles.

    Example: 'is a school backpack Yeah hers is a' -> 'Hers Is A School Backpack'
    """
    text = (raw_hook or "").strip()
    if not text and words:
        text = " ".join(w.w for w in words[:8]).strip()
    if not text:
        return ""

    # Detect duplicated wrap-around fragment like "is a school backpack Yeah hers is a"
    m_dup = re.match(
        r"^(?:is|are|was)\s+(a\s+[a-z0-9\s]{3,30}?)\s+(?:yeah|yes|no|so|and|but)?\s*(hers|his|mine|yours|theirs|ours|this|that|it)\s+(?:is|was)\s+a$",
        text,
        re.IGNORECASE,
    )
    if m_dup:
        noun_phrase = m_dup.group(1).strip()
        subj = m_dup.group(2).strip()
        text = f"{subj} is {noun_phrase}"
    else:
        # Strip leading mid-sentence fragments ("is a ...", "and ...") if enough words remain
        for _ in range(2):
            candidate = LEADING_FRAGMENT_RE.sub("", text).strip()
            if len(candidate.split()) >= 3:
                text = candidate
        # Strip trailing dangling articles/prepositions ("... hers is a" -> "... hers")
        for _ in range(3):
            cleaned = TRAILING_DANGLING_RE.sub("", text).strip()
            if cleaned and cleaned != text:
                text = cleaned
            else:
                break

    tokens = text.split()
    if len(tokens) > 8:
        tokens = tokens[:8]
    return " ".join(tokens)


def _escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")


HOOK_DURATION = 3.6


def write_ass(decision: RenderDecision, path: str, anchor_y: float | None = None,
              *, hook_duration: float = HOOK_DURATION) -> str:
    """Write the subtitle file and return it.

    `anchor_y` is the normalised vertical centre of the caption block, matching `baseY` in
    `drawCaption`, which centres the block on H * captionY whatever the line count. ASS bottom
    anchoring with a fixed MarginV cannot do that: a two-word page and a six-word page would sit at
    different heights. Each event therefore carries an `\\an5\\pos()` override, which pins the middle
    of the rendered box to the point the preview uses.
    """
    style = decision.style
    size, outline, shadow, bold, idle = PRESET_STYLE.get(style.preset, PRESET_STYLE["opus"])
    if style.font_size:
        size = int(style.font_size)
    active = _hex_to_ass(style.highlight)
    gold = _hex_to_ass(style.secondary_highlight)

    y = OUTPUT_HEIGHT * (max(0.0, min(1.0, anchor_y)) if anchor_y is not None else style.position_y)
    anchor = r"\an5\pos" + f"({OUTPUT_WIDTH // 2},{int(round(y))})"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUTPUT_WIDTH}",
        f"PlayResY: {OUTPUT_HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{style.font},{size},{idle},{active},&H00000000,&H80000000,{bold},0,0,0,"
        f"100,100,3,0,1,{outline},{shadow},2,64,64,0,1",
        f"Style: Hook,{style.font},{max(44, size - 38)},&H00120F0F,&H00120F0F,&H00FFFFFF,&H60000000,"
        "-1,0,0,0,100,100,1,0,3,18,0,7,60,420,60,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    cleaned_hook = clean_hook_title(decision.hook, decision.words)
    if style.show_hook and cleaned_hook and hook_duration > 0:
        hook = _escape(cleaned_hook.upper())
        lines.append(
            f"Dialogue: 1,{_stamp(0.0)},{_stamp(min(hook_duration, decision.duration))},Hook,,0,0,0,,{{\\fad(160,220)}}{hook}"
        )

    for page in paginate(decision.words, style.words_per_page, decision.start):
        parts: list[str] = []
        for w in page.words:
            centis = max(1, int(round((w.end - w.start) * 100)))
            token = _escape(w.w)
            # A gold keyword swaps the karaoke sweep to the accent colour for that word only.
            lead = r"\2c" + gold + r"\c" if is_gold_keyword(w.w) else ""
            parts.append(f"{{\\k{centis}{lead}}}{token} ")
        text = "".join(parts).rstrip()
        if style.show_emojis and page.emoji:
            text = f"{text} {{\\fscx120\\fscy120}}{page.emoji}"
        # The block fades in rather than springing, which is the documented approximation.
        text = r"{\fad(80,0)" + anchor + "}" + text
        lines.append(f"Dialogue: 0,{_stamp(page.start)},{_stamp(page.end)},Cap,,0,0,0,,{text}")

    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(content)
    return path
