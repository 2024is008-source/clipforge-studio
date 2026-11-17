import type { Word } from "./media";

export type CaptionStyle =
  | "opus"
  | "beast"
  | "karaoke"
  | "minimal"
  | "bounce"
  | "typewriter"
  | "glow"
  | "reveal"
  | "none";
export const CAPTION_PRESETS: { id: CaptionStyle; name: string; sample: string }[] = [
  { id: "opus", name: "ClipForge Viral", sample: "WIN A MILLION" },
  { id: "beast", name: "Bold pop", sample: "MAKE IT COUNT" },
  { id: "karaoke", name: "Karaoke", sample: "Every word matters" },
  { id: "bounce", name: "Bounce", sample: "Bring the energy" },
  { id: "typewriter", name: "Typewriter", sample: "One word at a time" },
  { id: "glow", name: "Neon glow", sample: "Light up the moment" },
  { id: "reveal", name: "Slide reveal", sample: "Tell your story" },
  { id: "minimal", name: "Clean", sample: "Less is more" },
  { id: "none", name: "No captions", sample: "" },
];

// Whole-word semantic emoji triggers calibrated for ClipForge Studio (zero false 🎮 on "store"/"entire")
const EMOJI_MAP: [RegExp, string][] = [
  [/\b(million|millions|billion|thousand|dollar|dollars|money|grand|cash|rich|charity|price|prize|paid|pay|\$\d+)\b/i, "💰"],
  [/\b(win|won|winner|champion|trophy|contestant|contestants|compete|first|victory)\b/i, "🏆"],
  [/\b(store|grocery|supermarket|aisle|cart|shop|shopping|bought|buy)\b/i, "🛒"],
  [/\b(weight|lbs|pounds|scale|lose|lost|gym|treadmill|mile|miles|run|running|workout|fitness|exercise)\b/i, "🏋️"],
  [/\b(shirt|xl|2xl|3xl|4xl|clothes|outfit|transformation)\b/i, "👕"],
  [/\b(ghost|ghosts|haunted|spooky|paranormal|demon|spirit|hunter)\b/i, "👻"],
  [/\b(trash|expired|rotten|spoiled|waste|garbage)\b/i, "🗑️"],
  [/\b(game|gaming|console|controller|arcade|player)\b/i, "🎮"],
  [/\b(crushing|sick|insane|fire|crazy|amazing|best|boom|unbelievable)\b/i, "🔥"],
  [/\b(scared|fear|omg|never|impossible|shock|wait|secret|risk|risking)\b/i, "😱"],
  [/\b(eat|eating|food|calories|hungry|taste|meal|feast)\b/i, "🍔"],
];

/** Return a contextual emoji for a caption page if any word triggers a high-impact whole-word keyword match. */
export function pageEmoji(page: Word[]): { emoji: string; triggerTime: number } | null {
  for (const w of page) {
    for (const [rx, emoji] of EMOJI_MAP) {
      if (rx.test(w.w)) return { emoji, triggerTime: w.s };
    }
  }
  return null;
}

/** Detect high-value numbers/power words that receive the secondary gold accent in Opus style. */
export function isGoldKeyword(word: string): boolean {
  return /(\$|\d|million|millions|billion|thousand|grand|everything|never|impossible|charity|winner)/i.test(
    word,
  );
}

const APOSTROPHE_CONT_RE = /^['’](all|s|t|re|ve|ll|d|m|em|cause|clock|tis|twas)\b/i;
const COMMA_DECIMAL_CONT_RE = /^[,.\u066B]\d+/;
const CURRENCY_ONLY_RE = /^[$€£¥₹]$/;
const LEADING_DIGITS_RE = /^\d[\d,.]*%?[,!?.]*$/;
const TRAILING_NUM_PREFIX_RE = /^\$?\d+[,.]$/;
const SHORT_INT_RE = /^\$?\d{1,3}$/;
const COMMA_THOUSANDS_RE = /^,\d{3}\b/;

/**
 * Rule 6 & 9: Merge sub-word continuation tokens emitted by faster-whisper:
 * - Apostrophe contractions ("y" + "'all" -> "y'all", "don" + "'t" -> "don't")
 * - Comma & decimal numbers ("6" + ",614" -> "6,614", "2" + ".4" -> "2.4", "$" + "6,614" -> "$6,614")
 * - Clothing/weight scale tokens ("4" + "XL" -> "4XL")
 * - Contextual ASR homophones ("105 looks" -> "105 loaves", "cans the pair" -> "cans of pears", "new hour" -> "new aisle")
 */
export function normalizeTranscriptWords(words: Word[]): Word[] {
  if (!words.length) return [];
  const merged: Word[] = [];
  for (const raw of words) {
    const text = (raw.w ?? "").trim();
    if (!text) continue;
    const prev = merged.at(-1);
    if (prev && raw.s - prev.e <= 0.45) {
      const p = prev.w.trim();
      const isApostrophe =
        APOSTROPHE_CONT_RE.test(text) ||
        (p.endsWith("'") || p.endsWith("’") ? /^[a-zA-Z]+[,!?.]*$/.test(text) : false);
      const isNumberCont =
        (COMMA_DECIMAL_CONT_RE.test(text) && /\d$/.test(p)) ||
        (CURRENCY_ONLY_RE.test(p) && LEADING_DIGITS_RE.test(text)) ||
        (TRAILING_NUM_PREFIX_RE.test(p) && LEADING_DIGITS_RE.test(text)) ||
        (SHORT_INT_RE.test(p) && COMMA_THOUSANDS_RE.test(text)) ||
        (/^\d{1,2}$/.test(p) && /^-?(xl|xxl|xxxl)[,!?.]*$/i.test(text));
      if (isApostrophe || isNumberCont) {
        const cleanSuffix =
          /^\d/.test(p) && text.toLowerCase().startsWith("-xl") ? text.slice(1) : text;
        prev.w = `${p}${cleanSuffix}`;
        prev.e = Math.max(prev.e, raw.e);
        continue;
      }
    }
    merged.push({ ...raw, w: text });
  }

  // Contextual multi-word homophone repairs
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const fixes: [string[], string[]][] = [
    [["105", "looks"], ["105", "loaves"]],
    [["cans", "the", "pair"], ["cans", "of", "pears"]],
    [["cans", "the", "pears"], ["cans", "of", "pears"]],
    [["new", "hour"], ["new", "aisle"]],
    [["jars", "of", "season"], ["jars", "of", "seasoning"]],
    [["good", "fucking", "god"], ["good", "fucking", "point"]],
  ];
  for (let i = 0; i < merged.length; i++) {
    for (const [pat, rep] of fixes) {
      if (i + pat.length > merged.length) continue;
      let match = true;
      for (let j = 0; j < pat.length; j++) {
        if (norm(merged[i + j]!.w) !== pat[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        for (let j = 0; j < rep.length; j++) {
          const orig = merged[i + j]!.w;
          const punct = /([,!?.]+)$/.exec(orig)?.[1] ?? "";
          const target = rep[j]!;
          const cased =
            orig === orig.toUpperCase() && orig !== orig.toLowerCase()
              ? target.toUpperCase()
              : orig[0] && orig[0] === orig[0].toUpperCase()
                ? target.charAt(0).toUpperCase() + target.slice(1)
                : target;
          merged[i + j] = { ...merged[i + j]!, w: `${cased}${punct}` };
        }
      }
    }
  }
  return merged;
}

const MAX_PAGE_WORDS = 6;
/** A page may not stay up long enough to be re-read as a new thought. */
const MAX_PAGE_SEC = 2.2;
/** Rule 5: Maximum silence gap allowed between two words inside the same caption page. */
export const MAX_INTRA_PAGE_GAP_SEC = 0.65;

/** Words that must not be the last thing on screen: the phrase continues. */
const DANGLING = new Set([
  "a", "an", "the", "of", "to", "and", "or", "but", "in", "on", "for", "with", "that", "this",
  "is", "was", "are", "be", "has", "have", "at", "as", "so", "if", "my", "your", "his", "her", "our",
]);

const bare = (w: string) => w.toLowerCase().replace(/[^a-z0-9$']/g, "");
const QUANTITY =
  /^(\$?\d[\d.,]*%?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|half|quarter|hundred|thousand|million|billion|trillion|percent)$/i;
const UNIT =
  /^(millions?|billions?|thousands?|hundreds?|dollars?|bucks|percent|seconds?|minutes?|hours?|days?|calories?|kgs?|lbs?|times|people)$/i;

/**
 * Words that must share a page. A stranded article reads as a mistake, and splitting "1 million
 * dollars" across two pages makes the viewer re-read the number as a new thought.
 */
function mustStayTogether(prev: Word, next: Word): boolean {
  if (next.s - prev.e > MAX_INTRA_PAGE_GAP_SEC) return false;
  return (
    DANGLING.has(bare(prev.w)) ||
    (QUANTITY.test(bare(prev.w)) && UNIT.test(bare(next.w)))
  );
}

/**
 * Page captions by spoken phrase rather than by a fixed word count.
 * Enforces Rule 5 (MAX_INTRA_PAGE_GAP_SEC = 0.65s) so words separated by a long pause or credit music
 * (e.g. "Good night!" at 31.8s and "Why" at 42.0s in WA0029) are never grouped into a frozen page.
 */
export function captionPages(words: Word[], count = 3): Word[][] {
  const normalized = normalizeTranscriptWords(words);
  if (!normalized.length) return [];
  const target = Math.max(2, Math.min(MAX_PAGE_WORDS, Math.round(count)));
  const groups: Word[][] = [];
  let group: Word[] = [];
  for (const word of normalized) {
    if (group.length && !mustStayTogether(group[group.length - 1]!, word)) {
      groups.push(group);
      group = [];
    }
    group.push(word);
  }
  if (group.length) groups.push(group);

  const pages: Word[][] = [];
  let page: Word[] = [];
  for (let gi = 0; gi < groups.length; gi++) {
    const g = groups[gi]!;
    if (page.length && page.length + g.length > MAX_PAGE_WORDS) {
      pages.push(page);
      page = [];
    }
    page.push(...g);

    const last = page[page.length - 1]!;
    const nextWord = groups[gi + 1]?.[0];
    const longSilenceGap = !!nextWord && nextWord.s - last.e > MAX_INTRA_PAGE_GAP_SEC;
    const boundary =
      /[.!?]$/.test(last.w) ||
      /[,:;]$/.test(last.w) ||
      (!!nextWord && nextWord.s - last.e >= 0.28) ||
      (!!nextWord && page.length >= target + 1 && /[A-Z]/.test(nextWord.w[0] ?? ""));
    if (
      longSilenceGap ||
      page.length >= MAX_PAGE_WORDS ||
      last.e - page[0]!.s >= MAX_PAGE_SEC ||
      (page.length >= target && boundary)
    ) {
      pages.push(page);
      page = [];
    }
  }
  if (page.length) pages.push(page);

  // Fold a one-word tail page into its neighbour ONLY when there is no long silence gap between them.
  if (
    pages.length > 1 &&
    pages.at(-1)!.length === 1 &&
    pages.at(-2)!.length < MAX_PAGE_WORDS &&
    pages.at(-1)![0]!.s - pages.at(-2)!.at(-1)!.e <= MAX_INTRA_PAGE_GAP_SEC
  ) {
    const tail = pages.pop()!;
    pages.at(-2)!.push(...tail);
  }
  return pages;
}

export type WordMotion = {
  active: boolean;
  scale: number;
  y: number;
  alpha: number;
  /** Eased 0→1→0 emphasis level across the word, for effects that must ramp instead of switching. */
  activity: number;
};

/** Ease with zero velocity at both ends, so a curve never arrives or leaves at speed. */
function smoothstep(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
}

/**
 * Continuous hump: 0 at rest, 1 at the peak, and no step where the phases meet. Used for the pop,
 * which previously jumped straight to full scale on the word's first frame and decayed linearly, and
 * for the bounce lift, whose raw `sin(p*PI)` reached the same peak but arrived back at rest with a
 * non-zero velocity and a `Math.sin(Math.PI)` residue that never quite settled to zero.
 */
function hump(age: number, attack: number, settle: number): number {
  if (age <= 0) return 0;
  if (age < attack) return smoothstep(age / attack);
  return 1 - smoothstep((age - attack) / settle);
}

/**
 * Eased in and out around the word's window so colour, glow and the karaoke chip can interpolate
 * across the boundary. The hold runs to the end of the word: a highlight that faded while the
 * narration still had the word reads as the caption lagging the voice. Short words get proportionally
 * shorter ramps, and the tail is capped at the word's own length so it cannot outlast the page.
 */
export function wordActivity(time: number, word: Word, rise = 0.09, fall = 0.16): number {
  const span = Math.max(0.05, word.e - word.s);
  const up = Math.min(rise, span / 2);
  const down = Math.min(fall, span);
  if (time <= word.s || time >= word.e + down) return 0;
  if (time < word.s + up) return smoothstep((time - word.s) / up);
  if (time > word.e) return 1 - smoothstep((time - word.e) / down);
  return 1;
}

const BOUNCE = { attack: 0.14, settle: 0.3, peak: 0.18, lift: 24 };
const POP = { attack: 0.09, settle: 0.3, peak: 0.12 };
const REVEAL = { duration: 0.34, slide: 32 };

/** Deterministic, word-relative motion shared by canvas preview and recorded export. */
export function wordMotion(
  style: CaptionStyle,
  time: number,
  word: Word,
  strength = 1,
): WordMotion {
  const elapsed = Math.max(0, time - word.s);
  const active = time >= word.s && time < word.e;
  const activity = wordActivity(time, word);
  let scale = 1,
    y = 0,
    alpha = 1;
  if (style === "bounce" && time >= word.s) {
    const a = hump(elapsed, BOUNCE.attack, BOUNCE.settle);
    scale += BOUNCE.peak * a * strength;
    y = -BOUNCE.lift * a * strength;
  } else if (style === "beast" || style === "opus") {
    scale += POP.peak * hump(elapsed, POP.attack, POP.settle) * strength;
  } else if (style === "reveal") {
    const p = Math.min(1, elapsed / REVEAL.duration);
    alpha = time < word.s ? 0 : 1 - Math.pow(1 - p, 3);
    y = (1 - alpha) * REVEAL.slide * strength;
  }
  return { active, scale, y: y === 0 ? 0 : y, alpha, activity };
}

/** Motion a preset applies when animation is off: the settled end state, with no ramp. */
export function staticMotion(style: CaptionStyle, time: number, word: Word): WordMotion {
  const active = time >= word.s && time < word.e;
  return {
    active,
    scale: 1,
    y: 0,
    alpha: style === "reveal" && time < word.s ? 0 : 1,
    activity: active ? 1 : 0,
  };
}

export function subtitleTimestamp(seconds: number, separator = ",") {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  const pad = (n: number, len = 2) => String(n).padStart(len, "0");
  return `${pad(Math.floor(ms / 3_600_000))}:${pad(Math.floor((ms % 3_600_000) / 60_000))}:${pad(
    Math.floor((ms % 60_000) / 1000),
  )}${separator}${pad(ms % 1000, 3)}`;
}

export function serializeSubtitles(
  words: Word[],
  start: number,
  end: number,
  count: number,
  format: "srt" | "vtt",
) {
  const separator = format === "srt" ? "," : ".";
  const pages = captionPages(words, count);
  const header = format === "vtt" ? "WEBVTT\n\n" : "";
  const cues = pages
    .map((page, i) => {
      const first = page[0]!,
        last = page[page.length - 1]!;
      return `${format === "srt" ? `${i + 1}\n` : ""}${subtitleTimestamp(Math.max(start, first.s) - start, separator)} --> ${subtitleTimestamp(Math.min(end, last.e) - start, separator)}\n${page.map((w) => w.w).join(" ")}`;
    })
    .join("\n\n");
  return `${header}${cues}${cues ? "\n" : ""}`;
}

export function parseSubtitles(text: string, offset = 0): Word[] {
  const words: Word[] = [];
  const seconds = (stamp: string) => {
    const parts = stamp.replace(",", ".").split(":").map(Number);
    return parts.length >= 3
      ? parts[0]! * 3600 + parts[1]! * 60 + parts[2]!
      : parts[0]! * 60 + parts[1]!;
  };
  const lines = text.replace(/\r/g, "").split("\n");
  for (let index = 0; index < lines.length; index++) {
    const timing = lines[index]?.match(
      /(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})/,
    );
    if (!timing?.[1] || !timing[2]) continue;
    const start = seconds(timing[1]) + offset,
      end = seconds(timing[2]) + offset;
    if (!Number.isFinite(start) || end <= start) continue;
    const body: string[] = [];
    for (let j = index + 1; j < lines.length; j++) {
      const line = lines[j]!;
      if (!line.trim() || line.includes("-->")) break;
      body.push(line);
    }
    index = body.length ? index + body.length : index;
    const tokens = body
      .join(" ")
      .replace(/<[^>]*>/g, "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    const step = (end - start) / Math.max(1, tokens.length);
    tokens.forEach((w, i) => words.push({ w, s: start + i * step, e: start + (i + 1) * step }));
  }
  return normalizeTranscriptWords(words.sort((a, b) => a.s - b.s));
}


const LEADING_FRAGMENT_RE =
  /^(?:is\s+a|are\s+a|was\s+a|were\s+a|is|are|was|were|and|but|so|or|yeah|um|uh|like|well|because|that|which)\s+/i;
const TRAILING_DANGLING_RE =
  /\s+\b(?:a|an|the|is|are|was|were|to|of|in|on|for|with|and|or|but|yeah|uh|um|so|like|that|could|would|should|can|will)\s*$/i;
const SENTENCE_STARTER_TOKEN_RE =
  /^(?:It|All|He|She|They|We|I|You|This|That|These|Those|There|Here|But|And|So|When|If|Now|Then)[,!?.]*$/;

/**
 * Repair mid-sentence lowercase fragments, dangling articles, and trailing 2-word sentence starters in Hook Pill titles.
 * Examples:
 * - "is a school backpack Yeah hers is a" -> "Hers is a school backpack"
 * - "are many different ways of communication It could" -> "Many different ways of communication"
 * - "To my left are 5 people All 5" -> "To my left are 5 people"
 */
export function cleanHookTitle(rawHook: string, words?: Word[]): string {
  let text = (rawHook || "").trim();
  if (!text && words?.length) {
    text = words
      .slice(0, 8)
      .map((w) => w.w)
      .join(" ")
      .trim();
  }
  if (!text) return "";

  const dupMatch =
    /^(?:is|are|was)\s+(a\s+[a-z0-9\s]{3,30}?)\s+(?:yeah|yes|no|so|and|but)?\s*(hers|his|mine|yours|theirs|ours|this|that|it)\s+(?:is|was)\s+a$/i.exec(
      text,
    );
  if (dupMatch) {
    text = `${dupMatch[2]!.trim()} is ${dupMatch[1]!.trim()}`;
  } else {
    for (let i = 0; i < 2; i++) {
      const candidate = text.replace(LEADING_FRAGMENT_RE, "").trim();
      if (candidate.split(/\s+/).length >= 3) text = candidate;
    }
  }
  let tokens = text.split(/\s+/).slice(0, 8);
  for (let idx = 4; idx < tokens.length; idx++) {
    const prevTok = tokens[idx - 1]!;
    const curTok = tokens[idx]!;
    if (/[.!?]$/.test(prevTok)) {
      tokens = tokens.slice(0, idx);
      break;
    }
    const curUpper = curTok[0] !== undefined && curTok[0] === curTok[0].toUpperCase() && curTok[0] !== curTok[0].toLowerCase();
    const prevUpper = prevTok[0] !== undefined && prevTok[0] === prevTok[0].toUpperCase() && prevTok[0] !== prevTok[0].toLowerCase();
    if (
      tokens.length - idx <= 2 &&
      curUpper &&
      !prevUpper &&
      SENTENCE_STARTER_TOKEN_RE.test(curTok)
    ) {
      tokens = tokens.slice(0, idx);
      break;
    }
  }
  text = tokens.join(" ");
  for (let i = 0; i < 3; i++) {
    const cleaned = text.replace(TRAILING_DANGLING_RE, "").trim();
    if (cleaned && cleaned !== text && cleaned.split(/\s+/).length >= 3) text = cleaned;
    else break;
  }
  tokens = text.split(/\s+/);
  return tokens
    .map((tok, idx) =>
      idx === 0 && tok.length > 0 ? tok.charAt(0).toUpperCase() + tok.slice(1) : tok,
    )
    .join(" ");
}

/** Hook card geometry at 1080x1920, mirrored by rendering/subtitles.py `Style: Hook`. */
export const HOOK_DURATION = 3.6;
export const HOOK_FONT_SIZE = 60;
export const HOOK_LINE_PITCH = 77;
export const HOOK_PAD_X = 46;
export const HOOK_PAD_Y = 18;
export const HOOK_TOP = 145;
export const HOOK_RADIUS = 13;
export const HOOK_MAX_WIDTH = 810;

/**
 * OpenShorts face-safe & caption-safe dynamic hook placement (`hook_placement.py`):
 * Scans `[0.06, 0.80]` in steps of `0.005`, avoids bottom captions `(0.70, 0.88)` and split-seam
 * captions `(0.42, 0.58)`, expands detected face boxes to full heads, and tries shrink scales
 * `(1.0, 0.85, 0.72)` until face overlap is `<= 8%`.
 */
export function chooseHookPlacement(
  faces: Array<[number, number, number, number]> = [],
  layoutMode: string = "fill",
  outW = 1080,
  outH = 1920,
  defaultTopPx = HOOK_TOP,
  boxWPx = HOOK_MAX_WIDTH,
  boxHPx = HOOK_LINE_PITCH * 2,
): { topNorm: number; topPx: number; scale: number; faceOverlap: number } {
  const TOP_MARGIN = 0.06;
  const BOTTOM_UI = 0.80;
  const STEP = 0.005;
  const MOVE_PENALTY = 0.05;
  const GOOD_ENOUGH = 0.08;
  const SHRINK_SCALES = [1.0, 0.85, 0.72];

  const preferredTop = Math.max(TOP_MARGIN, Math.min(BOTTOM_UI, defaultTopPx / Math.max(1, outH)));
  const baseW = Math.min(0.92, boxWPx / Math.max(1, outW));
  const baseH = Math.min(0.25, boxHPx / Math.max(1, outH));
  const forbidden: Array<[number, number]> =
    layoutMode === "split" ? [[0.42, 0.58]] : [[0.70, 0.88]];

  const headBox = (f: [number, number, number, number]): [number, number, number, number] => {
    const w = Math.max(0, f[2] - f[0]);
    const h = Math.max(0, f[3] - f[1]);
    return [
      Math.max(0, f[0] - 0.15 * w),
      Math.max(0, f[1] - 0.60 * h),
      Math.min(1, f[2] + 0.15 * w),
      Math.min(1, f[3] + 0.20 * h),
    ];
  };

  const faceCost = (b: [number, number, number, number]): number => {
    let worst = 0;
    for (const f of faces) {
      const hb = headBox(f);
      const area = Math.max(1e-6, (hb[2] - hb[0]) * (hb[3] - hb[1]));
      const ix = Math.max(0, Math.min(b[2], hb[2]) - Math.max(b[0], hb[0]));
      const iy = Math.max(0, Math.min(b[3], hb[3]) - Math.max(b[1], hb[1]));
      worst = Math.max(worst, (ix * iy) / area);
    }
    return worst;
  };

  const overlapsBand = (y0: number, y1: number) =>
    forbidden.some(([b0, b1]) => Math.min(y1, b1) > Math.max(y0, b0));

  const defaultBanner: [number, number, number, number] = [
    0.5 - baseW / 2,
    preferredTop,
    0.5 + baseW / 2,
    preferredTop + baseH,
  ];
  const defaultOv = faceCost(defaultBanner);
  if (defaultOv <= GOOD_ENOUGH && !overlapsBand(preferredTop, preferredTop + baseH)) {
    return { topNorm: preferredTop, topPx: defaultTopPx, scale: 1.0, faceOverlap: defaultOv };
  }

  let chosenY = preferredTop;
  let chosenScale = 1.0;
  let chosenOv = 1.0;

  for (const scale of SHRINK_SCALES) {
    const w = baseW * scale;
    const h = baseH * scale;
    const x0 = Math.max(0, 0.5 - w / 2);
    const x1 = Math.min(1, 0.5 + w / 2);
    const upper = Math.max(TOP_MARGIN, BOTTOM_UI - h);
    let bestY = preferredTop;
    let bestScore = Infinity;
    let bestOv = 1.0;
    for (let y0 = TOP_MARGIN; y0 <= upper + 1e-6; y0 += STEP) {
      const clampedY = Math.min(upper, y0);
      const y1 = clampedY + h;
      if (overlapsBand(clampedY, y1)) continue;
      const ov = faceCost([x0, clampedY, x1, y1]);
      const score = ov + MOVE_PENALTY * Math.abs(clampedY - preferredTop);
      if (score < bestScore - 1e-9) {
        bestScore = score;
        bestY = clampedY;
        bestOv = ov;
      }
    }
    chosenY = bestY;
    chosenScale = scale;
    chosenOv = bestOv;
    if (bestOv <= GOOD_ENOUGH) break;
  }

  return {
    topNorm: Number(chosenY.toFixed(4)),
    topPx: Math.round(chosenY * outH),
    scale: chosenScale,
    faceOverlap: Number(chosenOv.toFixed(4)),
  };
}
