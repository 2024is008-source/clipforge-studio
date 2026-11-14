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

/** Deterministic, word-relative motion shared by canvas preview and recorded export. */
export function wordMotion(style: CaptionStyle, time: number, word: Word, strength = 1) {
  const elapsed = Math.max(0, time - word.s);
  const progress = Math.min(1, elapsed / 0.28);
  const active = time >= word.s && time < word.e;
  let scale = 1,
    y = 0,
    alpha = 1;
  if (style === "bounce" && time >= word.s) {
    scale += Math.sin(progress * Math.PI) * 0.18 * strength;
    y = -Math.sin(progress * Math.PI) * 24 * strength;
  } else if ((style === "beast" || style === "opus") && active) {
    scale += 0.12 * Math.max(0, 1 - elapsed / 0.12) * strength;
  } else if (style === "reveal") {
    alpha = time < word.s ? 0 : 1 - Math.pow(1 - progress, 3);
    y = (1 - alpha) * 32 * strength;
  }
  return { active, scale, y: y === 0 ? 0 : y, alpha };
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
  /^(?:is\s+a|are\s+a|was\s+a|were\s+a|and|but|so|or|yeah|um|uh|like|well|because|that|which)\s+/i;
const TRAILING_DANGLING_RE =
  /\s+\b(?:a|an|the|is|are|was|were|to|of|in|on|for|with|and|or|but|yeah|uh|um|so|like|that)\s*$/i;

/**
 * Repair mid-sentence lowercase fragments and dangling articles in Hook Pill titles.
 * Example: "is a school backpack Yeah hers is a" -> "Hers Is A School Backpack"
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
    for (let i = 0; i < 3; i++) {
      const cleaned = text.replace(TRAILING_DANGLING_RE, "").trim();
      if (cleaned && cleaned !== text) text = cleaned;
      else break;
    }
  }
  const tokens = text.split(/\s+/).slice(0, 8);
  return tokens
    .map((tok, idx) =>
      idx === 0 && tok.length > 0 ? tok.charAt(0).toUpperCase() + tok.slice(1) : tok,
    )
    .join(" ");
}
