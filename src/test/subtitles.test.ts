import { describe, expect, it } from "vitest";
import { captionPages, normalizeTranscriptWords, pageEmoji, parseSubtitles, serializeSubtitles, wordMotion } from "@/lib/subtitles";

describe("Subtitle timing and animation", () => {
  it("keeps imported cue timing and distributes words within the cue", () => {
    expect(parseSubtitles("1\n00:00:00,500 --> 00:00:01,500\nHello world", 10)).toEqual([
      { w: "Hello", s: 10.5, e: 11 }, { w: "world", s: 11, e: 11.5 },
    ]);
  });
  it("exports clip-relative timestamps without exceeding clip boundaries", () => {
    const text = serializeSubtitles([{ w: "Hello", s: 9, e: 11 }, { w: "world", s: 11, e: 13 }], 10, 12, 3, "srt");
    expect(text).toBe("1\n00:00:00,000 --> 00:00:02,000\nHello world\n");
  });
  it("reads VTT cue settings and excludes invalid time ranges", () => {
    expect(parseSubtitles("WEBVTT\n\n00:00.500 --> 00:01.500 align:start\n<b>Hello</b>\n\n00:03.000 --> 00:02.000\nBad")).toEqual([{ w: "Hello", s: 0.5, e: 1.5 }]);
  });
  it("settles bounce motion and supports disabling motion", () => {
    const w = { w: "Hello", s: 1, e: 2 };
    expect(wordMotion("bounce", 1.14, w).scale).toBeCloseTo(1.18);
    expect(wordMotion("bounce", 1.5, w).scale).toBeCloseTo(1);
    expect(wordMotion("bounce", 1.14, w, 0).y).toBe(0);
  });
  it("hides slide reveal words until their narration starts", () => {
    expect(wordMotion("reveal", 0, { w: "Hello", s: 1, e: 2 }).alpha).toBe(0);
    expect(wordMotion("reveal", 1.3, { w: "Hello", s: 1, e: 2 }).alpha).toBe(1);
  });
  it("splits pages by phrase, not by a fixed word count", () => {
    const words = ["One", "two.", "Three", "four", "five"].map((w, i) => ({ w, s: i, e: i + 1 }));
    // "five" no longer gets its own one-word page: the sentence fragment it belongs to absorbs it.
    expect(captionPages(words, 2).map((p) => p.map((w) => w.w))).toEqual([
      ["One", "two."],
      ["Three", "four", "five"],
    ]);
  });

  it("keeps a number with its unit and never strands an article", () => {
    const w = (text: string[]) => text.map((t, i) => ({ w: t, s: i * 0.4, e: i * 0.4 + 0.35 }));
    const money = captionPages(w(["I", "spent", "1", "million", "dollars", "today"]), 2);
    const page = money.find((p) => p.some((x) => x.w === "1"))!;
    const at = page.findIndex((x) => x.w === "1");
    expect(page.slice(at, at + 3).map((x) => x.w)).toEqual(["1", "million", "dollars"]);

    const article = captionPages(w(["the", "first", "meal", "of", "the", "day"]), 2);
    for (const page of article) {
      expect(["the", "of", "a", "to"]).not.toContain(page.at(-1)!.w.toLowerCase());
    }
  });

  it("breaks on the silence between spoken phrases and caps a page at six words", () => {
    const words = [
      { w: "because", s: 0.0, e: 0.3 },
      { w: "every", s: 0.35, e: 0.6 },
      { w: "calorie", s: 0.7, e: 1.0 },
      { w: "counts", s: 1.05, e: 1.4 },
      { w: "and", s: 2.2, e: 2.5 },
      { w: "so", s: 2.55, e: 2.8 },
      { w: "does", s: 2.85, e: 3.1 },
    ];
    const pages = captionPages(words, 2);
    expect(pages.map((p) => p.map((x) => x.w))).toEqual([
      ["because", "every", "calorie", "counts"],
      ["and", "so", "does"],
    ]);
    expect(Math.max(...pages.map((p) => p.length))).toBeLessThanOrEqual(6);
  });
  it("merges apostrophe contractions, comma numbers, decimals, and repairs ASR homophones", () => {
    const raw = [
      { w: "See", s: 0.0, e: 0.2 },
      { w: "y", s: 0.2, e: 0.3 },
      { w: "'all", s: 0.3, e: 0.5 },
      { w: "with", s: 0.5, e: 0.7 },
      { w: "6", s: 0.7, e: 0.8 },
      { w: ",614", s: 0.8, e: 1.1 },
      { w: "and", s: 1.1, e: 1.2 },
      { w: "2", s: 1.2, e: 1.3 },
      { w: ".4", s: 1.3, e: 1.5 },
      { w: "in", s: 1.5, e: 1.6 },
      { w: "new", s: 1.6, e: 1.8 },
      { w: "hour", s: 1.8, e: 2.1 },
    ];
    expect(normalizeTranscriptWords(raw).map((x) => x.w)).toEqual([
      "See",
      "y'all",
      "with",
      "6,614",
      "and",
      "2.4",
      "in",
      "new",
      "aisle",
    ]);
  });

  it("never freezes a caption page across a 10-second credit music gap (Rule 5: MAX_INTRA_PAGE_GAP_SEC = 0.65s)", () => {
    const words = [
      { w: "Good", s: 31.2, e: 31.5 },
      { w: "night!", s: 31.5, e: 31.8 },
      { w: "Why", s: 42.0, e: 42.3 },
      { w: "are", s: 42.3, e: 42.5 },
      { w: "you", s: 42.5, e: 42.8 },
    ];
    const pages = captionPages(words, 3);
    expect(pages.map((p) => p.map((w) => w.w))).toEqual([
      ["Good", "night!"],
      ["Why", "are", "you"],
    ]);
  });

  it("maps store/grocery to 🛒 and ghost/hunter to 👻 instead of false 🎮", () => {
    expect(pageEmoji([{ w: "store", s: 1, e: 1.4 }])?.emoji).toBe("🛒");
    expect(pageEmoji([{ w: "ghost", s: 2, e: 2.4 }])?.emoji).toBe("👻");
    expect(pageEmoji([{ w: "treadmill", s: 3, e: 3.4 }])?.emoji).toBe("🏋️");
  });
});
