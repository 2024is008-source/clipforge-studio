import { describe, expect, it } from "vitest";

import { staticMotion, wordActivity, wordMotion, type CaptionStyle } from "./subtitles";

const FPS = 30;
const STYLES: CaptionStyle[] = [
  "opus",
  "beast",
  "karaoke",
  "minimal",
  "bounce",
  "typewriter",
  "glow",
  "reveal",
  "none",
];

/** A word spoken for 0.4 s starting at 1 s, the common case for a captioned token. */
const WORD = { w: "MILLION", s: 1.0, e: 1.4 };
const SHORT = { w: "a", s: 1.0, e: 1.08 };

function samples(style: CaptionStyle, from = 0.6, to = 2.4, step = 1 / FPS) {
  const out: number[] = [];
  for (let t = from; t <= to + 1e-9; t += step) out.push(wordMotion(style, t, WORD, 1).scale);
  return out;
}

describe("wordMotion continuity", () => {
  it("never steps the scale on the word's first frame", () => {
    // The linear ramp used to arrive at full 1.12 scale instantly, which is the snap being fixed.
    for (const style of STYLES) {
      const before = wordMotion(style, WORD.s - 1 / FPS, WORD, 1).scale;
      const at = wordMotion(style, WORD.s, WORD, 1).scale;
      expect(Math.abs(at - before), style).toBeLessThan(0.03);
    }
  });

  it("keeps every frame-to-frame scale step small enough to glide", () => {
    // Bounded by the pop attack, which is 3 frames at 30fps on purpose; what must never happen is
    // the old single-frame teleport from rest straight to full scale.
    for (const style of STYLES) {
      const xs = samples(style);
      const worst = Math.max(...xs.slice(1).map((x, i) => Math.abs(x - xs[i]!)));
      expect(worst, style).toBeLessThan(0.08);
    }
  });

  it("returns to rest and stays there", () => {
    for (const style of STYLES) {
      for (const t of [4, 9.5, 60]) {
        const m = wordMotion(style, t, WORD, 1);
        expect(m.scale, style).toBe(1);
        expect(m.y, style).toBe(0);
      }
    }
  });

  it("has a real pop that overshoots then settles", () => {
    for (const style of ["opus", "beast"] as CaptionStyle[]) {
      const xs = samples(style, WORD.s, WORD.s + 0.8);
      expect(Math.max(...xs)).toBeGreaterThan(1.1);
      expect(xs.at(-1)).toBeCloseTo(1, 5);
    }
  });

  it("lifts the bounce and lands it back on the baseline", () => {
    const lift = wordMotion("bounce", WORD.s + 0.2, WORD, 1).y;
    expect(lift).toBeLessThan(-1);
    expect(wordMotion("bounce", WORD.s + 0.6, WORD, 1).y).toBe(0);
  });

  it("fades reveal in monotonically over its window", () => {
    const xs: number[] = [];
    for (let t = WORD.s; t <= WORD.s + 0.4; t += 1 / FPS) xs.push(wordMotion("reveal", t, WORD, 1).alpha);
    expect(xs[0]).toBeLessThan(0.35);
    expect(xs.at(-1)).toBe(1);
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThanOrEqual(xs[i - 1]!);
  });
});

describe("wordActivity", () => {
  it("stays inside the unit range and starts at rest", () => {
    expect(wordActivity(WORD.s, WORD)).toBe(0);
    for (let t = WORD.s - 0.5; t <= WORD.e + 0.5; t += 1 / FPS) {
      const a = wordActivity(t, WORD);
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThanOrEqual(1);
    }
  });

  it("reaches full emphasis mid-word", () => {
    expect(wordActivity((WORD.s + WORD.e) / 2, WORD)).toBe(1);
  });

  it("shortens its ramps instead of overshooting a very short word", () => {
    const mid = wordActivity((SHORT.s + SHORT.e) / 2, SHORT);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThanOrEqual(1);
    // Past the word plus its tail there is nothing left to ramp.
    expect(wordActivity(SHORT.e + 0.5, SHORT)).toBe(0);
  });

  it("holds full emphasis for the whole word and fades only after it", () => {
    // Regression: the fall used to start a third of the word early, so the highlight left while the
    // narration still had the word and the caption read as lagging the voice.
    for (const frac of [0.3, 0.5, 0.7, 0.9]) {
      expect(wordActivity(WORD.s + (WORD.e - WORD.s) * frac, WORD), String(frac)).toBe(1);
    }
    expect(wordActivity(WORD.e - 0.01, WORD)).toBe(1);
    expect(wordActivity(WORD.e + 0.08, WORD)).toBeLessThan(1);
    expect(wordActivity(WORD.e + 0.17, WORD)).toBe(0);
  });

  it("moves continuously across the word edge", () => {
    // The exit ramp spans ~5 frames, so a third of the range per frame is the steepest honest slope.
    // The behaviour being ruled out is the old one: the chip and glow switching a whole unit at once.
    let previous = wordActivity(WORD.e - 0.2, WORD);
    for (let t = WORD.e - 0.2; t <= WORD.e + 0.3; t += 1 / FPS) {
      const next = wordActivity(t, WORD);
      expect(Math.abs(next - previous)).toBeLessThan(0.35);
      previous = next;
    }
  });
});

describe("motion switches", () => {
  it("strength 0 leaves the text at its resting transform", () => {
    for (const style of STYLES) {
      const m = wordMotion(style, WORD.s + 0.05, WORD, 0);
      expect(m.scale, style).toBe(1);
      expect(m.y, style).toBe(0);
    }
  });

  it("staticMotion holds the end state for every style", () => {
    for (const style of STYLES) {
      const m = staticMotion(style, WORD.s + 0.05, WORD);
      expect(m.scale, style).toBe(1);
      expect(m.y, style).toBe(0);
      expect(m.active, style).toBe(true);
      expect(m.alpha, style).toBe(1);
    }
    expect(staticMotion("reveal", WORD.s - 0.1, WORD).alpha).toBe(0);
  });

  it("never emits NaN for any style at any time", () => {
    for (const style of STYLES) {
      for (const t of [-1, 0, 0.999, 1.0, 1.2, 1.4, 1.401, 30]) {
        for (const word of [WORD, SHORT]) {
          const m = wordMotion(style, t, word, 1.5);
          expect(Number.isFinite(m.scale), `${style}@${t}`).toBe(true);
          expect(Number.isFinite(m.y), `${style}@${t}`).toBe(true);
          expect(Number.isFinite(m.alpha), `${style}@${t}`).toBe(true);
          expect(Number.isFinite(m.activity), `${style}@${t}`).toBe(true);
        }
      }
    }
  });
});
