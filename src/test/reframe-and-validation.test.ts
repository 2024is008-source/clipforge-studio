import { describe, expect, it } from "vitest";
import {
  boxIoU,
  detectSceneCut,
  sampleTrack,
  OpenShortsSpeakerTracker,
  OpenShortsSmoothedCameraman,
  analyzeSplitScene,
  analyzeThreeUpScene,
  filterEdgeBystanders,
  analyzeWideScene,
  type FrameSignature,
  type FaceBox,
} from "@/lib/reframe";
import { filterAndDedupeClips, validateMonotonicWords, getConfiguredGeminiKeys, type AIClip } from "@/lib/ai.functions";
import { numberCues } from "@/lib/render";
import { trimClipNarrativeBoundaries } from "@/lib/local-worker";
import { evaluateCampaignCompliance } from "@/components/VyroWorkspace";
import { isGoldKeyword, pageEmoji } from "@/lib/subtitles";

describe("Scene cut detection, Track ID IoU, and Shot Interpolation", () => {
  it("computes IoU accurately for persistent speaker track IDs", () => {
    const a = { cx: 0.5, cy: 0.4, w: 0.2, h: 0.2, score: 1 };
    const b = { cx: 0.52, cy: 0.41, w: 0.2, h: 0.2, score: 1 };
    const far = { cx: 0.1, cy: 0.8, w: 0.1, h: 0.1, score: 1 };
    expect(boxIoU(a, b)).toBeGreaterThan(0.65);
    expect(boxIoU(a, far)).toBe(0);
  });

  it("ignores uniform brightness flashes when histogram does not shift, and detects real cuts after hysteresis cooldown", () => {
    const sigA: FrameSignature = {
      hist: new Float32Array(48).fill(1 / 16),
      lumaGrid: new Float32Array(144).fill(0.2),
      meanLuma: 51,
    };
    const sigB: FrameSignature = {
      hist: new Float32Array(48),
      lumaGrid: new Float32Array(144).fill(0.8),
      meanLuma: 204,
    };
    sigB.hist[0] = 1;
    sigB.hist[16] = 1;
    sigB.hist[32] = 1;
    // Blocked during cooldown (< 0.4s)
    expect(detectSceneCut(sigA, sigB, 0.2)).toBe(false);
    // Triggers after cooldown when both histogram and spatial luma shift
    expect(detectSceneCut(sigA, sigB, 0.6)).toBe(true);
  });

  it("does not interpolate camera position across a hard scene cut", () => {
    const track = [
      { t: 0, x: 0.2, y: 0.4, mode: "fill" as const, cut: true },
      { t: 1.0, x: 0.8, y: 0.4, mode: "split" as const, cut: true },
    ];
    expect(sampleTrack(track, 0.5).x).toBeCloseTo(0.2);
    expect(sampleTrack(track, 0.99).x).toBeCloseTo(0.8);
  });

  it("carries the worker's audio-envelope punch-in zoom through interpolation", () => {
    const track = [
      { t: 0, x: 0.5, y: 0.4, mode: "fill" as const, zoom: 1.0 },
      { t: 1.0, x: 0.5, y: 0.4, mode: "fill" as const, zoom: 1.12 },
    ];
    expect(sampleTrack(track, 0.5).zoom).toBeCloseTo(1.06);
    expect(sampleTrack(track, 0.0).zoom).toBe(1.0);
    expect(sampleTrack([{ t: 0, x: 0.5, y: 0.4 }], 0.5).zoom).toBeUndefined();
  });
});

describe("OpenShorts Reframe Engine v2 (SpeakerTracker, SmoothedCameraman, split_layout, 5-Key Gemini Pool)", () => {
  it("SpeakerTracker holds on locked speaker during cooldown when active speaker blinks out", () => {
    const tracker = new OpenShortsSpeakerTracker(8, 8);
    const leftSpeaker: FaceBox = { cx: 0.25, cy: 0.35, w: 0.1, h: 0.18, score: 0.018 };
    const rightSpeaker: FaceBox = { cx: 0.75, cy: 0.35, w: 0.1, h: 0.18, score: 0.018 };

    // Frame 0: lock onto left speaker
    const first = tracker.getTarget([leftSpeaker], 0);
    expect(first?.box.cx).toBeCloseTo(0.25);

    // Frame 2 (inside cooldown=8): left speaker blinks out, only right speaker visible -> MUST return null (hold)
    const duringCooldown = tracker.getTarget([rightSpeaker], 2);
    expect(duringCooldown).toBeNull();

    // Frame 10 (after cooldown=8): right speaker sustained -> switches cleanly
    const afterCooldown = tracker.getTarget([rightSpeaker], 10);
    expect(afterCooldown?.box.cx).toBeCloseTo(0.75);
  });

  it("SmoothedCameraman rejects single-frame outlier jumps (jump_confirm_frames=3) and snaps at scene cuts", () => {
    const cam = new OpenShortsSmoothedCameraman(3);
    cam.beginScene();
    // First detection after scene cut snaps immediately
    cam.updateTarget({ cx: 0.25, cy: 0.36, w: 0.1, h: 0.18, score: 1 });
    expect(cam.getCenter().x).toBeCloseTo(0.25);

    // Single-frame detector glitch jumping to x=0.82 is ignored
    cam.updateTarget({ cx: 0.82, cy: 0.36, w: 0.1, h: 0.18, score: 1 });
    expect(cam.targetCenterX).toBeCloseTo(0.25);

    // Three consecutive consistent detections at x=0.82 confirm the move
    cam.updateTarget({ cx: 0.81, cy: 0.36, w: 0.1, h: 0.18, score: 1 });
    cam.updateTarget({ cx: 0.82, cy: 0.36, w: 0.1, h: 0.18, score: 1 });
    expect(cam.targetCenterX).toBeCloseTo(0.82);
  });

  it("analyzeSplitScene accepts true two-shots using median centres and rejects alternating single-shots", () => {
    const left: FaceBox = { cx: 0.25, cy: 0.35, w: 0.08, h: 0.15, score: 0.012 };
    const right: FaceBox = { cx: 0.75, cy: 0.38, w: 0.08, h: 0.15, score: 0.012 };
    // True two-shot (coexistence >= 50%)
    const twoShot = analyzeSplitScene([[left, right], [left, right], [left, right], [left]]);
    expect(twoShot).not.toBeNull();
    expect(twoShot?.left.x).toBeCloseTo(0.25);
    expect(twoShot?.right.x).toBeCloseTo(0.75);

    // Shot-reverse-shot (alternating 1-person closeups) -> rejected!
    const alternating = analyzeSplitScene([[left], [right], [left], [right]]);
    expect(alternating).toBeNull();
  });

  it("analyzeSplitScene rejects a pair whose panels frame the same body", () => {
    // One person leaning across the shot: centres clear the 0.2 separation test, but the two boxes
    // overlap, so a 2-up would stack two copies of the same region.
    const a: FaceBox = { cx: 0.38, cy: 0.35, w: 0.3, h: 0.4, score: 0.05 };
    const b: FaceBox = { cx: 0.62, cy: 0.36, w: 0.3, h: 0.4, score: 0.05 };
    expect(analyzeSplitScene([[a, b], [a, b], [a, b], [a, b]])).toBeNull();
  });

  it("analyzeWideScene flags a landscape group shot that a 9:16 crop would slice apart", () => {
    const far: FaceBox = { cx: 0.12, cy: 0.4, w: 0.1, h: 0.18, score: 0.02 };
    const mid: FaceBox = { cx: 0.5, cy: 0.4, w: 0.1, h: 0.18, score: 0.02 };
    const near: FaceBox = { cx: 0.88, cy: 0.4, w: 0.1, h: 0.18, score: 0.02 };
    const group = [far, mid, near];
    expect(analyzeWideScene([group, group, group])).toBe(true);

    const solo: FaceBox = { cx: 0.5, cy: 0.4, w: 0.12, h: 0.2, score: 0.03 };
    expect(analyzeWideScene([[solo], [solo], [solo]])).toBe(false);
    // Too few samples to call a shot wide on.
    expect(analyzeWideScene([group, group])).toBe(false);
  });

  it("getConfiguredGeminiKeys deduplicates and caps the pool at 5 keys", () => {
    const keys = getConfiguredGeminiKeys([
      "AIzaSyKeyOne1234567890",
      "AIzaSyKeyTwo1234567890",
      "AIzaSyKeyOne1234567890", // duplicate
      "AIzaSyKeyThree1234567890",
      "AIzaSyKeyFour1234567890",
      "AIzaSyKeyFive1234567890",
      "AIzaSyKeySix1234567890",
    ]);
    expect(keys).toHaveLength(5);
    expect(keys[0]).toBe("AIzaSyKeyOne1234567890");
    expect(keys[4]).toBe("AIzaSyKeyFive1234567890");
  });
});

describe("Gemini word timestamp & clip candidate validation", () => {
  it("enforces monotonic non-zero word timestamps bounded by duration", () => {
    const cleaned = validateMonotonicWords(
      [
        { w: "Win", s: 0.5, e: 0.4 }, // invalid end < start
        { w: "$1,000,000", s: 0.3, e: 1.2 }, // out-of-order start
      ],
      10
    );
    expect(cleaned[0]?.s).toBeCloseTo(0.5);
    expect(cleaned[0]?.e).toBeGreaterThan(0.5);
    expect(cleaned[1]?.s).toBeGreaterThanOrEqual(cleaned[0]!.s);
  });

  it("removes overlapping duplicate clip candidates and clamps scores", () => {
    const makeClip = (title: string, hook: string, start: number, end: number, score: number, reason: string, hashtags: string[]): AIClip => ({
      title,
      hook,
      start,
      end,
      score,
      reason,
      hashtags,
      category: "general",
      editingKind: "general",
      layoutIntent: "uncertain",
      suggestedLayout: "auto",
      suggestedBroll: "",
      editingCues: [],
    });
    const clips = filterAndDedupeClips(
      [
        makeClip("Clip A", "Hook A", 10, 40, 95, "Top hook", ["#test"]),
        makeClip("Clip A Overlap", "Hook B", 12, 38, 88, "Duplicate", []),
        makeClip("Clip B", "Hook C", 50, 80, 82, "Second moment", []),
      ],
      120,
      20,
      45,
      5
    );
    expect(clips).toHaveLength(2);
    expect(clips.map((c) => c.title)).toEqual(["Clip A", "Clip B"]);
  });
});

describe("Opus styling helpers & Vyro compliance preflight", () => {
  it("detects gold power words and contextual emojis", () => {
    expect(isGoldKeyword("$1,000,000")).toBe(true);
    expect(isGoldKeyword("the")).toBe(false);
    expect(pageEmoji([{ w: "contestants", s: 1, e: 1.5 }])?.emoji).toBe("🏆");
  });

  it("evaluates Vyro campaign rules with PASS/WARNING/FAIL preflight", () => {
    const rules = {
      campaignName: "Beast Challenge",
      platforms: ["TikTok"],
      minDurationSec: 15,
      maxDurationSec: 60,
      requiredHashtags: ["#mrbeast"],
      requiredMentions: [],
      requiredDisclosure: "#ad",
      requiredCta: "",
      prohibitedPhrases: ["guaranteed payout"],
      rightsNotice: "Client owns derivatives.",
    };
    const clip: AIClip = {
      title: "Test",
      hook: "Win big",
      start: 0,
      end: 25,
      score: 90,
      reason: "Good",
      hashtags: ["#mrbeast"],
      category: "general",
      editingKind: "general",
      layoutIntent: "uncertain",
      suggestedLayout: "auto",
      suggestedBroll: "",
      editingCues: [],
    };
    const checks = evaluateCampaignCompliance(rules, clip, [{ w: "Hello", s: 0, e: 1 }], "abc1234567890def");
    expect(checks.every((c) => c.status === "PASS")).toBe(true);
  });

  it("reuses a digit-bearing discovery cue as the callout", () => {
    const cues = [{ kind: "quote", start: 4, end: 5, quote: "win $1,000,000" }];
    expect(numberCues([{ w: "win", s: 4, e: 5 }], cues)).toEqual(cues);
  });

  it("synthesises callouts from transcript tokens when the model gave none", () => {
    const words = [
      { w: "I", s: 10, e: 10.2 },
      { w: "need", s: 10.2, e: 10.4 },
      { w: "$", s: 10.4, e: 10.5 },
      { w: "6", s: 10.5, e: 10.7 },
      { w: "614", s: 10.7, e: 10.9 },
      { w: "dollars", s: 10.9, e: 11.3 },
      { w: "out", s: 11.3, e: 11.5 },
    ];
    expect(numberCues(words, undefined)).toEqual([{ kind: "number", start: 10.4, end: 10.9, quote: "$6,614" }]);
  });

  it("does not pop an intensifier or a bare integer as a stake callout", () => {
    const words = [
      { w: "I", s: 0, e: 0.4 },
      { w: "100", s: 0.4, e: 0.76 },
      { w: "%", s: 0.76, e: 1.18 },
      { w: "channel", s: 1.18, e: 1.66 },
      { w: "a", s: 1.66, e: 1.92 },
      { w: "mob", s: 1.92, e: 2.26 },
    ];
    expect(numberCues(words, undefined)).toEqual([]);
  });

  it("keeps number callouts sparse and ignores prose-only cues", () => {
    const words = [
      { w: "$1", s: 0, e: 0.5 },
      { w: "link", s: 0.6, e: 1 },
      { w: "$2", s: 1, e: 1.5 },
      { w: "bag", s: 1.6, e: 2 },
      { w: "$3", s: 20, e: 20.5 },
      { w: "cap", s: 20.6, e: 21 },
      { w: "$4", s: 40, e: 40.5 },
      { w: "gone", s: 40.6, e: 41 },
    ];
    const long = [{ kind: "quote", start: 0, end: 3, quote: "which means i need 6,614 dollars out of your prize pool" }];
    expect(numberCues(words, long).map((c) => c.start)).toEqual([0, 20, 40]);
    expect(numberCues([], long)).toEqual([]);
  });

  it("Rule 1 & 2 & 3: rejects edge bystanders on 1-up closeups, vetoes 2-up on 3-person desks, and routes 3-person shots to 3-Up Grid", () => {
    const center: FaceBox = { cx: 0.48, cy: 0.36, w: 0.14, h: 0.24, score: 0.033 };
    const edgePeek: FaceBox = { cx: 0.93, cy: 0.38, w: 0.06, h: 0.10, score: 0.006 };
    expect(filterEdgeBystanders([center, edgePeek])).toHaveLength(1);
    expect(analyzeSplitScene([[center, edgePeek], [center, edgePeek], [center, edgePeek]])).toBeNull();

    const p1: FaceBox = { cx: 0.2, cy: 0.36, w: 0.09, h: 0.16, score: 0.015 };
    const p2: FaceBox = { cx: 0.5, cy: 0.36, w: 0.09, h: 0.16, score: 0.015 };
    const p3: FaceBox = { cx: 0.8, cy: 0.36, w: 0.09, h: 0.16, score: 0.015 };
    // Vetoed for 2-Up Split so p2 is never duplicated in both panels
    expect(analyzeSplitScene([[p1, p2, p3], [p1, p2, p3], [p1, p2, p3]])).toBeNull();
    // Accepted for 3-Up Grid
    const grid3 = analyzeThreeUpScene([[p1, p2, p3], [p1, p2, p3], [p1, p2, p3]]);
    expect(grid3).not.toBeNull();
    expect(grid3?.top.x).toBeCloseTo(0.5);
    expect(grid3?.bottomLeft.x).toBeCloseTo(0.2);
    expect(grid3?.bottomRight.x).toBeCloseTo(0.8);
  });

  it("Rules 4, 6, 7: trims episode outros/credits and trailing unanswered questions", () => {
    const words = [
      { w: "just", s: 0.0, e: 0.3 },
      { w: "kind", s: 0.3, e: 0.6 },
      { w: "of", s: 0.6, e: 0.8 },
      { w: "like", s: 0.8, e: 1.1 },
      { w: "room.", s: 1.1, e: 1.6 },
      { w: "On", s: 2.2, e: 2.5 },
      { w: "three,", s: 2.5, e: 3.0 },
      { w: "the", s: 3.0, e: 3.3 },
      { w: "real", s: 3.3, e: 3.7 },
      { w: "ghost", s: 3.7, e: 4.2 },
      { w: "hunter", s: 4.2, e: 4.8 },
      { w: "steps", s: 4.8, e: 5.5 },
      { w: "forward!", s: 5.5, e: 6.5 },
      { w: "Michelle", s: 15.0, e: 16.0 },
      { w: "was", s: 16.0, e: 16.5 },
      { w: "born", s: 16.5, e: 17.0 },
      { w: "in", s: 17.0, e: 17.5 },
      { w: "1600s.", s: 17.5, e: 26.5 },
      { w: "Hey,", s: 27.0, e: 27.4 },
      { w: "number", s: 27.4, e: 27.8 },
      { w: "five,", s: 27.8, e: 28.2 },
      { w: "mechanic", s: 28.2, e: 28.6 },
      { w: "or", s: 28.6, e: 28.8 },
      { w: "ghost", s: 28.8, e: 29.1 },
      { w: "hunter?", s: 29.1, e: 29.5 },
    ];
    const trimmed = trimClipNarrativeBoundaries(0, 29.5, words, 60);
    // Snaps past "just kind of like room." to "On three..." at 2.2s, and cuts before "Hey, number five..." at ~26.7s
    expect(trimmed.start).toBeGreaterThanOrEqual(2.1);
    expect(trimmed.end).toBeLessThanOrEqual(26.8);
  });
});
