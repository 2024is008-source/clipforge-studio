import { describe, expect, it } from "vitest";
import {
  enforceLayoutHold,
  layoutTransitionSec,
  medianShotSec,
  tuneEdits,
  type EditSignature,
} from "@/lib/edit-tuning";
import type { TrackPoint } from "@/lib/reframe";

const sig = (over: Partial<EditSignature>): EditSignature => ({
  shotGaps: [],
  samples: 200,
  faceSamples: 0,
  twoFaceShare: 0,
  personShare: 0,
  objectShare: 0,
  medianFaceWidth: 0,
  blackShare: 0,
  motionEnergy: 0,
  subjectTravel: 0,
  ...over,
});

// Goldens taken from local_worker/tuning.py on the same signatures: the preview must not edit to a
// different rhythm than the export.
const cases: [string, EditSignature, Record<string, unknown>][] = [
  [
    "rapid multicam",
    sig({
      shotGaps: [1.0, 1.2, 0.9, 1.1],
      faceSamples: 190,
      twoFaceShare: 0.3,
      medianFaceWidth: 0.16,
      motionEnergy: 90,
      subjectTravel: 0.05,
    }),
    {
      style: "rapid_multicam",
      layoutHoldSec: 3.0,
      splitMinHoldSec: 3.0,
      cutSnapMinSec: 0.63,
      tauX: 0.3,
      maxSpeedX: 0.7,
      splitAllowed: true,
      punchGain: 1.0,
    },
  ],
  [
    "dialogue cuts",
    sig({
      shotGaps: [2.5, 3.0, 2.2, 2.8],
      faceSamples: 200,
      twoFaceShare: 0.5,
      medianFaceWidth: 0.22,
      motionEnergy: 40,
      subjectTravel: 0.02,
    }),
    {
      style: "dialogue_cuts",
      layoutHoldSec: 3.0,
      cutSnapMinSec: 1.59,
      tauX: 0.38,
      maxSpeedX: 0.55,
      splitAllowed: true,
      punchGain: 1.0,
    },
  ],
  [
    "long take",
    sig({
      shotGaps: [6.0, 7.5],
      faceSamples: 180,
      twoFaceShare: 0.05,
      medianFaceWidth: 0.3,
      motionEnergy: 12,
      subjectTravel: 0.03,
    }),
    {
      style: "long_take",
      layoutHoldSec: 5.0,
      cutSnapMinSec: 4.05,
      deadbandX: 0.108,
      tauX: 0.46,
      splitAllowed: false,
      punchGain: 0.75,
    },
  ],
  [
    "one continuous take",
    sig({ faceSamples: 200, medianFaceWidth: 0.34, motionEnergy: 4, subjectTravel: 0.01 }),
    {
      style: "static_single_shot",
      layoutHoldSec: 5.0,
      cutSnapMinSec: 0.35,
      deadbandY: 0.08,
      tauX: 0.55,
      maxSpeedX: 0.35,
      splitAllowed: false,
    },
  ],
  [
    "object-led action",
    sig({
      shotGaps: [1.1, 1.3],
      faceSamples: 20,
      objectShare: 0.4,
      motionEnergy: 120,
      subjectTravel: 0.22,
    }),
    { style: "action_object", tauX: 0.2296, maxSpeedX: 0.8625, splitAllowed: false },
  ],
  [
    "night footage",
    sig({
      shotGaps: [2.2, 2.6, 1.9],
      faceSamples: 60,
      blackShare: 0.45,
      twoFaceShare: 0.1,
      medianFaceWidth: 0.09,
      motionEnergy: 25,
    }),
    {
      style: "dialogue_cuts",
      deadbandX: 0.1,
      deadbandY: 0.08,
      tauX: 0.494,
      maxSpeedX: 0.506,
      maxSpeedY: 0.35,
      punchGain: 0.6,
    },
  ],
  [
    "histogram-only pre-pass",
    sig({ shotGaps: [2.5, 3.0, 2.2, 2.8], motionEnergy: 80, subjectsMeasured: false }),
    { style: "dialogue_cuts", tauX: 0.38, maxSpeedX: 0.55, splitAllowed: false },
  ],
];

describe("Per-video edit tuning matches the local worker", () => {
  for (const [name, input, golden] of cases) {
    it(`derives ${name} the same way Python does`, () => {
      const tuned = tuneEdits(input);
      for (const [key, value] of Object.entries(golden)) {
        const actual = tuned[key as keyof typeof tuned];
        if (typeof value === "number") expect(actual).toBeCloseTo(value, 3);
        else expect(actual).toBe(value);
      }
    });
  }

  it("does not let an unmeasured pre-pass decide the footage is action", () => {
    // No detector has run, so faceSamples is necessarily 0 and motion is all the pre-pass measured.
    const prePass = tuneEdits(
      sig({ shotGaps: [3.0, 4.0], motionEnergy: 40, subjectsMeasured: false }),
    );
    const measured = tuneEdits(sig({ shotGaps: [3.0, 4.0], faceSamples: 0, motionEnergy: 40 }));
    expect(prePass.style).toBe("dialogue_cuts");
    expect(measured.style).toBe("action_object");
  });
});

const pt = (
  t: number,
  mode: NonNullable<TrackPoint["mode"]>,
  extra?: Omit<TrackPoint, "t" | "mode">,
): TrackPoint => ({ t, x: 0.5, y: 0.4, ...extra, mode });

describe("Layout hold", () => {
  it("folds a split that does not survive the clip's own hold into the longer neighbour", () => {
    const track = [
      pt(0, "fill"),
      pt(1, "fill"),
      pt(2, "split"),
      pt(2.5, "split"),
      pt(3, "fill"),
      pt(4, "fill"),
      pt(5, "fill"),
    ];
    expect(enforceLayoutHold(track, 2.0)).toBe(2);
    expect(track.map((p) => p.mode)).toEqual(Array(7).fill("fill"));
  });

  it("carries the panel geometry when a run is folded into a split", () => {
    const track: TrackPoint[] = [
      pt(0, "split"),
      pt(3, "split"),
      pt(4, "fill"),
      pt(4.5, "fill"),
      pt(5, "split"),
      pt(12, "split"),
    ];
    track.forEach((p) => {
      p.x2 = p.mode === "split" ? 0.8 : p.x;
      p.y2 = p.mode === "split" ? 0.6 : p.y;
    });
    enforceLayoutHold(track, 2.0);
    expect(track[3]!.mode).toBe("split");
    expect(track[3]!.x2).toBe(0.8);
  });

  it("refuses a split that cannot hold the screen for three seconds", () => {
    const track = [
      pt(0, "fill"),
      pt(4, "fill"),
      pt(5, "split"),
      pt(6.8, "split"),
      pt(7, "fill"),
      pt(12, "fill"),
    ];
    // 1.8 s clears the general hold, so only the split's own floor can object to it.
    expect(enforceLayoutHold(track, 1.5)).toBe(0);
    expect(enforceLayoutHold(track, 1.5, 3.0)).toBe(2);
    expect(track.map((p) => p.mode)).toEqual(Array(6).fill("fill"));
  });

  it("keeps a split on screen instead of flashing a different template inside it", () => {
    const track: TrackPoint[] = [
      pt(0, "split"),
      pt(4, "split"),
      pt(5, "fill"),
      pt(5.5, "fill"),
      pt(6, "split"),
      pt(12, "split"),
    ];
    track.forEach((p) => {
      p.x2 = p.mode === "split" ? 0.8 : p.x;
      p.y2 = p.mode === "split" ? 0.6 : p.y;
    });
    enforceLayoutHold(track, 1.5, 3.0);
    expect(track.map((p) => p.mode)).toEqual(Array(6).fill("split"));
    // The borrowed frames aim at the pair the split already showed, not at their own subject.
    expect(track[2]!.x2).toBe(0.8);
    expect(track[3]!.y2).toBe(0.6);
  });

  it("folds a short split at the start of the clip into the shot that follows", () => {
    const track = [pt(0, "split"), pt(1, "split"), pt(2, "fill"), pt(10, "fill")];
    enforceLayoutHold(track, 1.5, 3.0);
    expect(track.map((p) => p.mode)).toEqual(Array(4).fill("fill"));
  });

  it("reads the rhythm back out of a track", () => {
    const track = [pt(0, "fill"), pt(2, "fill"), pt(4, "fill"), pt(8, "fill")];
    track[1]!.cut = true;
    track[2]!.cut = true;
    track[3]!.cut = true;
    expect(medianShotSec(track)).toBeCloseTo(3.0, 3);
    expect(layoutTransitionSec(track)).toBeCloseTo(0.39, 2);
    expect(layoutTransitionSec([pt(0, "fill")])).toBe(0.34);
  });
});
