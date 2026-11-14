import type { TrackPoint } from "./reframe";

/**
 * Per-video editing-style calibration. Mirrors local_worker/tuning.py.
 *
 * A source that changes shot every 1.1 s and a source that holds one frame for 20 s cannot be framed
 * well by the same hold, tolerance and pursuit speed, so measure the footage and derive the numbers
 * from its own rhythm. The worker sends these back with the track; the browser computes them itself
 * when it tracks locally, and the renderer uses them for transition timing either way.
 */

export type EditStyle =
  "rapid_multicam" | "dialogue_cuts" | "long_take" | "static_single_shot" | "action_object";

export type EditSignature = {
  shotGaps: number[];
  samples: number;
  faceSamples: number;
  twoFaceShare: number;
  personShare: number;
  objectShare: number;
  medianFaceWidth: number;
  blackShare: number;
  motionEnergy: number;
  subjectTravel: number;
  /** False in a pre-pass that only saw histograms: no detector has ruled on the subjects yet. */
  subjectsMeasured?: boolean;
};

export type EditTuning = {
  style: EditStyle;
  layoutHoldSec: number;
  splitMinHoldSec: number;
  cutSnapMinSec: number;
  deadbandX: number;
  deadbandY: number;
  tauX: number;
  tauY: number;
  maxSpeedX: number;
  maxSpeedY: number;
  splitAllowed: boolean;
  punchGain: number;
};

/**
 * Three seconds is the shortest time any composition — solo crop, 2-up, or the wide blurred field —
 * stays on screen, whatever the footage's own cut rhythm says. Below that a layout reads as a twitch
 * instead of a shot the editor chose.
 */
export const HOLD_MIN_SEC = 3.0;
export const HOLD_MAX_SEC = 5.0;
/**
 * A split states that two people matter simultaneously, which is the hardest composition to read, so
 * it keeps its own floor even if a long-rhythm source raises the general hold above it.
 */
export const SPLIT_MIN_HOLD_SEC = 3.0;

const TAU_X: Record<EditStyle, number> = {
  rapid_multicam: 0.3,
  dialogue_cuts: 0.38,
  long_take: 0.46,
  static_single_shot: 0.55,
  action_object: 0.28,
};
const SPEED_X: Record<EditStyle, number> = {
  rapid_multicam: 0.7,
  dialogue_cuts: 0.55,
  long_take: 0.45,
  static_single_shot: 0.35,
  action_object: 0.75,
};

const clamp = (v: number, low: number, high: number) => Math.max(low, Math.min(high, v));

/** Nothing measured: the calmest law the table offers, used when a pass could not sample the clip. */
export function emptySignature(): EditSignature {
  return {
    shotGaps: [],
    samples: 1,
    faceSamples: 0,
    twoFaceShare: 0,
    personShare: 0,
    objectShare: 0,
    medianFaceWidth: 0,
    blackShare: 0,
    motionEnergy: 0,
    subjectTravel: 0,
    subjectsMeasured: false,
  };
}

export function median(values: number[]): number {
  const s = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!s.length) return 0;
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** Seconds between the camera shots the track itself declares. */
export function shotGaps(track: TrackPoint[]): number[] {
  const cuts = track.filter((p) => p.cut);
  const gaps: number[] = [];
  for (let i = 1; i < cuts.length; i++) gaps.push(cuts[i]!.t - cuts[i - 1]!.t);
  return gaps.filter((g) => g > 0);
}

export function medianShotSec(track: TrackPoint[]): number {
  return median(shotGaps(track));
}

function styleFor(msl: number): EditStyle {
  if (msl <= 0) return "static_single_shot";
  if (msl < 1.6) return "rapid_multicam";
  if (msl <= 3.5) return "dialogue_cuts";
  if (msl <= 8) return "long_take";
  return "static_single_shot";
}

export function tuneEdits(sig: EditSignature): EditTuning {
  const msl = median(sig.shotGaps);
  const hold = msl > 0 ? clamp(msl * 0.8, HOLD_MIN_SEC, HOLD_MAX_SEC) : HOLD_MAX_SEC;
  const snapMin = msl > 0 ? clamp(msl * 0.6, 0.35, hold) : 0.35;

  const measured = sig.subjectsMeasured !== false;
  let style = styleFor(msl);
  // A shot with no readable face or body in it is carried by a thing or by motion. Only say that
  // once a detector has looked: a pre-pass has no subjects by definition, and treating that as
  // "action" would put every video on the fastest camera before anything was measured.
  if (measured && (sig.objectShare >= 0.15 || (sig.faceSamples === 0 && sig.motionEnergy > 18)))
    style = "action_object";

  const busy = measured && (sig.subjectTravel > 0.1 || sig.objectShare >= 0.15);
  const tight = measured && sig.medianFaceWidth < 0.1; // wides and long lenses need more travel
  // Night footage is where a tracker lies: it drops the subject for a second at a time, and a
  // camera that chases every flicker reads as jitter.
  const dark = sig.blackShare > 0.12;

  let tauX = TAU_X[style];
  if (busy) tauX *= 0.82;
  if (dark) tauX *= 1.3;
  let maxSpeedX = SPEED_X[style];
  if (tight) maxSpeedX *= 1.15;
  if (dark) maxSpeedX *= 0.8;

  let deadbandX = 0.08 * (style === "long_take" || style === "static_single_shot" ? 1.35 : 1);
  if (dark) deadbandX *= 1.25;

  return {
    style,
    layoutHoldSec: hold,
    splitMinHoldSec: Math.max(hold, SPLIT_MIN_HOLD_SEC),
    cutSnapMinSec: snapMin,
    deadbandX: clamp(deadbandX, 0.05, 0.16),
    deadbandY: dark || style === "static_single_shot" ? 0.08 : 0.05,
    tauX,
    tauY: style === "rapid_multicam" ? 0.45 : 0.55,
    maxSpeedX,
    maxSpeedY: dark ? 0.35 : 0.45,
    splitAllowed: sig.twoFaceShare >= 0.12 && sig.medianFaceWidth <= 0.22,
    punchGain: dark
      ? 0.6
      : style === "rapid_multicam" || style === "dialogue_cuts" || style === "action_object"
        ? 1
        : 0.75,
  };
}

/**
 * Reject layout changes that do not survive their own minimum on screen, folding each short run into
 * whichever neighbour lasts longer. Returns how many keyframes were rewritten.
 *
 * The worker's version of this rule may instead lengthen the short run by taking frames from a
 * neighbour, because it can check that the borrowed frames still show whoever was talking. Here the
 * only evidence is the tracker's own geometry, so the safe move is the one that removes the change:
 * a template that cannot hold the screen is not a template, it is a flicker.
 *
 * A split gets a longer floor than a crop: it states that two people matter simultaneously, and a
 * viewer needs time to read that state before it changes. A two-shot that cannot hold it is framed on
 * one subject rather than flickering between the two compositions.
 */
export function enforceLayoutHold(
  points: TrackPoint[],
  minHoldSec: number,
  splitMinHoldSec = 0,
): number {
  if (points.length < 3 || minHoldSec <= 0) return 0;
  const modeOf = (p: TrackPoint) => p.mode ?? "fill";
  const spans = () => {
    const out: [number, number][] = [];
    points.forEach((p, i) => {
      const last = out[out.length - 1];
      if (last && modeOf(points[last[1] - 1]!) === modeOf(p)) last[1] = i + 1;
      else out.push([i, i + 1]);
    });
    return out;
  };
  // A layout is on screen until the next keyframe changes it, so the run's last second belongs to the
  // run after it. Measuring to the run's own final keyframe instead called a one-sample twitch a
  // four-second shot and left it untouched.
  const dur = ([begin, end]: [number, number]) =>
    (points[Math.min(end, points.length - 1)]!.t) - points[begin]!.t;
  const splitFloor = splitMinHoldSec > minHoldSec ? splitMinHoldSec : minHoldSec;
  const floorOf = (run: [number, number]) =>
    modeOf(points[run[0]]!) === "split" ? splitFloor : minHoldSec;

  let changed = 0;
  for (;;) {
    const runs = spans();
    if (runs.length < 2) break;
    const short = runs.findIndex((run) => dur(run) < floorOf(run));
    if (short < 0) break;
    // A run at either end of the clip has one neighbour to give itself to; an interior run joins
    // whichever side already holds the screen longer.
    const prevIdx = short > 0 ? short - 1 : null;
    const nextIdx = short < runs.length - 1 ? short + 1 : null;
    const keeperIdx =
      prevIdx === null ? nextIdx! : nextIdx === null ? prevIdx : dur(runs[prevIdx]!) >= dur(runs[nextIdx]!) ? prevIdx : nextIdx;
    const keeper = points[runs[keeperIdx]![0]]!;
    const keep = modeOf(keeper);
    for (let i = runs[short]![0]; i < runs[short]![1]; i++) {
      const pt = points[i]!;
      pt.mode = keep;
      if (keep === "split") {
        // A folded-in frame has to adopt the pair it is joining; its own x2 is its own subject,
        // which would put both panels on the same person.
        pt.x = keeper.x;
        pt.y = keeper.y;
        pt.x2 = keeper.x2 ?? keeper.x;
        pt.y2 = keeper.y2 ?? keeper.y;
      }
      changed++;
    }
  }
  return changed;
}

/**
 * How long a layout slide takes on this source. A third of a second reads as an edit on a two-second
 * shot and as a smear on a one-second one, so scale it with the rhythm the video actually cuts to.
 */
export function layoutTransitionSec(track: TrackPoint[]): number {
  const msl = medianShotSec(track);
  if (!msl) return 0.34;
  return clamp(msl * 0.13, 0.2, 0.45);
}

/**
 * How hard to punch in on a beat, for the browser-only word-emphasis fallback (the worker delivers
 * its zooms already scaled). A track carries no luma, so the damping the worker applies to night
 * footage is not available here.
 */
export function punchGainOf(track: TrackPoint[]): number {
  const style = styleFor(medianShotSec(track));
  return style === "rapid_multicam" || style === "dialogue_cuts" || style === "action_object"
    ? 1
    : 0.75;
}
