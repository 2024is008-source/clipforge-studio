// OpenShorts-Powered Reframe Engine v2 (Browser + Local Worker):
// - SpeakerTracker (cooldown_frames=30, 3.0x sticky hysteresis, associateFaces 2D velocity identity matching)
// - SmoothedCameraman ("Heavy Tripod" 25% safe-zone deadband, jump_confirm_frames=3 outlier rejection, begin_scene cut snap)
// - split_layout.analyzeScene (coexistence>=50%, separation>=20%, median left/right centres, independent_split IoU<=0.15)
// - active_speaker (mouth-region pixel activity 20th-80th percentile normalization)
// - Multi-stage Face (MediaPipe BlazeFace) + Portrait/Body/Action Object Cascade (EfficientDet/YOLOv8)
import type {
  FaceDetector,
  FaceDetectorResult,
  ObjectDetector,
  ObjectDetectorResult,
} from "@mediapipe/tasks-vision";
import {
  emptySignature,
  enforceLayoutHold,
  median,
  tuneEdits,
  type EditTuning,
} from "./edit-tuning";

export type FrameLayoutMode = "fill" | "split" | "grid3" | "fit" | "editorial";
export type TrackingEngineMode =
  "openshorts_reframe_v2" | "hybrid_yolo_bytetrack" | "mediapipe_fast";

export type TrackPoint = {
  t: number;
  x: number;
  y: number;
  trackId?: number | undefined;
  x2?: number | undefined;
  y2?: number | undefined;
  trackId2?: number | undefined;
  x3?: number | undefined;
  y3?: number | undefined;
  mode?: FrameLayoutMode | undefined;
  cut?: boolean | undefined;
  subjectType?: ("face" | "person" | "object" | "wide") | undefined;
  zoom?: number | undefined;
  hasBottomGraphic?: boolean | undefined;
  isFlashback?: boolean | undefined;
  trackingConfidence?: number | undefined;
  layoutDecision?: { scene: number; reason: string } | undefined;
};

/**
 * What the GPU worker decided about who had to be on screen, and whether the delivered track obeyed.
 * `main_lock` is the number that matters: of the keyframes where the rule demanded the main speaker,
 * how many actually kept his head in frame.
 */
export type SpeakerReport = {
  main_speaker?: number | null;
  windows?: number;
  overlap_windows?: number;
  ambiguous_windows?: number;
  silent_windows?: number;
  required?: { floor?: number; presumed_main?: number; silent?: number };
  shot?: Record<string, number>;
  hold_exempt_runs?: number;
  main_lock?: {
    required_samples?: number;
    on_screen?: number;
    coverage?: number | null;
    misses?: number[];
  };
  coverage_after?: { overall?: number | null; talking_samples?: number };
  coverage_before?: { overall?: number | null; talking_samples?: number };
};

type Vision = { faceDet: FaceDetector; objDet: ObjectDetector | null; delegate: "GPU" | "CPU" };
type DetectionTask<R> = {
  detectForVideo: (v: HTMLVideoElement, ts: number) => R;
  close: () => void;
};

const VISION_ASSETS = "/vision";

let visionPromise: Promise<{
  faceDet: DetectionTask<FaceDetectorResult>;
  objDet: DetectionTask<ObjectDetectorResult> | null;
  delegate: "GPU" | "CPU";
}> | null = null;

let visionNotice: string | null = null;
let visionError: string | null = null;
const delegateListeners = new Set<(delegate: string) => void>();

function publishDelegate(vision: Vision) {
  const label = `${vision.delegate} (${vision.objDet ? "Face + Person/Object" : "Face"})`;
  delegateListeners.forEach((listen) => listen(label));
}

export function onVisionDelegateChange(listener: (delegate: string) => void) {
  delegateListeners.add(listener);
  return () => delegateListeners.delete(listener);
}

export function lastVisionNotice() {
  return visionNotice;
}

export function lastVisionError() {
  return visionError;
}

// A failed fileset load rejects with the <script> element's error Event, not an Error, so the
// naive String(err) reported "[object Event]" and hid which asset was actually missing.
function describeVisionFailure(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  const src = (err as { target?: { src?: string } } | null)?.target?.src;
  return src ? `could not load ${src}` : "the vision runtime could not be loaded";
}

/**
 * A GPU delegate can be constructed successfully and still throw on the first real frame — weak
 * Intel/AMD drivers and software rasterisers do exactly that. The frame that fails is reported as
 * empty and the CPU pair is built in the background, so tracking continues instead of aborting.
 */
function resilient<R extends { detections: unknown[] }>(
  pick: (v: Vision) => DetectionTask<R> | null,
  state: { current: Vision },
  recover: () => Promise<void>,
  empty: R,
): DetectionTask<R> {
  return {
    detectForVideo(video, timestamp) {
      const task = pick(state.current);
      if (!task) return empty;
      try {
        return task.detectForVideo(video, timestamp);
      } catch (err) {
        if (state.current.delegate !== "GPU") throw err;
        visionNotice = `GPU vision failed mid-run (${String(err)}); continuing on CPU.`;
        void recover();
        return empty;
      }
    },
    close() {
      pick(state.current)?.close();
    },
  };
}

export function getFaceDetector() {
  const pending = (visionPromise ??= (async () => {
    const { FilesetResolver, FaceDetector, ObjectDetector } =
      await import("@mediapipe/tasks-vision");
    // Served from public/vision so the WASM binary is always the one shipped by the installed
    // package. A CDN `@latest` fileset pairs a newer binary with older JS glue, and any blocked
    // request used to leave vision permanently unavailable on a machine that is meant to work offline.
    const fs = await FilesetResolver.forVisionTasks(`${VISION_ASSETS}/wasm`);
    const faceModel = `${VISION_ASSETS}/models/blaze_face_short_range.tflite`;
    const objModel = `${VISION_ASSETS}/models/efficientdet_lite0.tflite`;
    const build = async (delegate: "GPU" | "CPU"): Promise<Vision> => {
      const faceDet = await FaceDetector.createFromOptions(fs, {
        baseOptions: { modelAssetPath: faceModel, delegate },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.45,
      });
      let objDet: ObjectDetector | null = null;
      try {
        objDet = await ObjectDetector.createFromOptions(fs, {
          baseOptions: { modelAssetPath: objModel, delegate },
          runningMode: "VIDEO",
          scoreThreshold: 0.35,
          maxResults: 8,
        });
      } catch {
        objDet = null;
      }
      return { faceDet, objDet, delegate };
    };

    let initial: Vision;
    try {
      initial = await build("GPU");
    } catch {
      initial = await build("CPU");
    }
    const state = { current: initial };
    let recovering: Promise<void> | null = null;
    const recover = () =>
      (recovering ??= build("CPU")
        .then((next) => {
          const previous = state.current;
          state.current = next;
          previous.faceDet.close();
          previous.objDet?.close();
          publishDelegate(next);
        })
        .catch(() => {
          recovering = null;
        }));

    publishDelegate(initial);
    visionError = null;
    return {
      faceDet: resilient((v) => v.faceDet, state, recover, { detections: [] }),
      objDet: state.current.objDet
        ? resilient((v) => v.objDet, state, recover, { detections: [] })
        : null,
      get delegate() {
        return state.current.delegate;
      },
    };
  })().catch((err) => {
    visionError = describeVisionFailure(err);
    visionPromise = null;
    throw err;
  }));
  return pending.then((v) => ({ det: v.faceDet, objDet: v.objDet, delegate: v.delegate }));
}

function seek(v: HTMLVideoElement, t: number, signal?: AbortSignal) {
  return new Promise<void>((res, rej) => {
    if (signal?.aborted) {
      rej(new Error("aborted"));
      return;
    }
    const timer = setTimeout(() => {
      v.removeEventListener("seeked", on);
      res();
    }, 1200);
    const on = () => {
      clearTimeout(timer);
      v.removeEventListener("seeked", on);
      res();
    };
    v.addEventListener("seeked", on);
    v.currentTime = t;
  });
}

export type FrameSignature = {
  hist: Float32Array;
  lumaGrid: Float32Array;
  meanLuma: number;
  meanSat?: number;
  hasBottomGraphic?: boolean;
};

export function computeFrameSignature(
  ctx: CanvasRenderingContext2D,
  v: HTMLVideoElement,
): FrameSignature {
  ctx.drawImage(v, 0, 0, 16, 9);
  const data = ctx.getImageData(0, 0, 16, 9).data;
  const hist = new Float32Array(48);
  const lumaGrid = new Float32Array(16 * 9);
  const totalPixels = 16 * 9;
  let lumaSum = 0;
  let satSum = 0;
  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4]!;
    const g = data[i * 4 + 1]!;
    const b = data[i * 4 + 2]!;
    hist[r >> 4]! += 1 / totalPixels;
    hist[16 + (g >> 4)]! += 1 / totalPixels;
    hist[32 + (b >> 4)]! += 1 / totalPixels;
    const luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    lumaGrid[i] = luma;
    lumaSum += luma * 255;
    const cmax = Math.max(r, g, b);
    const cmin = Math.min(r, g, b);
    satSum += cmax > 0 ? ((cmax - cmin) / cmax) * 255 : 0;
  }
  // Rule 10: Check lower-third rows (y=6..8, x=3..12) for high-contrast burned-in source graphics
  let botContrast = 0;
  let botPeak = 0;
  for (let y = 6; y <= 7; y++) {
    for (let x = 3; x <= 12; x++) {
      const idx = y * 16 + x;
      const lum = lumaGrid[idx]!;
      const nextLum = lumaGrid[idx + 1]!;
      botContrast += Math.abs(lum - nextLum);
      if (lum > botPeak) botPeak = lum;
    }
  }
  return {
    hist,
    lumaGrid,
    meanLuma: lumaSum / totalPixels,
    meanSat: satSum / totalPixels,
    hasBottomGraphic: botContrast / 20 > 0.18 && botPeak > 0.82,
  };
}

export function detectSceneCut(
  prev: FrameSignature | null,
  curr: FrameSignature,
  timeSinceLastCut: number,
  minCutInterval = 0.4,
): boolean {
  if (!prev) return true;
  if (timeSinceLastCut < minCutInterval) return false;

  let histDiff = 0;
  for (let i = 0; i < 48; i++) histDiff += Math.abs(curr.hist[i]! - prev.hist[i]!);
  histDiff /= 3;

  let lumaDiff = 0;
  for (let i = 0; i < curr.lumaGrid.length; i++)
    lumaDiff += Math.abs(curr.lumaGrid[i]! - prev.lumaGrid[i]!);
  lumaDiff /= curr.lumaGrid.length;

  // Catches both outdoor/high-contrast cuts and indoor studio/grocery camera angle cuts under constant lighting,
  // while still rejecting uniform brightness flashes where histDiff < 0.12.
  return (
    (histDiff >= 0.35 && lumaDiff >= 0.15) ||
    (histDiff >= 0.24 && lumaDiff >= 0.11) ||
    (histDiff >= 0.3 && lumaDiff >= 0.08)
  );
}

export type FaceBox = {
  cx: number;
  cy: number;
  w: number;
  h: number;
  score: number;
  kind?: "face" | "person" | "object";
  mouthActivity?: number;
};

export function boxIoU(a: FaceBox, b: FaceBox): number {
  const ax1 = a.cx - a.w / 2,
    ay1 = a.cy - a.h / 2,
    ax2 = a.cx + a.w / 2,
    ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2,
    by1 = b.cy - b.h / 2,
    bx2 = b.cx + b.w / 2,
    by2 = b.cy + b.h / 2;
  const ix1 = Math.max(ax1, bx1),
    iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2),
    iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1),
    ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * OpenShorts split_layout.dedupe_faces:
 * Drop duplicate detections of the same face (IoU > 0.30), keeping the largest.
 */
export function dedupeFaces(candidates: FaceBox[], maxOverlap = 0.3): FaceBox[] {
  const sorted = [...candidates].sort((a, b) => b.w * b.h - a.w * a.h);
  const kept: FaceBox[] = [];
  for (const c of sorted) {
    if (kept.every((k) => boxIoU(c, k) <= maxOverlap)) {
      kept.push(c);
    }
  }
  return kept;
}

/**
 * OpenShorts framing_geometry.needs_context:
 * True when cropping to a single 9:16 strip would slice off another primary face or person
 * (w >= 4.5% of frame width and area >= 35% of largest subject).
 */
export function needsContext(candidates: FaceBox[], cropWidthNorm = (9 / 16) * (9 / 16)): boolean {
  if (!candidates.length) return false;
  const largest = Math.max(...candidates.map((c) => c.w * c.h));
  const primary = candidates.filter((c) => c.w >= 0.045 && c.w * c.h >= largest * 0.35);
  if (primary.length <= 1) return false;
  const minX = Math.min(...primary.map((p) => p.cx - p.w / 2));
  const maxX = Math.max(...primary.map((p) => p.cx + p.w / 2));
  return maxX - minX > cropWidthNorm * 0.95;
}

/**
 * OpenShorts SpeakerTracker (ported from main.py & framing_geometry.associate_faces):
 * - 2D velocity-predicted identity matching (survives short occlusions/crossings).
 * - 3.0x sticky hysteresis bonus for the locked active speaker.
 * - cooldownFrames (default 8 samples ≈ 1.1s at 7Hz, equivalent to 30 frames at 30fps):
 *   if the active speaker blinks out or turns their head for a frame, returns null (holding
 *   the camera steady) rather than whipping to a bystander!
 */
export class OpenShortsSpeakerTracker {
  activeSpeakerId: number | null = null;
  speakerScores = new Map<number, number>();
  lastSwitchStep = -1000;
  nextId = 1;
  knownFaces: {
    id: number;
    cx: number;
    cy: number;
    area: number;
    lastStep: number;
    vx: number;
    vy: number;
  }[] = [];

  constructor(
    public cooldownSteps = 8,
    public ttlSteps = 8,
  ) {}

  reset(): void {
    this.activeSpeakerId = null;
    this.speakerScores.clear();
    this.lastSwitchStep = -1000;
    this.knownFaces = [];
  }

  getTarget(candidates: FaceBox[], stepIndex: number): { box: FaceBox; id: number } | null {
    const deduped = dedupeFaces(candidates);
    this.knownFaces = this.knownFaces.filter((k) => stepIndex - k.lastStep <= this.ttlSteps);

    type Pair = {
      cost: number;
      pid: number;
      idx: number;
      old: OpenShortsSpeakerTracker["knownFaces"][number];
    };
    const pairs: Pair[] = [];

    deduped.forEach((face, idx) => {
      const area = Math.max(1e-5, face.w * face.h);
      for (const old of this.knownFaces) {
        const dt = Math.max(1, stepIndex - old.lastStep);
        const px = old.cx + old.vx * Math.min(dt, this.ttlSteps * 0.4);
        const py = old.cy + old.vy * Math.min(dt, this.ttlSteps * 0.4);
        const dist = Math.hypot(face.cx - px, face.cy - py);
        const ratio = Math.max(area, old.area) / Math.max(1e-6, Math.min(area, old.area));
        if (dist < 0.16 && ratio <= 3.0) {
          pairs.push({ cost: dist + Math.abs(Math.log(ratio)) * 0.025, pid: old.id, idx, old });
        }
      }
    });

    pairs.sort((a, b) => a.cost - b.cost);
    const assigned = new Map<number, (typeof this.knownFaces)[number]>();
    const usedIds = new Set<number>();
    for (const p of pairs) {
      if (!assigned.has(p.idx) && !usedIds.has(p.pid)) {
        assigned.set(p.idx, p.old);
        usedIds.add(p.pid);
      }
    }

    const updatedKnown = this.knownFaces.filter((k) => !usedIds.has(k.id));
    const current: { box: FaceBox; id: number }[] = [];

    deduped.forEach((face, idx) => {
      const old = assigned.get(idx);
      const pid = old ? old.id : this.nextId++;
      const dt = old ? Math.max(1, stepIndex - old.lastStep) : 1;
      const vx = old ? Math.max(-0.15, Math.min(0.15, (face.cx - old.cx) / dt)) : 0;
      const vy = old ? Math.max(-0.15, Math.min(0.15, (face.cy - old.cy) / dt)) : 0;
      updatedKnown.push({
        id: pid,
        cx: face.cx,
        cy: face.cy,
        area: Math.max(1e-5, face.w * face.h),
        lastStep: stepIndex,
        vx,
        vy,
      });
      current.push({ box: face, id: pid });
    });

    this.knownFaces = updatedKnown;
    const aliveIds = new Set(this.knownFaces.map((k) => k.id));
    for (const [pid, score] of this.speakerScores.entries()) {
      if (!aliveIds.has(pid)) {
        this.speakerScores.delete(pid);
      } else {
        const decayed = score * 0.85;
        if (decayed < 0.05) this.speakerScores.delete(pid);
        else this.speakerScores.set(pid, decayed);
      }
    }

    for (const cand of current) {
      const mouthBoost = 1 + (cand.box.mouthActivity ?? 0) * 1.5;
      const rawScore = ((cand.box.w * cand.box.h) / 0.05) * mouthBoost;
      this.speakerScores.set(cand.id, (this.speakerScores.get(cand.id) ?? 0) + rawScore);
    }

    if (!current.length) return null;

    let best: { box: FaceBox; id: number } | null = null;
    let maxScore = -1;
    for (const cand of current) {
      let total = this.speakerScores.get(cand.id) ?? 0;
      if (cand.id === this.activeSpeakerId) total *= 3.0; // OpenShorts 3.0x sticky hysteresis
      if (total > maxScore) {
        maxScore = total;
        best = cand;
      }
    }

    if (!best) return null;
    if (best.id === this.activeSpeakerId) {
      return best;
    }

    // OpenShorts Cooldown Gate: hold current camera position if within cooldown
    if (stepIndex - this.lastSwitchStep < this.cooldownSteps) {
      const oldStillVisible = current.find((c) => c.id === this.activeSpeakerId);
      return oldStillVisible ?? null;
    }

    this.activeSpeakerId = best.id;
    this.lastSwitchStep = stepIndex;
    return best;
  }
}

/**
 * OpenShorts SmoothedCameraman ("Heavy Tripod" + jump_confirm_frames=3 outlier rejection + cut snap):
 * - Ignores single-sample target jumps outside the 25% safe-zone unless confirmed for `jumpConfirmFrames`
 *   consecutive detections in the same neighborhood.
 * - Immediately snaps to the new subject at a scene cut (`beginScene()`).
 */
/**
 * Camera pursuit tuning, in normalized-source units per second so the preview moves at the same
 * rate as the worker export regardless of how often frames are sampled.
 * Mirrors CAM_* in local_worker/openshorts_engine.py.
 */
export const CAM_DEADBAND_X = 0.08;
export const CAM_DEADBAND_Y = 0.05;
export const CAM_TAU_X = 0.38;
export const CAM_TAU_Y = 0.55;
export const CAM_MAX_SPEED_X = 0.55;
export const CAM_MAX_SPEED_Y = 0.45;
/** Floor between two detected shot changes, however calm the source looks. */
const SHOT_FLOOR_SEC = 0.45;
/** Smallest face width (fraction of source width) worth framing or splitting on. Mirrors MIN_FACE_WIDTH. */
const MIN_READABLE_FACE_W = 0.045;

export class OpenShortsSmoothedCameraman {
  currentCenterX = 0.5;
  targetCenterX = 0.5;
  currentCenterY = 0.38;
  targetCenterY = 0.38;
  cropWidthNorm = 0.3164; // (9/16) / (16/9) on 16:9 source
  cropHeightNorm = 1.0; // a 9:16 crop on a 16:9 source spans the whole source height
  deadbandX = CAM_DEADBAND_X;
  deadbandY = CAM_DEADBAND_Y;
  tauX = CAM_TAU_X;
  tauY = CAM_TAU_Y;
  maxSpeedX = CAM_MAX_SPEED_X;
  maxSpeedY = CAM_MAX_SPEED_Y;
  safeZoneRadius = 0.3164 * CAM_DEADBAND_X;
  trackingSafeZoneRadius = 0.3164 * CAM_DEADBAND_X;
  sampleDt = 0.14;
  private pendingTarget: number | null = null;
  private pendingCount = 0;
  private snapPending = true;

  constructor(public jumpConfirmFrames = 3) {}

  setSampleInterval(dt: number): void {
    this.sampleDt = Math.max(1e-3, dt);
  }

  /** Take the pursuit law from this video's measured editing rhythm instead of the defaults. */
  applyTuning(t: EditTuning): void {
    this.deadbandX = t.deadbandX;
    this.deadbandY = t.deadbandY;
    this.tauX = t.tauX;
    this.tauY = t.tauY;
    this.maxSpeedX = t.maxSpeedX;
    this.maxSpeedY = t.maxSpeedY;
    this.safeZoneRadius = this.cropWidthNorm * t.deadbandX;
    this.trackingSafeZoneRadius = this.safeZoneRadius;
  }

  /** Exponential approach capped by a speed limit that grows with the error; cannot overshoot. */
  private pursue(error: number, span: number, tau: number, maxSpeed: number): number {
    const dt = this.sampleDt;
    const eased = Math.abs(error) * (1 - Math.exp(-dt / tau));
    const limit =
      span * maxSpeed * dt * (1 + 3 * Math.min(1.5, Math.abs(error) / Math.max(1e-3, span)));
    return Math.sign(error) * Math.min(Math.abs(error), eased, limit);
  }

  beginScene(): void {
    this.pendingTarget = null;
    this.pendingCount = 0;
    this.snapPending = true;
    this.trackingSafeZoneRadius = this.safeZoneRadius;
  }

  updateTarget(face: FaceBox | null): void {
    if (!face) return;
    const newCenter = Math.max(0.08, Math.min(0.92, face.cx));
    // OpenShorts subject_position: place head in upper third (0.34 anchor) with chin/crown bounds
    const newY = Math.max(0.12, Math.min(0.85, face.cy));
    this.trackingSafeZoneRadius = Math.min(
      this.safeZoneRadius,
      Math.max(0.015, this.cropWidthNorm / 2 - face.w / 2 - this.cropWidthNorm * 0.08),
    );

    if (this.snapPending) {
      this.snapPending = false;
      this.pendingTarget = null;
      this.pendingCount = 0;
      this.targetCenterX = newCenter;
      this.currentCenterX = newCenter;
      this.targetCenterY = newY;
      this.currentCenterY = newY;
      return;
    }

    if (Math.abs(newCenter - this.targetCenterX) > this.safeZoneRadius) {
      if (
        this.pendingTarget !== null &&
        Math.abs(newCenter - this.pendingTarget) <= this.safeZoneRadius
      ) {
        this.pendingCount += 1;
      } else {
        this.pendingTarget = newCenter;
        this.pendingCount = 1;
      }
      if (this.pendingCount < this.jumpConfirmFrames) {
        return; // Hold frame: reject unconfirmed detector jump!
      }
    }

    this.pendingTarget = null;
    this.pendingCount = 0;
    this.targetCenterX = newCenter;
    this.targetCenterY = newY;
  }

  getCenter(forceSnap = false): { x: number; y: number } {
    if (forceSnap) {
      this.currentCenterX = this.targetCenterX;
      this.currentCenterY = this.targetCenterY;
    } else {
      const diff = this.targetCenterX - this.currentCenterX;
      if (Math.abs(diff) > this.trackingSafeZoneRadius) {
        const dir = diff > 0 ? 1 : -1;
        this.currentCenterX += this.pursue(diff, this.cropWidthNorm, this.tauX, this.maxSpeedX);
        const newDiff = this.targetCenterX - this.currentCenterX;
        if ((dir === 1 && newDiff < 0) || (dir === -1 && newDiff > 0))
          this.currentCenterX = this.targetCenterX;
      }
      const diffY = this.targetCenterY - this.currentCenterY;
      if (Math.abs(diffY) > this.cropHeightNorm * this.deadbandY) {
        this.currentCenterY += this.pursue(diffY, this.cropHeightNorm, this.tauY, this.maxSpeedY);
      }
    }
    return {
      x: Math.max(0.08, Math.min(0.92, this.currentCenterX)),
      y: Math.max(0.12, Math.min(0.85, this.currentCenterY)),
    };
  }
}

/**
 * OpenShorts split_layout.analyze_scene:
 * Decides whether a scene's sampled frames show a true stackable two-shot:
 * - Requires >= 50% coexistence of 2 primary faces (w >= 4.5% of frame width)
 * - Horizontal separation >= 20% of frame width
 * - Computes MEDIAN (cx, cy) for left and right speakers so a single stray detection never drags the split crop.
 */
/**
 * Rule 1 (Edge-Bystander Rejection):
 * When a central speaker dominates a close-up in a lineup (e.g. WA0029 0:01-0:02, WA0030 0:09-0:13),
 * ignore partial neighbor faces in the outer 12% margins (cx < 0.12 or cx > 0.88) whose area is < 55%
 * of the primary speaker's face area.
 */
export function filterEdgeBystanders(candidates: FaceBox[]): FaceBox[] {
  if (candidates.length <= 1) return candidates;
  const largestArea = Math.max(...candidates.map((c) => c.w * c.h));
  const hasCentralPrimary = candidates.some(
    (c) => c.cx >= 0.12 && c.cx <= 0.88 && c.w * c.h >= largestArea * 0.65,
  );
  if (!hasCentralPrimary) return candidates;
  const filtered = candidates.filter((c) => {
    const inEdgeMargin = c.cx < 0.12 || c.cx > 0.88;
    const clippedAtBorder = c.cx - c.w / 2 <= 0.015 || c.cx + c.w / 2 >= 0.985;
    if ((inEdgeMargin || clippedAtBorder) && c.w * c.h < largestArea * 0.55) {
      return false;
    }
    return true;
  });
  return filtered.length ? filtered : candidates;
}

export function analyzeSplitScene(
  shotSamples: FaceBox[][],
): { left: { x: number; y: number }; right: { x: number; y: number } } | null {
  if (shotSamples.length < 3) return null;
  const lefts: FaceBox[] = [];
  const rights: FaceBox[] = [];

  for (const raw of shotSamples) {
    const big = filterEdgeBystanders(dedupeFaces(raw)).filter((c) => c.w >= MIN_READABLE_FACE_W);
    if (big.length < 2) continue;
    const largestArea = Math.max(...big.map((c) => c.w * c.h));
    const primary = big.filter((c) => c.w * c.h >= largestArea * 0.35);
    if (primary.length !== 2) continue;
    primary.sort((a, b) => b.score - a.score);
    const pair = [primary[0]!, primary[1]!].sort((a, b) => a.cx - b.cx);
    if (pair[1]!.cx - pair[0]!.cx < 0.22) continue;
    // Rule 2 (Middle-Subject Veto): reject 2-Up Split if a 3rd readable face sits between left and right
    const hasMiddleBystander = big.some(
      (other) =>
        other !== pair[0] &&
        other !== pair[1] &&
        other.cx > pair[0]!.cx + 0.08 &&
        other.cx < pair[1]!.cx - 0.08,
    );
    if (hasMiddleBystander) continue;
    lefts.push(pair[0]!);
    rights.push(pair[1]!);
  }

  if (lefts.length < 0.5 * shotSamples.length) return null;

  const median = (arr: number[]) => {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
  };

  const leftMedian = { x: median(lefts.map((c) => c.cx)), y: median(lefts.map((c) => c.cy)) };
  const rightMedian = { x: median(rights.map((c) => c.cx)), y: median(rights.map((c) => c.cy)) };

  // Centre separation is not independence: one person leaning across the shot clears 0.2 while both
  // panels still frame the same body, and the export reads as a duplicated layer. The worker rejects
  // this with `independent_split`; the browser has to refuse it the same way or the two paths disagree.
  const lBox: FaceBox = {
    cx: leftMedian.x,
    cy: leftMedian.y,
    w: median(lefts.map((c) => c.w)),
    h: median(lefts.map((c) => c.h)),
    score: 0,
  };
  const rBox: FaceBox = {
    cx: rightMedian.x,
    cy: rightMedian.y,
    w: median(rights.map((c) => c.w)),
    h: median(rights.map((c) => c.h)),
    score: 0,
  };
  if (boxIoU(lBox, rBox) > 0.15) return null;
  // Each panel must actually contain its own subject and not the other one.
  const covers = (p: FaceBox, q: FaceBox) => {
    const ix = Math.max(
      0,
      Math.min(p.cx + p.w / 2, q.cx + q.w / 2) - Math.max(p.cx - p.w / 2, q.cx - q.w / 2),
    );
    const iy = Math.max(
      0,
      Math.min(p.cy + p.h / 2, q.cy + q.h / 2) - Math.max(p.cy - p.h / 2, q.cy - q.h / 2),
    );
    return (ix * iy) / Math.max(1e-6, q.w * q.h);
  };
  if (covers(lBox, rBox) > 0.1 || covers(rBox, lBox) > 0.1) return null;

  // Reject if either speaker walks/crosses more than 12% across the frame during the shot
  const leftMisses = lefts.filter((c) => Math.abs(c.cx - leftMedian.x) > 0.12).length;
  const rightMisses = rights.filter((c) => Math.abs(c.cx - rightMedian.x) > 0.12).length;
  if (
    leftMisses > Math.max(1, lefts.length * 0.15) ||
    rightMisses > Math.max(1, rights.length * 0.15)
  ) {
    return null;
  }

  return { left: leftMedian, right: rightMedian };
}

/**
 * Does this shot need the whole frame? `fill` slices a 9:16 strip out of a wide shot, so a group of
 * three or more people loses two of them. `needsContext` measures exactly that on one frame; a
 * single stray detection must not flip a whole shot to a letterbox, so the condition has to hold for
 * most of the shot's samples before the template changes.
 */
/**
 * Rule 3 (3-Up Split Grid for 3-Person Lineup / Desk Shots):
 * Detects shots where 3 primary speakers sit/stand side-by-side with >= 18% spacing so they can be
 * framed in a 3-Up Grid (1 Top + 2 Side-by-Side Bottom) without duplicating middle panelists.
 */
export function analyzeThreeUpScene(shotSamples: FaceBox[][]): {
  top: { x: number; y: number };
  bottomLeft: { x: number; y: number };
  bottomRight: { x: number; y: number };
} | null {
  if (shotSamples.length < 3) return null;
  const lefts: FaceBox[] = [];
  const mids: FaceBox[] = [];
  const rights: FaceBox[] = [];

  for (const raw of shotSamples) {
    const big = filterEdgeBystanders(dedupeFaces(raw)).filter((c) => c.w >= MIN_READABLE_FACE_W);
    if (big.length < 3) continue;
    const largestArea = Math.max(...big.map((c) => c.w * c.h));
    const primary = big.filter((c) => c.w * c.h >= largestArea * 0.32);
    if (primary.length !== 3) continue;
    const ordered = [...primary].sort((a, b) => a.cx - b.cx);
    if (ordered[1]!.cx - ordered[0]!.cx < 0.18 || ordered[2]!.cx - ordered[1]!.cx < 0.18) continue;
    lefts.push(ordered[0]!);
    mids.push(ordered[1]!);
    rights.push(ordered[2]!);
  }

  if (lefts.length < 0.5 * shotSamples.length) return null;
  return {
    top: { x: median(mids.map((c) => c.cx)), y: median(mids.map((c) => c.cy)) },
    bottomLeft: { x: median(lefts.map((c) => c.cx)), y: median(lefts.map((c) => c.cy)) },
    bottomRight: { x: median(rights.map((c) => c.cx)), y: median(rights.map((c) => c.cy)) },
  };
}

export function analyzeWideScene(shotSamples: FaceBox[][], minShare = 0.5): boolean {
  if (shotSamples.length < 3) return false;
  let n = 0;
  for (const raw of shotSamples) if (needsContext(dedupeFaces(raw))) n++;
  return n >= minShare * shotSamples.length;
}

/**
 * Bidirectional Zero-Phase Gaussian Trajectory Smoother:
 * Eliminates micro-jitter within each continuous shot while keeping hard camera cuts 100% sharp.
 */
export function smoothShotTrajectory(shot: TrackPoint[], sigma = 1.65): TrackPoint[] {
  const n = shot.length;
  if (n <= 2) return shot;
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const weights: number[] = [];
  for (let k = -radius; k <= radius; k++) {
    weights.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  }

  return shot.map((pt, i) => {
    let wx = 0,
      wy = 0,
      wx2 = 0,
      wy2 = 0,
      wsum = 0;
    for (let k = -radius; k <= radius; k++) {
      const idx = Math.max(0, Math.min(n - 1, i + k));
      const w = weights[k + radius]!;
      const sample = shot[idx]!;
      wx += sample.x * w;
      wy += sample.y * w;
      wx2 += (sample.x2 ?? sample.x) * w;
      wy2 += (sample.y2 ?? sample.y) * w;
      wsum += w;
    }
    return {
      ...pt,
      x: wx / wsum,
      y: wy / wsum,
      x2: wx2 / wsum,
      y2: wy2 / wsum,
      cut: i === 0,
    };
  });
}

let lastTimestampMs = 0;

/**
 * A cheap read of the clip's own editing rhythm, before any subject is chased: seek, draw 16x9,
 * compare histograms, no detectors. It costs a small fraction of the tracking pass and lets the
 * camera start with this video's tolerance and pursuit speed instead of the neutral defaults — the
 * same two-pass order the local worker uses, so the preview does not drift from the export.
 *
 * The shot test itself stays the browser's own RGB-histogram metric; the worker's self-calibrated
 * Bhattacharyya threshold is measured on a different scale and has not been re-derived here.
 */
async function measureEditRhythm(
  v: HTMLVideoElement,
  ctx: CanvasRenderingContext2D,
  start: number,
  end: number,
  step: number,
  signal?: AbortSignal,
  onProgress?: (p: number) => void,
): Promise<{ tuning: EditTuning; shotGaps: number[] }> {
  const shotGaps: number[] = [];
  let samples = 0;
  let blackSamples = 0;
  let prev: FrameSignature | null = null;
  let lastShotTime = -10;

  for (let t = start; t <= end; t += step) {
    if (signal?.aborted) break;
    await seek(v, t, signal);
    const sig = computeFrameSignature(ctx, v);
    samples++;
    if (sig.meanLuma < 16) blackSamples++;
    if (detectSceneCut(prev, sig, t - lastShotTime, SHOT_FLOOR_SEC)) {
      if (prev && lastShotTime >= 0) shotGaps.push(t - lastShotTime);
      lastShotTime = t;
    }
    prev = sig;
    onProgress?.((t - start) / Math.max(0.1, end - start));
  }

  const safeSamples = Math.max(1, samples);
  return {
    shotGaps,
    tuning: tuneEdits({
      shotGaps,
      samples: safeSamples,
      faceSamples: 0,
      twoFaceShare: 0,
      personShare: 0,
      objectShare: 0,
      medianFaceWidth: 0,
      blackShare: blackSamples / safeSamples,
      motionEnergy: 0,
      subjectTravel: 0,
      subjectsMeasured: false,
    }),
  };
}

/**
 * OpenShorts Reframe Engine v2 (MediaPipe BlazeFace + EfficientDet/YOLO Person & Object Cascade +
 * OpenShortsSpeakerTracker + SmoothedCameraman + split_layout + active_speaker mouth region).
 *
 * Every edit number — hold, snap gate, whether a two-up is ever allowed, camera tolerance — comes
 * from this clip's measured rhythm, not from a constant, because no single number fits both a show
 * that cuts every 1.1 s and one that holds a frame for 20.
 */
export async function trackFaces(
  src: string,
  start: number,
  end: number,
  onProgress?: (p: number) => void,
  signal?: AbortSignal,
): Promise<TrackPoint[]> {
  const v = document.createElement("video");
  v.src = src;
  v.muted = true;
  v.playsInline = true;
  v.crossOrigin = "anonymous";
  await new Promise((r, rej) => {
    v.onloadeddata = r;
    v.onerror = () => rej(new Error("Video load error during tracking"));
  });

  const probeCanvas = document.createElement("canvas");
  probeCanvas.width = 64;
  probeCanvas.height = 36;
  const probeCtx = probeCanvas.getContext("2d", { willReadFrequently: true });
  let prevMouthFrame: Uint8ClampedArray | null = null;

  const step = 0.14;
  const rhythm = probeCtx
    ? await measureEditRhythm(v, probeCtx, start, end, step * 2, signal, (p) =>
        onProgress?.(p * 0.2),
      )
    : { tuning: tuneEdits(emptySignature()), shotGaps: [] as number[] };
  const { det: faceDet, objDet } = await getFaceDetector();

  const raw: { pt: TrackPoint; candidates: FaceBox[] }[] = [];
  let prevSig: FrameSignature | null = null;
  let lastShotTime = -10;
  let lastCutTime = -10;
  let stepIndex = 0;
  let faceSamples = 0;
  let twoFaceSamples = 0;
  let personSamples = 0;
  let objectSamples = 0;
  let blackSamples = 0;
  let motionSum = 0;
  const faceWidths: number[] = [];
  const travel: number[] = [];
  let prevTargetX: number | null = null;
  let srcAspect = 0;

  const tracker = new OpenShortsSpeakerTracker(8, 8);
  const cameraman = new OpenShortsSmoothedCameraman(3);
  cameraman.setSampleInterval(step);
  cameraman.applyTuning(rhythm.tuning);
  // How long the previous shot has to have lasted before a new one may re-aim the camera in one
  // frame. Anything shorter is followed by a pan instead, which reads as an edit rather than a glitch.
  const snapMinSec = Math.min(
    rhythm.tuning.layoutHoldSec,
    Math.max(0.35, rhythm.tuning.cutSnapMinSec),
  );

  try {
    for (let t = start; t <= end; t += step, stepIndex++) {
      await seek(v, t, signal);
      const sig = probeCtx ? computeFrameSignature(probeCtx, v) : null;
      // Skip near-black fade frames (OpenShorts mean < 16 rule)
      const isBlackFade = sig ? sig.meanLuma < 16 : false;
      if (isBlackFade) blackSamples++;
      const shotChange = sig
        ? detectSceneCut(prevSig, sig, t - lastShotTime, SHOT_FLOOR_SEC)
        : false;
      if (shotChange) {
        if (prevSig) tracker.reset();
        lastShotTime = t;
      }
      // A shot change always drops the sticky speaker; only a shot the viewer has had time to read
      // gets an instant re-aim.
      const cut = shotChange && t - lastCutTime >= snapMinSec;
      if (cut) {
        lastCutTime = t;
        cameraman.beginScene();
      }
      prevSig = sig;

      // Capture 64x36 frame data for active_speaker mouth-region motion estimation
      let currMouthFrame: Uint8ClampedArray | null = null;
      if (probeCtx) {
        probeCtx.drawImage(v, 0, 0, 64, 36);
        currMouthFrame = new Uint8ClampedArray(probeCtx.getImageData(0, 0, 64, 36).data);
        if (prevMouthFrame) {
          // Whole-frame luma change, the browser's stand-in for the worker's small-frame motion energy.
          let d = 0;
          for (let i = 1; i < currMouthFrame.length; i += 4)
            d += Math.abs(currMouthFrame[i]! - prevMouthFrame[i]!);
          motionSum += d / (currMouthFrame.length / 4);
        }
      }

      const mediaMs = Math.max(lastTimestampMs + 1, Math.round(t * 1000));
      lastTimestampMs = mediaMs;
      const vw = v.videoWidth || 1920,
        vh = v.videoHeight || 1080;
      srcAspect = srcAspect || vw / vh;

      const candidates: FaceBox[] = [];

      if (!isBlackFade) {
        // Stage 1: MediaPipe BlazeFace
        const faceRes = faceDet.detectForVideo(v, mediaMs);
        for (const d of faceRes.detections) {
          const b = d.boundingBox;
          const conf = d.categories?.[0]?.score ?? 0.75;
          if (!b) continue;
          const cx = Math.max(0.05, Math.min(0.95, (b.originX + b.width / 2) / vw));
          const cy = Math.max(0.08, Math.min(0.92, (b.originY + b.height / 2) / vh));
          const w = b.width / vw;
          const h = b.height / vh;

          // Compute active_speaker mouth region frame-to-frame motion (mouth_y = y + h * 0.72)
          let mouthAct = 0;
          if (currMouthFrame && prevMouthFrame) {
            const mx0 = Math.max(0, Math.min(63, Math.floor((cx - w * 0.36) * 64)));
            const mx1 = Math.max(mx0 + 1, Math.min(64, Math.ceil((cx + w * 0.36) * 64)));
            const myCenter = (b.originY / vh + h * 0.72) * 36;
            const my0 = Math.max(0, Math.min(35, Math.floor(myCenter - h * 0.22 * 36)));
            const my1 = Math.max(my0 + 1, Math.min(36, Math.ceil(myCenter + h * 0.22 * 36)));
            let diffSum = 0,
              count = 0;
            for (let py = my0; py < my1; py++) {
              for (let px = mx0; px < mx1; px++) {
                const idx = (py * 64 + px) * 4;
                const curG = currMouthFrame[idx + 1]!;
                const prvG = prevMouthFrame[idx + 1]!;
                // Ignore stationary near-black podcast microphone foam (< 22 luma)
                if (curG <= 22 && prvG <= 22) continue;
                diffSum += Math.abs(curG - prvG) / 255;
                count++;
              }
            }
            if (count > 0) mouthAct = diffSum / count;
          }

          candidates.push({
            cx,
            cy,
            w,
            h,
            score: w * h * (1.2 + 0.4 * conf),
            kind: "face",
            mouthActivity: mouthAct,
          });
        }

        // Stage 2 & 3: Portrait Upper-Body & Context Object Detection when hats/profiles hide faces or action is wide
        if (objDet) {
          try {
            const objRes = objDet.detectForVideo(v, mediaMs + 1);
            for (const od of objRes.detections) {
              const b = od.boundingBox;
              const cat = (od.categories?.[0]?.categoryName ?? "").toLowerCase();
              const conf = od.categories?.[0]?.score ?? 0.4;
              if (!b || conf < 0.3) continue;
              const isPerson = cat === "person";
              const cx = Math.max(0.05, Math.min(0.95, (b.originX + b.width / 2) / vw));
              const cy = Math.max(
                0.08,
                Math.min(0.92, (b.originY + b.height * (isPerson ? 0.24 : 0.5)) / vh),
              );
              const bw = b.width / vw,
                bh = b.height / vh;
              const overlapsFace = candidates.some(
                (f) =>
                  f.kind === "face" &&
                  Math.abs(f.cx - cx) < bw * 0.45 &&
                  Math.abs(f.cy - cy) < bh * 0.45,
              );
              if (!overlapsFace) {
                let bodyMouthAct = 0;
                if (isPerson && currMouthFrame && prevMouthFrame) {
                  const headW = bw * 0.42;
                  const headH = bh * 0.28;
                  const mx0 = Math.max(0, Math.min(63, Math.floor((cx - headW * 0.36) * 64)));
                  const mx1 = Math.max(mx0 + 1, Math.min(64, Math.ceil((cx + headW * 0.36) * 64)));
                  const myCenter = (cy + headH * 0.22) * 36;
                  const my0 = Math.max(0, Math.min(35, Math.floor(myCenter - headH * 0.22 * 36)));
                  const my1 = Math.max(
                    my0 + 1,
                    Math.min(36, Math.ceil(myCenter + headH * 0.22 * 36)),
                  );
                  let diffSum = 0,
                    cnt = 0;
                  for (let py = my0; py < my1; py++) {
                    for (let px = mx0; px < mx1; px++) {
                      const idx = (py * 64 + px) * 4;
                      const curG = currMouthFrame[idx + 1]!;
                      const prvG = prevMouthFrame[idx + 1]!;
                      if (curG <= 22 && prvG <= 22) continue;
                      diffSum += Math.abs(curG - prvG) / 255;
                      cnt++;
                    }
                  }
                  if (cnt > 0) bodyMouthAct = diffSum / cnt;
                }
                candidates.push({
                  cx,
                  cy,
                  w: bw * 0.6,
                  h: bh * 0.35,
                  score: bw * bh * (isPerson ? 0.85 : 0.35) * conf,
                  kind: isPerson ? "person" : "object",
                  mouthActivity: bodyMouthAct,
                });
              }
            }
          } catch {
            // ignore single-frame object detector error
          }
        }
      }

      prevMouthFrame = currMouthFrame;

      const target = tracker.getTarget(candidates, stepIndex);
      if (target) {
        cameraman.updateTarget(target.box);
        if (prevTargetX !== null)
          travel.push(Math.abs(cameraman.targetCenterX - prevTargetX) / cameraman.cropWidthNorm);
        prevTargetX = cameraman.targetCenterX;
      }

      // What this clip is made of, in the same terms the worker measures: how often a subject
      // survives detection at a readable size, how often two of them share the frame, and whether
      // the anchor was a face, a body or a thing.
      const subjects = candidates.filter((c) => c.kind !== "object");
      const readable = dedupeFaces(subjects).filter((c) => c.w >= MIN_READABLE_FACE_W);
      if (subjects.length) faceSamples++;
      if (readable.length >= 2) twoFaceSamples++;
      if (readable.length) faceWidths.push(Math.max(...readable.map((c) => c.w)));
      if (candidates.some((c) => c.kind === "person")) personSamples++;
      if (candidates.some((c) => c.kind === "object")) objectSamples++;

      const center = cameraman.getCenter(cut);
      const wideNeeded = needsContext(candidates);
      const mode: FrameLayoutMode = "fill";

      raw.push({
        pt: {
          t,
          x: center.x,
          y: center.y,
          x2: center.x,
          y2: center.y,
          trackId: target?.id ?? tracker.activeSpeakerId ?? 1,
          mode,
          cut,
          subjectType: target?.box.kind ?? (wideNeeded ? "wide" : "face"),
          hasBottomGraphic: sig?.hasBottomGraphic ?? false,
          isFlashback: (sig?.meanSat ?? 128) < 18,
        },
        candidates,
      });

      onProgress?.(0.2 + (0.8 * (t - start)) / Math.max(0.1, end - start));
    }
  } finally {
    v.src = "";
  }

  // Group into continuous camera shots and run OpenShorts scene-level split_layout & strategy hysteresis
  const shots: (typeof raw)[] = [];
  for (const item of raw) {
    if (item.pt.cut || !shots.length) shots.push([item]);
    else shots[shots.length - 1]!.push(item);
  }

  const out: TrackPoint[] = [];
  for (const shot of shots) {
    const samples = shot.map((s) => s.candidates);
    const isFlashbackShot = shot.filter((s) => s.pt.isFlashback).length > shot.length * 0.5;
    const splitPair = isFlashbackShot ? null : analyzeSplitScene(samples);
    if (splitPair) {
      for (let i = 0; i < shot.length; i++) {
        out.push({
          ...shot[i]!.pt,
          x: splitPair.left.x,
          y: splitPair.left.y,
          x2: splitPair.right.x,
          y2: splitPair.right.y,
          mode: "split",
          cut: i === 0,
        });
      }
      continue;
    }

    const threeUp = isFlashbackShot ? null : analyzeThreeUpScene(samples);
    if (threeUp && srcAspect >= 1.2) {
      for (let i = 0; i < shot.length; i++) {
        out.push({
          ...shot[i]!.pt,
          x: threeUp.top.x,
          y: threeUp.top.y,
          x2: threeUp.bottomLeft.x,
          y2: threeUp.bottomLeft.y,
          x3: threeUp.bottomRight.x,
          y3: threeUp.bottomRight.y,
          mode: "grid3",
          cut: i === 0,
        });
      }
      continue;
    }

    // A landscape shot that stages several people at once cannot be cropped to one 9:16 strip
    // without cutting the others off, so it keeps its whole frame as one large panel instead. Only
    // landscape sources need this; a portrait source already fills the canvas.
    if (srcAspect >= 1.2 && analyzeWideScene(samples)) {
      out.push(...shot.map((s) => ({ ...s.pt, mode: "fit" as FrameLayoutMode })));
      continue;
    }

    const fillMode = shot.map((s) => ({ ...s.pt, mode: "fill" as FrameLayoutMode }));
    const smoothed = smoothShotTrajectory(fillMode, 1.55);
    out.push(...smoothed);
  }

  // Now that the detectors have run, finish the picture the pre-pass could only guess at: does this
  // clip ever really stage two people together, and how long does each of its own shots last?
  const safeSamples = Math.max(1, raw.length);
  const tuning = tuneEdits({
    shotGaps: rhythm.shotGaps,
    samples: safeSamples,
    faceSamples,
    twoFaceShare: twoFaceSamples / safeSamples,
    personShare: personSamples / safeSamples,
    objectShare: objectSamples / safeSamples,
    medianFaceWidth: median(faceWidths),
    blackShare: blackSamples / safeSamples,
    motionEnergy: motionSum / safeSamples / step,
    subjectTravel: median(travel) / step,
  });

  if (!tuning.splitAllowed) {
    // Two-up only where the video actually stages two people side by side; anywhere else the second
    // panel is pointed at empty air and every appearance reads as noise.
    for (const pt of out) {
      if (pt.mode === "split") {
        pt.mode = "fill";
        pt.x2 = pt.x;
        pt.y2 = pt.y;
      }
    }
  }
  enforceLayoutHold(out, tuning.layoutHoldSec, tuning.splitMinHoldSec);
  return out;
}

export function sampleTrack(track: TrackPoint[] | undefined, t: number): TrackPoint {
  if (!track || !track.length) return { t, x: 0.5, y: 0.38, x2: 0.5, y2: 0.38, mode: "fill" };
  if (t <= track[0]!.t) return track[0]!;
  for (let i = 1; i < track.length; i++) {
    const b = track[i]!;
    if (b.t >= t) {
      const a = track[i - 1]!;
      if (b.cut) return t >= b.t - 0.02 ? b : a;
      const k = Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t || 1)));
      // Smoothstep interpolation between bounded keyframes
      const e = k * k * (3 - 2 * k);
      return {
        t,
        x: a.x + (b.x - a.x) * e,
        y: a.y + (b.y - a.y) * e,
        trackId: a.trackId,
        x2: (a.x2 ?? a.x) + ((b.x2 ?? b.x) - (a.x2 ?? a.x)) * e,
        y2: (a.y2 ?? a.y) + ((b.y2 ?? a.y) - (a.y2 ?? a.y)) * e,
        trackId2: a.trackId2,
        x3:
          a.x3 !== undefined || b.x3 !== undefined
            ? (a.x3 ?? a.x) + ((b.x3 ?? b.x) - (a.x3 ?? a.x)) * e
            : undefined,
        y3:
          a.y3 !== undefined || b.y3 !== undefined
            ? (a.y3 ?? a.y) + ((b.y3 ?? b.y) - (a.y3 ?? a.y)) * e
            : undefined,
        mode: a.mode ?? "fill",
        subjectType: a.subjectType,
        zoom: (a.zoom ?? 1) + ((b.zoom ?? 1) - (a.zoom ?? 1)) * e,
        hasBottomGraphic: a.hasBottomGraphic ?? b.hasBottomGraphic,
        isFlashback: a.isFlashback ?? b.isFlashback,
        trackingConfidence: a.trackingConfidence,
        layoutDecision: a.layoutDecision,
      };
    }
  }
  return track[track.length - 1]!;
}
