// OpenShorts v2 & OpusClip-grade clip renderer:
// - Dynamic Multi-Layout: TRACK (fill), SPLIT (2-speaker vstack), EDITORIAL (detail top + full-width context bottom), BLUR (fit)
// - Punch-in zoom from BOTH sources: worker audio-envelope beats on the track + word emphasis (RISE=0.25s, HOLD=1.30s, FALL=0.70s, MAX_ZOOM=1.12)
// - Layout-aware & face-safe caption placement (caption_timing.py: 0.50 on SPLIT, 0.58 on EDITORIAL)
// - Auto polish: contrast/saturation grade, big number callouts from editing cues, lower third, audio fades
import type { Word } from "./media";
import { layoutTransitionSec, punchGainOf } from "./edit-tuning";
import { DEFAULT_MOTION, motionSpec } from "./motion";
import { sampleTrack, type TrackPoint } from "./reframe";
import {
  captionPages,
  cleanHookTitle,
  HOOK_DURATION,
  HOOK_FONT_SIZE,
  HOOK_LINE_PITCH,
  HOOK_MAX_WIDTH,
  HOOK_PAD_X,
  HOOK_PAD_Y,
  HOOK_RADIUS,
  HOOK_TOP,
  isGoldKeyword,
  pageEmoji,
  staticMotion,
  wordMotion,
  type CaptionStyle,
  type WordMotion,
} from "./subtitles";
import { renderBatchWithLocalBackend, renderWithLocalBackend } from "./backend-render";
import { HELPER } from "@/components/YouTubeImport";

export type { CaptionStyle } from "./subtitles";
export type Layout = "auto" | "fill" | "split" | "grid3" | "editorial" | "fit";
export type ExportMode = "nvenc" | "ffmpeg_cpu";

export type ExportResult = {
  blob: Blob;
  mode: ExportMode;
  encoder: string;
  verified: boolean;
  sha256: string;
  /** Frames per second the composition really delivered, which is what the master was written at. */
  deliveredFps: number;
};

export type ClipCue = { kind: string; start: number; end: number; quote: string };

export type ClipSettings = {
  layout: Layout;
  captionStyle: CaptionStyle;
  highlight: string;
  secondaryHighlight?: string;
  captionY: number;
  showHook: boolean;
  showEmojis?: boolean;
  progressBar: boolean;
  zoomPunch: boolean;
  wordsPerPage: number;
  fontSize?: number;
  animationStrength?: number;
  animations?: boolean;
  effects?: boolean;
  voiceEnhance?: boolean;
  brandLabel?: string;
  manualX?: number | null;
  manualY?: number | null;
  /** Pop/easing character for hooks, callouts, emojis and caption words. */
  motion?: import("./motion").MotionPreset;
  /** Custom cubic-bezier [x1,y1,x2,y2] used when motion === "custom". */
  customCurve?: import("./motion").Bezier;
  /** Force uppercase for every caption style, not just the punchy presets. */
  captionUppercase?: boolean;
  /** Extra word outline thickness in px; undefined keeps the per-style default. */
  captionOutline?: number;
  /** Draw a translucent box behind each caption line. */
  captionBox?: boolean;
  /** Extend the exported clip window before the cut / after the cut, in seconds. */
  startPad?: number;
  endPad?: number;
};

export type ClipJob = {
  start: number;
  end: number;
  hook: string;
  words: Word[];
  track?: TrackPoint[] | undefined;
  cues?: ClipCue[] | undefined;
  settings: ClipSettings;
};

export const W = 1080,
  H = 1920;

export const FIT_LANDSCAPE_HEIGHT_RATIO = 0.42;
const FIT_VIDEO_BOTTOM = 0.70;
const FIT_CAPTION_TOP = 0.72;
const FIT_CAPTION_BOTTOM = 0.94;

export function fitGeometry(sourceW: number, sourceH: number, outputW = W, outputH = H, captionSpace = false) {
  const videoBottom = captionSpace ? outputH * FIT_VIDEO_BOTTOM : outputH;
  const scale = Math.min(outputW / sourceW, videoBottom / sourceH);
  const width = sourceW * scale;
  let height = sourceH * scale;
  if (sourceW > sourceH && outputH > outputW) {
    const minLandscapeH = Math.min(videoBottom, Math.round((outputH * FIT_LANDSCAPE_HEIGHT_RATIO) / 2) * 2);
    height = Math.max(height, minLandscapeH);
  }
  // Proportional zoom + center crop (OpenShorts GENERAL layout) so increasing height to 42% (806px)
  // never stretches/squashes subjects vertically ("elastic" distortion).
  const boxAspect = width / Math.max(1, height);
  const srcAspect = sourceW / Math.max(1, sourceH);
  let sx = 0,
    sy = 0,
    sw = sourceW,
    sh = sourceH;
  if (srcAspect > boxAspect + 1e-4) {
    sw = sourceH * boxAspect;
    sx = (sourceW - sw) / 2;
  } else if (srcAspect < boxAspect - 1e-4) {
    sh = sourceW / boxAspect;
    sy = (sourceH - sh) / 2;
  }
  return {
    x: (outputW - width) / 2,
    y: Math.min((outputH - height) / 2, videoBottom - height),
    width,
    height,
    sx,
    sy,
    sw,
    sh,
  };
}

// OpenShorts punch_in.py smoothstep curve constants
const PUNCH_RISE_SEC = 0.25;
const PUNCH_HOLD_SEC = 1.3;
// A 0.55 s release read as a snap-back because the push out had nowhere near as much time as the
// pull in. 0.70 keeps the asymmetric feel while letting the frame settle. local_worker PUNCH_FALL_SEC.
const PUNCH_FALL_SEC = 0.7;
const PUNCH_SPAN_SEC = PUNCH_RISE_SEC + PUNCH_HOLD_SEC + PUNCH_FALL_SEC;
const PUNCH_MAX_ZOOM = 1.12;

// Auto-polish constants (OpenShorts editing_director + export_quality look)
const GRADE = "contrast(1.05) saturate(1.14) brightness(1.02)";

function smoothstep(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
}

/** Interpolate two #rrggbb colours so a glow can bloom across a word boundary instead of switching. */
function mixHex(a: string, b: string, x: number): string {
  const parse = (h: string) => {
    const s = (h || "").replace("#", "");
    const v = s.length === 3 ? s.replace(/./g, (c) => c + c) : s;
    return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
  };
  const from = parse(a);
  const to = parse(b);
  if (!from.every(Number.isFinite) || !to.every(Number.isFinite)) return x > 0.5 ? b : a;
  const t = Math.max(0, Math.min(1, x));
  return `#${from
    .map((c, i) => Math.round(c + (to[i]! - c) * t).toString(16).padStart(2, "0"))
    .join("")}`;
}

/** Overshoot-then-settle pop used by hook cards, number callouts and emojis. */
function popScale(age: number, m = DEFAULT_MOTION, peak = m.peak): number {
  if (age < m.attack) return 0.55 + (peak - 0.55) * m.ease(age / m.attack);
  if (age < m.attack + m.settle) return peak - (peak - 1) * m.ease((age - m.attack) / m.settle);
  return 1;
}

/**
 * OpenShorts punch_in.zoom_curve:
 * Asymmetric smoothstep push (fast 0.25s in, 1.30s hold, slow 0.55s out) with minimum gap between punches.
 */
function computeOpenShortsPunchZoom(words: Word[], t: number): number {
  let maxZoom = 1.0;
  let lastTrigger = -100;
  for (const w of words) {
    if (w.s - lastTrigger < 8.0) continue;
    if (/!$/.test(w.w) || isGoldKeyword(w.w) || w.w.length >= 9) {
      lastTrigger = w.s;
      const dt = t - w.s;
      if (dt >= 0 && dt <= PUNCH_SPAN_SEC) {
        let amount = 0;
        if (dt < PUNCH_RISE_SEC) {
          amount = smoothstep(dt / PUNCH_RISE_SEC);
        } else if (dt < PUNCH_RISE_SEC + PUNCH_HOLD_SEC) {
          amount = 1.0;
        } else {
          amount = 1.0 - smoothstep((dt - PUNCH_RISE_SEC - PUNCH_HOLD_SEC) / PUNCH_FALL_SEC);
        }
        maxZoom = Math.max(maxZoom, 1.0 + (PUNCH_MAX_ZOOM - 1.0) * amount);
      }
    }
  }
  return maxZoom;
}

function drawCroppedSubject(
  ctx: CanvasRenderingContext2D,
  v: HTMLVideoElement,
  fx: number,
  fy: number,
  zoom: number,
  dx: number,
  dy: number,
  dw: number,
  dh: number,
  excludeBottomSubtitles = false,
) {
  const vw = v.videoWidth || 1920,
    vh = v.videoHeight || 1080;
  const targetAspect = dw / dh;
  const usableVh = excludeBottomSubtitles ? vh * 0.84 : vh;
  let cropH = usableVh / zoom;
  let cropW = cropH * targetAspect;
  if (cropW > vw) {
    cropW = vw / zoom;
    cropH = cropW / targetAspect;
  }
  // OpenShorts subject_position: 0.34 upper-third anchor
  let sx = fx * vw - cropW / 2;
  let sy = fy * vh - cropH * 0.36;
  sx = Math.max(0, Math.min(vw - cropW, sx));
  sy = Math.max(0, Math.min(usableVh - cropH, sy));
  ctx.drawImage(v, sx, sy, cropW, cropH, dx, dy, dw, dh);
}

/** Desaturated emphasis beat, mirroring the worker's _mono_pulse_filter. */
function punchMonoActive(words: Word[], t: number): boolean {
  let last = -100;
  for (const w of words) {
    if (w.s - last < 8.0) continue;
    if (/!$/.test(w.w) || isGoldKeyword(w.w) || w.w.length >= 9) {
      last = w.s;
      if (t >= w.s && t <= w.s + 0.6) return true;
    }
  }
  return false;
}

const NUMBER_WORD = /(\$|\d|million|billion|thousand|percent)/i;
/** A callout has to state a stake. Percentages and bare integers do not: "I 100 % channel…" pops a
 *  giant "100" over the subject's face for a phrase that means "absolutely", and a year or a page
 *  number is the same noise. Currency and magnitude words are what the viewer is meant to react to. */
const STAKE_MARKER = /[$]|doll?ars?|bucks?|grand|cash|money|million|billion|thousand/i;
function isEmphaticNumber(phrase: string): boolean {
  return /\d/.test(phrase) && STAKE_MARKER.test(phrase);
}

/** Rejoin tokens Whisper splits apart ("$ 6 614") back into the number the viewer should read. */
function tidyNumberPhrase(raw: string): string {
  return raw
    .replace(/\s+/g, " ")
    .replace(/\$\s+/g, "$")
    .replace(/\s*([,.])\s*/g, "$1")
    .replace(/(\d)\s+(\d{3})(?!\d)/g, "$1,$2")
    .replace(/\s+(?=[!?])\s*/g, "")
    .trim();
}

/** Stakes numbers read off the transcript itself, in order, spaced and capped so they stay punchy. */
function scanNumberPhrases(words: Word[]): ClipCue[] {
  const out: ClipCue[] = [];
  let run: Word[] = [];
  const close = () => {
    const first = run[0],
      last = run.at(-1);
    const quote = tidyNumberPhrase(run.map((w) => w.w).join(" "));
    run = [];
    if (!first || !last || !isEmphaticNumber(quote) || quote.length > 22) return;
    if (out.length >= 4) return;
    if (out.length && first.s - (out.at(-1)?.start ?? 0) < 5) return;
    out.push({ kind: "number", start: first.s, end: Math.max(first.s + 0.4, last.e), quote });
  };
  for (const w of words) {
    if (NUMBER_WORD.test(w.w)) {
      run.push(w);
      continue;
    }
    if (run.length) close();
  }
  if (run.length) close();
  return out;
}

/**
 * Number callouts must not depend on the model volunteering them: word timestamps give the exact
 * phrase and timing, so the transcript scan wins and discovery cues only cover a wordless clip.
 */
export function numberCues(words: Word[], cues: ClipCue[] | undefined): ClipCue[] {
  const fromWords = scanNumberPhrases(words);
  if (fromWords.length) return fromWords;
  return (cues ?? [])
    .map((c) => ({ kind: c.kind, start: c.start, end: c.end, quote: tidyNumberPhrase(c.quote) }))
    .filter((c) => isEmphaticNumber(c.quote) && /^[^,.]/.test(c.quote) && c.quote.length <= 22);
}

/** Big animated number from the discovery cues (the "$1,000,000" beat), popped over the subject. */
function drawEmphasisCallouts(
  ctx: CanvasRenderingContext2D,
  cues: ClipCue[] | undefined,
  t: number,
  s: ClipSettings,
) {
  if (!cues?.length) return;
  const motion = motionSpec(s.motion, s.customCurve);
  for (const c of cues) {
    const text = c.quote.trim().toUpperCase();
    if (!/\d/.test(text)) continue;
    const enter = c.start - 0.1;
    const exit = Math.max(c.end + 1.05, c.start + 1.5);
    if (t < enter || t > exit) continue;
    const age = t - enter;
    const animate = s.animations !== false;
    const scale = animate ? popScale(age, motion) : 1;
    let size = text.length > 9 ? 104 : 138;
    let face = `900 ${size}px Montserrat, sans-serif`;
    ctx.font = face;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const tw = ctx.measureText(text).width;
    if (tw > W - 140) {
      size = Math.floor((size * (W - 140)) / tw);
      face = `900 ${size}px Montserrat, sans-serif`;
    }
    const sprite = styledTextSprite(ctx, {
      text,
      font: face,
      fill: s.secondaryHighlight ?? "#ffd400",
      stroke: "#000000",
      strokeWidth: Math.round(size * 0.2),
      shadowColor: "rgba(0,0,0,0.6)",
      shadowBlur: 26,
      shadowOffsetY: 10,
    });
    ctx.save();
    ctx.globalAlpha = animate ? Math.min(1, age / 0.06) * Math.min(1, (exit - t) / 0.24) : 1;
    ctx.translate(W / 2, H * 0.3);
    ctx.scale(scale, scale);
    if (sprite) {
      blitStyledText(ctx, sprite, 0, 0);
    } else {
      ctx.font = face;
      ctx.lineJoin = "round";
      ctx.miterLimit = 2;
      ctx.shadowColor = "rgba(0,0,0,0.6)";
      ctx.shadowBlur = 26;
      ctx.shadowOffsetY = 10;
      ctx.lineWidth = Math.round(size * 0.2);
      ctx.strokeStyle = "#000000";
      ctx.strokeText(text, 0, 0);
      ctx.shadowColor = "transparent";
      ctx.fillStyle = s.secondaryHighlight ?? "#ffd400";
      ctx.fillText(text, 0, 0);
    }
    ctx.restore();
  }
}

function drawLowerThird(
  ctx: CanvasRenderingContext2D,
  label: string,
  rel: number,
  s: ClipSettings,
) {
  const text = label.trim();
  if (!text) return;
  const enter = 0.35,
    exit = 4.9;
  if (rel < enter || rel > exit) return;
  ctx.save();
  ctx.globalAlpha =
    s.animations === false ? 1 : Math.min(1, (rel - enter) / 0.22) * Math.min(1, (exit - rel) / 0.3);
  ctx.font = "800 40px Manrope, Montserrat, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const tw = ctx.measureText(text).width;
  const bx = 64,
    by = H - 168,
    bh = 76;
  ctx.fillStyle = "rgba(10,10,12,0.72)";
  roundRect(ctx, bx, by - bh / 2, tw + 66, bh, 18);
  ctx.fill();
  ctx.fillStyle = s.highlight;
  ctx.fillRect(bx + 12, by - bh / 2 + 14, 8, bh - 28);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, bx + 40, by + 2);
  ctx.restore();
}

const LAYOUT_TRANSITION_SEC = 0.34;

/** Per-clip edit rhythm, read off the cuts the track itself declares. */
const rhythmCache = new WeakMap<TrackPoint[], { transitionSec: number; punchGain: number }>();
function editRhythm(track: TrackPoint[] | undefined) {
  if (!track || track.length < 3) return { transitionSec: LAYOUT_TRANSITION_SEC, punchGain: 1 };
  let rhythm = rhythmCache.get(track);
  if (!rhythm) {
    rhythm = { transitionSec: layoutTransitionSec(track), punchGain: punchGainOf(track) };
    rhythmCache.set(track, rhythm);
  }
  return rhythm;
}

/**
 * 0→1 progress of the shot into the layout it is currently showing, measured back to the last
 * keyframe that wanted a different layout. A panel stack that appears instantly next to a crop the
 * viewer has been watching for seconds reads as a glitch; sliding it in reads as an edit — for as
 * long as this clip's own shots last, since the same slide is an edit on a three-second shot and a
 * smear on a one-second one.
 */
function layoutEaseIn(
  track: TrackPoint[] | undefined,
  t: number,
  mode: string,
  transitionSec: number,
): number {
  if (!track || track.length < 2) return 1;
  let i = track.length - 1;
  while (i > 0 && track[i]!.t > t) i--;
  for (; i >= 0; i--) {
    const p = track[i]!;
    if ((p.mode ?? "fill") !== mode) return Math.max(0, Math.min(1, (t - p.t) / transitionSec));
    if (t - p.t > transitionSec) return 1;
  }
  return 1;
}

/**
 * The blurred field behind a `fit` panel, composed at a quarter of canvas size and upscaled.
 * Blurring 1080x1920 every frame is the most expensive thing the renderer does and is what drags the
 * recorder below the rate it reports; the same blur applied to a quarter-size bed and scaled back up is
 * visually equivalent for a sixteenth of the pixels.
 */
const contextBed = typeof document === "undefined" ? null : document.createElement("canvas");
const contextBedRaw = typeof document === "undefined" ? null : document.createElement("canvas");

/** Quarter-res Gaussian sigma, mirrored by filters.py `BED_BLUR_SIGMA`; CSS blur() length is the sigma. */
export const BED_BLUR_SIGMA = 24;
/** The reference bed keeps the shot's exposure and chroma; only the focus is gone. */
export const BED_SATURATION = 1.1;

function drawContextBed(ctx: CanvasRenderingContext2D, v: HTMLVideoElement) {
  if (!contextBed || !contextBedRaw) return;
  const vw = v.videoWidth || 1920,
    vh = v.videoHeight || 1080;
  const bw = Math.max(2, Math.round(W / 4)),
    bh = Math.max(2, Math.round(H / 4));
  // CSS blur samples transparent black outside the drawn rect, which would fade the bed's edges;
  // ffmpeg's gblur clamps to edge pixels instead. Replicating the edge rows/columns into a 3-sigma
  // margin before the single blur pass keeps both pipelines identical at the frame borders.
  const pad = BED_BLUR_SIGMA * 3;
  const cw = bw + pad * 2,
    ch = bh + pad * 2;
  if (contextBed.width !== cw || contextBed.height !== ch) {
    contextBed.width = cw;
    contextBed.height = ch;
    contextBedRaw.width = cw;
    contextBedRaw.height = ch;
  }
  const b = contextBed.getContext("2d");
  const r = contextBedRaw.getContext("2d");
  if (!b || !r) return;
  // Crop out top/bottom letterbox margins (~13%) from cinematic sources so the blurred bed fills edge-to-edge with color.
  const srcY = vh * 0.13;
  const srcH = vh * 0.74;
  const cover = Math.max(bw / vw, bh / srcH);
  const dw = vw * cover,
    dh = srcH * cover;
  const dx = pad + (bw - dw) / 2,
    dy = pad + (bh - dh) / 2;
  r.filter = "none";
  r.clearRect(0, 0, cw, ch);
  r.drawImage(v, 0, srcY, vw, srcH, dx, dy, dw, dh);
  const mL = dx,
    mT = dy,
    mR = cw - dx - dw,
    mB = ch - dy - dh;
  if (mT > 0) r.drawImage(v, 0, srcY, vw, 1, dx, 0, dw, mT);
  if (mB > 0) r.drawImage(v, 0, srcY + srcH - 1, vw, 1, dx, dy + dh, dw, mB);
  if (mL > 0) r.drawImage(v, 0, srcY, 1, srcH, 0, dy, mL, dh);
  if (mR > 0) r.drawImage(v, vw - 1, srcY, 1, srcH, dx + dw, dy, mR, dh);
  if (mL > 0 && mT > 0) r.drawImage(v, 0, srcY, 1, 1, 0, 0, mL, mT);
  if (mR > 0 && mT > 0) r.drawImage(v, vw - 1, srcY, 1, 1, dx + dw, 0, mR, mT);
  if (mL > 0 && mB > 0) r.drawImage(v, 0, srcY + srcH - 1, 1, 1, 0, dy + dh, mL, mB);
  if (mR > 0 && mB > 0) r.drawImage(v, vw - 1, srcY + srcH - 1, 1, 1, dx + dw, dy + dh, mR, mB);
  b.filter = "none";
  b.clearRect(0, 0, cw, ch);
  b.filter = `blur(${BED_BLUR_SIGMA}px) saturate(${BED_SATURATION})`;
  b.drawImage(contextBedRaw, 0, 0);
  b.filter = "none";
  ctx.drawImage(contextBed, pad, pad, bw, bh, 0, 0, W, H);
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  v: HTMLVideoElement,
  t: number,
  job: ClipJob,
) {
  const { settings: s } = job;
  const vw = v.videoWidth || 1920,
    vh = v.videoHeight || 1080;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);

  const tp = sampleTrack(job.track, t);
  const rhythm = editRhythm(job.track);
  // Audio-envelope punches arrive on the track from the local worker, already scaled to this clip's
  // style; word emphasis is the browser-only fallback, so take whichever of the two pushes harder.
  const zoom = s.zoomPunch
    ? Math.max(tp.zoom ?? 1, 1 + (computeOpenShortsPunchZoom(job.words, t) - 1) * rhythm.punchGain)
    : 1;

  const fx = typeof s.manualX === "number" ? s.manualX : tp.x;
  const fy = typeof s.manualY === "number" ? s.manualY : tp.y;
  const effectiveLayout = s.layout === "split" && tp.layoutDecision?.reason === "unsafe_manual_split"
    ? "fit" : s.layout === "auto" ? (tp.mode ?? "fill") : s.layout;

  const effects = s.effects !== false;
  const mono = effects && s.zoomPunch && punchMonoActive(job.words, t);
  ctx.filter = effects ? GRADE + (mono ? " saturate(0)" : "") : "none";
  // Anything the tracker does not name explicitly renders as a full-frame crop, never a pillarbox.
  if (
    effectiveLayout !== "split" &&
    effectiveLayout !== "grid3" &&
    effectiveLayout !== "editorial" &&
    effectiveLayout !== "fit"
  ) {
    drawCroppedSubject(ctx, v, fx, fy, zoom, 0, 0, W, H);
  } else if (effectiveLayout === "split") {
    const k = tp.layoutDecision ? 1 : layoutEaseIn(job.track, t, "split", rhythm.transitionSec);
    const seam = Math.round(H - (H / 2) * k);
    const splitZoom = Math.max(zoom * 1.32, 1.38);
    // A panel may only be aimed at a subject the tracker measured. `tp.x2 ?? 1 - fx` used to mirror
    // the single person of a one-subject shot into the second panel, and the separation push then
    // slid both crops off her face into the source's blurred pillar bars, so one body read as two
    // speakers and neither crop held focus. Both producers already vet independence before they
    // stamp a split (the worker's `independent_split`, the browser's 0.22 centre separation), so a
    // point that does not carry a second anchor is a single-subject shot and crops as one.
    const sx2 = tp.x2;
    const sy2 = tp.y2;
    const secondSubject = typeof sx2 === "number" &&
      (tp.layoutDecision && tp.mode === "split" || Math.abs(sx2 - fx) >= 0.04);
    if (!secondSubject) {
      drawCroppedSubject(ctx, v, fx, fy, zoom, 0, 0, W, H);
    } else {
      drawCroppedSubject(ctx, v, fx, fy, splitZoom, 0, 0, W, seam, true);
      if (k > 0.02) {
        drawCroppedSubject(ctx, v, sx2, sy2 ?? fy, splitZoom, 0, seam, W, H - seam, true);
        if (k > 0.98) {
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.fillRect(0, seam - 3, W, 6);
        }
      }
    }
  } else if (effectiveLayout === "grid3") {
    // Rule 3: 3-Up Split Grid (1 Wide Top + 2 Side-by-Side Bottom, matching Target Reference Videos 4 & 5)
    const seam = Math.round(H / 2);
    const bottomH = H - seam;
    const halfW = Math.round(W / 2);
    // Same rule as the split: a panel needs a subject the tracker measured. `fx - 0.30` / `fx + 0.30`
    // offsets turned a one-person shot into three copies of that person, each partly on the bars.
    const leftPanel =
      typeof tp.x2 === "number" && Math.abs(tp.x2 - fx) >= 0.04 ? { x: tp.x2, y: tp.y2 } : null;
    const rightPanel =
      typeof tp.x3 === "number" && Math.abs(tp.x3 - fx) >= 0.04 ? { x: tp.x3, y: tp.y3 } : null;
    if (!leftPanel && !rightPanel) {
      drawCroppedSubject(ctx, v, fx, fy, zoom, 0, 0, W, H);
    } else {
      drawCroppedSubject(ctx, v, fx, fy, Math.max(zoom * 1.35, 1.4), 0, 0, W, seam, true);
      if (leftPanel && rightPanel) {
        drawCroppedSubject(
          ctx, v, leftPanel.x, leftPanel.y ?? fy, Math.max(zoom * 1.15, 1.2), 0, seam, halfW, bottomH, true,
        );
        drawCroppedSubject(
          ctx, v, rightPanel.x, rightPanel.y ?? fy, Math.max(zoom * 1.15, 1.2), halfW, seam, W - halfW, bottomH, true,
        );
        ctx.fillStyle = "rgba(0,0,0,0.82)";
        ctx.fillRect(0, seam - 3, W, 6);
        ctx.fillRect(halfW - 3, seam, 6, bottomH);
      } else {
        // One measured neighbour: a 2-up, not a 3-up with a repeated face.
        const only = leftPanel ?? rightPanel!;
        drawCroppedSubject(
          ctx,
          v,
          only.x,
          only.y ?? fy,
          Math.max(zoom * 1.15, 1.2),
          0,
          seam,
          W,
          bottomH,
          true,
        );
        ctx.fillStyle = "rgba(0,0,0,0.82)";
        ctx.fillRect(0, seam - 3, W, 6);
      }
    }
  } else if (effectiveLayout === "editorial") {
    // OpenShorts reframe_v2.editorial_geometry:
    // Top panel = full-height detail view on subject; Bottom panel = full-width context view (no blurry bars!)
    const fullWidthH = Math.round((W * vh) / vw);
    const contextH = Math.max(2, Math.min(fullWidthH, Math.floor((H * 0.45) / 2) * 2));
    const detailH = H - contextH;
    drawCroppedSubject(ctx, v, fx, fy, zoom, 0, 0, W, detailH);
    ctx.drawImage(v, 0, 0, vw, vh, 0, detailH, W, contextH);
    ctx.fillStyle = "rgb(25,25,25)";
    ctx.fillRect(0, detailH - 2, W, 4);
  } else if (effectiveLayout === "fit") {
    // The bed is the shot thrown out of focus (drawContextBed): same exposure and chroma, no detail.
    drawContextBed(ctx, v);

    const hasCaptionSpace = s.captionStyle !== "none" && (job.words?.length ?? 0) > 0;
    const fit = fitGeometry(vw, vh, W, H, hasCaptionSpace);
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 28;
    ctx.drawImage(v, fit.sx, fit.sy, fit.sw, fit.sh, fit.x, fit.y, fit.width, fit.height);
    ctx.restore();
  }
  ctx.filter = "none";

  drawFrameOverlays(ctx, t, job);
  ctx.restore();
}

/** Source-time visual layer shared by preview and transparent native-export PNGs. No video draws. */
export function drawFrameOverlays(ctx: CanvasRenderingContext2D, t: number, job: ClipJob) {
  const { settings: s } = job;
  const rel = t - job.start;
  const tp = sampleTrack(job.track, t);
  const effectiveLayout = s.layout === "split" && tp.layoutDecision?.reason === "unsafe_manual_split"
    ? "fit" : s.layout === "auto" ? (tp.mode ?? "fill") : s.layout;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = "none";

  if (s.showHook && job.hook && rel < HOOK_DURATION) {
    const a = s.animations === false ? 1 : Math.min(1, rel / 0.2) * Math.min(1, (HOOK_DURATION - rel) / 0.28);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.font = `800 ${HOOK_FONT_SIZE}px Montserrat, Manrope, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const lines = wrap(ctx, cleanHookTitle(job.hook, job.words), HOOK_MAX_WIDTH - HOOK_PAD_X * 2);
    const widest = Math.max(...lines.map((l) => ctx.measureText(l).width));
    const bw = Math.min(HOOK_MAX_WIDTH, widest + HOOK_PAD_X * 2);
    const bh = lines.length * HOOK_LINE_PITCH + HOOK_PAD_Y * 2;
    const left = (W - bw) / 2;

    ctx.shadowColor = "rgba(0,0,0,0.30)";
    ctx.shadowBlur = 24;
    ctx.shadowOffsetY = 8;
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, left, HOOK_TOP, bw, bh, HOOK_RADIUS);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#0f0f12";
    const midY = HOOK_TOP + HOOK_PAD_Y + HOOK_LINE_PITCH / 2;
    lines.forEach((l, i) => ctx.fillText(l, W / 2, midY + i * HOOK_LINE_PITCH));
    ctx.restore();
  }

  drawEmphasisCallouts(ctx, job.cues, t, s);

  if (s.captionStyle !== "none") {
    const pg = captionPages(job.words, s.wordsPerPage).find((p) => {
      const first = p[0],
        last = p.at(-1);
      return first && last && t >= first.s - 0.04 && t <= last.e + 0.2;
    });
    if (pg) {
      // OpenShorts caption_timing.caption_position: place at 0.50 on SPLIT/GRID3, 0.58 on EDITORIAL,
      // safely below the foreground video (~0.83) on FIT/Blur, and lift to 0.65 only on 1-Up FILL with lower-third graphics.
      const fitSafeY = Math.max((FIT_CAPTION_TOP + FIT_CAPTION_BOTTOM) / 2, Math.min(0.88, s.captionY));
      const effectiveY =
        effectiveLayout === "split" || effectiveLayout === "grid3"
          ? 0.5
          : effectiveLayout === "editorial"
            ? 0.58
            : effectiveLayout === "fit"
              ? fitSafeY
              : tp.hasBottomGraphic
                ? Math.min(s.captionY, 0.65)
                : s.captionY;
      if (effectiveLayout === "split" || effectiveLayout === "grid3") {
        const bandH = Math.round((s.fontSize ?? 74) * 1.75);
        ctx.fillStyle = "rgba(8,8,10,0.84)";
        ctx.fillRect(0, Math.round(H * effectiveY - bandH / 2), W, bandH);
      }
      drawCaption(ctx, pg, t, { ...s, captionY: effectiveY });
    }
  }

  if (s.brandLabel) drawLowerThird(ctx, s.brandLabel, rel, s);

  if (s.progressBar) {
    const k = Math.min(1, Math.max(0, rel / Math.max(0.1, job.end - job.start)));
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    ctx.fillRect(0, H - 12, W, 12);
    ctx.fillStyle = s.highlight;
    ctx.fillRect(0, H - 12, W * k, 12);
  }
  ctx.restore();
}

export function drawCaption(ctx: CanvasRenderingContext2D, pg: Word[], t: number, s: ClipSettings) {
  if (s.captionStyle === "none") return;
  const isOpus = s.captionStyle === "opus";
  let size = s.fontSize ?? (s.captionStyle === "minimal" ? 56 : isOpus ? 74 : 78);
  const family =
    s.captionStyle === "typewriter"
      ? "JetBrains Mono, monospace"
      : s.captionStyle === "minimal"
        ? "Manrope, sans-serif"
        : "Montserrat, sans-serif";
  const weight = s.captionStyle === "minimal" ? 700 : 900;
  const font = () => `${weight} ${size}px ${family}`;
  ctx.font = font();
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  const mspec = motionSpec(s.motion, s.customCurve);
  const forceUpper = s.captionUppercase === true;
  const txt = (w: Word) =>
    forceUpper || ["opus", "beast", "bounce", "glow"].includes(s.captionStyle)
      ? w.w.toUpperCase()
      : w.w;
  const maxLineWidth = W - 160;
  const widest = Math.max(1, ...pg.map((w) => ctx.measureText(txt(w)).width));
  if (widest > maxLineWidth) {
    size *= maxLineWidth / widest;
    ctx.font = font();
  }
  // Rule 9: Extra word spacing so 1.12x scaled active words ("IN THE BEGINNING", "AND WHATEVER THE") never collide
  const space = size * (isOpus ? 0.44 : 0.46);

  const lines: Word[][] = [[]];
  let lw = 0;
  for (const w of pg) {
    const ww = ctx.measureText(txt(w)).width;
    const last = lines.at(-1);
    if (lw + ww > maxLineWidth && last?.length) {
      lines.push([]);
      lw = 0;
    }
    lines.at(-1)?.push(w);
    lw += ww + space;
  }

  const lh = size * (isOpus ? 1.18 : 1.28);
  const baseY = H * s.captionY - ((lines.length - 1) * lh) / 2;

  if ((s.showEmojis ?? true) && (isOpus || s.captionStyle === "beast")) {
    const em = pageEmoji(pg);
    if (em && t >= em.triggerTime - 0.05) {
      const elapsed = Math.max(0, t - em.triggerTime);
      // popScale already carries hook cards and number callouts; the emoji used to run its own
      // quarter-sine attack followed by a linear decay, so it settled differently from everything
      // else on screen.
      const pop = s.animations === false ? 1 : popScale(elapsed, mspec, 1.15);
      const emojiY = baseY - lh * 0.95;
      const emojiFont = `${Math.round(size * 1.55)}px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
      const sprite = styledTextSprite(ctx, {
        text: em.emoji,
        font: emojiFont,
        fill: "#fff",
        shadowColor: "rgba(0,0,0,0.55)",
        shadowBlur: 18,
      });
      ctx.save();
      ctx.translate(W / 2, emojiY);
      ctx.scale(pop, pop);
      if (sprite) {
        blitStyledText(ctx, sprite, 0, 0);
      } else {
        ctx.font = emojiFont;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.shadowColor = "rgba(0,0,0,0.55)";
        ctx.shadowBlur = 18;
        ctx.fillText(em.emoji, 0, 0);
      }
      ctx.restore();
      ctx.font = font();
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
    }
  }

  const goldColor = s.secondaryHighlight ?? "#ffd400";

  lines.forEach((line, li) => {
    const widths = line.map((w) => ctx.measureText(txt(w)).width);
    const total = widths.reduce((a, b) => a + b, 0) + space * (line.length - 1);
    let x = (W - total) / 2;
    const y = baseY + li * lh;

    if (s.captionBox && line.length) {
      ctx.save();
      ctx.fillStyle = "rgba(8,8,10,0.62)";
      roundRect(ctx, x - 24, y - size * 0.68, total + 48, size * 1.36, 14);
      ctx.fill();
      ctx.restore();
    }

    line.forEach((w, i) => {
      const motion: WordMotion =
        s.animations === false
          ? staticMotion(s.captionStyle, t, w)
          : wordMotion(s.captionStyle, t, w, s.animationStrength ?? 1, mspec.ease);
      const active = motion.active;
      const past = t > w.e;
      const ww = widths[i] ?? 0;

      ctx.save();
      ctx.globalAlpha = motion.alpha * (s.captionStyle === "karaoke" && t < w.s ? 0.85 : 1);
      ctx.translate(x + ww / 2, y + motion.y);
      ctx.scale(motion.scale, motion.scale);
      ctx.translate(-(x + ww / 2), -y);

      // The chip used to appear and vanish on a single frame at each word boundary. Easing its
      // alpha and inset with the word makes it arrive as a sweep rather than a strobe.
      const chip = s.captionStyle === "karaoke" ? motion.activity : 0;
      if (chip > 0.002) {
        const grow = size * 0.12 * (1 - chip);
        ctx.save();
        ctx.globalAlpha = motion.alpha * chip;
        ctx.fillStyle = s.highlight;
        roundRect(ctx, x - 14 - grow, y - size * 0.62 - grow, ww + 28 + grow * 2, size * 1.2 + grow * 2, 16);
        ctx.fill();
        ctx.restore();
      }

      let visible = txt(w);
      if (s.captionStyle === "typewriter") {
        const fraction =
          s.animations === false
            ? Number(t >= w.s)
            : Math.max(0, Math.min(1, (t - w.s) / Math.max(0.08, w.e - w.s)));
        visible = visible.slice(0, Math.ceil(fraction * visible.length));
      }

      ctx.lineJoin = "round";
      ctx.miterLimit = 2;

      const glow = s.captionStyle === "glow";
      // Quantised to a ladder because the sprite cache keys on shadowBlur and colour: a continuous
      // value would build a new offscreen canvas every frame, the cost the vignette gradient avoids.
      const glowLevel = glow ? Math.round(motion.activity * 5) / 5 : 0;
      const shadow: Pick<StyledTextSprite, "shadowColor" | "shadowBlur" | "shadowOffsetY"> = glow
        ? {
            shadowColor: mixHex("#000000", s.highlight, glowLevel),
            shadowBlur: 12 + 16 * glowLevel,
            shadowOffsetY: 0,
          }
        : { shadowColor: "rgba(0,0,0,0.85)", shadowBlur: 10, shadowOffsetY: 6 };

      let fill: string;
      if (isOpus) {
        if (active) {
          fill = isGoldKeyword(w.w) ? goldColor : s.highlight;
        } else if (past && isGoldKeyword(w.w)) {
          fill = goldColor;
        } else {
          fill = "#ffffff";
        }
      } else {
        fill = ["beast", "bounce", "glow", "reveal", "typewriter"].includes(s.captionStyle)
          ? active
            ? s.highlight
            : "#fff"
          : s.captionStyle === "karaoke"
            ? // Flip at the chip's half-ramp instead of the hard word edge, so text and box cross
              // over together and never leave white letters on a half-faded box.
              motion.activity >= 0.5
              ? "#000"
              : past
                ? "#fff"
                : "rgba(255,255,255,0.85)"
            : active
              ? s.highlight
              : "#fff";
      }

      const skipStroke = s.captionStyle === "karaoke" && motion.activity >= 0.5;
      const baseStroke = isOpus
        ? Math.max(14, Math.round(size * 0.22))
        : 14;
      const outline = s.captionOutline;
      const strokeWidth =
        s.captionStyle === "minimal" || skipStroke ? 0 : Math.max(0, outline ?? baseStroke);
      const sprite = styledTextSprite(ctx, {
        text: visible,
        font: font(),
        fill,
        stroke: strokeWidth > 0 ? "#000000" : undefined,
        strokeWidth,
        ...shadow,
      });

      if (sprite) {
        blitStyledText(ctx, sprite, x + ww / 2, y);
      } else {
        ctx.fillStyle = fill;
        ctx.fillText(visible, x, y);
      }
      ctx.restore();
      x += ww + space;
    });
  });
}

function wrap(ctx: CanvasRenderingContext2D, text: string, max: number) {
  const words = text.split(" ");
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const n = cur ? cur + " " + w : w;
    if (ctx.measureText(n).width > max && cur) {
      lines.push(cur);
      cur = w;
    } else cur = n;
  }
  if (cur) lines.push(cur);
  return lines.slice(0, 3);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

/**
 * A canvas shadow is a Gaussian blur recomputed for every draw call, and the styled caption path
 * makes one per word per frame. On a 1080x1920 export that cost more than the rest of the frame put
 * together, so the capture loop could not keep up with playback and the master had to repeat frames
 * to reach its frame rate — the lag that reads as a stuttering clip. A styled word changes only when
 * its text or its highlight state changes, so it is blurred once into a sprite and every later frame
 * blits that sprite.
 */
const styledTextSprites = new Map<string, HTMLCanvasElement>();

/** Load the actual canvas faces, not just fonts used by DOM text, before any export rasterization. */
export async function prepareOverlayFonts(job: ClipJob): Promise<void> {
  if (!document.fonts) throw new Error("Caption overlays require the browser Font Loading API.");
  const { settings: s } = job;
  const faces = new Map<string, string>();
  if (s.showHook && job.hook) faces.set(`800 ${HOOK_FONT_SIZE}px "Montserrat"`, cleanHookTitle(job.hook, job.words));
  if (job.cues?.length) faces.set('900 138px "Montserrat"', job.cues.map((c) => c.quote.toUpperCase()).join(" "));
  if (s.brandLabel?.trim()) faces.set('800 40px "Manrope"', s.brandLabel.trim());
  if (s.captionStyle !== "none" && job.words.length) {
    const font = s.captionStyle === "typewriter"
      ? '900 78px "JetBrains Mono"'
      : s.captionStyle === "minimal" ? '700 56px "Manrope"' : '900 74px "Montserrat"';
    // Include both cases so unicode-range font subsets match the renderer's uppercase presets too.
    const text = job.words.map((w) => w.w).join(" ");
    faces.set(font, `${text} ${text.toUpperCase()}`);
  }
  await document.fonts.ready;
  await Promise.all(Array.from(faces, async ([font, text]) => {
    const loaded = await document.fonts.load(font, text || "Aa");
    if (!loaded.length || !document.fonts.check(font, text || "Aa")) {
      throw new Error(`Required caption overlay font is unavailable: ${font}`);
    }
  }));
  if ((s.showEmojis ?? true) && (s.captionStyle === "opus" || s.captionStyle === "beast")) {
    const text = captionPages(job.words, s.wordsPerPage).map((pg) => pageEmoji(pg)?.emoji ?? "").join("");
    if (text) {
      // System emoji faces have no CSS FontFace entries; an empty load result is valid for these.
      await document.fonts.load('115px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif', text);
    }
  }
  await document.fonts.ready;
  // Preview may have cached fallback glyphs before the web fonts finished loading.
  styledTextSprites.clear();
}

type StyledTextSprite = {
  text: string;
  font: string;
  fill: string;
  stroke?: string | undefined;
  strokeWidth?: number | undefined;
  shadowColor?: string | undefined;
  shadowBlur?: number | undefined;
  shadowOffsetY?: number | undefined;
};

function styledTextSprite(
  measure: CanvasRenderingContext2D,
  o: StyledTextSprite,
): HTMLCanvasElement | null {
  if (!o.text) return null;
  const key = [
    o.text,
    o.font,
    o.fill,
    o.stroke ?? "-",
    o.strokeWidth ?? 0,
    o.shadowColor ?? "-",
    o.shadowBlur ?? 0,
    o.shadowOffsetY ?? 0,
  ].join("|");
  const cached = styledTextSprites.get(key);
  if (cached) return cached;

  const prevFont = measure.font;
  measure.font = o.font;
  const tw = measure.measureText(o.text).width;
  measure.font = prevFont;
  if (tw < 1) return null;

  const size = Number(/(\d+(?:\.\d+)?)px/.exec(o.font)?.[1] ?? 64);
  const grow = (o.shadowBlur ?? 0) + (o.strokeWidth ?? 0) / 2 + 10;
  const c = document.createElement("canvas");
  c.width = Math.ceil(tw + grow * 2);
  c.height = Math.ceil(size * 1.9 + grow * 2);
  const x = c.getContext("2d");
  if (!x) return null;
  x.font = o.font;
  x.textAlign = "center";
  x.textBaseline = "middle";
  x.lineJoin = "round";
  x.miterLimit = 2;
  x.shadowColor = o.shadowColor ?? "transparent";
  x.shadowBlur = o.shadowBlur ?? 0;
  x.shadowOffsetY = o.shadowOffsetY ?? 0;
  if (o.stroke && o.strokeWidth) {
    x.lineWidth = o.strokeWidth;
    x.strokeStyle = o.stroke;
    x.strokeText(o.text, c.width / 2, c.height / 2);
  }
  x.shadowColor = "transparent";
  x.shadowBlur = 0;
  x.shadowOffsetY = 0;
  x.fillStyle = o.fill;
  x.fillText(o.text, c.width / 2, c.height / 2);

  // A clip's vocabulary is bounded, but a long session of many clips is not: drop the whole layer
  // rather than let the cache grow without limit.
  if (styledTextSprites.size > 900) styledTextSprites.clear();
  styledTextSprites.set(key, c);
  return c;
}

/** Blit a cached styled word so its centre lands where `fillText` would have put it. */
function blitStyledText(
  ctx: CanvasRenderingContext2D,
  sprite: HTMLCanvasElement,
  cx: number,
  cy: number,
) {
  ctx.drawImage(sprite, cx - sprite.width / 2, cy - sprite.height / 2);
}

async function computeSha256Hex(blob: Blob): Promise<string> {
  try {
    const buf = await blob.arrayBuffer();
    const hash = await crypto.subtle.digest("SHA-256", buf);
    return Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return "";
  }
}

export async function exportClip(
  src: string,
  job: ClipJob,
  onProgress: (p: number, stageLabel?: string) => void,
  signal?: AbortSignal,
  backend?: { sourceId?: string | undefined; file?: File | Blob | null | undefined },
): Promise<ExportResult> {
  if (signal?.aborted) throw new Error("cancelled");
  // Final exports are always mastered by the local worker's native FFmpeg/NVENC pipeline. There is no
  // browser MediaRecorder path: an offline or failed renderer surfaces as an error instead of silently
  // falling back to a real-time canvas recording that bypasses the backend.
  const out = await renderWithLocalBackend(job, backend?.sourceId, onProgress, signal, {
    file: backend?.file,
    src,
  });
  const sha256 = await computeSha256Hex(out.blob);
  if (signal?.aborted) throw new Error("cancelled");
  return {
    blob: out.blob,
    mode: out.encoder === "h264_nvenc" ? "nvenc" : "ffmpeg_cpu",
    encoder: out.encoder,
    verified: out.verified,
    sha256,
    deliveredFps: out.deliveredFps,
  };
}

/**
 * Export several accepted clips in one batch against the same source. The video is cached once and
 * FFmpeg passes are streamed by the worker, so this is materially faster than looping exportClip.
 * Results come back in the same order as `jobs`.
 */
export async function exportClipsBatch(
  src: string,
  jobs: ClipJob[],
  onProgress: (p: number, stageLabel?: string) => void,
  signal?: AbortSignal,
  backend?: { sourceId?: string | undefined; file?: File | Blob | null | undefined },
): Promise<ExportResult[]> {
  if (signal?.aborted) throw new Error("cancelled");
  if (!jobs.length) return [];
  const outs = await renderBatchWithLocalBackend(jobs, backend?.sourceId, onProgress, signal, {
    file: backend?.file,
    src,
  });
  const results: ExportResult[] = [];
  for (const out of outs) {
    const sha256 = await computeSha256Hex(out.blob);
    if (signal?.aborted) throw new Error("cancelled");
    results.push({
      blob: out.blob,
      mode: out.encoder === "h264_nvenc" ? "nvenc" : "ffmpeg_cpu",
      encoder: out.encoder,
      verified: out.verified,
      sha256,
      deliveredFps: out.deliveredFps,
    });
  }
  return results;
}

export function shortGpuName(raw: string): string {
  const angle = /^ANGLE\s*\((.*)\)\s*$/i.exec(raw);
  const parts = (angle ? angle[1]! : raw)
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const driver = parts.find((p) => /driver$/i.test(p));
  const candidate = (driver ? driver.replace(/\s*driver$/i, "") : (parts[1] ?? parts[0] ?? raw)).replace(
    /^ANGLE\s+Metal\s+Renderer:\s*/i,
    "",
  );
  const cleaned = candidate
    .replace(/\s*(Direct3D|D3D|OpenGL|Vulkan|Metal).*$/i, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned.length >= 3 ? cleaned : raw;
}

export async function gpuInfo(): Promise<string | null> {
  const nav = navigator as Navigator & {
    gpu?: {
      requestAdapter: () => Promise<{
        info?: { vendor?: string; architecture?: string; description?: string };
      } | null>;
    };
  };
  try {
    const a = await nav.gpu?.requestAdapter();
    if (a) {
      const i = a.info;
      return (
        [i?.vendor, i?.architecture, i?.description].filter(Boolean).join(" · ") || "WebGPU ready"
      );
    }
  } catch {
    /* ignore */
  }
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    if (gl && ext) return String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL));
  } catch {
    /* ignore */
  }
  return null;
}
