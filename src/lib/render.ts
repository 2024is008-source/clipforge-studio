// OpenShorts v2 & OpusClip-grade clip renderer:
// - Dynamic Multi-Layout: TRACK (fill), SPLIT (2-speaker vstack), EDITORIAL (detail top + full-width context bottom), BLUR (fit)
// - Punch-in zoom from BOTH sources: worker audio-envelope beats on the track + word emphasis (RISE=0.25s, HOLD=1.30s, FALL=0.55s, MAX_ZOOM=1.12)
// - Layout-aware & face-safe caption placement (caption_timing.py: 0.50 on SPLIT, 0.58 on EDITORIAL)
// - Auto polish: scene-cut dips, vignette, contrast/saturation grade, big number callouts from editing cues, lower third, audio fades
import type { Word } from "./media";
import { layoutTransitionSec, punchGainOf } from "./edit-tuning";
import { sampleTrack, type TrackPoint } from "./reframe";
import { captionPages, cleanHookTitle, isGoldKeyword, pageEmoji, wordMotion, type CaptionStyle } from "./subtitles";
import { browserCaptureForced, renderWithLocalBackend } from "./backend-render";
import { HELPER } from "@/components/YouTubeImport";

export type { CaptionStyle } from "./subtitles";
export type Layout = "auto" | "fill" | "split" | "grid3" | "editorial" | "fit";
export type TargetFps = "29.97" | "30";
export type ExportMode = "nvenc" | "ffmpeg_cpu" | "browser_fallback";

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
  fps?: TargetFps;
  voiceEnhance?: boolean;
  brandLabel?: string;
  manualX?: number | null;
  manualY?: number | null;
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

const FIT_VIDEO_BOTTOM = 0.70;
const FIT_CAPTION_TOP = 0.72;
const FIT_CAPTION_BOTTOM = 0.94;

export function fitGeometry(sourceW: number, sourceH: number, outputW = W, outputH = H, captionSpace = false) {
  const videoBottom = captionSpace ? outputH * FIT_VIDEO_BOTTOM : outputH;
  const scale = Math.min(outputW / sourceW, videoBottom / sourceH);
  const width = sourceW * scale;
  const height = sourceH * scale;
  return { x: (outputW - width) / 2, y: Math.min((outputH - height) / 2, videoBottom - height), width, height };
}

// OpenShorts punch_in.py smoothstep curve constants
const PUNCH_RISE_SEC = 0.25;
const PUNCH_HOLD_SEC = 1.3;
const PUNCH_FALL_SEC = 0.55;
const PUNCH_SPAN_SEC = PUNCH_RISE_SEC + PUNCH_HOLD_SEC + PUNCH_FALL_SEC;
const PUNCH_MAX_ZOOM = 1.12;

// Auto-polish constants (OpenShorts editing_director + export_quality look)
const CUT_DIP_SEC = 0.15;
const GRADE = "contrast(1.05) saturate(1.14) brightness(1.02)";

function smoothstep(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
}

/** Overshoot-then-settle pop used by hook cards, number callouts and emojis. */
function popScale(age: number, attack = 0.16, settle = 0.14, peak = 1.1): number {
  if (age < attack) return 0.55 + (peak - 0.55) * smoothstep(age / attack);
  if (age < attack + settle) return peak - (peak - 1) * smoothstep((age - attack) / settle);
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

/** Dip-to-black centred on each verified scene cut, so hard cuts read as edits, not glitches. */
function drawCutDip(ctx: CanvasRenderingContext2D, track: TrackPoint[] | undefined, t: number) {
  if (!track || track.length < 2) return;
  const firstT = track[0]!.t;
  let k = 0;
  for (const p of track) {
    if (!p.cut || p.t <= firstT + 0.001) continue;
    const d = Math.abs(t - p.t);
    if (d < CUT_DIP_SEC) k = Math.max(k, 1 - d / CUT_DIP_SEC);
  }
  if (k <= 0) return;
  ctx.fillStyle = `rgba(0,0,0,${(0.85 * k * k).toFixed(3)})`;
  ctx.fillRect(0, 0, W, H);
}

const EDGE_BLUR_SEC = 0.16;
/** Blur-dip straddling layout seams and verified cuts, mirroring the worker's _edge_blur_filter. */
function seamDipBlur(track: TrackPoint[] | undefined, t: number): number {
  if (!track || track.length < 2) return 0;
  let d = Infinity;
  let prevMode = track[0]!.mode;
  for (let i = 1; i < track.length; i++) {
    const p = track[i]!;
    if (p.cut || p.mode !== prevMode) d = Math.min(d, Math.abs(t - p.t));
    prevMode = p.mode;
  }
  return d < EDGE_BLUR_SEC ? 26 * (1 - d / EDGE_BLUR_SEC) : 0;
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

/** Cinematic edge falloff; Opus keeps the centre of interest brightest. */
let vignetteFill: CanvasGradient | null = null;
function drawVignette(ctx: CanvasRenderingContext2D) {
  // The gradient is identical on every frame of every render; rebuilding it 30 times a second is
  // cost the capture loop cannot spend without dropping the frames it then pads in mastering.
  if (!vignetteFill) {
    const g = ctx.createRadialGradient(
      W / 2,
      H * 0.46,
      H * 0.3,
      W / 2,
      H * 0.5,
      Math.hypot(W / 2, H / 2),
    );
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(0.62, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0,0.36)");
    vignetteFill = g;
  }
  ctx.fillStyle = vignetteFill;
  ctx.fillRect(0, 0, W, H);
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
  for (const c of cues) {
    const text = c.quote.trim().toUpperCase();
    if (!/\d/.test(text)) continue;
    const enter = c.start - 0.1;
    const exit = Math.max(c.end + 1.05, c.start + 1.5);
    if (t < enter || t > exit) continue;
    const age = t - enter;
    const animate = s.animations !== false;
    const scale = animate ? popScale(age) : 1;
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

function drawContextBed(ctx: CanvasRenderingContext2D, v: HTMLVideoElement) {
  if (!contextBed) return;
  const vw = v.videoWidth || 1920,
    vh = v.videoHeight || 1080;
  const bw = Math.max(2, Math.round(W / 4)),
    bh = Math.max(2, Math.round(H / 4));
  if (contextBed.width !== bw || contextBed.height !== bh) {
    contextBed.width = bw;
    contextBed.height = bh;
  }
  const b = contextBed.getContext("2d");
  if (!b) return;
  // Crop out top/bottom letterbox margins (~13%) from cinematic sources so the blurred bed fills edge-to-edge with color.
  const srcY = vh * 0.13;
  const srcH = vh * 0.74;
  const cover = Math.max(bw / vw, bh / srcH);
  b.filter = "none";
  b.clearRect(0, 0, bw, bh);
  b.filter = "blur(12px) saturate(0.8) brightness(0.9)";
  b.drawImage(v, 0, srcY, vw, srcH, (bw - vw * cover) / 2, (bh - srcH * cover) / 2, vw * cover, srcH * cover);
  b.filter = "none";
  ctx.drawImage(contextBed, 0, 0, W, H);
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
  const dip = effects ? seamDipBlur(job.track, t) : 0;
  const mono = effects && s.zoomPunch && punchMonoActive(job.words, t);
  ctx.filter = effects
    ? GRADE + (dip > 0.5 ? ` blur(${dip.toFixed(1)}px)` : "") + (mono ? " saturate(0)" : "")
    : "none";
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
    // OpenShorts contain_background: the same shot enlarged to fill, blurred, only slightly darkened.
    // The browser used to add brightness(0.62) and a 0.32/0.48 gradient on top of a 44px blur, which is
    // why every letterboxed export read as two black bars rather than a background.
    drawContextBed(ctx, v);

    const hasCaptionSpace = s.captionStyle !== "none" && (job.words?.length ?? 0) > 0;
    const fit = fitGeometry(vw, vh, W, H, hasCaptionSpace);
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 28;
    ctx.drawImage(v, 0, 0, vw, vh, fit.x, fit.y, fit.width, fit.height);
    ctx.restore();
  }
  ctx.filter = "none";

  if (effects) {
    drawCutDip(ctx, job.track, t);
    drawVignette(ctx);
  }
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

  if (s.showHook && job.hook && rel < 3.6) {
    const a = s.animations === false ? 1 : Math.min(1, rel / 0.2) * Math.min(1, (3.6 - rel) / 0.28);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.font = "800 44px Montserrat, Manrope, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lines = wrap(ctx, cleanHookTitle(job.hook, job.words), W * 0.62);
    const lh = 56,
      left = 60,
      top = 96;
    const bw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 72;
    const bh = lines.length * lh + 36;

    ctx.shadowColor = "rgba(0,0,0,0.38)";
    ctx.shadowBlur = 24;
    ctx.shadowOffsetY = 8;
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, left, top - 44, bw, bh, 22);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#0f0f12";
    lines.forEach((l, i) => ctx.fillText(l, left + 36, top + i * lh));
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

  const txt = (w: Word) =>
    ["opus", "beast", "bounce", "glow"].includes(s.captionStyle) ? w.w.toUpperCase() : w.w;
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
      const pop =
        s.animations === false
          ? 1
          : elapsed < 0.18
            ? Math.sin((elapsed / 0.18) * (Math.PI / 2)) * 1.15
            : Math.max(1, 1.15 - (elapsed - 0.18) * 0.5);
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

    line.forEach((w, i) => {
      const motion =
        s.animations === false
          ? { active: t >= w.s && t < w.e, scale: 1, y: 0, alpha: s.captionStyle === "reveal" && t < w.s ? 0 : 1 }
          : wordMotion(s.captionStyle, t, w, s.animationStrength ?? 1);
      const active = motion.active;
      const past = t > w.e;
      const ww = widths[i] ?? 0;

      ctx.save();
      ctx.globalAlpha = motion.alpha;
      ctx.translate(x + ww / 2, y + motion.y);
      ctx.scale(motion.scale, motion.scale);
      ctx.translate(-(x + ww / 2), -y);

      if (s.captionStyle === "karaoke" && active) {
        ctx.fillStyle = s.highlight;
        roundRect(ctx, x - 14, y - size * 0.62, ww + 28, size * 1.2, 16);
        ctx.fill();
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
      const shadow: Pick<StyledTextSprite, "shadowColor" | "shadowBlur" | "shadowOffsetY"> = glow
        ? {
            shadowColor: active ? s.highlight : "#000",
            shadowBlur: active ? 28 : 12,
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
            ? active
              ? "#000"
              : past
                ? "#fff"
                : "rgba(255,255,255,0.85)"
            : active
              ? s.highlight
              : "#fff";
      }

      const skipStroke = s.captionStyle === "karaoke" && active;
      const sprite = styledTextSprite(ctx, {
        text: visible,
        font: font(),
        fill,
        stroke: s.captionStyle === "minimal" || skipStroke ? undefined : "#000000",
        strokeWidth:
          s.captionStyle === "minimal" || skipStroke
            ? 0
            : isOpus
              ? Math.max(14, Math.round(size * 0.22))
              : 14,
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
  if (s.showHook && job.hook) faces.set('800 50px "Montserrat"', cleanHookTitle(job.hook, job.words));
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

export function pickMime() {
  const opts = [
    "video/mp4;codecs=avc1.64002A,mp4a.40.2",
    "video/mp4;codecs=avc1.640028,mp4a.40.2",
    "video/mp4;codecs=avc1.4D4028,mp4a.40.2",
    "video/mp4",
    "video/webm;codecs=h264,opus",
    "video/webm;codecs=vp9,opus",
    "video/webm",
  ];
  return opts.find((m) => MediaRecorder.isTypeSupported(m)) ?? "video/webm";
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

const NAMED_RATES: [string, number][] = [
  ["23.976", 24000 / 1001],
  ["24", 24],
  ["25", 25],
  ["29.97", 30000 / 1001],
  ["30", 30],
  ["59.94", 60000 / 1001],
  ["60", 60],
];

/**
 * The rate the master should write: what the capture actually delivered, snapped to the nearest named
 * rate. Never the requested rate when delivery fell short — padding a short pass up to 29.97 is what
 * turns skipped frames into a stutter, and it lets the file claim a frame rate it does not have.
 */
function masterRateText(_measured: number, requested: number): string {
  // Always master at the requested broadcast constant frame rate (29.97 or 30 fps) — never write
  // a throttled 8.8 fps / 9.1 fps rational into the final H.264 MP4 container.
  for (const [label, value] of NAMED_RATES) if (Math.abs(requested - value) < 0.05) return label;
  return requested >= 29.98 ? "30" : "29.97";
}

/**
 * Draw and capture one canvas frame per video frame the decoder presents. Reports the frame rate that
 * actually came out of the capture (frames drawn / span of their presentation timestamps) and how much
 * of the requested window the pass really covered — a backgrounded tab stops presenting frames, and a
 * truncated pass must never be mistaken for a finished clip.
 */
async function capturePresentedFrames(
  v: HTMLVideoElement,
  ctx: CanvasRenderingContext2D,
  job: ClipJob,
  canvasTrack: (MediaStreamTrack & { requestFrame?: () => void }) | undefined,
  dur: number,
  signal: AbortSignal | undefined,
  onProgress: (p: number, stageLabel?: string) => void,
  fallbackFps: number,
): Promise<{ fps: number; covered: number }> {
  const requestNext = (
    v as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }
  ).requestVideoFrameCallback;
  const stamps: number[] = [];
  const startedAt = performance.now();
  const minFrameIntervalMs = Math.max(12, (1000 / fallbackFps) * 0.72);
  let lastDrawPerf = 0;

  return new Promise<{ fps: number; covered: number }>((resolve) => {
    let pace: ReturnType<typeof setInterval> | null = null;
    let watchdog: ReturnType<typeof setInterval> | null = null;
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      if (pace !== null) clearInterval(pace);
      if (watchdog !== null) clearInterval(watchdog);
      const covered = Math.max(0, Math.min(1, (v.currentTime - job.start) / Math.max(0.1, dur)));
      const trimmed = stamps.length > 30 ? stamps.slice(2) : stamps;
      const span = trimmed.length > 1 ? (trimmed[trimmed.length - 1]! - trimmed[0]!) / 1000 : 0;
      if (trimmed.length < 12 || span <= 0.25) {
        resolve({ fps: covered >= 0.9 ? fallbackFps : 0, covered });
        return;
      }
      resolve({ fps: Math.min(fallbackFps, (trimmed.length - 1) / span), covered });
    };
    const renderOneFrame = () => {
      if (finished) return;
      if (signal?.aborted || v.currentTime >= job.end || v.ended) {
        finish();
        return;
      }
      const now = performance.now();
      if (now - lastDrawPerf < minFrameIntervalMs) return;
      lastDrawPerf = now;
      stamps.push(now);
      drawFrame(ctx, v, v.currentTime, job);
      canvasTrack?.requestFrame?.();
      onProgress(
        Math.min(0.88, (v.currentTime - job.start) / Math.max(0.1, dur)),
        "Capturing smooth 29.97fps composition…",
      );
    };
    const onVideoFrame = () => {
      if (finished) return;
      renderOneFrame();
      if (requestNext) requestNext.call(v, onVideoFrame);
    };

    // Always run a steady CFR cadence timer alongside requestVideoFrameCallback so animated captions
    // and video frames never drop to 8.8 fps or freeze on Frame 0 if the browser throttles callbacks.
    if (requestNext) requestNext.call(v, onVideoFrame);
    pace = setInterval(renderOneFrame, Math.round(1000 / fallbackFps));
    watchdog = setInterval(() => {
      const lastActivity = stamps.length ? stamps[stamps.length - 1]! : startedAt;
      if (performance.now() - lastActivity <= (stamps.length ? 3000 : 9000)) return;
      finish();
    }, 500);
  });
}

export async function exportClip(
  src: string,
  job: ClipJob,
  onProgress: (p: number, stageLabel?: string) => void,
  signal?: AbortSignal,
  backend?: { sourceId?: string | undefined; file?: File | Blob | null | undefined },
): Promise<ExportResult> {
  if (signal?.aborted) throw new Error("cancelled");
  // Native export is mandatory except for the explicit development A/B switch. Never silently
  // replace a failed renderer or missing overlay API with a real-time browser recording.
  if (!browserCaptureForced()) {
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

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  // `desynchronized` trades a copy per frame for direct scanout, and a copy is exactly what the
  // capture path needs to hand the frame to the encoder. Without it the pass keeps up with playback.
  const firstContext = canvas.getContext("2d", { alpha: false });
  if (!firstContext) throw new Error("Your browser cannot create a video canvas.");
  const ctx: CanvasRenderingContext2D = firstContext;
  await document.fonts.ready;

  const dur = Math.max(0.4, job.end - job.start);
  // Measured from the frames the decoder actually presented; the master encodes at this rate so it
  // never has to duplicate a frame to reach a rate the capture did not deliver.
  const requestedFps = job.settings.fps === "30" ? 30 : 30000 / 1001;
  const mime = pickMime();

  /**
   * One real-time pass of the whole composition into a recorded blob, with the frame rate that
   * actually came out. A pass needs its own video element and AudioContext — a media element can only
   * ever be routed into one graph — so each attempt builds both from scratch.
   */
  async function capturePass(): Promise<{ parts: Blob[]; fps: number; covered: number }> {
    const v = document.createElement("video");
    v.src = src;
    v.playsInline = true;
    v.crossOrigin = "anonymous";
    v.preload = "auto";
    // CRITICAL CHROMIUM FIX: Mount `<video>` in a 1x1px viewport-resident container during export!
    // An unmounted off-DOM `<video>` has its hardware frame decoding suspended after Frame 0 while
    // WebAudio keeps playing — which caused the 8.8fps Frame-0 freeze on podcast exports.
    v.style.cssText =
      "position:fixed;bottom:0;right:0;width:2px;height:2px;opacity:0.001;pointer-events:none;z-index:-1;";
    if (typeof document !== "undefined" && document.body) {
      document.body.appendChild(v);
    }
    await new Promise((r, rej) => {
      v.onloadeddata = r;
      v.onerror = () => rej(new Error("Failed to load source video for export."));
    });

    const ac = new AudioContext({ sampleRate: 48000 });
    const node = ac.createMediaElementSource(v);
    const dest = ac.createMediaStreamDestination();

    const limiter = ac.createDynamicsCompressor();
    limiter.threshold.value = -1.0;
    limiter.knee.value = 0.0;
    limiter.ratio.value = 20.0;
    limiter.attack.value = 0.001;
    limiter.release.value = 0.05;

    if (job.settings.voiceEnhance ?? true) {
      const hp = ac.createBiquadFilter();
      hp.type = "highpass";
      hp.frequency.value = 75;
      const comp = ac.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.knee.value = 10;
      comp.ratio.value = 2.8;
      comp.attack.value = 0.005;
      comp.release.value = 0.18;
      const gain = ac.createGain();
      gain.gain.value = 1.15;
      node.connect(hp).connect(comp).connect(gain).connect(limiter);
    } else {
      node.connect(limiter);
    }

    // Edit-style fade in/out so clips don't start or end on a hard audio cut.
    const fader = ac.createGain();
    limiter.connect(fader).connect(dest);

    if (Math.abs(v.currentTime - job.start) > 0.001) {
      await new Promise<void>((res) => {
        v.onseeked = () => res();
        v.currentTime = job.start;
      });
    }
    drawFrame(ctx, v, job.start, job);

    // captureStream(0) plus requestFrame() puts exactly one canvas frame on the wire per frame the
    // decoder presents. A fixed-rate capture plus a wall-clock interval instead samples the canvas on
    // its own clock, so a hiccup duplicates one video frame and skips the next — the judder that reads
    // as lag once the master re-timed everything to a constant rate.
    const stream = canvas.captureStream(requestedFps);
    const canvasTrack = stream.getVideoTracks()[0] as
      (MediaStreamTrack & { requestFrame?: () => void }) | undefined;
    dest.stream.getAudioTracks().forEach((tr) => stream.addTrack(tr));

    const rec = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: 16_000_000,
      audioBitsPerSecond: 288_000,
    });
    const parts: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    const done = new Promise<void>((r) => {
      rec.onstop = () => r();
    });

    let capture = { fps: 0, covered: 0 };
    try {
      rec.start(200);
      await v.play();
      const a0 = ac.currentTime;
      fader.gain.setValueAtTime(0.0001, a0);
      fader.gain.linearRampToValueAtTime(1, a0 + Math.min(0.12, dur * 0.25));
      fader.gain.setValueAtTime(1, a0 + Math.max(0.15, dur - 0.3));
      fader.gain.linearRampToValueAtTime(0.0001, a0 + dur);
      capture = await capturePresentedFrames(
        v,
        ctx,
        job,
        canvasTrack,
        dur,
        signal,
        onProgress,
        requestedFps,
      );
    } finally {
      v.pause();
      if (rec.state !== "inactive") rec.stop();
      await done;
      stream.getTracks().forEach((tr) => tr.stop());
      await ac.close().catch(() => {});
      // Clearing `src` alone leaves the hardware decoder attached until the element is collected, and
      // a fourth element in the same page stops presenting frames at all. load() after removing the
      // attribute tears the media pipeline down now, so the next pass gets a decoder.
      v.removeAttribute("src");
      v.load();
      v.remove();
    }
    return { parts, fps: capture.fps, covered: capture.covered };
  }

  let pass = await capturePass();
  // The first pass pays for every styled word once, and a media element only starts presenting at
  // full rate after it has settled. If that pass fell behind, the retry draws from the warm caches
  // and is the pass that gets shipped — a repeat of the same content costs far more than it saves.
  if ((pass.fps < 0.8 * requestedFps || pass.covered < 0.97) && !signal?.aborted) {
    onProgress(0.03, "Re-capturing with a warmed renderer…");
    const again = await capturePass();
    if (again.covered * again.fps > pass.covered * pass.fps) pass = again;
  }
  if (signal?.aborted) throw new Error("cancelled");
  // A pass that never reached the end of the window is a truncated clip, not a shorter export: the
  // encoder would happily master the fragment and report it as verified.
  if (pass.covered < 0.97) {
    throw new Error(
      `The capture only reached ${Math.round(pass.covered * 100)}% of the clip. Keep this tab in the foreground and export again.`,
    );
  }
  const capturedFps = pass.fps;
  const rawBlob = new Blob(pass.parts, { type: mime.split(";")[0] ?? "video/mp4" });
  // Blob construction copies the bytes, so leaving the recorder's chunk array alive holds the whole
  // capture twice. The master response is another ~100 MB, and the renderer refuses the transfer
  // long before the disk does.
  pass.parts.length = 0;

  try {
    const health = await fetch(`${HELPER}/health`, { signal: AbortSignal.timeout(1000) });
    if (health.ok) {
      const hData = (await health.json()) as { encoder?: string | null };
      const encLabel =
        hData.encoder === "h264_nvenc"
          ? "Smoothing & mastering on NVIDIA NVENC…"
          : "Smoothing & mastering via local FFmpeg…";
      onProgress(0.92, encLabel);
      // FormData is single-use once the body is consumed, so every attempt rebuilds it around the
      // same blob. Returns null when the worker answered with a real rejection — repeating a failed
      // render only costs another minute — and throws for transport faults, which are worth a retry.
      const masterOnce = async (): Promise<ExportResult | null> => {
        const fd = new FormData();
        fd.append("file", rawBlob, "raw_capture.webm");
        fd.append("fps", masterRateText(capturedFps, requestedFps));
        fd.append("voiceEnhance", String(job.settings.voiceEnhance ?? true));
        const reqInit: RequestInit = { method: "POST", body: fd };
        if (signal) reqInit.signal = signal;
        const mastered = await fetch(`${HELPER}/master`, reqInit);
        if (!mastered.ok) {
          console.warn(
            `Local master rejected with HTTP ${mastered.status}; shipping browser stream`,
          );
          return null;
        }
        // Chrome refuses `response.blob()` once a body is big enough to need blob storage: it throws
        // TypeError: Failed to fetch within milliseconds, with the bytes already delivered and the
        // master already written. Taking the buffer and wrapping it costs ~1.6 s per 100 MB and is
        // the difference between shipping an NVENC master and silently shipping the browser stream.
        const mp4Blob = new Blob([await mastered.arrayBuffer()], { type: "video/mp4" });
        const encoderUsed = mastered.headers.get("X-Video-Encoder") || hData.encoder || "libx264";
        const verified = mastered.headers.get("X-FFprobe-Verified") === "true";
        const sha256 = await computeSha256Hex(mp4Blob);
        onProgress(1, "Master complete");
        return {
          blob: mp4Blob,
          mode: encoderUsed === "h264_nvenc" ? "nvenc" : "ffmpeg_cpu",
          encoder: encoderUsed,
          verified,
          sha256,
          deliveredFps: capturedFps,
        };
      };
      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const out = await masterOnce();
          if (out) return out;
          break;
        } catch (e) {
          if (signal?.aborted || attempt === 2) throw e;
          console.warn(`Local master attempt ${attempt} broke in transport; retrying once`, e);
          await new Promise((r) => setTimeout(r, 1500));
        }
      }
    } else {
      console.warn(
        `Local master skipped: worker health HTTP ${health.status}; shipping browser stream`,
      );
    }
  } catch (e) {
    if (signal?.aborted || (e as Error).name === "AbortError") throw new Error("cancelled");
    console.warn("Local master failed; shipping browser stream instead", e);
  }

  const sha256 = await computeSha256Hex(rawBlob);
  if (signal?.aborted) throw new Error("cancelled");
  onProgress(1, "Browser fallback export complete");
  return {
    blob: rawBlob,
    mode: "browser_fallback",
    encoder: mime,
    verified: false,
    sha256,
    deliveredFps: capturedFps,
  };
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
