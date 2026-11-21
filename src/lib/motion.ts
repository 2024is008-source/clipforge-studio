// Shared motion shaping for the "pop" elements (hook cards, number callouts, emojis) and caption
// word pops. One source of truth so the interactive preview and the exported overlay PNGs — which
// are rasterised from this same draw path — animate identically.
export type MotionPreset = "smooth" | "snappy" | "bounce" | "custom";
export type Bezier = readonly [number, number, number, number];

export type Easing = (x: number) => number;

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

export const easeSmoothstep: Easing = (x) => {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
};

export const easeOutCubic: Easing = (x) => 1 - Math.pow(1 - clamp01(x), 3);

/**
 * CSS-style cubic-bezier easing solver: given control points (x1,y1,x2,y2) on the unit square,
 * return y for progress x. Newton-Raphson with a bisection guard so a non-monotonic handle still
 * resolves instead of diverging.
 */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number): Easing {
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const sampleX = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sampleY = (t: number) => ((ay * t + by) * t + cy) * t;
  const sampleDX = (t: number) => (3 * ax * t + 2 * bx) * t + cx;
  return (p) => {
    const x = clamp01(p);
    if (x === 0 || x === 1) return x;
    let t = x;
    for (let i = 0; i < 8; i++) {
      const err = sampleX(t) - x;
      if (Math.abs(err) < 1e-6) break;
      const d = sampleDX(t);
      if (Math.abs(d) < 1e-6) break;
      t -= err / d;
    }
    if (t < 0 || t > 1) {
      let lo = 0;
      let hi = 1;
      t = x;
      for (let i = 0; i < 20; i++) {
        const err = sampleX(t) - x;
        if (Math.abs(err) < 1e-6) break;
        if (err > 0) hi = t;
        else lo = t;
        t = (lo + hi) / 2;
      }
    }
    return sampleY(clamp01(t));
  };
}

export type MotionSpec = {
  /** Seconds from trigger to peak scale. */
  attack: number;
  /** Seconds from peak back to rest. */
  settle: number;
  /** Peak scale multiplier of the overshoot. */
  peak: number;
  /** Curve shaping both the attack and the settle ramp. */
  ease: Easing;
};

export const DEFAULT_MOTION: MotionSpec = {
  attack: 0.16,
  settle: 0.14,
  peak: 1.1,
  ease: easeSmoothstep,
};

const PRESETS: Record<Exclude<MotionPreset, "custom">, MotionSpec> = {
  // "smooth" is the shipped look: gentle overshoot, symmetric-ish settle.
  smooth: DEFAULT_MOTION,
  // "snappy" lands almost immediately with a hair of overshoot — reads punchy for hype/fast cuts.
  snappy: { attack: 0.09, settle: 0.1, peak: 1.04, ease: easeOutCubic },
  // "bounce" pushes harder and rides the settle longer — playful / reaction content.
  bounce: { attack: 0.2, settle: 0.34, peak: 1.28, ease: easeSmoothstep },
};

/** Resolve a preset (plus an optional custom bezier) into concrete pop parameters. */
export function motionSpec(preset: MotionPreset | undefined, curve?: Bezier | null): MotionSpec {
  if (preset === "custom") {
    const base = PRESETS.smooth;
    if (!curve) return base;
    const [x1, y1, x2, y2] = curve;
    return { ...base, ease: cubicBezier(clamp01(x1), y1, clamp01(x2), y2) };
  }
  return PRESETS[preset ?? "smooth"] ?? DEFAULT_MOTION;
}
