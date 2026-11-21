import { describe, expect, it } from "vitest";
import {
  cubicBezier,
  DEFAULT_MOTION,
  easeSmoothstep,
  motionSpec,
  type Bezier,
} from "./motion";

describe("cubicBezier easing", () => {
  it("pins the endpoints exactly", () => {
    const ease = cubicBezier(0.34, 1.56, 0.64, 1);
    expect(ease(0)).toBe(0);
    expect(ease(1)).toBe(1);
  });

  it("reproduces the linear identity curve", () => {
    const linear = cubicBezier(1 / 3, 1 / 3, 2 / 3, 2 / 3);
    for (const x of [0, 0.25, 0.5, 0.75, 1]) expect(linear(x)).toBeCloseTo(x, 2);
  });

  it("clamps progress outside the unit window", () => {
    const ease = cubicBezier(0.2, 0, 0.2, 1);
    expect(ease(-1)).toBe(0);
    expect(ease(2)).toBe(1);
  });

  it("overshoots past 1 for a back curve mid-progress", () => {
    const back = cubicBezier(0.34, 1.56, 0.64, 1);
    let peak = 0;
    for (let x = 0; x <= 1; x += 0.02) peak = Math.max(peak, back(x));
    expect(peak).toBeGreaterThan(1);
  });
});

describe("motionSpec presets", () => {
  it("defaults to the shipped smooth curve when unset", () => {
    expect(motionSpec(undefined)).toEqual(DEFAULT_MOTION);
    expect(motionSpec("smooth")).toEqual(DEFAULT_MOTION);
  });

  it("makes snappy faster and less bouncy than smooth", () => {
    const snappy = motionSpec("snappy");
    expect(snappy.attack).toBeLessThan(DEFAULT_MOTION.attack);
    expect(snappy.peak).toBeLessThan(DEFAULT_MOTION.peak);
  });

  it("makes bounce overshoot harder and ride longer", () => {
    const bounce = motionSpec("bounce");
    expect(bounce.peak).toBeGreaterThan(DEFAULT_MOTION.peak);
    expect(bounce.settle).toBeGreaterThan(DEFAULT_MOTION.settle);
  });

  it("uses the supplied bezier for the custom preset", () => {
    const curve: Bezier = [0.34, 1.56, 0.64, 1];
    const custom = motionSpec("custom", curve);
    expect(custom.ease(0)).toBe(0);
    expect(custom.ease(1)).toBe(1);
    // custom differs from smoothstep mid-progress because the bezier control points differ
    expect(Math.abs(custom.ease(0.4) - easeSmoothstep(0.4))).toBeGreaterThan(0.01);
  });

  it("falls back to smooth for custom with no curve", () => {
    expect(motionSpec("custom", null)).toEqual(DEFAULT_MOTION);
  });
});
