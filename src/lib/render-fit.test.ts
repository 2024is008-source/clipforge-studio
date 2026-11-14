import { afterEach, describe, expect, it, vi } from "vitest";
import { drawFrame, fitGeometry, H, W, type ClipJob } from "./render";

const sources = [
  ["landscape", 1920, 1080],
  ["portrait", 1080, 1920],
  ["tall portrait", 1080, 2400],
  ["ultrawide", 3440, 1440],
  ["square", 1080, 1080],
  ["odd dimensions", 1919, 1079],
] as const;

afterEach(() => vi.restoreAllMocks());

describe.each(sources)("BLUR + FIT: %s", (_name, sourceW, sourceH) => {
  it("contains the entire source, centered and without changing its aspect ratio", () => {
    const fit = fitGeometry(sourceW, sourceH);
    expect(fit.x).toBeGreaterThanOrEqual(0);
    expect(fit.y).toBeGreaterThanOrEqual(0);
    expect(fit.x + fit.width).toBeLessThanOrEqual(W);
    expect(fit.y + fit.height).toBeLessThanOrEqual(H);
    expect(fit.width / fit.height).toBeCloseTo(sourceW / sourceH, 10);
    expect(fit.x * 2 + fit.width).toBeCloseTo(W, 10);
    expect(fit.y * 2 + fit.height).toBeCloseTo(H, 10);
    expect(Math.max(fit.width / W, fit.height / H)).toBeCloseTo(1, 10);
  });

  it.each(["fit", "auto"] as const)("draws every source pixel despite punch and manual offsets (%s)", (layout) => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const drawImage = vi.fn();
    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      fillRect: vi.fn(),
      drawImage,
      createRadialGradient: () => ({ addColorStop: vi.fn() }),
    } as unknown as CanvasRenderingContext2D;
    const video = { videoWidth: sourceW, videoHeight: sourceH } as HTMLVideoElement;
    const job: ClipJob = {
      start: 10,
      end: 12,
      hook: "",
      words: [{ w: "Everything!", s: 10, e: 11 }],
      track: [{ t: 10, x: 0.05, y: 0.1, zoom: 1.8, mode: "fit" }],
      settings: {
        layout,
        captionStyle: "none",
        highlight: "#22ff4e",
        captionY: 0.78,
        showHook: false,
        progressBar: false,
        zoomPunch: true,
        manualX: 0.95,
        manualY: 0.9,
        wordsPerPage: 3,
      },
    };
    drawFrame(ctx, video, 10.5, job);
    const fit = fitGeometry(sourceW, sourceH);
    expect(drawImage).toHaveBeenCalledExactlyOnceWith(
      video, 0, 0, sourceW, sourceH, fit.x, fit.y, fit.width, fit.height,
    );
  });
});

it("uses the requested output bounds rather than a fixed-height card", () => {
  expect(fitGeometry(1920, 1080)).toEqual({ x: 0, y: 656.25, width: 1080, height: 607.5 });
  expect(fitGeometry(1080, 2400)).toEqual({ x: 108, y: 0, width: 864, height: 1920 });
  expect(fitGeometry(1080, 1920, 1280, 720)).toEqual({ x: 437.5, y: 0, width: 405, height: 720 });
});

it("places fit captions safely below the foreground video even when hasBottomGraphic is set", async () => {
  const { drawFrameOverlays } = await import("./render");
  const drawnY: number[] = [];
  const ctx = {
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn((_x: number, y: number) => {
      if (y > 0) drawnY.push(y);
    }),
    scale: vi.fn(),
    measureText: vi.fn((text: string) => ({ width: text.length * 24 })),
    fillText: vi.fn((_text: string, _x: number, y: number) => { drawnY.push(y); }),
    strokeText: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    quadraticCurveTo: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    drawImage: vi.fn((_img: unknown, _sx: number, _sy: number, _sw: number, _sh: number, _dx?: number, dy?: number) => {
      if (typeof dy === "number") drawnY.push(dy);
    }),
    imageSmoothingEnabled: true,
    imageSmoothingQuality: "high",
    filter: "none",
    font: "",
    textAlign: "left",
    textBaseline: "middle",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineJoin: "round",
    shadowColor: "",
    shadowBlur: 0,
    shadowOffsetY: 0,
    globalAlpha: 1,
  } as unknown as CanvasRenderingContext2D;

  const spriteCtx = {
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    measureText: vi.fn((text: string) => ({ width: text.length * 24 })),
    fillText: vi.fn(),
    strokeText: vi.fn(),
    clearRect: vi.fn(),
    font: "",
    textAlign: "left",
    textBaseline: "middle",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineJoin: "round",
    shadowColor: "",
    shadowBlur: 0,
    shadowOffsetY: 0,
  };
  const origCreateElement = globalThis.document?.createElement;
  vi.stubGlobal("document", {
    createElement: vi.fn(() => ({ width: 0, height: 0, getContext: () => spriteCtx })),
  });
  try {
    const job: ClipJob = {
      start: 0,
      end: 3,
      hook: "",
      words: [
        { w: "What", s: 0.1, e: 0.4 },
        { w: "I", s: 0.4, e: 0.6 },
        { w: "had", s: 0.6, e: 0.9 },
      ],
      track: [{ t: 0, x: 0.5, y: 0.35, mode: "fit", hasBottomGraphic: true }],
      settings: {
        layout: "fit",
        captionStyle: "opus",
        highlight: "#22ff4e",
        captionY: 0.65,
        showHook: false,
        showEmojis: false,
        progressBar: false,
        zoomPunch: false,
        wordsPerPage: 3,
      },
    };
    drawFrameOverlays(ctx, 0.5, job);
    const fit = fitGeometry(1920, 1080, W, H, true);
    const fgBottom = fit.y + fit.height; // 1263.75
    expect(drawnY.length).toBeGreaterThan(0);
    for (const y of drawnY) {
      expect(y).toBeGreaterThan(fgBottom + 100);
    }
  } finally {
    if (origCreateElement) {
      vi.stubGlobal("document", { createElement: origCreateElement });
    } else {
      vi.unstubAllGlobals();
    }
  }
});
