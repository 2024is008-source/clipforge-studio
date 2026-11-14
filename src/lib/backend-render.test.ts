import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BackendUnavailable, browserCaptureForced, decisionFromJob, renderWithLocalBackend } from "./backend-render";
import { exportClip, type ClipJob } from "./render";
import { uploadRenderOverlays } from "./render-overlays";

vi.mock("@/components/YouTubeImport", () => ({ HELPER: "http://localhost:8765" }));
vi.mock("./local-worker", () => ({ workerSourceId: vi.fn() }));
vi.mock("./reframe", () => ({ sampleTrack: vi.fn() }));
vi.mock("./render-overlays", async (importOriginal) => ({
  ...await importOriginal<typeof import("./render-overlays")>(),
  uploadRenderOverlays: vi.fn(),
}));

const job: ClipJob = {
  start: 120.5,
  end: 146.0,
  hook: "The runway was far",
  words: [
    { w: "The", s: 120.5, e: 120.8 },
    { w: "runway", s: 120.8, e: 121.4 },
    { w: "far!", s: 121.4, e: 122.0 },
  ],
  track: [
    { t: 120.5, x: 0.42, y: 0.31, mode: "fill", zoom: 1.08, cut: true, subjectType: "face" },
    { t: 124.0, x: 0.55, y: 0.28, mode: "split", x2: 0.71, y2: 0.33, hasBottomGraphic: true },
  ],
  settings: {
    layout: "split",
    captionStyle: "opus",
    highlight: "#22ff4e",
    secondaryHighlight: "#ffd400",
    captionY: 0.78,
    showHook: true,
    showEmojis: true,
    progressBar: false,
    zoomPunch: true,
    wordsPerPage: 3,
    fps: "29.97",
    voiceEnhance: true,
    manualX: null,
    manualY: null,
  },
};

describe("decisionFromJob", () => {
  it("carries the preview's edit in the keys the renderer validates", () => {
    const d = decisionFromJob(job);
    expect(d.start).toBe(120.5);
    expect(d.end).toBe(146.0);
    expect(d.layout).toBe("split");
    expect(d.hook).toBe("The runway was far");
    // decision_from_payload requires these four; omitting any one of them fails the job at 400.
    for (const key of ["start", "end", "layout", "track"] as const) expect(d[key]).toBeDefined();
  });

  it("sends track and word times in absolute source seconds, as the preview draws them", () => {
    const d = decisionFromJob(job);
    expect(d.track[0]).toMatchObject({ t: 120.5, x: 0.42, y: 0.31, mode: "fill", zoom: 1.08 });
    expect(d.track[1]).toMatchObject({
      x2: 0.71,
      y2: 0.33,
      hasBottomGraphic: true,
      subjectType: "face",
      cut: false,
    });
    expect(d.words[2]).toEqual({ w: "far!", start: 121.4, end: 122.0 });
  });

  it("maps the caption settings onto the style object", () => {
    const d = decisionFromJob(job);
    expect(d.style).toMatchObject({
      preset: "opus",
      highlight: "#22ff4e",
      secondaryHighlight: "#ffd400",
      positionY: 0.78,
      wordsPerPage: 3,
      showHook: true,
      font: "Montserrat",
    });
  });

  it("leaves geometry to the worker instead of guessing at it", () => {
    const d = decisionFromJob(job);
    expect(d).not.toHaveProperty("source_width");
    expect(d).not.toHaveProperty("source_fps");
  });
});

const sourceFps = 24000 / 1001;
const nativeJob: ClipJob = {
  ...job,
  settings: { ...job.settings, fps: "30" },
  track: [{
    t: job.start, x: 0.5, y: 0.3, mode: "fit", trackingConfidence: 0,
    layoutDecision: { scene: 2, reason: "low_tracking_confidence" },
  }],
};
const base = "http://localhost:8765/v1";
const sourceId = "cached/source";
const sourceUrl = `${base}/sources/cached%2Fsource`;
const infoUrl = `${sourceUrl}/render-info?fps=30`;
const overlayId = "overlay/one";
const overlayUrl = `${base}/render/overlays/overlay%2Fone`;
const decisionUrl = `${base}/render/decisions`;
const jobUrl = `${base}/render/jobs/job%2Fone`;
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
const fetcher = vi.fn<typeof fetch>();
const upload = vi.mocked(uploadRenderOverlays);
const progress = vi.fn();
const timeout = vi.fn((_ms: number) => new AbortController().signal);
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.useFakeTimers();
  vi.stubGlobal("fetch", fetcher);
  vi.stubGlobal("AbortSignal", { timeout });
  vi.stubGlobal("localStorage", { getItem: vi.fn().mockReturnValue(null) });
  fetcher.mockRejectedValue(new Error("Unexpected fetch"));
  upload.mockResolvedValue(overlayId);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("native render transport", () => {
  it("resolves server FPS before PNGs, sends overlayId and provenance, downloads, and cleans up", async () => {
    const info = deferred<Response>();
    const controller = new AbortController();
    fetcher.mockReturnValueOnce(info.promise)
      .mockResolvedValueOnce(json({ job_id: "job/one", encoder: "ffmpeg" }))
      .mockResolvedValueOnce(json({ status: "completed", progress: 1, result: { encoder_kind: "gpu", qc: { passed: true } } }))
      .mockResolvedValueOnce(new Response("mp4", { headers: { "X-Video-Encoder": "h264_nvenc", "X-FFprobe-Verified": "true" } }))
      .mockResolvedValueOnce(json({}));
    upload.mockImplementationOnce(async (_job, _fps, onProgress) => {
      onProgress(0, "Preparing PNGs");
      onProgress(1, "PNGs ready");
      return overlayId;
    });
    const pending = renderWithLocalBackend(nativeJob, sourceId, progress, controller.signal);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(infoUrl, { signal: controller.signal });
    expect(upload).not.toHaveBeenCalled();
    info.resolve(json({ fps: sourceFps }));
    await vi.advanceTimersByTimeAsync(800);
    const result = await pending;

    expect(upload).toHaveBeenCalledExactlyOnceWith(nativeJob, sourceFps, expect.any(Function), controller.signal);
    expect(upload.mock.invocationCallOrder[0]).toBeLessThan(fetcher.mock.invocationCallOrder[1]!);
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      infoUrl, decisionUrl, jobUrl, `${jobUrl}/download`, overlayUrl,
    ]);
    const body = JSON.parse(fetcher.mock.calls[1]![1]!.body as string);
    expect(body).toMatchObject({
      sourceId, overlayId, punchGain: 1,
      decision: {
        fps: "30", start: nativeJob.start, end: nativeJob.end,
        track: [{ mode: "fit", trackingConfidence: 0, layoutDecision: { scene: 2, reason: "low_tracking_confidence" } }],
      },
    });
    expect(body.decision).toEqual(JSON.parse(JSON.stringify(decisionFromJob(nativeJob))));
    expect(fetcher.mock.calls[1]![1]).toMatchObject({ method: "POST", headers: { "Content-Type": "application/json" } });
    expect(timeout.mock.calls).toEqual([[30000], [5000]]);
    expect(fetcher.mock.calls[1]![1]!.signal).not.toBe(controller.signal);
    expect(result).toMatchObject({
      encoder: "h264_nvenc", encoderKind: "gpu", verified: true, deliveredFps: sourceFps,
      duration: nativeJob.end - nativeJob.start, elapsedSec: 0, qc: { passed: true },
    });
    expect(result.blob.type).toBe("video/mp4");
    expect(result.blob.size).toBe(3);
    expect(progress).toHaveBeenCalledWith(0.02, "Preparing PNGs");
    expect(progress).toHaveBeenCalledWith(0.35, "PNGs ready");
    expect(progress).toHaveBeenLastCalledWith(1, "Render verified");
    expect(fetcher).toHaveBeenLastCalledWith(overlayUrl, { method: "DELETE", signal: expect.anything() });
  });

  it("uploads a missing source on 409, then refetches render-info before preparing PNGs", async () => {
    const file = new File(["video"], "original.mp4", { type: "video/mp4" });
    const controller = new AbortController();
    const stop = new Error("stop after timing resolution");
    fetcher.mockResolvedValueOnce(json({ detail: "missing source" }, 409))
      .mockResolvedValueOnce(json({})).mockResolvedValueOnce(json({ fps: sourceFps }));
    upload.mockRejectedValueOnce(stop);
    await expect(renderWithLocalBackend(nativeJob, sourceId, progress, controller.signal, { file })).rejects.toBe(stop);
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([infoUrl, sourceUrl, infoUrl]);
    const request = fetcher.mock.calls[1]![1]!;
    expect(request).toMatchObject({ method: "POST", signal: controller.signal });
    const uploadedFile = (request.body as FormData).get("file") as File;
    expect(uploadedFile.name).toBe("original.mp4");
    expect(uploadedFile.size).toBe(file.size);
    expect(uploadedFile.type).toBe("video/mp4");
    expect(upload).toHaveBeenCalledExactlyOnceWith(nativeJob, sourceFps, expect.any(Function), controller.signal);
    expect(upload.mock.invocationCallOrder[0]).toBeGreaterThan(fetcher.mock.invocationCallOrder[2]!);
  });

  it("reports a missing source without PNGs when no upload is available", async () => {
    fetcher.mockResolvedValueOnce(json({ detail: "missing source" }, 409));
    await expect(renderWithLocalBackend(nativeJob, sourceId, progress)).rejects.toBeInstanceOf(BackendUnavailable);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(upload).not.toHaveBeenCalled();
  });

  it.each([undefined, "30", 5.99, 60.01, null])("rejects invalid server FPS %s before PNGs", async (fps) => {
    fetcher.mockResolvedValueOnce(json({ fps }));
    await expect(renderWithLocalBackend(nativeJob, sourceId, progress)).rejects.toThrow("invalid output frame rate");
    expect(upload).not.toHaveBeenCalled();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it.each([404, 409, 422, 503, 507, 500])("cleans overlays after queue HTTP %s without cancelling an unknown job", async (status) => {
    fetcher.mockResolvedValueOnce(json({ fps: sourceFps }))
      .mockResolvedValueOnce(json({ detail: "queue failed" }, status))
      .mockRejectedValueOnce(new Error("cleanup offline"));
    await expect(renderWithLocalBackend(nativeJob, sourceId, progress)).rejects.toBeInstanceOf(
      status === 500 ? Error : BackendUnavailable,
    );
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([infoUrl, decisionUrl, overlayUrl]);
    expect(fetcher).toHaveBeenLastCalledWith(overlayUrl, { method: "DELETE", signal: expect.anything() });
    expect(timeout).toHaveBeenLastCalledWith(5000);
  });

  it.each(["poll error", "lost contact", "download"])("cancels before releasing overlays after %s failure", async (failure) => {
    fetcher.mockResolvedValueOnce(json({ fps: sourceFps }))
      .mockResolvedValueOnce(json({ job_id: "job/one" }));
    if (failure === "poll error") fetcher.mockResolvedValueOnce(json({ status: "error", error: "encoder failed" }));
    else if (failure === "lost contact") fetcher.mockRejectedValueOnce(new Error("offline"));
    else fetcher.mockResolvedValueOnce(json({ status: "completed", progress: 1 }))
      .mockResolvedValueOnce(json({ detail: "download failed" }, 500));
    fetcher.mockRejectedValueOnce(new Error("cancel offline")).mockResolvedValueOnce(json({}));
    const pending = renderWithLocalBackend(nativeJob, sourceId, progress);
    const rejected = expect(pending).rejects.toThrow(
      failure === "poll error" ? "encoder failed" : failure === "lost contact" ? "Lost contact" : "download failed",
    );
    await vi.advanceTimersByTimeAsync(800);
    await rejected;
    expect(fetcher.mock.calls.slice(-2)).toEqual([
      [`${jobUrl}/cancel`, { method: "POST", signal: expect.anything() }],
      [overlayUrl, { method: "DELETE", signal: expect.anything() }],
    ]);
    expect(timeout.mock.calls).toEqual([[30000], [5000], [5000]]);
  });

  it("obtains the queued ID despite user abort, then cancels it before deleting PNGs", async () => {
    const controller = new AbortController();
    const entered = deferred<void>();
    const queued = deferred<Response>();
    fetcher.mockResolvedValueOnce(json({ fps: sourceFps }))
      .mockImplementationOnce(() => { entered.resolve(); return queued.promise; })
      .mockResolvedValueOnce(json({})).mockResolvedValueOnce(json({}));
    const pending = renderWithLocalBackend(nativeJob, sourceId, progress, controller.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    await entered.promise;
    controller.abort();
    const creationSignal = fetcher.mock.calls[1]![1]!.signal!;
    expect(timeout).toHaveBeenCalledExactlyOnceWith(30000);
    expect(creationSignal).not.toBe(controller.signal);
    expect(creationSignal.aborted).toBe(false);
    expect(fetcher).toHaveBeenCalledTimes(2);
    queued.resolve(json({ job_id: "job/one" }));
    await rejected;
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([infoUrl, decisionUrl, `${jobUrl}/cancel`, overlayUrl]);
    expect(fetcher.mock.calls[2]![1]).toMatchObject({ method: "POST" });
    expect(fetcher.mock.calls[3]![1]).toMatchObject({ method: "DELETE" });
    for (const [, init] of fetcher.mock.calls.slice(2)) {
      expect(init!.signal).not.toBe(controller.signal);
      expect(init!.signal!.aborted).toBe(false);
    }
    expect(timeout.mock.calls).toEqual([[30000], [5000], [5000]]);
  });

  it("cancels only once during polling and removes the abort listener and poll timer", async () => {
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, "removeEventListener");
    fetcher.mockResolvedValueOnce(json({ fps: sourceFps }))
      .mockResolvedValueOnce(json({ job_id: "job/one" }))
      .mockResolvedValueOnce(json({})).mockResolvedValueOnce(json({}));
    const pending = renderWithLocalBackend(nativeJob, sourceId, progress, controller.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await rejected;
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([infoUrl, decisionUrl, `${jobUrl}/cancel`, overlayUrl]);
    expect(remove).toHaveBeenCalledWith("abort", expect.any(Function));
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("no automatic browser capture fallback", () => {
  it.each(["render-info", "overlays", "queue"])("propagates %s failure through exportClip without browser capture", async (stage) => {
    const createElement = vi.spyOn(document, "createElement").mockImplementation(() => {
      throw new Error("Browser capture must not run");
    });
    const recorder = vi.fn();
    const audio = vi.fn();
    vi.stubGlobal("MediaRecorder", recorder);
    vi.stubGlobal("AudioContext", audio);
    const file = new File(["source"], "video.mp4");
    fetcher.mockResolvedValueOnce(stage === "render-info" ? json({ detail: "worker unavailable" }, 503) : json({ fps: sourceFps }));
    if (stage === "overlays") upload.mockRejectedValueOnce(new Error("PNG overlay API unavailable"));
    if (stage === "queue") fetcher.mockResolvedValueOnce(json({ detail: "queue unavailable" }, 503)).mockResolvedValueOnce(json({}));
    await expect(exportClip("blob:source", nativeJob, progress, undefined, { sourceId, file })).rejects.toThrow(
      stage === "render-info" ? "worker unavailable" : stage === "overlays" ? "PNG overlay API unavailable" : "queue unavailable",
    );
    expect(createElement).not.toHaveBeenCalled();
    expect(recorder).not.toHaveBeenCalled();
    expect(audio).not.toHaveBeenCalled();
  });

  it("requires the explicit development switch and stays native when storage is inaccessible", () => {
    expect(browserCaptureForced()).toBe(false);
    vi.mocked(localStorage.getItem).mockReturnValue("1");
    expect(browserCaptureForced()).toBe(true);
    expect(localStorage.getItem).toHaveBeenLastCalledWith("clipforge_browser_capture");
    vi.mocked(localStorage.getItem).mockReturnValue("true");
    expect(browserCaptureForced()).toBe(false);
    vi.mocked(localStorage.getItem).mockImplementation(() => { throw new Error("storage blocked"); });
    expect(browserCaptureForced()).toBe(false);
  });
});
