import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { drawFrameOverlays, prepareOverlayFonts, type ClipJob } from "./render";
import { overlayTiming, uploadRenderOverlays } from "./render-overlays";

vi.mock("@/components/YouTubeImport", () => ({ HELPER: "http://localhost:8765" }));
vi.mock("./render", () => ({ drawFrameOverlays: vi.fn(), prepareOverlayFonts: vi.fn() }));

const fps = 24000 / 1001;
const job: ClipJob = {
  start: 12.5,
  end: 12.5 + 17 / fps,
  hook: "Keep the shared visual layer",
  words: [{ w: "shared", s: 12.5, e: 13 }],
  settings: {
    layout: "fit", captionStyle: "opus", highlight: "#22ff4e", captionY: 0.78,
    showHook: true, progressBar: true, zoomPunch: false, wordsPerPage: 3, fps: "30",
  },
};
const base = "http://localhost:8765/v1/render/overlays";
const overlayId = "overlay/one";
const overlayUrl = `${base}/overlay%2Fone`;
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

const fetcher = vi.fn<typeof fetch>();
const progress = vi.fn();
const timeout = vi.fn((_ms: number) => new AbortController().signal);
const clearRect = vi.fn();
const ctx = { clearRect } as unknown as CanvasRenderingContext2D;
let canvas: HTMLCanvasElement;

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("fetch", fetcher);
  vi.stubGlobal("AbortSignal", { timeout });
  fetcher.mockRejectedValue(new Error("Unexpected fetch"));
  vi.mocked(prepareOverlayFonts).mockResolvedValue(undefined);
  canvas = document.createElement("canvas");
  vi.spyOn(document, "createElement").mockReturnValue(canvas);
  vi.spyOn(canvas, "getContext").mockReturnValue(ctx);
  vi.spyOn(canvas, "toBlob").mockImplementation((callback) => callback(new Blob(["png"], { type: "image/png" })));
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("overlayTiming", () => {
  it("uses resolved source 24000/1001 fps instead of UI 30, without an extra rounding frame", () => {
    expect(overlayTiming(job, fps)).toEqual({ fps, frameCount: 17, width: 1080, height: 1920 });
    expect(overlayTiming({ ...job, end: job.start + 1.001 }, fps).frameCount).toBe(24);
    expect(overlayTiming({ ...job, end: job.start + 1.01 }, fps).frameCount).toBe(25);
  });

  it.each([6, 60])("accepts the inclusive %s fps and 300 second bounds", (resolvedFps) => {
    expect(overlayTiming({ ...job, start: 0, end: 300 }, resolvedFps).frameCount).toBe(300 * resolvedFps);
    expect(overlayTiming({ ...job, start: 0, end: 0.001 }, resolvedFps).frameCount).toBe(1);
  });

  it.each([5.999, 60.001, NaN, Infinity, -Infinity])("rejects invalid fps %s", (resolvedFps) => {
    expect(() => overlayTiming(job, resolvedFps)).toThrow("between 6 and 60 fps");
  });

  it.each([
    [-1, 1], [NaN, 1], [Infinity, Infinity], [0, 0], [2, 1], [0, 300.001], [0, NaN], [0, Infinity],
  ])("rejects invalid clip bounds %s..%s", (start, end) => {
    expect(() => overlayTiming({ ...job, start, end }, fps)).toThrow("between 0 and 300 seconds");
  });
});

describe("uploadRenderOverlays", () => {
  it("draws exact source times on a transparent 1080x1920 canvas and uploads sequential batches of eight PNGs", async () => {
    const firstBatch = deferred<void>();
    const releaseBatch = deferred<Response>();
    const controller = new AbortController();
    fetcher.mockResolvedValueOnce(json({ overlayId }))
      .mockImplementationOnce(() => { firstBatch.resolve(); return releaseBatch.promise; })
      .mockResolvedValueOnce(json({})).mockResolvedValueOnce(json({}))
      .mockResolvedValueOnce(json({ overlayId }));

    const pending = uploadRenderOverlays(job, fps, progress, controller.signal);
    await firstBatch.promise;
    expect(prepareOverlayFonts).toHaveBeenCalledExactlyOnceWith(job);
    expect(prepareOverlayFonts).toHaveBeenCalledBefore(fetcher);
    expect(canvas.width).toBe(1080);
    expect(canvas.height).toBe(1920);
    expect(canvas.getContext).toHaveBeenCalledExactlyOnceWith("2d", { alpha: true });
    expect(drawFrameOverlays).toHaveBeenCalledTimes(8);
    expect(canvas.toBlob).toHaveBeenCalledTimes(8);
    // A pending upload must hold back drawing/encoding the next batch.
    await Promise.resolve();
    expect(drawFrameOverlays).toHaveBeenCalledTimes(8);
    releaseBatch.resolve(json({}));
    await expect(pending).resolves.toBe(overlayId);

    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      base, `${overlayUrl}/frames`, `${overlayUrl}/frames`, `${overlayUrl}/frames`, `${overlayUrl}/complete`,
    ]);
    expect(JSON.parse(fetcher.mock.calls[0]![1]!.body as string)).toEqual(overlayTiming(job, fps));
    expect(fetcher.mock.calls[0]![1]).toMatchObject({ method: "POST", headers: { "Content-Type": "application/json" } });
    expect(timeout).toHaveBeenCalledExactlyOnceWith(30000);
    expect(fetcher.mock.calls[0]![1]!.signal).not.toBe(controller.signal);
    for (const [batch, size] of [[0, 8], [1, 8], [2, 1]] as const) {
      const request = fetcher.mock.calls[batch + 1]![1]!;
      expect(request).toMatchObject({ method: "POST", signal: controller.signal });
      const body = request.body as FormData;
      expect(body.get("start")).toBe(String(batch * 8));
      const files = body.getAll("files") as File[];
      expect(files).toHaveLength(size);
      files.forEach((file, i) => {
        expect(file.name).toBe(`frame_${String(batch * 8 + i).padStart(8, "0")}.png`);
        expect(file.type).toBe("image/png");
        expect(file.size).toBe(3);
      });
    }
    expect(drawFrameOverlays).toHaveBeenCalledTimes(17);
    expect(clearRect).toHaveBeenCalledTimes(17);
    for (let n = 0; n < 17; n++) {
      expect(clearRect).toHaveBeenNthCalledWith(n + 1, 0, 0, 1080, 1920);
      expect(drawFrameOverlays).toHaveBeenNthCalledWith(n + 1, ctx, job.start + n / fps, job);
      expect(canvas.toBlob).toHaveBeenNthCalledWith(n + 1, expect.any(Function), "image/png");
      expect(clearRect.mock.invocationCallOrder[n]).toBeLessThan(vi.mocked(drawFrameOverlays).mock.invocationCallOrder[n]!);
      expect(vi.mocked(drawFrameOverlays).mock.invocationCallOrder[n]).toBeLessThan(vi.mocked(canvas.toBlob).mock.invocationCallOrder[n]!);
    }
    expect(fetcher.mock.calls[4]![1]).toMatchObject({ method: "POST", signal: controller.signal });
    expect(progress).toHaveBeenLastCalledWith(1, "Caption overlays ready");
    expect(document.createElement).toHaveBeenCalledExactlyOnceWith("canvas");
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);
    // Successful overlays stay on the worker until the backend job releases them.
    expect(fetcher.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false);
  });

  it.each([
    ["upload", "Caption overlay upload failed: disk full."],
    ["completion", "Caption overlay completion failed: disk full."],
    ["mismatch", "Caption overlay completion returned a different overlayId."],
    ["null PNG", "Caption overlay PNG encoding failed."],
    ["wrong MIME", "Caption overlay PNG encoding failed."],
    ["canvas", "Your browser cannot create a transparent caption overlay canvas."],
  ])("releases the canvas and deletes the overlay on %s failure, even if cleanup fails", async (stage, message) => {
    fetcher.mockResolvedValueOnce(json({ overlayId }));
    if (stage === "upload") fetcher.mockResolvedValueOnce(json({ detail: "disk full" }, 507));
    if (stage === "completion" || stage === "mismatch") {
      fetcher.mockResolvedValueOnce(json({})).mockResolvedValueOnce(
        stage === "completion" ? json({ detail: "disk full" }, 507) : json({ overlayId: "wrong" }),
      );
    }
    if (stage === "null PNG" || stage === "wrong MIME") {
      vi.mocked(canvas.toBlob).mockImplementation((callback) => callback(
        stage === "null PNG" ? null : new Blob(["not png"], { type: "image/jpeg" }),
      ));
    }
    if (stage === "canvas") vi.mocked(canvas.getContext).mockReturnValue(null);
    fetcher.mockRejectedValueOnce(new Error("cleanup offline"));
    await expect(uploadRenderOverlays({ ...job, end: job.start + 1 / fps }, fps, progress)).rejects.toThrow(message);
    expect(fetcher).toHaveBeenLastCalledWith(overlayUrl, { method: "DELETE", signal: expect.anything() });
    expect(timeout).toHaveBeenLastCalledWith(5000);
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);
    expect(progress).not.toHaveBeenCalledWith(1, expect.anything());
  });

  it.each(["old worker", "missing ID"])("does not allocate canvas or attempt unknown-ID cleanup for %s", async (failure) => {
    fetcher.mockResolvedValueOnce(failure === "old worker" ? json({}, 404) : json({ overlayId: " " }));
    await expect(uploadRenderOverlays(job, fps, progress)).rejects.toThrow(
      failure === "old worker" ? "Restart/update the local worker" : "returned no overlayId",
    );
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(document.createElement).not.toHaveBeenCalled();
  });

  it("does no work when already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(uploadRenderOverlays(job, fps, progress, controller.signal)).rejects.toThrow("cancelled");
    expect(prepareOverlayFonts).not.toHaveBeenCalled();
    expect(fetcher).not.toHaveBeenCalled();
    expect(document.createElement).not.toHaveBeenCalled();
  });

  it("stops waiting for fonts on abort and removes cancellation listeners", async () => {
    const controller = new AbortController();
    const entered = deferred<void>();
    const fonts = deferred<void>();
    const remove = vi.spyOn(controller.signal, "removeEventListener");
    vi.mocked(prepareOverlayFonts).mockImplementation(() => { entered.resolve(); return fonts.promise; });
    const pending = uploadRenderOverlays(job, fps, progress, controller.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    await entered.promise;
    controller.abort();
    await rejected;
    expect(remove).toHaveBeenCalledWith("abort", expect.any(Function));
    fonts.resolve();
    await Promise.resolve();
    expect(fetcher).not.toHaveBeenCalled();
    expect(document.createElement).not.toHaveBeenCalled();
  });

  it("waits for a creation ID after abort using its own 30s timeout, then cleans it up", async () => {
    const controller = new AbortController();
    const entered = deferred<void>();
    const created = deferred<Response>();
    fetcher.mockImplementationOnce(() => { entered.resolve(); return created.promise; })
      .mockResolvedValueOnce(json({}));
    const pending = uploadRenderOverlays(job, fps, progress, controller.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    await entered.promise;
    controller.abort();
    expect(timeout).toHaveBeenCalledWith(30000);
    expect(fetcher.mock.calls[0]![1]!.signal).not.toBe(controller.signal);
    expect(fetcher.mock.calls[0]![1]!.signal!.aborted).toBe(false);
    expect(fetcher).toHaveBeenCalledTimes(1);
    created.resolve(json({ overlayId }));
    await rejected;
    expect(fetcher).toHaveBeenLastCalledWith(overlayUrl, { method: "DELETE", signal: expect.anything() });
    expect(fetcher.mock.calls[1]![1]!.signal!.aborted).toBe(false);
    expect(timeout).toHaveBeenLastCalledWith(5000);
    expect(document.createElement).not.toHaveBeenCalled();
  });

  it("abandons an unabortable PNG callback and cleans up without sending that frame", async () => {
    const controller = new AbortController();
    const encoding = deferred<BlobCallback>();
    vi.mocked(canvas.toBlob).mockImplementation((callback) => encoding.resolve(callback));
    fetcher.mockResolvedValueOnce(json({ overlayId })).mockResolvedValueOnce(json({}));
    const pending = uploadRenderOverlays(job, fps, progress, controller.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    const callback = await encoding.promise;
    controller.abort();
    await rejected;
    callback(new Blob(["late"], { type: "image/png" }));
    await Promise.resolve();
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([base, overlayUrl]);
    expect(fetcher.mock.calls[1]![1]!.signal!.aborted).toBe(false);
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);
  });

  it.each(["upload", "completion"])("cleans up and reports cancellation during %s", async (stage) => {
    const controller = new AbortController();
    fetcher.mockResolvedValueOnce(json({ overlayId }));
    if (stage === "completion") fetcher.mockResolvedValueOnce(json({}));
    fetcher.mockImplementationOnce(async () => {
      controller.abort();
      if (stage === "upload") throw new DOMException("aborted", "AbortError");
      return json({ overlayId });
    }).mockRejectedValueOnce(new Error("cleanup offline"));
    await expect(uploadRenderOverlays({ ...job, end: job.start + 1 / fps }, fps, progress, controller.signal))
      .rejects.toThrow("cancelled");
    expect(fetcher).toHaveBeenLastCalledWith(overlayUrl, { method: "DELETE", signal: expect.anything() });
    expect(fetcher.mock.lastCall![1]!.signal!.aborted).toBe(false);
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);
    expect(progress).not.toHaveBeenCalledWith(1, expect.anything());
  });
});
