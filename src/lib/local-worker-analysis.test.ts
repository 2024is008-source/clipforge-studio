import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeReframeWithLocalWorker } from "./local-worker";

const file = new File(["video"], "synthetic.mp4", { type: "video/mp4" });
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });

afterEach(() => vi.unstubAllGlobals());

describe("queued local analysis", () => {
  it("retains safe fit and provenance without requiring a successful detection", async () => {
    const track = [{ t: 0, x: .5, y: .3, mode: "fit", trackingConfidence: 0,
      layoutDecision: { scene: 0, reason: "low_tracking_confidence" } }];
    const fetcher = vi.fn().mockResolvedValueOnce(json({ job_id: "test" }))
      .mockResolvedValueOnce(json({ status: "completed", progress: 1, result: { ok: true, track } }));
    vi.stubGlobal("fetch", fetcher);
    const result = await analyzeReframeWithLocalWorker(file, 0, 2, "fill", undefined, { sourceId: "cached" });
    expect(result.track).toEqual(track);
    const body = fetcher.mock.calls[0]![1].body as FormData;
    expect(body.get("queue")).toBe("true");
    expect(body.get("framing_mode")).toBe("fill");
    expect(fetcher.mock.calls[1]![0]).toMatch(/analyze\/jobs\/test$/);
  });

  it("reports a worker error rather than manufacturing a successful track", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ job_id: "test" }))
      .mockResolvedValueOnce(json({ status: "error", progress: .5, error: "No frames decoded" })));
    await expect(analyzeReframeWithLocalWorker(file, 0, 2, "auto", undefined, { sourceId: "cached" }))
      .rejects.toThrow("No frames decoded");
  });

  it("sends cancellation even when abort races the queue response", async () => {
    const controller = new AbortController();
    const fetcher = vi.fn().mockImplementationOnce(async () => {
      controller.abort();
      return json({ job_id: "test" });
    }).mockResolvedValue(json({ status: "cancelled" }));
    vi.stubGlobal("fetch", fetcher);
    await expect(analyzeReframeWithLocalWorker(file, 0, 2, "auto", controller.signal, { sourceId: "cached" }))
      .rejects.toMatchObject({ name: "AbortError" });
    expect(fetcher.mock.calls[1]![0]).toMatch(/analyze\/jobs\/test\/cancel$/);
  });
});
