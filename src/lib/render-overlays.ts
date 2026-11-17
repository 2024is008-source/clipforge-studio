import { HELPER } from "@/components/YouTubeImport";
import type { ClipJob } from "./render";

const BATCH_SIZE = 8;

export function overlayTiming(job: ClipJob, fps: number) {
  const duration = job.end - job.start;
  if (!Number.isFinite(job.start) || job.start < 0 || !Number.isFinite(duration) || duration <= 0 || duration > 300) {
    throw new Error("Caption overlays require a clip between 0 and 300 seconds.");
  }
  if (!Number.isFinite(fps) || fps < 6 || fps > 60) {
    throw new Error("The local renderer must resolve a frame rate between 6 and 60 fps.");
  }
  const frameCount = Math.ceil(duration * fps - 1e-8);
  if (!Number.isSafeInteger(frameCount) || frameCount < 1) {
    throw new Error("Caption overlay frame count is invalid.");
  }
  return { fps, frameCount, width: 1080 as const, height: 1920 as const };
}

/** Keep aborts distinguishable from unavailable workers at every transport boundary. */
export function checkRenderCancellation(signal?: AbortSignal, error?: unknown): void {
  if (signal?.aborted || (error instanceof Error && (error.name === "AbortError" || error.message === "cancelled"))) {
    throw new Error("cancelled");
  }
}

/** Font loading and canvas.toBlob cannot themselves be aborted; stop waiting without leaking listeners. */
export function withRenderCancellation<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending;
  return new Promise<T>((resolve, reject) => {
    const abort = () => {
      signal.removeEventListener("abort", abort);
      reject(new Error("cancelled"));
    };
    signal.addEventListener("abort", abort, { once: true });
    pending.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        if (signal.aborted) reject(new Error("cancelled"));
        else resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", abort);
        reject(signal.aborted ? new Error("cancelled") : error);
      },
    );
    if (signal.aborted) abort();
  });
}

async function requireOverlayResponse(response: Response, action: string): Promise<void> {
  if (response.ok) return;
  const data = await response.json().catch(() => null) as { detail?: unknown } | null;
  const detail = typeof data?.detail === "string" ? data.detail : `HTTP ${response.status}`;
  const hint = response.status === 404 || response.status === 405
    ? " Restart/update the local worker to enable PNG overlay export."
    : "";
  throw new Error(`Caption overlay ${action} failed: ${detail}.${hint}`);
}

/** Best-effort cleanup uses its own timeout: the user's signal is usually already aborted. */
export async function deleteRenderOverlays(overlayId: string): Promise<void> {
  await fetch(`${HELPER}/v1/render/overlays/${encodeURIComponent(overlayId)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(5000),
  }).catch(() => {});
}

function pngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob || blob.type !== "image/png") reject(new Error("Caption overlay PNG encoding failed."));
      else resolve(blob);
    }, "image/png");
  });
}

/**
 * Rasterize only the shared visual layer at exact output-frame source times. No video element,
 * decoder, model inference, playback clock, or MediaRecorder participates in this path.
 * Eight PNGs at most are retained, and each upload completes before the next batch starts.
 */
export async function uploadRenderOverlays(
  job: ClipJob,
  fps: number,
  onProgress: (p: number, stageLabel?: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  checkRenderCancellation(signal);
  const timing = overlayTiming(job, fps);
  let overlayId: string | undefined;
  let canvas: HTMLCanvasElement | undefined;
  try {
    onProgress(0, "Preparing caption overlays: loading fonts…");
    // render -> backend-render -> this module: keep the renderer dependency asynchronous.
    const { drawFrameOverlays, prepareOverlayFonts } = await withRenderCancellation(import("./render"), signal);
    checkRenderCancellation(signal);
    await withRenderCancellation(prepareOverlayFonts(job), signal);
    checkRenderCancellation(signal);

    const created = await fetch(`${HELPER}/v1/render/overlays`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(timing),
      signal: AbortSignal.timeout(30000),
    });
    await requireOverlayResponse(created, "creation");
    const data = await created.json() as { overlayId?: unknown };
    if (typeof data.overlayId !== "string" || !data.overlayId.trim()) {
      throw new Error("Caption overlay creation returned no overlayId.");
    }
    overlayId = data.overlayId;
    checkRenderCancellation(signal);
    const url = `${HELPER}/v1/render/overlays/${encodeURIComponent(overlayId)}`;
    canvas = document.createElement("canvas");
    canvas.width = timing.width;
    canvas.height = timing.height;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) throw new Error("Your browser cannot create a transparent caption overlay canvas.");

    for (let start = 0; start < timing.frameCount; start += BATCH_SIZE) {
      const body = new FormData();
      body.append("start", String(start));
      const end = Math.min(start + BATCH_SIZE, timing.frameCount);
      for (let n = start; n < end; n++) {
        checkRenderCancellation(signal);
        ctx.clearRect(0, 0, timing.width, timing.height);
        drawFrameOverlays(ctx, job.start + n / timing.fps, job);
        const blob = await withRenderCancellation(pngBlob(canvas), signal);
        checkRenderCancellation(signal);
        body.append("files", blob, `frame_${String(n).padStart(8, "0")}.png`);
        onProgress(Math.min(0.99, (n + 1) / timing.frameCount), `Preparing caption overlays: ${n + 1}/${timing.frameCount}…`);
      }
      checkRenderCancellation(signal);
      const uploaded = await fetch(`${url}/frames`, { method: "POST", body, signal: signal ?? null });
      await requireOverlayResponse(uploaded, "upload");
    }
    checkRenderCancellation(signal);
    const completed = await fetch(`${url}/complete`, { method: "POST", signal: signal ?? null });
    await requireOverlayResponse(completed, "completion");
    const result = await completed.json() as { overlayId?: unknown };
    if (result.overlayId !== overlayId) throw new Error("Caption overlay completion returned a different overlayId.");
    checkRenderCancellation(signal);
    onProgress(1, "Caption overlays ready");
    return overlayId;
  } catch (error) {
    if (overlayId) await deleteRenderOverlays(overlayId);
    checkRenderCancellation(signal, error);
    throw error;
  } finally {
    // Release the full-size pixel buffer immediately, including on failed uploads.
    if (canvas) { canvas.width = 0; canvas.height = 0; }
  }
}
