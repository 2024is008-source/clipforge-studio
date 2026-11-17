// Backend export path: deterministic canvas PNGs preserve the visual layer; the local worker's
// FFmpeg pipeline decodes, composes and encodes video. The browser never captures video here.
import type { TrackPoint } from "./reframe";
import { punchGainOf } from "./edit-tuning";
import { HELPER } from "@/components/YouTubeImport";
import type { ClipJob } from "./render";
import { cleanHookTitle } from "./subtitles";
import { workerSourceId } from "./local-worker";
import { checkRenderCancellation, deleteRenderOverlays, uploadRenderOverlays, withRenderCancellation } from "./render-overlays";

export type BackendRenderResult = {
  blob: Blob;
  encoder: string;
  /** Encoder class as measured by the worker: "gpu" when an NVENC/AMF pass actually ran. */
  encoderKind: string;
  verified: boolean;
  deliveredFps: number;
  duration: number;
  elapsedSec: number;
  qc: unknown;
};

/**
 * The wire form of `RenderDecision` (local_worker/rendering/decision.py). Everything here is the
 * same value the live preview drew with — track points in absolute source seconds, word windows in
 * absolute source seconds — so the two renderers are looking at one edit, not two.
 */
export function decisionFromJob(job: ClipJob) {
  const s = job.settings;
  const track = (job.track ?? []).map((p: TrackPoint) => ({
    t: p.t,
    x: p.x,
    y: p.y,
    mode: p.mode ?? "fill",
    x2: p.x2 ?? null,
    y2: p.y2 ?? null,
    x3: p.x3 ?? null,
    y3: p.y3 ?? null,
    zoom: p.zoom ?? 1,
    cut: !!p.cut,
    hasBottomGraphic: !!p.hasBottomGraphic,
    subjectType: p.subjectType ?? "face",
    layoutDecision: p.layoutDecision,
    trackingConfidence: p.trackingConfidence,
  }));
  return {
    start: job.start,
    end: job.end,
    layout: s.layout,
    hook: cleanHookTitle(job.hook, job.words),
    track,
    words: job.words.map((w) => ({ w: w.w, start: w.s, end: w.e })),
    style: {
      preset: s.captionStyle,
      highlight: s.highlight,
      secondaryHighlight: s.secondaryHighlight ?? "#ffd400",
      positionY: s.captionY,
      wordsPerPage: s.wordsPerPage,
      fontSize: s.fontSize ?? null,
      showEmojis: s.showEmojis ?? true,
      showHook: s.showHook,
      font: "Montserrat",
    },
    zoomPunch: s.zoomPunch ?? true,
    visualEffects: s.effects ?? true,
    manualX: s.manualX ?? null,
    manualY: s.manualY ?? null,
    voiceEnhance: s.voiceEnhance ?? true,
    fps: s.fps ?? "30",
    brandLabel: s.brandLabel || undefined,
  };
}

/** Match the preview's own damping of beat punches, which only kicks in on a real track. */
function punchGainFor(job: ClipJob): number {
  const t = job.track;
  return t && t.length >= 3 ? punchGainOf(t) : 1;
}

async function errorDetail(res: Response): Promise<string> {
  const data = await res.json().catch(() => null);
  const detail = (data as { detail?: string } | null)?.detail;
  return detail || `local worker returned HTTP ${res.status}`;
}

/** The worker cannot take this native job. This is reported, never an automatic capture fallback. */
export class BackendUnavailable extends Error {}

/** Explicit development escape hatch so the two paths can still be A/B'd on the same clip. */
export function browserCaptureForced(): boolean {
  try {
    return typeof localStorage !== "undefined" && localStorage.getItem("clipforge_browser_capture") === "1";
  } catch {
    return false;
  }
}

async function resolveOrCacheSourceOnBackend(
  sourceId: string | undefined,
  options?: { file?: File | Blob | null | undefined; src?: string | undefined },
  signal?: AbortSignal,
): Promise<{ sourceId: string; blob: File | Blob | null }> {
  checkRenderCancellation(signal);
  let blob: File | Blob | null = options?.file ?? null;
  if (!blob && options?.src) {
    try {
      const res = await fetch(options.src, { signal: signal ?? null });
      if (res.ok) blob = await res.blob();
    } catch (error) {
      checkRenderCancellation(signal, error);
      blob = null;
    }
  }
  checkRenderCancellation(signal);
  if (sourceId) return { sourceId, blob };
  if (blob instanceof File) {
    return { sourceId: await withRenderCancellation(workerSourceId(blob), signal), blob };
  }
  if (blob) {
    const f = new File([blob], "source_video.mp4", { type: blob.type || "video/mp4", lastModified: 1 });
    return { sourceId: await withRenderCancellation(workerSourceId(f), signal), blob: f };
  }
  throw new BackendUnavailable("No source video identifier or blob available for backend rendering.");
}

function pollDelay(ms: number, signal?: AbortSignal): Promise<void> {
  let timer: ReturnType<typeof setTimeout>;
  const pending = new Promise<void>((resolve) => { timer = setTimeout(resolve, ms); });
  return withRenderCancellation(pending, signal).finally(() => clearTimeout(timer));
}

/** Prepare PNGs, queue, watch weighted FFmpeg progress, cancel-with-the-user, and download. */
export async function renderWithLocalBackend(
  job: ClipJob,
  sourceId: string | undefined,
  onProgress: (p: number, stageLabel?: string) => void,
  signal?: AbortSignal,
  options?: { file?: File | Blob | null | undefined; src?: string | undefined },
): Promise<BackendRenderResult> {
  const sig = signal ?? null;
  const overlayWeight = 0.35;
  let overlayId: string | undefined;
  let jobId: string | undefined;
  let cancellation: Promise<unknown> | undefined;
  const cancel = () => {
    if (jobId && !cancellation) {
      cancellation = fetch(`${HELPER}/v1/render/jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
        signal: AbortSignal.timeout(5000),
      }).catch(() => {});
    }
  };

  try {
    checkRenderCancellation(signal);
    onProgress(0.01, "Preparing source for native FFmpeg export…");
    const resolved = await resolveOrCacheSourceOnBackend(sourceId, options, signal);
    const sid = resolved.sourceId;
    const sourceUrl = `${HELPER}/v1/sources/${encodeURIComponent(sid)}`;
    const infoUrl = `${sourceUrl}/render-info?fps=${encodeURIComponent(job.settings.fps ?? "30")}`;
    const readInfo = () => fetch(infoUrl, { signal: sig }).catch((error: unknown) => {
      checkRenderCancellation(signal, error);
      throw new BackendUnavailable("The local renderer is offline. Start the local worker and export again.");
    });
    let infoResponse = await readInfo();
    if (infoResponse.status === 409 && resolved.blob) {
      onProgress(0.01, "Syncing source video to local FFmpeg renderer…");
      const body = new FormData();
      body.append("file", resolved.blob, resolved.blob instanceof File ? resolved.blob.name : `${sid}.mp4`);
      const uploaded = await fetch(sourceUrl, { method: "POST", body, signal: sig });
      if (!uploaded.ok) throw new Error(`Source sync failed: ${await errorDetail(uploaded)}`);
      infoResponse = await readInfo();
    }
    if (!infoResponse.ok) throw new BackendUnavailable(`Cannot resolve native export timing: ${await errorDetail(infoResponse)}`);
    const info = await infoResponse.json() as { fps?: unknown };
    if (typeof info.fps !== "number" || !Number.isFinite(info.fps) || info.fps < 6 || info.fps > 60) {
      throw new Error("The local renderer returned an invalid output frame rate.");
    }
    const outputFps = info.fps;
    overlayId = await uploadRenderOverlays(job, outputFps, (p, label) => onProgress(0.02 + (overlayWeight - 0.02) * p, label), signal);
    checkRenderCancellation(signal);

    const postDecision = () =>
      fetch(`${HELPER}/v1/render/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceId: sid,
          overlayId,
          decision: decisionFromJob(job),
          punchGain: punchGainFor(job),
        }),
        signal: AbortSignal.timeout(30000),
      }).catch((error: unknown) => {
        checkRenderCancellation(signal, error);
        throw new BackendUnavailable(
          `No local renderer answered on 127.0.0.1:8765 (${(error as Error).message})`,
        );
      });

    onProgress(overlayWeight, "Handing the edit and overlays to the local FFmpeg renderer…");
    const createRes = await postDecision();
    if (!createRes.ok) {
      checkRenderCancellation(signal);
      if (createRes.status === 404) {
        throw new BackendUnavailable(
          "The local worker answering on 127.0.0.1:8765 is an older build without the native renderer. Restart it (local_worker) and export again.",
        );
      }
      if (createRes.status === 409) {
        throw new BackendUnavailable(
          "The local worker no longer holds a copy of this video. Re-open the file so it can cache it, then export again.",
        );
      }
      const detail = await errorDetail(createRes);
      if (createRes.status === 422 || createRes.status === 503 || createRes.status === 507) {
        throw new BackendUnavailable(detail);
      }
      throw new Error(`Backend render could not start: ${detail}`);
    }
    const queued = (await createRes.json()) as { job_id: string; encoder?: string };
    if (typeof queued.job_id !== "string" || !queued.job_id) throw new Error("Native renderer returned no job_id.");
    jobId = queued.job_id;
    signal?.addEventListener("abort", cancel, { once: true });
    checkRenderCancellation(signal);
    const jobUrl = `${HELPER}/v1/render/jobs/${encodeURIComponent(jobId)}`;

    let status = "";
    let backendProgress = 0;
    for (;;) {
      checkRenderCancellation(signal);
      await pollDelay(status === "queued" ? 400 : 800, signal);
      const res = await fetch(jobUrl, { signal: sig }).catch((error: unknown) => {
        checkRenderCancellation(signal, error);
        throw new Error("Lost contact with the local renderer.");
      });
      if (!res.ok) throw new Error("Lost contact with the local renderer.");
      const st = (await res.json()) as {
        status: string;
        progress: number;
        error?: string | null;
        result?: { fps?: number; duration?: number; elapsed_sec?: number; encoder_kind?: string; qc?: unknown } | null;
      };
      checkRenderCancellation(signal);
      status = st.status;
      if (status === "error") throw new Error(st.error || "Backend render failed.");
      if (status === "cancelled") throw new Error("cancelled");
      backendProgress = Math.max(backendProgress, Math.max(0, Math.min(1, st.progress || 0)));
      onProgress(
        overlayWeight + (0.99 - overlayWeight) * backendProgress,
        status === "queued" ? "Queued on the local renderer…" : "FFmpeg rendering the clip…",
      );
      if (status === "completed") {
        const file = await fetch(`${jobUrl}/download`, { signal: sig });
        if (!file.ok) throw new Error(`Rendered file could not be fetched: ${await errorDetail(file)}`);
        // Chrome aborts response.blob() on bodies large enough to need blob storage.
        const blob = new Blob([await file.arrayBuffer()], { type: "video/mp4" });
        checkRenderCancellation(signal);
        onProgress(1, "Render verified");
        return {
          blob,
          encoder: file.headers.get("X-Video-Encoder") || (queued.encoder ?? "ffmpeg"),
          encoderKind: st.result?.encoder_kind ?? "cpu",
          verified: file.headers.get("X-FFprobe-Verified") === "true",
          deliveredFps: st.result?.fps ?? outputFps,
          duration: st.result?.duration ?? job.end - job.start,
          elapsedSec: st.result?.elapsed_sec ?? 0,
          qc: st.result?.qc ?? null,
        };
      }
    }
  } catch (error) {
    // Stop a possibly still-running native job before releasing its PNG input on failure/cancel.
    cancel();
    await cancellation;
    checkRenderCancellation(signal, error);
    throw error;
  } finally {
    signal?.removeEventListener("abort", cancel);
    if (overlayId) await deleteRenderOverlays(overlayId);
  }
}
