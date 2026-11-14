// Zero-token automated client for the local FastAPI faster-whisper, OpenShorts Reframe v2 & 5-Key Gemini Pool worker (http://127.0.0.1:8765).
import type { Word } from "./media";
import type { AIClip } from "./ai.functions";
import type { SpeakerReport, TrackPoint } from "./reframe";
import { normalizeTranscriptWords } from "./subtitles";

export const WORKER_URL = "http://127.0.0.1:8765";
const LOCAL_GEMINI_KEYS_STORAGE = "clipforge_gemini_5key_pool_v1";

export type WhisperModelId = "tiny" | "base" | "small" | "medium" | "large-v3";

export type WhisperConfig = {
  model: WhisperModelId;
  device: "auto" | "cuda" | "cpu";
  computeType: "auto" | "int8_float16" | "int8" | "float16";
  language: string;
  beamSize: number;
  wordTimestamps: boolean;
  vadFilter: boolean;
  chunkSeconds: number;
  initialPrompt: string;
  trackingEngine?: "openshorts_reframe_v2" | "hybrid_yolo_bytetrack" | "mediapipe_fast" | undefined;
  gimbalSmoothing?: number | undefined;
  jumpConfirmFrames?: number | undefined;
  speakerSwitchCooldownFrames?: number | undefined;
  activeSpeakerGate?: boolean | undefined;
};

export const DEFAULT_WHISPER_CONFIG: WhisperConfig = {
  model: "small",
  device: "auto",
  computeType: "int8_float16",
  language: "auto",
  beamSize: 5,
  wordTimestamps: true,
  vadFilter: true,
  chunkSeconds: 360,
  initialPrompt: "",
  trackingEngine: "openshorts_reframe_v2",
  gimbalSmoothing: 1.55,
  jumpConfirmFrames: 3,
  speakerSwitchCooldownFrames: 30,
  activeSpeakerGate: true,
};

export type AcceleratorTier = "discrete" | "integrated" | "cpu" | "unavailable";

export type AcceleratorSummary = {
  tier: AcceleratorTier;
  vendor: string;
  name: string;
  video_encoder: string | null;
  video_encoder_label: string;
  transcription_device: string;
  transcription_compute: string;
  cpu_threads: number;
  quality_locked: boolean;
};

export type EncoderCapabilities = {
  tier: AcceleratorTier;
  encoder: {
    name: string;
    kind: string;
    vendor: string;
    integrated: boolean;
    label: string;
    note: string;
    args: string[];
  } | null;
  software_fallback: { name: string; label: string };
  delivery: {
    width: number;
    height: number;
    pixel_format: string;
    codec: string;
    profile: string;
    bitrate: string;
    max_bitrate: string;
  };
  quality_locked: boolean;
};

export type WorkerHealth = {
  ok: boolean;
  app: string;
  version?: string;
  authRequired: boolean;
  encoder: string | null;
  encoderKind?: string | null;
  encoderLabel?: string | null;
  accelerator?: AcceleratorSummary;
  whisperInstalled: boolean;
  cudaAvailable: boolean;
  gpuName: string | null;
  activeDevice: string | null;
  geminiKeysConfigured?: number;
  geminiMaxKeys?: number;
};

export type WhisperModelStatus = {
  id: WhisperModelId;
  label: string;
  download_size_mb: number;
  min_vram_mb: number;
  description: string;
  cached: boolean;
  vram_compatible: boolean;
  recommended: boolean;
};

export type GeminiWorkerPoolSlot = {
  slot: number;
  env_name: string;
  configured: boolean;
  masked: string;
  status: string;
  requests: number;
  successes: number;
  rate_limits: number;
  latency_ms?: number | null;
  last_error?: string | null;
};

export type GeminiWorkerPoolStatus = {
  max_keys: number;
  configured_count: number;
  healthy_count: number;
  model: string;
  parallel_workers: number;
  effective_rpm_multiplier: number;
  slots: GeminiWorkerPoolSlot[];
};

export type SystemInfoResponse = {
  hardware: {
    whisper_installed: boolean;
    cuda_available: boolean;
    gpu_name: string | null;
    vram_total_mb: number | null;
    vram_free_mb: number | null;
    recommended_model: WhisperModelId;
    recommended_device: string;
    recommended_compute_type: string;
    cpu_fallback_available: boolean;
    cpu_threads?: number;
    active_device: string | null;
    active_compute_type: string | null;
    active_model: string | null;
    diarization: { available: boolean; backend: string | null; reason: string };
  };
  ffmpeg: { ffmpeg: boolean; ffprobe: boolean };
  tracking?: {
    ultralytics_yolo: boolean;
    opencv: boolean;
    mediapipe: boolean;
    active_backend: string;
    supported_trackers: string[];
    openshorts_features?: string[];
  };
  gemini_pool?: GeminiWorkerPoolStatus;
  video_encoder: string | null;
  accelerator?: AcceleratorSummary;
  capabilities?: EncoderCapabilities;
  supported_languages: { code: string; name: string }[];
};

export type TranscriptionJobResult = {
  text: string;
  words: Word[];
  duration: number;
  language: string;
  model: string;
  device: string;
  compute_type: string;
  timing_source: "whisper";
  missing_timing_count: number;
  duplicates_removed: number;
  warnings: string[];
  cached: boolean;
  file_sha256: string;
};

export function getLocalGeminiKeyPool(): string[] {
  if (typeof window === "undefined") return ["", "", "", "", ""];
  try {
    const raw = window.localStorage.getItem(LOCAL_GEMINI_KEYS_STORAGE);
    if (!raw) return ["", "", "", "", ""];
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return [0, 1, 2, 3, 4].map((i) => (typeof parsed[i] === "string" ? parsed[i].trim() : ""));
    }
  } catch {
    // ignore storage errors
  }
  return ["", "", "", "", ""];
}

export function saveLocalGeminiKeyPool(keys: string[]): string[] {
  const normalized = [0, 1, 2, 3, 4].map((i) =>
    typeof keys[i] === "string" ? keys[i].trim() : "",
  );
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(LOCAL_GEMINI_KEYS_STORAGE, JSON.stringify(normalized));
    } catch {
      // ignore storage errors
    }
  }
  return normalized;
}

export async function fetchGeminiKeyPoolStatus(): Promise<GeminiWorkerPoolStatus | null> {
  try {
    const res = await fetch(`${WORKER_URL}/v1/gemini/keys`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return null;
    return (await res.json()) as GeminiWorkerPoolStatus;
  } catch {
    return null;
  }
}

export async function saveGeminiKeyPoolToWorker(
  keys: string[],
  model?: string,
  parallelWorkers?: number,
): Promise<GeminiWorkerPoolStatus | null> {
  saveLocalGeminiKeyPool(keys);
  try {
    const res = await fetch(`${WORKER_URL}/v1/gemini/keys`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys, model, parallel_workers: parallelWorkers }),
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    return (await res.json()) as GeminiWorkerPoolStatus;
  } catch {
    return null;
  }
}

export async function checkGeminiKeyPoolHealth(): Promise<GeminiWorkerPoolStatus | null> {
  try {
    const res = await fetch(`${WORKER_URL}/v1/gemini/keys/check`, {
      method: "POST",
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return null;
    return (await res.json()) as GeminiWorkerPoolStatus;
  } catch {
    return null;
  }
}

export async function fetchWorkerHealth(): Promise<WorkerHealth | null> {
  try {
    const res = await fetch(`${WORKER_URL}/health`, { signal: AbortSignal.timeout(1500) });
    if (!res.ok) return null;
    return (await res.json()) as WorkerHealth;
  } catch {
    return null;
  }
}

export async function fetchWorkerSystemInfo(): Promise<SystemInfoResponse> {
  const res = await fetch(`${WORKER_URL}/v1/system/info`, {
    signal: AbortSignal.timeout(3000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `Worker returned ${res.status}`);
  }
  return (await res.json()) as SystemInfoResponse;
}

export async function fetchWhisperModels(): Promise<WhisperModelStatus[]> {
  const res = await fetch(`${WORKER_URL}/v1/models`, {
    signal: AbortSignal.timeout(3000),
  });
  if (!res.ok) throw new Error("Failed to query Whisper models");
  const data = (await res.json()) as { models: WhisperModelStatus[] };
  return data.models;
}

export async function downloadAndCacheModel(
  modelId: WhisperModelId,
  device = "auto",
  computeType = "auto",
): Promise<{ model: string; device: string; compute_type: string }> {
  const res = await fetch(
    `${WORKER_URL}/v1/models/${modelId}/download?device=${encodeURIComponent(device)}&compute_type=${encodeURIComponent(computeType)}`,
    { method: "POST" },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to download/load Whisper model");
  }
  return await res.json();
}

export async function runTestSampleTranscription(
  sampleFile: File,
  config: WhisperConfig,
): Promise<{
  text: string;
  words: Word[];
  duration: number;
  language: string;
  device: string;
  compute_type: string;
  model: string;
}> {
  const fd = new FormData();
  fd.append("file", sampleFile);
  fd.append("model", config.model);
  fd.append("language", config.language);
  fd.append("device", config.device);
  fd.append("compute_type", config.computeType);
  const res = await fetch(`${WORKER_URL}/v1/transcription/test`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Test transcription failed");
  }
  const out = await res.json();
  return { ...out, words: normalizeTranscriptWords(out.words ?? []) };
}

export async function transcribeWithLocalWhisper(
  file: File,
  config: WhisperConfig,
  onProgress?: (progress: number, statusText: string, logs: string[]) => void,
  signal?: AbortSignal,
  forceRetranscribe = false,
): Promise<TranscriptionJobResult> {
  const sourceId = await workerSourceId(file);
  // Automatically enrich initial_prompt with the video filename keywords if user left it empty
  const autoPrompt =
    config.initialPrompt.trim() ||
    file.name
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ")
      .trim();
  const buildForm = (withFile: boolean) => {
    const fd = new FormData();
    if (withFile) fd.append("file", file);
    fd.append("model", config.model);
    fd.append("device", config.device);
    fd.append("compute_type", config.computeType);
    fd.append("language", config.language);
    fd.append("beam_size", String(config.beamSize));
    fd.append("word_timestamps", String(config.wordTimestamps));
    fd.append("vad_filter", String(config.vadFilter));
    fd.append("chunk_seconds", String(config.chunkSeconds));
    fd.append("initial_prompt", autoPrompt);
    fd.append("force_retranscribe", String(forceRetranscribe));
    fd.append("source_id", sourceId);
    return fd;
  };

  let createRes = await fetch(`${WORKER_URL}/v1/transcription/jobs`, {
    method: "POST",
    body: buildForm(false),
    signal: signal ?? null,
  });
  if (createRes.status === 404) {
    createRes = await fetch(`${WORKER_URL}/v1/transcription/jobs`, {
      method: "POST",
      body: buildForm(true),
      signal: signal ?? null,
    });
  }
  if (!createRes.ok) {
    const err = await createRes.json().catch(() => ({}));
    throw new Error(err.detail || err.error || "Failed to start local Whisper transcription job");
  }
  const { job_id: jobId } = (await createRes.json()) as { job_id: string };

  const cancelListener = () => {
    void fetch(`${WORKER_URL}/v1/transcription/jobs/${jobId}/cancel`, {
      method: "POST",
    }).catch(() => {});
  };
  signal?.addEventListener("abort", cancelListener, { once: true });

  try {
    for (;;) {
      if (signal?.aborted) throw new Error("Transcription cancelled");
      await new Promise((r) => setTimeout(r, 700));

      const [statusRes, logsRes] = await Promise.all([
        fetch(`${WORKER_URL}/v1/transcription/jobs/${jobId}`, { signal: signal ?? null }),
        fetch(`${WORKER_URL}/v1/transcription/jobs/${jobId}/logs`, {
          signal: signal ?? null,
        }).catch(() => null),
      ]);

      if (!statusRes.ok) throw new Error("Lost connection to local Whisper worker");
      const st = (await statusRes.json()) as {
        status: string;
        progress: number;
        error?: string | null;
      };
      const logsData =
        logsRes && logsRes.ok ? ((await logsRes.json()) as { logs: string[] }) : { logs: [] };

      onProgress?.(st.progress ?? 0, st.status, logsData.logs);

      if (st.status === "error" || st.status === "interrupted_restart") {
        throw new Error(st.error || "Local Whisper transcription job failed");
      }
      if (st.status === "cancelled") {
        throw new Error("Transcription cancelled");
      }
      if (st.status === "completed") {
        const resultRes = await fetch(`${WORKER_URL}/v1/transcription/jobs/${jobId}/result`, {
          signal: signal ?? null,
        });
        if (!resultRes.ok) throw new Error("Failed to fetch completed Whisper transcript");
        const data = (await resultRes.json()) as TranscriptionJobResult;
        return {
          ...data,
          words: normalizeTranscriptWords(data.words ?? []),
        };
      }
    }
  } finally {
    signal?.removeEventListener("abort", cancelListener);
  }
}

const OUTRO_SIGN_OFF_RE =
  /\b(if you made it this far|see y'all next week|see you next week|see you next time|all in a day's work|good night|today's panel|sub or you're|like and subscribe|thanks for watching)\b/i;
const WEAK_FILLER_LEAD_RE =
  /^(just kind of like|i don't know|let's start it off|uh\b|um\b|so yeah|year\.?\s+i don't know)/i;
const DANGLING_NEW_TURN_RE =
  /\b(hey,?\s*number\s+(one|two|three|four|five|six|\d+)|mechanic or ghost hunter)\b/i;

/**
 * Rules 4, 6, 7: Clean narrative boundaries of a candidate clip:
 * - Rule 4 (Outro / Credits / Blooper Hard Stop): Cut before episode sign-offs ("if you made it this far",
 *   "see y'all next week", "good night") and before long >= 4.5s credit-music silence gaps (WA0029 0:27-0:43.5).
 * - Rule 6 (Payoff-Boundary Trimming & Trailing Question Veto): If a new sentence starting in the final 4.0s
 *   asks an unanswered question ("?") or addresses a new contestant ("Hey, number five, mechanic or ghost hunter?"
 *   in WA0030 0:41.8-0:44.4) after a completed payoff, trim `end` back to the completed payoff.
 * - Rule 7 (Hook-First Lead-In Snapping): If the first <= 5.5s is mid-sentence filler ("just kind of like held
 *   the room down") before a strong sentence boundary, advance `start` to the clean sentence start.
 */
export function trimClipNarrativeBoundaries(
  startSec: number,
  endSec: number,
  words: Word[],
  videoDuration: number,
): { start: number; end: number } {
  const inside = words.filter((w) => w.e >= startSec - 0.25 && w.s <= endSec + 0.25);
  if (inside.length < 8) {
    return { start: startSec, end: endSec };
  }

  let cleanStart = startSec;
  let cleanEnd = endSec;

  // Rule 7: Hook-First Lead-In Snapping (skip mid-sentence / filler opener in first 5.5s)
  const leadWords = inside.filter((w) => w.s <= startSec + 5.5);
  const leadText = leadWords.map((w) => w.w).join(" ");
  const startsLowercase = /^[a-z]/.test(inside[0]!.w);
  if ((startsLowercase || WEAK_FILLER_LEAD_RE.test(leadText)) && cleanEnd - cleanStart >= 16) {
    for (let k = 1; k < inside.length; k++) {
      const prev = inside[k - 1]!;
      const curr = inside[k]!;
      if (curr.s - startSec > 5.8) break;
      if (/[.!?]$/.test(prev.w) || curr.s - prev.e >= 0.45) {
        if (cleanEnd - curr.s >= 10) {
          cleanStart = Math.max(0, curr.s - 0.08);
          break;
        }
      }
    }
  }

  // Rule 4A: Detect long credit/music silence gap (>= 4.5s) in the second half of the clip (e.g. WA0029 0:32-0:42)
  const clipMid = cleanStart + (cleanEnd - cleanStart) * 0.45;
  for (let k = 1; k < inside.length; k++) {
    const prev = inside[k - 1]!;
    const curr = inside[k]!;
    if (prev.e >= clipMid && curr.s - prev.e >= 4.5 && prev.e - cleanStart >= 10) {
      cleanEnd = Math.min(cleanEnd, prev.e + 0.25);
      break;
    }
  }

  // Rule 4B: Detect episode sign-off / outro phrases ("if you made it this far", "see y'all next week", "good night")
  for (let k = 0; k < inside.length; k++) {
    const w = inside[k]!;
    if (w.s < cleanStart + 10 || w.s > cleanEnd) continue;
    const windowText = inside
      .slice(k, k + 7)
      .map((x) => x.w)
      .join(" ");
    if (OUTRO_SIGN_OFF_RE.test(windowText)) {
      const prevEnd = k > 0 ? inside[k - 1]!.e : w.s;
      if (prevEnd - cleanStart >= 10) {
        cleanEnd = Math.min(cleanEnd, prevEnd + 0.25);
        break;
      }
    }
  }

  // Rule 6: Trailing Unanswered Question & New Contestant Turn Veto in the final 4.0s (e.g. WA0030 0:41.8-0:44.4)
  const scoped = inside.filter((w) => w.s >= cleanStart && w.e <= cleanEnd + 0.15);
  if (scoped.length >= 6) {
    // Find the last sentence boundary before the finalword
    let lastSentenceEndIdx = -1;
    for (let k = scoped.length - 2; k >= Math.max(0, scoped.length - 12); k--) {
      if (/[.!?]$/.test(scoped[k]!.w) && cleanEnd - scoped[k]!.e <= 4.2) {
        lastSentenceEndIdx = k;
        break;
      }
    }
    if (lastSentenceEndIdx >= 0) {
      const tailSlice = scoped.slice(lastSentenceEndIdx + 1);
      const tailText = tailSlice.map((w) => w.w).join(" ");
      const endsWithQuestion = /\?$/.test(tailText.trim());
      const endsUnfinished = !/[.!?]$/.test(tailText.trim());
      const addressesNewTurn = DANGLING_NEW_TURN_RE.test(tailText);
      const payoffEnd = scoped[lastSentenceEndIdx]!.e;
      if (
        (endsWithQuestion || endsUnfinished || addressesNewTurn) &&
        payoffEnd - cleanStart >= 10
      ) {
        cleanEnd = Math.min(videoDuration, payoffEnd + 0.2);
      }
    }
  }

  return {
    start: Math.max(0, cleanStart),
    end: Math.min(videoDuration, Math.max(cleanStart + 3, cleanEnd)),
  };
}

export function snapClipToTranscriptWords(
  startSec: number,
  endSec: number,
  words: Word[],
  videoDuration: number,
): { start: number; end: number } {
  if (!words.length) {
    return {
      start: Math.max(0, startSec),
      end: Math.min(videoDuration, Math.max(startSec + 3, endSec)),
    };
  }
  const trimmed = trimClipNarrativeBoundaries(startSec, endSec, words, videoDuration);
  const nearStart = words.find((w) => w.e >= trimmed.start - 0.35) ?? words[0]!;
  let nearEnd = words[words.length - 1]!;
  for (let i = words.length - 1; i >= 0; i--) {
    if (words[i]!.s <= trimmed.end + 0.35) {
      nearEnd = words[i]!;
      break;
    }
  }
  const snappedStart = Math.max(0, nearStart.s);
  const snappedEnd = Math.min(videoDuration, Math.max(snappedStart + 3, nearEnd.e));
  return { start: snappedStart, end: snappedEnd };
}

/**
 * Local Smart Viral Clip Discovery Engine (OpenShorts editing_skills.py + Rules 4, 6, 7):
 * Evaluates hook patterns (CHALLENGE_PREMISE, STAKES, IMPORTANT_NUMBER, SURPRISE, STORY_PAYOFF),
 * skips weak filler starts, avoids outro/credit bleed, and advances with zero clip overlap.
 */
export function discoverClipsLocallyFromTranscript(
  words: Word[],
  duration: number,
  minLen: number,
  maxLen: number,
  count: number,
  timeframe?: [number, number],
): AIClip[] {
  const normalized = normalizeTranscriptWords(words);
  const tfStart = timeframe ? timeframe[0] : 0;
  const tfEnd = timeframe ? timeframe[1] : duration;
  const scoped = normalized.filter((w) => w.s >= tfStart && w.e <= tfEnd);
  if (!scoped.length) return [];

  const targetLen = Math.min(maxLen, Math.max(minLen, (minLen + maxLen) / 2));
  const candidates: AIClip[] = [];
  const hookRegex =
    /(\$|\d+|million|thousand|win|challenge|contestant|prize|risk|secret|never|how|why|what|best|crazy|insane|first|ever|imagine|wait|watch|ghost|hunter)/i;

  let i = 0;
  while (i < scoped.length) {
    // Rule 7: If current word is mid-sentence filler, advance to the next clean sentence start within 4s
    while (
      i + 1 < scoped.length &&
      /^(uh|um|yeah|so|like|just)$/i.test(scoped[i]!.w.replace(/[^a-z]/gi, ""))
    ) {
      i++;
    }
    const startWord = scoped[i]!;
    const rawClipStart = startWord.s;
    if (rawClipStart + minLen > tfEnd + 1) break;

    let bestEndIdx = i;
    for (let j = i; j < scoped.length; j++) {
      const w = scoped[j]!;
      const span = w.e - rawClipStart;
      if (span >= minLen) {
        bestEndIdx = j;
        if (/[.!?]$/.test(w.w) && span >= targetLen * 0.8) break;
      }
      if (span >= maxLen) break;
    }

    const rawClipEnd = scoped[bestEndIdx]?.e ?? Math.min(tfEnd, rawClipStart + targetLen);
    const cleanedBounds = trimClipNarrativeBoundaries(rawClipStart, rawClipEnd, scoped, tfEnd);
    const slice = scoped.filter(
      (w) => w.s >= cleanedBounds.start - 0.05 && w.e <= cleanedBounds.end + 0.05,
    );
    const clipStart = slice[0]?.s ?? cleanedBounds.start;
    const clipEnd = slice.at(-1)?.e ?? cleanedBounds.end;

    if (clipEnd - clipStart >= Math.min(5, minLen) && slice.length > 0) {
      // Pick the strongest hook clause in the first 12 words (avoiding weak filler like "I don't know")
      let hookSlice = slice.slice(0, 8);
      for (let k = 0; k < Math.min(6, slice.length - 4); k++) {
        const sub = slice.slice(k, k + 8);
        const subText = sub.map((w) => w.w).join(" ");
        if (!WEAK_FILLER_LEAD_RE.test(subText) && hookRegex.test(subText)) {
          hookSlice = sub;
          break;
        }
      }
      const openingWords = hookSlice.map((w) => w.w).join(" ");
      let score = 76;
      const editingCues: AIClip["editingCues"] = [];
      for (const w of slice) {
        if (hookRegex.test(w.w)) score += 2.2;
        if (/[!?]$/.test(w.w)) score += 1.8;
        if (/(\$|\d+|million|thousand)/i.test(w.w) && editingCues.length < 4) {
          editingCues.push({
            kind: "number",
            start: w.s,
            end: Math.max(w.s + 0.4, w.e),
            quote: w.w,
            confidence: 0.9,
          });
        }
      }
      if (hookRegex.test(openingWords)) score += 7;
      score = Math.min(98, Math.round(score));

      const cleanHook =
        openingWords
          .replace(/[^\p{L}\p{N}\s$!?'-]/gu, "")
          .trim()
          .slice(0, 56) || "Must-Watch Moment";
      candidates.push({
        title: cleanHook,
        hook: cleanHook,
        start: clipStart,
        end: clipEnd,
        score,
        reason: `OpenShorts 2-Second Hook ("${hookSlice
          .slice(0, 6)
          .map((w) => w.w)
          .join(
            " ",
          )}…") with ${slice.length} synchronized Whisper words and ${editingCues.length} emphasis cues.`,
        category: "viral_hook",
        editingKind: "challenge",
        layoutIntent: "uncertain",
        suggestedLayout: "auto",
        suggestedBroll: "",
        hashtags: ["#shorts", "#viral", "#highlights"],
        storyScores: {
          hook_strength: Math.min(99, score + 2),
          standalone_context: 88,
          narrative_completeness: 86,
          pacing: 91,
          ending_quality: 85,
        },
        editingCues,
      });
    }

    // Advance past clipEnd so generated clips never overlap each other
    const nextTime = Math.max(rawClipStart + Math.max(8, minLen * 0.8), clipEnd + 0.4);
    const nextIdx = scoped.findIndex((w, idx) => idx > i && w.s >= nextTime);
    if (nextIdx === -1) break;
    i = nextIdx;
  }

  return candidates.sort((a, b) => b.score - a.score).slice(0, count);
}

export async function workerSourceId(file: File): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(`${file.name}:${file.size}:${file.lastModified}`),
  );
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 32);
}

export async function analyzeReframeWithLocalWorker(
  file: File,
  start: number,
  end: number,
  framingMode: "auto" | "fill" | "split" | "fit" | "grid3" | "editorial",
  signal?: AbortSignal,
  options?: { sourceId?: string; onProgress?: ((p: number) => void) | undefined },
): Promise<{ track: TrackPoint[]; speakers?: SpeakerReport }> {
  const sourceId = options?.sourceId ?? (await workerSourceId(file));
  const post = async (withFile: boolean) => {
    const fd = new FormData();
    if (withFile) fd.append("file", file);
    fd.append("start", String(start));
    fd.append("end", String(end));
    fd.append("framing_mode", framingMode);
    fd.append("source_id", sourceId);
    fd.append("queue", "true");
    if (withFile) fd.append("total_bytes", String(file.size));
    return fetch(`${WORKER_URL}/v1/openshorts/analyze`, {
      method: "POST",
      body: fd,
      signal: signal ?? null,
    });
  };
  let response = await post(false);
  if (response.status === 404) response = await post(true);
  if (!response.ok) throw new Error(`Worker reframe analysis failed (${response.status})`);
  const queued = (await response.json()) as { job_id: string };
  const jobUrl = `${WORKER_URL}/v1/openshorts/analyze/jobs/${queued.job_id}`;
  const cancel = () => {
    void fetch(`${jobUrl}/cancel`, { method: "POST" }).catch(() => {});
  };
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    if (signal?.aborted) {
      cancel();
      signal.throwIfAborted();
    }
    for (;;) {
      signal?.throwIfAborted();
      const statusResponse = await fetch(jobUrl, { signal: signal ?? null });
      if (!statusResponse.ok) throw new Error("Lost contact with the analysis worker");
      const state = (await statusResponse.json()) as {
        status: string;
        progress: number;
        error?: string;
        result?: { ok: boolean; track: TrackPoint[]; speakers?: SpeakerReport };
      };
      if (state.status === "error") throw new Error(state.error || "Video analysis failed");
      if (state.status === "cancelled")
        throw new DOMException("Video analysis cancelled", "AbortError");
      options?.onProgress?.(Math.min(0.99, state.progress));
      if (state.status === "completed") {
        if (!state.result?.ok || !state.result.track?.length) {
          throw new Error("Worker reframe analysis returned no track points");
        }
        options?.onProgress?.(1);
        return state.result.speakers
          ? { track: state.result.track, speakers: state.result.speakers }
          : { track: state.result.track };
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}
