import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

// Strict Architecture Enforcement (OpenShorts v2 + 5-Key Gemini Pool):
// - Speech transcription is NEVER performed in cloud functions (100% handled by local faster-whisper worker).
// - Supports up to 5 Google Gemini Free Tier API keys (GEMINI_API_KEY_1 .. GEMINI_API_KEY_5) with
//   round-robin job distribution and automatic 429/quota failover rotation to maximize video quality.

const FREE_TIER_ONLY = (process.env["FREE_TIER_ONLY"] ?? "true").toLowerCase() !== "false";
let activeGeminiRequests = 0;
let roundRobinCounter = 0;

const keyCooldownUntil = new Map<string, number>();

export function maskApiKey(key: string): string {
  const k = (key || "").trim();
  if (!k) return "Not set";
  if (k.length <= 10) return "****";
  return `${k.slice(0, 6)}…${k.slice(-4)}`;
}

/**
 * Retrieve up to 5 configured Gemini API keys in stable order without duplicates.
 * Supports GEMINI_API_KEY_1 .. GEMINI_API_KEY_5 (plus fallback GEMINI_API_KEY) and optional client-supplied keys.
 */
export function getConfiguredGeminiKeys(overrideKeys?: string[]): string[] {
  const rawList: string[] = [];
  if (Array.isArray(overrideKeys)) {
    for (const k of overrideKeys) {
      if (typeof k === "string" && k.trim().length > 10) rawList.push(k.trim());
    }
  }
  const envNames = [
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4",
    "GEMINI_API_KEY_5",
    "GEMINI_API_KEY",
  ];
  for (const name of envNames) {
    const val = process.env[name]?.trim();
    if (val && val.length > 10) rawList.push(val);
  }
  const unique: string[] = [];
  for (const k of rawList) {
    if (!unique.includes(k)) unique.push(k);
  }
  return unique.slice(0, 5);
}

export function isGeminiKeyRetryable(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toUpperCase();
  return [
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "QUOTA",
    "401",
    "403",
    "API_KEY",
    "API KEY",
    "PERMISSION_DENIED",
  ].some((tok) => msg.includes(tok));
}

function getConfiguredGeminiModel(): string {
  return process.env["GEMINI_VIDEO_MODEL"] || process.env["GEMINI_ANALYSIS_MODEL"] || "gemini-2.5-flash";
}

export type GeminiKeySlotStatus = {
  slot: number;
  envName: string;
  configured: boolean;
  masked: string;
  status: "ready" | "rate_limited" | "empty";
};

export type GeminiStatusInfo = {
  apiKeyConfigured: boolean;
  configuredKeyCount: number;
  maxKeySlots: number;
  effectiveThroughputMultiplier: number;
  model: string;
  freeTierOnly: boolean;
  maxConcurrentRequests: number;
  activeRequests: number;
  slots: GeminiKeySlotStatus[];
};

export const getGeminiConfigStatus = createServerFn({ method: "GET" }).handler(async (): Promise<GeminiStatusInfo> => {
  const now = Date.now();
  const slots: GeminiKeySlotStatus[] = [];
  for (let i = 1; i <= 5; i++) {
    const envName = `GEMINI_API_KEY_${i}`;
    const raw = (process.env[envName] || (i === 1 ? process.env["GEMINI_API_KEY"] : "") || "").trim();
    const configured = raw.length > 10;
    const cooldown = configured ? (keyCooldownUntil.get(raw) ?? 0) : 0;
    slots.push({
      slot: i,
      envName,
      configured,
      masked: configured ? maskApiKey(raw) : "Not set",
      status: !configured ? "empty" : cooldown > now ? "rate_limited" : "ready",
    });
  }
  const keys = getConfiguredGeminiKeys();
  const maxConcurrent = Math.max(1, Number(process.env["GEMINI_MAX_CONCURRENT_REQUESTS"] || String(Math.max(1, keys.length))));
  return {
    apiKeyConfigured: keys.length > 0,
    configuredKeyCount: keys.length,
    maxKeySlots: 5,
    effectiveThroughputMultiplier: Math.max(1, keys.length),
    model: getConfiguredGeminiModel(),
    freeTierOnly: FREE_TIER_ONLY,
    maxConcurrentRequests: maxConcurrent,
    activeRequests: activeGeminiRequests,
    slots,
  };
});

export type TimedWord = { w: string; s: number; e: number; speaker_id?: string | null };

/** Validate and normalize word-level timestamps so they are monotonic, bounded, and non-zero. */
export function validateMonotonicWords(rawWords: TimedWord[], maxDuration: number): TimedWord[] {
  const valid: TimedWord[] = [];
  let cursor = 0;
  for (const item of rawWords) {
    const w = String(item.w ?? "").trim();
    if (!w) continue;
    let s = Number(item.s);
    let e = Number(item.e);
    if (!Number.isFinite(s) || s < cursor) s = cursor;
    if (s > maxDuration) break;
    if (!Number.isFinite(e) || e <= s) e = Math.min(maxDuration, s + 0.12);
    e = Math.min(maxDuration, Math.max(s + 0.04, e));
    valid.push({ w, s, e, speaker_id: item.speaker_id ?? null });
    cursor = s;
  }
  return valid;
}

const EditingCueSchema = z.object({
  kind: z.enum(["explanation", "reveal", "surprise", "reaction", "serious", "action", "number", "pause"]).default("action"),
  start: z.number().nonnegative(),
  end: z.number().positive(),
  quote: z.string().default(""),
  confidence: z.number().min(0).max(1).default(0.85),
});

const StoryScoresSchema = z.object({
  hook_strength: z.number().min(0).max(100).default(85),
  standalone_context: z.number().min(0).max(100).default(82),
  narrative_completeness: z.number().min(0).max(100).default(84),
  pacing: z.number().min(0).max(100).default(86),
  ending_quality: z.number().min(0).max(100).default(80),
});

const ClipCandidateItemSchema = z.object({
  title: z.string().min(1).max(100),
  hook: z.string().min(1).max(90),
  start: z.number().nonnegative(),
  end: z.number().positive(),
  score: z.number().min(0).max(100),
  reason: z.string().min(1).max(400),
  category: z.string().default("highlight"),
  editingKind: z.enum(["challenge", "emotional", "comedy", "podcast", "educational", "general"]).default("general"),
  layoutIntent: z
    .enum(["preserve_scene", "single_subject", "two_speaker_split", "three_speaker_grid", "demonstration", "uncertain"])
    .default("uncertain"),
  suggestedLayout: z.enum(["auto", "fill", "split", "grid3", "fit", "editorial"]).default("auto"),
  suggestedBroll: z.string().default(""),
  hashtags: z.array(z.string()).max(12).default([]),
  storyScores: StoryScoresSchema.optional(),
  editingCues: z.array(EditingCueSchema).max(12).default([]),
});

const ClipRequestSchema = z
  .object({
    segments: z.array(z.object({ t: z.number(), text: z.string() })).max(4000),
    duration: z.number().positive(),
    minLen: z.number().min(5).max(180),
    maxLen: z.number().min(10).max(180),
    count: z.number().min(1).max(15),
    goal: z.string().max(1500).optional(),
    apiKeysPool: z.array(z.string()).max(5).optional(),
  })
  .refine((d) => d.minLen <= d.maxLen, { message: "minLen must be <= maxLen" });

export type AIClip = z.infer<typeof ClipCandidateItemSchema>;

export function filterAndDedupeClips(
  candidates: AIClip[],
  duration: number,
  minLen: number,
  maxLen: number,
  maxCount: number
): AIClip[] {
  const sorted = candidates
    .map((c) => {
      const start = Math.max(0, Math.min(duration - 3, Number(c.start) || 0));
      const end = Math.max(start + 3, Math.min(duration, Number(c.end) || start + minLen));
      return {
        title: String(c.title || "Highlight Clip").slice(0, 100),
        hook: String(c.hook || c.title || "").slice(0, 90),
        start,
        end,
        score: Math.max(0, Math.min(100, Math.round(Number(c.score) || 75))),
        reason: String(c.reason || "Strong 2-second hook and self-contained story beat.").slice(0, 400),
        category: String(c.category || "highlight"),
        editingKind: c.editingKind ?? "general",
        layoutIntent: c.layoutIntent ?? "uncertain",
        suggestedLayout: c.suggestedLayout ?? "auto",
        suggestedBroll: String(c.suggestedBroll || ""),
        hashtags: Array.isArray(c.hashtags) ? c.hashtags.slice(0, 8).map((h) => String(h)) : [],
        storyScores: c.storyScores,
        editingCues: Array.isArray(c.editingCues) ? c.editingCues : [],
      };
    })
    .filter((c) => {
      const len = c.end - c.start;
      return c.end > c.start && len >= Math.max(3, minLen * 0.75) && len <= maxLen * 1.25;
    })
    .sort((a, b) => b.score - a.score);

  const accepted: AIClip[] = [];
  for (const cand of sorted) {
    if (accepted.length >= maxCount) break;
    // Enforce zero overlap (> 5% overlap rejected) so consecutive clips never duplicate scenes
    const overlaps = accepted.some((ex) => {
      const inter = Math.max(0, Math.min(ex.end, cand.end) - Math.max(ex.start, cand.start));
      const minDur = Math.min(ex.end - ex.start, cand.end - cand.start);
      return minDur > 0 && inter / minDur > 0.05;
    });
    if (!overlaps) accepted.push(cand);
  }
  return accepted;
}

const OPENSHORTS_DIRECTOR_PROMPT = `You are a senior short-form video editor and OpenShorts Story Director.
Using ONLY the local Whisper timestamped transcript provided, select the strongest self-contained viral shorts.
Apply OpenShorts' strict editing rules:
1. THE 2-SECOND HOOK RULE (Rule 7): The clip MUST open directly on its strongest hook sentence. Never start mid-sentence or on filler ("just kind of like...", "I don't know, let's start it off...").
2. PAYOFF BOUNDARY & TRAILING QUESTION VETO (Rule 6): End the clip immediately after the punchline, reveal, or payoff completes. NEVER include a trailing unanswered question ("?") or a new turn addressing another contestant at the end of a clip.
3. HARD STOP BEFORE OUTROS & CREDITS (Rule 4): Never include episode sign-offs ("if you made it this far", "see y'all next week", "good night"), credit rolls, or post-credits bloopers.
4. ZERO OVERLAP: Never return overlapping time ranges across clips.
5. STORY & EDITING DIRECTOR SIGNALS: For each clip, include:
   - editingKind: "challenge" | "emotional" | "comedy" | "podcast" | "educational" | "general"
   - layoutIntent: "preserve_scene" (wide action/sports/group) | "single_subject" (1 speaker) | "two_speaker_split" (2 people talking) | "three_speaker_grid" (3 people interacting) | "demonstration" | "uncertain"
   - suggestedLayout: "auto" | "fill" | "split" | "grid3" | "fit" | "editorial"
   - storyScores: { hook_strength, standalone_context, narrative_completeness, pacing, ending_quality } (0-100)
   - editingCues: up to 4 grounded moments ({ kind: "number" | "surprise" | "reaction" | "reveal" | "explanation" | "action", start, end, quote, confidence }) for automatic zoom-punch and number emphasis.
Return strict JSON \`{ "clips": [...] }\`.`;

/**
 * Execute a Gemini SDK call rotating across the 5-key pool (round-robin start + automatic 429 failover).
 */
async function executeWithGeminiKeyPool<T>(
  keys: string[],
  runner: (apiKey: string, keySlotIndex: number) => Promise<T>
): Promise<{ result: T; usedKeySlot: number }> {
  const now = Date.now();
  const startIdx = roundRobinCounter++ % keys.length;
  const rotated = [...keys.slice(startIdx), ...keys.slice(0, startIdx)];
  const ready = rotated.filter((k) => (keyCooldownUntil.get(k) ?? 0) <= now);
  const ordered = [...ready, ...rotated.filter((k) => !ready.includes(k))];

  let lastError: unknown = null;
  for (let i = 0; i < ordered.length; i++) {
    const key = ordered[i]!;
    const slotNum = keys.indexOf(key) + 1;
    try {
      const result = await runner(key, slotNum);
      keyCooldownUntil.delete(key);
      return { result, usedKeySlot: slotNum };
    } catch (err) {
      lastError = err;
      if (isGeminiKeyRetryable(err)) {
        keyCooldownUntil.set(key, Date.now() + 60_000);
        continue;
      }
      throw err;
    }
  }
  throw lastError ?? new Error("All configured Gemini API keys in the 5-key pool are exhausted.");
}

export const findViralClips = createServerFn({ method: "POST" })
  .validator((d: unknown) => ClipRequestSchema.parse(d))
  .handler(
    async ({
      data,
    }): Promise<{ clips: AIClip[]; quotaExhausted?: boolean; error?: string; usedKeySlot?: number; totalPoolKeys?: number }> => {
      const keys = getConfiguredGeminiKeys(data.apiKeysPool);
      if (!keys.length) {
        throw new Error(
          "No Gemini API keys configured. Add up to 5 free-tier Google AI Studio keys (GEMINI_API_KEY_1..5) in Settings → AI & Processing."
        );
      }

      const maxConcurrent = Math.max(1, keys.length);
      if (activeGeminiRequests >= maxConcurrent) {
        throw new Error(
          `Concurrency limit (${maxConcurrent} across ${keys.length} key(s)) reached. Please wait a moment.`
        );
      }

      const modelName = getConfiguredGeminiModel();
      const transcript = data.segments.map((s) => `[${s.t.toFixed(1)}] ${s.text}`).join("\n");
      const userPrompt = `Return strict JSON \`{ "clips": [{ "title": string, "hook": string, "start": number, "end": number, "score": number, "reason": string, "category": string, "editingKind": "challenge"|"emotional"|"comedy"|"podcast"|"educational"|"general", "layoutIntent": "preserve_scene"|"single_subject"|"two_speaker_split"|"three_speaker_grid"|"demonstration"|"uncertain", "suggestedLayout": "auto"|"fill"|"split"|"grid3"|"fit"|"editorial", "suggestedBroll": string, "hashtags": string[], "storyScores": { "hook_strength": number, "standalone_context": number, "narrative_completeness": number, "pacing": number, "ending_quality": number }, "editingCues": [{ "kind": "number"|"surprise"|"reaction"|"reveal"|"explanation"|"action", "start": number, "end": number, "quote": string, "confidence": number }] }] }\`. Video duration ${data.duration.toFixed(1)}s. Find up to ${data.count} non-overlapping clips, each between ${data.minLen} and ${data.maxLen} seconds, aligned to sentence boundaries.${data.goal ? ` Discovery instruction / campaign brief: ${data.goal}` : ""}\n\nLocal Whisper Transcript:\n${transcript}`;

      activeGeminiRequests++;
      try {
        const { GoogleGenAI } = await import("@google/genai");
        const { result: text, usedKeySlot } = await executeWithGeminiKeyPool(keys, async (apiKey) => {
          const ai = new GoogleGenAI({ apiKey });
          const response = await ai.models.generateContent({
            model: modelName,
            contents: [{ role: "user", parts: [{ text: `${OPENSHORTS_DIRECTOR_PROMPT}\n\n${userPrompt}` }] }],
            config: { responseMimeType: "application/json" },
          });
          return response.text ?? "{}";
        });

        const rawParsed = JSON.parse(text) as { clips?: unknown[] };
        const rawList = Array.isArray(rawParsed.clips) ? rawParsed.clips : [];
        const validItems: AIClip[] = [];
        for (const item of rawList) {
          const res = ClipCandidateItemSchema.safeParse(item);
          if (res.success) validItems.push(res.data);
        }
        return {
          clips: filterAndDedupeClips(validItems, data.duration, data.minLen, data.maxLen, data.count),
          usedKeySlot,
          totalPoolKeys: keys.length,
        };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (/429|quota|RESOURCE_EXHAUSTED|rate limit/i.test(msg)) {
          return {
            clips: [],
            quotaExhausted: true,
            totalPoolKeys: keys.length,
            error: `All ${keys.length} configured Gemini Free Tier key(s) hit HTTP 429 rate limits. Your local Whisper transcript is safely preserved — Local Smart Discovery has been engaged automatically.`,
          };
        }
        throw err;
      } finally {
        activeGeminiRequests = Math.max(0, activeGeminiRequests - 1);
      }
    }
  );

export const CampaignRulesSchema = z.object({
  campaignName: z.string().default("Untitled Campaign"),
  platforms: z.array(z.string()).default(["TikTok", "Instagram Reels", "YouTube Shorts"]),
  minDurationSec: z.number().default(15),
  maxDurationSec: z.number().default(60),
  requiredHashtags: z.array(z.string()).default([]),
  requiredMentions: z.array(z.string()).default([]),
  requiredDisclosure: z.string().default("#ad"),
  requiredCta: z.string().default(""),
  prohibitedPhrases: z.array(z.string()).default([]),
  rightsNotice: z.string().default("New Client Content / derivatives remain Client property per Vyro Creator Terms."),
});

export type CampaignRules = z.infer<typeof CampaignRulesSchema>;

export const parseCampaignBrief = createServerFn({ method: "POST" })
  .validator((d: unknown) =>
    z.object({ briefText: z.string().min(10).max(10000), apiKeysPool: z.array(z.string()).max(5).optional() }).parse(d)
  )
  .handler(async ({ data }): Promise<CampaignRules> => {
    const keys = getConfiguredGeminiKeys(data.apiKeysPool);
    if (!keys.length) {
      throw new Error("No Gemini API keys configured for campaign brief extraction.");
    }
    const prompt = `Extract structured clipping campaign rules from this campaign brief. Return strict JSON matching: { "campaignName": string, "platforms": string[], "minDurationSec": number, "maxDurationSec": number, "requiredHashtags": string[], "requiredMentions": string[], "requiredDisclosure": string, "requiredCta": string, "prohibitedPhrases": string[], "rightsNotice": string }.\n\nBrief:\n${data.briefText}`;
    const { GoogleGenAI } = await import("@google/genai");
    const { result: text } = await executeWithGeminiKeyPool(keys, async (apiKey) => {
      const ai = new GoogleGenAI({ apiKey });
      const res = await ai.models.generateContent({
        model: getConfiguredGeminiModel(),
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        config: { responseMimeType: "application/json" },
      });
      return res.text ?? "{}";
    });
    try {
      return CampaignRulesSchema.parse(JSON.parse(text));
    } catch {
      return CampaignRulesSchema.parse({});
    }
  });
