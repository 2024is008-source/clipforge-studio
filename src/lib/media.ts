// Audio & Word schema helpers. Primary transcription runs via local faster-whisper worker (not browser memory).

export type Word = {
  w: string;
  s: number;
  e: number;
  speaker_id?: string | null;
  source?: "whisper" | "whisper_fallback" | "imported" | "fallback";
  timing_warning?: boolean;
};

export type Chunk = { start: number; end: number; wav: Blob; voicedStart: number; voicedEnd: number };

const SR = 16000;

export async function decodeMono16k(file: File): Promise<Float32Array> {
  const buf = await file.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(buf);
  await ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

export function energyEnvelope(pcm: Float32Array, frame = 320): Float32Array {
  const n = Math.floor(pcm.length / frame);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const base = i * frame;
    for (let j = 1; j < frame; j++) {
      const diff = pcm[base + j]! - 0.94 * pcm[base + j - 1]!;
      sum += diff * diff;
    }
    out[i] = Math.sqrt(sum / Math.max(1, frame - 1));
  }
  return out;
}

function percentile(a: Float32Array, p: number) {
  const s = Array.from(a).sort((x, y) => x - y);
  return s[Math.floor(s.length * p)] ?? 0;
}

/**
 * Legacy fallback word aligner — retained ONLY for offline test/import edge cases.
 * Primary transcription MUST use genuine word-level timestamps from local faster-whisper.
 */
export function alignWords(text: string, chunk: Chunk, env: Float32Array): Word[] {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const fps = 50;
  const a = Math.floor(chunk.voicedStart * fps);
  const maxSpeechSec = Math.min(chunk.voicedEnd - chunk.voicedStart, Math.max(1.8, words.length * 0.42 + 1.2));
  const rawB = Math.ceil(chunk.voicedEnd * fps);
  const winFrames = Math.min(rawB - a, Math.ceil(maxSpeechSec * fps));
  const b = Math.max(a + 1, a + winFrames);
  const sub = env.subarray(a, b);
  const thr = Math.max(0.008, percentile(sub, 0.35));
  const mass: number[] = [0];
  for (let i = a; i < b; i++) {
    const eVal = env[i]!;
    mass.push(mass[mass.length - 1]! + (eVal > thr ? 1.0 + Math.min(1.5, eVal * 4) : 0.06));
  }
  const totalMass = mass[mass.length - 1]! || 1;
  const weights = words.map((w) => Math.max(2, w.replace(/[^\p{L}\p{N}]/gu, "").length) + (/[.,!?;:]$/.test(w) ? 2.5 : 0));
  const totalW = weights.reduce((x, y) => x + y, 0);
  const timeAtMass = (m: number) => {
    let lo = 0, hi = mass.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (mass[mid]! < m) lo = mid + 1; else hi = mid; }
    return (a + lo) / fps;
  };
  const out: Word[] = [];
  let acc = 0;
  for (let i = 0; i < words.length; i++) {
    const s = timeAtMass((acc / totalW) * totalMass);
    acc += weights[i]!;
    const rawE = timeAtMass((acc / totalW) * totalMass);
    const e = Math.min(s + 0.65, Math.max(rawE, s + 0.08));
    out.push({ w: words[i]!, s, e, speaker_id: null, source: "fallback", timing_warning: true });
  }
  return out;
}
