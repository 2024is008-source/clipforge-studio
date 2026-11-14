import { describe, expect, it } from "vitest";
import { snapClipToTranscriptWords, discoverClipsLocallyFromTranscript, DEFAULT_WHISPER_CONFIG } from "@/lib/local-worker";
import { smoothShotTrajectory } from "@/lib/reframe";
import { captionPages, serializeSubtitles, wordMotion } from "@/lib/subtitles";
import type { Word } from "@/lib/media";

describe("Phase 1 — Local Whisper Word Schema & Boundary Compatibility", () => {
  const whisperWords: Word[] = [
    { w: "Today", s: 1.25, e: 1.58, speaker_id: null, source: "whisper" },
    { w: "we", s: 1.58, e: 1.75, speaker_id: null, source: "whisper" },
    { w: "forge", s: 1.75, e: 2.15, speaker_id: null, source: "whisper" },
    { w: "clips.", s: 2.15, e: 2.65, speaker_id: "Host", source: "whisper" },
    { w: "Next", s: 3.10, e: 3.45, speaker_id: "Host", source: "whisper" },
    { w: "sentence!", s: 3.45, e: 4.10, speaker_id: "Host", source: "whisper" },
  ];

  it("defaults to small model with int8_float16 and word_timestamps=true for RTX 2050 profile", () => {
    expect(DEFAULT_WHISPER_CONFIG.model).toBe("small");
    expect(DEFAULT_WHISPER_CONFIG.computeType).toBe("int8_float16");
    expect(DEFAULT_WHISPER_CONFIG.wordTimestamps).toBe(true);
    expect(DEFAULT_WHISPER_CONFIG.vadFilter).toBe(true);
  });

  it("maintains 100% compatibility with captionPages, wordMotion, and serializeSubtitles", () => {
    const pages = captionPages(whisperWords, 3);
    // Phrase paging: "clips." ends the sentence, so the page runs to four words instead of being
    // cut at three mid-phrase. The point of this test is that the extended Whisper schema still
    // flows through every caption consumer unchanged.
    expect(pages.length).toBe(2);
    expect(pages[0]?.map((w) => w.w)).toEqual(["Today", "we", "forge", "clips."]);
    const motion = wordMotion("opus", 1.3, whisperWords[0]!);
    expect(motion.active).toBe(true);
    expect(motion.scale).toBeGreaterThan(1);
    const srt = serializeSubtitles(whisperWords, 1.0, 3.0, 4, "srt");
    expect(srt).toContain("Today we forge clips.");
  });

  it("snaps Gemini clip boundaries to actual local Whisper word timestamps", () => {
    const snapped = snapClipToTranscriptWords(1.1, 4.2, whisperWords, 10);
    expect(snapped.start).toBeCloseTo(1.25);
    expect(snapped.end).toBeCloseTo(4.25); // minimum 3s clip length enforced from 1.25s start
  });
});

describe("Automated Local Viral Clip Discovery & Zero-Phase Gimbal Smoothing", () => {
  it("automatically discovers ranked clips from a Whisper transcript even without Gemini API key", () => {
    const words: Word[] = Array.from({ length: 40 }, (_, i) => ({
      w: i % 8 === 0 ? "Million!" : "contestants",
      s: i * 1.0,
      e: i * 1.0 + 0.85,
      speaker_id: null,
      source: "whisper",
    }));
    const clips = discoverClipsLocallyFromTranscript(words, 45, 10, 20, 3);
    expect(clips.length).toBeGreaterThan(0);
    expect(clips[0]!.score).toBeGreaterThanOrEqual(74);
  });

  it("smooths camera jitter within a shot while keeping the first frame marked as a scene cut", () => {
    const noisy = [
      { t: 0, x: 0.3, y: 0.4, cut: true },
      { t: 0.14, x: 0.7, y: 0.4, cut: false },
      { t: 0.28, x: 0.3, y: 0.4, cut: false },
    ];
    const smoothed = smoothShotTrajectory(noisy, 1.5);
    expect(smoothed[0]?.cut).toBe(true);
    expect(Math.abs((smoothed[1]?.x ?? 0) - 0.5)).toBeLessThan(0.15);
  });
});
