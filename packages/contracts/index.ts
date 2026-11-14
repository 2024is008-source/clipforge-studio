/**
 * ClipForge Shared Domain Contracts
 * Centralizes data structures exchanged between the Studio Frontend, Server Functions, and Local GPU Worker.
 */
export type { Word } from "../../src/lib/media";
export type {
  FrameLayoutMode,
  FaceBox,
  TrackPoint,
  SpeakerReport,
  FrameSignature,
} from "../../src/lib/reframe";
export type {
  ClipSettings,
  ClipJob,
  ExportResult,
} from "../../src/lib/render";
export type {
  WhisperConfig,
  WhisperModelId,
  WhisperModelStatus,
  WorkerHealth,
  AcceleratorSummary,
  SystemInfoResponse,
  GeminiWorkerPoolStatus,
} from "../../src/lib/local-worker";
export type { AIClip, CampaignRules, GeminiStatusInfo } from "../../src/lib/ai.functions";
