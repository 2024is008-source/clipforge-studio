import { useEffect, useRef, useState } from "react";
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Download,
  Play,
  RefreshCw,
  ScanFace,
  Sparkles,
  KeyRound,
  Layers,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  fetchWorkerHealth,
  fetchWorkerSystemInfo,
  fetchWhisperModels,
  downloadAndCacheModel,
  runTestSampleTranscription,
  getLocalGeminiKeyPool,
  saveGeminiKeyPoolToWorker,
  checkGeminiKeyPoolHealth,
  fetchGeminiKeyPoolStatus,
  type WhisperConfig,
  type WhisperModelId,
  type WhisperModelStatus,
  type SystemInfoResponse,
  type WorkerHealth,
  type GeminiWorkerPoolStatus,
} from "@/lib/local-worker";
import { getGeminiConfigStatus, type GeminiStatusInfo } from "@/lib/ai.functions";

export function WhisperSettingsPanel({
  config,
  onChange,
  lastStatus,
  speakerAliases,
  onSpeakerAliasesChange,
}: {
  config: WhisperConfig;
  onChange: (patch: Partial<WhisperConfig>) => void;
  lastStatus: string;
  speakerAliases: Record<string, string>;
  onSpeakerAliasesChange: (aliases: Record<string, string>) => void;
}) {
  const [health, setHealth] = useState<WorkerHealth | null>(null);
  const [sysInfo, setSysInfo] = useState<SystemInfoResponse | null>(null);
  const [models, setModels] = useState<WhisperModelStatus[]>([]);
  const [geminiStatus, setGeminiStatus] = useState<GeminiStatusInfo | null>(null);
  const [workerPoolStatus, setWorkerPoolStatus] = useState<GeminiWorkerPoolStatus | null>(null);
  const [keySlots, setKeySlots] = useState<string[]>(() => getLocalGeminiKeyPool());
  const [savingKeys, setSavingKeys] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [testingSample, setTestingSample] = useState(false);
  const [testOutput, setTestOutput] = useState<string | null>(null);
  const sampleInputRef = useRef<HTMLInputElement>(null);

  async function refreshStatus() {
    const h = await fetchWorkerHealth();
    setHealth(h);
    if (h?.ok) {
      try {
        const [si, ms, wp] = await Promise.all([
          fetchWorkerSystemInfo(),
          fetchWhisperModels(),
          fetchGeminiKeyPoolStatus(),
        ]);
        setSysInfo(si);
        setModels(ms);
        if (wp) setWorkerPoolStatus(wp);
      } catch {
        setSysInfo(null);
      }
    } else {
      setSysInfo(null);
    }
    try {
      const gs = await getGeminiConfigStatus();
      setGeminiStatus(gs);
    } catch {
      // ignore server status error in offline mode
    }
  }

  useEffect(() => {
    void refreshStatus();
    const timer = setInterval(refreshStatus, 5000);
    return () => clearInterval(timer);
  }, []);

  async function handleSaveAndVerifyGeminiKeys() {
    setSavingKeys(true);
    try {
      const activeCount = keySlots.filter((k) => k.trim().length > 10).length;
      const saved = await saveGeminiKeyPoolToWorker(
        keySlots,
        "gemini-2.5-flash",
        Math.max(1, activeCount),
      );
      const checked = await checkGeminiKeyPoolHealth();
      if (checked) setWorkerPoolStatus(checked);
      else if (saved) setWorkerPoolStatus(saved);
      toast.success(
        `Saved ${activeCount}/5 Gemini API key(s) — ${Math.max(1, activeCount)}x Free-Tier Rotation Pool active!`,
      );
      await refreshStatus();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save Gemini key pool");
    } finally {
      setSavingKeys(false);
    }
  }

  async function handleDownloadModel(modelId: WhisperModelId, sizeMb: number) {
    setDownloadingModel(modelId);
    toast.info(
      `Downloading & caching faster-whisper '${modelId}' (~${sizeMb} MB) on local machine…`,
    );
    try {
      const res = await downloadAndCacheModel(modelId, config.device, config.computeType);
      toast.success(`Loaded '${res.model}' on ${res.device.toUpperCase()} (${res.compute_type})`);
      await refreshStatus();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Model download failed");
    } finally {
      setDownloadingModel(null);
    }
  }

  async function handleTestSample(file: File) {
    setTestingSample(true);
    setTestOutput(null);
    try {
      const res = await runTestSampleTranscription(file, config);
      setTestOutput(
        `[${res.device.toUpperCase()} · ${res.compute_type} · lang=${res.language} · ${res.words.length} words]: "${res.text.slice(0, 160)}"`,
      );
      toast.success(`Test transcription succeeded (${res.words.length} word timestamps verified)`);
      await refreshStatus();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Test transcription failed");
    } finally {
      setTestingSample(false);
    }
  }

  const hw = sysInfo?.hardware;
  const accel = sysInfo?.accelerator ?? health?.accelerator ?? null;
  const caps = sysInfo?.capabilities ?? null;
  const tierLabel =
    accel?.tier === "discrete"
      ? "Dedicated GPU"
      : accel?.tier === "integrated"
        ? "Integrated graphics"
        : accel?.tier === "cpu"
          ? "CPU only"
          : "Measuring";
  const localActiveKeysCount = keySlots.filter((k) => k.trim().length > 10).length;
  const totalConfiguredKeys = Math.max(
    localActiveKeysCount,
    workerPoolStatus?.configured_count ?? 0,
    geminiStatus?.configuredKeyCount ?? 0,
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4">
        <h2 className="font-display text-2xl font-extrabold tracking-tight">AI &amp; Processing</h2>
        <Button variant="outline" size="sm" onClick={() => refreshStatus()}>
          <RefreshCw className="size-4" />
          Refresh
        </Button>
      </div>

      <div className="rounded-xl border border-border bg-card p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2">
            <Layers className="mt-0.5 size-5 shrink-0 text-primary" />
            <div className="min-w-0">
              <h3 className="font-display font-bold">Render Hardware</h3>
              <p className="font-mono text-[11px] text-muted-foreground">
                Measured on this machine, not assumed
              </p>
            </div>
          </div>
          <span
            className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1 font-mono text-xs ${
              accel?.tier === "cpu"
                ? "bg-yellow-500/15 text-yellow-400"
                : "bg-primary/15 text-primary"
            }`}
          >
            <CheckCircle2 className="size-3.5" />
            {tierLabel}
          </span>
        </div>

        <div className="mt-4 grid gap-2 rounded-lg border border-border bg-surface p-3 font-mono text-xs text-muted-foreground sm:grid-cols-2">
          <div className="min-w-0 truncate" title={accel?.name}>
            Accelerator: <span className="text-foreground">{accel?.name ?? "—"}</span>
          </div>
          <div className="min-w-0 truncate" title={caps?.encoder?.args.join(" ")}>
            Encoder:{" "}
            <span className="text-primary">{accel?.video_encoder ?? health?.encoder ?? "—"}</span>
          </div>
          <div className="min-w-0">
            Transcription:{" "}
            <span className="text-foreground">
              {accel
                ? `${accel.transcription_device.toUpperCase()} · ${accel.transcription_compute} · ${accel.cpu_threads} threads`
                : "—"}
            </span>
          </div>
          <div className="min-w-0 truncate">
            Fallback:{" "}
            <span className="text-foreground">{caps?.software_fallback.name ?? "libx264"}</span>
          </div>
          <div className="min-w-0 sm:col-span-2">
            Output:{" "}
            <span className="text-foreground">
              {caps
                ? `${caps.delivery.width}×${caps.delivery.height} · ${caps.delivery.codec.toUpperCase()} ${caps.delivery.profile} · ${caps.delivery.bitrate} (max ${caps.delivery.max_bitrate})`
                : "1080×1920 · H264 high · 16M (max 18M)"}
            </span>
          </div>
        </div>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        {/* Card 1: Local Faster-Whisper Engine */}
        <div className="space-y-4 rounded-xl border border-border bg-card p-5">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2">
              <Cpu className="mt-0.5 size-5 shrink-0 text-primary" />
              <div className="min-w-0">
                <h3 className="font-display font-bold">Whisper Transcription</h3>
                <p className="font-mono text-[11px] text-muted-foreground">
                  100% local · no audio upload
                </p>
              </div>
            </div>
            {health?.ok ? (
              <span className="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-primary/15 px-3 py-1 font-mono text-xs text-primary">
                <CheckCircle2 className="size-3.5" />
                Online
              </span>
            ) : (
              <span className="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-yellow-500/15 px-3 py-1 font-mono text-xs text-yellow-400">
                <AlertTriangle className="size-3.5" />
                Offline
              </span>
            )}
          </div>

          <div className="grid gap-2 rounded-lg border border-border bg-surface p-3 font-mono text-xs text-muted-foreground sm:grid-cols-2">
            <div className="min-w-0">
              Engine:{" "}
              <span className="text-foreground">
                {hw
                  ? hw.whisper_installed
                    ? "faster-whisper"
                    : "not installed"
                  : health?.whisperInstalled
                    ? "faster-whisper"
                    : "—"}
              </span>
            </div>
            <div className="min-w-0 truncate" title={hw?.gpu_name ?? health?.gpuName ?? undefined}>
              CUDA:{" "}
              <span className="text-foreground">
                {hw?.cuda_available
                  ? (hw.gpu_name ?? "NVIDIA GPU")
                  : health?.cudaAvailable
                    ? (health.gpuName ?? "CUDA")
                    : "not available"}
              </span>
            </div>
            <div className="min-w-0">
              Device:{" "}
              <span className="text-primary">
                {hw?.active_device
                  ? `${hw.active_device.toUpperCase()} · ${hw.active_compute_type}`
                  : `${(hw?.recommended_device ?? "auto").toUpperCase()} · ready`}
              </span>
            </div>
            <div className="min-w-0">
              VRAM:{" "}
              <span className="text-foreground">
                {hw?.vram_total_mb
                  ? `${hw.vram_free_mb ?? "?"}/${hw.vram_total_mb} MB`
                  : "System RAM"}
              </span>
            </div>
            <div className="min-w-0 sm:col-span-2">
              Last run: <span className="text-foreground">{lastStatus || "Idle"}</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="block font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              Whisper model
            </label>
            <select
              value={config.model}
              aria-label="Whisper model"
              onChange={(e) => onChange({ model: e.target.value as WhisperModelId })}
              className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-base sm:text-sm"
            >
              <option value="tiny">Tiny · 75 MB · fastest</option>
              <option value="base">Base · 145 MB · fast</option>
              <option value="small">Small · 484 MB · recommended</option>
              <option value="medium">Medium · 1.5 GB · more RAM</option>
              <option value="large-v3">Large-v3 · 3.1 GB · most accurate</option>
            </select>

            {models.length > 0 && (
              <div className="rounded-md border border-border bg-background p-3">
                {models
                  .filter((m) => m.id === config.model)
                  .map((m) => (
                    <div
                      key={m.id}
                      className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <p className="min-w-0 text-xs font-semibold">
                        {m.label} · {m.download_size_mb} MB{" "}
                        <span className={m.cached ? "text-primary" : "text-muted-foreground"}>
                          · {m.cached ? "cached" : "not downloaded"}
                        </span>
                      </p>
                      <Button
                        variant="outline"
                        size="sm"
                        className="shrink-0 self-start sm:self-auto"
                        disabled={downloadingModel !== null}
                        onClick={() => handleDownloadModel(m.id, m.download_size_mb)}
                      >
                        {downloadingModel === m.id ? (
                          <Loader2 className="size-3.5 animate-spin" />
                        ) : (
                          <Download className="size-3.5" />
                        )}
                        {m.cached ? "Verify" : "Download"}
                      </Button>
                    </div>
                  ))}
              </div>
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="min-w-0">
              <label className="block font-mono text-[11px] uppercase text-muted-foreground">
                Compute device
              </label>
              <select
                value={config.device}
                aria-label="Compute device"
                onChange={(e) => onChange({ device: e.target.value as WhisperConfig["device"] })}
                className="mt-1.5 w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-base sm:text-xs"
              >
                <option value="auto">Auto · CUDA → CPU</option>
                <option value="cuda">CUDA GPU · int8_float16</option>
                <option value="cpu">CPU only · int8</option>
              </select>
            </div>
            <div className="min-w-0">
              <label className="block font-mono text-[11px] uppercase text-muted-foreground">
                Language
              </label>
              <select
                value={config.language}
                aria-label="Transcription language"
                onChange={(e) => onChange({ language: e.target.value })}
                className="mt-1.5 w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-base sm:text-xs"
              >
                <option value="auto">Auto-detect</option>
                <option value="en">English</option>
                <option value="si">Sinhala · සිංහල</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="hi">Hindi</option>
                <option value="ta">Tamil</option>
              </select>
            </div>
          </div>

          <div className="space-y-3 pt-1 text-sm">
            <label className="flex items-center justify-between gap-3">
              <span>Word-level timestamps</span>
              <Switch
                checked={config.wordTimestamps}
                onCheckedChange={(v) => onChange({ wordTimestamps: v })}
              />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>Silence filtering</span>
              <Switch
                checked={config.vadFilter}
                onCheckedChange={(v) => onChange({ vadFilter: v })}
              />
            </label>
            <input
              value={config.initialPrompt}
              aria-label="Vocabulary hints"
              onChange={(e) => onChange({ initialPrompt: e.target.value })}
              placeholder="Vocabulary, brand or name hints"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-base sm:text-xs"
            />
          </div>

          <div className="space-y-2 border-t border-border pt-3">
            <input
              ref={sampleInputRef}
              type="file"
              accept="audio/*,video/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void handleTestSample(f);
                e.target.value = "";
              }}
            />
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              disabled={!health?.ok || testingSample}
              onClick={() => sampleInputRef.current?.click()}
            >
              {testingSample ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              Test transcription
            </Button>
            {testOutput && (
              <p className="rounded bg-surface p-2.5 font-mono text-xs text-primary">
                {testOutput}
              </p>
            )}
          </div>
        </div>

        {/* Right Column: ClipForge Reframe Engine v2 + 5-Key Gemini Pool */}
        <div className="space-y-6">
          {/* Card 2: ClipForge Reframe Engine v2 & Editing Director */}
          <div className="space-y-4 rounded-xl border border-border bg-card p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                <ScanFace className="mt-0.5 size-5 shrink-0 text-primary" />
                <div className="min-w-0">
                  <h3 className="font-display font-bold">Reframe Engine</h3>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    Speaker tracking · smoothed camera
                  </p>
                </div>
              </div>
              <span className="shrink-0 whitespace-nowrap rounded-full bg-primary/15 px-2.5 py-1 font-mono text-[11px] font-semibold text-primary">
                v2 active
              </span>
            </div>

            <div className="space-y-2">
              <label className="block font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Tracking engine
              </label>
              <select
                value={config.trackingEngine ?? "openshorts_reframe_v2"}
                aria-label="Tracking engine"
                onChange={(e) =>
                  onChange({ trackingEngine: e.target.value as WhisperConfig["trackingEngine"] })
                }
                className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-base sm:text-xs"
              >
                <option value="openshorts_reframe_v2">ClipForge Reframe v2 · recommended</option>
                <option value="hybrid_yolo_bytetrack">Hybrid YOLO + ByteTrack</option>
                <option value="mediapipe_fast">MediaPipe GPU · fastest</option>
              </select>
            </div>

            <div className="grid gap-2 rounded-lg border border-border bg-surface p-3 font-mono text-[11px] text-muted-foreground sm:grid-cols-2">
              <div className="min-w-0">
                Jump rejection:{" "}
                <span className="text-primary">{config.jumpConfirmFrames ?? 3} frames</span>
              </div>
              <div className="min-w-0">
                Switch cooldown:{" "}
                <span className="text-primary">
                  {config.speakerSwitchCooldownFrames ?? 30} frames
                </span>
              </div>
              <div className="min-w-0">
                Split gate: <span className="text-foreground">50% overlap · 20% apart</span>
              </div>
              <div className="min-w-0">
                Speaker gate:{" "}
                <span className="text-primary">
                  {(config.activeSpeakerGate ?? true) ? "Mouth + audio" : "Geometry only"}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="block font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Camera smoothing · {(config.gimbalSmoothing ?? 1.55).toFixed(2)}σ
              </label>
              <Slider
                min={0.5}
                max={3.0}
                step={0.05}
                aria-label="Camera smoothing"
                value={[config.gimbalSmoothing ?? 1.55]}
                onValueChange={(v) => onChange({ gimbalSmoothing: v[0] ?? 1.55 })}
              />
            </div>

            <div className="space-y-2 border-t border-border pt-3">
              <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Speaker names
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {(["Speaker 1", "Speaker 2"] as const).map((spk) => (
                  <input
                    key={spk}
                    value={speakerAliases[spk] ?? ""}
                    aria-label={`${spk} name`}
                    onChange={(e) =>
                      onSpeakerAliasesChange({ ...speakerAliases, [spk]: e.target.value })
                    }
                    placeholder={`${spk} → ${spk === "Speaker 1" ? "Host" : "Guest"}`}
                    className="min-w-0 rounded-md border border-input bg-background px-3 py-1.5 text-base sm:text-xs"
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Card 3: 5-Gemini-API-Key Rotation Pool (Free Tier Booster) */}
          <div className="space-y-4 rounded-xl border border-border bg-card p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                <Sparkles className="mt-0.5 size-5 shrink-0 text-primary" />
                <div className="min-w-0">
                  <h3 className="font-display font-bold">Gemini Key Pool</h3>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    Round-robin · 429 failover
                  </p>
                </div>
              </div>
              <span
                className="flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full bg-primary/15 px-3 py-1 font-mono text-xs font-bold text-primary"
                title={`${Math.max(1, totalConfiguredKeys)}x throughput`}
              >
                <Layers className="size-3.5" />
                {totalConfiguredKeys}/5 keys
              </span>
            </div>

            <div className="space-y-2">
              {([0, 1, 2, 3, 4] as const).map((idx) => {
                const slotNum = idx + 1;
                const workerSlot = workerPoolStatus?.slots?.[idx];
                const serverSlot = geminiStatus?.slots?.[idx];
                const val = keySlots[idx] ?? "";
                const isSet =
                  val.trim().length > 10 || workerSlot?.configured || serverSlot?.configured;
                const statusLabel =
                  workerSlot?.status ?? serverSlot?.status ?? (isSet ? "ready" : "empty");
                return (
                  <div key={slotNum} className="flex flex-wrap items-center gap-2">
                    <span className="w-full shrink-0 font-mono text-[11px] text-muted-foreground sm:w-32">
                      GEMINI_API_KEY_{slotNum}
                    </span>
                    <input
                      type="password"
                      value={val}
                      aria-label={`Gemini API key ${slotNum}`}
                      onChange={(e) => {
                        const next = [...keySlots];
                        next[idx] = e.target.value;
                        setKeySlots(next);
                      }}
                      placeholder={
                        workerSlot?.configured
                          ? `On server · ${workerSlot.masked}`
                          : serverSlot?.configured
                            ? `In .env · ${serverSlot.masked}`
                            : `AIza… key ${slotNum}`
                      }
                      className="min-w-0 flex-1 rounded-md border border-input bg-background px-3 py-1.5 font-mono text-base sm:text-xs"
                    />
                    <span
                      className={`w-20 shrink-0 rounded px-2 py-1 text-center font-mono text-[10px] uppercase ${
                        statusLabel === "healthy" || statusLabel === "ready"
                          ? "bg-primary/15 text-primary"
                          : statusLabel === "rate_limited"
                            ? "bg-yellow-500/15 text-yellow-400"
                            : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {statusLabel === "healthy" || statusLabel === "ready"
                        ? "Ready"
                        : statusLabel === "rate_limited"
                          ? "Cooldown"
                          : "Empty"}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
              <div className="min-w-0 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground">
                  {workerPoolStatus?.model ?? geminiStatus?.model ?? "gemini-2.5-flash"}
                </span>{" "}
                · <span className="text-primary">{Math.max(1, totalConfiguredKeys)}</span> workers
              </div>
              <Button
                size="sm"
                onClick={() => void handleSaveAndVerifyGeminiKeys()}
                disabled={savingKeys}
              >
                {savingKeys ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <KeyRound className="size-3.5" />
                )}
                Save &amp; verify
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
