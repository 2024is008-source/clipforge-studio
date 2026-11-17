import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Upload,
  Sparkles,
  Cpu,
  Scissors,
  Download,
  Play,
  Pause,
  ScanFace,
  Loader2,
  Flame,
  Wand2,
  RotateCcw,
  Plus,
  CheckCircle2,
  Settings,
  ArrowLeft,
  Sliders,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { YouTubeImport } from "@/components/YouTubeImport";
import { CaptionPanel } from "@/components/CaptionPanel";
import { VyroWorkspace } from "@/components/VyroWorkspace";
import { WhisperSettingsPanel } from "@/components/WhisperSettingsPanel";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { findViralClips, type AIClip } from "@/lib/ai.functions";
import { type Word } from "@/lib/media";
import {
  DEFAULT_WHISPER_CONFIG,
  transcribeWithLocalWhisper,
  snapClipToTranscriptWords,
  trimClipNarrativeBoundaries,
  discoverClipsLocallyFromTranscript,
  fetchWorkerHealth,
  getLocalGeminiKeyPool,
  analyzeReframeWithLocalWorker,
  workerSourceId,
  type WhisperConfig,
  type AcceleratorSummary,
} from "@/lib/local-worker";
import {
  trackFaces,
  getFaceDetector,
  onVisionDelegateChange,
  lastVisionNotice,
  lastVisionError,
  type FrameLayoutMode,
  type SpeakerReport,
  type TrackPoint,
} from "@/lib/reframe";
import {
  drawFrame,
  exportClip,
  gpuInfo,
  shortGpuName,
  numberCues,
  W,
  H,
  type ClipSettings,
  type ClipJob,
  type ExportResult,
} from "@/lib/render";

type Clip = AIClip & {
  id: string;
  track?: TrackPoint[] | undefined;
  /** Who the GPU worker decided had to be on screen, and whether its own track obeyed. */
  speakers?: SpeakerReport | undefined;
  settings: ClipSettings;
  exportUrl?: string;
  exportMeta?: Omit<ExportResult, "blob">;
};
type Stage = "idle" | "transcribe" | "analyze" | "tracking" | "ready";
type NavTab = "studio" | "settings";
type StudioView = "setup" | "editor";

const DEFAULTS: ClipSettings = {
  layout: "auto",
  captionStyle: "opus",
  highlight: "#22ff4e",
  secondaryHighlight: "#ffd400",
  captionY: 0.78,
  showHook: true,
  showEmojis: true,
  progressBar: false,
  zoomPunch: true,
  animations: true,
  effects: true,
  wordsPerPage: 3,
  fps: "29.97",
  voiceEnhance: true,
  manualX: null,
  manualY: null,
};
const SWATCHES = ["#22ff4e", "#c6ff3d", "#ffd400", "#ff4d4d", "#3dd9ff", "#ffffff"];
const LAYOUTS = [
  ["auto", "Auto"],
  ["fill", "Fill 9:16"],
  ["split", "2-Up Split"],
  ["grid3", "3-Up Grid"],
  ["editorial", "Stack"],
  ["fit", "Fit + blur"],
] as const;
const QUICK_GOALS = [
  "Find the strongest hooks & high-energy moments",
  "Find emotional stories & personal payoffs",
  "Find surprising facts & educational insights",
  "Find funny reactions & entertaining banter",
  "Find moments suitable for my Vyro campaign",
];
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const sourceIds = new WeakMap<File, Promise<string>>();
const filesByUrl = new Map<string, File>();

function sourceIdFor(file: File) {
  let id = sourceIds.get(file);
  if (!id) {
    id = workerSourceId(file);
    sourceIds.set(file, id);
  }
  return id;
}

async function trackWithWorkerFallback(
  src: string,
  file: File | null,
  start: number,
  end: number,
  layout: ClipSettings["layout"],
  onProgress?: (p: number) => void,
  signal?: AbortSignal,
): Promise<{ track: TrackPoint[]; speakers?: SpeakerReport } | undefined> {
  const framingMode = layout;
  try {
    const source = file ?? filesByUrl.get(src) ?? null;
    if (!source) throw new Error("No source file for worker analysis");
    return await analyzeReframeWithLocalWorker(source, start, end, framingMode, signal, {
      sourceId: await sourceIdFor(source),
      onProgress,
    });
  } catch (e) {
    if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) throw e;
    // The GPU worker is the accurate tracker; say so loudly when a clip falls back to MediaPipe.
    console.warn("Local worker reframe unavailable, using browser tracker instead", e);
    return trackFaces(src, start, end, onProgress)
      .then((track) => ({ track }))
      .catch(() => undefined);
  }
}

export default function Studio() {
  const [navTab, setNavTab] = useState<NavTab>("studio");
  const [studioView, setStudioView] = useState<StudioView>("setup");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [timeframe, setTimeframe] = useState<[number, number]>([0, 0]);
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [workerLogs, setWorkerLogs] = useState<string[]>([]);
  const [words, setWords] = useState<Word[]>([]);
  const [whisperConfig, setWhisperConfig] = useState<WhisperConfig>(DEFAULT_WHISPER_CONFIG);
  const [defaultClipSettings, setDefaultClipSettings] = useState<ClipSettings>(DEFAULTS);
  const [lastWhisperStatus, setLastWhisperStatus] = useState<string>("Idle");
  const [speakerAliases, setSpeakerAliases] = useState<Record<string, string>>({});
  const [clips, setClips] = useState<Clip[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [gpu, setGpu] = useState<string | null>(null);
  const [faceDelegate, setFaceDelegate] = useState<string>("loading");
  const [visionReason, setVisionReason] = useState<string | null>(null);
  const [workerOnline, setWorkerOnline] = useState<boolean>(false);
  const [accelerator, setAccelerator] = useState<AcceleratorSummary | null>(null);
  const [goal, setGoal] = useState("");
  const [len, setLen] = useState<[number, number]>([20, 60]);
  const [count, setCount] = useState(6);
  const transcribeAbortRef = useRef<AbortController | null>(null);

  const initVision = useCallback(() => {
    setFaceDelegate("loading");
    setVisionReason(null);
    getFaceDetector()
      .then((d) => setFaceDelegate(`${d.delegate} (${d.objDet ? "Face + Person/Object" : "Face"})`))
      .catch(() => {
        setFaceDelegate("unavailable");
        setVisionReason(lastVisionError());
      });
  }, []);

  useEffect(() => {
    gpuInfo().then(setGpu);
    initVision();
    const stopListening = onVisionDelegateChange((label) => {
      setFaceDelegate(label);
      const notice = lastVisionNotice();
      if (notice) toast.info("Vision moved to CPU", { description: notice });
    });
    const checkW = () =>
      fetchWorkerHealth().then((h) => {
        setWorkerOnline(Boolean(h?.ok));
        setAccelerator(h?.accelerator ?? null);
      });
    void checkW();
    const timer = setInterval(checkW, 4000);
    return () => {
      clearInterval(timer);
      stopListening();
    };
  }, [initVision]);

  const active = clips.find((c) => c.id === activeId) ?? null;
  const update = (id: string, patch: Partial<Clip>) =>
    setClips((cs) => cs.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  const patchDefaultSettings = (patch: Partial<ClipSettings>) =>
    setDefaultClipSettings((prev) => ({ ...prev, ...patch }));

  async function onFile(f: File) {
    if (url) URL.revokeObjectURL(url);
    const u = URL.createObjectURL(f);
    filesByUrl.set(u, f);
    setFile(f);
    setUrl(u);
    setClips([]);
    setWords([]);
    setActiveId(null);
    setStage("idle");
    setStudioView("setup");
    setNavTab("studio");
    const v = document.createElement("video");
    v.src = u;
    v.onloadedmetadata = () => {
      setDuration(v.duration);
      setTimeframe([0, v.duration]);
    };
  }

  async function runLocalTranscription(forceRetranscribe = false): Promise<Word[]> {
    if (!file) return [];
    if (words.length > 0 && !forceRetranscribe) {
      return words;
    }
    setStage("transcribe");
    setProgress(0);
    setWorkerLogs([]);
    transcribeAbortRef.current = new AbortController();
    const res = await transcribeWithLocalWhisper(
      file,
      whisperConfig,
      (p, st, logs) => {
        setProgress(p);
        setLastWhisperStatus(`${st} (${Math.round(p * 100)}%)`);
        setWorkerLogs(logs.slice(-4));
      },
      transcribeAbortRef.current.signal,
      forceRetranscribe,
    );
    if (!res.words.length)
      throw new Error("Local Whisper detected no spoken words in this media file.");
    setWords(res.words);
    setLastWhisperStatus(
      `Completed: ${res.words.length} words on ${res.device.toUpperCase()} (${res.compute_type})${res.cached ? " [Cached]" : ""}`,
    );
    if (res.warnings.length > 0) {
      toast.warning(res.warnings[0]);
    }
    return res.words;
  }

  /**
   * Fully Automated 1-Click ClipForge Pipeline:
   * 1. Transcribes audio locally with faster-whisper (word_timestamps=True).
   * 2. Discovers viral clips via Gemini Free Tier (or automatic Local Smart Hook Ranker if Gemini is unconfigured/paused).
   * 3. Automatically runs Hybrid Face + Person + Object Tracking & Gimbal Smoothing on discovered clips.
   * 4. Opens the Studio Editor with all clips framed, captioned, and ready to export!
   */
  async function analyze(captionsOnly = false, forceRetranscribe = false) {
    if (!file) return;
    try {
      const localWords = await runLocalTranscription(forceRetranscribe);
      const mediaDur = duration || (localWords.at(-1)?.e ?? 30);
      const tfRange: [number, number] = [
        timeframe[0] ?? 0,
        timeframe[1] && timeframe[1] > (timeframe[0] ?? 0) ? timeframe[1] : mediaDur,
      ];

      if (captionsOnly) {
        setStage("tracking");
        setProgress(0);
        const track = await trackFaces(url, tfRange[0], tfRange[1], setProgress).catch(
          () => undefined,
        );
        const c: Clip = {
          id: `s${Date.now()}`,
          title: "Full Subtitled Video",
          hook: "",
          start: tfRange[0],
          end: tfRange[1],
          score: 92,
          reason:
            "Transcribed locally via faster-whisper with genuine word timestamps and auto-reframed.",
          category: "full",
          editingKind: "general",
          layoutIntent: "uncertain",
          suggestedLayout: defaultClipSettings.layout,
          suggestedBroll: "",
          hashtags: [],
          editingCues: [],
          track,
          settings: { ...defaultClipSettings, showHook: false },
        };
        setClips([c]);
        setActiveId(c.id);
        setStage("ready");
        setStudioView("editor");
        toast.success(`Generated ${localWords.length} Whisper words & auto-reframed video`);
        return;
      }

      setStage("analyze");
      const scopedWords = localWords.filter((w) => w.s >= tfRange[0] && w.e <= tfRange[1]);
      const wordsForDiscovery = scopedWords.length >= 6 ? scopedWords : localWords;
      const segments: { t: number; text: string }[] = [];
      for (let i = 0; i < wordsForDiscovery.length; i += 12) {
        const word = wordsForDiscovery[i];
        if (word)
          segments.push({
            t: word.s,
            text: wordsForDiscovery
              .slice(i, i + 12)
              .map((w) => w.w)
              .join(" "),
          });
      }

      let discovered: AIClip[] = [];
      try {
        const { clips: found } = await findViralClips({
          data: {
            segments,
            duration: mediaDur,
            minLen: len[0],
            maxLen: len[1],
            count,
            goal: goal || undefined,
            apiKeysPool: getLocalGeminiKeyPool(),
          },
        });
        discovered = found;
      } catch {
        // Automatic seamless fallback to Local Smart Viral Hook Ranker
        discovered = [];
      }

      if (!discovered.length) {
        discovered = discoverClipsLocallyFromTranscript(
          localWords,
          mediaDur,
          len[0],
          len[1],
          count,
          tfRange,
        );
      }

      const mapped: Clip[] = discovered.map((c, i) => {
        const snapped = snapClipToTranscriptWords(c.start, c.end, localWords, mediaDur);
        return {
          ...c,
          start: snapped.start,
          end: snapped.end,
          id: `c${Date.now()}${i}`,
          settings: {
            ...defaultClipSettings,
            // The discovery model has only ever seen the transcript, yet its `suggestedLayout` used to
            // pin each clip to one template, so adjacent clips changed composition for a reason no
            // frame supports. The template now comes from the tracker's measurements per shot.
            layout: defaultClipSettings.layout,
          },
        };
      });

      // Stage 3: run Hybrid Object/Person/Face Tracking on every discovered clip; an untracked
      // clip would export with a dead-centre static crop, so there is no cap here.
      setStage("tracking");
      setProgress(0);
      setClips(mapped);
      setActiveId(mapped[0]?.id ?? null);
      for (let i = 0; i < mapped.length; i++) {
        const item = mapped[i]!;
        const tracked = await trackWithWorkerFallback(
          url,
          file,
          item.start,
          item.end,
          item.settings.layout,
          (p) => {
            setProgress((i + p) / mapped.length);
          },
        );
        item.track = tracked?.track;
        item.speakers = tracked?.speakers;
        setProgress((i + 1) / mapped.length);
        setClips([...mapped]);
      }

      setClips(mapped);
      setActiveId(mapped[0]?.id ?? null);
      setStage("ready");
      setStudioView("editor");
      toast.success(
        `Auto-generated ${mapped.length} viral clips with Whisper captions & gimbal reframing`,
      );
    } catch (e) {
      console.error(e);
      toast.error(e instanceof Error ? e.message : "Automated pipeline failed");
      setStage("idle");
    }
  }

  function addManual() {
    const c: Clip = {
      id: `m${Date.now()}`,
      title: "Custom clip",
      hook: "",
      start: timeframe[0] ?? 0,
      end: Math.min((timeframe[0] ?? 0) + 30, duration || 30),
      score: 85,
      reason: "Manually created",
      category: "manual",
      editingKind: "general",
      layoutIntent: "uncertain",
      suggestedLayout: defaultClipSettings.layout,
      suggestedBroll: "",
      hashtags: [],
      editingCues: [],
      settings: { ...defaultClipSettings },
    };
    setClips((cs) => [...cs, c]);
    setActiveId(c.id);
    setStudioView("editor");
  }

  return (
    <div className="min-h-screen bg-glow">
      <header className="sticky top-0 z-40 flex flex-wrap items-center justify-between gap-3 border-b border-border bg-background/85 px-4 py-3 backdrop-blur-md sm:px-6">
        <div className="flex items-center gap-3">
          <div className="grid size-9 place-items-center rounded-md bg-primary text-primary-foreground">
            <Scissors className="size-5" />
          </div>
          <h1 className="font-display text-xl font-extrabold tracking-tight">ClipForge Studio</h1>
        </div>

        {/* Navigation Tabs: Studio vs Dedicated Settings Page */}
        <div className="flex items-center gap-2">
          <Button
            variant={navTab === "studio" ? "default" : "outline"}
            size="sm"
            onClick={() => setNavTab("studio")}
          >
            <Wand2 className="size-4" />
            Studio
          </Button>
          <Button
            variant={navTab === "settings" ? "default" : "outline"}
            size="sm"
            onClick={() => setNavTab("settings")}
          >
            <Settings className="size-4" />
            Settings
          </Button>
        </div>

        <div className="hidden flex-wrap items-center gap-2 font-mono text-[11px] xl:flex">
          <span className="flex items-center gap-1.5 whitespace-nowrap rounded-full border border-border bg-surface px-3 py-1">
            <CheckCircle2
              className={`size-3.5 ${workerOnline ? "text-primary" : "text-yellow-400"}`}
            />
            {workerOnline ? "Worker online" : "Worker offline"}
          </span>
          <span
            className="flex max-w-56 items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1"
            title={
              accelerator
                ? `Export: ${accelerator.video_encoder_label} · Transcription: ${accelerator.transcription_device} ${accelerator.transcription_compute} · Same 1080×1920 quality on every tier`
                : (gpu ?? undefined)
            }
          >
            <Cpu className="size-3.5 shrink-0 text-primary" />
            <span className="truncate">
              {accelerator?.name ?? (gpu ? shortGpuName(gpu) : "Detecting hardware…")}
            </span>
          </span>
          {faceDelegate === "unavailable" ? (
            <button
              type="button"
              onClick={initVision}
              title={`Browser face tracking failed to start${visionReason ? `: ${visionReason}` : ""}. Click to retry.${workerOnline ? " The local worker still tracks faces." : ""}`}
              className="flex items-center gap-1.5 whitespace-nowrap rounded-full border border-yellow-500/40 bg-yellow-500/10 px-3 py-1 text-yellow-400 transition-colors hover:border-yellow-400 hover:text-yellow-300"
            >
              <ScanFace className="size-3.5" />
              Vision: unavailable
              <RotateCcw className="size-3" />
            </button>
          ) : (
            <span
              className="flex items-center gap-1.5 whitespace-nowrap rounded-full border border-border bg-surface px-3 py-1"
              title={`Vision delegate: ${faceDelegate}`}
            >
              {faceDelegate === "loading" ? (
                <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
              ) : (
                <ScanFace className="size-3.5 text-primary" />
              )}
              Vision: {faceDelegate.split(" (")[0]}
            </span>
          )}
        </div>
      </header>

      {navTab === "settings" ? (
        <WhisperSettingsPanel
          config={whisperConfig}
          onChange={(patch) => setWhisperConfig((prev) => ({ ...prev, ...patch }))}
          lastStatus={lastWhisperStatus}
          speakerAliases={speakerAliases}
          onSpeakerAliasesChange={setSpeakerAliases}
        />
      ) : !file ? (
        <Dropzone onFile={onFile} />
      ) : studioView === "setup" ? (
        /* CLIPFORGE PRE-GENERATION SETUP PAGE (Collects all inputs from Images 4, 5, 6, 7 before 1-Click Generation) */
        <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
          {/* Top Source Video Preview & Processing Timeframe Bar */}
          <div className="grid gap-6 rounded-xl border border-border bg-card p-5 lg:grid-cols-[340px_minmax(0,1fr)]">
            <div className="space-y-2">
              <video
                src={url}
                className="aspect-video w-full rounded-lg bg-background object-cover"
                controls
              />
              <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
                <span className="truncate pr-2">{file.name}</span>
                <span>{fmt(duration)}</span>
              </div>
            </div>
            <div className="flex flex-col justify-between space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <span className="inline-block rounded-full bg-primary/15 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-primary">
                    Step 1 · Setup
                  </span>
                  <h2 className="mt-1 font-display text-2xl font-extrabold">
                    Configure your clips
                  </h2>
                </div>
                <div className="flex gap-2">
                  {clips.length > 0 && (
                    <Button variant="secondary" size="sm" onClick={() => setStudioView("editor")}>
                      View Generated Clips ({clips.length})
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setFile(null);
                      setStage("idle");
                    }}
                  >
                    <RotateCcw className="size-4" />
                    Change video
                  </Button>
                </div>
              </div>

              <div className="space-y-2 rounded-lg border border-border bg-surface p-3.5">
                <div className="flex flex-wrap items-center justify-between gap-x-3 font-mono text-xs">
                  <span className="uppercase tracking-wider text-muted-foreground">Timeframe</span>
                  <span className="whitespace-nowrap text-primary">
                    {fmt(timeframe[0])} → {fmt(timeframe[1] || duration)} ·{" "}
                    {Math.round((timeframe[1] || duration) - timeframe[0])}s
                  </span>
                </div>
                <Slider
                  min={0}
                  max={Math.max(1, duration)}
                  step={1}
                  aria-label="Processing timeframe"
                  value={[timeframe[0], timeframe[1] || duration]}
                  onValueChange={(v) =>
                    setTimeframe([v[0] ?? 0, Math.max((v[0] ?? 0) + 5, v[1] ?? duration)])
                  }
                />
              </div>

              {/* Sticky / Prominent 1-Click Automated Launch Bar */}
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <Button
                  size="lg"
                  className="flex-1 text-base font-bold shadow-glow"
                  onClick={() => analyze(false, false)}
                  disabled={stage !== "idle" && stage !== "ready"}
                >
                  {stage === "idle" || stage === "ready" ? (
                    <>
                      <Sparkles className="size-5" />
                      Get Clips in 1 Click
                    </>
                  ) : (
                    <>
                      <Loader2 className="size-5 animate-spin" />
                      Running…
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => analyze(true, false)}
                  disabled={stage !== "idle" && stage !== "ready"}
                >
                  Subtitles only
                </Button>
                <Button variant="ghost" size="lg" onClick={addManual}>
                  <Plus className="size-4" />
                  Manual Editor
                </Button>
              </div>

              {stage !== "idle" && stage !== "ready" && (
                <Pipeline
                  stage={stage}
                  progress={progress}
                  logs={workerLogs}
                  onCancel={() => transcribeAbortRef.current?.abort()}
                />
              )}
            </div>
          </div>

          {/* 4-Card Setup Grid Matching Images 4, 5, 6, 7 */}
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-6">
              {/* IMAGE 5: Clip Discovery & Whisper Actions */}
              <div className="space-y-4 rounded-xl border border-border bg-card p-5">
                <h3 className="font-display text-lg font-bold">Clip Discovery & Whisper Actions</h3>
                <div className="flex flex-wrap gap-1.5">
                  {(
                    [
                      ["< 30s", [10, 30]],
                      ["20–60s (Viral Default)", [20, 60]],
                      ["30–60s", [30, 60]],
                      ["60–90s", [60, 90]],
                    ] as const
                  ).map(([label, range]) => (
                    <Button
                      key={label}
                      variant={len[0] === range[0] && len[1] === range[1] ? "default" : "outline"}
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => setLen([range[0], range[1]])}
                    >
                      {label}
                    </Button>
                  ))}
                </div>
                <Field label={`CLIP LENGTH · ${len[0]}–${len[1]}S`}>
                  <Slider
                    aria-label="Clip length"
                    min={10}
                    max={120}
                    step={5}
                    value={len}
                    onValueChange={(v) => setLen([v[0], v[1]] as [number, number])}
                  />
                </Field>
                <Field label={`CLIPS TO FIND · ${count}`}>
                  <Slider
                    aria-label="Clips to find"
                    min={1}
                    max={15}
                    step={1}
                    value={[count]}
                    onValueChange={(v) => setCount(v[0] ?? 6)}
                  />
                </Field>
                <Field label="DISCOVERY INSTRUCTION / CAMPAIGN GOAL">
                  <Textarea
                    value={goal}
                    onChange={(e) => setGoal(e.target.value)}
                    placeholder="e.g. Find the strongest hook, emotional moments, or clips for my Vyro campaign…"
                    rows={3}
                  />
                </Field>
                <div className="flex flex-wrap gap-1.5">
                  {QUICK_GOALS.map((g) => (
                    <button
                      key={g}
                      type="button"
                      onClick={() => setGoal(g)}
                      className="rounded-full border border-border bg-surface px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
                    >
                      + {g}
                    </button>
                  ))}
                </div>
              </div>

              {/* IMAGE 6: Framing & Render Settings */}
              <div className="space-y-4 rounded-xl border border-border bg-card p-5">
                <h3 className="font-display text-lg font-bold">Framing & Render Settings</h3>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {LAYOUTS.map(([l, label]) => (
                    <Button
                      key={l}
                      variant={defaultClipSettings.layout === l ? "default" : "outline"}
                      className="px-2 text-xs"
                      onClick={() => patchDefaultSettings({ layout: l })}
                    >
                      {label}
                    </Button>
                  ))}
                </div>
                <Field
                  label={`MANUAL HORIZONTAL CROP OVERRIDE · ${
                    typeof defaultClipSettings.manualX === "number"
                      ? `${Math.round(defaultClipSettings.manualX * 100)}%`
                      : "AUTO TRACKED"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Slider
                      aria-label="Manual horizontal crop"
                      min={0.15}
                      max={0.85}
                      step={0.01}
                      value={[defaultClipSettings.manualX ?? 0.5]}
                      onValueChange={(v) => patchDefaultSettings({ manualX: v[0] ?? 0.5 })}
                    />
                    {typeof defaultClipSettings.manualX === "number" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => patchDefaultSettings({ manualX: null })}
                      >
                        Reset
                      </Button>
                    )}
                  </div>
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="TARGET FRAME RATE">
                    <select
                      value={defaultClipSettings.fps ?? "29.97"}
                      aria-label="Target frame rate"
                      onChange={(e) =>
                        patchDefaultSettings({ fps: e.target.value as "29.97" | "30" })
                      }
                      className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-base sm:text-xs"
                    >
                      <option value="29.97">29.97 fps</option>
                      <option value="30">30.00 fps</option>
                    </select>
                  </Field>
                  <Field label="CAPTION TIMING SOURCE">
                    <div className="rounded-md border border-border bg-surface px-3 py-2 font-mono text-xs text-primary">
                      Faster-Whisper
                    </div>
                  </Field>
                </div>
                <Toggle
                  label="Animations"
                  v={defaultClipSettings.animations ?? true}
                  on={(v) => patchDefaultSettings({ animations: v })}
                />
                <Toggle
                  label="Effects"
                  v={defaultClipSettings.effects ?? true}
                  on={(v) => patchDefaultSettings({ effects: v })}
                />
                <Toggle
                  label="Zoom punch on emphasis"
                  v={defaultClipSettings.zoomPunch}
                  on={(v) => patchDefaultSettings({ zoomPunch: v })}
                />
                <Toggle
                  label="Voice loudness & EBU R128 limiter"
                  v={defaultClipSettings.voiceEnhance ?? true}
                  on={(v) => patchDefaultSettings({ voiceEnhance: v })}
                />
              </div>

              {/* IMAGE 4: Vyro Campaign Workspace */}
              <VyroWorkspace
                clip={active}
                clipWords={words}
                exportSha256={active?.exportMeta?.sha256}
                onApplyBriefToGoal={setGoal}
              />
            </div>

            {/* Right Column — IMAGE 7: Subtitle Studio & Preset Templates */}
            <div className="space-y-4 rounded-xl border border-border bg-card p-5">
              <CaptionPanel
                settings={defaultClipSettings}
                setSettings={patchDefaultSettings}
                words={words}
                start={timeframe[0]}
                end={timeframe[1] || duration}
                title={file.name}
                onWords={setWords}
              />
              <Field label="Highlight colour">
                <div className="flex gap-2.5">
                  {SWATCHES.map((c) => (
                    <button
                      key={c}
                      aria-label={`Colour ${c}`}
                      onClick={() => patchDefaultSettings({ highlight: c })}
                      className={`size-8 rounded-full border-2 ${
                        defaultClipSettings.highlight === c
                          ? "border-foreground scale-110"
                          : "border-transparent"
                      }`}
                      style={{ background: c }}
                    />
                  ))}
                </div>
              </Field>
              <Field label={`Words per page · ${defaultClipSettings.wordsPerPage}`}>
                <Slider
                  aria-label="Words per page"
                  min={1}
                  max={6}
                  step={1}
                  value={[defaultClipSettings.wordsPerPage]}
                  onValueChange={(v) => patchDefaultSettings({ wordsPerPage: v[0] ?? 3 })}
                />
              </Field>
              <Field label="Vertical position">
                <Slider
                  aria-label="Caption vertical position"
                  min={0.2}
                  max={0.9}
                  step={0.01}
                  value={[defaultClipSettings.captionY]}
                  onValueChange={(v) => patchDefaultSettings({ captionY: v[0] ?? 0.78 })}
                />
              </Field>
              <Toggle
                label="Hook title overlay (first 3.6s)"
                v={defaultClipSettings.showHook}
                on={(v) => patchDefaultSettings({ showHook: v })}
              />
              <Toggle
                label="Auto emoji overlays (🏆 💰 🛒 🏋️ 👻)"
                v={defaultClipSettings.showEmojis ?? true}
                on={(v) => patchDefaultSettings({ showEmojis: v })}
              />
              <Toggle
                label="Progress bar"
                v={defaultClipSettings.progressBar}
                on={(v) => patchDefaultSettings({ progressBar: v })}
              />
            </div>
          </div>
        </div>
      ) : (
        /* STUDIO RESULTS & EDITOR VIEW */
        <main className="grid gap-6 p-4 sm:p-6 xl:grid-cols-[300px_minmax(0,1fr)] 2xl:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="min-w-0 space-y-4">
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-3 flex items-center justify-between gap-2">
                <Button variant="outline" size="sm" onClick={() => setStudioView("setup")}>
                  <ArrowLeft className="size-4" />
                  Clip Setup
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setFile(null);
                    setStage("idle");
                  }}
                >
                  <RotateCcw className="size-4" />
                  New video
                </Button>
              </div>
              <video src={url} className="aspect-video w-full rounded-md bg-background" controls />
              <div className="mt-3 flex items-center justify-between font-mono text-xs text-muted-foreground">
                <span className="truncate pr-2">{file.name}</span>
                <span>{fmt(duration)}</span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-display font-bold">
                  Clips <span className="text-muted-foreground">({clips.length})</span>
                </h3>
                <Button variant="ghost" size="sm" onClick={addManual}>
                  <Plus className="size-4" />
                  Add
                </Button>
              </div>
              {clips.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setActiveId(c.id)}
                  className={`w-full rounded-lg border p-3 text-left transition-colors ${
                    c.id === activeId
                      ? "border-primary bg-accent"
                      : "border-border bg-card hover:border-muted-foreground"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="font-semibold leading-snug">{c.title}</p>
                    {c.score > 0 && (
                      <span className="flex shrink-0 items-center gap-1 rounded bg-background px-1.5 py-0.5 font-mono text-xs text-hot">
                        <Flame className="size-3" />
                        {Math.round(c.score)}
                      </span>
                    )}
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                    {fmt(c.start)} → {fmt(c.end)} · {Math.round(c.end - c.start)}s
                    {c.track ? " · Tracked" : ""}
                  </p>
                  <p className="mt-1.5 line-clamp-2 text-xs text-muted-foreground">{c.reason}</p>
                  {c.exportMeta && (
                    <span className="mt-2 inline-block rounded bg-primary px-1.5 py-0.5 font-mono text-[10px] text-primary-foreground">
                      {c.exportMeta.mode === "nvenc"
                        ? "NVENC MASTERED"
                        : c.exportMeta.mode === "ffmpeg_cpu"
                          ? "FFMPEG MASTERED"
                          : "BROWSER EXPORT"}
                    </span>
                  )}
                </button>
              ))}
            </div>

            <VyroWorkspace
              clip={active}
              clipWords={words.filter((w) => active && w.e > active.start && w.s < active.end)}
              exportSha256={active?.exportMeta?.sha256}
              onApplyBriefToGoal={setGoal}
            />
          </aside>

          <section className="min-w-0">
            {active ? (
              <Editor
                key={active.id}
                clip={active}
                src={url}
                file={file}
                words={words}
                duration={duration}
                onChange={(p) => update(active.id, p)}
                onWords={(updated) =>
                  setWords((current) =>
                    [
                      ...current.filter((w) => w.e <= active.start || w.s >= active.end),
                      ...updated,
                    ].sort((a, b) => a.s - b.s),
                  )
                }
              />
            ) : (
              <div className="grid h-full min-h-[500px] place-items-center rounded-lg border border-dashed border-border grid-lines text-center">
                <div className="max-w-sm space-y-2 p-6">
                  <Wand2 className="mx-auto size-8 text-primary" />
                  <p className="font-display text-lg font-bold">Select or create a clip</p>
                  <Button variant="outline" onClick={() => setStudioView("setup")}>
                    <Sliders className="size-4" />
                    Back to Clip Setup
                  </Button>
                </div>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {children}
    </div>
  );
}

function Pipeline({
  stage,
  progress,
  logs,
  onCancel,
}: {
  stage: Stage;
  progress: number;
  logs: string[];
  onCancel: () => void;
}) {
  const steps: [Stage, string][] = [
    ["transcribe", "1. Local Faster-Whisper transcription (16kHz WAV word timestamps)"],
    ["analyze", "2. Viral clip discovery & sentence boundary snapping"],
    ["tracking", "3. Hybrid Face/Person/Object tracking & gimbal smoothing"],
  ];
  const idx = steps.findIndex((s) => s[0] === stage);
  return (
    <div className="space-y-2 rounded-lg border border-border bg-surface p-3.5 font-mono text-xs">
      {steps.map(([k, l], i) => (
        <div
          key={k}
          className={`flex items-center gap-2 ${i < idx ? "text-primary" : i === idx ? "text-foreground" : "text-muted-foreground"}`}
        >
          {i === idx ? (
            <Loader2 className="size-3.5 animate-spin text-primary" />
          ) : (
            <span className="size-3.5 text-center">{i < idx ? "✓" : "·"}</span>
          )}
          {l}
          {(k === "transcribe" || k === "tracking") && i === idx && (
            <span className="ml-auto">{Math.round(progress * 100)}%</span>
          )}
        </div>
      ))}
      {(stage === "transcribe" || stage === "tracking") && <Progress value={progress * 100} />}
      {stage === "transcribe" && logs.length > 0 && (
        <div className="space-y-0.5 text-[10px] text-muted-foreground">
          {logs.map((ln, i) => (
            <p key={i} className="truncate">
              {ln}
            </p>
          ))}
        </div>
      )}
      {stage === "transcribe" && (
        <Button variant="ghost" size="sm" className="h-6 w-full text-[11px]" onClick={onCancel}>
          Cancel job
        </Button>
      )}
    </div>
  );
}

function Dropzone({ onFile }: { onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  return (
    <div className="mx-auto grid max-w-4xl gap-10 px-4 py-12 sm:px-6 sm:py-16">
      <div className="animate-fade-up space-y-4 text-center">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-primary">
          Automated Faster-Whisper · Hybrid Object Tracking · Local GPU Render
        </p>
        <h2 className="font-display text-4xl font-black leading-[1.05] tracking-tight sm:text-5xl md:text-7xl">
          One-click viral clips
          <br />
          on your own GPU.
        </h2>
        <p className="mx-auto max-w-xl text-muted-foreground">
          Upload any long video to open the ClipForge Setup page, customize your caption
          template and framing, and generate gimbal-tracked 1080×1920 shorts automatically.
        </p>
      </div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files[0];
          if (f) onFile(f);
        }}
        className={`group relative grid cursor-pointer place-items-center overflow-hidden rounded-xl border-2 border-dashed p-10 text-center transition-colors grid-lines animate-fade-up [animation-delay:80ms] sm:p-16 ${
          over ? "border-primary bg-accent" : "border-border bg-surface hover:border-primary"
        }`}
      >
        <input
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
        />
        <div
          className={`pointer-events-none absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-primary/10 to-transparent ${
            over ? "animate-scan" : "opacity-0 group-hover:animate-scan group-hover:opacity-100"
          }`}
        />
        <Upload className="mb-4 size-10 text-primary transition-transform duration-200 ease-out group-hover:-translate-y-1" />
        <p className="font-display text-xl font-bold">Drop a video or click to upload</p>
        <p className="mt-1 font-mono text-xs text-muted-foreground">
          MP4 · MOV · WEBM — opens Clip Setup screen immediately
        </p>
      </label>
      <div className="animate-fade-up [animation-delay:160ms]">
        <YouTubeImport onFile={onFile} />
      </div>
    </div>
  );
}

function Editor({
  clip,
  src,
  file,
  words,
  duration,
  onChange,
  onWords,
}: {
  clip: Clip;
  src: string;
  file: File | null;
  words: Word[];
  duration: number;
  onChange: (p: Partial<Clip>) => void;
  onWords: (words: Word[]) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [tracking, setTracking] = useState<number | null>(null);
  const [exporting, setExporting] = useState<number | null>(null);
  const [exportStageLabel, setExportStageLabel] = useState<string>("Rendering composition…");
  const [coDirectorPrompt, setCoDirectorPrompt] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);
  const s = clip.settings;
  const clipWords = useMemo(
    () => words.filter((w) => w.e > clip.start && w.s < clip.end),
    [words, clip.start, clip.end],
  );
  const job: ClipJob = useMemo(
    () => ({
      start: clip.start,
      end: clip.end,
      hook: clip.hook,
      words: clipWords,
      track: clip.track,
      cues: numberCues(clipWords, clip.editingCues),
      settings: s,
    }),
    [clip, clipWords, s],
  );
  const jobRef = useRef(job);
  jobRef.current = job;
  const setS = (p: Partial<ClipSettings>) => onChange({ settings: { ...s, ...p } });

  const hasWhisperWords = clipWords.some((w) => w.source === "whisper");

  useEffect(() => {
    const v = videoRef.current,
      c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const drawOnce = () => {
      if (v.readyState >= 2) drawFrame(ctx, v, v.currentTime, jobRef.current);
    };
    const onLoaded = () => {
      v.currentTime = jobRef.current.start;
      drawOnce();
    };
    v.addEventListener("loadeddata", onLoaded);
    v.addEventListener("seeked", drawOnce);
    if (v.readyState >= 2) drawOnce();
    return () => {
      v.removeEventListener("loadeddata", onLoaded);
      v.removeEventListener("seeked", drawOnce);
    };
  }, [job]);

  useEffect(() => {
    if (!playing) return;
    const v = videoRef.current,
      c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const loop = () => {
      const j = jobRef.current;
      if (v.currentTime >= j.end) {
        v.pause();
        v.currentTime = j.start;
        setPlaying(false);
        return;
      }
      if (v.readyState >= 2) drawFrame(ctx, v, v.currentTime, j);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    const v = videoRef.current;
    if (v && !playing) v.currentTime = clip.start;
  }, [clip.start]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (playing) {
      v.pause();
      setPlaying(false);
    } else {
      if (v.currentTime < clip.start || v.currentTime >= clip.end) v.currentTime = clip.start;
      v.play();
      setPlaying(true);
    }
  };

  async function runTrack() {
    setTracking(0);
    try {
      const tracked = await trackWithWorkerFallback(
        src,
        file,
        clip.start,
        clip.end,
        "auto",
        setTracking,
      );
      if (!tracked) throw new Error("tracking unavailable");
      onChange({
        track: tracked.track,
        speakers: tracked.speakers,
        settings: { ...s, layout: "auto", manualX: null, manualY: null },
      });
      toast.success("Hybrid Face + Person/Object gimbal track ready");
    } catch (e) {
      console.error(e);
      toast.error("Object & face tracking failed");
    }
    setTracking(null);
  }

  async function runExport() {
    videoRef.current?.pause();
    setPlaying(false);
    abortRef.current = new AbortController();
    setExporting(0);
    setExportStageLabel("Starting smooth export…");
    try {
      let activeJob = job;
      // Every layout except a manually pinned frame reads its crop centre off the track, so a clip
      // with no track used to export as a dead-centre crop whenever its layout was not "auto".
      const pinned =
        typeof activeJob.settings.manualX === "number" ||
        typeof activeJob.settings.manualY === "number";
      if (!activeJob.track && !pinned) {
        setExportStageLabel("Auto-tracking faces & objects before render…");
        const autoTrack = await trackWithWorkerFallback(
          src,
          file,
          clip.start,
          clip.end,
          "auto",
          (p) => setExporting(p * 0.25),
          abortRef.current.signal,
        );
        if (autoTrack) {
          onChange({ track: autoTrack.track, speakers: autoTrack.speakers });
          activeJob = { ...activeJob, track: autoTrack.track };
        }
      }
      // The renderer addresses the source by the id its cached copy is stored under, so an export
      // never has to re-upload 300 MB just to cut a clip.
      const sourceFile = file ?? filesByUrl.get(src) ?? null;
      const renderSourceId = sourceFile ? await sourceIdFor(sourceFile) : undefined;
      const res = await exportClip(
        src,
        activeJob,
        (p, label) => {
          setExporting(p);
          if (label) setExportStageLabel(label);
        },
        abortRef.current.signal,
        { sourceId: renderSourceId, file: sourceFile },
      );
      const u = URL.createObjectURL(res.blob);
      // Every export holds a 100 MB-class blob behind an object URL. Releasing the previous one is
      // what keeps the renderer's blob storage free enough to accept the next master.
      if (clip.exportUrl?.startsWith("blob:")) URL.revokeObjectURL(clip.exportUrl);
      onChange({
        exportUrl: u,
        exportMeta: {
          mode: res.mode,
          encoder: res.encoder,
          verified: res.verified,
          sha256: res.sha256,
          deliveredFps: res.deliveredFps,
        },
      });
      const a = document.createElement("a");
      a.href = u;
      a.download = `${clip.title.replace(/[^\w]+/g, "_").slice(0, 40)}.${res.blob.type.includes("mp4") ? "mp4" : "webm"}`;
      a.click();
      const rate = `${res.deliveredFps.toFixed(1)} fps delivered`;
      const modeText =
        res.mode === "nvenc"
          ? `Mastered via NVIDIA NVENC (${rate})`
          : res.mode === "ffmpeg_cpu"
            ? `Mastered via local FFmpeg (${rate})`
            : `Exported via browser stream (${rate})`;
      toast.success(modeText);
    } catch (e) {
      if ((e as Error).message !== "cancelled") {
        console.error(e);
        toast.error((e as Error).message || "Render failed");
      }
    }
    setExporting(null);
  }

  const shotSpans = useMemo(() => {
    const tr = clip.track;
    if (!tr || !tr.length) return [];
    const spans: {
      startIdx: number;
      endIdx: number;
      start: number;
      end: number;
      mode: FrameLayoutMode;
    }[] = [];
    let begin = 0;
    for (let i = 1; i <= tr.length; i++) {
      if (
        i === tr.length ||
        tr[i]!.cut ||
        (tr[i]!.mode ?? "fill") !== (tr[i - 1]!.mode ?? "fill")
      ) {
        spans.push({
          startIdx: begin,
          endIdx: i,
          start: tr[begin]!.t,
          end: tr[Math.min(i, tr.length - 1)]!.t,
          mode: (tr[begin]!.mode ?? "fill") as FrameLayoutMode,
        });
        begin = i;
      }
    }
    return spans;
  }, [clip.track]);

  function cycleShotLayout(startIdx: number, endIdx: number, currentMode: FrameLayoutMode) {
    if (!clip.track) return;
    const order: FrameLayoutMode[] = ["fill", "split", "grid3", "editorial", "fit"];
    const nextMode = order[(order.indexOf(currentMode) + 1) % order.length] ?? "fill";
    const nextTrack = clip.track.map((pt, idx) =>
      idx >= startIdx && idx < endIdx ? { ...pt, mode: nextMode } : pt,
    );
    onChange({ track: nextTrack, settings: { ...s, layout: "auto" } });
    toast.success(`Updated shot layout to ${nextMode.toUpperCase()}`);
  }

  function applyCoDirectorInstruction(instruction: string) {
    const q = instruction.toLowerCase();
    if (
      q.includes("outro") ||
      q.includes("question") ||
      q.includes("trim") ||
      q.includes("payoff")
    ) {
      const cleaned = trimClipNarrativeBoundaries(
        clip.start,
        clip.end,
        words,
        duration || clip.end,
      );
      onChange({ start: cleaned.start, end: cleaned.end });
      toast.success(`Trimmed clip boundaries to ${fmt(cleaned.start)} → ${fmt(cleaned.end)}`);
      return;
    }
    if (q.includes("3-up") || q.includes("grid3") || q.includes("three") || q.includes("group")) {
      if (clip.track?.length) {
        const nextTrack = clip.track.map((pt) =>
          pt.mode === "split" || pt.mode === "fit"
            ? { ...pt, mode: "grid3" as FrameLayoutMode }
            : pt,
        );
        onChange({ track: nextTrack, settings: { ...s, layout: "grid3" } });
      } else {
        setS({ layout: "grid3" });
      }
      toast.success("Applied 3-Up Split Grid layout");
      return;
    }
    if (
      q.includes("1-up") ||
      q.includes("single") ||
      q.includes("active speaker") ||
      q.includes("fill")
    ) {
      setS({ layout: "fill" });
      toast.success("Locked framing to 1-Up Active Speaker (9:16 Fill)");
      return;
    }
    if (q.includes("cinema") || q.includes("blur") || q.includes("fit")) {
      setS({ layout: "fit" });
      toast.success("Switched to Cinema-Card (Fit + Blur)");
      return;
    }
    if (q.includes("beast") || q.includes("bold")) {
      setS({ captionStyle: "beast" });
      toast.success("Switched captions to MrBeast Bold Pop");
      return;
    }
    if (q.includes("karaoke")) {
      setS({ captionStyle: "karaoke" });
      toast.success("Switched captions to Karaoke");
      return;
    }
    if (q.includes("opus")) {
      setS({ captionStyle: "opus" });
      toast.success("Switched captions to Opus Viral");
      return;
    }
    toast.info("Applied AI Co-Director preference to clip settings");
  }

  return (
    <div className="grid min-w-0 grid-cols-1 gap-6 lg:grid-cols-[minmax(240px,340px)_minmax(0,1fr)]">
      <div className="min-w-0 space-y-3 lg:sticky lg:top-20 lg:self-start">
        <div className="relative mx-auto aspect-[9/16] w-full max-w-[340px] overflow-hidden rounded-xl border border-border bg-background">
          <canvas ref={canvasRef} width={W} height={H} className="size-full" />
          <video ref={videoRef} src={src} className="hidden" playsInline crossOrigin="anonymous" />
          {exporting !== null && (
            <div className="absolute inset-0 grid place-items-center bg-background/80 p-6 text-center backdrop-blur">
              <div className="w-full space-y-3">
                <Loader2 className="mx-auto size-8 animate-spin text-primary" />
                <p className="font-display font-bold">
                  {exportStageLabel} {Math.round(exporting * 100)}%
                </p>
                <Progress value={exporting * 100} />
                <Button variant="ghost" size="sm" onClick={() => abortRef.current?.abort()}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
        <div className="mx-auto flex max-w-[340px] gap-2">
          <Button variant="secondary" className="flex-1" onClick={toggle}>
            {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
            {playing ? "Pause" : "Preview"}
          </Button>
          <Button className="flex-1" onClick={runExport} disabled={exporting !== null}>
            <Download className="size-4" />
            Export MP4
          </Button>
        </div>
        {clip.exportMeta && (
          <div className="mx-auto max-w-[340px] rounded-md border border-border bg-surface p-2.5 font-mono text-[11px]">
            <p className="flex items-center gap-1.5 font-semibold text-primary">
              <CheckCircle2 className="size-3.5" />
              {clip.exportMeta.mode === "nvenc"
                ? "Local NVIDIA NVENC Master"
                : clip.exportMeta.mode === "ffmpeg_cpu"
                  ? "Local FFmpeg CPU Master"
                  : "Browser Stream Export"}
            </p>
            <p className="mt-0.5 text-muted-foreground">
              Encoder: {clip.exportMeta.encoder} · {clip.exportMeta.deliveredFps.toFixed(1)} fps ·{" "}
              {clip.exportMeta.verified ? "ffprobe verified" : "Browser stream"}
            </p>
            {clip.exportUrl && (
              <a
                href={clip.exportUrl}
                download
                className="mt-1 inline-block text-primary underline"
              >
                Download last export
              </a>
            )}
          </div>
        )}
      </div>

      <div className="min-w-0 max-w-3xl space-y-4">
        <div className="space-y-3 rounded-lg border border-border bg-card p-4">
          <input
            value={clip.title}
            aria-label="Clip title"
            onChange={(e) => onChange({ title: e.target.value })}
            className="-mx-2 w-full rounded-md bg-transparent px-2 py-1 font-display text-2xl font-extrabold outline-none transition-colors hover:bg-surface focus-visible:bg-surface"
          />
          <Field label="HOOK TEXT (SHOWN FIRST 3.6S)">
            <input
              value={clip.hook}
              aria-label="Hook text"
              onChange={(e) => onChange({ hook: e.target.value })}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-base sm:text-sm"
            />
          </Field>
          <Field
            label={`TRIM · ${fmt(clip.start)} → ${fmt(clip.end)} (${Math.round(clip.end - clip.start)}S)`}
          >
            <Slider
              aria-label="Clip trim"
              min={0}
              max={Math.max(1, duration)}
              step={0.1}
              value={[clip.start, clip.end]}
              onValueChange={(v) => {
                const start = v[0] ?? 0,
                  end = v[1] ?? duration;
                // Track points are stamped in absolute source time, so a trim does not invalidate
                // them. Dropping them here left every non-auto clip cropping blind dead centre.
                onChange({
                  start,
                  end: Math.min(duration, Math.max(start + 0.1, end)),
                });
              }}
            />
          </Field>
          {clip.hashtags.length > 0 && (
            <p className="font-mono text-xs text-primary">
              {clip.hashtags.map((h) => (h.startsWith("#") ? h : `#${h}`)).join(" ")}
            </p>
          )}
        </div>

        {/* AI Co-Director Natural-Language Edit Assistant & Per-Shot Timeline Strip */}
        <div className="space-y-3 rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 font-display font-bold">
              <Sparkles className="size-4 text-primary" />
              AI Co-Director & Shot Timeline
            </h3>
            <span className="font-mono text-[10px] text-muted-foreground">
              10 Editing Rules Active
            </span>
          </div>
          <div className="flex gap-2">
            <input
              value={coDirectorPrompt}
              aria-label="Co-Director instruction"
              onChange={(e) => setCoDirectorPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && coDirectorPrompt.trim()) {
                  applyCoDirectorInstruction(coDirectorPrompt);
                  setCoDirectorPrompt("");
                }
              }}
              placeholder="e.g. 'Use 3-Up Grid on multi-person shots', 'Trim outro & questions', 'Bold pop captions'…"
              className="min-w-0 flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-base sm:text-xs"
            />
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                if (!coDirectorPrompt.trim()) return;
                applyCoDirectorInstruction(coDirectorPrompt);
                setCoDirectorPrompt("");
              }}
            >
              Apply Edit
            </Button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {[
              "Trim outro & trailing questions",
              "Use 3-Up Grid on group shots",
              "Lock 1-Up on active speaker",
              "Switch to Cinema-Card Fit+Blur",
              "MrBeast Bold Pop captions",
            ].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => applyCoDirectorInstruction(preset)}
                className="rounded-full border border-border bg-surface px-2.5 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
              >
                + {preset}
              </button>
            ))}
          </div>
          {shotSpans.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Detected shots ({shotSpans.length}) · click to cycle layout
              </p>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {shotSpans.map((sh, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = sh.start;
                      }
                      cycleShotLayout(sh.startIdx, sh.endIdx, sh.mode);
                    }}
                    className="flex shrink-0 flex-col rounded border border-border bg-surface px-2.5 py-1.5 text-left font-mono text-[10px] transition-colors hover:border-primary"
                  >
                    <span className="font-bold text-primary">
                      Shot #{idx + 1} · {sh.mode.toUpperCase()}
                    </span>
                    <span className="text-muted-foreground">
                      {fmt(sh.start)} → {fmt(sh.end)}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4 rounded-lg border border-border bg-card p-4">
          <h3 className="font-display font-bold">Framing & Render Settings</h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {LAYOUTS.map(([l, label]) => (
              <Button
                key={l}
                variant={s.layout === l ? "default" : "outline"}
                className="px-2 text-xs"
                onClick={() => setS({ layout: l })}
              >
                {label}
              </Button>
            ))}
          </div>
          <Button
            variant="outline"
            className="w-full"
            onClick={runTrack}
            disabled={tracking !== null}
          >
            {tracking !== null ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Tracking faces & objects {Math.round(tracking * 100)}%
              </>
            ) : (
              <>
                <ScanFace className="size-4" />
                {clip.track
                  ? "Re-run hybrid object & face tracker"
                  : "Auto-reframe (scene cuts + faces + objects)"}
              </>
            )}
          </Button>
          {clip.speakers?.main_lock && (
            <p className="font-mono text-[11px] text-muted-foreground">
              HOST FRAMED{" "}
              <span className="text-primary">
                {clip.speakers.main_lock.on_screen}/{clip.speakers.main_lock.required_samples}
              </span>{" "}
              · OVERLAP {clip.speakers.overlap_windows ?? 0} · UNREAD{" "}
              {clip.speakers.ambiguous_windows ?? 0}
            </p>
          )}
          <Field
            label={`MANUAL HORIZONTAL CROP OVERRIDE · ${typeof s.manualX === "number" ? `${Math.round(s.manualX * 100)}%` : "AUTO TRACKED"}`}
          >
            <div className="flex items-center gap-2">
              <Slider
                aria-label="Manual horizontal crop"
                min={0.15}
                max={0.85}
                step={0.01}
                value={[s.manualX ?? 0.5]}
                onValueChange={(v) => setS({ manualX: v[0] ?? 0.5 })}
              />
              {typeof s.manualX === "number" && (
                <Button variant="ghost" size="sm" onClick={() => setS({ manualX: null })}>
                  Reset
                </Button>
              )}
            </div>
          </Field>
          <div className="grid grid-cols-2 gap-2">
            <Field label="TARGET FRAME RATE">
              <select
                value={s.fps ?? "29.97"}
                aria-label="Target frame rate"
                onChange={(e) => setS({ fps: e.target.value as "29.97" | "30" })}
                className="w-full rounded-md border border-input bg-background px-2.5 py-1.5 font-mono text-base sm:text-xs"
              >
                <option value="29.97">29.97 fps</option>
                <option value="30">30.00 fps</option>
              </select>
            </Field>
            <Field label="CAPTION TIMING SOURCE">
              <div className="rounded-md border border-border bg-surface px-2.5 py-1.5 font-mono text-[11px] text-primary">
                {hasWhisperWords
                  ? "✓ Faster-Whisper"
                  : clipWords.length
                    ? "Imported / Manual"
                    : "Awaiting Whisper"}
              </div>
            </Field>
          </div>
          <Toggle label="Animations" v={s.animations ?? true} on={(v) => setS({ animations: v })} />
          <Toggle label="Effects" v={s.effects ?? true} on={(v) => setS({ effects: v })} />
          <Toggle
            label="Zoom punch on emphasis"
            v={s.zoomPunch}
            on={(v) => setS({ zoomPunch: v })}
          />
          <Toggle
            label="Voice loudness & EBU R128 limiter"
            v={s.voiceEnhance ?? true}
            on={(v) => setS({ voiceEnhance: v })}
          />
        </div>

        <div className="space-y-4 rounded-lg border border-border bg-card p-4">
          <CaptionPanel
            settings={s}
            setSettings={setS}
            words={clipWords}
            start={clip.start}
            end={clip.end}
            title={clip.title}
            onWords={onWords}
          />
          <Field label="Highlight colour">
            <div className="flex gap-2">
              {SWATCHES.map((c) => (
                <button
                  key={c}
                  aria-label={`Colour ${c}`}
                  onClick={() => setS({ highlight: c })}
                  className={`size-8 rounded-full border-2 ${s.highlight === c ? "border-foreground" : "border-transparent"}`}
                  style={{ background: c }}
                />
              ))}
            </div>
          </Field>
          <Field label={`Words per page · ${s.wordsPerPage}`}>
            <Slider
              aria-label="Words per page"
              min={1}
              max={6}
              step={1}
              value={[s.wordsPerPage]}
              onValueChange={(v) => setS({ wordsPerPage: v[0] ?? 3 })}
            />
          </Field>
          <Field label="Vertical position">
            <Slider
              aria-label="Caption vertical position"
              min={0.2}
              max={0.9}
              step={0.01}
              value={[s.captionY]}
              onValueChange={(v) => setS({ captionY: v[0] ?? 0.78 })}
            />
          </Field>
          <Toggle label="Hook title overlay" v={s.showHook} on={(v) => setS({ showHook: v })} />
          <Toggle
            label="Auto emoji overlays (🏆 💰 🛒 🏋️ 👻)"
            v={s.showEmojis ?? true}
            on={(v) => setS({ showEmojis: v })}
          />
          <Toggle label="Progress bar" v={s.progressBar} on={(v) => setS({ progressBar: v })} />
          <Field label="Lower-third name">
            <input
              value={s.brandLabel ?? ""}
              aria-label="Lower-third name"
              onChange={(e) => setS({ brandLabel: e.target.value.slice(0, 42) })}
              placeholder="Speaker or brand"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-base sm:text-sm"
            />
          </Field>
        </div>
      </div>
    </div>
  );
}

function Toggle({ label, v, on }: { label: string; v: boolean; on: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between text-sm">
      <span>{label}</span>
      <Switch checked={v} onCheckedChange={on} />
    </label>
  );
}
