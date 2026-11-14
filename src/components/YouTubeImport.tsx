import { useEffect, useState } from "react";
import { Youtube, Loader2, Download, CircleCheck, CircleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

export const HELPER = "http://127.0.0.1:8765";

export function getHelperToken(): string {
  return "auto-local";
}

export function YouTubeImport({ onFile }: { onFile: (f: File) => void }) {
  const [online, setOnline] = useState<boolean | null>(null);
  const [encoder, setEncoder] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [quality, setQuality] = useState(1080);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const check = async () => {
    try {
      const r = await fetch(`${HELPER}/health`, { signal: AbortSignal.timeout(1500) });
      if (r.ok) {
        const data = (await r.json()) as { ok?: boolean; encoder?: string | null };
        setOnline(Boolean(data.ok));
        setEncoder(data.encoder ?? null);
      } else {
        setOnline(false);
      }
    } catch {
      setOnline(false);
    }
  };
  useEffect(() => {
    void check();
    const i = setInterval(check, 4000);
    return () => clearInterval(i);
  }, []);

  async function start() {
    setError("");
    setProgress(0);
    setBusy("Starting download…");
    try {
      const r = await fetch(`${HELPER}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, quality, rightsConfirmed: true }),
      });
      const { id, error: e } = await r.json();
      if (!id) throw new Error(e ?? "Could not start download");
      for (;;) {
        await new Promise((res) => setTimeout(res, 900));
        const s = await (await fetch(`${HELPER}/status?id=${id}`)).json();
        setProgress(s.progress ?? 0);
        setBusy(s.title ? `Downloading “${s.title}”` : "Downloading…");
        if (s.status === "error") throw new Error(s.error || "Download failed");
        if (s.status === "done") {
          setBusy("Loading into studio…");
          const blob = await (await fetch(`${HELPER}/file?id=${id}`)).blob();
          onFile(new File([blob], `${(s.title || "youtube").replace(/[^\w ]+/g, "").slice(0, 60)}.mp4`, { type: "video/mp4" }));
          return;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2 font-display font-bold">
          <Youtube className="size-5 text-hot" />
          Import from YouTube link
        </p>
        <span
          className={`flex items-center gap-1.5 font-mono text-[11px] ${online ? "text-primary" : "text-muted-foreground"}`}
          title={online ? undefined : "Start the local worker for YouTube import & NVENC export"}
        >
          {online ? <CircleCheck className="size-3.5" /> : <CircleAlert className="size-3.5" />}
          {online === null
            ? "Connecting…"
            : online
              ? `Auto-connected (${encoder ?? "FFmpeg ready"})`
              : "Worker offline"}
        </span>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          value={url}
          aria-label="YouTube URL"
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://www.youtube.com/watch?v=…"
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-base sm:text-sm"
        />
        <select
          value={quality}
          aria-label="Download quality"
          onChange={(e) => setQuality(Number(e.target.value))}
          className="rounded-md border border-input bg-background px-2 py-2 font-mono text-sm"
        >
          {[2160, 1440, 1080, 720, 480].map((q) => (
            <option key={q} value={q}>
              {q}p
            </option>
          ))}
        </select>
        <Button onClick={start} disabled={!online || !/^https?:\/\//.test(url) || !!busy}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}Import
        </Button>
      </div>

      {busy && (
        <div className="space-y-1.5">
          <p className="truncate font-mono text-xs text-muted-foreground">
            {busy} {Math.round(progress * 100)}%
          </p>
          <Progress value={progress * 100} />
        </div>
      )}
      {error && <p className="font-mono text-xs text-destructive">{error}</p>}
    </div>
  );
}
