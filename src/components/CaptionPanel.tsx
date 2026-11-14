import { useEffect, useRef, useState } from "react";
import { Download, Upload, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { CAPTION_PRESETS, captionPages, parseSubtitles, serializeSubtitles, type CaptionStyle } from "@/lib/subtitles";
import { drawCaption, W, H, type ClipSettings } from "@/lib/render";
import type { Word } from "@/lib/media";

function StylePreview({ style, settings }: { style: CaptionStyle; settings: ClipSettings }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current, ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const sample = CAPTION_PRESETS.find((p) => p.id === style)?.sample ?? "";
    const words = sample.split(" ").map((w, i) => ({ w, s: i * 0.5, e: (i + 1) * 0.5 }));
    const reduced = settings.animations === false || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    const start = performance.now();
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      const scale = canvas.width / W;
      ctx.translate(0, canvas.height / 2 - H * scale / 2);
      ctx.scale(scale, scale);
      drawCaption(ctx, words, reduced ? 0.65 : ((performance.now() - start) / 1000) % (words.length * 0.5 + 0.6), { ...settings, captionStyle: style, captionY: 0.5, fontSize: 96 });
      ctx.restore();
      if (!reduced) frame = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(frame);
  }, [style, settings.highlight, settings.animationStrength, settings.animations]);
  return <canvas ref={ref} width={432} height={192} className="aspect-[9/4] w-full" aria-hidden="true" />;
}

export function CaptionPanel({ settings, setSettings, words, start, end, title, onWords }: {
  settings: ClipSettings; setSettings: (patch: Partial<ClipSettings>) => void;
  words: Word[]; start: number; end: number; title: string; onWords: (words: Word[]) => void;
}) {
  const [editing, setEditing] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pages = captionPages(words, settings.wordsPerPage);
  function download(format: "srt" | "vtt") {
    const url = URL.createObjectURL(new Blob([serializeSubtitles(words, start, end, settings.wordsPerPage, format)], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url; a.download = `${title.replace(/[^\p{L}\p{N}]+/gu, "_") || "subtitles"}.${format}`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function importFile(file: File) {
    try {
      const parsed = parseSubtitles(await file.text(), start).filter((w) => w.s < end && w.e > start);
      if (!parsed.length) throw new Error("No timed subtitles found within this clip. Choose an SRT or VTT file timed from the beginning of this clip.");
      onWords(parsed); toast.success(`Imported ${parsed.length} subtitle words`);
    } catch (error) { toast.error(error instanceof Error ? error.message : "Subtitle import failed"); }
  }
  return <div className="space-y-4">
    <div className="flex items-center justify-between gap-2">
      <h3 className="font-display font-bold">Subtitle studio</h3>
      <span className="font-mono text-[10px] text-primary">ClipForge Caption Engine</span>
    </div>
    <div className="grid grid-cols-2 gap-2" aria-label="Subtitle styles">
      {CAPTION_PRESETS.map((preset) => <Button key={preset.id} variant="outline" aria-pressed={settings.captionStyle === preset.id} onClick={() => setSettings({ captionStyle: preset.id })} className={`h-auto min-w-0 flex-col gap-0 overflow-hidden p-0 ${settings.captionStyle === preset.id ? "border-primary bg-accent" : "bg-background"}`}>
        <StylePreview style={preset.id} settings={settings} />
        <span className="pb-2 text-xs">{preset.name}</span>
      </Button>)}
    </div>
    <label className="block space-y-2 text-xs text-muted-foreground">Text size · {settings.fontSize ?? 84}px
      <Slider aria-label="Subtitle text size" min={40} max={110} step={2} value={[settings.fontSize ?? 84]} onValueChange={(v) => setSettings({ fontSize: v[0] ?? 84 })} />
    </label>
    <label className="block space-y-2 text-xs text-muted-foreground">Motion · {Math.round((settings.animationStrength ?? 1) * 100)}%
      <Slider aria-label="Subtitle motion" disabled={settings.animations === false} min={0} max={1.5} step={0.1} value={[settings.animationStrength ?? 1]} onValueChange={(v) => setSettings({ animationStrength: v[0] ?? 1 })} />
    </label>
    <div className="flex flex-wrap gap-2">
      <Button variant="outline" size="sm" onClick={() => input.current?.click()}><Upload className="size-3.5" />Import subtitles</Button>
      <Button variant="outline" size="sm" disabled={!words.length} onClick={() => download("srt")}><Download className="size-3.5" />SRT</Button>
      <Button variant="outline" size="sm" disabled={!words.length} onClick={() => download("vtt")}><Download className="size-3.5" />VTT</Button>
      <input ref={input} type="file" accept=".srt,.vtt" className="hidden" aria-label="Import subtitle file" onChange={(e) => { const file = e.target.files?.[0]; if (file) void importFile(file); e.target.value = ""; }} />
    </div>
    {!!words.length && <>
      <Button variant="ghost" size="sm" className="w-full justify-between" onClick={() => setEditing(!editing)}>Edit transcript · {words.length} words {editing ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}</Button>
      {editing && <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
        {pages.map((page, pageIndex) => <div key={pageIndex} className="space-y-1 border-b border-border pb-3">
          <p className="font-mono text-[10px] text-muted-foreground">{((page[0]?.s ?? start) - start).toFixed(2)}s</p>
          <div className="flex flex-wrap gap-1.5">{page.map((word) => {
            const index = words.indexOf(word);
            return <input key={index} aria-label={`Subtitle word ${index + 1}`} value={word.w} onChange={(e) => onWords(words.map((w, i) => i === index ? { ...w, w: e.target.value } : w))} className="min-w-0 max-w-full rounded border border-input bg-background px-2 py-1 text-sm" size={Math.max(3, word.w.length + 1)} />;
          })}</div>
        </div>)}
      </div>}
    </>}
  </div>;
}