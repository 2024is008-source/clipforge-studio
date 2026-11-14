import { useState } from "react";
import { ShieldCheck, AlertTriangle, XCircle, Sparkles, Loader2, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { parseCampaignBrief, type CampaignRules, type AIClip } from "@/lib/ai.functions";
import type { Word } from "@/lib/media";

export type ComplianceCheckItem = {
  rule: string;
  status: "PASS" | "WARNING" | "FAIL";
  detail: string;
};

export function evaluateCampaignCompliance(
  rules: CampaignRules,
  clip: AIClip,
  clipWords: Word[],
  exportSha256?: string
): ComplianceCheckItem[] {
  const checks: ComplianceCheckItem[] = [];
  const dur = clip.end - clip.start;
  const transcriptText = clipWords.map((w) => w.w).join(" ").toLowerCase();
  const tagSet = new Set(clip.hashtags.map((h) => (h.startsWith("#") ? h.toLowerCase() : `#${h.toLowerCase()}`)));

  // 1. Duration check
  if (dur >= rules.minDurationSec && dur <= rules.maxDurationSec) {
    checks.push({
      rule: "Clip duration",
      status: "PASS",
      detail: `${dur.toFixed(1)}s is within required ${rules.minDurationSec}–${rules.maxDurationSec}s window.`,
    });
  } else {
    checks.push({
      rule: "Clip duration",
      status: "FAIL",
      detail: `${dur.toFixed(1)}s is outside required ${rules.minDurationSec}–${rules.maxDurationSec}s window.`,
    });
  }

  // 2. Required hashtags & disclosure
  const missingTags = rules.requiredHashtags.filter((t) => {
    const norm = t.startsWith("#") ? t.toLowerCase() : `#${t.toLowerCase()}`;
    return !tagSet.has(norm);
  });
  if (missingTags.length === 0) {
    checks.push({
      rule: "Required hashtags",
      status: "PASS",
      detail: rules.requiredHashtags.length ? `All required tags present (${rules.requiredHashtags.join(", ")}).` : "No mandatory tags missing.",
    });
  } else {
    checks.push({
      rule: "Required hashtags",
      status: "WARNING",
      detail: `Add required campaign tags before native posting: ${missingTags.join(", ")}`,
    });
  }

  // 3. Prohibited phrases check
  const violated = rules.prohibitedPhrases.filter(
    (p) => p.trim() && (transcriptText.includes(p.toLowerCase()) || clip.hook.toLowerCase().includes(p.toLowerCase()))
  );
  if (violated.length === 0) {
    checks.push({
      rule: "Prohibited claims / phrases",
      status: "PASS",
      detail: "No prohibited campaign phrases detected in spoken transcript or hook.",
    });
  } else {
    checks.push({
      rule: "Prohibited claims / phrases",
      status: "FAIL",
      detail: `Contains restricted phrase(s): ${violated.join(", ")}`,
    });
  }

  // 4. Post-match source file checksum
  checks.push({
    rule: "Post-match export checksum",
    status: exportSha256 ? "PASS" : "WARNING",
    detail: exportSha256
      ? `SHA-256 recorded (${exportSha256.slice(0, 14)}…) for Vyro source-file submission.`
      : "Export the final MP4 first so its exact SHA-256 checksum is locked for post-match verification.",
  });

  return checks;
}

export function VyroWorkspace({
  clip,
  clipWords,
  exportSha256,
  onApplyBriefToGoal,
}: {
  clip: AIClip | null;
  clipWords: Word[];
  exportSha256?: string | undefined;
  onApplyBriefToGoal: (brief: string) => void;
}) {
  const [campaignUrl, setCampaignUrl] = useState("");
  const [briefText, setBriefText] = useState("");
  const [parsing, setParsing] = useState(false);
  const [rules, setRules] = useState<CampaignRules | null>(null);
  const [livePostUrl, setLivePostUrl] = useState("");
  const [contentStatus, setContentStatus] = useState<"not_submitted" | "pending_review" | "approved" | "revision_required">("not_submitted");
  const [postMatchStatus, setPostMatchStatus] = useState<"unverified" | "pending_match" | "matched">("unverified");

  async function handleParse() {
    if (briefText.trim().length < 10) {
      toast.error("Paste at least one sentence from the campaign brief.");
      return;
    }
    setParsing(true);
    try {
      const extracted = await parseCampaignBrief({ data: { briefText } });
      setRules(extracted);
      onApplyBriefToGoal(briefText);
      toast.success(`Parsed rules for “${extracted.campaignName}”`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to parse brief");
    } finally {
      setParsing(false);
    }
  }

  const checks = rules && clip ? evaluateCampaignCompliance(rules, clip, clipWords, exportSha256) : [];

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-display font-bold">Vyro Campaign Workspace</h3>
        <ShieldCheck className="size-5 text-primary" />
      </div>

      <input
        value={campaignUrl}
        aria-label="Campaign reference URL"
        onChange={(e) => setCampaignUrl(e.target.value)}
        placeholder="Campaign reference URL (manual record)"
        className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-base sm:text-xs"
      />
      <Textarea
        value={briefText}
        onChange={(e) => setBriefText(e.target.value)}
        placeholder="Paste authorized Vyro campaign brief (required duration, hashtags, disclosure, prohibited claims)…"
        rows={3}
      />
      <Button variant="outline" size="sm" className="w-full" onClick={handleParse} disabled={parsing}>
        {parsing ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
        Parse brief
      </Button>

      {rules && (
        <div className="space-y-3 rounded-md border border-border bg-surface p-3 text-xs">
          <div className="flex items-center justify-between font-semibold">
            <span>{rules.campaignName}</span>
            <span className="font-mono text-[10px] text-primary">
              {rules.minDurationSec}–{rules.maxDurationSec}s · {rules.requiredDisclosure}
            </span>
          </div>
          {rules.requiredHashtags.length > 0 && (
            <p className="font-mono text-[11px] text-muted-foreground">Required tags: {rules.requiredHashtags.join(" ")}</p>
          )}
          <p className="text-[10px] text-muted-foreground">{rules.rightsNotice}</p>

          {clip && (
            <div className="space-y-1.5 border-t border-border pt-2">
              <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Preflight Advisory (Not official Vyro approval)
              </p>
              {checks.map((c) => (
                <div key={c.rule} className="flex items-start gap-2 rounded bg-background p-2">
                  {c.status === "PASS" ? (
                    <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-primary" />
                  ) : c.status === "WARNING" ? (
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-yellow-400" />
                  ) : (
                    <XCircle className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                  )}
                  <div>
                    <p className="font-semibold">
                      {c.rule} · <span className="font-mono text-[10px]">{c.status}</span>
                    </p>
                    <p className="text-[11px] text-muted-foreground">{c.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="space-y-2 border-t border-border pt-2">
            <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Native Post & Submission Log
            </p>
            <input
              value={livePostUrl}
              aria-label="Live post URL"
              onChange={(e) => setLivePostUrl(e.target.value)}
              placeholder="Paste live native post URL (TikTok / Reels / Shorts)"
              className="w-full rounded border border-input bg-background px-2.5 py-1.5 text-base sm:text-xs"
            />
            <div className="grid grid-cols-2 gap-2">
              <select
                value={contentStatus}
                onChange={(e) => setContentStatus(e.target.value as typeof contentStatus)}
                className="rounded border border-input bg-background px-2 py-1 font-mono text-[11px]"
              >
                <option value="not_submitted">Content: Not submitted</option>
                <option value="pending_review">Content: Pending review</option>
                <option value="approved">Content: Approved</option>
                <option value="revision_required">Content: Revision required</option>
              </select>
              <select
                value={postMatchStatus}
                onChange={(e) => setPostMatchStatus(e.target.value as typeof postMatchStatus)}
                className="rounded border border-input bg-background px-2 py-1 font-mono text-[11px]"
              >
                <option value="unverified">Post-Match: Unverified</option>
                <option value="pending_match">Post-Match: Pending</option>
                <option value="matched">Post-Match: Matched</option>
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
