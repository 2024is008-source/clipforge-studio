# ClipForge Studio — Repository Comparison & Integration Architecture

Pre-implementation document. Nothing in this file has been coded yet.
Scope: the 10 referenced open-source projects, the two generated quality reports, and where each
recommendation should land in the existing ClipForge codebase.

Ground rules this document is written against:

- ClipForge is **not** rebuilt. Heavy video work stays in the browser on the user's GPU
  (`src/lib/media|reframe|render.ts`); server functions only make AI calls (AGENTS.md).
- The local Python worker (`local_worker/`) is the accurate GPU tracker/ASR side-car and stays.
- Preserved as-is: the Opus caption preset, dynamic layouts, scene-cut detection, H.264 High
  1080×1920 / 30000-1001 CFR / BT.709 / AAC 48k export profile, local-only processing.
- The "duplicated / competing caption text layers" finding from the DOCX report is **out of scope**
  by explicit instruction.

### Verdict at a glance

| Repo | Verdict | One-line reason |
|---|---|---|
| ClipForge Reframe Engine v2 | **PORT** (MIT; constants + 4 unported mechanisms) | We already ported ~80% of it; the missing parts fix T2, T6, T7, T8 directly. |
| SYSTRAN/faster-whisper | **ADOPTED — tune** | Already running; T6 fragmentation is structural in its tokenizer and needs our own merge pass. |
| Breakthrough/PySceneDetect | **PORT the detector maths** (BSD-3) | Its adaptive rolling-mean ratio is the proven answer to missed indoor cuts (T10). |
| FFmpeg | **ADOPTED — extend post-passes** | Cinema-card and 2-up already exist in our worker; lanczos/unsharp, grade, loudnorm, dead-air are cheap adds. |
| google-ai-edge/mediapipe | **ADOPT selectively** (Apache-2.0) | FaceLandmarker (3.76 MB) gives the eye/mouth boxes the safe-zone work needs; segmenter/pose are opt-in. |
| remotion-dev/remotion | **SKIP package, PORT the ideas** | Node/Rust compositor contradicts browser-only rendering; its spring/interpolate/transition math is worth rewriting from scratch. |
| openvideodev/react-video-editor | **REFERENCE only** | Best data model we've seen (µs timebase, display/trim split, snapping) but an unresolved licence contradiction. |
| remotion-dev/skills | **REFERENCE / doc format** | `pageBreakAfter` caption contract and the router+topic skill-pack layout; no licence file, copy nothing. |
| tryvinci/vinci-clips | **REFERENCE only** (AGPL-3.0) | Ideas only; strictly weaker than our reframe/audio stack, and AGPL bars code reuse. |
| pyannote/pyannote-audio | **DEFER** | Licence is fine (CC-BY-4.0, 33.7 MB); there is no torchaudio cp314 CUDA wheel for our index — needs a separate py3.12 sidecar. |

Ship order: **P0.0 → P0.1 → P0.2 → P0.3 → P1.1 → P1.2 → P1.3 → P2.** §5 lists the eight decisions I
need from you before writing code.

---

## 1. Report claims verified against our actual code

Every claim below was checked by reading the source, not by watching video. `TRUE` means the code
does what the report says. `FALSE` means our own rendered frames contradict it.

### 1.1 TRUE — fix these

| # | Claim | Evidence in code |
|---|---|---|
| T1 | Split panels crop ~63% of source width each, so two halves can overlap by ~510 px | `drawCroppedSubject` uses `targetAspect = 1080/960 = 1.125`; at `zoom = 1` → `cropW = 1215` of 1920 — [src/lib/render.ts:110](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/render.ts#L110) |
| T2 | No minimum layout-hold gate exists | `MIN_SCENE_SECONDS = 2.5` is **declared but never referenced** — [local_worker/openshorts_engine.py:47](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/openshorts_engine.py#L47). No `MIN_LAYOUT_HOLD_SEC` / `MIN_SPLIT_SEPARATION` / `splitZoom` anywhere in `src/` or `local_worker/`. |
| T3 | Caption paging is fixed word count, no phrase/pause grouping, no orphan-line rule | `captionPages()` closes a page on `page.length >= count` or `/[.!?]$/` only — [src/lib/subtitles.ts:38](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/subtitles.ts#L38) |
| T4 | Emoji regex fires 🎮 on "store", "entire", "attempting" | `[/\b(game\|challenge\|attempting\|level\|player\|store\|entire)/i, "🎮"]` — [src/lib/subtitles.ts:19](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/subtitles.ts#L19) |
| T5 | Active-word pop is drawn into a layout computed at scale 1 → words can touch | Line widths measured at `ctx.font = font()` (scale 1), then each word is scaled about its own centre with no extra advance — [src/lib/render.ts:455-495](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/render.ts#L455) |
| T6 | Whisper continuation fragments are never merged: the leading space that marks a fragment is stripped first, and upstream fragmentation is structural | `text = str(...).strip()` before any token classification; `normalize_token()` also strips — [local_worker/transcript_validator.py:51](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/transcript_validator.py#L51), :27. Root cause is in faster-whisper itself (§2.6): BPE emits `.`/`,` as standalone tokens, so `2.4` arrives as `" 2"|"."|"4"` and `merge_punctuations` never rejoins glued fragments. |
| T7 | Discovery window steps by 70% of target length → clips overlap | `const nextTime = clipStart + Math.max(8, targetLen * 0.7)` — [src/lib/local-worker.ts:489](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/local-worker.ts#L489). Confirmed in our own job set: clip 1 = 0–33.8 s, clip 2 = 27.86–60.9 s. |
| T8 | Hook text is the first 8 transcript words, verbatim | `slice.slice(0, 8).map(w => w.w).join(" ")` → `cleanHook` → `title`/`hook` — [src/lib/local-worker.ts:435-464](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/local-worker.ts#L435) |
| T9 | Hook card is placed in the top band with no face test → it can sit on the eyes | Hook drawn at a fixed `top = 210` rounded plate, no face/person input — [src/lib/render.ts:299 area](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/render.ts#L299). Reproduced in our own export frames for the first ~3.6 s. |
| T10 | Indoor scene cuts need *both* colour-histogram and luma deltas simultaneously | `return histDiff >= 0.35 && lumaDiff >= 0.15` (report said 0.36/0.16 — same AND gate, values differ slightly) — [src/lib/reframe.ts:144](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/reframe.ts#L144) |
| T11 | `progressBar` defaults to on | `progressBar: true` in `defaultClipSettings` — [src/components/Studio.tsx:66](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/components/Studio.tsx#L66) |
| T12 | Whisper gets no domain prompt by default → homophones ("loaves"→"looks") | `initialPrompt: ""` → `initial_prompt or None` in the model call — [src/lib/local-worker.ts:37](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/local-worker.ts#L37), [local_worker/whisper_engine.py:211](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/whisper_engine.py#L211). A UI field exists, it is just empty. |
| T13 | **Our own bug, found while checking the reports:** the active-speaker gate is dead code, and our status output claims it is running | `mouth_region`, `normalise_activity`, `attribute_windows` and `hold_speaker_verdicts` are fully implemented and unit-tested ([local_worker/openshorts_engine.py:646-745](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/openshorts_engine.py#L646), [tests:157-164](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/tests/test_openshorts_and_gemini_pool.py#L157)) but **no production call site exists** — the only references are the test file. Meanwhile `main.py:189` and `object_tracker.py:58,66` advertise `split_layout_active_speaker_gate` and "8kHz audio RMS gate" as active. Split decisions today rest on face coexistence alone. Fix by wiring the gate in, or stop claiming it. |

### 1.2 FALSE or not reproduced — do not "fix" these

| Claim | Why we reject it |
|---|---|
| "Auto-Opus stayed in 1-Up Fill for 100% of both videos" | Contradicted by the exports I frame-checked after the tracker fix: 2-Up split and editorial layouts do render. The earlier pillarbox run was caused by a real bug (silent MediaPipe fallback + `auto`→`fit` promotion), both now fixed. |
| "Both 2-Up panels show the exact same two people stacked" | Not observed in our verified frames. The **geometry that would allow it** is real (T1), so we fix the gate, but we do not treat it as a confirmed defect. |
| "Route 4+ person / wide aisle / scoreboard shots to Cinema-Card Fit+Blur" | Directly contradicts the rule you approved in the last round: the talking person must fill the frame. We removed `auto`→`fit` deliberately. Rejected as written; see §4.5 for the compromise. |
| "Lower the scene-cut threshold" (unstated but implied by T10) | Only safe together with an *edge-change* signal. Histogram-only lowering will produce false cuts under studio lighting. Our cut gate is also Bhattacharyya-based in the worker (`diff > 0.48`), which is a different, better-conditioned metric than the browser's absolute-difference sum. |
| "Add OCR-based burned-in subtitle detection" | Deferred. Requires an OCR model in the worker, and the DOCX itself concedes it could not tell source text from our overlays. Not a code-verified defect. |

### 1.3 Report claims we cannot evaluate

The reports were generated from **exported clips only** (no source video, no ground-truth transcript,
no project settings). Anything that depends on the source — "slices the -6,614 counter", "drops the
yellow source text" — is plausible but not provable from the artefacts given. We treat those as
*hypotheses to test on our own footage*, not as defects to fix.

---

## 2. Repository comparison

For each repo: what it is, what is actually reusable, its dependency weight, the compatibility
problem it creates for ClipForge, and the verdict.

Verdict vocabulary:
- **PORT** — reimplement the algorithm inside our existing modules; no dependency added.
- **REFERENCE** — read for design, adopt nothing.
- **ADOPT** — install/require it as a dependency in its native runtime (browser package or worker).
- **SKIP** — no fit.

### 2.1 ClipForge Reframe Engine v2 — Python, scene-aware reframing + editing skills

- **Relevance: highest.** Licence: MIT (its `cloud/` subtree is separately commercial — do not touch).
  We already ported its core: `SpeakerTracker` (sticky hysteresis, cooldown, `seen` maturity,
  centrality), `SmoothedCameraman` (25% deadband, `jump_confirm_frames=3`, `begin_scene()` snap),
  `split_layout.analyze_scene` (coexistence ≥ 50%, separation ≥ 20%, median centres), `active_speaker`,
  HSV/Bhattacharyya cuts, punch-in beats, `compact_keyframes`.
- **Exact constants verified in its source** (these replace guessed values in §4):

  | Area | Constant | Source |
  |---|---|---|
  | Cameraman | deadband `crop_width*0.25`; pan 3 px/frame, 15 px/frame when `|diff| > crop_width*0.5`; `JUMP_CONFIRM_FRAMES=3`; `force_snap` at scene start | `main.py:94-219` |
  | Tracker | score `face_area/(w²*0.05)`, decay ×0.85/frame, drop `<0.1`, sticky ×3.0, switch cooldown 30 frames, ID radius `width*0.15` | `main.py:221-346` |
  | Sampling | `DETECT_STRIDE=4`, `YOLO_FALLBACK_STRIDE=8`, `DETECT_MAX_WIDTH=640`, black-frame skip `mean<16` | `main.py` |
  | Scene strategy | GENERAL if `avg_faces > 1.2 or < 0.5`; **a scene shorter than 2.0 s inherits its neighbours' strategy** | `main.py:494-559` |
  | Split gate | `MIN_COEXISTENCE 0.5`, `MIN_SEPARATION 0.20`, `MIN_FACE_WIDTH 0.045`, `MIN_SCENE_SECONDS 2.5`, `MAX_FACE_OVERLAP 0.30`, `SPLIT_TIGHTNESS 0.8`, crop `y = cy − crop_h*0.42`, median centres, 1 sample/1.5 s clamped 8–24, needs `≥max(4, n//2)` readable samples | `split_layout.py` |
  | **Group layout** | 3–4 people → **2×2 grid** (540×960 tiles at 1080×1920); 3 bands explicitly **rejected as too landscape**; `MIN_SEPARATION 0.12` Euclidean; `PANEL_TIGHTNESS 0.85`; crop clamped to `neighbour_gap*0.9`; headcount = the count that held longest | `panel_layout.py` |
  | Active speaker | `WINDOW 0.4 s`, `AUDIO_FLOOR 0.18` of peak RMS, `MIN_MARGIN 0.15`, `MIN_SHARE 0.2`, 20th/80th-percentile normalisation per speaker, `hold()` needs 3 consecutive windows, mouth ROI `y+0.72h` × `0.36w × 0.22h`, decode 480 px grayscale, 8 kHz mono | `active_speaker.py` |
  | Punch-in | `MAX_ZOOM 1.12`, 0.25 rise / 1.30 hold / 0.55 fall, smoothstep, overlapping beats take **max, not sum**, `MIN_GAP_SECONDS 18.0`, `BEAT_PROMINENCE 0.60` | `punch_in.py` |
  | Fit/contain | `GENERAL_CONTENT_HEIGHT_RATIO 0.42` (0.55 measured worse) | `general_filtergraph` |
  | Camera inset | `MAX_SUBJECT_HEIGHT 0.35`, `MIN_OFFSET 0.18`, `INSET_PADDING 1.45`, inset 0.10–0.38 height, `MAX_CENTRE_SPREAD 0.05` | `camera_inset.detect` |
  | Captions | one ASS event **per word**, word end = next word's **start** (removes gap flicker); `SAFE_MARGIN_V 43` of PlayResY 288 ≈ 15%; max 16 chars / 1.4 s; pop `\fscx90→108` over 110 ms; inactive dimming by RGB scaling `0.5+0.5*opacity` (alpha tested muddier); `merge_continuation_words()` glues faster-whisper tokens that lack a leading space | `subtitles.py` |
  | Clip edges | windows 90 s / 30 s overlap snapped to segment ends (cap 1.25×); `snap_clip_to_words(search_window=1.5, max_lead=0.35, max_tail=0.45, +0.2 s)`; two-pass scoring, batches of 8, shortlist `max(3, min(10, dur//90 + 2))` | `clip_selection.py`, `main.py:1251-1270` |
  | Effect restraint | `EFFECT_LIMITS` zoom/punch ≤ 0.15, pulse ≤ 0.10, `MAX_EDITS 12`, `MIN_EFFECT_SECONDS 0.15`, `MAX_ZOOM_SECONDS 8`, `ZOOM_CENTER_Y 0.45`, and **all zoom effects are dropped when captions are burned in** | `edit_builder.py` |
  | Audio/encode | `loudnorm=I=-14:TP=-2.0:LRA=11` (measured: TP −1.5 clips after AAC); NVENC `cq 25/29` for p5/p4, `cq ≈ crf+7`; per-stream metadata scrub; `+faststart` after concat | `ffmpeg_utils.py` |

- **Reusable now, not yet ported:** the two rows above that fix our confirmed defects — `merge_continuation_words`
  (T6), the <2.0 s strategy inheritance (T2), `GENERAL_CONTENT_HEIGHT_RATIO 0.42` (fit-card size, §5.1),
  `_complete_speech_edges`/`snap_clip_to_words` (T7 mid-sentence starts), the >0.65 s dead-air trim in
  `smart_split.py`, `framing_geometry.py` / `screencast_layout.py` text-and-scoreboard bbox handling
  (the honest version of the report's scoreboard claim), and the word-per-ASS-event caption timing.
- **Dependencies:** ultralytics 8.4, torch 2.11, mediapipe 0.10.14, scenedetect 0.7, TransNetV2,
  faster-whisper 1.2.1 — the first four already exist in our worker stack; TransNetV2 is the only new
  heavy weight and we do not need it.
- **Compatibility:** its render path is ffmpeg `sendcmd` + `crop@c` keyframes. Legal for us **only in
  the local worker** (AGENTS.md bans server-side heavy video; the worker is the user's own machine).
  The **browser port** (`src/lib/reframe.ts`) must keep parity or the two trackers drift — that is
  exactly how the silent-fallback bug became visible.
- **Verdict: PORT** (targeted algorithms + verified constants), **no new dependency**. Skip TransNetV2.

### 2.2 tryvinci/vinci-clips — TypeScript/React clipping app

- **Licence: AGPL-3.0.** Consequence: ideas and published parameter values may inform an independent
  implementation, but **no code, no constants copied into a derived file, no vendoring**. Everything
  from this repo in §4 is written from scratch against our own types.
- **Relevance:** closest *architecture* match to us — browser-side face detection plus a Node/Express
  backend with no database (JSON file) and fluent-ffmpeg.
- **Worth knowing (verified):**
  - Browser identity persistence by face-descriptor Euclidean distance `< 0.6`, prune after 2000 ms,
    sample every 0.5 s by seeking a `<video>` — a genuinely browser-native tracking pattern
    (`frontend/src/components/SubjectDetection.tsx:96-145`).
  - "Visual Director": group detections per 100 ms, protagonist = most-centre face, average the box
    per scene, then one crop filter with 0.5 s interpolated pans (`backend/src/routes/reframe.js:84-188`).
    Cheaper than a per-frame trajectory, weaker than our SmoothedCameraman.
  - Caption phrase grouping: break after ≥4 words or on `.!?` (`captions.js:150`) — the same rule the
    DOCX report asks for, in its crudest form. Our P0.2 does better.
  - Manual drag-box region picker + webcam-band percentage (`StreamerGameplayCrop.tsx`) — the
    interaction we lack for manual crop override.
- **Confirmed absent** (so the report cannot blame us for lagging here): no scene-cut detection, no
  audio processing of any kind (zero `loudnorm`/`afftdn`/`acompressor`/ducking hits in `backend/src`),
  no hook scoring, no duplicate detection, no dead-air removal, no keyword highlight, no emoji, 720p out.
- **Verdict: REFERENCE.** Adopt the *state-split* idea (layout plan vs caption plan vs overlay plan)
  and the manual-region-picker interaction. Steal no code.

### 2.3 remotion-dev/remotion — React video rendering (v4.0.541 inspected locally)

- **Relevance:** conceptually strong — deterministic frame-driven rendering (`useCurrentFrame` reads a
  timeline store; `<Sequence>` re-bases the frame), so every visual is a pure function of frame `t`.
  That is the missing discipline in our own `drawFrame`, where motion is computed from wall-clock `time`
  inside the caption and layout code.
- **Architecturally wrong for us:** its export path is *not* browser-side — `<OffthreadVideo>` picks a
  native Rust compositor in `packages/compositor` under Node/Lambda. That contradicts AGENTS.md (no
  long jobs server-side) and duplicates a pipeline we already run. Also note: `serializeAudio` does not
  exist in v4 (that is a 1.x API); current audio automation is
  `VolumeProp = number | ((frame: number) => number)`.
- **Licence terms — verified from the repo's own `LICENSE.md`:** individuals and small companies may use
  Remotion for free including commercially (eligible: individual, for-profit with **up to 3 employees**,
  non-profit); larger firms need a paid Company License. Critically: *"It is not allowed to copy or modify
  Remotion code for the purpose of selling, renting, licensing… your own derivate."*
  Consequence for us: **we may adopt the concepts and the public mathematics (spring oscillators,
  piecewise linear interpolation, easing), but we must write our own implementations from a blank file —
  no translation of their source, no vendoring, no `@remotion/*` package in a paid product** until the
  current licence text is re-read against ClipForge's commercial status. npm footprint if ever adopted:
  ≈2.09 MB unpacked (+0.49 MB transitions), peer `react >= 16.8`.
- **Verdict: SKIP the dependency. PORT the ideas** — an independently written
  `src/lib/motion.ts` (`spring(t, {mass, damping, stiffness, overshootClamping, durationRestThreshold})`,
  `interpolate(v, inRange, outRange, easing, extrapolate)` ≈200 LOC) and `src/lib/transitions.ts`
  (fade / slide / wipe / clock-wipe / zoom-blur presets + linear & spring timings) wired into the shot
  boundaries in `render.ts`. This is exactly what task #8 needs, and it kills the ad-hoc `Math.sin`
  ramps in [src/lib/subtitles.ts:56](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/subtitles.ts#L56).
- Also portable: the `VolumeProp` frame-function shape for audio automation (gain node in preview,
  FFmpeg volume expression in the NVENC mux).

### 2.4 remotion-dev/skills — agent skill packs for Remotion

- **Structure (inspected):** `skills/remotion-best-practices/SKILL.md` is a router into ~14 sub-skills
  (`remotion-captions`, `remotion-markup` with ~30 topic files incl. `transitions.md`, `timing.md`,
  `multi-scene-video.md`, `silence-detection.md`, `remotion-saas` with `player.md`/`rendering.md`,
  `remotion-license`). All are markdown instruction assets, not libraries; versioned to 4.0.541.
- **Licence:** the repo exposes **no licence file** (GitHub API returns `license: null`) — treat as
  reference text, do not copy verbatim.
- **Worth adopting:** (1) its caption data contract `Caption {text, startMs, endMs, timestampMs,
  confidence, pageBreakAfter}` — `pageBreakAfter` is a clean way to express what our P0.2 grouper decides,
  and it fits our existing `Word {w,s,e}` with one added boolean; (2) the router-plus-topic-files layout
  as the documentation shape for our own editing skills, so an agent (or a future "Auto OpenShorts v3")
  can drive features by name.
- **Verdict: REFERENCE.** Adopt the `pageBreakAfter` convention and the skill-pack documentation format;
  copy no text.

### 2.5 openvideodev/react-video-editor — browser NLE

- **Relevance:** the best *data-model* reference we have. It is not a component library — a Next.js 16 /
  React 19 app built on `@openvideo/core` + `@openvideo/engine-pixi` + `@openvideo/timeline`.
- **Architecture worth copying (as a concept, in our own types):**
  - Everything is measured in **microseconds** (`new Core({settings:{width,height,fps,duration}})`),
    which is why its timeline math never drifts — our `Clip.start/end` are float seconds.
  - `IClip` is normalised: `display:{from,to}` separate from `trim:{from,to}`, plus `playbackRate`,
    `left/top/width/height/angle/zIndex/opacity`, `effects[]`; `ITimelineTrack {clipIds}` with track
    types including Caption / Effect / Transition.
  - The timeline strip is **canvas-rendered** with registered item types and explicit snapping targets
    (`snappable` in `utils/target.ts`) — the affordance we lack for manual word/crop override.
  - Export is browser-side via **mediabunny** (MPL-2.0, ≈11 MB unpacked; weak copyleft — usable, changes
    to its files must be shared).
- **Compatibility / licence:** repo `LICENSE` is an "OpenVideo License" restricted to individuals /
  ≤3-employee companies / non-profits and it says it applies to *all packages in the monorepo*, while the
  published `@openvideo/*` packages claim MIT and point at a repository that 404s. **Unresolved
  contradiction — treat as a licensing red flag and take no dependency on it.**
- **Verdict: REFERENCE + PORT the interaction and model ideas** (µs timebase, display/trim split,
  snapping targets, per-word caption edit). No dependency, no code.

### 2.6 SYSTRAN/faster-whisper — CTranslate2 ASR (MIT, pure-python `py3-none-any` wheel)

- **Relevance: already adopted and running** (`small`/CUDA in the worker, word timestamps, VAD,
  chunk+overlap merge). Our call path is unbatched —
  [local_worker/whisper_engine.py:204](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/whisper_engine.py#L204).
- **Root cause of the number-splitting defect (T6) is upstream and structural, not our bug alone.**
  `tokenizer.py:199-215` `split_tokens_on_spaces` starts a new "word" when a subword begins with `" "`,
  is a special token, **or `subword.strip() in string.punctuation`** — and Whisper BPE emits `.` and `,`
  as standalone tokens inside numbers, so `2.4` arrives as `" 2" | "." | "4"`. Then
  `transcribe.py:1971-1983` `merge_punctuations` only pulls a *following* word backwards when that word
  is exactly in `append_punctuations`, so you get `"2."` + `"4"`, and glued fragments like `".4"` or
  `",614"` are never merged. **There is no upstream knob for this.** The fix must be our own adjacency
  pass (`transcript_validator.py` first, then the browser grouping), matching on digit/punctuation
  adjacency rather than on the leading space we currently strip at :51.
- **Verified defaults worth knowing:** `vad_filter=False`, `condition_on_previous_text=True`,
  `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
  `temperatures=(0,.2,.4,.6,.8,1.0)`, `prompt_reset_on_temperature=0.5`,
  `prepend_punctuations="\"'“¿([{-"`, `append_punctuations="\"'.。,，!！?？:：”)]}、"`,
  `suppress_tokens=[-1]`. Alignment is cross-attention (`ctranslate2 model.align`, hardcoded
  `median_filter_width=7`); **`word_timestamps_max_jump` does not exist** in any version — do not plan
  around it. Timing guards: median word duration capped at `min(0.7, median)`, max `2×median`,
  sentence-end truncation on `.。!！?？`.
- **VRAM:** `BatchedInferencePipeline` defaults `batch_size=8` and forces VAD
  `max_speech_duration_s = 30 s`. The project's own RTX 3070 Ti 8 GB table: int8 bs=1 → 2926 MB,
  int8 bs=8 → 4500 MB, fp16 bs=8 → 6090 MB. On our RTX 2050 / 4 GB that means `int8_float16` or `int8`
  with `batch_size=4–8`; **avoid fp16 and `large-v3-turbo` fp16**. `VadOptions` defaults
  (`threshold=0.5`, `min_silence_duration_ms=2000`, `speech_pad_ms=400`) are too coarse for interview
  cutting — try `min_silence_duration_ms≈400–600`.
- **Compatibility:** `ctranslate2>=4,<5` ships cp314 win_amd64 (4.8.5), `onnxruntime>=1.14,<2` (1.31.1 ✓),
  `av>=11` (19.0.2 ✓, requires-python ≥3.12 ✓) — so our Python 3.14 worker is compatible today. The
  `initial_prompt` digit-formatting hint is a community trick, **not documented behaviour**: A/B it on
  our footage before claiming it works.
- **Verdict: ADOPT (already) — tune and post-process.** Add `BatchedInferencePipeline` + int8 +
  VadOptions tuning; write our own continuation-token merge; seed `initial_prompt` (T12).

### 2.7 Breakthrough/PySceneDetect — scene detection (BSD-3-Clause, `scenedetect-core`)

- **Relevance: exactly the right fix for T10, and it explains *why* our current detector misses indoor
  cuts.** Our worker test at
  [local_worker/openshorts_engine.py:1277-1286](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/openshorts_engine.py#L1277)
  (HSV histogram, Bhattacharyya > 0.48) is the analogue of PySceneDetect's own `HistogramDetector`
  (Y-hist, 128 bins, `HISTCMP_CORREL`, threshold 0.20→0.8) — and a **palette comparison cannot
  distinguish two camera angles of the same room**. That is the documented reason indoor cuts slip
  through, independent of threshold choice.
- **`ContentDetector` (verified):** per frame `h,s,l = cv2.split(cv2.cvtColor(frame, BGR2HSV))`; each
  component score is `_mean_pixel_distance = sum(abs(int32(a)-b)) / npix`;
  `score = (Δh·1 + Δs·1 + Δl·1 + Δedges·0) / Σ|w|`; cut when `score >= threshold` (default **27.0**),
  `min_scene_len` **15 frames**; `luma_only` = weights (0,0,1,0). Edge weighting is opt-in: Canny on luma
  with `sigma=1/3`, `low/high = (1∓σ)·median(luma)`, dilate kernel `4 + round(sqrt(w·h)/192)` forced odd.
  Note **Δhue is a plain absolute difference with no 0/179 wrap-around** — copy that for parity.
- **`AdaptiveDetector` (the piece we need):** two-pass. Buffer `1 + 2·window_width` content values
  (default window 2 → 5). `avg` = mean of the 4 neighbours; `ratio = min(target/avg, 255.0)`; if
  `avg < 1e-5 and target >= min_content_val` → 255. Cut iff `ratio >= adaptive_threshold (3.0)` **and**
  `target >= min_content_val (15.0)` **and** gap ≥ `min_scene_len (15)`. The absolute base threshold is
  pinned at 255 so it never fires alone. A rolling-mean ratio is what surfaces low-contrast cuts while
  suppressing fast-motion false positives. `FlashFilter` offers MERGE vs SUPPRESS for white flashes.
  Downscale convention: `compute_downscale_factor(width, 256) = width/256`.
- **`TransNetV2Detector` exists** (ONNXRuntime, frames resized 48×27 INTER_AREA, 100-frame windows,
  25-frame overlap, uses `pred[25:75]`, threshold 0.5, rising edge only) **but the `.onnx` weights are
  not in the repository** (default path points at `tests/resources/`) and download size could not be
  verified — so this stays skipped.
- **Dependencies:** numpy at runtime plus OpenCV; `requires-python >= 3.10` — installs cleanly in our
  worker. We want the algorithm, not the `SceneManager` file reader.
- **Verdict: PORT ContentDetector + AdaptiveDetector** into the worker (every frame at ~256 px width
  instead of our current stride-4 sample) and mirror it in the browser for parity; keep the 0.48
  histogram test only as an OR-signal. BSD-3 attribution goes in `THIRD_PARTY_NOTICES.md`.

### 2.9 pyannote/pyannote-audio — speaker diarization

- **Relevance:** the real answer to "who is speaking". It would let split-layout pick panels by
  *speaker change* and stamp a `speaker_id` onto each Whisper word. It is **not** what fixes T13 —
  that gate is our own dead code and should be wired up first, cheaply, with mouth pixels + RMS.
- **Licence facts, verified (this replaces the earlier caution):**
  `pyannote/speaker-diarization-community-1` is **CC-BY-4.0**, gated "auto" (fill company/use-case),
  and the model card states it "will always remain freely accessible" → **commercial use is permitted
  with attribution** (attribution line goes in `THIRD_PARTY_NOTICES.md`).
  `pyannote/speaker-diarization-3.1`, `pyannote/segmentation`, `pyannote/segmentation-3.0` are **MIT**.
  Code itself is MIT. So the licence is not the blocker; the install is.
- **Download cost is small:** community-1 totals **33.7 MB** (segmentation 5.91 MB, embedding 26.65 MB,
  plda 0.27 MB); `wespeaker-voxceleb-resnet34-LM` 26.6 MB; `segmentation-3.0` 5.9 MB. Still needs a
  user-supplied HF token in environment variables (.env) — never in the repo.
- **Real blocker: the CUDA / Python 3.14 pairing.** Deps are `torch>=2.8`, `torchaudio>=2.8`,
  `torchcodec>=0.7` (needs system ffmpeg — we have it), `lightning`, `pytorch-metric-learning`,
  `safetensors`, `opentelemetry-*`; `requires-python>=3.11`. PyPI torch 2.15.0 cp314 win_amd64 is
  **CPU-only (127 MB)**; `torch-2.9.0+cu129-cp314` exists on download.pytorch.org, but **torchaudio
  cp314 CUDA wheels exist only for +cu128/+cu130 and none for the cu129 index** — fragile. RTX 2050 is
  Turing sm_75, still present in cu12x builds (its status under CUDA 13 could not be verified).
  Runtime VRAM and wall-clock cost are **unverifiable** from published figures.
- **Defaults if we ever ship it:** `segmentation_step=0.1` (0.5 s hop on the 5 s window),
  `clustering="VBxClustering"`, `embedding_batch_size=1`, `min_duration_off=0.0`,
  clustering `threshold=0.6, Fa=0.07, Fb=0.8`; use `DiarizeOutput.exclusive_speaker_diarization` to map
  one speaker per time offset.
- **Verdict: DEFER.** Design the seam now (`speakerOf(t): string | null` on the caption plan). If it is
  ever adopted, run community-1 in a **separate py3.12 + torch 2.9.0+cu128 sidecar venv** rather than
  installing torch into the 3.14 worker, and keep it opt-in behind an explicit user decision.

### 2.8 google-ai-edge/mediapipe — vision tasks (Apache-2.0, `@mediapipe/tasks-vision@1.2.2` already installed)

- **Relevance:** already used in-browser (BlazeFace `FaceDetector` in
  [src/lib/reframe.ts:7](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/src/lib/reframe.ts#L7)).
  The worker now uses real YuNet faces + YOLOv8n, so the browser stack is the *fallback*, not the authority.
- **Model inventory — sizes verified by HTTP HEAD, options verified against the installed `vision.d.ts`:**

  | Task | Model | Size | What it buys us |
  |---|---|---|---|
  | FaceDetector | `blaze_face_short_range` | 230 KB | already used; 6 keypoints — enough for an eye-band proxy |
  | FaceLandmarker | `face_landmarker` | 3.76 MB | 478 landmarks + blendshapes + head transform → **real eye/mouth boxes for the P0.3 safe zone** |
  | ImageSegmenter | `selfee_segmenter` | 250 KB | **person pixel mask** — survives head-turns, enables true background blur |
  | PoseLandmarker | `lite` / `full` / `heavy` | 5.8 / 9.4 / 30.7 MB | 33 3D landmarks → **torso anchor when the face is lost** (the "back of dreadlocks" case) |
  | ObjectDetector | `efficientdet_lite0` | 7.25 MB | nothing over YOLOv8n — **skip** |
  | FaceMatcher | — | URL 404 on both float16 and float32 | cannot verify; do not plan around identity matching |

- **Reusable knobs (verbatim names):** `baseOptions.modelAssetPath` + `delegate: "CPU"|"GPU"`,
  `runningMode`, `numFaces`, `minDetectionConfidence`, `minSuppressionThreshold`, `maxResults`,
  `categoryAllowlist`, `numPoses`, `minPoseDetectionConfidence/Presence/Tracking`.
- **Compatibility:** the browser GPU delegate is WebGL-based and needs a `gpuContext` canvas; CPU is the
  safe default. Each added model is a first-run download (3.76 MB landmark, 250 KB segmenter,
  5.8 MB+ pose) via the existing `FilesetResolver.forVisionTasks` CDN pattern. Any of these must stay
  optional and cached, and must **never** silently replace the worker's result — that regression is
  the one we just fixed.
- **Verdict: ADOPT selectively.** FaceLandmarker for safe zones (P0.3). ImageSegmenter + PoseLandmarker
  are the honest fix for the head-turn/torso-anchor claims — propose them as a separate opt-in phase
  because they cost download and per-frame CPU. Skip ObjectDetector and FaceMatcher.

### 2.10 FFmpeg/FFmpeg — filtergraph

- **Relevance:** already our encoder and audio analyser. Everything the reports ask for already exists
  as a filter, and **two of the "missing" features are already in our worker**:
  `contain_background()` (split → cover-scale → `gblur=sigma=N:steps=2` → `eq=brightness=-0.06:
  saturation=0.7` → overlay) is the Cinema-Card the PDF report says we lack, and `split_filtergraph`
  (split → crop → scale → vstack) is the 2-Up. Both live in
  [local_worker/openshorts_engine.py](file:///D:/My%20Project/clipcraft-studio-opus-upgrade%20(3)/clipcraft-studio-main/local_worker/openshorts_engine.py).
- **Verified against the installed Gyan 9.0-full build** (`-h filter=X`, `ffmpeg -filters`,
  `--enable-nvenc` present in its configure line):

  | Need | Recipe | Cost |
  |---|---|---|
  | 3-up grid, one pass | `xstack=inputs=3:layout=0_0_WxH\|0_H_wxh\|w_H_wxh:fill=black` or `split=3` → crop/scale → `hstack`+`vstack` | cheap |
  | Fit + blur bed | existing `contain_background()`; `gblur sigma 0–1024, steps 1–6`. `boxblur` radius must stay `< half the smallest dim`, hence the existing downscale-first order | medium |
  | Upscale sharpen | `scale=1080:1920:flags=lanczos,unsharp=lx=5:ly=5:la=0.8` | cheap, **canvas cannot do this** → worker |
  | Highlight protection | `eq=gamma=0.98:gamma_weight=0.6` (`gamma_weight` reduces gamma's effect on bright areas) or `curves=master='0/0 0.75/0.71 0.93/0.87 1/0.94'` | cheap |
  | Loudness | two-pass `loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json` then `:linear=1:measured_I/TP/LRA/thresh/offset`. OpenShorts ships `I=-14:TP=-2.0:LRA=11` with measured evidence that `TP -1.5` clips after AAC — **pick TP −2.0** | pass 1 is cheap |
  | Denoise | `afftdn=nr=12:nf=-50:tn=1`. `arnndn` needs an external `.rnnn` file that **is not bundled** in Gyan builds | cheap |
  | Music duck | `[music][voice]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300[dm];[dm][voice]amix=inputs=2:duration=first` (sidechain = input #1) | cheap |
  | Audio fades | `afade=t=in:st=0:d=0.3:curve=hsin`, `afade=t=out:st=END-0.5:d=0.5:curve=exp` | cheap |
  | Dead-air | `silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB:stop_periods=-1:stop_duration=0.6:stop_threshold=-40dB:timestamp=write` — **audio-only; video needs the matching EDL**, and we already detect silence in `audio_processor.py` | cheap |
  | NVENC master | `-c:v h264_nvenc -preset p5 -tune hq -profile:v high -rc vbr -b:v 16M -maxrate 18M -bufsize 32M -spatial-aq 1 -temporal-aq 1 -r 30000/1001 -fps_mode cfr`; tier map `cq ≈ crf+7`, `cq 25/29` for p5/p4 | — |
  | BT.709 | the encoder flags only **tag**; the actual conversion needs `colorspace=all=bt709:iall=bt601-6-625:fast=1`, which we already apply in `clipforge_helper.py:353` | cheap |
  | Motion blur | `tblend=all_mode=average:all_opacity=0.3` is ghosty, not shutter-like; `minterpolate=mi_mode=mci` is true MC but very slow CPU and **can break 30000/1001 CFR** → do not use | — |

- **Compatibility:** any frame composited by FFmpeg diverges from the canvas preview unless kept in
  sync; AGENTS.md deliberately shares one renderer. **Rule: FFmpeg owns post-passes (blur bed, lanczos +
  unsharp, eq/curves grade, loudnorm, denoise, duck, dead-air, encode); canvas owns compositing.**
  The two exceptions worth moving the *other* way: preview-side motion blur (draw the previous decoded
  frame at `globalAlpha ≈ 0.2` in the capture loop — free, and preview equals export) and the fit/blur
  bed (`ctx.filter='blur()'` on a downscaled offscreen canvas).
- **Verdict: ADOPT (already adopted) — extend the post-pass graph; do not move compositing out of canvas.**

### 2.11 OCR (for the burned-in-subtitle claim, not one of the 10 repos)

The report's "detect yellow lower-third source graphics" ask needs an OCR signal. Measured options:
- **Tesseract.js** — 67 KB JS + 3.46 MB WASM core + ~11 MB `eng.traineddata.gz`, Apache-2.0, runs in
  the browser on sampled frames, zero backend change. Accuracy on stylised low-contrast burned-in
  subs is mediocre — good enough for a binary "is there text in the bottom band?", not for reading it.
- **PaddleOCR** — best accuracy on styled text, but pulls in paddlepaddle (large install).
- **EasyOCR** — pip-simple but drags ~2 GB PyTorch, which this worker deliberately avoids.

Recommendation: a Tesseract.js probe used only to *shift* the caption band away from detected source
text. Do not build a transcription feature on it, and do not add a 2 GB torch install for it.

---

## 3. Integration architecture

### 3.1 Layering

```
upload / source cache                      (unchanged)
        │
        ├─ ASR          faster-whisper (worker)         → Word[] + speaker hints
        ├─ Detection    YuNet faces + YOLOv8 persons     → Detection[] per sampled frame
        ├─ Cut detect   adaptive LUV delta               → cut frames
        │
        ├─ CAMERA PLAN   SpeakerTracker + Cameraman      → keyframes {x,y,zoom,mode}
        ├─ LAYOUT PLAN   scene-scoped chooser            → LayoutPlan[]   ← NEW gate layer
        ├─ TEXT PLAN     safe-zone solver                → TextPlan[]    ← NEW
        ├─ CAPTION PLAN  phrase grouper                  → CaptionPage[] ← NEW
        └─ OVERLAY PLAN  hook / callout / emoji / bar    → OverlayPlan[] ← NEW
        │
        render.ts (canvas compositor, unchanged structure)
        │
        encode (MediaRecorder | NVENC master) + FFmpeg post-pass
```

The five "PLAN" blocks are the architectural change. Today `reframe.ts` decides camera *and* layout
*and* the renderer decides text placement, which is why fixing one thing breaks another. Each plan
becomes a pure function with a typed output, so a plan is inspectable in the UI and testable offline.

### 3.2 New modules and their interfaces

| Module | Interface | Notes |
|---|---|---|
| `src/lib/plan/layout.ts` | `planLayout(scenes, detections, opts): LayoutPlan[]` | Owns the gates: `MIN_LAYOUT_HOLD_SEC`, `MIN_SPLIT_SEPARATION`, split zoom. Replaces layout decisions buried in `reframe.ts:672-721`. |
| `src/lib/plan/textZone.ts` | `solveTextZone(regions, faces, layout, opts): {y, band, plate}` | Pure scoring: overlap with eyes/mouth/subject/other text, platform-UI margins. Consumed by hook, caption and callout drawing. |
| `src/lib/plan/captions.ts` | `buildCaptionPages(words, opts): CaptionPage[]` | Phrase + pause + reading-speed grouping, orphan-line rule, keeps ≤ 2 lines. Replaces `captionPages()` call sites; `captionPages` stays for SRT/VTT export so file interchange is unaffected. |
| `src/lib/plan/overlays.ts` | `planOverlays(clip, words, zones): OverlayPlan[]` | Emoji selection (whole-word, contextual), number callout relevance, progress-bar default. |
| `src/lib/motion.ts` | `spring(t, {mass, damping, stiffness, overshootClamping, restThreshold})`, `interpolate(v, inRange, outRange, easing, extrapolate)` | Independently written, ~200 LOC. Replaces the ad-hoc `Math.sin`/linear ramps in `wordMotion`. Remotion-style API shape only — see the licence note in §2.3. |
| `src/lib/transitions.ts` | `drawTransition(ctx, kind, progress, layers)` | fade / slide / wipe / clock-wipe / zoom-blur presets + linear & spring timings, canvas-compatible. Task #8. |
| `src/lib/text/numbers.ts` | `mergeContinuationTokens(words): Word[]` | Shared by worker and browser so `2`+`.4` never splits. Port of the OpenShorts rule, written against our `Word` type. |
| `local_worker/scene_detect.py` | `adaptive_cuts(frames) -> list[int]` | PySceneDetect's LUV-delta + rolling-average algorithm against our 64×36 sampler. |
| `local_worker/asr_hints.py` | `build_initial_prompt(title, domain_terms) -> str` | Seeds Whisper domain vocabulary. |
| `local_worker/grapass.py` | `post_pass(input, opts) -> path` | lanczos+unsharp, eq/curves, two-pass loudnorm, afade edges. Worker-side only, after the canvas composite. |

`ClipSettings` gains optional fields only (`captionZone: "auto" | "lower" | "center" | "upper"`,
`sourceHasBurnedInCaptions`, `motionIntensity`). Old exports keep parsing.

### 3.3 Non-negotiable invariants

1. The worker is the tracker of record. Any browser fallback logs a visible warning (just landed).
2. `auto` framing never produces a pillarbox. `fit` is an explicit user choice.
3. `src/lib/render.ts` remains the single compositor shared by preview and export.
4. No new runtime dependency in the browser bundle without a size/first-download number attached.
5. No gated-model or HF-token work without an explicit user decision.
6. Licences have been read from the repos themselves (see §2 and the provenance note below). Anything
   still marked unverified must be re-read against ClipForge's commercial status **before** a package is
   added or its text is copied. Attribution entries go in `THIRD_PARTY_NOTICES.md`: OpenShorts (MIT),
   PySceneDetect (BSD-3) for derived cut-detection maths, MediaPipe (Apache-2.0), faster-whisper (MIT),
   pyannote community-1 (CC-BY-4.0) **only if** it is ever enabled, and the existing ClipForge Caption Engine MIT notice.
7. Reference clones are read-only under `./worker_data/ref_repos\{remotion, remotion-skills,
   ove-rve, faster-whisper, PySceneDetect, pyannote-audio}`. Nothing is vendored from them.

---

## 4. Phased plan (each phase ships with tests and a real render check)

Every constant below is either read from our own code/frames or taken from the verified OpenShorts
table in §2.1. Values the PDF report invented without evidence are marked `(report-only)` and are
tuned against our footage before they are trusted.

### P0.1 — Layout gating, hold, and geometry (fixes T1, T2)

- **Layout hold.** Adopt OpenShorts' rule directly: a scene shorter than **2.0 s inherits the
  neighbours' layout** instead of choosing its own (`main.py:494-559`). This is the structural fix for
  the 0:04 / 0:22 flashes; the report's `MIN_LAYOUT_HOLD_SEC = 2.0` is a re-statement of it.
- **Split gate.** Keep coexistence ≥ 50% and add a hard centre-distance requirement. OpenShorts uses
  `MIN_SEPARATION 0.20`; the report asks for 0.38 `(report-only)`. Ship it **measured in normalised
  frame width with the value set from our own two-shot footage**, starting at 0.30 and re-tuned; 0.20 is
  demonstrably too loose given T1.
- **Panel geometry.** Enforce `splitZoom >= 1.55` so each panel crops ≈41% of width instead of 63.3%,
  and adopt OpenShorts' `MAX_FACE_OVERLAP 0.30` IoU rejection plus `crop y = cy − crop_h*0.42`
  (we currently use `0.36`).
- **New capability, not in the report's form:** 3–4 person shots → **2×2 grid** (540×960 tiles at
  1080×1920). OpenShorts explicitly rejects 1-top/2-bottom as too landscape; the PDF report's "3-Up
  Split Grid (1 Wide Top + 2 Bottom)" therefore does **not** come from OpenShorts. Worker side is one
  `xstack=inputs=4:grid=0_0|1_0|0_1|1_1` pass; canvas preview draws four `drawImage` quads.
- Tests: hold-merge and separation-rejection units; a synthetic multi-scene fixture; then re-track and
  export the two clips that flashed at 0:04 and 0:22 and diff those frames.

### P0.0 — Make the split gate real (fixes T13, our own bug)

Before any layout tuning, either wire the existing active-speaker functions into
`analyze_openshorts_reframe` (mouth-pixel activity per split panel + 8 kHz RMS + percentile
normalisation + `hold(min_windows=3)`, all already implemented and unit-tested) **or** remove the
`split_layout_active_speaker_gate` / "8kHz audio RMS gate" strings from `main.py:189` and
`object_tracker.py:58,66` so the worker stops reporting a gate it never runs.
Decision: I want to wire it in, and if it rejects too many real two-shots on our footage we fall back
to coexistence-only and say so in the status.
- Tests: the existing unit test keeps passing; a new test asserts the analyze path actually calls the
  gate; a status-output test asserts nothing unimplemented is advertised.

### P0.2 — Phrase-aware, collision-free captions (fixes T3, T5, T6)

- **Token integrity first:** merge faster-whisper continuation fragments before whitespace is stripped
  (OpenShorts `merge_continuation_words()`), in `local_worker/transcript_validator.py:51` **and** the
  browser path. Until this lands, every downstream grouping test is measuring the wrong words.
- **Grouping:** punctuation → silence gap (> 0.45 s) → reading speed, max 2 lines, orphan rule (never
  strand a single short word on line 2 unless it is the highlight keyword). Carry a
  `pageBreakAfter` flag on `Word` (remotion-skills convention, §2.4) so paging is data, not a recompute.
  OpenShorts' hard caps — 16 chars / 1.4 s per event — are the reference for readability limits.
- **Per-word event timing:** a word's displayed end is the **next word's start**, not its own end.
  That is what removes gap flicker between words; our `wordMotion` currently stops animating at `word.e`.
- **Collision fix (T5):** advance each word by `width * maxScale + minSpace` (`maxScale = 1.12` is
  already our `MAX_ZOOM`), i.e. lay out at the pop size, not at 1.0. OpenShorts pops 90→108% over 110 ms
  and dims inactive words by **RGB scaling `0.5 + 0.5*opacity`, not alpha** (alpha measured muddier) —
  adopt both.
- **Safe margin:** 15% of frame height from the bottom edge (OpenShorts `43/288`), because our current
  `captionY` default sits inside platform UI zones at 0.78–0.85.
- Tests: golden fixtures for `"2"+" .4"→"2.4"`, `"6"+" ,614"→"6,614"`, orphan and 2-line cases, plus the
  exact strings the report complained about rebuilt from our 10 357-word transcript cache.

### P0.3 — Face-aware text safe zones (fixes T9)

- Eye/mouth boxes: worker from YuNet landmarks; browser fallback from MediaPipe `face_landmarker`
  (3.76 MB, opt-in and cached; fall back to a face-box eye-band proxy if the model is not loaded).
- Score upper / middle / lower bands with penalties for eyes, mouth, main subject, other on-screen text,
  platform UI margins, and the split seam. Pick per shot; allow manual override.
- Hook card: ≤ 2 lines, visible 1.5–2.5 s, fallback order shrink → backing plate → hide. **The hook must
  never be drawn where a face is, and must never be the first 8 spoken words** (ties into P1.1).
- Tests: geometry units with synthetic face boxes; then frame-diff at every hook entry/exit and caption
  change on the two reference clips. Acceptance = zero hook or caption pixels inside an eye or mouth box.

### P1.1 — Clip boundaries, hooks, ASR vocabulary (fixes T7, T8, T12)

- Replace the `targetLen * 0.7` step with **word-snapped, non-overlapping windows**: OpenShorts
  `snap_clip_to_words(search_window=1.5, max_lead=0.35, max_tail=0.45, +0.2 s)` plus
  `_complete_speech_edges`, stepping from `clipEnd + 1.0 s`. Scoring in two passes over batches of 8
  windows with shortlist `max(3, min(10, duration//90 + 2))`.
- Build the hook from stake/question patterns; keep Gemini's options when online; reject filler openings.
- Seed Whisper `initial_prompt` with title + domain nouns (the field and the plumbing already exist).
  A/B it against the cached transcript before claiming the homophone rate improved — digit/noun hinting
  is community practice, not documented behaviour.
- Throughput on our 4 GB card: move the unbatched call to `BatchedInferencePipeline(batch_size=4–8)`
  with `int8_float16`/`int8`, and tighten VAD to `min_silence_duration_ms≈400–600`, `speech_pad_ms=400`
  (the library default 2000 ms is far too coarse for interview pacing). No new package — the batched
  pipeline ships inside the faster-whisper we already use.
- Optional (OpenShorts `smart_split.py`): > 0.65 s dead-air trim, which needs the matching video EDL
  because `silenceremove` is audio-only.
- Tests: zero-overlap assertion across a full discovery run; hook ≠ first 8 words verbatim; homophone
  fixture against the cached transcript; dead-air trim does not cut mid-word; batched int8 output matches
  the current unbatched word count within ±1% with no timing regression.

### P1.2 — Overlay relevance and motion restraint (fixes T4, T11, the smear claim)

- Whole-word emoji with a corrected map: `store|grocery|aisle` → 🛒; remove store/entire/attempting from
  the game regex; keep money/trophy/food/power-word classes.
- Number callouts only when the number is transcript-verified **and** not the active caption word.
- **The important documented rule:** OpenShorts drops *all* zoom effects when captions are burned in
  (`edit_builder.py`), and caps punch at 0.15 with `MAX_EDITS 12`. Our exports always burn captions, so
  the honest fix for "excessive smear" is **cap simultaneous punch-in + emphasis zoom**, not a smaller
  blur. Implement `motionIntensity` (default conservative), keep `MAX_ZOOM 1.12`, and use browser-side
  motion smear (previous decoded frame at `globalAlpha ≈ 0.2`) instead of `tblend` — free, and preview
  equals export.
- Punch-in beats: `MIN_GAP_SECONDS 18.0`, `BEAT_PROMINENCE 0.60`, overlapping beats take **max, not sum**.
- `progressBar: false` default (T11), still toggleable.
- Tests: emoji-map table units, callout-dedup unit, zoom-vs-caption exclusivity unit, one rendered clip
  compared frame-by-frame at the previous smear timestamps.

### P1.3 — Adaptive scene cuts (fixes T10 honestly)

- Port PySceneDetect's real maths, not a lowered threshold: **HSV mean-pixel-distance score**
  (`Δh + Δs + Δl` over `int32` abs diffs / pixel count, hue compared as a plain abs diff for parity),
  then the **AdaptiveDetector ratio test** — each frame's score against the mean of its 2 neighbours on
  each side, cut iff `ratio >= 3.0` **and** `score >= 15.0` **and** gap ≥ 15 frames. This is what
  surfaces low-contrast indoor angle changes while suppressing fast-motion false positives, because it
  is relative to local history.
- Sample **every frame at ~256 px width** (`compute_downscale_factor(w, 256)`) instead of the current
  stride-4 at 64×36 — the reason short indoor cuts are skipped over entirely.
- Keep the existing Bhattacharyya > 0.48 histogram test only as an **OR-signal**; on its own it is a
  palette comparison, which is exactly why two shots of the same room score low.
- Cut-influence zone (`min_scene_len`, plus `FlashFilter` MERGE for white-flash frames) so a cut lands
  on the dissolve, not at the peak frame; `begin_scene()` snaps on frame 0 of every accepted cut.
- Do **not** adopt TransNetV2 — its `.onnx` weights are not in the repo and its size is unverifiable.
- Mirror the same algorithm in the browser port so preview and worker agree.
- Tests: our own footage must flag the previously-missed indoor cuts; false-cut count on a static
  talking-head segment must not rise; a fast-pan fixture must not produce a cut; browser and worker
  cut lists must agree within one frame on the same sample.

### P2 — Grade, watermark, export profiles (task #8 + report §4 P2)

- Worker post-pass: `scale=flags=lanczos, unsharp=lx=5:ly=5:la=0.8` for upscales;
  `eq=gamma=0.98:gamma_weight=0.6` or a soft-clip `curves` preset for highlight protection; per-shot
  exposure match; vignette; `afade` in/out on clip edges. All cheap single passes; canvas cannot sharpen.
- Loudness: two-pass `loudnorm=I=-16:TP=-2.0:LRA=11` (OpenShorts measured that TP −1.5 clips after AAC —
  use **−2.0**, not the report's implied −1.5).
- Transitions at shot boundaries via the new `src/lib/transitions.ts` (fade / slide / wipe / clock-wipe /
  zoom-blur, linear + spring timing), independently written.
- Watermark on/off, size, opacity, safe position; respect the 15% caption margin.
- Add a smaller social export profile next to the 16 Mbps master
  (`h264_nvenc -preset p5 -rc vbr -b:v/-maxrate/-bufsize`, `cq ≈ crf+7` tier map) and validate with ffprobe.
- Optional, only if a manual timeline ships: mediabunny (MPL-2.0, 11 MB) as a `captureStream` alternative.

---

## 5. Decisions needed before coding

1. **Group shots: 2×2 grid vs Cinema-Card vs fill-only.** The PDF report asks for "3-Up Grid (1 wide
   top + 2 bottom)" and attributes it to OpenShorts — **OpenShorts actually rejects 3 bands as too
   landscape and uses a 2×2 grid for 3–4 people** (`panel_layout.py`). It also asks for Fit+Blur on wide
   shots, which contradicts your "focus the talking person" rule. Pick one:
   a) 2×2 grid for 3–4 people, fill for everything else *(reference-accurate, my recommendation)*;
   b) large fit card at `GENERAL_CONTENT_HEIGHT_RATIO 0.42` — note our current fit card is 717/1920 ≈
   0.373, so OpenShorts' own ratio is *bigger* than what we ship, not smaller;
   c) fill only, no group layout at all.
2. **`MIN_SPLIT_SEPARATION` value.** 0.20 (OpenShorts, currently in our worker), 0.30 (my proposed
   start), or 0.38 (report-only, unverified). I want to tune it on our footage; approve a tuning pass?
3. **Zoom-vs-caption exclusivity.** OpenShorts drops all zoom effects when captions are burned in. We
   always burn captions in Opus style, so adopting that rule literally would remove every punch-in.
   Proposed: keep punch-in but forbid simultaneous punch + emphasis-callout zoom, cap total at 1.12×.
4. **pyannote diarization.** The licence turned out to be fine (community-1 is CC-BY-4.0, commercial use
   allowed with attribution; 33.7 MB of weights). The blocker is the install: **no torchaudio cp314 CUDA
   wheel exists for the cu129 index**, so it means a separate py3.12 + torch 2.9.0+cu128 sidecar venv.
   Wire the existing pixel+RMS gate first (P0.0) and revisit only if it still mis-picks speakers?
5. **Model downloads in the browser.** FaceLandmarker 3.76 MB (needed for real eye/mouth safe zones),
   SelfieSegmenter 250 KB, PoseLite 5.8 MB. Which are acceptable as opt-in cached downloads?
6. **`progressBar` default → `false`.** Changes what existing users get on first export. Confirm.
7. **Timebase.** Adopt a microsecond integer timebase for the timeline model (OpenVideo's strongest
   architectural idea) now, while the editor is small, or keep float seconds and accept rounding drift?
8. **Worker restart.** Still pending your approval: the running worker (PID 35860) executes the
   pre-edit engine, so the `auto`-never-`fit` change is not live there. Several phases below need it.

## 6. What this plan does not do

- No new browser npm dependency, and no `@openvideo/*`, `remotion`, or `mediabunny` install. No torch or
  diarization install in the Python 3.14 worker.
- No code copied from AGPL (vinci-clips), source-derivative-restricted (Remotion), licence-contradictory
  (OpenVideo) or unlicensed (remotion-skills) projects — only independent implementations of observed
  behaviour and published constants.
- No server-side video rendering, no database/migration changes, no secrets in the repo, no change to
  the AI gateway path.
- No duplicated-caption work, per your instruction.
- No claim that any of this is fixed until a real clip has been rendered and frame-checked.
