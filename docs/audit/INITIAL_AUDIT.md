# ClipForge — Initial Repository Architecture & Quality Audit

**Project:** ClipForge (`ClipForge Studio`)  
**Maintainer:** Minuja Ranthika Amarasekara (`2024is008-source`)  
**Target Repositories:** `https://github.com/2024is008-source/ClipForge.git` / `https://github.com/2024is008-source/clipforge-next.git`  
**Audit Date:** 2026-11-14

---

## 1. Existing Architecture

ClipForge is a hybrid browser + local GPU/CPU workstation platform for converting long-form 16:9 videos into vertical 9:16 short-form clips:

1. **Frontend Studio (`src/`)**:
   - **Framework**: TanStack Start (`@tanstack/react-start` 1.168.60), TanStack Router (`1.170.41`), React 19.2, TypeScript 5.8 (`strict: true`), Tailwind CSS v4.2, and Radix UI primitives.
   - **Browser Preview & Interactive Canvas (`src/lib/render.ts`, `src/lib/subtitles.ts`, `src/lib/render-overlays.ts`)**: Renders 1080×1920 frames at interactive framerates using HTML5 Canvas 2D, word-level subtitle animations (`opus`, `beast`, `karaoke`, `bounce`, `typewriter`, `glow`, `reveal`, `minimal`), and transparent PNG overlay batching for native FFmpeg compositing.
   - **Browser Vision Fallback (`src/lib/reframe.ts`)**: Same-origin MediaPipe Tasks Vision WASM + float16 models (`blaze_face_short_range.tflite`, `efficientdet_lite0.tflite`) with automatic GPU-to-CPU delegate recovery.
   - **Server Functions (`src/lib/ai.functions.ts`)**: SSR RPC endpoints supporting a 5-Key Google Gemini Free Tier rotation pool (`GEMINI_API_KEY_1`..`5`) with automatic HTTP 429 cooldown and local transcript ranker fallback.

2. **Local Processing Worker (`local_worker/`)**:
   - **Framework**: FastAPI + Uvicorn service bound to `127.0.0.1:8765` with Bearer token authentication (`X-ClipForge-Token` / `Authorization: Bearer`).
   - **Speech Transcription (`local_worker/whisper_engine.py`, `local_worker/transcription_jobs.py`, `local_worker/transcript_validator.py`)**: Local `faster-whisper` (CTranslate2) inference with `int8_float16` CUDA and `int8` CPU execution, word-level timestamp extraction, BPE continuation token merging, and monotonic timestamp validation.
   - **Computer Vision & Reframe Engine (`local_worker/openshorts_engine.py`, `local_worker/detection_service.py`, `local_worker/segmentation_service.py`, `local_worker/speaker_priority.py`, `local_worker/layout_decision.py`)**:
     - Face detection via OpenCV YuNet ONNX (`face_detection_yunet_2023mar.onnx`) with Haar cascade fallback.
     - Person & object detection via Ultralytics YOLOv8 (`yolov8n.pt`, optional `yolov8n-pose.pt`).
     - Optional video segmentation adapter for Meta AI SAM 2 (`local_worker/segmentation_service.py`).
     - Active-speaker attribution combining mouth-region pixel motion and 8 kHz audio RMS energy envelopes.
     - Camera smoothing via `SmoothedCameraman` (One-Euro filter, deadband hysteresis, 3-frame outlier jump rejection, and hard scene-cut snapping).
     - Automated layout scoring across `fill` (1-up 9:16), `split` (2-up vertical stack), `grid3` (3-up panel grid), `editorial` (context strip + subject), and `fit` (blurred background + full frame).
   - **Native FFmpeg Mastering Engine (`local_worker/rendering/`)**:
     - Hardware encoder probing (`h264_nvenc`, `h264_qsv`, `h264_amf`, `h264_videotoolbox`, fallback `libx264`).
     - Multi-pass segment rendering with `sendcmd` per-frame crop trajectories, two-pass EBU R128 `loudnorm` (`I=-16:TP=-2.0:LRA=11`) mixed once across layout seams, and `ffprobe` output quality verification.

---

## 2. Current Repository Structure (Pre-Migration)

```text
clipforge-main/
├── src/
│   ├── components/         # Studio.tsx, WhisperSettingsPanel.tsx, CaptionPanel.tsx, VyroWorkspace.tsx, YouTubeImport.tsx, ui/*
│   ├── integrations/       # Supabase client & auth middleware
│   ├── lib/                # render.ts, reframe.ts, subtitles.ts, local-worker.ts, backend-render.ts, render-overlays.ts, ai.functions.ts
│   ├── routes/             # __root.tsx, index.tsx
│   └── test/               # Vitest unit tests
├── local_worker/
│   ├── rendering/          # renderer.py, filters.py, overlays.py, subtitles.py, audio.py, encoders.py, validator.py
│   ├── tests/              # Python unit and integration test suites (17 test modules)
│   ├── main.py             # FastAPI worker entrypoint
│   ├── openshorts_engine.py# Core scene cut, speaker tracking & reframe engine
│   ├── detection_service.py# YOLOv8 adapter
│   ├── segmentation_service.py # Optional SAM 2 adapter
│   ├── whisper_engine.py   # faster-whisper adapter
│   └── gemini_pool.py      # 5-Key Gemini rotation pool
├── public/                 # Static assets, vision TFLite models, helper script
├── scripts/                # prepare-vision-assets.mjs
├── docs/                   # Setup, licensing, and architecture notes
├── package.json
└── vite.config.ts
```

---

## 3. Verified Working Functionality

- **Local Analysis Cache (`test_analysis_cache.py`)**: 23/23 tests passing (LRU eviction, UTF-8 byte cap, checkpoint invalidation, secret exclusion).
- **Layout Decision Engine (`test_layout_decision.py`)**: 29/29 tests passing (weighted scoring for `fill`/`split`/`fit`, overlap vetoes, hysteresis hold).
- **Speaker Priority & Attribution (`test_speaker_priority.py`)**: 40/40 tests passing (floor-holder locking, cutaway suppression, overlap handling).
- **SAM 2 Segmentation Service (`test_segmentation_service.py`)**: 38/38 tests passing (graceful degradation when SAM 2 weights are absent, mask bbox refinement).
- **YOLO Detection Service (`test_detection_service.py`)**: 33/33 tests passing (offline weight resolution, device fallback, candidate conversion).
- **Reframe & 5-Key Gemini Pool (`test_openshorts_and_gemini_pool.py`)**: 28/28 tests passing (SpeakerTracker cooldown, SmoothedCameraman outlier rejection, 2-up/3-up scene analysis, 5-key rotation).
- **Hardware Encoder Portability (`test_hardware_portability.py`)**: 17/17 tests passing (NVENC/QSV/AMF/CPU probing, bitrate ceilings, corrupt output rejection).
- **Framerate & Motion Mathematics (`test_render_framerate.py`, `test_render_motion.py`, `test_fit_preservation.py`, `test_edit_tuning.py`, `test_effects_toggle.py`, `test_phase1_whisper.py`)**: 63/63 tests passing.
- **Intelligence & Overlay HTTP API (`test_intelligence_api.py`)**: 15/15 tests passing after compatibility fixes.

---

## 4. Incomplete or Optional Functionality

1. **SAM 2 Full Video Propagation**: `local_worker/segmentation_service.py` implements an isolated optional predictor adapter (`SAM2_MODEL_PATH`). No SAM 2 `.pt` checkpoint is bundled in the repository to keep clone sizes fast; when unconfigured, the pipeline automatically uses YOLOv8 + YuNet bounding boxes.
2. **Remotion Integration**: Mentioned only in historical planning notes; actual preview and production rendering are implemented via the shared HTML5 Canvas 2D + native FFmpeg multi-pass filtergraph pipeline (`src/lib/render.ts` and `local_worker/rendering/renderer.py`).
3. **Campaign Compliance Workspace (`VyroWorkspace.tsx`)**: Performs local preflight validation (duration bounds, required hashtags, prohibited phrases, and SHA-256 export checksum verification); external platform submission is tracked manually by URL.

---

## 5. Technical Debt & Bugs Identified (and Resolved)

1. **FastAPI `APIRouter(lifespan=...)` Version Incompatibility**:
   - *Finding*: `local_worker/rendering/overlays.py` passed `lifespan=_lifespan` to `APIRouter()`, which raises `TypeError: APIRouter.__init__() got an unexpected keyword argument 'lifespan'` on FastAPI versions prior to 0.93.
   - *Resolution*: Added a `try/except TypeError` fallback in `local_worker/rendering/overlays.py` that registers startup/shutdown event handlers when running on older FastAPI installations.
2. **Pydantic v1 vs v2 `StrictFloat` Integer Rejection in `OverlayManifest`**:
   - *Finding*: `OverlayManifest.fps` was typed as `StrictFloat`, causing JSON payloads with integer `"fps": 30` to fail with HTTP 422 on Pydantic v1 instead of reaching `OverlayStore.create()` validation (HTTP 400).
   - *Resolution*: Updated `fps` to `StrictFloat | StrictInt` and added dual Pydantic v1/v2 config support so both `30` and `30.0` validate consistently.
3. **FFmpeg Subprocess Stdin Blocking**:
   - *Finding*: `local_worker/rendering/renderer.py` spawned `ffmpeg` via `subprocess.Popen` without `stdin=subprocess.DEVNULL` or `-nostdin`, allowing `ffmpeg` to wait on TTY stdin in non-interactive shells.
   - *Resolution*: Added `-nostdin` to `_base_args()` and `stdin=subprocess.DEVNULL` to `subprocess.Popen`.
4. **Legacy Third-Party Platform Coupling**:
   - *Finding*: `vite.config.ts` and `package.json` depended on external `@lovable.dev/*` packages and preview auth brokers, and `docs/repository-integration-architecture.md` contained hardcoded local `file:///D:/...` paths.
   - *Resolution*: Replaced with native Vite + `@tanstack/react-start/plugin/vite` configuration, standard `localStorage` auth storage, and repository-relative paths.

---

## 6. Security Findings

- **Secret Isolation**: No hardcoded API keys or credentials exist in source files. `.env` and `.env.*` (except `.env.example`) are excluded via `.gitignore`.
- **Path Traversal Protection**: `source_id` and `overlay_id` parameters are validated against strict regex/UUID patterns (`local_worker/main.py` and `local_worker/rendering/overlays.py`), preventing directory traversal outside `WORK_DIR`.
- **Authentication**: All mutating local worker routes require `verify_token` (`X-ClipForge-Token` or `Authorization: Bearer`), while `/health` remains accessible for readiness probes.

---

## 7. Migration Plan

1. Initialize a clean Git repository on `main` with `.gitignore`, `.editorconfig`, `.gitattributes`, `.env.example`, and `LICENSE`.
2. Organize modular architecture (`apps/`, `workers/`, `engines/`, `packages/contracts/`, `tests/`, `docs/`, `assets/`, `scripts/`) while preserving root-level compatibility so all existing scripts and tests run unmodified.
3. Apply frontend, backend, worker, video, vision, audio, and rendering improvements in atomic Conventional Commits.
4. Establish comprehensive QA test runners (`pyproject.toml`, `tests/`) and GitHub Actions workflows (`ci.yml`, `frontend.yml`, `backend.yml`, `tests.yml`, `security.yml`, `release.yml`).
5. Author complete documentation (`README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, `docs/*`) with Mermaid architecture diagrams and visual branding assets.
6. Verify all tracked files, tests, security checks, and generate `docs/audit/FINAL_VERIFICATION.md` and `docs/audit/MIGRATION_REPORT.md`.
