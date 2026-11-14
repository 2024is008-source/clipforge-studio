# ClipForge Studio — Architecture & Maintainer Guide

**Creator & Lead Developer:** Minuja Ranthika Amarasekara (`2024is008-source`)  
**Repository:** `https://github.com/2024is008-source/clipforge.git`

## Core Architecture
- **Local GPU Worker (`local_worker/`)**: Runs FastAPI on `http://127.0.0.1:8765` for local `faster-whisper` speech transcription with word-level timestamps, YOLOv8 + OpenCV YuNet face/person/object tracking, optional SAM 2 segmentation, 5-key Gemini Free Tier rotation pool, and native FFmpeg NVENC/CPU mastering.
- **Studio Frontend (`src/`)**: TanStack Start + React 19 + Tailwind CSS v4 studio interface providing 1-click clip discovery, real-time 9:16 canvas preview, ClipForge Caption Engine presets, and campaign preflight checks.
- **Unified Overlay & Caption Pipeline (`src/lib/subtitles.ts`, `src/lib/render.ts`, `src/lib/render-overlays.ts`)**: Word-level animation timing, phrase-aware pagination, and transparent PNG overlay sequences shared between the interactive browser preview and the native FFmpeg export pipeline.
- **Environment & Secrets**: Store API keys (`GEMINI_API_KEY_1`..`5`, `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`) in `.env` locally; never commit secret values to the repository.
