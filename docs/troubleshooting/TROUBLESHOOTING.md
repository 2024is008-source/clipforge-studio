# Troubleshooting Guide

## 1. Port 8765 or 8080 Already in Use
Run `run.bat --restart` on Windows, or terminate stale processes listening on `:8765` / `:8080`.

## 2. CUDA Out of Memory or Missing cuBLAS/cuDNN DLLs
`local_worker/whisper_engine.py` automatically catches CUDA initialization errors and falls back to CPU `int8` inference without aborting the job. You can also force CPU mode in the Settings UI or via `WHISPER_DEVICE=cpu`.

## 3. FFmpeg Not Found
Install FFmpeg 7.0+ and ensure `ffmpeg` and `ffprobe` are available on your system `PATH`.
