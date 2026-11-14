@echo off
title ClipForge Studio Local Worker (Faster-Whisper + FFmpeg)
echo ============================================================
echo   ClipForge Studio — Local Whisper & GPU Render Worker
echo   Optimized for NVIDIA RTX 2050 (CUDA int8_float16) & CPU
echo ============================================================
python -m pip install -U faster-whisper ctranslate2 fastapi "uvicorn[standard]" python-multipart pydantic "yt-dlp[default]"
python -m local_worker.main
pause
