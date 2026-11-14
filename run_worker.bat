@echo off
setlocal enabledelayedexpansion
title ClipForge Local Worker
chcp 65001 >nul 2>&1

cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "WORKER_DIR=%~dp0worker_data"
set "CLIPFORGE_WORK_DIR=%WORKER_DIR%"
if not exist "%WORKER_DIR%\models_cache\yolov8n.pt" (
  if exist "%TEMP%\clipforge_local_worker\models_cache\yolov8n.pt" (
    if not exist "%WORKER_DIR%\models_cache" mkdir "%WORKER_DIR%\models_cache"
    copy /y "%TEMP%\clipforge_local_worker\models_cache\yolov8n.pt" "%WORKER_DIR%\models_cache\yolov8n.pt" >nul 2>&1
  )
)

set "PYTHON_CMD="
if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
  set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
) else (
  set "PYTHON_CMD=python"
)

echo ============================================================
echo  ClipForge Local Render Worker
echo ============================================================
echo.

"%PYTHON_CMD%" -m local_worker.main
if errorlevel 1 (
  echo.
  echo [X] Worker exited with error code !ERRORLEVEL!.
  pause
)

