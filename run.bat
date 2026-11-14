@echo off
setlocal enabledelayedexpansion
title ClipForge Launcher
chcp 65001 >nul 2>&1

rem ============================================================
rem  ClipForge - One-Command Launcher & Process Manager
rem    1. Checks Node, web dependencies, Python, worker deps, FFmpeg
rem    2. Starts the local render worker on http://127.0.0.1:8765
rem    3. Starts the studio (npm run dev) and opens the browser
rem    4. Closes the worker again when the studio exits
rem
rem  Commands:
rem    run.bat           - Start the local worker and studio
rem    run.bat --check   - Check system dependencies without starting
rem    run.bat --stop    - Stop all running ClipForge background processes
rem    run.bat --restart - Stop running instances and restart fresh
rem ============================================================

cd /d "%~dp0"

set "WORKER_DIR=%~dp0worker_data"
set "CLIPFORGE_WORK_DIR=%WORKER_DIR%"
set "PORT=8765"
set "STUDIO_PORT=8080"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

rem ---- Handle --stop flag ------------------------------------
if /I "%~1"=="--stop" goto do_stop
if /I "%~1"=="--restart" (
  call :kill_processes
  echo  [ok] Cleaned previous instances. Restarting...
  echo.
)

set "CHECK_ONLY=0"
if /I "%~1"=="--check" set "CHECK_ONLY=1"

echo.
echo  ========================================
echo   ClipForge Production Launcher
echo  ========================================

rem ---- Node + web dependencies -------------------------------
where node >nul 2>&1
if errorlevel 1 (
  echo  [X] Node.js not found on PATH. Install Node 20 or newer, then run this again.
  exit /b 1
)
for /f "delims=" %%v in ('node -v 2^>nul') do echo  [ok] Node %%v

if exist "node_modules\" goto node_ready
if "%CHECK_ONLY%"=="1" (
  echo  [X] node_modules is missing - run: npm install
  goto deps_done
)
echo  [..] Installing web dependencies, this happens once.
call npm install
if errorlevel 1 (
  echo  [X] npm install failed.
  exit /b 1
)
:node_ready
echo  [ok] Web dependencies present
:deps_done

rem ---- Python + worker dependencies --------------------------
set "PYTHON_CMD="
if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
  echo  [ok] Python environment: .venv
) else if exist "%~dp0venv\Scripts\python.exe" (
  set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
  echo  [ok] Python environment: venv
) else (
  where python >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
  )
)

if "%PYTHON_CMD%"=="" (
  echo  [X] Python not found on PATH. Install Python 3.10+ from python.org and tick "Add python.exe to PATH".
  exit /b 1
)

%PYTHON_CMD% -V > "%TEMP%\clipforge_pyver.tmp" 2>&1
if errorlevel 1 (
  del "%TEMP%\clipforge_pyver.tmp" >nul 2>&1
  echo  [X] Python failed to execute. Ensure a valid Python 3.10+ installation is in PATH.
  exit /b 1
)
set "PYVER="
for /f "tokens=2" %%v in ('type "%TEMP%\clipforge_pyver.tmp"') do set "PYVER=%%v"
del "%TEMP%\clipforge_pyver.tmp" >nul 2>&1
if "%PYVER%"=="" (
  echo  [ok] Python present
) else (
  echo  [ok] Python %PYVER%
)

%PYTHON_CMD% -c "import fastapi, uvicorn, pydantic" >nul 2>&1
if not errorlevel 1 goto py_ready
if "%CHECK_ONLY%"=="1" (
  echo  [X] Worker packages missing - run: %PYTHON_CMD% -m pip install -r local_worker\requirements.txt
  goto py_ready
)
echo  [..] Installing worker packages, this happens once.
%PYTHON_CMD% -m pip install -r "local_worker\requirements.txt"
if errorlevel 1 (
  echo  [X] Worker package install failed.
  exit /b 1
)
:py_ready
echo  [ok] Worker packages present

rem ---- FFmpeg detection --------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
  if exist "C:\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=C:\ffmpeg\bin;!PATH!"
  )
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo  [X] FFmpeg not found on PATH - video exports cannot render natively. Install FFmpeg and reopen.
) else (
  for /f "tokens=3" %%v in ('ffmpeg -version 2^>^&1 ^| findstr /i /b "ffmpeg version"') do echo  [ok] FFmpeg %%v
)

if not exist "%WORKER_DIR%" mkdir "%WORKER_DIR%"
echo  [ok] Render folder: %WORKER_DIR%

rem ---- Detect curl vs powershell probe ------------------------
set "CURL_BIN="
where curl.exe >nul 2>&1
if not errorlevel 1 set "CURL_BIN=curl.exe"

rem ---- Is a worker already answering? ------------------------
set "WORKER_UP=0"
set "HEALTH_CODE=12"

if not "%CURL_BIN%"=="" (
  %CURL_BIN% -s -f --connect-timeout 2 -m 3 "http://127.0.0.1:%PORT%/health" > "%TEMP%\clipforge_health.tmp" 2>nul
  if not errorlevel 1 (
    findstr /i "nativeRenderer native_renderer" "%TEMP%\clipforge_health.tmp" >nul 2>&1
    if not errorlevel 1 (
      set "HEALTH_CODE=10"
    ) else (
      set "HEALTH_CODE=11"
    )
  ) else (
    set "HEALTH_CODE=12"
  )
  del "%TEMP%\clipforge_health.tmp" >nul 2>&1
) else (
  powershell -NoProfile -Command "try{$h=(Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').Content; if($h -match 'nativeRenderer|native_renderer'){exit 10}; exit 11}catch{exit 12}"
  set "HEALTH_CODE=!ERRORLEVEL!"
)

if "!HEALTH_CODE!"=="10" (
  set "WORKER_UP=1"
  echo  [ok] Worker already running on port %PORT% with native FFmpeg renderer
)
if "!HEALTH_CODE!"=="11" (
  set "WORKER_UP=1"
  echo  [*] Worker on port %PORT% answered without native renderer flag.
)
if "!HEALTH_CODE!"=="12" (
  echo  [..] No worker on port %PORT% currently answering
  rem Check if port is held by an unresponsive zombie process
  powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue; if ($conn) { $p = $conn.OwningProcess | Select-Object -Unique; foreach($id in $p){ Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }; exit 1 } else { exit 0 }" >nul 2>&1
  if "!ERRORLEVEL!"=="1" echo  [*] Freed stale socket holding port %PORT%
)

if "%CHECK_ONLY%"=="1" (
  echo  ----------------------------------------
  echo  Check complete: all pre-flight checks evaluated.
  echo  Run "run.bat" to start ClipForge Studio.
  exit /b 0
)

rem ---- Start the worker if needed ----------------------------
if "%WORKER_UP%"=="1" goto worker_ready

echo  [..] Starting ClipForge Local Worker on http://127.0.0.1:%PORT%...
if exist "%~dp0run_worker.bat" (
  start "" /D "%~dp0" "%~dp0run_worker.bat"
) else (
  start "ClipForge Local Worker" /D "%~dp0" "%PYTHON_CMD%" -m local_worker.main
)

set "TRIES=0"
:wait_worker
set /a TRIES+=1
if not "%CURL_BIN%"=="" (
  %CURL_BIN% -s -f --connect-timeout 1 "http://127.0.0.1:%PORT%/health" >nul 2>&1
  if not errorlevel 1 goto worker_ready
) else (
  powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health' | Out-Null; exit 0}catch{exit 1}"
  if not errorlevel 1 goto worker_ready
)

if !TRIES! LSS 60 (
  ping -n 2 127.0.0.1 >nul
  goto wait_worker
)
echo  [X] The worker did not answer within 60s. Read the "ClipForge Local Worker" window for errors.
exit /b 1

:worker_ready
echo  [ok] Worker ready on http://127.0.0.1:%PORT%
echo.

rem ---- Check if port 8080 is held by an old node process -----
powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %STUDIO_PORT% -ErrorAction SilentlyContinue; if ($conn) { exit 1 } else { exit 0 }" >nul 2>&1
if "!ERRORLEVEL!"=="1" (
  echo  [*] Port %STUDIO_PORT% is already occupied. Checking process...
  powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %STUDIO_PORT% -ErrorAction SilentlyContinue; foreach($p in ($conn.OwningProcess | Select-Object -Unique)){ $proc = Get-Process -Id $p -ErrorAction SilentlyContinue; if ($proc.ProcessName -match 'node') { Stop-Process -Id $p -Force; Write-Host ' [ok] Freed port ' + %STUDIO_PORT% + ' from stale node process PID ' + $p } }"
)

echo  Starting ClipForge Studio on http://localhost:%STUDIO_PORT%...
echo  The browser will open automatically once the Vite server is ready.
echo  Press Ctrl+C to stop the studio and worker.
echo.

rem ---- Start browser auto-opener in background ---------------
start "" /min powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%STUDIO_PORT%/'|Out-Null; Start-Process 'http://127.0.0.1:%STUDIO_PORT%/'; break}catch{Start-Sleep -Seconds 1}}"

rem ---- Start the studio -------------------------------------
call npm run dev

rem ---- Cleanup on Studio Exit -------------------------------
echo.
echo  Studio stopped. Stopping local worker...
call :kill_processes
echo  [ok] Goodbye!
exit /b 0

rem ============================================================
rem  Helper: Kill running worker and studio processes
rem ============================================================
:kill_processes
rem 1. Kill worker via PID file if present
if exist "%WORKER_DIR%\worker.pid" (
  for /f "delims=" %%p in ('type "%WORKER_DIR%\worker.pid" 2^>nul') do (
    powershell -NoProfile -Command "Stop-Process -Id %%p -Force -ErrorAction SilentlyContinue" >nul 2>&1
  )
  del /f /q "%WORKER_DIR%\worker.pid" >nul 2>&1
)
rem 2. Kill any python running local_worker.main
powershell -NoProfile -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*local_worker.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
rem 3. Kill worker window if orphaned
powershell -NoProfile -Command "Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '*ClipForge Local Worker*' } | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }" >nul 2>&1
exit /b 0

:do_stop
echo.
echo  Stopping ClipForge local worker and studio processes...
call :kill_processes
echo  [ok] ClipForge worker and studio processes stopped.
exit /b 0
