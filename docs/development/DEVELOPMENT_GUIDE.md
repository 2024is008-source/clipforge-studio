# ClipForge Development Guide

## Local Development Workflow

1. **Install dependencies**:
   ```bash
   npm install
   pip install -r local_worker/requirements.txt
   ```
2. **Stage MediaPipe Vision Runtime**:
   ```bash
   npm run prepare
   ```
3. **Run Worker & Studio**:
   - Terminal 1: `python -m local_worker.main`
   - Terminal 2: `npm run dev`
4. **Pre-Commit Verification**:
   - `npx tsc --noEmit`
   - `npm run lint`
   - `npm test`
   - `python -m unittest discover -s tests -p "test_*.py" -v`
