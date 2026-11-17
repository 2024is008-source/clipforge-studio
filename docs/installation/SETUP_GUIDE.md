# ClipForge — Complete Setup & Deployment Guide

This guide provides explicit, isolated installation instructions for CPU, NVIDIA GPU, optional SAM 2 segmentation, model checkpoints, development, and production environments.

---

## 1. System Requirements

### Hardware
- **CPU**: x86_64 quad-core processor or higher (Intel Core i5/i7/i9 or AMD Ryzen).
- **GPU (Recommended)**: NVIDIA GPU with Turing, Ampere, Ada, or Blackwell architecture (minimum 4 GB VRAM, e.g. GeForce RTX 2050 / 3050 / 3060 / 4060).
- **RAM**: Minimum 8 GB system RAM (16 GB recommended for multi-tasking).
- **Storage**: Minimum 10 GB free space for model weights and temporary render buffers.

### Core Binaries
- **Python**: 3.10 to 3.14 (3.11/3.12 recommended for full PyTorch ecosystem compatibility).
- **FFmpeg**: Version 7.0+ (Version 9.0 full-build recommended, compiled with `--enable-libass`, `--enable-libx264`, and `--enable-nvenc`).
- **Node.js**: Version 20.0+ (or Bun 1.2+).

---

## 2. Environment Setup Configurations

### A. Standard CPU Installation

For machines without an NVIDIA GPU or running in standard cloud containers:

```bash
# 1. Clone repository
git clone https://github.com/2024is008-source/clipforge.git
cd clipforge

# 2. Create Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install core dependencies (CPU PyTorch)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r local_worker/requirements.txt

# 4. Set environment configuration for CPU
export CLIPFORGE_WORKER_HOST=127.0.0.1
export CLIPFORGE_WORKER_PORT=8765
export YOLO_DEVICE=cpu
export WHISPER_DEVICE=cpu
```

---

### B. NVIDIA GPU Installation (CUDA 12.x)

For systems with NVIDIA hardware acceleration:

```bash
# 1. Verify NVIDIA Driver & CUDA
nvidia-smi

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install CUDA-accelerated PyTorch
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install worker dependencies
pip install -r local_worker/requirements.txt

# 5. Configure environment variables for GPU
export YOLO_DEVICE=cuda
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE_TYPE=int8_float16
```

---

### C. Optional SAM 2 Video Segmentation Setup

SAM 2 is an **optional** enhancement for precise pixel masking and object isolation. Ordinary video editing and exports function completely without it.

#### Requirements
- Python 3.10–3.12 (SAM 2 requires PyTorch with CUDA support).
- PyTorch 2.3+ with torchvision.
- CUDA compute capability >= 7.0.

#### Installation Steps:

```bash
# 1. Clone official SAM 2 repository
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# 2. Install SAM 2 package in your active virtual environment
pip install -e .

# 3. Download SAM 2 model weights into models directory
mkdir -p checkpoints
cd checkpoints
# Option 1: SAM 2.1 Hiera Tiny (Fastest, ~150 MB)
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
# Option 2: SAM 2.1 Hiera Small (~180 MB)
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
cd ../..

# 4. Configure ClipForge environment variables
export SAM2_MODEL_PATH=/path/to/checkpoints/sam2.1_hiera_tiny.pt
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export SAM2_DEVICE=cuda
```

*Note: If `SAM2_MODEL_PATH` or the `sam2` package is missing, ClipForge automatically skips segmentation and performs standard reframing without errors.*

---

### D. Model Checkpoint Setup

ClipForge uses local-only model checkpoints stored in `MODELS_DIR` (`%TEMP%/clipforge_local_worker/models_cache` or custom path via `CLIPFORGE_WORK_DIR`):

1. **YOLOv8 Object Detector**:
   - Standard: `yolov8n.pt` (Nano, ~6.2 MB)
   - Pose (Optional): `yolov8n-pose.pt` (~6.5 MB)
   - Path configuration: `YOLO_MODEL_PATH=/path/to/yolov8n.pt`

2. **OpenCV YuNet Face Detector**:
   - `face_detection_yunet_2023mar.onnx` (~380 KB)
   - Stored in `MODELS_DIR`.

3. **Faster-Whisper Speech Recognition**:
   - `Systran/faster-whisper-small` (~484 MB)
   - Downloaded once by CTranslate2 and cached permanently in `MODELS_DIR`.

---

### E. Frontend Setup & Development

```bash
# 1. Install Node.js dependencies
npm install
# or with bun:
bun install

# 2. Run TypeScript check
npm run typecheck   # or: npx tsc --noEmit

# 3. Run frontend unit tests
npm run test        # or: npx vitest run

# 4. Start local development servers
# Terminal 1 - Local Python Worker:
python -m local_worker.main

# Terminal 2 - Frontend Development Server:
npm run dev
```

---

### F. Production Deployment

1. **Local Worker Daemon**:
   Run the local worker behind an init system (systemd, Windows Service, or PM2):
   ```bash
   uvicorn local_worker.main:app --host 127.0.0.1 --port 8765 --workers 1
   ```

2. **Frontend Production Build**:
   ```bash
   npm run build
   # Outputs prebuilt production artifacts in .output/
   ```

3. **Production Run**:
   ```bash
   npx vite preview
   # or deploy prebuilt bundle to Cloudflare Pages / Workers / Nitro host
   ```

