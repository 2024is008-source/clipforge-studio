# AI Model Integrations

## 1. Speech Recognition (`local_worker/whisper_engine.py`)
- Uses `faster-whisper` with automatic CUDA `int8_float16` to CPU `int8` fallback.
- Enforces monotonic word timestamps and merges sub-word BPE continuation tokens (`local_worker/transcript_validator.py`).

## 2. Face, Person & Object Detection (`local_worker/detection_service.py`)
- **OpenCV YuNet ONNX**: Fast face bounding-box and landmark detection.
- **Ultralytics YOLOv8**: Person (`class 0`) and context object detection (`yolov8n.pt`).

## 3. Optional SAM 2 Segmentation (`local_worker/segmentation_service.py`)
- Configured via `SAM2_MODEL_PATH` and `SAM2_CONFIG`.
- Refines subject silhouettes and bounding boxes when available; cleanly bypassed when unconfigured.

## 4. 5-Key Gemini Free Tier Pool (`local_worker/gemini_pool.py` & `src/lib/ai.functions.ts`)
- Rotates across up to 5 API keys (`GEMINI_API_KEY_1`..`5`) with per-key 60-second cooldowns on HTTP 429 rate limits.
