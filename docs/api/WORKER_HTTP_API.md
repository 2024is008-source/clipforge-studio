# Local Worker HTTP API Reference (`http://127.0.0.1:8765`)

All non-health routes accept `X-ClipForge-Token` or `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns worker readiness, GPU/encoder capabilities, and Whisper status. |
| `GET` | `/v1/system/info` | Returns detailed hardware telemetry, VRAM, and measured encoder profiles. |
| `POST` | `/v1/whisper/transcribe` | Queues or executes local `faster-whisper` word-level transcription. |
| `POST` | `/v1/openshorts/analyze` | Queues Reframe Engine v2 analysis (`fill`, `split`, `grid3`, `editorial`, `fit`). |
| `GET` | `/v1/openshorts/analyze/jobs/{job_id}` | Polls analysis progress and returns `track` + `layout_decisions`. |
| `POST` | `/v1/render/overlays` | Initializes a bounded RGBA PNG overlay upload session. |
| `POST` | `/v1/render/overlays/{id}/frames` | Uploads a batch of sequential transparent PNG overlay frames. |
| `POST` | `/v1/render/decisions` | Queues a native FFmpeg render job from a validated `RenderDecision`. |
