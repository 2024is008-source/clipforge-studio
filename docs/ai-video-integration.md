# AI video integration plan and verification ledger

## Scope and architecture

This plan implements the supplied Ultralytics / SAM 2 / ClipForge EDL brief incrementally, without replacing ClipForge's editor, transcript, speaker registry, or timeline. The current brief supersedes the older browser-capture-only recommendations in repository-integration-architecture.md: final composition belongs to the existing local FFmpeg worker, not the hosted AI server. No database, gateway, secret, or generated-file changes are needed.

Audit findings: the worker already has face/body fallback, scene sampling, speaker attribution, camera smoothing, per-output-frame crop commands, blur-fit and split rendering. It currently calls YOLO twice on frames without people, loses detector confidence in candidate conversion, and discards object context whenever faces exist. Separate post-processors can rewrite layouts. Analysis results are not cached. SAM currently means optional single-image inference inside the ordinary person loop, not video propagation. Native caption rendering uses ASS and does not yet reproduce all canvas animation/overlays. These are gaps, not delivered capabilities.

## Implementation order

1. Isolate local-checkpoint YOLO inference in a worker-only service. Reuse models under a lock; return normalized XYWH boxes, classes, confidence and optional pose keypoints. Cache detections by pixels, checkpoint identity and settings with bounded storage. Preserve the existing speaker/identity tracker; do not stack a second tracker on the same people. Call object detection once per sampled frame and carry object context even when faces exist. Make sampling cadence configurable. Never download models during tests.
2. Add one final layout decision stage after speaker attribution and trajectory smoothing. Use actual renderer crop geometry to score fill / split / fit with visibility 25%, preservation 25%, confidence 15%, balance 15%, editorial relevance 10%, resolution 10%. Hard safety constraints override score and hold: uncertain tracking, clipped required content, and duplicate/overlapping panels fall back to fit. Reuse scene boundaries and hold durations; record component scores and reasons. Preserve explicit manual fill/fit and existing advanced layouts; manual split still cannot duplicate content.
3. Replace implicit per-person SAM inference with an explicitly requested, bounded video-segmentation service using the supplied SAM 2 API. Sample short segments into capped-size frames, use selected point/box prompts, propagate masks, keep timestamps, cache mask products and release predictor state. Missing package/checkpoint or insufficient resources must be reported; ordinary exports continue without masks. No fake segmentation or claims that boxes are accurate masks.
4. Extend the existing ClipJob-to-RenderDecision contract rather than add a competing EDL. Carry decision provenance and deterministic caption page boundaries. Preserve current caption visuals; planned alpha-overlay transport must be finished and visually compared before replacing the existing caption path. Backend-only export must not silently fall back to browser capture. Preview must consume the same finalized layout and caption decisions.
5. Verify unit and integration behavior, then exercise the UI with the supplied full source, Auto enabled and Hook disabled. Inspect actual MP4 frames, timing, audio and frame rate. Produce before/after artifacts and measured timings. Keep unsupported categories and unavailable model paths explicitly unverified.

## Reference-source findings

- Ultralytics Model.predict/track and tracker callbacks: shared model tracking state requires clip-level ownership, not just frame locks. The supplied BYTETracker advances one step per update; its buffer is in sampled updates, not source frames. Existing ClipForge identities will be retained in this increment rather than create conflicting ID systems.
- SAM 2 build_sam2_video_predictor / init_state / add_new_points_or_box / propagate_in_video: init_state loads video into memory. CPU offload is not a memory bound. Therefore only bounded, resized frame sets may enter it. Box prompts are pixel XYXY with normalize_coords=True. Output logits are thresholded at zero, paired with original source timestamps.
- ClipForge EDL EditCommand and timelineToEdl: validate serializable decisions once and keep source/program time explicit. renderEdl composes trim, effects and overlays from that decision. cropForAspect uses even dimensions. transcriptToCues and captionFrames separate word timing, grouping and rasterization. Its renderer is not transplanted; its center crop is less capable than ClipForge's current tracked geometry.

## Licensing and deployment

Ultralytics source and advertised models use AGPL-3.0 or a separately acquired Enterprise license. Commercial use is not prohibited by AGPL, but covered distributions/network deployments have source and license obligations. A proprietary integration needs a reviewed compliance strategy or suitable commercial license. A separate worker process is an engineering boundary, not an automatic copyleft exemption. No vendor source or new weights are bundled by this work.

The supplied SAM 2 README explicitly licenses both code and checkpoints under Apache-2.0; the optional connected-components implementation carries BSD-3-Clause terms in LICENSE_cctorch. Preserve licenses/notices on redistribution. No checkpoint binary was supplied in that ZIP and no SAM 2 model inference is verified yet.

ClipForge EDL is MIT with a NOTICE file. Only architecture patterns are adopted; retain the original copyright and permission notice if substantial source is copied. FFmpeg compliance depends on the installed binary's actual build options, not on invoking it as a subprocess. This is an engineering license review, not legal clearance.

## Verification ledger at plan creation

- Full supplied source ingest, discovery and an earlier browser-mastered export were exercised in the preceding live test. This is not verification of the planned native integration.
- Sequential sampling previously measured 0.57 seconds versus 4.04 seconds for 48 samples; the complete old/new tracking comparison failed from process memory exhaustion. Output parity is NOT established.
- The 13-category real-video matrix is not available in the supplied media; no category-wide visual acceptance is claimed.
- SAM 2 and MediaPipe Python packages were absent at audit time; no package installation or model download is implicit in this plan.

## Implemented services and setup

- `local_worker/detection_service.py`: local `.pt` checkpoint only, serialized reused models, bounded 128-frame result cache, explicit optional pose, normalized detections and reported CPU retry. Configure `YOLO_MODEL_PATH`, optional `YOLO_POSE_MODEL_PATH`, `YOLO_DEVICE=auto|cpu|cuda`, `YOLO_IMAGE_SIZE` (default 640), `YOLO_CONFIDENCE` (default 0.4). Without a path, an existing `yolov8n.pt` in MODELS_DIR or the working directory is used; this service never downloads it.
- Analysis sampling uses `REFRAME_SAMPLE_FPS` (default 7.5, bounded 1–15). The final safety policy accepts `LAYOUT_CONFIDENCE`, `LAYOUT_VISIBILITY`, `LAYOUT_MAX_PANEL_OVERLAP`, `LAYOUT_DETECTION_SHARE`, and `LAYOUT_SWITCH_MARGIN`; defaults are 0.45, 0.98, 0.15, 0.65, and 0.08. Object IDs use the existing association algorithm in separate class-specific namespaces; they survive short gaps but are not appearance re-identification across cuts.
- `local_worker/segmentation_service.py`: original SAM 2 package plus local checkpoint required. Configure `SAM2_MODEL_PATH`, `SAM2_CONFIG` (the installed package's Hydra config name), and `SAM2_DEVICE=auto|cpu|cuda`. Maximum 10 seconds, 32 frames, 512-pixel long edge, sampled at 3 FPS. Explicit `/v1/segmentation/jobs` accepts cached sourceId and normalized point/box prompts; job status and cancellation are available under that job ID. Ordinary analysis/export never runs SAM. Missing setup returns `available:false`, not fabricated masks. Real checkpoint inference and VFR mask alignment remain unverified.
- Analysis supports asynchronous per-job results and cancellation, without relying on the global progress pointer. The cached source's modification timestamp is no longer touched just to access it. Complete source/results are addressed by safe IDs rather than browser-supplied filesystem paths.
- BLUR+FIT contains the entire source in both renderers, including portrait, square and ultrawide sources. Native fit is one source-decode pass instead of a cropped foreground intermediate.

## Executed checks during integration

- Full worker discovery suite: 230 tests executed, one skipped, no failures at this checkpoint. Includes mocked model boundaries, 38 segmentation tests, queued API checks, layout safety and existing speaker/render regressions. Later overlay changes require a fresh final run.
- Frontend type check currently fails, including pre-existing AIClip fixture/default and strict-optional errors; this is not a passing build claim.
- The full 796,149,590-byte source was loaded again in the browser, Hook disabled through its switch, Auto retained, and six-clip discovery reached tracking. Real inference reported YuNet and YOLO ready. Native MP4 acceptance is still pending.
- A live polling race was found: completion could be published before the result payload. Result and terminal status are now published together. The running worker must be restarted before retesting this change.
- No model packages/checkpoints were installed or downloaded by this integration work. The cached YOLO and YuNet weights were already present.

Caption-overlay integration, final source-rate alignment, actual MP4 visual comparisons, and the requested broader media matrix remain pending verification.
