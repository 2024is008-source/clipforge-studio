# ClipForge Native FFmpeg Rendering Pipeline

## Multi-Pass Layout Window Architecture

1. **Layout Segmentation (`layout_segments`)**: Splits a clip into continuous layout spans (`fill`, `split`, `grid3`, `editorial`, `fit`) while enforcing minimum hold durations (`enforce_layout_hold`).
2. **Silent Window Rendering**: When a clip contains multiple layout segments, individual windows are rendered without audio (`-an`) so `loudnorm` and compressor states never reset at layout cuts.
3. **Full-Clip Audio Concatenation (`_concat_pass`)**: Concatenates all video segments and mixes the source audio once across the entire clip duration using EBU R128 loudness normalization (`I=-16:TP=-2.0:LRA=11`).
4. **Canvas RGBA Overlay Burn (`_overlay_pass`)**: Composites transparent 1080×1920 PNG sequences rendered by the browser's HTML5 Canvas (`src/lib/render-overlays.ts`) so exported typography and emoji animations match the preview pixel-for-pixel.
