"""Face-safe and caption-safe dynamic Hook Pill placement (ported from OpenShorts `hook_placement.py`).

Instead of pinning the opening hook banner to a fixed vertical offset that can cover a speaker's
forehead or eyes in close-up shots or collide with the seam captions in 2-Up `split` layouts, this
module scans candidate vertical positions in `[TOP_MARGIN, BOTTOM_UI]`, avoids active caption bands
(`BOTTOM_CAPTIONS` for single-panel shots and `SEAM_CAPTIONS` for 2-Up split shots), expands detected
face boxes to include the forehead/crown and chin (`head_box`), and picks the vertical placement and
shrink scale (`1.0`, `0.85`, `0.72`) that keeps face overlap at or below `GOOD_ENOUGH` (`8%`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

TOP_MARGIN = 0.06
BOTTOM_UI = 0.80
LEGACY_TOP = 0.20
STEP = 0.005

# Normalized vertical intervals reserved for burned-in captions on 9:16 output
BOTTOM_CAPTIONS: tuple[float, float] = (0.70, 0.88)
SEAM_CAPTIONS: tuple[float, float] = (0.42, 0.58)

SHRINK_SCALES: tuple[float, ...] = (1.0, 0.85, 0.72)
GOOD_ENOUGH = 0.08
MOVE_PENALTY = 0.05
DEFAULT_WINDOW = 5.0


@dataclass(frozen=True)
class HookPlacementResult:
    """Resolved normalized and pixel placement for the opening hook pill."""

    top_norm: float
    top_px: int
    scale: float
    face_overlap: float
    avoided_seam: bool


def head_box(face: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    """Expand a brow-to-chin face detection `(x0, y0, x1, y1)` to cover the full head (hair/forehead + chin)."""
    x0, y0, x1, y1 = float(face[0]), float(face[1]), float(face[2]), float(face[3])
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return (
        max(0.0, x0 - 0.15 * w),
        max(0.0, y0 - 0.60 * h),
        min(1.0, x1 + 0.15 * w),
        min(1.0, y1 + 0.20 * h),
    )


def overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection area of two normalized `(x0, y0, x1, y1)` rectangles."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def face_cost(
    banner: tuple[float, float, float, float],
    faces: Iterable[tuple[float, float, float, float]],
) -> float:
    """Worst-case fraction of any detected head box covered by `banner`."""
    worst = 0.0
    for f in faces:
        hb = head_box(f)
        area = max(1e-6, (hb[2] - hb[0]) * (hb[3] - hb[1]))
        coverage = overlap_area(banner, hb) / area
        if coverage > worst:
            worst = coverage
    return worst


def caption_bands(
    layout_modes: Iterable[str] | str,
    caption_y: float | None = None,
) -> list[tuple[float, float]]:
    """Forbidden vertical intervals where captions are rendered during the hook window."""
    if isinstance(layout_modes, str):
        modes = {layout_modes.lower()}
    else:
        modes = {str(m).lower() for m in layout_modes if m}
    bands: list[tuple[float, float]] = []
    if "split" in modes:
        bands.append(SEAM_CAPTIONS)
    if not modes or (modes - {"split"}):
        if caption_y is not None and 0.05 < caption_y < 0.95:
            bands.append((max(0.0, caption_y - 0.09), min(1.0, caption_y + 0.09)))
        else:
            bands.append(BOTTOM_CAPTIONS)
    return bands


def _overlaps_band(y0: float, y1: float, bands: Iterable[tuple[float, float]]) -> bool:
    for b0, b1 in bands:
        if min(y1, b1) > max(y0, b0):
            return True
    return False


def choose_y(
    width_frac: float,
    height_frac: float,
    faces: list[tuple[float, float, float, float]],
    forbidden: list[tuple[float, float]],
    preferred_top: float = LEGACY_TOP,
) -> tuple[float, float]:
    """Scan `[TOP_MARGIN, BOTTOM_UI - height_frac]` for the lowest-cost hook top `y`."""
    x0 = max(0.0, 0.5 - width_frac / 2.0)
    x1 = min(1.0, 0.5 + width_frac / 2.0)
    upper = max(TOP_MARGIN, BOTTOM_UI - height_frac)

    best_y = max(TOP_MARGIN, min(upper, preferred_top))
    best_score = float("inf")
    best_overlap = 1.0

    steps = max(1, int(round((upper - TOP_MARGIN) / STEP)))
    for i in range(steps + 1):
        y0 = min(upper, TOP_MARGIN + i * STEP)
        y1 = y0 + height_frac
        if _overlaps_band(y0, y1, forbidden):
            continue
        banner = (x0, y0, x1, y1)
        ov = face_cost(banner, faces)
        score = ov + MOVE_PENALTY * abs(y0 - preferred_top)
        if score < best_score - 1e-9:
            best_score = score
            best_y = y0
            best_overlap = ov

    return round(best_y, 4), round(best_overlap, 4)


def best_scale(
    base_w: float,
    base_h: float,
    faces: list[tuple[float, float, float, float]],
    forbidden: list[tuple[float, float]],
    preferred_top: float = LEGACY_TOP,
) -> tuple[float, float, float]:
    """Try `SHRINK_SCALES` (`1.0`, `0.85`, `0.72`) until face overlap is `<= GOOD_ENOUGH` (`8%`)."""
    if not faces and not _overlaps_band(preferred_top, preferred_top + base_h, forbidden):
        return preferred_top, 1.0, 0.0

    chosen_y = preferred_top
    chosen_scale = 1.0
    chosen_ov = 1.0

    for scale in SHRINK_SCALES:
        w = base_w * scale
        h = base_h * scale
        y0, ov = choose_y(w, h, faces, forbidden, preferred_top=preferred_top)
        if ov < chosen_ov or chosen_scale == 1.0:
            chosen_y, chosen_scale, chosen_ov = y0, scale, ov
        if ov <= GOOD_ENOUGH:
            return y0, scale, ov

    return chosen_y, chosen_scale, chosen_ov


def place_hook_banner(
    faces: list[tuple[float, float, float, float]] | None = None,
    layout_modes: Iterable[str] | str = "fill",
    out_w: int = 1080,
    out_h: int = 1920,
    default_top_px: int = 145,
    box_w_px: int = 810,
    box_h_px: int = 140,
    caption_y: float | None = None,
) -> HookPlacementResult:
    """Compute pixel and normalized hook placement for a 9:16 frame."""
    preferred_top = max(TOP_MARGIN, min(BOTTOM_UI, default_top_px / max(1, out_h)))
    base_w = min(0.92, box_w_px / max(1, out_w))
    base_h = min(0.25, box_h_px / max(1, out_h))
    forbidden = caption_bands(layout_modes, caption_y=caption_y)
    face_list = list(faces or [])

    if not face_list and not _overlaps_band(preferred_top, preferred_top + base_h, forbidden):
        return HookPlacementResult(
            top_norm=round(preferred_top, 4),
            top_px=int(default_top_px),
            scale=1.0,
            face_overlap=0.0,
            avoided_seam="split" in ({layout_modes.lower()} if isinstance(layout_modes, str) else {str(m).lower() for m in layout_modes}),
        )

    # Check if the default position already has <= GOOD_ENOUGH face overlap and avoids caption bands
    default_banner = (0.5 - base_w / 2.0, preferred_top, 0.5 + base_w / 2.0, preferred_top + base_h)
    default_ov = face_cost(default_banner, face_list)
    if default_ov <= GOOD_ENOUGH and not _overlaps_band(preferred_top, preferred_top + base_h, forbidden):
        return HookPlacementResult(
            top_norm=round(preferred_top, 4),
            top_px=int(default_top_px),
            scale=1.0,
            face_overlap=round(default_ov, 4),
            avoided_seam=any(b == SEAM_CAPTIONS for b in forbidden),
        )

    top_norm, scale, ov = best_scale(base_w, base_h, face_list, forbidden, preferred_top=preferred_top)
    return HookPlacementResult(
        top_norm=round(top_norm, 4),
        top_px=int(round(top_norm * out_h)),
        scale=scale,
        face_overlap=round(ov, 4),
        avoided_seam=any(b == SEAM_CAPTIONS for b in forbidden),
    )


def extract_output_faces_from_track(
    track: list[Any],
    duration: float = DEFAULT_WINDOW,
) -> tuple[list[tuple[float, float, float, float]], set[str]]:
    """Derive approximate normalized 9:16 output face boxes during the opening hook window `[0, duration]`."""
    faces: list[tuple[float, float, float, float]] = []
    modes: set[str] = set()
    if not track:
        return faces, modes
    first = track[0]
    t0 = float(first.get("t", 0.0) if isinstance(first, dict) else getattr(first, "t", 0.0))
    for pt in track:
        pt_t = float(pt.get("t", 0.0) if isinstance(pt, dict) else getattr(pt, "t", 0.0))
        if pt_t - t0 > duration:
            break
        raw_mode = pt.get("mode") if isinstance(pt, dict) else getattr(pt, "mode", None)
        modes.add(str(raw_mode or "fill").lower())
        out_box = pt.get("outputFaceBox") if isinstance(pt, dict) else getattr(pt, "outputFaceBox", None)
        if out_box and len(out_box) == 4:
            faces.append((float(out_box[0]), float(out_box[1]), float(out_box[2]), float(out_box[3])))
    return faces, modes
