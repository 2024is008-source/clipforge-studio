"""Frame-level motion and pacing diagnostics for a finished render.

Everything here is measured from decoded pixels and packet timestamps, because "29.97 fps" in the
container header says nothing about whether the frames arrived evenly or whether the camera path
actually moved between them.

Two questions get answered separately, because they have different causes:

* pacing  - are the frames evenly spaced, and how many are repeats? (capture/encode problem)
* motion  - when the framing moves, does it glide or does it jump? (crop path problem)

A held crop that then snaps shows up as a frame with a large global displacement surrounded by
frames with none, so the reported jerk and step ratio distinguish it from real source motion.

    python -m local_worker.rendering.diagnostics <file.mp4> [--source src.mp4 --start 10 --end 63]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

# Analysis is done on a downscaled grey image: global camera displacement survives it, and 53 s of
# 1080x1920 would otherwise be 8 GB of pixels held in RAM.
PROBE_WIDTH, PROBE_HEIGHT = 270, 480
FINE_RANGE, FINE_STEP = 4, 1
COARSE_STRIDE, FINE_STRIDE = 4, 2
# Widening passes: a crop that teleports the framing can move further in one frame than the first
# window covers, and a clipped reading would look like a smaller, calmer move than the defect is.
COARSE_PASSES = ((16, 4), (48, 6), (96, 8))
EXACT_HOLD_DIFF = 0.02                # sampled luma units; a true repeat of the previous frame
STATIC_DIFF = 0.35                    # below this the pair reads as "nothing moved" to the eye
STEP_PX = 2.0                         # probe px; one frame carrying this much displacement is a step
HOLD_FRAMES = 2                       # still frames before a step that make it a hold-then-jump


@dataclass
class FrameReport:
    path: str
    frames: int = 0
    duration: float = 0.0
    measured_fps: float = 0.0
    exact_duplicates: int = 0
    static_pairs: int = 0
    longest_freeze_sec: float = 0.0
    pts_nominal_gap: float = 0.0
    pts_min_gap: float = 0.0
    pts_max_gap: float = 0.0
    pts_gap_stdev: float = 0.0
    pts_uneven_gaps: int = 0
    shifts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    costs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    diff_series: list = field(default_factory=list)

    def as_dict(self) -> dict:
        travel = np.hypot(self.shifts[:, 0], self.shifts[:, 1]) if len(self.shifts) else np.zeros(0)
        # A step only counts as framing motion when a pure translation explains the frame pair;
        # otherwise it is a cut or real subject motion, which the renderer is not responsible for.
        floor = float(np.median(self.costs)) if len(self.costs) else 0.0
        clean = self.costs <= max(2.0, floor * 6.0) if len(self.costs) else np.ones_like(travel, dtype=bool)
        big = (travel >= STEP_PX) & clean
        cut_like = int(((travel >= STEP_PX) & ~clean).sum())
        # jerk = frame-to-frame change in displacement: what the eye reads as a stumble.
        jerk = np.abs(np.diff(travel)) if len(travel) > 1 else np.zeros(0)
        hold_then_jump = 0
        run = 0
        for i, t in enumerate(travel):
            if t < 0.35:
                run += 1
                continue
            if t >= STEP_PX and clean[i] and run >= HOLD_FRAMES:
                hold_then_jump += 1
            run = 0
        return {
            "path": self.path,
            "frames": self.frames,
            "duration_sec": round(self.duration, 3),
            "measured_fps": round(self.measured_fps, 3),
            "exact_duplicate_frames": self.exact_duplicates,
            "exact_duplicate_pct": round(100.0 * self.exact_duplicates / max(1, self.frames - 1), 2),
            "static_frame_pct": round(100.0 * self.static_pairs / max(1, self.frames - 1), 2),
            "longest_frozen_sec": round(self.longest_freeze_sec, 3),
            "pts_nominal_gap": round(self.pts_nominal_gap, 5),
            "pts_min_gap": round(self.pts_min_gap, 5),
            "pts_max_gap": round(self.pts_max_gap, 5),
            "pts_gap_stdev": round(self.pts_gap_stdev, 6),
            "pts_uneven_gaps": self.pts_uneven_gaps,
            "motion_steps": int(big.sum()),
            "motion_steps_per_sec": round(float(big.sum()) / self.duration, 2) if self.duration else 0.0,
            "displacement_events_excluded_as_cuts": cut_like,
            "hold_then_jump_events": hold_then_jump,
            "camera_travel_px": round(float(travel[clean].sum()), 1),
            "mean_frame_shift_px": round(float(travel[clean].mean()), 3) if clean.any() else 0.0,
            "p95_frame_shift_px": round(float(np.percentile(travel[clean], 95)), 2) if clean.any() else 0.0,
            "max_frame_shift_px": round(float(travel[clean].max()), 2) if clean.any() else 0.0,
            "mean_jerk_px": round(float(jerk.mean()), 3) if len(jerk) else 0.0,
            "p95_jerk_px": round(float(np.percentile(jerk, 95)), 3) if len(jerk) else 0.0,
            "max_jerk_px": round(float(jerk.max()), 3) if len(jerk) else 0.0,
            # How much of the total travel happens in step frames rather than spread over time.
            "step_travel_ratio": round(float(travel[big].sum() / travel[clean].sum()), 3)
            if big.any() and travel[clean].sum() else 0.0,
            "median_frame_diff": round(float(np.median(self.diff_series)), 3) if self.diff_series else 0.0,
            "median_translation_residual": round(floor, 3),
        }


def _run(args: list[str], timeout: int) -> str:
    out = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return out.stdout.decode("utf-8", "replace")


def _packet_times(path: str) -> list[float]:
    raw = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", path],
        240,
    )
    times = []
    for line in raw.splitlines():
        try:
            times.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    # Packets come back in decode order; presentation order is the one that can be differenced.
    return sorted(times)


def _video_stream(path: str) -> dict:
    raw = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,width,height",
         "-of", "json", path],
        60,
    )
    streams = json.loads(raw or "{}").get("streams", [])
    return streams[0] if streams else {}


def _rate(value: str) -> float:
    try:
        num, _, den = value.partition("/")
        n, d = float(num), float(den or 1)
    except (TypeError, ValueError):
        return 0.0
    return n / d if d else 0.0


def _frames_gray(path: str):
    """Yield (frame, digest, sampled-diff baseline) as float32 grey planes, streaming the decode."""
    args = [
        "ffmpeg", "-v", "error", "-i", path, "-an", "-vf",
        f"scale={PROBE_WIDTH}:{PROBE_HEIGHT}:force_original_aspect_ratio=disable,format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = PROBE_WIDTH * PROBE_HEIGHT
    try:
        while True:
            buf = proc.stdout.read(n)
            if not buf or len(buf) < n:
                break
            plane = np.frombuffer(buf, dtype=np.uint8).reshape(PROBE_HEIGHT, PROBE_WIDTH).astype(np.float32)
            yield plane, hashlib.md5(buf).hexdigest()
    finally:
        proc.stdout.close()
        proc.wait(timeout=180)


def _diff_at_offset(prev: np.ndarray, cur: np.ndarray, dy: int, dx: int, stride: int) -> float:
    """Mean absolute luma difference between `cur` and `prev` shifted by (dy, dx)."""
    h, w = prev.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    a = prev[ys0:ys1:stride, xs0:xs1:stride]
    b = cur[ys0 - dy : ys1 - dy : stride, xs0 - dx : xs1 - dx : stride]
    if a.size == 0:
        return math.inf
    return float(np.abs(a - b).mean())


def _search(prev: np.ndarray, cur: np.ndarray, cands: list[tuple[int, int]], stride: int) -> tuple[float, int, int]:
    best = (math.inf, 0, 0)
    for dy, dx in cands:
        cost = _diff_at_offset(prev, cur, dy, dx, stride)
        if cost < best[0]:
            best = (cost, dy, dx)
    return best


def _global_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """Coarse-to-fine global displacement (dx, dy) in probe px, plus the residual difference.

    The residual is what a pure translation cannot explain: it is large for a cut or for genuinely
    moving subjects, small for a reframe of an otherwise held shot. Reporting it is what keeps a
    scene change from being counted as a crop jump.
    """
    cost = math.inf
    cy = cx = 0
    for rng, step in COARSE_PASSES:
        cands = [(dy, dx) for dy in range(-rng, rng + 1, step) for dx in range(-rng, rng + 1, step)]
        cost, ny, nx = _search(prev, cur, cands, COARSE_STRIDE)
        cy, cx = ny, nx
        if abs(ny) < rng and abs(nx) < rng:
            break
    fine = [(dy, dx) for dy in range(cy - FINE_RANGE, cy + FINE_RANGE + 1, FINE_STEP)
            for dx in range(cx - FINE_RANGE, cx + FINE_RANGE + 1, FINE_STEP)]
    cost, cy, cx = _search(prev, cur, fine, FINE_STRIDE)
    # Sub-pixel from the parabola through the neighbours of the integer minimum.
    sx = sy = 0.0
    left = _diff_at_offset(prev, cur, cy, cx - 1, FINE_STRIDE)
    right = _diff_at_offset(prev, cur, cy, cx + 1, FINE_STRIDE)
    denom = left - 2 * cost + right
    if denom > 1e-6:
        off = 0.5 * (left - right) / denom
        sx = off if abs(off) <= 1.0 else 0.0
    up = _diff_at_offset(prev, cur, cy - 1, cx, FINE_STRIDE)
    down = _diff_at_offset(prev, cur, cy + 1, cx, FINE_STRIDE)
    denom = up - 2 * cost + down
    if denom > 1e-6:
        off = 0.5 * (up - down) / denom
        sy = off if abs(off) <= 1.0 else 0.0
    return cx + sx, cy + sy, cost


def analyse(path: str) -> FrameReport:
    # A missing or unreadable file once reported every metric as 0, which reads as a clean video.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no video to measure at {path}")
    rep = FrameReport(path=path)
    stream = _video_stream(path)
    if not stream:
        raise ValueError(f"ffprobe found no video stream in {path}; there is nothing to measure")
    times = _packet_times(path)
    nominal = _rate(stream.get("r_frame_rate") or "0") or _rate(stream.get("avg_frame_rate") or "0")
    if len(times) >= 2:
        gaps = np.diff(np.asarray(times, dtype=np.float64))
        nominal = nominal or 1.0 / float(np.median(gaps))
        rep.pts_nominal_gap = 1.0 / nominal if nominal else 0.0
        rep.pts_min_gap = float(gaps.min())
        rep.pts_max_gap = float(gaps.max())
        rep.pts_gap_stdev = float(gaps.std())
        rep.pts_uneven_gaps = int(((gaps > rep.pts_nominal_gap * 1.5) | (gaps < rep.pts_nominal_gap * 0.6)).sum())
        rep.duration = times[-1] - times[0] + (rep.pts_nominal_gap or float(gaps.mean()))
        rep.measured_fps = (len(times) - 1) / max(1e-6, times[-1] - times[0])

    prev: np.ndarray | None = None
    prev_hash: str | None = None
    run_start_idx = None
    index = 0
    interval = rep.pts_nominal_gap or (1.0 / nominal if nominal else 1 / 30)
    shifts: list[tuple[float, float]] = []
    costs: list[float] = []
    for plane, digest in _frames_gray(path):
        if prev is None:
            prev, prev_hash = plane, digest
            index = 1
            continue
        exact = digest == prev_hash
        if exact:
            rep.exact_duplicates += 1
            diff = 0.0
        else:
            diff = float(np.abs(prev[::8, ::8] - plane[::8, ::8]).mean())
            rep.diff_series.append(round(diff, 4))
        if diff < STATIC_DIFF:
            rep.static_pairs += 1
        # A "held" pair is one the decoder reproduced almost exactly; several in a row is a freeze.
        held = diff < EXACT_HOLD_DIFF
        if held:
            if run_start_idx is None:
                run_start_idx = index - 1
        elif run_start_idx is not None:
            rep.longest_freeze_sec = max(rep.longest_freeze_sec, (index - 1 - run_start_idx) * interval)
            run_start_idx = None
        if exact:
            shifts.append((0.0, 0.0))
            costs.append(0.0)
        else:
            dx, dy, cost = _global_shift(prev, plane)
            shifts.append((dx, dy))
            costs.append(cost)
        prev, prev_hash = plane, digest
        index += 1
    if run_start_idx is not None:
        rep.longest_freeze_sec = max(rep.longest_freeze_sec, (index - run_start_idx) * interval)

    rep.frames = index
    rep.shifts = np.asarray(shifts, dtype=np.float32) if shifts else np.zeros((0, 2), dtype=np.float32)
    rep.costs = np.asarray(costs, dtype=np.float32)
    if not rep.duration and nominal and rep.frames:
        rep.duration = rep.frames / nominal
        rep.measured_fps = nominal
    return rep


def _source_window(source: str, start: float, end: float, dest_dir: str) -> str:
    out = dest_dir
    args = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.4f}", "-t", f"{end - start:.4f}", "-i", source,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-qp", "14", "-bf", "0", out]
    subprocess.run(args, check=True, timeout=1800)
    return out


def main() -> None:
    import os
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--source", help="also analyse this window of the source, as the reference motion")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--series", help="write the per-frame travel series to this JSON file")
    args = ap.parse_args()

    report = analyse(args.file)
    out: dict = {"output": report.as_dict()}
    if args.source and args.end > args.start:
        tmp = os.path.join(tempfile.gettempdir(), "cf_diag_source_window.mp4")
        out["source_reference"] = analyse(_source_window(args.source, args.start, args.end, tmp)).as_dict()
    if args.series:
        travel = np.hypot(report.shifts[:, 0], report.shifts[:, 1]).round(3).tolist()
        with open(args.series, "w", encoding="utf-8") as handle:
            json.dump({"travel_px": travel, "residual": [round(c, 3) for c in report.costs.tolist()]}, handle)
        out["series_file"] = args.series
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
