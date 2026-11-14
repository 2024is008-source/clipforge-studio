"""The output frame rate comes from the footage, not from the export dropdown's default.

24 fps film pushed to 29.97 by frame duplication judders on any slow move: the extra frames land in
an uneven 4-4-5-4 pattern rather than spread out. The studio's control only offers 29.97 or 30, and
neither is a request to resample someone's footage, so the measured source cadence wins unless the
requested rate is the same cadence or a clean whole multiple of it.

No optical-flow interpolation happens anywhere here: frames are never invented, only counted.
"""

from __future__ import annotations

import math

# (exact rate, the rational FFmpeg should be given). Drop-frame rates stay rationals: a decimal
# 23.976 is 23976/1000, which drifts against the 24000/1001 grid the filters build.
STANDARD_RATES: tuple[tuple[float, str], ...] = (
    (23.976023976023978, "24000/1001"),
    (24.0, "24/1"),
    (25.0, "25/1"),
    (29.97002997002997, "30000/1001"),
    (30.0, "30/1"),
    (47.95204795204795, "48000/1001"),
    (48.0, "48/1"),
    (50.0, "50/1"),
    (59.94005994005994, "60000/1001"),
    (60.0, "60/1"),
)

# A source further from a standard rate than this is its own cadence and is not snapped.
SNAP_TOLERANCE = 0.15
# How close a rate ratio must be to a whole number before resampling keeps the cadence.
MULTIPLE_TOLERANCE = 0.02
# Frame intervals this far apart describe different streams, not one cadence read two ways.
VFR_INTERVAL_MS = 0.5
MIN_OUTPUT_FPS = 6.0
MAX_OUTPUT_FPS = 60.0
DEFAULT_OUTPUT_FPS = 29.97002997002997


def requested_fps(target: str | float) -> float:
    """The studio's export setting as a number, with the historical default behind it.

    Snapped on the way in: the control sends the decimal string "29.97", and a decimal that reaches
    `fps=` runs against the 30000/1001 grid the frames are actually cut on, so a two-hour clip ends
    a third of a second short and the crop commands land off the frame slots.
    """
    if isinstance(target, (int, float)):
        value = float(target)
    else:
        try:
            value = float(str(target).strip())
        except (TypeError, ValueError):
            return DEFAULT_OUTPUT_FPS
    if not MIN_OUTPUT_FPS <= value <= MAX_OUTPUT_FPS:
        return DEFAULT_OUTPUT_FPS
    return snap_to_standard(value)


def snap_to_standard(fps: float) -> float:
    """Use the exact rational of the standard rate the footage is nearest to, if it is close at all.

    The nearest, not the first inside tolerance: scanning in order turned a genuine 30 into 29.97 and
    a genuine 60 into 59.94, because each sits only 0.03 from its drop-frame neighbour.
    """
    if fps <= 0:
        return fps
    distance, standard = min((abs(fps - s), s) for s, _arg in STANDARD_RATES)
    return standard if distance <= SNAP_TOLERANCE else fps


def rate_arg(fps: float) -> str:
    """The `-r` / `fps=` argument for a rate: always the rational when one exists."""
    for standard, arg in STANDARD_RATES:
        if abs(fps - standard) < 1e-6:
            return arg
    return f"{fps:.6f}"


def is_variable_rate(nominal: float, average: float) -> bool:
    """`r_frame_rate` is the shortest interval the container can express, `avg` the measured mean.

    A constant-frame-rate file has both the same. The comparison is milliseconds per frame, because
    29.97 and 30.0 are one cadence read two ways (33.367 against 33.333) while 30 against 60 is a
    genuinely variable stream described by its fastest instant.
    """
    if nominal <= 0 or average <= 0:
        return False
    return abs(1000.0 / nominal - 1000.0 / average) > VFR_INTERVAL_MS


def choose_source_rate(average: float, nominal: float = 0.0) -> float:
    """The source's own cadence, preferring the faster rate when the file is genuinely variable.

    Oversampling a VFR source duplicates frames, which is a smooth result; undersampling drops
    content, which is a missing moment in the user's clip.
    """
    if average <= 0:
        return nominal
    if nominal <= 0 or not is_variable_rate(nominal, average):
        return average
    return max(average, nominal)


def resolve_output_fps(source_fps: float, target: str | float, nominal_fps: float = 0.0) -> float:
    """The rate the render should be cut at, preserving the source cadence where it can.

    Three cases are safe to honour the requested rate: it already is the source's rate, it is a whole
    multiple of the source (30 to 60 duplicates evenly), or the source is a whole multiple of it
    (60 to 30 drops every other frame evenly). Anything else - 23.976 to 29.97, 24 to 30, 25 to 29.97
    - repeats frames unevenly, and that unevenness is the judder.
    """
    requested = requested_fps(target)
    source = snap_to_standard(choose_source_rate(source_fps, nominal_fps))
    if not (MIN_OUTPUT_FPS <= source <= MAX_OUTPUT_FPS):
        return requested
    if abs(source - requested) <= SNAP_TOLERANCE:
        return source
    up, down = requested / source, source / requested
    if _is_whole_multiple(up):
        return snap_to_standard(requested)
    if _is_whole_multiple(down):
        # Halve or third the source's own rate rather than the requested number: 60 to 29.97 would
        # repeat one frame in a thousand, while 60 to 30 drops exactly every other one.
        return snap_to_standard(source / round(down))
    return source


def _is_whole_multiple(ratio: float) -> bool:
    whole = round(ratio)
    return whole >= 2 and abs(ratio - whole) <= MULTIPLE_TOLERANCE
