"""Per-video editing-style calibration for Auto OpenShorts v2.

Every source is cut differently: a reality show changes shot every 1.2 s, a podcast sits on one
frame for 20 s, a vlog walks the camera through a room. A single global hold therefore has to be
wrong somewhere — too tight and the framing snaps on every micro-cut, too loose and the camera
pans across cuts it should have landed on.

So measure the footage first and derive the numbers from it. `signature_of` reads only cheap
frame statistics (histograms, small-frame motion energy, face/person counts) over the same samples
the tracker already walks; `tune` turns that into an `EditTuning` that the engine applies to the
scene-cut gate, the layout hold, the camera pursuit and whether a 2-up split is ever allowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import median
from typing import Any


# How long a derived hold is allowed to be, whatever the footage says about itself. Three seconds is
# the shortest time a composition — solo crop, 2-up, or the wide blurred field — can be on screen and
# still be read as a deliberate shot rather than a twitch; the ceiling is reached by sources that
# genuinely hold one shot for several seconds.
HOLD_MIN_SEC = float(os.environ.get("HOLD_MIN_SEC", "3.0"))
HOLD_MAX_SEC = float(os.environ.get("HOLD_MAX_SEC", "5.0"))
# Kept as its own number because a two-panel split asserts that two people matter at once, which is
# the hardest composition to read. With every layout now floored at three seconds it is normally the
# same number as the hold; it only diverges if footage with a very long rhythm raises the hold above it.
SPLIT_MIN_HOLD_SEC = float(os.environ.get("SPLIT_MIN_HOLD_SEC", "3.0"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Seconds to close ~63% of the remaining aim error, and crop widths per second at unit error, per
# editing style. Fast-cut footage needs a camera that lands before the next cut; a long take can
# afford to drift, and drifting is what makes a static interview feel alive rather than robotic.
_TAU_X = {
    "rapid_multicam": 0.30,
    "dialogue_cuts": 0.38,
    "long_take": 0.46,
    "static_single_shot": 0.55,
    "action_object": 0.28,
}
_SPEED_X = {
    "rapid_multicam": 0.70,
    "dialogue_cuts": 0.55,
    "long_take": 0.45,
    "static_single_shot": 0.35,
    "action_object": 0.75,
}


@dataclass
class Signature:
    """What the footage itself says about how it was edited."""

    shot_gaps: list[float] = field(default_factory=list)  # seconds between detected shot changes
    samples: int = 0
    face_samples: int = 0  # samples where at least one face survived detection
    two_face_share: float = 0.0  # fraction of samples with two readable faces
    person_share: float = 0.0  # fraction with a body anchor (wide or occluded framing)
    object_share: float = 0.0  # fraction where the subject was a thing, not a face or a body
    median_face_width: float = 0.0  # fraction of frame width
    black_share: float = 0.0  # fade/night frames the tracker should not chase
    motion_energy: float = 0.0  # mean small-frame luma change per second
    subject_travel: float = 0.0  # median subject centre movement, crop widths per second
    subjects_measured: bool = True  # False in the pre-pass, where no detector has run yet

    @property
    def median_shot_sec(self) -> float:
        return median(self.shot_gaps) if self.shot_gaps else 0.0


@dataclass
class EditTuning:
    style: str
    layout_hold_sec: float
    split_min_hold_sec: float
    cut_snap_min_sec: float
    deadband_x: float
    deadband_y: float
    tau_x: float
    tau_y: float
    max_speed_x: float
    max_speed_y: float
    split_allowed: bool
    punch_gain: float
    signature: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "layout_hold_sec": round(self.layout_hold_sec, 3),
            "split_min_hold_sec": round(self.split_min_hold_sec, 3),
            "cut_snap_min_sec": round(self.cut_snap_min_sec, 3),
            "deadband_x": round(self.deadband_x, 4),
            "deadband_y": round(self.deadband_y, 4),
            "tau_x": round(self.tau_x, 3),
            "tau_y": round(self.tau_y, 3),
            "max_speed_x": round(self.max_speed_x, 3),
            "max_speed_y": round(self.max_speed_y, 3),
            "split_allowed": self.split_allowed,
            "punch_gain": round(self.punch_gain, 3),
            "signature": self.signature,
        }

    def keep_measured_camera(self, applied: "EditTuning") -> None:
        """Adopt the pursuit numbers a camera actually ran with.

        A pass tunes twice: the pre-pass from histograms alone, which is what the camera followed,
        and the post-pass once the detectors have filled in face widths and subject travel, which is
        what decides the layouts. Both see the same shot rhythm so the style agrees, but the busy
        and tight modifiers only exist after detection — and the payload has to describe the crop
        that was rendered, not a law that was never applied.
        """
        self.deadband_x, self.deadband_y = applied.deadband_x, applied.deadband_y
        self.tau_x, self.tau_y = applied.tau_x, applied.tau_y
        self.max_speed_x, self.max_speed_y = applied.max_speed_x, applied.max_speed_y


def signature_of(
    shot_gaps: list[float],
    samples: int,
    face_hits: int,
    two_face_hits: int,
    person_hits: int,
    object_hits: int,
    face_widths: list[float],
    black_hits: int,
    motion_sum: float,
    travel: list[float],
    sample_dt: float,
    subjects_measured: bool = True,
) -> Signature:
    safe = max(1, samples)
    dt = max(1e-3, sample_dt)
    return Signature(
        shot_gaps=[g for g in shot_gaps if g > 0],
        samples=samples,
        face_samples=face_hits,
        two_face_share=two_face_hits / safe,
        person_share=person_hits / safe,
        object_share=object_hits / safe,
        median_face_width=median(face_widths) if face_widths else 0.0,
        black_share=black_hits / safe,
        motion_energy=motion_sum / safe / dt,
        subject_travel=(median(travel) if travel else 0.0) / dt,
        subjects_measured=subjects_measured,
    )


def _style_for(msl: float) -> str:
    if msl <= 0:
        # Nothing cut inside the window: it is one continuous take, however long the clip is.
        return "static_single_shot"
    if msl < 1.6:
        return "rapid_multicam"
    if msl <= 3.5:
        return "dialogue_cuts"
    if msl <= 8.0:
        return "long_take"
    return "static_single_shot"


def tune(sig: Signature) -> EditTuning:
    msl = sig.median_shot_sec
    # The hold is a fraction of the video's own shot rhythm, never an absolute. With no cuts to
    # measure there is no rhythm to follow, so the calmest reading is the right one.
    hold = _clamp(msl * 0.8, HOLD_MIN_SEC, HOLD_MAX_SEC) if msl > 0 else HOLD_MAX_SEC
    snap_min = _clamp(msl * 0.6, 0.35, hold) if msl > 0 else 0.35

    style = _style_for(msl)
    measured = sig.subjects_measured
    # A shot with no readable face or body in it is carried by a thing or by motion, and the cut
    # rhythm above says nothing about it; those edits cut on action, so they get the fast law. Only
    # say that once a detector has actually looked — the pre-pass has no subjects by definition, and
    # treating that as "action" put every video on the fastest camera on the first pass.
    if measured and (sig.object_share >= 0.15 or (sig.face_samples == 0 and sig.motion_energy > 18.0)):
        style = "action_object"

    busy = measured and (sig.subject_travel > 0.10 or sig.object_share >= 0.15)
    tight = measured and sig.median_face_width < 0.10  # wides and long lenses need more travel

    tau_x = _TAU_X[style]
    if busy:
        tau_x *= 0.82
    max_speed_x = _SPEED_X[style]
    if tight:
        max_speed_x *= 1.15
    # Night footage is where a tracker lies: the detector drops the subject for a second at a time,
    # and a camera that chases every flicker reads as jitter. Give it a wider tolerance and less
    # zoom drama for the frames the detector cannot see through.
    dark = sig.black_share > 0.12
    if dark:
        tau_x *= 1.3
        max_speed_x *= 0.8

    # Two-up only where the video actually stages two people together; otherwise the panel is empty
    # air and every fold looks like noise.
    split_allowed = sig.two_face_share >= 0.12 and sig.median_face_width <= 0.22

    deadband_x = 0.08 * (1.35 if style in ("long_take", "static_single_shot") else 1.0)
    if dark:
        deadband_x *= 1.25
    deadband_x = _clamp(deadband_x, 0.05, 0.16)
    return EditTuning(
        style=style,
        layout_hold_sec=hold,
        split_min_hold_sec=max(hold, SPLIT_MIN_HOLD_SEC),
        cut_snap_min_sec=snap_min,
        deadband_x=deadband_x,
        deadband_y=0.08 if dark or style == "static_single_shot" else 0.05,
        tau_x=tau_x,
        tau_y=0.55 if style != "rapid_multicam" else 0.45,
        max_speed_x=max_speed_x,
        max_speed_y=0.35 if dark else 0.45,
        split_allowed=split_allowed,
        punch_gain=0.6 if dark else (1.0 if style in ("rapid_multicam", "dialogue_cuts", "action_object") else 0.75),
        signature={
            "median_shot_sec": round(msl, 3),
            "shots_detected": len(sig.shot_gaps),
            "two_face_share": round(sig.two_face_share, 3),
            "person_share": round(sig.person_share, 3),
            "object_share": round(sig.object_share, 3),
            "median_face_width": round(sig.median_face_width, 4),
            "black_share": round(sig.black_share, 3),
            "motion_energy": round(sig.motion_energy, 2),
            "subject_travel": round(sig.subject_travel, 3),
            "face_hit_share": round(sig.face_samples / max(1, sig.samples), 3),
        },
    )
