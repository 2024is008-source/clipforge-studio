"""What the gimbal focuses on, and how it moves between subjects.

The layout pass decides *which composition* shows the people who matter; this module decides
*where the camera's attention rests* inside that composition, and makes the attention move like
a camera operator rather than a servo. Rules, in priority order:

  R1 active_talker   — one attributed talker holds the floor: they own the frame.
  R2 presumed_main   — loud audio, mouths this pass cannot separate, main speaker on screen:
                       he carries the frame rather than whoever the tracker happened to pick.
  R3 reaction_hold   — the floor just went silent: stay on the last talker for a beat. People
                       look at whoever just spoke; a camera that swings away the instant a
                       sentence ends reads as inattentive.
  R4 object_subject  — no human in shot: the thing the tracker chose is the subject. An object
                       never takes focus from a human; with a human present it only joins the
                       frame through the object-demonstration union crop.
  R5 tracker_hold    — silence with no recent talker: keep the tracker's framing.

A shared floor (two or three talking at once) is a group shot, not a single focus, so the
layout pass's group/split answer stands and the director does not re-aim it.

Motion rules:
  - Handoff glide: when focus changes inside a shot, the aim eases from where the camera is to
    the new subject over FOCUS_HANDOFF_SEC with a smoothstep — zero velocity at both ends, no
    mid-pan lurch. A cut is not a handoff: the edit moved the frame, so the camera snaps.
  - Gaze lead room: the aim leads slightly toward the direction the speaker faces, the way an
    operator leaves room into a gaze instead of pinning the head dead centre.
  - Breathing drift: a calm held shot drifts almost imperceptibly so a static interview feels
    alive rather than frozen. The drift is zero at a cut and ramped in, so it never reads as a
    transition, and it is far below the pursuit speed cap.

The track these rules write is the same keyframe list the browser preview smoothsteps
(reframe.ts sampleTrack) and the export turns into sendcmd commands, so both pipelines move
identically by construction — there is no second motion law to keep in parity.
"""

from __future__ import annotations

import math
from typing import Any

from .layout_decision import foreground_humans
from .speaker_priority import WINDOW_SECONDS, head_zone, prominently_shown

# How long the camera stays with the last talker after the floor goes silent. Two audio windows:
# long enough to cover the pause between sentences, short enough that a genuine change of
# subject is not held hostage to the previous one.
REACTION_HOLD_SEC = 0.8
# Seconds a focus handoff takes inside a shot. Long enough to read as a deliberate pan, short
# enough that the new talker is framed before their first sentence is finished.
FOCUS_HANDOFF_SEC = 0.55
# Normalised aim offset per unit of head yaw bias. Mirrors the solo close-up lead room in
# classify_human_behaviors so a solo shot and a multi-person shot lead the gaze by the same law.
GAZE_LEAD_ROOM = 0.018
# Breathing drift: peak aim offset (normalised), full cycle seconds, and the styles calm enough
# to earn it. Fast-cut footage is already alive; drifting it would read as instability.
BREATH_AMPLITUDE = 0.004
BREATH_PERIOD_SEC = 7.0
BREATH_STYLES = frozenset({"long_take", "static_single_shot"})
BREATH_RAMP_SEC = 1.5
# Mirrors clipforge_engine.BEHAVIOR_OBJECT_DEMO: the union crop already frames the person and
# the thing they are showing, so the director must not pull the aim back onto the person alone.
OBJECT_DEMO_BEHAVIOR = "object_demonstration"

RULE_ACTIVE_TALKER = "active_talker"
RULE_PRESUMED_MAIN = "presumed_main"
RULE_REACTION_HOLD = "reaction_hold"
RULE_OBJECT_SUBJECT = "object_subject"
RULE_TRACKER_HOLD = "tracker_hold"
RULE_SHARED_FLOOR = "shared_floor"


def _smoothstep(k: float) -> float:
    k = max(0.0, min(1.0, k))
    return k * k * (3.0 - 2.0 * k)


def _clamp_axis(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def focus_aim(person: dict[str, Any], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Head centre with a little room into the gaze, clamped like every other track centre."""
    hx, hy, hw, hh = head_zone(person)
    yaw = float(person.get("yaw_bias", 0.0) or 0.0)
    return (
        _clamp_axis((hx + hw / 2.0) / max(1, frame_w) + yaw * GAZE_LEAD_ROOM, 0.05, 0.95),
        _clamp_axis((hy + hh / 2.0) / max(1, frame_h), 0.08, 0.92),
    )


def focus_subject(
    sample: dict[str, Any],
    talk: Any,
    main_key: int | None,
    last_talkers: frozenset[int],
    last_talk_end: float,
    t: float,
) -> tuple[dict[str, Any] | None, str]:
    """The one person the camera owes its attention to at this sample, and the rule that says so.

    Returns (None, rule) when the answer is a group shot, an object, or the tracker's own
    framing — none of which the director re-aims.
    """
    humans = foreground_humans(sample.get("people", []))
    by_key = {int(p["key"]): p for p in humans if int(p.get("key", -1)) >= 0}
    talking = bool(talk is not None and getattr(talk, "talking", False))
    if talking:
        present = [k for k in sorted(talk.keys) if k in by_key]
        if len(present) == 1:
            return by_key[present[0]], RULE_ACTIVE_TALKER
        if len(present) > 1:
            return None, RULE_SHARED_FLOOR
        # The floor belongs to a voice that is not in shot. If the host is, he carries it.
        if main_key is not None and main_key in by_key and getattr(talk, "loud", False):
            return by_key[main_key], RULE_PRESUMED_MAIN
        return None, RULE_TRACKER_HOLD
    if (
        talk is not None
        and getattr(talk, "loud", False)
        and getattr(talk, "ambiguous", False)
        and main_key is not None
        and main_key in by_key
    ):
        return by_key[main_key], RULE_PRESUMED_MAIN
    if last_talkers and t <= last_talk_end + REACTION_HOLD_SEC:
        present = [k for k in sorted(last_talkers) if k in by_key]
        if len(present) == 1:
            return by_key[present[0]], RULE_REACTION_HOLD
    if not humans:
        objects = [p for p in sample.get("people", []) if p.get("kind") == "object" and p.get("box")]
        if objects:
            return None, RULE_OBJECT_SUBJECT
    return None, RULE_TRACKER_HOLD


def direct_speaker_focus(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, Any],
    main_key: int | None,
    frame_w: int,
    frame_h: int,
    behaviors: dict[float, str] | None = None,
) -> dict[str, Any]:
    """Rest the camera's attention on the rule ladder's subject, gliding when it changes.

    Runs after the layout pass and before the trajectory smoothing. Only `fill` aims are
    touched: a split already gives each talker a panel, and fit/editorial carry the whole room.
    A sample whose framing already shows the focus subject prominently is left alone — the
    least intrusive answer, same philosophy as the layout pass. The report counts which rule
    named the subject at every sample, so the guarantee is a number, not a claim.
    """
    if not track_points or len(track_points) != len(samples):
        return {"rules": {}, "handoffs": 0, "reaction_holds": 0}
    start = float(samples[0]["t"])
    rules: dict[str, int] = {}
    handoffs = reaction_holds = 0
    last_talkers: frozenset[int] = frozenset()
    last_talk_end = -1e9
    active_key: int | None = None
    blend_from: tuple[float, float] | None = None
    handoff_t0 = 0.0

    for pt, s in zip(track_points, samples):
        t = float(s["t"])
        w = max(0, int((t - start) / WINDOW_SECONDS))
        talk = talks.get(w)
        if talk is not None and getattr(talk, "talking", False):
            present = frozenset(
                int(p["key"]) for p in foreground_humans(s.get("people", []))
                if int(p.get("key", -1)) >= 0 and int(p["key"]) in talk.keys
            )
            if present:
                last_talkers = present
                last_talk_end = start + (w + 1) * WINDOW_SECONDS
        subject, rule = focus_subject(s, talk, main_key, last_talkers, last_talk_end, t)
        rules[rule] = rules.get(rule, 0) + 1
        if rule == RULE_REACTION_HOLD:
            reaction_holds += 1
        pt["focusRule"] = rule

        if pt.get("cut"):
            active_key, blend_from = None, None
        if subject is None or (pt.get("mode") or "fill") != "fill":
            continue
        if behaviors is not None and behaviors.get(t) == OBJECT_DEMO_BEHAVIOR:
            # The union crop frames the person and the thing they are showing; keep it.
            active_key, blend_from = int(subject["key"]), None
            continue
        key = int(subject["key"])
        if prominently_shown(subject, pt, frame_w, frame_h):
            active_key, blend_from = key, None
            continue
        desired = focus_aim(subject, frame_w, frame_h)
        if pt.get("cut"):
            pt["x"], pt["y"] = round(desired[0], 4), round(desired[1], 4)
            active_key, blend_from = key, None
            continue
        if key != active_key:
            handoffs += 1
        if key != active_key or blend_from is None:
            blend_from = (float(pt["x"]), float(pt["y"]))
            handoff_t0 = t
            active_key = key
        e = _smoothstep((t - handoff_t0) / FOCUS_HANDOFF_SEC)
        pt["x"] = round(_clamp_axis(blend_from[0] + (desired[0] - blend_from[0]) * e, 0.05, 0.95), 4)
        pt["y"] = round(_clamp_axis(blend_from[1] + (desired[1] - blend_from[1]) * e, 0.08, 0.92), 4)
    return {"rules": rules, "handoffs": handoffs, "reaction_holds": reaction_holds}


def apply_gimbal_breathing(
    points: list[dict[str, Any]],
    style: str,
    amplitude: float = BREATH_AMPLITUDE,
    period: float = BREATH_PERIOD_SEC,
) -> list[dict[str, Any]]:
    """A calm held shot drifts almost imperceptibly, so a static interview feels alive.

    Two slow sinusoids at different periods trace a small organic loop rather than a mechanical
    left-right sway. The drift is zero at a cut and smoothstep-ramped in over BREATH_RAMP_SEC,
    so it never reads as a transition; its peak speed (~0.004 crop widths/s) is two orders below
    the pursuit cap, so it cannot fight the rate limiter. Only calm styles and only full-frame
    crops earn it — a split or a contained shot is already carrying enough information.
    """
    if style not in BREATH_STYLES or len(points) < 3 or amplitude <= 0:
        return points
    out = [dict(p) for p in points]
    shot_t0 = float(out[0]["t"])
    for p in out:
        if p.get("cut"):
            shot_t0 = float(p["t"])
        if (p.get("mode") or "fill") != "fill":
            continue
        age = max(0.0, float(p["t"]) - shot_t0)
        ramp = _smoothstep(age / BREATH_RAMP_SEC)
        dx = amplitude * math.sin(2.0 * math.pi * age / period) * ramp
        dy = 0.5 * amplitude * math.sin(2.0 * math.pi * age / (period * 1.7)) * ramp
        p["x"] = round(_clamp_axis(float(p["x"]) + dx, 0.05, 0.95), 4)
        p["y"] = round(_clamp_axis(float(p["y"]) + dy, 0.08, 0.92), 4)
    return out