from __future__ import annotations

import math
import os
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from .rendering.filters import crop_rect
from .speaker_priority import WINDOW_SECONDS, head_zone

WEIGHTS = {"visibility": 0.25, "preservation": 0.25, "confidence": 0.15,
           "balance": 0.15, "relevance": 0.10, "resolution": 0.10}

# A sustained solo shot can outlive a tracker ID swap at a source cut.
SOLO_FALLBACK_BEHAVIORS = {"solo_monologue", "emphatic_gesture", "object_demonstration", "physical_action"}
WIDE_HEAD_FRACTION = 0.055
BYSTANDER_WIDTH_RATIO = 0.7
DUPLICATE_HEAD_OVERLAP = 0.25
# Two heads farther apart than this cannot share one 9:16 fill crop, but each fits its own half-height
# split panel (which sees ~2x the horizontal span). Below it, a single fill two-shot holds both and is
# preferred; above it the reference edit stacks the pair as a vertical split.
SPLIT_SEPARATION = 0.22
# OpenShorts active_speaker.is_conversation minimum window share (20%) required for a second speaker
# to justify a 2-Up SPLIT instead of keeping a single-speaker TRACK (fill) crop on the floor-holder.
CONVERSATION_MIN_SHARE = 0.20
STATIC_HEAD_RATE = 2.0
STATIC_HEAD_RELATIVE = 0.3
STATIC_MIN_OBSERVATIONS = 4


@dataclass(frozen=True)
class LayoutPolicy:
    confidence: float = 0.45
    visibility: float = 0.98
    max_panel_overlap: float = 0.15
    detection_share: float = 0.65
    switch_margin: float = 0.08
    hold_seconds: float = 3.0

    @classmethod
    def from_environment(cls, hold_seconds: float = 3.0) -> "LayoutPolicy":
        values = {}
        for name in ("confidence", "visibility", "max_panel_overlap", "detection_share", "switch_margin"):
            raw = float(os.environ.get(f"LAYOUT_{name.upper()}", getattr(cls(), name)))
            if not math.isfinite(raw) or not 0 <= raw <= 1:
                raise ValueError(f"LAYOUT_{name.upper()} must be in [0, 1]")
            values[name] = raw
        return cls(**values, hold_seconds=hold_seconds)


def _intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box(person):
    # Only non-human objects require their full bounding box in frame; for humans ("face" or "person"),
    # portrait 9:16 framing centers and preserves the head/upper-shoulder zone (`head_zone`), otherwise
    # any medium close-up whose torso/arms exceed ~542px width is forced out of Full 9:16 into blur-fit.
    x, y, w, h = person["box"] if person.get("kind") == "object" else head_zone(person)
    return x, y, x + w, y + h


def _coverage(person, rects):
    box = _box(person)
    return max((_intersection(box, rect) / max(1.0, _area(box)) for rect in rects), default=0.0)


def _rects(point, mode, width, height):
    if mode == "fit":
        return [(0.0, 0.0, float(width), float(height))]
    zoom = max(1.0, float(point.get("zoom", 1.0)))
    centres = [(point["x"], point["y"])]
    if mode == "split":
        if point.get("x2") is None or point.get("y2") is None:
            return []
        centres.append((point["x2"], point["y2"]))
        zoom = max(zoom * 1.32, 1.38)
    rects = []
    for x, y in centres:
        w, h, left, top = crop_rect(x, y, zoom, width, height, 1080, 960 if mode == "split" else 1920, mode == "split")
        rects.append((left, top, left + w, top + h))
    return rects


def _focal_objects(people, width, height):
    humans = [p for p in people if p.get("kind") != "object"]
    return [
        p for p in people
        if p.get("kind") == "object"
        and (not humans or (not p.get("background_context") and _area(_box(p)) < 0.30 * width * height))
    ]


def _without_set_pieces(samples, width):
    paths: dict[Any, list[tuple[float, float, float, float]]] = {}
    for sample in samples:
        for person in sample.get("people", []):
            if person.get("kind") == "face" and person.get("box"):
                hx, hy, hw, hh = head_zone(person)
                paths.setdefault(person["key"], []).append(
                    (hx + hw / 2, hy + hh / 2, hw, float(person.get("confidence", 0.0))))
    paths = {key: path for key, path in paths.items()
             if len(path) >= max(STATIC_MIN_OBSERVATIONS, math.ceil(len(samples) * 0.65))}
    if len(paths) < 2:
        return samples
    sizes = {key: mean(point[2] for point in path) for key, path in paths.items()}
    confidence = {key: mean(point[3] for point in path) for key, path in paths.items()}
    lead = max(sizes, key=sizes.get)
    if sizes[lead] < 0.085 * width or confidence[lead] < 0.9:
        return samples
    rates = {key: mean(math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:]))
             for key, path in paths.items()}
    limit = min(STATIC_HEAD_RATE * width / 1920, STATIC_HEAD_RELATIVE * rates[lead])
    # Stillness alone cannot distinguish a sculpture from a listening person.
    still = {key for key, rate in rates.items() if key != lead and rate <= limit
             and sizes[key] < 0.8 * sizes[lead] and confidence[key] < confidence[lead] - 0.08}
    if not still:
        return samples
    return [{**sample, "people": [p for p in sample.get("people", [])
                                  if p.get("kind") != "face" or p["key"] not in still]}
            for sample in samples]


def _required(people, talk, main_key, width, height):
    keys = getattr(talk, "keys", frozenset())
    humans = foreground_humans(people)
    humans += [p for p in people if p.get("kind") != "object" and p not in humans
               and p.get("key") in keys and head_zone(p)[2] >= WIDE_HEAD_FRACTION * width]
    objects = _focal_objects(people, width, height)
    if keys:
        selected = [p for p in humans if p["key"] in keys]
        if {p["key"] for p in selected} != set(keys):
            return humans + objects, False
        if len(humans) >= 3:
            selected = humans
    elif getattr(talk, "loud", False) and not getattr(talk, "ambiguous", False):
        selected = [p for p in humans if p["key"] == main_key] or humans
    else:
        selected = humans
    return selected + objects, True


def foreground_humans(people: list[dict[str, Any]], min_confidence: float = 0.0) -> list[dict[str, Any]]:
    """Confident foreground heads without strongly overlapping duplicate detections."""
    humans = [p for p in people if p.get("box") and p.get("kind") != "object"
              and float(p.get("confidence", 0.0)) >= min_confidence]
    if not humans:
        return humans
    widths = [head_zone(p)[2] for p in humans]
    largest = max(widths)
    humans = [p for p, w in zip(humans, widths) if w >= BYSTANDER_WIDTH_RATIO * largest]
    kept: list[dict[str, Any]] = []
    for person in sorted(humans, key=lambda p: -float(p.get("confidence", 0.0))):
        box = _box(person)
        if any(_intersection(box, _box(q)) / max(1.0, _area(box) + _area(_box(q))
                                               - _intersection(box, _box(q))) >= DUPLICATE_HEAD_OVERLAP
               for q in kept):
            continue
        kept.append(person)
    return kept


def _solo_subject(sample, talk, main_key, policy, width, behavior=None):
    """One confidently tracked human who is the attributed speaker (or the only candidate): a close-up."""
    humans = foreground_humans(sample.get("people", []), policy.confidence)
    if len(humans) != 1:
        return None
    person = humans[0]
    if head_zone(person)[2] < WIDE_HEAD_FRACTION * width:
        return None
    keys = getattr(talk, "keys", frozenset())
    if getattr(talk, "ambiguous", False) or len(keys) > 1:
        return None
    if keys and person["key"] not in keys:
        if behavior not in SOLO_FALLBACK_BEHAVIORS:
            return None
        if any(p.get("kind") != "object" and p.get("key") in keys
               and float(p.get("confidence", 0.0)) >= policy.confidence
               and head_zone(p)[2] >= WIDE_HEAD_FRACTION * width
               for p in sample.get("people", [])):
            return None
    return person


def anchor_solo_subjects(points, samples, talks, main_key, width, height, policy, behaviors):
    if not points:
        return
    origin = samples[0]["t"]
    starts = [0] + [i for i in range(1, len(points)) if points[i].get("cut")] + [len(points)]
    for lo, hi in zip(starts, starts[1:]):
        evidence = _without_set_pieces(samples[lo:hi], width)
        targets = {}
        for i, sample in enumerate(evidence, lo):
            talk = talks.get(int((sample["t"] - origin) / WINDOW_SECONDS))
            solo = _solo_subject(sample, talk, main_key, policy, width, behaviors.get(sample["t"]))
            if solo is not None:
                hx, hy, hw, hh = head_zone(solo)
                targets[i] = ((hx + hw / 2) / width, (hy + hh / 2) / height)
        # OpenShorts solo-floor-holder anchoring: when a multi-person shot has a single dominant
        # speaker who holds the floor (`_speaker_focus` returns `len(keys) == 1, hint == "fill"`),
        # anchor the shot's camera path directly to that speaker across brief pauses so a silent
        # listener never pulls the crop away from the speaker.
        used_speaker_focus = False
        if len(targets) / (hi - lo) < policy.detection_share and talks:
            focus = _speaker_focus(points[lo:hi], evidence, talks, origin, width, height, policy)
            if focus is not None and focus[2] == "fill" and len(focus[0]) == 1:
                solo_key = next(iter(focus[0]))
                focus_targets = {}
                for i, sample in enumerate(evidence, lo):
                    humans = {p["key"]: p for p in foreground_humans(sample.get("people", []), policy.confidence)}
                    if solo_key in humans and head_zone(humans[solo_key])[2] >= WIDE_HEAD_FRACTION * width:
                        hx, hy, hw, hh = head_zone(humans[solo_key])
                        focus_targets[i] = ((hx + hw / 2) / width, (hy + hh / 2) / height)
                if len(focus_targets) / (hi - lo) >= policy.detection_share:
                    targets = focus_targets
                    used_speaker_focus = True
        if len(targets) / (hi - lo) < policy.detection_share:
            continue
        measured = sorted(targets)
        left = right = measured[0]
        cursor = 0
        for i in range(lo, hi):
            while cursor + 1 < len(measured) and measured[cursor] < i:
                left = measured[cursor]
                cursor += 1
            right = measured[cursor]
            if right <= i:
                left = right
            fraction = min(1.0, max(0.0, (i - left) / max(1, right - left)))
            points[i]["x"], points[i]["y"] = (
                a + (b - a) * fraction for a, b in zip(targets[left], targets[right]))
            if used_speaker_focus and points[i].get("mode") == "split":
                points[i]["mode"] = "fill"
                points[i]["x2"], points[i]["y2"] = points[i]["x"], points[i]["y"]


def _speaker_focus(points, samples, talks, origin, width, height, policy):
    """The speaker set a shot should be framed around, and how to frame it.

    The reference edit crops to whoever holds the floor — one speaker as a full-bleed fill, two as a
    vertical split when they stand apart or one fill two-shot when adjacent — and only keeps the blurred
    wide card for genuinely diffuse group/establishing shots. Bystanders are not required content for a
    speaker crop. The returned set always contains every attributed talker of the shot (after filtering
    single-window non-conversation listener blips under OpenShorts' 20% `is_conversation` threshold),
    so reducing the layout evidence to it can never drop a genuine speaker.

    Returns (keys, anchors, hint): anchors are stable split panel centres when the pair stands apart,
    else None; hint is "split" or "fill". Returns None when the shot has no clean speaker focus.
    """
    n = len(points)
    if n < 4:
        return None
    raw_union: set = set()
    key_counts: dict[Any, int] = {}
    attributed_samples = 0
    for point in points:
        talk = talks.get(int((point["t"] - origin) / WINDOW_SECONDS))
        keys = frozenset(getattr(talk, "keys", frozenset()))
        if keys:
            attributed_samples += 1
            for k in keys:
                raw_union.add(k)
                key_counts[k] = key_counts.get(k, 0) + 1

    # OpenShorts `active_speaker.is_conversation(min_share=0.20)` gate:
    # If multiple identities were attributed in this shot, keep only speakers who hold at least 20%
    # of attributed samples, provided any dropped minor listener blip occurred in <= 20% of total
    # shot samples (within `_evaluate`'s 25% `speaker_missing` tolerance).
    union = set(raw_union)
    if len(raw_union) >= 2 and attributed_samples > 0:
        major = {k for k, c in key_counts.items() if c / attributed_samples >= CONVERSATION_MIN_SHARE}
        dropped_samples = sum(c for k, c in key_counts.items() if k not in major)
        if major and dropped_samples / n <= CONVERSATION_MIN_SHARE:
            union = major

    if len(union) >= 3:
        return None
    if len(union) == 1:
        keys = frozenset(union)
        present = 0
        for point, sample in zip(points, samples):
            humans = {p["key"]: p for p in foreground_humans(sample.get("people", []), policy.confidence)}
            if keys <= humans.keys() and all(
                head_zone(humans[k])[2] >= WIDE_HEAD_FRACTION * width for k in keys
            ):
                present += 1
        if present / n < policy.detection_share:
            return None
        return keys, None, "fill"

    per_sample: list[tuple[frozenset, tuple[float, float], tuple[float, float]]] = []
    for point, sample in zip(points, samples):
        talk = talks.get(int((point["t"] - origin) / WINDOW_SECONDS))
        keys = frozenset(getattr(talk, "keys", frozenset()))
        humans = foreground_humans(sample.get("people", []), policy.confidence)
        by_key = {p["key"]: p for p in humans}
        if len(union) == 2 and union <= by_key.keys():
            pair = sorted(union, key=lambda k: head_zone(by_key[k])[0])
        elif len( keys) == 2 and keys <= by_key.keys():
            pair = sorted(keys, key=lambda k: head_zone(by_key[k])[0])
        elif (not union and not keys and len(humans) == 2
              and min(head_zone(p)[2] for p in humans) >= WIDE_HEAD_FRACTION * width):
            pair = sorted((p["key"] for p in humans), key=lambda k: head_zone(by_key[k])[0])
        else:
            continue
        a, b = by_key[pair[0]], by_key[pair[1]]
        ha, hb = head_zone(a), head_zone(b)
        ca = (ha[0] + ha[2] / 2, ha[1] + ha[3] / 2)
        cb = (hb[0] + hb[2] / 2, hb[1] + hb[3] / 2)
        per_sample.append((frozenset(pair), ca, cb))

    if not per_sample:
        return None
    counts: dict[frozenset, int] = {}
    for pair, _, _ in per_sample:
        counts[pair] = counts.get(pair, 0) + 1
    pair = max(counts, key=counts.get)
    good = [(ca, cb) for p, ca, cb in per_sample if p == pair
            and (cb[0] - ca[0]) >= SPLIT_SEPARATION * width]
    adjacent = [(ca, cb) for p, ca, cb in per_sample if p == pair
                and (cb[0] - ca[0]) < SPLIT_SEPARATION * width]
    if len(good) / n >= policy.detection_share:
        lx, ly = median(c[0][0] for c in good) / width, median(c[0][1] for c in good) / height
        rx, ry = median(c[1][0] for c in good) / width, median(c[1][1] for c in good) / height
        return pair, [(lx, ly, rx, ry)] * n, "split"
    if len(adjacent) / n >= policy.detection_share:
        return pair, None, "fill"
    return None


def _evaluate(mode, points, samples, talks, origin, main_key, width, height, policy, behaviors=None):
    values = {name: [] for name in WEIGHTS}
    reasons = set()
    measured = 0
    speaker_missing = 0
    bottom_graphic_hits = 0
    outside_crop_hits = 0
    split_overlap_hits = 0
    split_distinct_hits = 0
    split_subject_hits = 0
    for point, sample in zip(points, samples):
        people = sample.get("people", [])
        w_idx = int((point["t"] - origin) / WINDOW_SECONDS)
        talk = talks.get(w_idx)
        behavior = behaviors.get(sample["t"]) if behaviors else None
        required, visible_talkers = _required(people, talk, main_key, width, height)
        solo = _solo_subject(sample, talk, main_key, policy, width, behavior)
        if solo is not None:
            required, visible_talkers = [solo, *_focal_objects(people, width, height)], True
        confidence = min((float(p.get("confidence", 0.0)) for p in required), default=0.0)
        reliable = bool(required) and confidence >= policy.confidence and visible_talkers
        measured += int(reliable)
        rects = _rects(point, mode, width, height)
        coverage = min((_coverage(p, rects) for p in required), default=0.0)
        if mode != "fit":
            if required and coverage < policy.visibility:
                outside_crop_hits += 1
            if not visible_talkers:
                speaker_missing += 1
            # A close-up trades the source's lower-third graphic for the speaker's face.
            if point.get("hasBottomGraphic") and solo is None:
                bottom_graphic_hits += 1
        if mode == "split":
            humans = foreground_humans(people, policy.confidence)
            if len(rects) != 2:
                reasons.add("missing_panel_anchor")
            else:
                overlap = _intersection(*rects)
                iou = overlap / max(1.0, _area(rects[0]) + _area(rects[1]) - overlap)
                if iou > policy.max_panel_overlap:
                    split_overlap_hits += 1
                owners = [{p["key"] for p in humans if _coverage(p, [r]) >= policy.visibility} for r in rects]
                if not owners[0] or not owners[1] or owners[0] & owners[1]:
                    split_distinct_hits += 1
                if len({p["key"] for p in humans}) != 2:
                    split_subject_hits += 1
        visible = 1.0 if mode == "fit" else coverage
        conf_humans = foreground_humans(people, policy.confidence)
        wide = len(conf_humans) == 1 and head_zone(conf_humans[0])[2] < WIDE_HEAD_FRACTION * width
        context_needed = ((point.get("hasBottomGraphic") and solo is None) or not reliable
                          or len(required) > 2 or wide)
        preservation = 1.0 if mode == "fit" else coverage
        if rects and required:
            distance = []
            for person in required:
                box = _box(person)
                cx = (box[0] + box[2]) / 2
                distance.append(min(abs(cx - (r[0] + r[2]) / 2) / max(1, r[2] - r[0]) for r in rects))
            balance = max(0.0, 1.0 - 2 * mean(distance))
        else:
            balance = 1.0 if mode == "fit" else 0.0
        if mode == "fit":
            relevance = 1.0 if context_needed else 0.2
            balance = 0.8
            resolution = min(1.0, width / 1080)
        else:
            relevance = 0.2 if wide else (1.0 if mode == "fill" and len(required) == 1 else 0.85)
            resolution = min((min(1.0, (r[2] - r[0]) / 1080) for r in rects), default=0.0)
        for name, value in zip(WEIGHTS, (visible, preservation, confidence if mode != "fit" else 1.0, balance, relevance, resolution)):
            values[name].append(value)
    n_pts = max(1, len(points))
    if mode != "fit":
        if measured / n_pts < policy.detection_share:
            reasons.add("low_tracking_confidence")
        # Tolerate isolated 1-frame cut-boundary blur/pan jitter while enforcing shot-level rules cleanly.
        if outside_crop_hits / n_pts > (0.20 if n_pts >= 4 else 0.0):
            reasons.add("required_content_outside_crop")
        if speaker_missing / n_pts > (0.25 if n_pts >= 4 else 0.0):
            reasons.add("speaker_not_observed")
        if bottom_graphic_hits >= max(1, math.ceil(n_pts * 0.45)):
            reasons.add("preserve_source_graphics")
    if mode == "split":
        # Same isolated-frame tolerance as fill: a detector flicker on one sample must not disqualify
        # a shot that is cleanly a two-shot for the rest of its duration.
        tol = 0.20 if n_pts >= 4 else 0.0
        if split_overlap_hits / n_pts > tol:
            reasons.add("overlapping_panels")
        if split_distinct_hits / n_pts > tol:
            reasons.add("panels_not_distinct")
        if split_subject_hits / n_pts > tol:
            reasons.add("split_requires_two_subjects")
    components = {key: round(mean(value), 4) if value else 0.0 for key, value in values.items()}
    score = sum(components[key] * weight for key, weight in WEIGHTS.items())
    return {"score": round(score, 4), "components": components, "eligible": not reasons, "reasons": sorted(reasons)}


def finalize_layouts(points: list[dict[str, Any]], samples: list[dict[str, Any]], width: int, height: int,
                     talks: dict, main_key: int | None, framing_mode: str = "auto",
                     policy: LayoutPolicy | None = None,
                     behaviors: dict | None = None) -> list[dict[str, Any]]:
    if len(points) != len(samples):
        raise ValueError("Layout evidence and track must have matching samples")
    if not points:
        return []
    policy = policy or LayoutPolicy.from_environment()
    starts = [0] + [i for i in range(1, len(points)) if points[i].get("cut")]
    starts.append(len(points))
    report = []
    previous = None
    switched_at = points[0]["t"]
    origin = samples[0]["t"]
    for scene, (lo, hi) in enumerate(zip(starts, starts[1:])):
        current = points[lo:hi]
        evidence = _without_set_pieces(samples[lo:hi], width)
        focus = (_speaker_focus(current, evidence, talks, origin, width, height, policy)
                 if framing_mode == "auto" else None)
        anchors = None
        hint = None
        if focus:
            keys, anchors, hint = focus
            # Frame the shot around its speakers: bystanders are not required content for a speaker
            # crop, so drop them from the layout evidence. The kept set always contains every
            # attributed talker, which is what the delivered-coverage guarantee promises to show.
            evidence = [{**s, "people": [p for p in s.get("people", [])
                                         if p.get("kind") == "object" or p.get("key") in keys]}
                        for s in evidence]
        split_points = current
        if anchors:
            split_points = [{**p, "x": a[0], "y": a[1], "x2": a[2], "y2": a[3]}
                            for p, a in zip(current, anchors)]
        candidates = {"fill": _evaluate("fill", current, evidence, talks, origin, main_key, width, height, policy, behaviors),
                      "split": _evaluate("split", split_points, evidence, talks, origin, main_key, width, height, policy, behaviors),
                      "fit": _evaluate("fit", current, evidence, talks, origin, main_key, width, height, policy, behaviors)}
        if framing_mode == "auto":
            viable = [mode for mode in candidates if candidates[mode]["eligible"]]
            if hint and hint in viable:
                # The reference edit crops to whoever holds the floor; honour that composition whenever
                # it is safe instead of letting the blurred wide card win on a flat context score.
                chosen, reason = hint, "speaker_focus"
            else:
                chosen = max(viable, key=lambda mode: candidates[mode]["score"])
                reason = "highest_safe_score"
            if previous in viable and chosen != previous:
                # A hard cut is an edit point: the new shot's composition wins outright, so the layout
                # follows the content (solo crop / two-up / wide bed) instead of staying pinned. Hold and
                # hysteresis only damp oscillation inside continuous, un-cut footage.
                if not current[0].get("cut"):
                    holding = current[0]["t"] - switched_at < policy.hold_seconds
                    marginal = candidates[chosen]["score"] - candidates[previous]["score"] < policy.switch_margin
                    if holding or marginal:
                        chosen, reason = previous, "layout_hold" if holding else "score_hysteresis"
        else:
            chosen, reason = framing_mode, "manual_override"
            if chosen == "split" and not candidates["split"]["eligible"]:
                chosen, reason = "fit", "unsafe_manual_split"
        if chosen != previous:
            switched_at = current[0]["t"]
        if chosen == "fit" and framing_mode == "auto":
            failed = sorted({r for mode in ("fill", "split") for r in candidates[mode]["reasons"]})
            if failed:
                reason = ",".join(failed)
        if chosen == "split" and anchors:
            for point, a in zip(current, anchors):
                point["x"], point["y"], point["x2"], point["y2"] = a
        for point in current:
            point["mode"] = chosen
            point["layoutDecision"] = {"scene": scene, "reason": reason}
            point["trackingConfidence"] = candidates["fill"]["components"]["confidence"]
            if chosen == "fit":
                point["zoom"] = 1.0
        report.append({"scene": scene, "start": current[0]["t"],
                       "end": points[hi]["t"] if hi < len(points) else current[-1]["t"],
                       "layout": chosen, "reason": reason, "candidates": candidates})
        previous = chosen
    return report
