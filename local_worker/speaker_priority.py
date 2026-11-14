"""Who must be on screen: the main speaker, the current talker, or all of them at once.

The tracker picks a subject per frame from size, centrality and stickiness. On multi-person footage
that is not the same question as "who is talking", so a clip can spend its best seconds pointed at a
listener while the person everyone came to hear is at the edge of the source and outside the 9:16
crop. Three rules, in priority order, come out of this module:

1. While the main speaker talks, he is on screen. Non-negotiable — and when the audio is loud, he is
   standing in the shot, and the mouths cannot be separated, he is the one presumed to be talking.
2. While somebody else talks, that person is on screen — the main speaker waits his turn.
3. While two or three talk over each other, all of them are on screen together, using the least
   artificial layout that fits: a natural two-shot crop if they are close enough, otherwise split
   panels, otherwise the room held wide with the blurred source behind it.

In every case the person who owns the audio is the *subject* of the frame, not a head tucked into a
context band: candidates that show him small lose to candidates that show him large.

How it knows who talks: faces are clustered into people by appearance (a cut hands the tracker a fresh
id for the same person, so an id cannot carry an identity across shots), each person's mouth region is
differenced frame to frame, and the mouth movement is compared with the audio envelope in fixed
windows. A person whose mouth moves when the audio is loud is talking. Two people whose mouths move
by similar amounts in the same loud window are talking over each other, which is what triggers rule 3.

Every decision is reported back as measured coverage, so the guarantee is a number a test can assert
rather than a claim in a comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

WINDOW_SECONDS = 0.4
AUDIO_FLOOR = 0.18
# Two mouths moving by similar amounts in a loud window: nobody clearly has the floor.
OVERLAP_MARGIN = 0.25
# How many people at once the shared-floor verdict will name. Three is the most the mouth signal can be
# trusted to separate; past that the group is a crowd and the source's own cut is the better read.
MAX_OVERLAP_TALKERS = 3
# Above this share of the person's own typical mouth motion, that window counts as them talking.
MOUTH_ACTIVE = 0.35
# A mouth whose loudest window is under this share of the loudest mouth in the clip never counts, so a
# still face's own noise cannot normalise itself into a turn at the microphone.
MOUTH_SHARE_OF_LOUDEST = 0.25
# Appearance gate for treating two detections as the same person. Deliberately looser than IoU —
# a handheld multicam shot moves people around — but tight enough that a wide two-shot does not merge.
MAX_WIDTH_RATIO = 1.32
MAX_CENTRE_GAP = 0.12
# How far a head may travel between two samples of the same shot and still be the same person. Inside a
# shot position is the strongest cue there is, and ignoring it is what turned a five-person panel into a
# dozen people.
MAX_MOVE_WITHIN_SHOT = 0.09
# Detector slop on a head measurement, in source pixels. Ratio gates alone lose small heads in a wide
# shot, because a few pixels of jitter is a third of their width.
HEAD_NOISE_PX = 28.0
# How much horizontal position is allowed to prefer one identity over another across a cut: enough to
# stop four locked-off heads being handed out in dict order, never enough to veto a reframe.
CUT_X_TIE_WEIGHT = 0.35
# The crop has to keep this much of a person's head for them to count as being on screen.
MIN_HEAD_VISIBLE = 0.7
# A body box is not a head: only its top slice is the face, and framing the middle of a torso puts the
# head out of shot on a standing person.
HEAD_SLICE_OF_BODY = 0.35
# A YOLO person box is shoulder width across, so the head inside it is a fraction of that. Identity and
# the mouth patch both work on this head box, which is what lets a face and a body be the same human.
HEAD_W_OF_BODY = 0.40
HEAD_CY_OF_BODY = HEAD_SLICE_OF_BODY / 2.0
# How large a step the rule may ask the camera to make, in crop widths, before it is reported as a
# re-anchor the pursuit law has to absorb rather than an ordinary correction.
MAX_REAIM_GAP = 0.35
# Windows a newly read talker must repeat before the framing turns to them (0.4 s each).
FLOOR_HOLD_WINDOWS = 2
# Side of the square the mouth patch is resampled to before differencing, so a close-up and a face at
# the back of the room land on the same grid and the reading survives the box changing size.
PATCH_SIDE = 24
# A head narrower than this in source pixels has a mouth patch smaller than the patch grid, so the
# reading would be upsampled noise. Crowds of small heads are the common case in this kind of footage
# and pretending to read them is what made every panel window demand a group shot.
MIN_MOUTH_HEAD_W = 60.0


def _median(values: list[float]) -> float:
    s = sorted(v for v in values if math.isfinite(v))
    if not s:
        return 0.0
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: list[float], q: float) -> float:
    s = sorted(v for v in values if math.isfinite(v))
    if not s:
        return 0.0
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


@dataclass
class Person:
    """One cluster of detections treated as the same human being across the clip."""

    key: int
    widths: list[float] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)
    screen_sec: float = 0.0
    talk_sec: float = 0.0


def head_box(box: list[float], kind: str = "face") -> list[float]:
    """The head inside a detection, in the same coordinate space as the detection.

    A face is already a head. A body box is shoulder width and full height, so its head is the centred
    top slice — and until both kinds are put through this, the same seated person measured as a face and
    measured as a body is two different people to the matcher, which is how a five-person panel became a
    dozen identities.
    """
    x, y, w, h = [float(v) for v in box]
    if kind != "person":
        return [x, y, w, h]
    hw = w * HEAD_W_OF_BODY
    return [x + (w - hw) / 2.0, y, hw, h * HEAD_SLICE_OF_BODY]


class PersonRegistry:
    """Cluster detections into people and keep them alive across cuts."""

    def __init__(self) -> None:
        self.people: dict[int, Person] = {}
        self._seen_at: dict[int, float] = {}
        self._next_key = 0

    def _distance(self, person: Person, hw: float, hx: float, hy: float, within_shot: bool) -> float:
        med_w = _median(person.widths)
        if med_w <= 0 or not person.centers:
            return math.inf
        ratio = max(hw / med_w, med_w / hw)
        # A 40 px head in a wide shot is not measured to the same precision as a 400 px one, and a fixed
        # ratio gate throws the small ones out of their own identity on every sample. The slack is
        # therefore a few pixels of head width, whatever the head size happens to be.
        if ratio > 1.0 + max(MAX_WIDTH_RATIO - 1.0, HEAD_NOISE_PX / max(med_w, hw)):
            return math.inf
        if within_shot:
            # Inside one shot the head barely moves and its size does not change, so whoever is closest
            # is the same person. This is what separates four seated panel members who look identical.
            last_x, last_y = person.centers[-1]
            drift = max(abs(hx - last_x), abs(hy - last_y))
            if drift > MAX_MOVE_WITHIN_SHOT:
                return math.inf
            return (ratio - 1.0) + drift / MAX_MOVE_WITHIN_SHOT
        # A cut reframes everybody, so horizontal position cannot carry identity through it — but heads
        # in a locked-off wide shot really do keep their places, and without some preference the first
        # four people in the dict order would be handed out to the four detections in whatever order
        # they came. So position stays a tie-break only, never a veto.
        return (
            (ratio - 1.0) * 2.0
            + abs(hy - _median([c[1] for c in person.centers])) / MAX_CENTRE_GAP
            + CUT_X_TIE_WEIGHT * min(1.0, abs(hx - _median([c[0] for c in person.centers])) / MAX_CENTRE_GAP)
        )

    def observe(
        self,
        box: list[float],
        frame_w: int,
        frame_h: int,
        dt: float,
        t: float,
        kind: str = "face",
        within_shot: bool = False,
    ) -> int:
        """Cluster one detection into a person. Detections from the same frame never merge.

        Identity is judged on the head — its width and how far down the frame it sits — because a
        multicam cut or a handheld pan moves people sideways freely. Inside a shot, where nothing has
        cut, sideways position is used too, and that is what keeps a row of similar heads apart. The
        same-instant gate is still needed on top: two people in one shot agree on every appearance axis,
        and only the fact that they were measured at the same instant says they are not one person.
        """
        head = head_box(box, kind)
        hx, hy = head[0] + head[2] / 2.0, head[1] + head[3] / 2.0
        if head[2] <= 0 or head[3] <= 0:
            return -1
        nx, ny = hx / max(1.0, frame_w), hy / max(1.0, frame_h)
        best_key: int | None = None
        best_dist = math.inf
        for key, person in self.people.items():
            if self._seen_at.get(key) == t:
                continue
            dist = self._distance(person, head[2], nx, ny, within_shot)
            if dist < best_dist:
                best_key, best_dist = key, dist
        if best_key is None or best_dist > 1.6:
            best_key = self._next_key
            self._next_key += 1
            self.people[best_key] = Person(key=best_key)
        person = self.people[best_key]
        person.widths.append(head[2])
        person.centers.append((nx, ny))
        person.screen_sec += dt
        self._seen_at[best_key] = t
        return best_key

    def main_speaker(self) -> int | None:
        """The person who holds the floor the longest; screen time settles it when nobody speaks.

        There is deliberately no screen-share veto here. Real multicam footage splits one host across
        several identities, so even the genuine main speaker's share of the clip looks small, and a veto
        on that share used to hand back None — which silently switched the whole framing rule off.
        """
        if not self.people:
            return None
        ranked = sorted(
            self.people.values(),
            key=lambda p: (round(p.talk_sec, 3), round(p.screen_sec, 3)),
            reverse=True,
        )
        top = ranked[0]
        return top.key if (top.talk_sec > 0 or top.screen_sec > 0) else None

    def report(self, main_key: int | None) -> dict[str, Any]:
        return {
            "main_speaker": main_key,
            "people": [
                {
                    "key": p.key,
                    "screen_sec": round(p.screen_sec, 2),
                    "talk_sec": round(p.talk_sec, 2),
                    "median_face_w": round(_median(p.widths), 1),
                    "is_main": p.key == main_key,
                }
                for p in sorted(self.people.values(), key=lambda q: (-q.talk_sec, -q.screen_sec))
            ],
        }


@dataclass
class MouthMemory:
    """Last mouth patch per person, so the next sample can be differenced against it."""

    patches: dict[int, Any] = field(default_factory=dict)

    def activity(
        self, key: int, box: list[float], gray, frame_w: int, frame_h: int, scale: float
    ) -> float | None:
        import cv2

        x, y, w, h = [v / scale for v in box]
        if w < MIN_MOUTH_HEAD_W:
            # Too small to read, and no threshold on noise makes it legible.
            return None
        # Podcast Mic-Occlusion & Profile Resilience:
        # Sample from upper-lip/cheek (0.48*h) to jaw (0.92*h) and mask out stationary near-black
        # microphone foam pixels (< 22 luma) when computing frame-to-frame speaking motion.
        cx, mouth_y = x + w / 2.0, y + h * 0.70
        half_w, half_h = w * 0.36, h * 0.22
        x0 = int(max(0.0, min(cx - half_w, frame_w - 2)))
        x1 = int(max(x0 + 2, min(cx + half_w, frame_w)))
        y0 = int(max(0.0, min(mouth_y - half_h, frame_h - 2)))
        y1 = int(max(y0 + 2, min(mouth_y + half_h, frame_h)))
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            return None
        probe = cv2.resize(patch, (PATCH_SIDE, PATCH_SIDE), interpolation=cv2.INTER_AREA)
        prev = self.patches.get(key)
        self.patches[key] = probe
        if prev is None:
            return None
        diff = cv2.absdiff(probe, prev)
        skin_mask = (probe > 22) | (prev > 22)
        if int(skin_mask.sum()) >= (PATCH_SIDE * PATCH_SIDE) // 4:
            return float(diff[skin_mask].mean()) / 255.0
        return float(diff.mean()) / 255.0


class TalkWatcher:
    """Mouth motion per person, bucketed into fixed windows next to the audio envelope."""

    def __init__(self, start_t: float | None = None) -> None:
        self.start_t = start_t
        self.totals: dict[int, dict[int, float]] = {}
        self.counts: dict[int, dict[int, int]] = {}

    def window(self, t: float) -> int:
        base = self.start_t if self.start_t is not None else t
        return max(0, int((t - base) / WINDOW_SECONDS))

    def observe(self, key: int, t: float, delta: float | None) -> None:
        # The clock starts on the first sample the caller saw, not the first sample with mouth motion:
        # these window numbers have to line up with the ones the framing reads them from.
        if self.start_t is None:
            self.start_t = t
        if delta is None:
            return
        w = self.window(t)
        totals = self.totals.setdefault(key, {})
        counts = self.counts.setdefault(key, {})
        totals[w] = totals.get(w, 0.0) + delta
        counts[w] = counts.get(w, 0) + 1

    def activity(self) -> dict[int, dict[int, float]]:
        return {
            key: {w: total / self.counts[key][w] for w, total in totals.items() if self.counts[key].get(w)}
            for key, totals in self.totals.items()
        }


@dataclass
class TalkWindow:
    keys: frozenset[int]
    overlap: bool
    loud: bool
    # Loud audio, several faces on screen, and the mouths do not separate them: the source's own cut
    # is the best read of the floor available, so the framing rule abstains instead of guessing.
    ambiguous: bool = False

    @property
    def talking(self) -> bool:
        return bool(self.keys)


def settle_floor(
    talks: dict[int, TalkWindow],
    hold: int = FLOOR_HOLD_WINDOWS,
    present: dict[int, set[int]] | None = None,
) -> dict[int, TalkWindow]:
    """A new talker has to be read twice before the camera turns to them.

    One window's worth of mouth motion is the same class of noise the tracker's sticky speaker exists to
    reject, and a framing that swings to it and back inside a second reads as a flicker, not a cut.
    An overlap passes through untouched: showing everybody is not a change of subject to be earned.
    """
    out: dict[int, TalkWindow] = {}
    current: frozenset[int] = frozenset()
    pending: frozenset[int] | None = None
    run = 0
    for w in sorted(talks):
        tw = talks[w]
        if not tw.talking or tw.overlap:
            # Silence hands the framing back to the tracker, so the next speaker starts a fresh count.
            current, pending, run = frozenset(), None, 0
            out[w] = tw
            continue
        # A hard camera cut where the prior speaker exited shot must immediately release floor ownership
        # instead of attributing the window to an absent identity from the previous shot.
        if current and present is not None and not current.issubset(present.get(w, set())):
            current, pending, run = tw.keys, None, 0
            out[w] = tw
            continue
        if tw.keys == current:
            pending, run = None, 0
            out[w] = tw
            continue
        if not current:
            current, pending, run = tw.keys, None, 0
            out[w] = tw
            continue
        pending, run = (tw.keys, run + 1) if tw.keys == pending else (tw.keys, 1)
        if run >= hold:
            current, pending, run = tw.keys, None, 0
            out[w] = tw
        else:
            out[w] = TalkWindow(current, False, tw.loud)
    return out


def visible_by_window(samples: list[dict[str, Any]], base_t: float | None) -> dict[int, set[int]]:
    """Which identities are on screen in each window, on the same clock the mouth buckets use.

    Attribution needs this to keep two different things apart: nobody talking, and somebody talking who
    is not in shot. Only the second is a framing failure, and an off-screen voice cannot be framed anyway.
    """
    out: dict[int, set[int]] = {}
    for sample in samples:
        t = float(sample.get("t", 0.0))
        w = max(0, int((t - (t if base_t is None else base_t)) / WINDOW_SECONDS))
        keys = out.setdefault(w, set())
        for person in sample.get("people", []):
            keys.add(int(person["key"]))
    return out


def attribute_talkers(
    watcher: TalkWatcher,
    registry: PersonRegistry,
    loudness: list[float],
    windows: int,
    present: dict[int, set[int]] | None = None,
) -> dict[int, TalkWindow]:
    """Read the floor for each window: who is talking, and whether more than one has it at once.

    Each person is measured against their own range, so a close-up that fills the frame and a wide shot
    at the back of the room are compared on the same scale instead of the big face always winning. Then
    three cases are kept apart, because collapsing them is what made every panel window demand a group
    shot: one mouth moving is a verdict, two or three moving alike is the floor being shared, and a crowd
    of four or more that cannot be told apart is something this pass cannot see — so it abstains and the
    source's own cut stands. Verdicts are then settled, so what the caller reads is who had the floor, not
    one window.
    """
    activity = watcher.activity()
    peaks = {key: _percentile(list(series.values()), 0.95) for key, series in activity.items()}
    loudest = max(peaks.values(), default=0.0)
    # Measuring each person against their *own* loudest window is what makes a close-up and a face at the
    # back of the room comparable, but it also lets a still face normalise its own noise into a turn. So
    # the same-scale figures only ever compete with someone whose mouth is genuinely moving in absolute
    # terms too; alone, a quiet speaker does not need to out-shout anybody to be believed.
    scaled: dict[int, dict[int, float]] = {}
    for key, series in activity.items():
        peak = peaks.get(key, 0.0)
        if peak > 0.0:
            scaled[key] = {w: v / peak for w, v in series.items()}
    floor_peak = MOUTH_SHARE_OF_LOUDEST * loudest

    out: dict[int, TalkWindow] = {}
    for w in range(windows):
        loud = (loudness[w] if w < len(loudness) else 0.0) >= AUDIO_FLOOR
        if not loud:
            # A quiet window has no audio to attribute, so nobody is called a talker.
            out[w] = TalkWindow(frozenset(), False, False)
            continue
        active = {key: s[w] for key, s in scaled.items() if s.get(w, 0.0) > MOUTH_ACTIVE}
        if present is not None:
            # A mouth that is not in shot cannot be framed anyway, so it does not get to hold the floor.
            active = {k: v for k, v in active.items() if k in present.get(w, set())}
        if not active:
            crowd = present is not None and len(present.get(w, set())) > 1
            out[w] = TalkWindow(frozenset(), False, loud, ambiguous=crowd)
            continue
        if len(active) == 1:
            out[w] = TalkWindow(frozenset(active), False, loud)
            continue
        rivals = {k: v for k, v in active.items() if peaks.get(k, 0.0) >= floor_peak} or active
        ranked = sorted(rivals.items(), key=lambda kv: -kv[1])
        if len(ranked) == 1:
            out[w] = TalkWindow(frozenset([ranked[0][0]]), False, loud)
            continue
        top = ranked[0][1]
        # Everyone whose mouth is within margin of the loudest one is holding the floor *with* them, and
        # naming all of them is what puts all of them in the frame. One person a clear way ahead takes it.
        crew = [key for key, score in ranked if (top - score) / (top + score) < OVERLAP_MARGIN]
        if len(crew) == 1:
            out[w] = TalkWindow(frozenset(crew), False, loud)
        elif len(crew) <= MAX_OVERLAP_TALKERS:
            out[w] = TalkWindow(frozenset(crew), True, loud)
        else:
            # Four or more and the signal cannot say which of them the editor cut to. Guessing here would
            # swing the camera away from whoever is actually on the source's own cut, so the rule defers.
            out[w] = TalkWindow(frozenset(), False, loud, ambiguous=True)
    out = settle_floor(out, present=present)
    for w, tw in out.items():
        for key in tw.keys:
            person = registry.people.get(key)
            if person:
                person.talk_sec += WINDOW_SECONDS
    return out


# --- What the viewer is actually shown ---------------------------------------------------------
#
# Every decision below is checked against the rects the renderer really draws, not against an idealised
# crop. The renderer clamps a crop into the source and anchors the face about a third of the way down,
# and each layout hands its panels a different output shape, so a split panel shows a far wider slice
# of the source than a full-frame crop does. Getting this wrong makes the coverage number meaningless.

OUT_W = 1080
OUT_H = 1920


def head_zone(person: dict[str, Any]) -> list[float]:
    """The part of a detection that has to be on screen for that person to be *shown*."""
    return head_box(person["box"], str(person.get("kind") or "face"))


def panel_geometry(mode: str, frame_w: int, frame_h: int, zoom: float = 1.0) -> list[dict[str, float]]:
    """Per-panel output shape and the source slice it shows, mirroring drawCroppedSubject.

    A panel of output size ow x oh shows a source rect of oh/zoom tall and (oh/zoom)(ow/oh) wide,
    widened from the height unless that would run past the source, in which case it is fitted from the
    width instead. Two half-height split panels therefore see about twice the horizontal span of one
    full-frame crop, which is exactly why a pair that will not fit a crop often fits a split.
    """
    zoom = max(1.0, zoom)
    shapes = {
        "split": [(OUT_W, OUT_H / 2.0), (OUT_W, OUT_H / 2.0)],
        "editorial": [(OUT_W, OUT_H - min(OUT_W * frame_h / frame_w, OUT_H * 0.45))],
        "fit": [],
    }.get(mode, [(OUT_W, OUT_H)])
    panels: list[dict[str, float]] = []
    for ow, oh in shapes:
        sh = frame_h / zoom
        sw = sh * (ow / oh)
        if sw > frame_w:
            sw = frame_w / zoom
            sh = sw * (oh / ow)
        panels.append({"cw": sw / frame_w, "ch": sh / frame_h, "anchor": 0.36, "height_share": oh / OUT_H})
    return panels


def source_rect(cx: float, cy: float, cw: float, ch: float, anchor: float = 0.36) -> tuple[float, float, float, float]:
    """The normalised source rect a centre lands on, after the renderer's own in-bounds clamp."""
    x0 = min(max(cx - cw / 2.0, 0.0), max(0.0, 1.0 - cw))
    y0 = min(max(cy - ch * anchor, 0.0), max(0.0, 1.0 - ch))
    return (x0, y0, x0 + cw, y0 + ch)


def covered(person: dict[str, Any], rect: tuple[float, float, float, float], frame_w: int, frame_h: int) -> float:
    """Share of this person's head that one source rect keeps."""
    x, y, w, h = head_zone(person)
    if w <= 0 or h <= 0:
        return 0.0
    x0, y0, x1, y1 = rect[0] * frame_w, rect[1] * frame_h, rect[2] * frame_w, rect[3] * frame_h
    ix = max(0.0, min(x + w, x1) - max(x, x0))
    iy = max(0.0, min(y + h, y1) - max(y, y0))
    return (ix * iy) / (w * h)


def shown(person: dict[str, Any], layout: dict[str, Any], frame_w: int, frame_h: int) -> bool:
    """Is this person on screen in this layout?

    A contained shot (fit) and the context band of an editorial shot both carry the whole source, so
    everyone standing in the room is in the frame — small, but visible. That is what makes them the
    fallback that can always honour the rule.
    """
    mode = layout.get("mode") or "fill"
    if mode in ("fit", "editorial"):
        return True
    zoom = max(1.0, float(layout.get("zoom") or 1.0))
    panels = panel_geometry(mode, frame_w, frame_h, zoom)
    centres = [(float(layout["x"]), float(layout["y"]))]
    if mode == "split":
        centres.append((float(layout.get("x2") or layout["x"]), float(layout.get("y2") or layout["y"])))
    for centre, panel in zip(centres, panels):
        rect = source_rect(centre[0], centre[1], panel["cw"], panel["ch"], panel["anchor"])
        if covered(person, rect, frame_w, frame_h) >= MIN_HEAD_VISIBLE:
            return True
    return False


def _axis_interval(b0: float, b1: float, lead: float, tail: float, need: float) -> tuple[float, float]:
    """Centres on one axis that cover at least `need` source units of a box.

    The crop runs from `c - lead` to `c + tail`. When `need` fits inside both the box and the crop, the
    centres that satisfy it are exactly this interval; the caller has already ruled out the case where
    it does not, so a head wider than any crop cannot slip through as feasible.
    """
    return (b0 + need - tail, b1 - need + lead)


def group_target(
    people: list[dict[str, Any]],
    cw: float,
    ch: float,
    bias: tuple[float, float],
    frame_w: int,
    frame_h: int,
    anchor: float = 0.36,
) -> tuple[float, float] | None:
    """One crop that keeps every listed person's head on screen, biased toward whoever owns the audio.

    Returns the centre, or None when no single crop can hold them all — two people at opposite ends of a
    wide shot cannot, and that is the case the split panels and the context band exist for. Each axis
    has to keep its own share so the two together add up to MIN_HEAD_VISIBLE of the head area.
    """
    share = math.sqrt(MIN_HEAD_VISIBLE)
    x_lo, x_hi = cw / 2.0, 1.0 - cw / 2.0
    y_lo, y_hi = ch * anchor, 1.0 - ch * (1.0 - anchor)
    for person in people:
        x, y, w, h = head_zone(person)
        bw, bh = w / max(1, frame_w), h / max(1, frame_h)
        if bw <= 0 or bh <= 0:
            continue
        if share * bw > cw or share * bh > ch:
            return None
        ix = _axis_interval(x / frame_w, (x + w) / frame_w, cw / 2.0, cw / 2.0, share * bw)
        iy = _axis_interval(y / frame_h, (y + h) / frame_h, ch * anchor, ch * (1.0 - anchor), share * bh)
        x_lo, x_hi = max(x_lo, ix[0]), min(x_hi, ix[1])
        y_lo, y_hi = max(y_lo, iy[0]), min(y_hi, iy[1])
    if x_lo > x_hi or y_lo > y_hi:
        return None
    return (
        max(0.05, min(0.95, min(max(bias[0], x_lo), x_hi))),
        max(0.08, min(0.92, min(max(bias[1], y_lo), y_hi))),
    )


def _centre(person: dict[str, Any], frame_w: int, frame_h: int) -> tuple[float, float]:
    x, y, w, h = head_zone(person)
    return ((x + w / 2.0) / max(1, frame_w), (y + h / 2.0) / max(1, frame_h))


# A two-panel split carries exactly two people. Asking it to hold a six-person shout is what made the
# export alternate split and wide inside one continuous moment, because the detector reported three
# talkers in one 0.4 s window and two in the next.
MAX_SPLIT_TALKERS = 2


def crowd_runs(talker_counts: list[int]) -> list[tuple[int, int]]:
    """Which stretches of the clip are a crowd roar rather than a two-person exchange.

    A shout is one editorial event, and the decision about it has to be made once for the whole of it.
    Any window with more talkers than a split can carry seeds a burst, and the burst then grows across
    every neighbouring window that still has more than one voice in it — that wobble between "three"
    and "two" is the detector losing a head for a fifth of a second, not the room changing the subject.
    """
    seeds = [i for i, n in enumerate(talker_counts) if n > MAX_SPLIT_TALKERS]
    runs: list[tuple[int, int]] = []
    for i in seeds:
        begin = end = i
        while begin > 0 and talker_counts[begin - 1] >= 2:
            begin -= 1
        while end < len(talker_counts) - 1 and talker_counts[end + 1] >= 2:
            end += 1
        span = (begin, end + 1)
        if runs and runs[-1][1] >= span[0]:
            runs[-1] = (runs[-1][0], max(runs[-1][1], span[1]))
        else:
            runs.append(span)
    return runs


def _crowd_layout(
    talkers: list[dict[str, Any]],
    people: list[dict[str, Any]],
    lead: dict[str, Any],
    frame_w: int,
    frame_h: int,
    zoom: float,
) -> dict[str, Any]:
    """One framing for a whole crowd burst — everybody who matters in a single panel, never a 2-up.

    Nobody is taking turns in a shout, so there is no speaker to cut to and no conversation to split.
    The subject is the group, and the honest way to show it is the wide crop that keeps all of them;
    when the group is too spread for any single crop, the contained landscape frame is the one that
    still shows the moment instead of cutting six people down to two.
    """
    for panel in panel_geometry("fill", frame_w, frame_h, zoom):
        bias = _centre(lead, frame_w, frame_h)
        for group, shot in ((talkers, "talkers"), (people, "group")):
            if not group:
                continue
            target = group_target(group, panel["cw"], panel["ch"], bias, frame_w, frame_h, panel["anchor"])
            if target is not None:
                return {"mode": "fill", "x": target[0], "y": target[1], "shot": shot}
    return {"mode": "fit", "shot": "wide"}


def candidates_for(
    talkers: list[dict[str, Any]],
    people: list[dict[str, Any]],
    lead: dict[str, Any],
    main_key: int | None,
    frame_w: int,
    frame_h: int,
    zoom: float,
    allow_editorial: bool = False,
) -> list[dict[str, Any]]:
    """Every layout that could hold this moment, best-looking first.

    The order is the editorial preference, not a guess about feasibility: a natural single crop of the
    group beats a manufactured seam, a split beats a picture-in-picture of nothing, and the wide
    contained shot is the last thing that still honours the rule rather than a first choice. Feasibility
    is then tested, and the first candidate that satisfies it wins, so a layout that would cut a speaker
    off is never chosen and repaired later.
    """
    panels = panel_geometry("fill", frame_w, frame_h, zoom)
    if not panels:
        return [{"mode": "fit"}]
    cw, ch, anchor = panels[0]["cw"], panels[0]["ch"], panels[0]["anchor"]
    bias = _centre(lead, frame_w, frame_h)
    out: list[dict[str, Any]] = []

    if len(people) >= 2:
        target = group_target(people, cw, ch, bias, frame_w, frame_h, anchor)
        if target is not None:
            out.append({"mode": "fill", "x": target[0], "y": target[1], "shot": "group"})
    target = group_target(talkers, cw, ch, bias, frame_w, frame_h, anchor)
    if target is not None:
        out.append({"mode": "fill", "x": target[0], "y": target[1], "shot": "talkers"})

    if 2 <= len(talkers) <= MAX_SPLIT_TALKERS:
        split = panel_geometry("split", frame_w, frame_h, zoom)
        if len(split) == 2:
            first = group_target([lead], split[0]["cw"], split[0]["ch"], bias, frame_w, frame_h, split[0]["anchor"])
            partners = [p for p in talkers if p is not lead]
            second_bias = _centre(partners[0], frame_w, frame_h)
            second = group_target(partners, split[1]["cw"], split[1]["ch"], second_bias, frame_w, frame_h, split[1]["anchor"])
            if first and second and abs(second[0] - first[0]) >= 0.25:
                # Rule 2 (Middle-Subject Veto & Zero Duplicate Panelists):
                # Do not propose a 2-up split if a 3rd person sits horizontally between the two speakers
                # such that both panels would duplicate the middle panelist(s).
                x_left, x_right = min(first[0], second[0]), max(first[0], second[0])
                has_middle_bystander = any(
                    p is not lead
                    and p not in partners
                    and (x_left + 0.08) < _centre(p, frame_w, frame_h)[0] < (x_right - 0.08)
                    for p in people
                )
                if not has_middle_bystander:
                    out.append(
                        {
                            "mode": "split",
                            "x": first[0],
                            "y": first[1],
                            "x2": second[0],
                            "y2": second[1],
                            "shot": "two_panel",
                        }
                    )

    # Only propose "editorial" when the clip is already explicitly in editorial mode:
    # auto-selecting "editorial" on a 3-5 person studio panel duplicates the same speaker and
    # middle panelists on both the top crop and the bottom full-room strip (0:09-0:12, 0:33-0:36).
    if allow_editorial:
        editorial = panel_geometry("editorial", frame_w, frame_h, zoom)
        if editorial:
            panel = editorial[0]
            aim = group_target([lead], panel["cw"], panel["ch"], bias, frame_w, frame_h, panel["anchor"]) or bias
            out.append({"mode": "editorial", "x": aim[0], "y": aim[1], "shot": "context"})

    out.append({"mode": "fit", "shot": "wide"})
    return out


def apply_speaker_priority(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    main_key: int | None,
    talks: dict[int, TalkWindow],
    frame_w: int,
    frame_h: int,
    min_hold_sec: float = 0.0,
    split_min_hold_sec: float = 0.0,
) -> dict[str, Any]:
    """Hold the frame to the speaking rule: the floor decides who is on screen, the geometry decides how.

    `samples[i]` says which people the detectors saw at `track_points[i]`, and `talks` gives one verdict
    per 0.4 s window — as short a span as the audio test can be honest about, so the layout cannot
    change faster than the evidence behind it. Rules, in priority order:

      1. while the main speaker talks he is on screen;
      2. while somebody else talks, that person is on screen — he waits his turn;
      3. while two or more talk over each other, all of them are on screen together.

    The framing the clip is already in is the first candidate, so nothing moves unless one of those
    three rules is actually being broken; the rest of the ladder is ordered by how natural the shot
    looks, and the first entry that satisfies the rule wins. Then the shot-hold is applied to the
    *result*, and a fold that would break one of the rules is refused rather than obeyed.
    """
    if not track_points or len(track_points) != len(samples) or not samples:
        return {"error": "track and sample counts do not match"}

    start = float(samples[0]["t"])
    original = [{k: p.get(k) for k in ("mode", "x", "y", "x2", "y2", "zoom")} for p in track_points]
    counts: dict[str, Any] = {
        "windows": len(talks),
        "overlap_windows": sum(1 for tw in talks.values() if tw.overlap),
        "ambiguous_windows": sum(1 for tw in talks.values() if tw.ambiguous),
        "silent_windows": sum(1 for tw in talks.values() if not tw.talking),
        "shot": {"group": 0, "talkers": 0, "two_panel": 0, "context": 0, "wide": 0, "untouched": 0},
        "required": {"floor": 0, "presumed_main": 0, "silent": 0, "no_verdict": 0},
        "far_reaims": 0,
        "far_reaim_max_crop_widths": 0.0,
    }

    verdicts = [_required_in_sample(s, talks, start, main_key) for s in samples]
    for _, _, reason in verdicts:
        counts["required"][reason] = counts["required"].get(reason, 0) + 1

    # A shout is one editorial event, so it gets one framing decided over the whole of it. Deciding it
    # per window is what produced the split/fill argument: the detector named three voices one window
    # and two the next, and each answer looked reasonable for its own 0.4 s.
    burst_layout: dict[int, dict[str, Any]] = {}
    bursts = crowd_runs([len(v[0]) if v[0] else 0 for v in verdicts])
    for begin, end in bursts:
        voices: dict[int, dict[str, Any]] = {}
        room: dict[int, dict[str, Any]] = {}
        for i in range(begin, end):
            talkers, people, _ = verdicts[i]
            for p in talkers or []:
                voices[int(p["key"])] = p
            for p in people:
                room.setdefault(int(p["key"]), p)
        if not voices:
            continue
        lead = voices.get(main_key) if main_key is not None else None
        if lead is None:
            lead = voices[min(voices)]
        layout = _crowd_layout(
            list(voices.values()),
            list(room.values()),
            lead,
            frame_w,
            frame_h,
            float(track_points[begin].get("zoom") or 1.0),
        )
        for i in range(begin, end):
            burst_layout[i] = layout
    counts["crowd_bursts"] = len(bursts)

    for i, (pt, s) in enumerate(zip(track_points, samples)):
        talkers, people, _ = verdicts[i]
        if talkers is None:
            continue
        if not talkers:
            counts["shot"]["untouched"] += 1
            continue
        lead = next((p for p in talkers if p.get("key") == main_key), talkers[0])
        zoom = float(pt.get("zoom") or 1.0)
        # Where the clip already is, if it already honours the rule. That is the least intrusive answer
        # and it keeps a framing the tracker chose for good reason from being rewritten every time a
        # listener happens to fall silent.
        current = {"mode": pt.get("mode") or "fill", "x": pt["x"], "y": pt["y"], "shot": "untouched"}
        if current["mode"] == "split":
            current["x2"], current["y2"] = pt.get("x2"), pt.get("y2")
        if i in burst_layout:
            winner = burst_layout[i]
        else:
            layouts = [current] + candidates_for(
                talkers, people, lead, main_key, frame_w, frame_h, zoom,
                allow_editorial=(current["mode"] == "editorial"),
            )
            visible = [c for c in layouts if all(shown(p, c, frame_w, frame_h) for p in talkers)]
            # Of the layouts that keep every speaker in shot, prefer the ones that make whoever owns the audio
            # the *subject* of the frame rather than a head somewhere inside a context band or a wide shot.
            # "Show him" means show him, and a host the viewer has to go looking for does not count.
            front = [c for c in visible if prominently_shown(lead, c, frame_w, frame_h)]
            # Only change a clip's layout when that layout cannot make the owner of the audio the subject at
            # all. An editorial clip whose window is pointed at the wrong person needs its aim fixed, not its
            # layout replaced — the band already carries the room, which is why the clip is editorial.
            same_mode = [c for c in front if c["mode"] == current["mode"]]
            winner = layouts[-1]
            for pool in (same_mode, front, visible):
                if pool:
                    winner = pool[0]
                    break
        if _same_aim(pt, winner):
            counts["shot"]["untouched"] += 1
            continue
        # The tracker's own aim at this sample is where the rendered camera already is, so this is the
        # size of the step the rule is asking the pursuit law to make.
        step = _reaim_cost(pt, winner, frame_w, frame_h, zoom)
        if step > MAX_REAIM_GAP:
            counts["far_reaims"] += 1
            counts["far_reaim_max_crop_widths"] = max(counts["far_reaim_max_crop_widths"], round(step, 3))
        pt["mode"] = winner["mode"]
        pt["x"], pt["y"] = winner.get("x", pt["x"]), winner.get("y", pt["y"])
        if winner["mode"] == "split":
            pt["x2"], pt["y2"] = winner["x2"], winner["y2"]
        counts["shot"][winner.get("shot", "wide")] += 1

    exempt = _fold_short_layout_runs(
        track_points, samples, talks, start, frame_w, frame_h, min_hold_sec, main_key, split_min_hold_sec
    )
    counts["hold_exempt_runs"] = exempt
    # Both halves of the same measurement: what the tracker's framing did for the speakers, and what
    # this rule did. Only the second one is allowed to be near 1.0.
    counts["coverage_after"] = coverage_of(track_points, samples, talks, frame_w, frame_h, main_key)
    counts["coverage_before"] = coverage_of(original, samples, talks, frame_w, frame_h, main_key)
    # Read off the delivered track after the hold has had its say, so the number is the guarantee rather
    # than the intention: how much of the time the rule demanded the host, was he actually on screen.
    counts["main_lock"] = main_lock_of(track_points, samples, talks, main_key, start, frame_w, frame_h)
    return {**counts, "main_speaker": main_key}


def main_lock_of(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, TalkWindow],
    main_key: int | None,
    start: float,
    frame_w: int,
    frame_h: int,
) -> dict[str, Any]:
    """Rule 1 as a number: of the samples that demanded the host on screen, how many showed him.

    Rule 2 is why this counts only the samples where the host *is* the requirement. When somebody else
    holds the floor they take the frame, and holding the host in shot through that would be the rule
    working against the clip rather than for it.
    """
    if main_key is None:
        return {"required_samples": 0, "on_screen": 0, "coverage": None, "misses": []}
    required = shown_count = 0
    misses: list[float] = []
    for pt, s in zip(track_points, samples):
        need, _, _ = _required_in_sample(s, talks, start, main_key)
        host = next((p for p in (need or []) if int(p.get("key", -1)) == int(main_key)), None)
        if host is None:
            continue
        required += 1
        if shown(host, pt, frame_w, frame_h):
            shown_count += 1
        elif len(misses) < 8:
            misses.append(round(float(s["t"]), 2))
    return {
        "required_samples": required,
        "on_screen": shown_count,
        "coverage": round(shown_count / required, 4) if required else None,
        "misses": misses,
    }


# Which rule says the frame owes somebody a place in it. `floor` is a read verdict; `presumed_main` is
# rule 1 running with no verdict behind it — loud audio, the host standing right there, mouths this pass
# cannot separate — where framing him is the answer that is right far more often than leaving it to
# whatever the tracker happened to be pointed at.
FLOOR = "floor"
PRESUMED = "presumed_main"


def _required_in_sample(
    s: dict[str, Any],
    talks: dict[int, TalkWindow],
    start: float,
    main_key: int | None,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]], str]:
    """The people this sample must show, and which rule put them there.

    Returns (None, ...) for a sample with no window verdict so the caller leaves the tracker's framing
    alone; objects are not speakers and take no part in this rule.
    """
    people = [
        p
        for p in s.get("people", [])
        if p.get("key", -1) >= 0 and p.get("box") and p.get("kind") != "object"
    ]
    w = max(0, int((float(s["t"]) - start) / WINDOW_SECONDS))
    talk = talks.get(w)
    if talk is None:
        return None, people, "no_verdict"
    by_key = {int(p["key"]): p for p in people}
    if talk.talking:
        required = [by_key[k] for k in sorted(talk.keys) if k in by_key]
        if not required and main_key is not None and main_key in by_key:
            # The floor belongs to a voice that is not in shot while the host is: he carries the frame.
            return [by_key[int(main_key)]], people, PRESUMED
        return required, people, FLOOR
    if talk.loud and talk.ambiguous and main_key is not None and main_key in by_key:
        return [by_key[int(main_key)]], people, PRESUMED
    return [], people, "silent"


def prominently_shown(person: dict[str, Any], layout: dict[str, Any], frame_w: int, frame_h: int) -> bool:
    """Is this person a subject of the frame, rather than a head somewhere inside a context band?

    Only ever asked of a layout that already passes `shown`: a contained wide shot cannot make anybody
    prominent whatever it is aimed at, and an editorial window can, which is the difference the caller
    acts on.
    """
    mode = layout.get("mode") or "fill"
    if mode == "fit" or "x" not in layout:
        return False
    if mode != "editorial":
        return shown(person, layout, frame_w, frame_h)
    panels = panel_geometry(mode, frame_w, frame_h, float(layout.get("zoom") or 1.0))
    centres = [(float(layout["x"]), float(layout["y"]))]
    for centre, panel in zip(centres, panels):
        rect = source_rect(centre[0], centre[1], panel["cw"], panel["ch"], panel["anchor"])
        if covered(person, rect, frame_w, frame_h) >= MIN_HEAD_VISIBLE:
            return True
    return False


def _reaim_cost(pt: dict[str, Any], layout: dict[str, Any], frame_w: int, frame_h: int, zoom: float) -> float:
    """How far this layout asks the camera to slide, in crop widths of the panel that moves.

    Nothing here refuses a large value: the engine's own pursuit law spreads an anchor change over
    several samples, so a far target becomes a fast glide rather than a jump, and the anchor is exactly
    what that law was missing. This is reported so the size of the moves the rule asks for is measured
    instead of assumed.
    """
    mode = layout.get("mode") or "fill"
    if mode == "fit" or mode != (pt.get("mode") or "fill"):
        return 0.0
    panels = panel_geometry(mode, frame_w, frame_h, zoom)
    if not panels:
        return 0.0
    travel = max(abs(float(layout.get(axis, pt[axis])) - float(pt[axis])) for axis in ("x", "y"))
    return travel / max(1e-3, panels[0]["cw"])


def _same_aim(pt: dict[str, Any], layout: dict[str, Any]) -> bool:
    """Already framed like this: leave the camera alone rather than move it for the sake of it."""
    if (pt.get("mode") or "fill") != layout["mode"]:
        return False
    if layout["mode"] == "fit":
        return True
    return abs(float(pt["x"]) - float(layout.get("x", pt["x"]))) < 0.02 and abs(
        float(pt["y"]) - float(layout.get("y", pt["y"]))
    ) < 0.02


def _fold_short_layout_runs(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, TalkWindow],
    start: float,
    frame_w: int,
    frame_h: int,
    min_hold_sec: float,
    main_key: int | None = None,
    split_min_hold_sec: float = 0.0,
) -> int:
    """Merge layouts that do not survive the clip's own hold, unless merging would hide a speaker.

    Returns how many short runs were refused a fold, which is how often the speaking rule and the
    smoothness rule disagree on this footage.
    """
    if min_hold_sec <= 0:
        return 0
    mode_of = lambda i: track_points[i].get("mode") or "fill"
    # A layout is on screen until the next keyframe changes it, so a run's final second belongs to the
    # run after it; measuring to the run's own last keyframe reads a one-sample twitch as a long shot.
    span_of = lambda run: float(
        track_points[min(run[1], len(track_points) - 1)]["t"]
    ) - float(track_points[run[0]]["t"])
    split_floor = split_min_hold_sec if split_min_hold_sec > min_hold_sec else min_hold_sec
    floor_of = lambda run: split_floor if mode_of(run[0]) == "split" else min_hold_sec
    exempt = 0
    protected: set[int] = set()
    for _ in range(2 * len(track_points) + 64):
        runs: list[list[int]] = []
        for i in range(len(track_points)):
            if runs and mode_of(runs[-1][1] - 1) == mode_of(i):
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        if len(runs) < 2:
            break
        short = next(
            (
                k
                for k in range(len(runs))
                if runs[k][0] not in protected and span_of(runs[k]) < floor_of(runs[k])
            ),
            None,
        )
        if short is None:
            break
        begin, end = runs[short]
        first_frame = begin
        own_mode = mode_of(begin)
        anchor = track_points[begin]
        floor = floor_of(runs[short])
        # A run at either end of the clip has one neighbour; an interior run has two.
        prev_run = runs[short - 1] if short > 0 else None
        next_run = runs[short + 1] if short < len(runs) - 1 else None
        sides = [r for r in (prev_run, next_run) if r is not None and r[0] < r[1]]
        if not sides:
            protected.add(begin)
            continue

        def absorb(side: list[int]) -> bool:
            """Hand the frame at this side of the short run over to it. False when that is not possible."""
            nonlocal begin, end
            if side is prev_run:
                i = side[1] - 1
                if i < side[0] or i + 1 != begin:
                    return False
            else:
                i = side[0]
                if i >= side[1] or i != end:
                    return False
            if not _fold_is_safe(
                i, anchor, own_mode, track_points, samples, talks, start, frame_w, frame_h, main_key
            ):
                return False
            pt = track_points[i]
            pt["mode"] = own_mode
            if own_mode == "split":
                pt["x"], pt["y"] = anchor["x"], anchor["y"]
                pt["x2"], pt["y2"] = anchor.get("x2"), anchor.get("y2")
            else:
                aim = _refold_aim(pt, samples[i], own_mode, frame_w, frame_h)
                if aim:
                    pt["x"], pt["y"] = aim
            if side is prev_run:
                side[1] -= 1
                begin -= 1
            else:
                side[0] += 1
                end += 1
            return True

        # Deleting the twitch is what makes a clip read as edited, and it is the only move that can
        # bring the number of layouts down to what the clip's runtime can actually hold, so it goes
        # first: the run earns the right to stand only when folding it into the neighbour that lasts
        # longest would put a speaker out of shot.
        donor = max(sides, key=span_of)
        keeper_mode = mode_of(donor[0])
        first = track_points[donor[0]]
        if all(
            _fold_is_safe(
                i, first, keeper_mode, track_points, samples, talks, start, frame_w, frame_h, main_key
            )
            for i in range(begin, end)
        ):
            for i in range(begin, end):
                pt = track_points[i]
                pt["mode"] = keeper_mode
                if keeper_mode == "split":
                    pt["x"], pt["y"] = first["x"], first["y"]
                    pt["x2"], pt["y2"] = first.get("x2"), first.get("y2")
                else:
                    aim = _refold_aim(pt, samples[i], keeper_mode, frame_w, frame_h)
                    if aim:
                        pt["x"], pt["y"] = aim
            continue

        # The hold cannot have this shot, so the shot has to fit inside the hold: take the seconds it
        # is short by from whichever neighbour has the less to lose, frame by frame, and only where
        # that frame is happy to show whoever is talking in it.
        for side in sorted(sides, key=span_of):
            while span_of([begin, end]) < floor and absorb(side):
                pass
        protected.update({begin, first_frame})
        exempt += 1
    return exempt


def _fold_is_safe(
    index: int,
    donor: dict[str, Any],
    keeper_mode: str,
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, TalkWindow],
    start: float,
    frame_w: int,
    frame_h: int,
    main_key: int | None = None,
) -> bool:
    """Would the frame this fold actually produces still show whoever the rule demands?

    The fold re-aims a merged sample at the people in it, so the test has to judge that aim, not the
    donor's raw centre — otherwise a fold that lands on the wrong half of the room looks safe.
    """
    pt = track_points[index]
    people = [p for p in samples[index].get("people", []) if p.get("box") and p.get("kind") != "object"]
    # Never fold a solo close-up shot (bounded by a hard scene cut or marked cutProtected) into a
    # letterboxed "fit" card (e.g. 0:23 solo host close-up between two wide group shots).
    if keeper_mode in ("fit", "editorial") and (pt.get("mode") or "fill") == "fill":
        if pt.get("cutProtected") or (
            pt.get("cut")
            and len(people) == 1
            and head_zone(people[0])[2] >= 0.085 * frame_w
        ):
            return False
    talkers, _, _ = _required_in_sample(samples[index], talks, start, main_key)
    if not talkers:
        return True
    if keeper_mode == "split":
        trial = {**pt, "mode": "split", "x": donor["x"], "y": donor["y"]}
        trial["x2"], trial["y2"] = donor.get("x2"), donor.get("y2")
    else:
        trial = {**pt, "mode": keeper_mode}
        aim = _refold_aim(pt, samples[index], keeper_mode, frame_w, frame_h)
        if aim:
            trial["x"], trial["y"] = aim
    return all(shown(p, trial, frame_w, frame_h) for p in talkers)


def _refold_aim(
    pt: dict[str, Any], s: dict[str, Any], keeper_mode: str, frame_w: int, frame_h: int
) -> tuple[float, float] | None:
    """Where a folded sample has to point to keep the people it is now expected to show."""
    panels = panel_geometry(keeper_mode, frame_w, frame_h, float(pt.get("zoom") or 1.0))
    if not panels:
        return None
    panel = panels[0]
    people = [p for p in s.get("people", []) if p.get("box") and p.get("kind") != "object"]
    if not people:
        return None
    bias = _centre(max(people, key=lambda p: p["box"][2] * p["box"][3]), frame_w, frame_h)
    return group_target(people, panel["cw"], panel["ch"], bias, frame_w, frame_h, panel["anchor"])


def coverage_of(
    track_points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    talks: dict[int, TalkWindow],
    frame_w: int,
    frame_h: int,
    main_key: int | None = None,
) -> dict[str, Any]:
    """Of the samples where the rule demanded a person on screen, how many actually showed their head.

    Only counts a person the detectors saw in that sample. A shot that cuts away to a door or a product
    is not the framing losing its speaker, and charging those seconds to the camera would make the
    number mean nothing.
    """
    start = float(samples[0]["t"]) if samples else 0.0
    per_key: dict[int, dict[str, int]] = {}
    total_talk = total_shown = 0
    for pt, s in zip(track_points, samples):
        required, _, reason = _required_in_sample(s, talks, start, main_key)
        if reason == "no_verdict" or not required:
            continue
        for person in required:
            key = int(person["key"])
            stat = per_key.setdefault(key, {"seen": 0, "shown": 0})
            stat["seen"] += 1
            total_talk += 1
            if shown(person, pt, frame_w, frame_h):
                stat["shown"] += 1
                total_shown += 1
    return {
        "overall": round(total_shown / total_talk, 4) if total_talk else None,
        "talking_samples": total_talk,
        "per_speaker": {
            str(key): {
                "talking_samples": v["seen"],
                "on_screen": v["shown"],
                "coverage": round(v["shown"] / v["seen"], 4) if v["seen"] else None,
            }
            for key, v in sorted(per_key.items())
        },
    }
