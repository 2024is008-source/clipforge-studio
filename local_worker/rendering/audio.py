"""Audio composition: original, music, voiceover, ducking, fades and loudness."""

from __future__ import annotations

from .decision import AudioTrack, RenderDecision

LOUDNORM = "loudnorm=I=-16:TP=-2.0:LRA=11"
CEILING = "alimiter=limit=0.891"


def extra_audio_inputs(decision: RenderDecision) -> list[AudioTrack]:
    """The tracks that need their own FFmpeg input, in the order the renderer must add them."""
    return [t for t in decision.audio if t.kind != "original"]


def build_audio_graph(
    decision: RenderDecision,
    video_index: int,
    extra_indices: dict[int, int],
) -> tuple[str, str]:
    """Return (filter_complex fragment, audio output label).

    `extra_indices` maps a track's position in `decision.audio` to the FFmpeg input index the
    renderer opened it at; the original track always rides the video input.
    """
    tracks = decision.audio or [AudioTrack(kind="original", path="", volume=1.0)]
    chains: list[str] = []
    labels: list[str] = []
    duration = decision.duration

    for position, track in enumerate(tracks):
        index = video_index if track.kind == "original" else extra_indices[position]
        label = f"a{position}"
        # Each link is a separate filter: joining with "," matters, because a trailing ":" inside
        # `aresample=...:async=1:` would make the next filter read as an unknown aresample option.
        links = [f"[{index}:a]aresample=48000:async=1"]
        if track.kind == "original" and decision.voice_enhance:
            # The same voice chain the master endpoint applies: drop the rumble, then hold the
            # speaker at a steady level so the caption band never has to fight a loud burst.
            links += ["highpass=f=75", "acompressor=threshold=0.05:ratio=4:attack=20:release=250"]
        if track.volume != 1.0:
            links.append(f"volume={track.volume:.3f}")
        if track.start > 0:
            links.append(f"adelay=delays={int(round(track.start * 1000))}:all=1")
        if track.fade_in > 0:
            links.append(f"afade=t=in:st=0:d={track.fade_in:.3f}")
        if track.fade_out > 0:
            links.append(f"afade=t=out:st={max(0.0, duration - track.fade_out):.3f}:d={track.fade_out:.3f}")
        chains.append(",".join(links) + f"[{label}]")
        labels.append(label)

    # Ducking needs one key copy per ducked track, and the voice itself must stay audible, so the
    # key source splits into (1 main + N keys) rather than being consumed by the first compressor.
    key_label = next((labels[i] for i, t in enumerate(tracks) if t.kind == "voiceover"), None)
    if key_label is None:
        key_label = next((labels[i] for i, t in enumerate(tracks) if t.kind == "original"), None)
    ducked = [i for i, t in enumerate(tracks) if t.duck_under_voice and key_label and labels[i] != key_label]

    mixed = list(labels)
    if ducked and key_label:
        pads = ["vmain"] + [f"vkey{j}" for j in range(len(ducked))]
        chains.append(f"[{key_label}]asplit={len(pads)}" + "".join(f"[{p}]" for p in pads))
        mixed = [pads[0] if l == key_label else l for l in labels]
        for j, i in enumerate(ducked):
            out = f"duck{i}"
            chains.append(
                f"[{labels[i]}][vkey{j}]sidechaincompress=threshold=0.03:ratio=8:attack=20:"
                f"release=400:makeup=1[{out}]"
            )
            mixed[i] = out

    if len(mixed) == 1:
        chains.append(f"[{mixed[0]}]anull[amix]")
    else:
        source = "".join(f"[{m}]" for m in mixed)
        chains.append(f"{source}amix=inputs={len(mixed)}:duration=longest:dropout_transition=0:normalize=0[amix]")

    # `apad` keeps audio at least as long as the picture and `-shortest` cuts it at the video's end.
    # Without the pad, a source whose AAC track is a second shorter than its video track truncates
    # the whole window to the audio length and every later window slides out of sync.
    chains.append(f"[amix]{LOUDNORM},{CEILING},aresample=48000,asetpts=PTS-STARTPTS,apad[outa]")
    return ";".join(chains), "[outa]"
