from __future__ import annotations

import io
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
import zlib
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from local_worker.rendering import overlays, renderer
from local_worker.rendering.decision import CaptionStyle, RenderDecision, TrackPoint, Word
from local_worker.rendering.encoders import EncoderChoice, cpu_encoder, encoder_video_args
from local_worker.rendering.framerate import rate_arg, resolve_output_fps


def chunk(kind, data=b""):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_header(width=1080, height=1920, depth=8, color=6, interlace=0):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, interlace))


def png_stream(compressed, header=None):
    return overlays.PNG_SIGNATURE + (header or png_header()) + chunk(b"IDAT", compressed) + chunk(b"IEND")


@lru_cache(maxsize=4)
def rgba_png(color=(255, 0, 0, 255), striped=False):
    row = bytes(color) * 1080
    if striped:
        row = bytes(color) * 360 + bytes((0, 255, 0, 128)) * 360 + bytes((0, 0, 0, 0)) * 360
    return png_stream(zlib.compress((b"\0" + row) * 1920))


def decision(source, fps=24.0, duration=2.0, start=0.0, layout="fill"):
    return RenderDecision(
        str(source), 180, 320, fps, start, start + duration, layout,
        [TrackPoint(start, .5, .38, x2=.8, y2=.38, mode=layout)],
        [Word("Canvas", start, start + duration)], "Hook", CaptionStyle(show_emojis=False),
        brand_label="ClipForge", zoom_punch=False, voice_enhance=False,
    )


class StoreCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="clipforge_overlay_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = overlays.OverlayStore(self.root / "overlays")

    def create(self, count=1, fps=24.0):
        return self.store.create(fps, count, 1080, 1920)

    def completed(self, count=1, fps=24.0):
        key = self.create(count, fps)
        self.store.upload(key, 0, [io.BytesIO(rgba_png()) for _ in range(count)])
        self.store.complete(key)
        return key, self.store.resolve(key, fps, count / fps)


class ManifestTests(StoreCase):
    def test_supported_rates_are_not_snapped(self):
        for fps in (6, 15.125, 23.976, 24000 / 1001, 24, 29.97, 30000 / 1001, 30 + 1e-10, 60):
            with self.subTest(fps=fps):
                key = self.create(fps=fps)
                self.assertEqual(self.store._sessions[key].fps, fps)
                self.store.delete(key)

    def test_invalid_rates(self):
        for fps in (True, False, None, "24", 0, 5.9999, 60.0001, math.inf, -math.inf, math.nan, 10 ** 1000):
            with self.subTest(fps=str(fps)[:30]), self.assertRaises(ValueError):
                self.create(fps=fps)

    def test_duration_and_frame_limits(self):
        self.assertEqual(overlays.overlay_frame_count(60, 300), 18000)
        self.assertEqual(overlays.overlay_frame_count(6, 300), 1800)
        self.assertEqual(overlays.overlay_frame_count(24000 / 1001, 300), 7193)
        for fps in (24000 / 1001, 24, 60):
            self.assertEqual(overlays.overlay_frame_count(fps, 3 / fps), 3)
            self.assertEqual(overlays.overlay_frame_count(fps, 3.2 / fps), 4)
        key = self.create(18000, 60)
        self.store.delete(key)
        for duration in (True, None, "1", 0, -1, 300.001, math.inf, math.nan, 10 ** 1000):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                overlays.overlay_frame_count(24, duration)
        for count in (True, 0, -1, 1.0, "1", 18001):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.create(count, 60)
        with self.assertRaisesRegex(ValueError, "300 seconds"):
            self.create(7201, 24)
        key = self.create(7193, 24000 / 1001)
        self.store.delete(key)
        with self.assertRaisesRegex(ValueError, "300 seconds"):
            self.create(7194, 24000 / 1001)

    def test_dimensions_and_session_limit(self):
        for width, height in ((1920, 1080), (1080.0, 1920), (1080, True), (0, 0)):
            with self.assertRaises(ValueError):
                self.store.create(24, 1, width, height)
        for _ in range(overlays.MAX_SESSIONS):
            self.create()
        with self.assertRaises(HTTPException) as caught:
            self.create()
        self.assertEqual(caught.exception.status_code, 429)

    def test_router_manifest_and_multipart_contract(self):
        app = FastAPI()
        app.include_router(overlays.router)
        with patch.object(overlays, "store", self.store), TestClient(app) as client:
            url = "/v1/render/overlays"
            manifest = dict(fps=24000 / 1001, frameCount=1, width=1080, height=1920)
            for change in ({"fps": True}, {"fps": "24"}, {"frameCount": 1.0}, {"extra": 1}):
                self.assertEqual(client.post(url, json=manifest | change).status_code, 422)
            self.assertEqual(client.post(url, json=manifest | {"fps": 61}).status_code, 400)
            created = client.post(url, json=manifest)
            self.assertEqual(created.status_code, 200)
            key = created.json()["overlayId"]
            endpoint = f"{url}/{key}"
            self.assertEqual(client.post(endpoint + "/complete").status_code, 400)
            response = client.post(endpoint + "/frames", data={"start": 0},
                                   files=[("files", ("../../escape.png", rgba_png(), "text/plain"))])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"frameCount": 1})
            self.assertEqual(client.post(endpoint + "/complete").status_code, 200)
            self.assertEqual(client.delete(endpoint).status_code, 204)
            self.assertFalse((self.root / "escape.png").exists())


class PNGTests(unittest.TestCase):
    def test_rgba_scanlines_and_contiguous_idat(self):
        for filter_byte in range(5):
            raw = (bytes([filter_byte]) + b"\0" * 4320) * 1920
            compressed = zlib.compress(raw)
            data = (overlays.PNG_SIGNATURE + png_header() + chunk(b"tEXt", b"key\0value")
                    + chunk(b"IDAT", compressed[:10]) + chunk(b"IDAT", compressed[10:]) + chunk(b"IEND"))
            overlays._validate_png(data)
        overlays._validate_png(rgba_png(striped=True))

    def test_malformed_oversized_non_rgba_and_bad_streams(self):
        valid = rgba_png()
        compressed = zlib.compress((b"\0" + b"\0" * 4320) * 1920)
        prefix = overlays.PNG_SIGNATURE + png_header()
        malformed = {
            "not png": b"jpeg", "truncated": valid[:-1], "missing end": valid[:-12],
            "crc": valid[:-1] + bytes([valid[-1] ^ 1]), "trailing": valid + b"extra",
            "oversized": b"x" * (overlays.MAX_FRAME_BYTES + 1),
            "rgb": png_stream(compressed, png_header(color=2)),
            "palette": png_stream(compressed, png_header(color=3)),
            "grayscale": png_stream(compressed, png_header(color=0)),
            "16bit": png_stream(compressed, png_header(depth=16)),
            "interlace": png_stream(compressed, png_header(interlace=1)),
            "size": png_stream(compressed, png_header(width=1079)),
            "duplicate header": prefix + png_header() + chunk(b"IDAT", compressed) + chunk(b"IEND"),
            "no header": overlays.PNG_SIGNATURE + chunk(b"IEND"),
            "no data": prefix + chunk(b"IEND"),
            "apng": prefix + chunk(b"acTL", b"\0" * 8) + chunk(b"IDAT", compressed) + chunk(b"IEND"),
            "critical": prefix + chunk(b"ABCD") + chunk(b"IDAT", compressed) + chunk(b"IEND"),
            "noncontiguous": prefix + chunk(b"IDAT", compressed[:10]) + chunk(b"tEXt", b"a\0b")
                             + chunk(b"IDAT", compressed[10:]) + chunk(b"IEND"),
            "invalid zlib": png_stream(b"not zlib"),
            "truncated zlib": png_stream(compressed[:-1]),
            "trailing zlib": png_stream(compressed + b"garbage"),
            "second stream": png_stream(compressed + compressed),
            "short stream": png_stream(zlib.compress(b"\0" * (4321 * 1920 - 1))),
            "long stream": png_stream(zlib.compress(b"\0" * (4321 * 1920 + 1))),
            "inflate bomb": png_stream(zlib.compress(b"\0" * (4321 * 1920 * 2))),
            "filter": png_stream(zlib.compress((b"\5" + b"\0" * 4320) * 1920)),
        }
        for name, data in malformed.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                overlays._validate_png(data)


class UploadTests(StoreCase):
    def test_batches_are_dense_sequential_and_immutable_after_completion(self):
        key = self.create(18)
        for start, count in ((1, 1), (False, 1), (0, 0), (0, 17)):
            with self.assertRaises(ValueError):
                self.store.upload(key, start, [io.BytesIO(rgba_png()) for _ in range(count)])
        self.assertEqual(self.store.upload(key, 0, [io.BytesIO(rgba_png()) for _ in range(16)]), 16)
        with self.assertRaises(ValueError):
            self.store.upload(key, 15, [io.BytesIO(rgba_png())])
        with self.assertRaises(ValueError):
            self.store.upload(key, 16, [io.BytesIO(rgba_png()) for _ in range(3)])
        with self.assertRaises(ValueError):
            self.store.complete(key)
        with self.assertRaises(ValueError):
            self.store.resolve(key, 24, 18 / 24)
        self.store.upload(key, 16, [io.BytesIO(rgba_png()), io.BytesIO(rgba_png())])
        self.store.complete(key)
        path = Path(self.store.resolve(key, 24, 18 / 24))
        self.assertEqual(sorted(p.name for p in path.iterdir()), [f"{i:06d}.png" for i in range(18)])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.upload(key, 18, [io.BytesIO(rgba_png())])

    def test_atomic_validation_and_byte_accounting(self):
        key = self.create(3)
        session = self.store._sessions[key]
        with self.assertRaises(ValueError):
            self.store.upload(key, 0, [io.BytesIO(rgba_png()), io.BytesIO(b"bad")])
        self.assertEqual((session.uploaded, session.size, list(session.path.iterdir())), (0, 0, []))
        with self.assertRaises(HTTPException) as caught:
            self.store.upload(key, 0, [io.BytesIO(b"x" * (overlays.MAX_FRAME_BYTES + 1))])
        self.assertEqual(caught.exception.status_code, 413)
        size = len(rgba_png())
        with patch.object(overlays, "MAX_SESSION_BYTES", 2 * size):
            self.store.upload(key, 0, [io.BytesIO(rgba_png())])
            with self.assertRaises(HTTPException) as caught:
                self.store.upload(key, 1, [io.BytesIO(rgba_png()), io.BytesIO(rgba_png())])
            self.assertEqual(caught.exception.status_code, 413)
            self.assertEqual((session.uploaded, session.size), (1, size))
            self.assertEqual([p.name for p in session.path.iterdir()], ["000000.png"])
            self.store.upload(key, 1, [io.BytesIO(rgba_png())])
            self.assertEqual(session.size, 2 * size)
            other = self.create()
            self.store.upload(other, 0, [io.BytesIO(rgba_png())])

    def test_commit_failure_rolls_back_the_whole_batch(self):
        key = self.create(2)
        session = self.store._sessions[key]
        real_replace = os.replace
        calls = 0

        def fail_second(src, dst):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full")
            real_replace(src, dst)

        with patch.object(overlays.os, "replace", side_effect=fail_second), self.assertRaises(OSError):
            self.store.upload(key, 0, [io.BytesIO(rgba_png()), io.BytesIO(rgba_png())])
        self.assertEqual((session.uploaded, session.size, list(session.path.iterdir())), (0, 0, []))

    def test_exact_fps_count_and_dense_files_are_revalidated(self):
        fps = 24000 / 1001
        key, directory = self.completed(2, fps)
        for bad_fps, duration in ((23.976, 2 / fps), (fps + 1e-10, 2 / fps), (fps, 3 / fps)):
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.store.resolve(key, bad_fps, duration)
        extra = Path(directory) / "extra.png"
        extra.write_bytes(rgba_png())
        with self.assertRaisesRegex(ValueError, "dense"):
            self.store.resolve(key, fps, 2 / fps)
        extra.unlink()
        (Path(directory) / "000001.png").unlink()
        with self.assertRaisesRegex(ValueError, "dense"):
            self.store.resolve(key, fps, 2 / fps)


class LifetimeTests(StoreCase):
    def test_canonical_uuid_and_trusted_directory_only(self):
        key, directory = self.completed()
        for invalid in ("../" + key, key.upper(), key.replace("-", ""), "{" + key + "}",
                        directory, key + "/..", "", None):
            for operation in (self.store.delete, self.store.complete, lambda value: self.store.resolve(value, 24, 1 / 24)):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    operation(invalid)
        for path in (str(self.root / key), key, str(self.root / "overlays" / ".." / key)):
            with self.assertRaises(ValueError), self.store.reference(path, 24, 1 / 24):
                pass
        self.assertTrue(Path(directory).exists())

    def test_nested_pins_block_delete_and_expiry_and_release_after_failure(self):
        key, directory = self.completed()
        session = self.store._sessions[key]
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with self.store.reference(directory, 24, 1 / 24), self.store.reference(directory, 24, 1 / 24):
                self.assertEqual(session.references, 2)
                session.touched = 0
                self.store.cleanup()
                self.assertTrue(Path(directory).exists())
                with self.assertRaises(HTTPException) as caught:
                    self.store.delete(key)
                self.assertEqual(caught.exception.status_code, 409)
                raise RuntimeError("failure")
        self.assertEqual(session.references, 0)
        self.store.delete(key)
        self.assertFalse(Path(directory).exists())

    def test_expiry_reaps_abandoned_uuids_not_neighbors(self):
        key = self.create()
        session = self.store._sessions[key]
        session.touched = 0
        orphan = self.store.root / str(uuid.uuid4())
        orphan.mkdir()
        os.utime(orphan, (0, 0))
        neighbor = self.store.root / "keep-me"
        neighbor.mkdir()
        os.utime(neighbor, (0, 0))
        fresh = self.store.root / str(uuid.uuid4())
        fresh.mkdir()
        self.store.cleanup()
        self.assertNotIn(key, self.store._sessions)
        self.assertFalse(orphan.exists())
        self.assertTrue(neighbor.exists())
        self.assertTrue(fresh.exists())
        with self.assertRaises(ValueError):
            self.store.resolve(key, 24, 1 / 24)


class LayoutTests(unittest.TestCase):
    def test_short_finalized_fit_safety_run_is_never_folded(self):
        d = decision("unused", duration=2, layout="auto")
        d.track = [TrackPoint(t, .5, .38, mode=mode, layout_decision={"layout": mode})
                   for t, mode in ((0, "fill"), (1, "fit"), (1.2, "fill"), (2, "fill"))]
        with patch.object(renderer, "effective_layout", side_effect=AssertionError("must not remeasure")):
            segments = renderer.layout_segments(d, 24)
        self.assertEqual([s.layout for s in segments], ["fill", "fit", "fill"])
        self.assertLess(segments[1].duration, .5)
        self.assertAlmostEqual(segments[1].start, 1)
        for finalized_count in (0, 3):
            legacy = replace(d, track=[replace(p, layout_decision=p.layout_decision if i < finalized_count else None)
                                       for i, p in enumerate(d.track)])
            self.assertEqual([s.layout for s in renderer.layout_segments(legacy, 24)], ["fill"])

    def test_finalized_panels_are_not_downgraded(self):
        d = decision("unused", layout="auto")
        d.track = [TrackPoint(0, .5, .38, mode="split", layout_decision={})]
        self.assertEqual([s.layout for s in renderer.layout_segments(d, 24)], ["split"])
        d.track[0].layout_decision = None
        self.assertEqual([s.layout for s in renderer.layout_segments(d, 24)], ["fill"])


class RenderPlanTests(StoreCase):
    def test_native_text_and_band_only_suppressed_with_canvas(self):
        d = decision(self.root / "source.mp4", layout="split")
        encoder = EncoderChoice("h264_qsv", "gpu", ["-global_quality", "23"])
        for canvas in (None, str(self.root / "overlay")):
            request = renderer.RenderRequest(d, str(self.root), encoder=encoder, fps=24, overlay_dir=canvas)
            with patch.object(renderer, "_write_captions", return_value="ass=filename=c.ass") as ass, \
                 patch.object(renderer, "crop_keyframes", wraps=renderer.crop_keyframes) as crops:
                passes, _, events = renderer.plan_passes(request, renderer.Segment("split", 0, 2), 0,
                                                        True, None, str(self.root / "vignette.png"))
            graph = passes[-1].args[passes[-1].args.index("-filter_complex") + 1]
            self.assertEqual("ass=" in graph, canvas is None)
            self.assertEqual("drawbox=" in graph, canvas is None)
            self.assertEqual(ass.call_count, 1 if canvas is None else 0)
            self.assertEqual(events, 2 if canvas is None else 0)
            sub = crops.call_args.args[0]
            self.assertEqual(sub.words, d.words)
            self.assertEqual(sub.hook, d.hook if canvas is None else "")
            self.assertEqual(sub.brand_label, d.brand_label if canvas is None else None)
            self.assertEqual(sub.style.show_hook, canvas is None)
            self.assertIn("[outa]", passes[-1].args)
            self.assertIn("h264_qsv", passes[-1].args)
            self.assertIn("-global_quality", passes[-1].args)
        self.assertEqual(d.hook, "Hook")
        self.assertEqual(d.brand_label, "ClipForge")
        self.assertTrue(d.style.show_hook)

    def test_overlay_plan_uses_chosen_rate_encoder_full_timeline_and_audio_copy(self):
        encoder = EncoderChoice("h264_nvenc", "gpu", ["-preset", "p5"])
        fps = 24000 / 1001
        d = decision(self.root / "source.mp4", fps=fps, duration=3.2 / fps, start=12.5)
        request = renderer.RenderRequest(d, str(self.root), fps=fps, encoder=encoder,
                                         overlay_dir=str(self.root / "overlay"))
        for audio in (True, False):
            p = renderer._overlay_pass(request, str(self.root / "assembled.mp4"), audio)
            graph = p.args[p.args.index("-filter_complex") + 1]
            self.assertEqual(p.expected_frames, 4)
            self.assertEqual(p.args[p.args.index("-framerate") + 1], "24000/1001")
            self.assertEqual(p.args[p.args.index("-r") + 1], "24000/1001")
            self.assertEqual(p.args[p.args.index("-c:v") + 1], encoder.name)
            self.assertAlmostEqual(float(p.args[p.args.index("-t") + 1]), 4 / fps)
            self.assertIn("p5", p.args)
            self.assertIn("trim=end_frame=4", graph)
            self.assertIn("alpha=straight", graph)
            self.assertIn("setpts=PTS-STARTPTS", graph)
            self.assertNotIn("-ss", p.args)
            self.assertNotIn("-shortest", p.args)
            self.assertEqual("-an" in p.args, not audio)
            if audio:
                self.assertEqual(p.args[p.args.index("-c:a") + 1], "copy")
                self.assertIn("0:a:0", p.args)
        with self.assertRaises(renderer.RenderFailed):
            renderer._overlay_pass(replace(request, overlay_dir=None), "assembled.mp4", False)
        concat = renderer._concat_pass(request, ["a.mp4", "b.mp4"], True)
        self.assertTrue(concat.output.endswith("out_assembled.mp4"))

    def test_reference_wrapper_keeps_frames_for_retry_and_removes_partial_delivery(self):
        key, directory = self.completed()
        request = renderer.RenderRequest(decision(self.root / "source.mp4", duration=1 / 24),
                                         str(self.root), fps=24, overlay_dir=directory)
        pending = self.root / "out_canvas_overlay.mp4"
        for error in (renderer.RenderCancelled, renderer.EncoderUnavailable, renderer.RenderFailed):
            def fail(*_args):
                self.assertEqual(self.store._sessions[key].references, 2)
                pending.write_bytes(b"partial")
                raise error("injected")

            with patch.object(overlays, "store", self.store), patch.object(renderer, "_run_render", side_effect=fail):
                with self.store.reference(directory, 24, 1 / 24), self.assertRaises(error):
                    renderer.run_render(request)
                self.assertFalse(pending.exists())
                self.assertEqual(self.store._sessions[key].references, 0)
                self.assertTrue((Path(directory) / "000000.png").exists())

    def test_frame_count_qc_failure_keeps_upload_and_removes_partial_output(self):
        key, directory = self.completed()
        source = self.root / "source.mp4"
        source.touch()
        request = renderer.RenderRequest(decision(source, duration=1 / 24), str(self.root), fps=24,
                                         encoder=cpu_encoder(), overlay_dir=directory)

        def fake_encode(args, *_rest):
            Path(args[-1]).write_bytes(b"injected output")

        with patch.object(overlays, "store", self.store), \
             patch.object(renderer.shutil, "which", return_value="ffmpeg"), \
             patch.object(renderer, "source_has_audio", return_value=False), \
             patch.object(renderer, "_ensure_vignette", return_value=str(self.root / "vignette.png")), \
             patch.object(renderer, "_run", side_effect=fake_encode), \
             patch.object(renderer, "frame_count", return_value=0):
            with self.assertRaisesRegex(renderer.RenderFailed, "expected exactly 1"):
                renderer.run_render(request)
        self.assertFalse((self.root / "out_canvas_overlay.mp4").exists())
        self.assertFalse((self.root / "out_backend.mp4").exists())
        self.assertEqual(self.store._sessions[key].references, 0)
        self.assertTrue((Path(directory) / "000000.png").exists())

    def test_pre_cancelled_render_releases_pin_without_launching(self):
        key, directory = self.completed()
        request = renderer.RenderRequest(decision("unused", duration=1 / 24), str(self.root),
                                         fps=24, overlay_dir=directory)
        with patch.object(overlays, "store", self.store), patch.object(renderer.subprocess, "Popen") as popen:
            with self.assertRaises(renderer.RenderCancelled):
                renderer.run_render(request, is_cancelled=lambda: True)
        popen.assert_not_called()
        self.assertEqual(self.store._sessions[key].references, 0)


class ProcessTests(StoreCase):
    def run_script(self, script, callback=lambda _: None, cancelled=None, budget=5, encoder=None):
        args = [sys.executable, "-u", "-c", script]
        if encoder:
            args += ["-c:v", encoder]
        renderer._run(args, str(self.root), 1, callback, cancelled, time.time(), budget)

    def test_progress_continue_does_not_report_premature_completion(self):
        fractions = []
        self.run_script("print('out_time_us=250000\\nprogress=continue\\nout_time_us=500000\\nprogress=end')",
                        fractions.append)
        self.assertEqual(fractions, [.25, .5, 1.0])

    def test_silent_process_cancellation_and_deadline_reap_child(self):
        for cancel in (True, False):
            began = time.monotonic()
            processes = []
            popen = subprocess.Popen

            def capture(*args, **kwargs):
                process = popen(*args, **kwargs)
                processes.append(process)
                return process

            with patch.object(renderer.subprocess, "Popen", side_effect=capture):
                with self.assertRaises(renderer.RenderCancelled if cancel else renderer.RenderFailed):
                    self.run_script("import threading; threading.Event().wait(30)",
                                    cancelled=(lambda: time.monotonic() - began > .2) if cancel else None,
                                    budget=5 if cancel else .2)
            self.assertLess(time.monotonic() - began, 3)
            self.assertIsNotNone(processes[0].poll())

    def test_stderr_is_drained_and_encoder_failure_keeps_retry_classification(self):
        cases = (
            ("h264_nvenc", "nvenc error: no encode device"),
            ("h264_nvenc", "cuInit(0) failed -> CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected"),
            ("h264_qsv", "Error initializing the MFX video session: unsupported"),
            ("h264_qsv", "Error while opening encoder - maybe incorrect parameters"),
            ("h264_amf", "AMFEncoderCore: Failed to create the encoder component"),
            ("h264_videotoolbox", "compression session create failed"),
        )
        for encoder, message in cases:
            with self.subTest(encoder=encoder, message=message):
                with self.assertRaises(renderer.EncoderUnavailable):
                    self.run_script(
                        "import sys; sys.stderr.write('warning\\n' * 20000); "
                        f"sys.stderr.write({message!r} + '\\n'); sys.exit(1)",
                        encoder=encoder,
                    )

    def test_software_pass_and_filter_errors_are_not_retryable_on_cpu(self):
        for encoder, message in (
            ("libx264", "nvenc error: no encode device"),
            ("h264_qsv", "No such filter: 'not_a_real_filter'"),
        ):
            with self.subTest(encoder=encoder, message=message):
                with self.assertRaises(renderer.RenderFailed) as caught:
                    self.run_script(
                        f"import sys; sys.stderr.write({message!r} + '\\n'); sys.exit(1)",
                        encoder=encoder,
                    )
                self.assertNotIsInstance(caught.exception, renderer.EncoderUnavailable)

    def test_progress_callback_failure_stops_process(self):
        def fail(_fraction):
            raise RuntimeError("callback failed")

        began = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            self.run_script("import threading; print('out_time_us=250000'); threading.Event().wait(30)", fail)
        self.assertLess(time.monotonic() - began, 3)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local FFmpeg and ffprobe required")
class FFmpegSmokeTests(StoreCase):
    def test_synthetic_film_and_24fps_frames_alpha_color_audio_and_progress(self):
        encoder = cpu_encoder()
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=180x320:r=24:d=0.1",
            *encoder_video_args(encoder, "24/1"), "-an", "-f", "null", "-",
        ], check=True, capture_output=True, timeout=30)
        for source_fps in (24000 / 1001, 24.0):
            with self.subTest(fps=source_fps):
                fps = resolve_output_fps(source_fps, "30")
                self.assertEqual(fps, source_fps)
                source = self.root / f"source_{fps}.mp4"
                duration = 3.2 / fps
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c=blue:s=180x320:r={rate_arg(fps)}:d=0.6",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
                ], check=True, capture_output=True, timeout=30)
                key = self.create(4, fps)
                colors = [(255, 0, 0, 255), (0, 255, 0, 255)] * 2
                self.store.upload(key, 0, [io.BytesIO(rgba_png(color, striped=True)) for color in colors])
                self.store.complete(key)
                directory = self.store.resolve(key, fps, duration)
                job = self.root / f"job_{fps}"
                request = renderer.RenderRequest(decision(source, fps, duration, start=2 / fps), str(job),
                                                 fps=fps, encoder=encoder, overlay_dir=directory)
                progress = []
                with patch.object(overlays, "store", self.store):
                    result = renderer.run_render(request, on_progress=lambda value, label: progress.append(value))
                self.assertEqual(result.encoder, encoder.name)
                self.assertEqual(result.caption_events, 0)
                self.assertTrue(result.source_has_audio)
                self.assertEqual(result.passes[-1], "composite canvas overlays")
                self.assertFalse(list(job.glob("*.ass")))
                self.assertFalse((job / "out_canvas_overlay.mp4").exists())
                self.assertEqual(self.store._sessions[key].references, 0)
                self.assertEqual(progress, sorted(progress))
                self.assertEqual(progress[-1], 1)
                info = renderer.ffprobe_json(result.output_path)
                video = next(s for s in info["streams"] if s["codec_type"] == "video")
                audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
                self.assertEqual(int(video["nb_frames"]), 4)
                self.assertEqual((video["width"], video["height"]), (1080, 1920))
                self.assertAlmostEqual(float(Fraction(video["avg_frame_rate"])), fps, places=7)
                self.assertEqual(audio["codec_name"], "aac")
                audio_hashes = []
                for path in (job / "segment_000.mp4", result.output_path):
                    audio_hashes.append(subprocess.run([
                        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "copy",
                        "-f", "hash", "-hash", "sha256", "pipe:1",
                    ], check=True, capture_output=True, timeout=30).stdout)
                self.assertEqual(audio_hashes[0], audio_hashes[1])
                decoded = subprocess.run([
                    "ffmpeg", "-v", "error", "-i", result.output_path, "-an", "-vf",
                    "scale=108:192", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                ], check=True, capture_output=True, timeout=30).stdout
                frame_bytes = 108 * 192 * 3
                self.assertEqual(len(decoded), 4 * frame_bytes)
                samples = []
                for i, color in enumerate(colors):
                    def pixel(x):
                        offset = i * frame_bytes + (96 * 108 + x) * 3
                        return tuple(decoded[offset:offset + 3])
                    opaque, blended, transparent = pixel(18), pixel(54), pixel(90)
                    samples.append(opaque)
                    for got, expected in zip(opaque, color[:3]):
                        self.assertAlmostEqual(got, expected, delta=15)
                    self.assertLess(blended[0], 25)
                    self.assertTrue(80 < blended[1] < 180)
                    self.assertTrue(60 < blended[2] < 180)
                    self.assertLess(transparent[0], 25)
                    self.assertLess(transparent[1], 25)
                    self.assertGreater(transparent[2], 170)
                pcm = subprocess.run([
                    "ffmpeg", "-v", "error", "-i", result.output_path, "-vn", "-ac", "1",
                    "-ar", "48000", "-f", "f32le", "pipe:1",
                ], check=True, capture_output=True, timeout=30).stdout
                values = [sample[0] for sample in struct.iter_unpack("<f", pcm)]
                rms = math.sqrt(sum(value * value for value in values) / len(values))
                self.assertGreater(rms, .01)
                self.assertAlmostEqual(len(values) / 48000, duration, delta=.05)
                print(f"overlay smoke: fps={video['avg_frame_rate']} frames=4 colors={samples} "
                      f"audio=aac samples={len(values)} rms={rms:.4f} encoder={result.encoder}")
                self.store.delete(key)


if __name__ == "__main__":
    unittest.main()
