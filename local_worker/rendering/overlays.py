"""Bounded, process-local storage for canvas-painted RGBA frame sequences.

Only UUIDs cross the HTTP boundary. resolve_overlay validates a completed upload;
reference_overlay pins its trusted directory for the renderer (including cleanup on
failure). The router's lifespan sweeps expired uploads, including abandoned uploads
from a previous worker process. Register the router behind the worker's token check.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import struct
import threading
import time
import uuid
import zlib
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

from ..config import WORK_DIR

MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_SESSION_BYTES = 512 * 1024 * 1024
MAX_FRAMES = 18000
MAX_DURATION = 300.0
MAX_BATCH = 16
MAX_SESSIONS = 4
SESSION_TTL = 3600.0
SWEEP_INTERVAL = 60.0
WIDTH, HEIGHT = 1080, 1920
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _fps(value: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 6 <= value <= 60 or not math.isfinite(value)):
        raise ValueError("fps must be a finite number between 6 and 60")
    return value


def overlay_frame_count(fps: float, duration: float) -> int:
    fps = _fps(fps)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("duration must be a finite number in (0, 300]")
    if not 0 < duration <= MAX_DURATION or not math.isfinite(duration):
        raise ValueError("duration must be a finite number in (0, 300]")
    count = math.ceil(duration * fps - 1e-8)
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError("overlay must contain between 1 and 18000 frames")
    return count


def _uuid(value: str) -> str:
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError, TypeError):
        raise ValueError("overlayId must be a canonical UUID") from None
    return value


def _validate_png(data: bytes) -> None:
    """Validate browser PNGs without loading a model or requiring an imaging library.

    Check every chunk CRC and the complete, bounded zlib scanline stream. Canvas
    PNGs are 8-bit non-interlaced RGBA; palette/RGB/APNG and trailing payloads are
    deliberately rejected. At most one 1080x1920 RGBA image is inflated at a time.
    """
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError("PNG exceeds the 2 MiB frame limit")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("frame must be a PNG")
    offset = 8
    seen_header = seen_data = seen_end = data_ended = False
    compressed = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        payload = data[offset + 8:end - 4]
        crc, = struct.unpack_from(">I", data, end - 4)
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise ValueError("invalid PNG CRC")
        if not seen_header and kind != b"IHDR":
            raise ValueError("PNG must begin with IHDR")
        if kind == b"IHDR":
            if seen_header or length != 13:
                raise ValueError("invalid PNG header")
            if struct.unpack(">IIBBBBB", payload) != (WIDTH, HEIGHT, 8, 6, 0, 0, 0):
                raise ValueError("PNG must be 1080x1920, 8-bit non-interlaced RGBA")
            seen_header = True
        elif kind == b"IDAT":
            if data_ended:
                raise ValueError("PNG IDAT chunks must be contiguous")
            seen_data = True
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or not seen_data or end != len(data):
                raise ValueError("invalid PNG end")
            seen_end = True
        elif kind in (b"acTL", b"fcTL", b"fdAT"):
            raise ValueError("animated PNGs are not overlay frames")
        elif kind == b"PLTE":
            if seen_data or not length or length % 3 or length > 768:
                raise ValueError("invalid PNG palette")
        elif not (kind[0] & 0x20):
            raise ValueError("unsupported critical PNG chunk")
        if seen_data and kind != b"IDAT":
            data_ended = True
        offset = end
    if not seen_end:
        raise ValueError("PNG has no IEND")
    row_size = WIDTH * 4 + 1
    expected = row_size * HEIGHT
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise ValueError("invalid PNG image data") from exc
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid PNG scanline size or compressed stream")
    if any(raw[row] > 4 for row in range(0, expected, row_size)):
        raise ValueError("invalid PNG scanline filter")


@dataclass
class _Session:
    path: Path
    fps: float
    frame_count: int
    touched: float
    uploaded: int = 0
    size: int = 0
    complete: bool = False
    references: int = 0


class OverlayStore:
    """One worker process owns the registry and lock; uploads are atomic per batch."""

    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path)

    def cleanup(self) -> None:
        with self._lock:
            now = time.time()
            for key, session in list(self._sessions.items()):
                if not session.references and now - session.touched >= SESSION_TTL:
                    self._remove(session.path)
                    del self._sessions[key]
            # A restarted worker cannot resume the old in-memory manifest. Reap only
            # expired UUID directories, never arbitrary neighbouring files.
            if self.root.is_dir():
                for path in self.root.iterdir():
                    if path.name in self._sessions:
                        continue
                    try:
                        _uuid(path.name)
                        if now - path.lstat().st_mtime >= SESSION_TTL:
                            self._remove(path)
                    except (ValueError, FileNotFoundError):
                        continue

    def create(self, fps: float, count: int, width: int, height: int) -> str:
        fps = _fps(fps)
        if type(count) is not int or not 1 <= count <= MAX_FRAMES:
            raise ValueError("frameCount must be an integer between 1 and 18000")
        if count > overlay_frame_count(fps, MAX_DURATION):
            raise ValueError("frameCount exceeds 300 seconds at the supplied fps")
        if type(width) is not int or type(height) is not int or (width, height) != (WIDTH, HEIGHT):
            raise ValueError("overlay dimensions must be 1080x1920")
        with self._lock:
            self.cleanup()
            if len(self._sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "At most four overlay sessions may be active")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            key = str(uuid.uuid4())
            path = self.root / key
            path.mkdir(mode=0o700)
            self._sessions[key] = _Session(path, fps, count, time.time())
            return key

    def _get(self, key: str) -> _Session:
        _uuid(key)
        self.cleanup()
        session = self._sessions.get(key)
        if session is None or session.path.is_symlink() or not session.path.is_dir():
            raise ValueError("overlay does not exist or has expired")
        return session

    @staticmethod
    def _touch(session: _Session) -> None:
        session.touched = time.time()
        os.utime(session.path, None)

    def upload(self, key: str, start: int, files: list[BinaryIO]) -> int:
        # FastAPI invokes this synchronous endpoint in its thread pool. Multipart
        # spools have already been parsed; never await network I/O while locked.
        with self._lock:
            session = self._get(key)
            if session.complete:
                raise ValueError("completed overlays are immutable")
            if type(start) is not int or start != session.uploaded:
                raise ValueError("start must be the next contiguous frame index")
            if not 1 <= len(files) <= MAX_BATCH:
                raise ValueError("upload between 1 and 16 frames per batch")
            if start + len(files) > session.frame_count:
                raise ValueError("batch exceeds the declared frameCount")
            staged: list[tuple[Path, Path]] = []
            committed: list[Path] = []
            size = 0
            try:
                for offset, stream in enumerate(files):
                    data = stream.read(MAX_FRAME_BYTES + 1)
                    if len(data) > MAX_FRAME_BYTES or session.size + size + len(data) > MAX_SESSION_BYTES:
                        raise HTTPException(413, "Overlay frame or session byte limit exceeded")
                    _validate_png(data)
                    final = session.path / f"{start + offset:06d}.png"
                    pending = session.path / f"{start + offset:06d}.pending"
                    staged.append((pending, final))
                    with pending.open("xb") as handle:
                        handle.write(data)
                    size += len(data)
                for pending, final in staged:
                    os.replace(pending, final)
                    committed.append(final)
                self._touch(session)
                session.uploaded += len(files)
                session.size += size
                return session.uploaded
            except BaseException:
                for path in committed:
                    path.unlink(missing_ok=True)
                raise
            finally:
                for pending, _ in staged:
                    pending.unlink(missing_ok=True)

    @staticmethod
    def _dense(session: _Session) -> None:
        if session.uploaded != session.frame_count:
            raise ValueError("overlay frameCount is not complete")
        expected = {f"{i:06d}.png" for i in range(session.frame_count)}
        entries = list(session.path.iterdir())
        if {p.name for p in entries} != expected or any(p.is_symlink() or not p.is_file() for p in entries):
            raise ValueError("overlay frames are not a dense sequence")

    def complete(self, key: str) -> None:
        with self._lock:
            session = self._get(key)
            self._dense(session)
            session.complete = True
            self._touch(session)

    def _resolve(self, key: str, fps: float, duration: float) -> _Session:
        count = overlay_frame_count(fps, duration)
        session = self._get(key)
        if not session.complete:
            raise ValueError("overlay upload has not been completed")
        if session.fps != _fps(fps) or session.frame_count != count:
            raise ValueError("overlay fps or frameCount does not match the render")
        self._dense(session)
        self._touch(session)
        return session

    def resolve(self, key: str, fps: float, duration: float) -> str:
        with self._lock:
            return str(self._resolve(key, fps, duration).path)

    @contextmanager
    def reference(self, directory: str, fps: float, duration: float) -> Iterator[str]:
        with self._lock:
            path = Path(directory)
            key = _uuid(path.name)
            session = self._resolve(key, fps, duration)
            if path != session.path:
                raise ValueError("overlay directory must come from resolve_overlay")
            session.references += 1
        try:
            yield str(session.path)
        finally:
            with self._lock:
                session.references -= 1
                self._touch(session)

    def delete(self, key: str) -> None:
        with self._lock:
            session = self._get(key)
            if session.references:
                raise HTTPException(409, "Overlay is in use by a render")
            self._remove(session.path)
            del self._sessions[key]


store = OverlayStore(Path(WORK_DIR) / "render_overlays")


def resolve_overlay(overlay_id: str, fps: float, duration: float) -> str:
    """Return a trusted completed directory, or ValueError; never accept a path."""
    return store.resolve(overlay_id, fps, duration)


def reference_overlay(directory: str, fps: float, duration: float):
    """Pin a resolved directory while rendering; DELETE/expiry cannot remove it."""
    return store.reference(directory, fps, duration)


@asynccontextmanager
async def _lifespan(_app):
    async def sweep():
        while True:
            await asyncio.to_thread(store.cleanup)
            await asyncio.sleep(SWEEP_INTERVAL)

    task = asyncio.create_task(sweep())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


try:
    router = APIRouter(prefix="/v1/render/overlays", lifespan=_lifespan)
except TypeError:
    router = APIRouter(prefix="/v1/render/overlays")
    _sweep_task = None

    @router.on_event("startup")
    async def _start_sweep():
        global _sweep_task
        async def sweep():
            while True:
                await asyncio.to_thread(store.cleanup)
                await asyncio.sleep(SWEEP_INTERVAL)
        _sweep_task = asyncio.create_task(sweep())

    @router.on_event("shutdown")
    async def _stop_sweep():
        global _sweep_task
        if _sweep_task is not None:
            _sweep_task.cancel()
            with suppress(asyncio.CancelledError):
                await _sweep_task
            _sweep_task = None


class OverlayManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fps: StrictFloat | StrictInt
    frameCount: StrictInt
    width: StrictInt
    height: StrictInt


@router.post("")
def create_overlay(manifest: OverlayManifest):
    try:
        key = store.create(manifest.fps, manifest.frameCount, manifest.width, manifest.height)
        return {"overlayId": key}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{overlay_id}/frames")
def upload_frames(overlay_id: str, start: int = Form(...), files: list[UploadFile] = File(...)):
    try:
        # Client filenames and content types are not filesystem paths or evidence
        # of the format. Only bounded, validated bytes reach our numbered files.
        uploaded = store.upload(overlay_id, start, [file.file for file in files])
        return {"frameCount": uploaded}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        for file in files:
            file.file.close()


@router.post("/{overlay_id}/complete")
def complete_overlay(overlay_id: str):
    try:
        store.complete(overlay_id)
        return {"overlayId": overlay_id}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{overlay_id}", status_code=204)
def delete_overlay(overlay_id: str):
    try:
        store.delete(overlay_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
