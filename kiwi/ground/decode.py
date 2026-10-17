"""Sequential frame extraction with ffmpeg - never seek.

One pass per video; the caller consumes a generator. On NVDEC ``hevc_cuvid``
decodes HEVC with a hardware resize; the generic ``-hwaccel cuda`` path
covers other codecs, the CPU path everything else.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

_MODES = ("cuvid", "hwaccel", "cpu")


def ffmpeg_binary() -> str:
    """The env's own ffmpeg when the interpreter's bin/ is not on PATH."""
    own = Path(sys.executable).parent / "ffmpeg"
    return str(own) if own.exists() else shutil.which("ffmpeg") or "ffmpeg"


def grid(indices: Iterable[int]) -> tuple[int, int]:
    """(start, stride): the coarsest regular grid containing every index."""
    idx = sorted(set(int(i) for i in indices))
    stride = int(np.gcd.reduce(np.diff(idx))) if len(idx) > 1 else 1
    return idx[0], stride


def _cmd(video, size, select, mode, pre=()):
    dec = {
        "cuvid": ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        + (["-resize", f"{size}x{size}"] if size else []),
        "hwaccel": ["-hwaccel", "cuda"],
        "cpu": [],
    }[mode]
    vf = ([select] if select else []) + (
        [f"scale={size}:{size}"] if size and mode != "cuvid" else []
    )
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", *dec, *pre]
    cmd += ["-i", str(video)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    return cmd + [
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def _stream(video, size, select, n_bytes, modes, pre=(), limit=None):
    """Yield raw frame buffers; the first mode whose first read yields a frame wins."""
    errs = []
    for mode in modes:
        with tempfile.TemporaryFile() as err:
            p = subprocess.Popen(
                _cmd(video, size, select, mode, pre),
                stdout=subprocess.PIPE,
                stderr=err,
                bufsize=n_bytes * 4,
            )
            buf = p.stdout.read(n_bytes)
            if len(buf) < n_bytes:
                p.kill()
                p.wait()
                err.seek(0)
                msg = err.read().decode(errors="replace").strip()[-300:]
                errs.append(f"[{mode}] {msg}")
                continue
            k = 0
            try:
                while len(buf) == n_bytes and (limit is None or k < limit):
                    yield buf
                    k += 1
                    buf = p.stdout.read(n_bytes)
            finally:
                p.kill()
                p.wait()
            return
    raise RuntimeError(f"ffmpeg decoded no frame from {video}:\n" + "\n".join(errs))


def frames(
    video: Path, indices: Iterable[int], size: int | None = None, native: int = 1920
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, rgb uint8 (size, size, 3)) for the indices, in order."""
    start, stride = grid(indices)
    want = set(int(i) for i in indices)
    last = max(want)
    w = size or native
    n = start
    select = f"select=not(mod(n-{start}\\,{stride}))*gte(n\\,{start})"
    for buf in _stream(video, size, select, w * w * 3, _MODES):
        if n in want:
            yield n, np.frombuffer(buf, np.uint8).reshape(w, w, 3)
        n += stride
        if n > last:
            break


def keyframes(video: Path, size=None, native=1920, limit=None) -> Iterator[np.ndarray]:
    """Yield rgb uint8 keyframes only (``-skip_frame nokey``; the generic NVDEC
    decoder honours it, hevc_cuvid does not)."""
    w = size or native
    for buf in _stream(
        video,
        size,
        None,
        w * w * 3,
        ("hwaccel", "cpu"),
        pre=("-skip_frame", "nokey"),
        limit=limit,
    ):
        yield np.frombuffer(buf, np.uint8).reshape(w, w, 3)


def prefetch(gen, depth: int = 48):
    """Run a frame generator in a thread with a bounded queue so the decoder keeps
    working while windows are processed. A generator exception re-raises here."""
    q: queue.Queue = queue.Queue(depth)
    end = object()

    def run():
        try:
            for item in gen:
                q.put(item)
            q.put(end)
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
            q.put(e)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not end:
        if isinstance(item, BaseException):
            raise item
        yield item
