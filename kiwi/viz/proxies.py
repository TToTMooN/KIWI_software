"""H.264 proxy renditions - the player's video source for HEVC recordings.

The rerun viewer decodes H.264 (and AV1), not the HEVC our cameras record,
so every stream the player shows needs a small H.264 rendition unless the
source already is one (the GO3 ego mp4 sometimes is). Three encode settings:

* ``-force_key_frames`` at 1 s;
* ``-movflags +faststart`` - ``moov`` ahead of ``mdat``;
* an fps **cap** - a source at or below the cap keeps its native timing
  (29.97 is never re-timed to 30) while 119.88 and 50 come down. Time maps
  in seconds, so streams never need to agree on a frame rate.

Inside an episode a rendition lands as ``proxy.mp4`` in the stream's own
directory (:mod:`kiwi.core.layout`) - the caller passes ``dst`` - never
beside the source. Standalone, with no ``dst``, it is
``<stem>_proxy.mp4`` beside the source. Renditions are reused on mtime
freshness and written atomically (``.part`` + rename, the demux pattern
from ``kiwi.core.camera.convert``) so an interrupted encode never leaves a
half-written file under the canonical name.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "DEFAULT_FPS_CAP",
    "DEFAULT_SIZE",
    "PLAYABLE_CODECS",
    "PROXY_SUFFIX",
    "ProxyJob",
    "build_proxy",
    "ffmpeg_cmd",
    "is_stale",
    "playable_video",
    "probe_stream",
    "proxy_path",
]

PROXY_SUFFIX = "_proxy.mp4"
DEFAULT_SIZE = 480
DEFAULT_FPS_CAP = 30.0

# What the rerun viewer decodes natively (H.264 via its bundled decode path,
# AV1 via dav1d).
PLAYABLE_CODECS = frozenset({"h264", "av1"})


def proxy_path(video: Path) -> Path:
    """The rendition path for a source video: ``<stem>_proxy.mp4`` beside it."""
    video = Path(video)
    return video.with_name(video.stem + PROXY_SUFFIX)


@dataclasses.dataclass(frozen=True)
class ProxyJob:
    """One source stream and the rendition it should produce."""

    src: Path
    dst: Path


def probe_stream(
    path: Path, ffprobe: str = "ffprobe", *, video_stream: int = 0
) -> tuple[str, float]:
    """``(codec_name, fps)`` of a selected video stream (the first by default).

    ``("", 0.0)`` when the file is missing or unreadable - the caller
    treats that as "nothing to play", not an error, so a half-written file
    never crashes planning.
    """
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                f"v:{video_stream}",
                "-show_entries",
                "stream=codec_name,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        stream = json.loads(out)["streams"][0]
        codec = str(stream.get("codec_name", ""))
        rate = str(stream.get("r_frame_rate", "0/1"))
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
    ):
        return ("", 0.0)
    num, _, den = rate.partition("/")
    try:
        den_f = float(den or 1)
        fps = float(num) / den_f if den_f else 0.0
    except ValueError:
        fps = 0.0
    return (codec, fps)


def is_stale(job: ProxyJob) -> bool:
    """True when the rendition is missing or older than its source."""
    return not job.dst.is_file() or job.dst.stat().st_mtime < job.src.stat().st_mtime


def ffmpeg_cmd(
    src: Path,
    dst: Path,
    *,
    src_fps: float,
    size: int = DEFAULT_SIZE,
    fps_cap: float = DEFAULT_FPS_CAP,
    crf: int = 26,
    preset: str = "veryfast",
    video_stream: int = 0,
    hwaccel: str | None = None,
) -> list[str]:
    """The argv that renders one proxy. Pure - the caller probes ``src_fps``.

    ``-f mp4`` is explicit because the atomic ``.part`` temp name defeats
    ffmpeg's format-from-extension inference.
    """
    chain = [
        f"scale={size}:{size}:force_original_aspect_ratio=decrease",
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
    ]
    # Only decimate when meaningfully above the cap, so 29.97 stays 29.97.
    if src_fps > fps_cap + 1.0:
        chain.insert(0, f"fps={fps_cap:g}")
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        *(["-hwaccel", hwaccel] if hwaccel else []),
        "-i",
        str(src),
        "-map",
        f"0:v:{video_stream}",
        "-an",
        "-vf",
        ",".join(chain),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        "expr:gte(t,n_forced*1)",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(dst),
    ]


@contextmanager
def _proxy_lock(dst: Path):
    """Keep one writer per destination, across threads and processes.

    Leave the sidecar in place: removing it could let another caller lock
    a different inode while an existing waiter still holds the old one.
    """
    lock = dst.with_name(f".{dst.name}.lock")
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_proxy(
    job: ProxyJob,
    *,
    src_fps: float,
    size: int = DEFAULT_SIZE,
    fps_cap: float = DEFAULT_FPS_CAP,
    crf: int = 26,
    preset: str = "veryfast",
    video_stream: int = 0,
    hwaccel: str | None = None,
    force: bool = True,
) -> Path:
    """Encode atomically with one writer per destination.

    Direct calls rebuild by default. Cache-aware callers pass
    ``force=False`` to recheck freshness after acquiring the lock and reuse
    a rendition another writer has just finished.
    """
    job.dst.parent.mkdir(parents=True, exist_ok=True)
    with _proxy_lock(job.dst):
        if not force and not is_stale(job):
            return job.dst
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH (pixi.toml includes it; run inside `pixi shell`)"
            )
        tmp = job.dst.with_name(job.dst.name + ".part")
        cmd = ffmpeg_cmd(
            job.src,
            tmp,
            src_fps=src_fps,
            size=size,
            fps_cap=fps_cap,
            crf=crf,
            preset=preset,
            video_stream=video_stream,
            hwaccel=hwaccel,
        )
        # Flushed so the GUI can report progress while a worker is running.
        print(f"$ {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
            os.replace(tmp, job.dst)
        finally:
            tmp.unlink(missing_ok=True)
    return job.dst


def playable_video(
    video: Path,
    *,
    dst: Path | None = None,
    build: bool = True,
    force: bool = False,
    size: int = DEFAULT_SIZE,
    fps_cap: float = DEFAULT_FPS_CAP,
    probe: Callable[[Path], tuple[str, float]] = probe_stream,
    hwaccel: str | None = None,
) -> Path | None:
    """A path the rerun viewer can decode for ``video``, or ``None``.

    The source itself when its codec is already playable; otherwise the
    rendition at ``dst`` (an episode stream's ``proxy.mp4``; beside the
    source when omitted), reused when fresh, encoded when stale and
    ``build`` is true. ``None`` means no playable path exists: the source
    is unreadable, or it needs a proxy and ``build`` is false.

    ``hwaccel`` is forwarded to the decoder; only the decode path changes.
    """
    video = Path(video)
    codec, fps = probe(video)
    if not codec:
        return None
    if codec in PLAYABLE_CODECS:
        return video
    job = ProxyJob(src=video, dst=Path(dst) if dst is not None else proxy_path(video))
    if force or is_stale(job):
        if not build:
            return None
        build_proxy(
            job, src_fps=fps, size=size, fps_cap=fps_cap, hwaccel=hwaccel, force=force
        )
    return job.dst
