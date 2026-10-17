"""Resolve viewer videos, including cached previews of unconverted recordings."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kiwi.viz.proxies import ProxyJob, build_proxy, playable_video, probe_stream
from kiwi.viz.sources import StreamSource


def _cache_path(source: Path, cache: Path, variant: str) -> Path:
    stamp = source.stat()
    identity = (
        "proxy-v2",
        stamp.st_dev,
        stamp.st_ino,
        stamp.st_size,
        stamp.st_mtime_ns,
        variant,
    )
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    return cache / "proxies" / f"{key}-{variant}.mp4"


def review_video(stream: StreamSource, cache: Path, *, lens: int = 0) -> Path | None:
    """Reuse derived video, or encode one raw INSV lens directly into the cache.

    The pipeline owns full-resolution demuxes and trajectory generation. The
    preview is a small H.264 file in the cache; no source episode files are
    created or changed here. Lens selection and native timing use the same
    video-stream ordering and proxy recipe as demuxed playback.
    """
    if lens not in (0, 1):
        raise ValueError(f"Expected lens 0 or 1, got {lens}")
    source = stream.video if lens == 0 else stream.stream_dir / "lens1.mp4"
    recording = stream.recording
    insv = recording.suffix.lower() == ".insv"
    # A processed episode may link to the same raw recording as the import
    # directory. Its per-lens demux is a stream-copy, so the cached raw lens
    # is already the matching preview, even though stream_dir is different.
    if insv and recording.is_file():
        cached = _cache_path(recording, cache, f"lens{lens}")
        if cached.is_file():
            return cached
    if source.is_file() and (
        not insv
        or not recording.is_file()
        or source.stat().st_mtime_ns >= recording.stat().st_mtime_ns
    ):
        proxy = (
            stream.proxy
            if stream.kind == "camera" and lens == 0
            else stream.stream_dir
            / ("proxy_front.mp4" if lens == 0 else "proxy_rear.mp4")
        )
        if proxy.is_file() and proxy.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return proxy
        return playable_video(
            source, dst=_cache_path(source, cache, "video"), hwaccel="auto"
        )

    if not insv or not recording.is_file():
        return None
    output = _cache_path(recording, cache, f"lens{lens}")
    if output.is_file():
        return output
    codec, fps = probe_stream(recording, video_stream=lens)
    if not codec:
        return None
    print(
        f"Preparing {stream.label} {'front' if lens == 0 else 'rear'} video preview "
        f"from {recording.name}…",
        flush=True,
    )
    return build_proxy(
        ProxyJob(recording, output),
        src_fps=fps,
        video_stream=lens,
        hwaccel="auto",
        force=False,
    )


def _proxy_workers(count: int) -> int:
    """How many encodes to keep in flight."""
    return max(1, min(4, count, (os.cpu_count() or 4) // 3))


def prefetch_videos(
    requests: Sequence[tuple[StreamSource, int]],
    cache: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Warm the preview cache for several streams at once.

    Nothing is returned: callers resolve through :func:`review_video`, which
    finds the prefetched file. Failures are swallowed here; that resolve
    re-runs the encode and turns a real failure into a planner note.
    """
    ordered: list[tuple[StreamSource, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for stream, lens in requests:
        source = stream.video if lens == 0 else stream.stream_dir / "lens1.mp4"
        # Avoid identical requests. Distinct streams can still resolve to
        # one cache destination; build_proxy serializes those writers.
        key = (str(source), str(stream.recording), lens)
        if key not in seen:
            seen.add(key)
            ordered.append((stream, lens))
    if not ordered:
        return

    def encode(stream: StreamSource, lens: int) -> None:
        try:
            review_video(stream, cache, lens=lens)
        except Exception:
            pass

    done = 0
    if on_progress:
        on_progress(done, len(ordered))
    with ThreadPoolExecutor(max_workers=_proxy_workers(len(ordered))) as pool:
        futures = [pool.submit(encode, stream, lens) for stream, lens in ordered]
        for _ in as_completed(futures):
            done += 1
            if on_progress:
                on_progress(done, len(ordered))
