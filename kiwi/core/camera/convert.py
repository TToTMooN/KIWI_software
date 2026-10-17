#!/usr/bin/env python3
"""Convert an Insta360 ``.insv`` recording into per-lens ``.mp4`` files.

X4 / X5 cameras store the two lenses as **separate video streams** in
one ``.insv`` container, not side-by-side in a single dual-fisheye
frame. ``cv2.VideoCapture`` only reads stream 0, so we must demux the
two tracks up-front with ffmpeg. The video tracks are standard H.264,
so this is a stream-copy - no re-encode::

    pixi run python -m kiwi.core.camera.convert \\
        --input data/VID_20260515_120000_00_001.insv

Output (next to the input, by default)::

    <stem>_lens0.mp4
    <stem>_lens1.mp4

Stream 0 is typically the screen-side ("front") lens on the X4/X5, but
**verify with a one-frame inspection** before labelling - wiring the
wrong lens into pose code will silently produce mirrored / 180°-rotated
poses. The IMU track lives in an Insta360-proprietary box and is **not**
carried across.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path  # noqa: TC003 - tyro needs Path at runtime

import tyro

from kiwi.core.layout import refuse_raw_output

# ffmpeg muxer names for the output containers we emit, keyed by file suffix.
# Needed because the atomic ".part" temp name defeats extension-based inference.
_MUXER_BY_SUFFIX = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska", ".m4v": "mp4"}


@dataclasses.dataclass
class Args:
    """Demux an .insv into two per-lens .mp4 streams (ffmpeg stream-copy)."""

    input: Path
    """Source .insv file."""

    output_lens0: Path | None = None
    """Output for stream 0. Defaults to <input>_lens0.mp4."""

    output_lens1: Path | None = None
    """Output for stream 1. Defaults to <input>_lens1.mp4."""

    force: bool = False
    """Overwrite any existing outputs."""


def count_video_streams(path: Path, *, ffprobe: str = "ffprobe") -> int:
    """How many video streams ``path`` carries (X4/X5 .insv: 2, GO3 .mp4: 1).

    The demux/no-demux decision for calibration inputs hangs on this: a
    single-lens camera original has nothing to demux. Read with ffprobe
    (``VideoCapture`` only ever opens stream 0).
    """
    if shutil.which(ffprobe) is None:
        raise RuntimeError(
            f"{ffprobe} not found on PATH (ffmpeg ships it; pixi.toml includes "
            "ffmpeg; run inside `pixi shell`)"
        )
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        why = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"ffprobe failed for {path}: {why[-1] if why else proc.returncode}"
        )
    return len(proc.stdout.decode("ascii", "replace").split())


def _extract_stream(
    input_path: Path, output_path: Path, stream_index: int, force: bool
) -> Path:
    if output_path.exists() and not force:
        print(f"{output_path} exists; pass --force to overwrite. Skipping.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sidecar temp and atomically rename on success only: an
    # interrupted demux leaves only the .part, never the canonical name.
    tmp_path = output_path.with_name(output_path.name + ".part")
    # The ".part" extension defeats ffmpeg's format-from-extension inference
    # ("Unable to choose an output format"), so pass the muxer explicitly,
    # derived from the real output suffix (these demuxes are always .mp4).
    fmt = _MUXER_BY_SUFFIX.get(
        output_path.suffix.lower(), output_path.suffix.lstrip(".")
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:v:{stream_index}",
        "-c:v",
        "copy",
        "-an",
        "-f",
        fmt,
        str(tmp_path),
    ]
    print(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")
    return output_path


def convert(
    input_path: Path,
    output_lens0: Path,
    output_lens1: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Demux the two video streams of ``input_path`` into separate mp4 files."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH (pixi.toml includes it; run inside `pixi shell`)"
        )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    _extract_stream(input_path, output_lens0, 0, force)
    _extract_stream(input_path, output_lens1, 1, force)
    return output_lens0, output_lens1


def resolve_lens0_mp4(video: Path, force: bool = False) -> Path:
    """Return a per-lens-0 mp4 for ``video``, demuxing the .insv if needed.

    If ``video`` is already an mp4 it's returned as-is. For an .insv the
    sibling ``<stem>_lens0.mp4`` is used, or built via ``convert`` when
    either lens mp4 is missing, is older than the ``.insv``, or
    ``force=True``. Convenience wrapper for the common "load lens 0"
    pattern in the ``kiwi eef`` verbs.

    The freshness key is mtime: ``kiwi import`` copies with ``copystat``
    (:mod:`kiwi.core.camera.ingest`), so re-importing an *older* take over
    the same name keeps the card's mtime and its stale demux still tests
    fresh.

    Refuses a recording living in an episode's ``raw/`` - the demux there
    belongs to the convert stage, which lands it in the stream's own
    directory (:mod:`kiwi.core.layout`); pass that ``lens0.mp4`` instead.
    """
    if video.suffix.lower() != ".insv":
        return video
    lens0 = video.with_name(video.stem + "_lens0.mp4")
    refuse_raw_output(lens0, "demux")
    lens1 = video.with_name(video.stem + "_lens1.mp4")
    if not video.is_file():
        raise FileNotFoundError(
            f"{video} is not there - a demux beside it cannot be checked "
            f"against a source that is gone. Re-import the clip, or pass the "
            f"pre-demuxed lens0 mp4 instead."
        )
    # Freshness covers the PAIR this call writes, not just lens0.
    stale = [
        p
        for p in (lens0, lens1)
        if p.exists() and p.stat().st_mtime < video.stat().st_mtime
    ]
    if lens0.exists() and not force and not stale:
        print(f"reusing existing {lens0} (mtime newer than {video.name})")
        return lens0
    if stale:
        names = " + ".join(p.name for p in stale)
        print(f"{names} older than {video.name}; re-demuxing")
    print(f"demuxing {video} -> {lens0.name}, {lens1.name}")
    # The announced re-demux carries its own force: the caller's is False on
    # the staleness path by construction, and _extract_stream skips an
    # existing output unless forced.
    convert(video, lens0, lens1, force=force or bool(stale))
    return lens0


def main(args: Args) -> None:
    out0 = args.output_lens0 or args.input.with_name(args.input.stem + "_lens0.mp4")
    out1 = args.output_lens1 or args.input.with_name(args.input.stem + "_lens1.mp4")
    try:
        convert(args.input, out0, out1, force=args.force)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
