"""Container frame times of a video stream.

Frame times come from the stream's own packet PTS (probed with ffprobe,
converted through the probed time base) - the frame's own container time,
never a ``frame_index / fps`` reconstruction.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

__all__ = ["frame_times_ns", "video_time_base"]


def video_time_base(video: Path) -> Fraction:
    """The video stream's time base (seconds per PTS tick), via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # ffprobe can print stream-level entries more than once for real camera
    # containers (once per grouping it walks); the values must agree.
    values = [line.strip() for line in out.splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"{video}: ffprobe reported no video time base")
    if len(set(values)) != 1:
        raise RuntimeError(f"{video}: ambiguous video time bases {sorted(set(values))}")
    num, den = values[0].split("/")
    return Fraction(int(num), int(den))


def frame_times_ns(video: Path) -> list[int]:
    """Every video frame's container time in integer nanoseconds.

    Packet PTS (demux only, no decode) sorted into presentation order -
    with one packet per frame this IS the decoded frame times. Verified
    strictly increasing at nanosecond resolution; a recording violating
    that cannot feed the engine.
    """
    video = Path(video)
    time_base = video_time_base(video)
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ticks = sorted(int(line) for line in out.splitlines() if line and line != "N/A")
    if not ticks:
        raise RuntimeError(f"{video}: no video packets with timestamps")
    t_ns = [
        int(t * time_base.numerator * 1_000_000_000 // time_base.denominator)
        for t in ticks
    ]
    if any(b <= a for a, b in zip(t_ns, t_ns[1:])):
        raise RuntimeError(
            f"{video}: container timestamps are not strictly increasing at "
            f"nanosecond resolution - the recording cannot feed the engine"
        )
    return t_ns
