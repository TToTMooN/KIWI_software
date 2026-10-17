"""Frame export for the engine, the mapping subset and the engine settings.

The engine reads a directory of ``<t_ns>.jpg`` frames plus a ``timestamps.txt``
listing the frames to feed. Each stream is exported once at the map image size
(every frame, named by its own container time); mapping then lists a 30 fps
subset, localization lists every frame.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

MAP_IMAGE_SIZE = 960  # edge of the square frames the engine reads, pixels
MAP_FPS = 30.0  # frame rate of the mapping subset
N_FEATURES = 3500

_SETTINGS_TEMPLATE = """\
%YAML:1.0

File.version: "1.0"

Camera.type: "KannalaBrandt8"

Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}

Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.k3: {k3}
Camera1.k4: {k4}

Camera.width: {size}
Camera.height: {size}

Camera.fps: {fps}
Camera.RGB: 1

ORBextractor.nFeatures: {n_features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


def write_engine_settings(path: Path, kb4_params: np.ndarray, fps: float) -> Path:
    """Render the KannalaBrandt8 settings; ``kb4_params`` is the calibration at the map size."""
    p = [float(v) for v in np.asarray(kb4_params, dtype=float).ravel()]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _SETTINGS_TEMPLATE.format(
            fx=p[0],
            fy=p[1],
            cx=p[2],
            cy=p[3],
            k1=p[4],
            k2=p[5],
            k3=p[6],
            k4=p[7],
            size=MAP_IMAGE_SIZE,
            fps=round(fps),
            n_features=N_FEATURES,
        ),
        encoding="utf-8",
    )
    return path


def write_timestamps(path: Path, t_ns: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(t)}\n" for t in np.asarray(t_ns).ravel()), encoding="utf-8"
    )
    return path


def read_timestamps(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def export_stream_frames(
    video: Path, frame_times_ns: np.ndarray, out_dir: Path
) -> Path:
    """Decode every frame of ``video`` to ``out_dir/<t_ns>.jpg`` at the map image size.

    ``out_dir`` is emptied first. ffmpeg decodes in presentation order, so the i-th
    decoded frame carries the i-th container time. Returns the ``timestamps.txt`` that
    lists every frame.
    """
    out_dir = Path(out_dir)
    t_ns = np.asarray(frame_times_ns, dtype=np.int64).ravel()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough",
        "-vf", f"scale={MAP_IMAGE_SIZE}:{MAP_IMAGE_SIZE}", "-q:v", "2", str(out_dir / "%d.jpg"),
    ]  # fmt: skip
    subprocess.run(cmd)
    for seq_no, t in enumerate(t_ns, start=1):
        (out_dir / f"{seq_no}.jpg").rename(out_dir / f"{t}.jpg")
    return write_timestamps(out_dir / "timestamps.txt", t_ns)


def subset_times(t_ns: np.ndarray, native_fps_hz: float) -> np.ndarray:
    """Every ``round(native_fps_hz / MAP_FPS)``-th frame time, starting with the first."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    stride = int(round(native_fps_hz / MAP_FPS))
    return t_ns[::stride]


def native_fps(t_ns: np.ndarray) -> float:
    """Frame rate from the median inter-frame interval."""
    return 1e9 / float(np.median(np.diff(np.asarray(t_ns, dtype=np.int64))))


def rescale_kb4(
    kb4: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]
) -> np.ndarray:
    """Rescale ``[fx fy cx cy k1 k2 k3 k4]`` from one image size to another.

    The Kannala-Brandt distortion acts on angles, so only the pinhole part
    changes with the image size.
    """
    p = np.array(kb4, dtype=float).ravel()
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    p[0] *= sx
    p[1] *= sy
    p[2] *= sx
    p[3] *= sy
    return p
