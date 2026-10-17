"""UMI-style replay-buffer ``.zarr.zip`` - the Diffusion-Policy side export.

Layout (the ``ReplayBuffer`` shape the Diffusion-Policy toolchain consumes;
zarr 2.x store)::

    <out>/replay_buffer.zarr.zip
      data/
        robot0_eef_pos             (T, 3) f4   world frame, meters
        robot0_eef_rot_axis_angle  (T, 3) f4   world frame, radians
        robot0_gripper             (T, 1) f4   normalized [0, 1]
        robot1_...                             (second arm, when present)
        camera0_rgb                (T, H, W, 3) u8, chunked per frame
      meta/
        episode_ends               (E,) i8   cumulative frame counts

Arms map to ``robot{i}`` and video keys to ``camera{i}_rgb`` in feature
order - the UMI convention - and the root attrs record which KIWI label each
one is, plus the units, so nothing about the mapping is implicit. Rotation is
axis-angle (UMI's convention), converted from the state's quaternion columns.

No ``action`` array: the Diffusion-Policy toolchain derives actions from the
recorded poses at training time, so ``DatasetInfo.action_policy`` is ignored
here. Frames stream through in blocks (an episode's pixels never sit in RAM),
compressed with Blosc/zstd; resize/decimate with ``--resize``/``--fps``.

Zarr's ZipStore is write-once per key, and appending rewrites array metadata,
so episodes accumulate in a plain directory store that is packed into the
``.zarr.zip`` (and removed) on ``close()``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kiwi.core.observation_episode import ObservationEpisode

from ..schema import STATE_KEY
from .base import DatasetInfo, quat_spans

__all__ = ["UmiZarrWriter"]

_FRAME_BLOCK = 32  # frames appended per zarr write while streaming a video


class UmiZarrWriter:
    """Streaming writer for the UMI-style replay-buffer zip."""

    format = "umi-zarr"

    def __init__(self) -> None:
        self._root_dir: Path | None = None
        self._store_dir: Path | None = None
        self._info: DatasetInfo | None = None
        self._group = None
        self._episode_ends: list[int] = []
        self._total = 0
        self._arm_slices: list[tuple[str, int]] = []  # (label, state col offset)
        self._video_names: dict[str, str] = {}  # lerobot key -> camera{i}_rgb

    # -- lifecycle ----------------------------------------------------------

    def open(self, root: Path, info: DatasetInfo) -> None:
        from ..schema import GROUND_PLANE_KEY, HEAD_MARKER_POSE_KEY

        if HEAD_MARKER_POSE_KEY in info.features:
            raise ValueError("UMI does not support head markers; use LeRobot export")
        if GROUND_PLANE_KEY in info.features:
            raise ValueError(
                "UMI does not support the ground plane; use LeRobot export"
            )
        import zarr

        self._root_dir = Path(root)
        self._store_dir = self._root_dir / "replay_buffer.zarr"
        self._info = info
        state = info.features.get(STATE_KEY)
        if state is None or not state.names:
            raise ValueError(
                f"umi-zarr needs a named {STATE_KEY!r} feature to locate "
                f"the per-arm pose/gripper columns"
            )
        # One arm per quaternion block; its 8-column block starts 3 before it.
        self._arm_slices = []
        for q in quat_spans(state.names):
            label = state.names[q][: -len("_ee_qx")]
            self._arm_slices.append((label, q - 3))
        if not self._arm_slices:
            raise ValueError(
                "umi-zarr found no '<label>_ee_qx..qw' blocks in the state "
                "names - nothing to export as robot poses"
            )
        self._video_names = {
            k: f"camera{i}_rgb"
            for i, (k, f) in enumerate(
                (k, f) for k, f in info.features.items() if f.is_video
            )
        }
        if self._store_dir.exists():
            shutil.rmtree(self._store_dir)
        self._group = zarr.open_group(str(self._store_dir), mode="w")
        self._group.create_group("data")
        self._group.create_group("meta")
        self._group.attrs["kiwi_export"] = {
            "arms": {
                f"robot{i}": label for i, (label, _) in enumerate(self._arm_slices)
            },
            "cameras": {v: k for k, v in self._video_names.items()},
            "position": "meters, world frame",
            "rotation": "axis-angle, radians (from the state quaternions)",
            "gripper": "normalized [0, 1] (endpoints in export_provenance.json)",
        }

    def add_episode(
        self, episode: ObservationEpisode, episode_index: int, task_index: int
    ) -> None:
        assert self._group is not None and self._info is not None, "open() first"
        episode.validate()
        n = episode.num_frames
        state = np.asarray(episode.features[STATE_KEY].data, dtype=np.float64).reshape(
            n, -1
        )

        for i, (label, off) in enumerate(self._arm_slices):
            pos = state[:, off : off + 3].astype(np.float32)
            rotvec = (
                Rotation.from_quat(state[:, off + 3 : off + 7])
                .as_rotvec()
                .astype(np.float32)
            )
            grip = state[:, off + 7 : off + 8].astype(np.float32)
            self._append(f"robot{i}_eef_pos", pos, chunks=(4096, 3))
            self._append(f"robot{i}_eef_rot_axis_angle", rotvec, chunks=(4096, 3))
            self._append(f"robot{i}_gripper", grip, chunks=(4096, 1))

        for key, fc in episode.video_features().items():
            name = self._video_names[key]
            block: list[np.ndarray] = []
            for frame in fc.data:
                block.append(np.asarray(frame, dtype=np.uint8))
                if len(block) >= _FRAME_BLOCK:
                    self._append_frames(name, block, fc)
                    block = []
            if block:
                self._append_frames(name, block, fc)

        self._total += n
        self._episode_ends.append(self._total)

    def close(self) -> None:
        import zarr

        assert self._group is not None and self._store_dir is not None
        self._group["meta"].create_dataset(
            "episode_ends", data=np.asarray(self._episode_ends, dtype=np.int64)
        )
        zip_path = self._root_dir / "replay_buffer.zarr.zip"
        with zarr.ZipStore(str(zip_path), mode="w") as dst:
            zarr.copy_store(zarr.DirectoryStore(str(self._store_dir)), dst)
        shutil.rmtree(self._store_dir)

    # -- helpers -------------------------------------------------------------

    def _append(self, name: str, rows: np.ndarray, *, chunks: tuple) -> None:
        import numcodecs

        data = self._group["data"]
        if name not in data:
            data.create_dataset(
                name,
                shape=(0, *rows.shape[1:]),
                chunks=chunks,
                dtype=rows.dtype,
                compressor=numcodecs.Blosc(
                    cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
                ),
            )
        data[name].append(rows)

    def _append_frames(self, name: str, block: list[np.ndarray], fc) -> None:
        rows = np.stack(block)
        self._append(name, rows, chunks=(1, fc.data.height, fc.data.width, 3))
