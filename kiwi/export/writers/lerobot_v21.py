"""LeRobot ``v2.1`` dataset writer (one parquet + one mp4 per episode).

Layout produced (matched against a real v2.1 reference dataset)::

    <root>/
      data/chunk-000/episode_000000.parquet        # tabular + action + bookkeeping
      videos/chunk-000/<video_key>/episode_000000.mp4
      meta/info.json  meta/tasks.jsonl  meta/episodes.jsonl
      meta/episodes_stats.jsonl  meta/stats.json

Video frames are streamed straight from each episode's :class:`FrameSource`
into a piped ``ffmpeg`` encode (HEVC / yuv420p), so a whole episode's pixels
never live in memory; a sample of frames is tapped off that same pass for
``meta/stats.json``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kiwi.core.observation_episode import ObservationEpisode

from ..schema import ACTION_KEY, HEAD_MARKER_POSE_KEY, HEAD_MARKER_VALID_KEY, STATE_KEY
from ..stats import RunningStats
from . import info as info_schema
from .base import DatasetInfo, chunk_index, quat_spans, synthesize_action

__all__ = ["CODEC_ENCODERS", "LeRobotV21Writer", "tabular_column"]


def tabular_column(arr: np.ndarray, shape: tuple[int, ...]) -> pa.Array:
    """One tabular feature as the parquet column lerobot reads it back as.

    lerobot derives the column type from the feature's declared ``shape``
    when it loads the parquet (``get_hf_features_from_features``): ``(1,)``
    is a scalar ``Value`` column, any longer vector a ``Sequence`` (list)
    column. The writer follows the same rule. ``arr`` is ``(n, width)`` in
    the feature's own dtype.
    """
    typ = pa.from_numpy_dtype(arr.dtype)
    if tuple(shape) == (1,):
        return pa.array(arr[:, 0], type=typ)
    return pa.array(arr.tolist(), type=pa.list_(typ))


CODEC_ENCODERS = {"hevc": "libx265", "h264": "libx264"}

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


class LeRobotV21Writer:
    """Incremental writer for the LeRobot v2.1 (episode-per-file) format."""

    format = "v2.1"

    def __init__(self) -> None:
        self._root: Path | None = None
        self._info: DatasetInfo | None = None
        self._stats = RunningStats()
        self._episodes: list[dict] = []
        self._episodes_stats: list[dict] = []
        self._tasks: dict[int, str] = {}
        self._video_keys: list[str] = []
        self._action = None
        self._quat_spans: tuple[int, ...] = ()
        self._total_frames = 0
        self._global_index = 0
        self._max_chunk = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self, root: Path, info: DatasetInfo) -> None:
        self._root = Path(root)
        self._info = info
        self._video_keys = [k for k, f in info.features.items() if f.is_video]
        state = info.features.get(STATE_KEY)
        if info.include_action and state is not None:
            self._action = info_schema.action_feature(state)
            self._quat_spans = quat_spans(state.names)
        (self._root / "meta").mkdir(parents=True, exist_ok=True)

    def add_episode(
        self, episode: ObservationEpisode, episode_index: int, task_index: int
    ) -> None:
        assert self._root is not None and self._info is not None, "open() first"
        episode.validate()
        n = episode.num_frames
        chunk = chunk_index(episode_index, self._info.chunks_size)
        self._max_chunk = max(self._max_chunk, chunk)

        # Per-episode stats feed meta/episodes_stats.jsonl (what the v2.1
        # loader reads); the global accumulator feeds the meta/stats.json
        # aggregate. Both are updated from the same data in one pass.
        ep_stats = RunningStats()
        self._write_parquet(episode, episode_index, task_index, chunk, n, ep_stats)
        for key, fc in episode.video_features().items():
            self._write_video(key, fc, episode_index, chunk, ep_stats)

        self._tasks[task_index] = episode.task
        self._episodes.append(
            {"episode_index": episode_index, "tasks": [episode.task], "length": n}
        )
        self._episodes_stats.append(
            {"episode_index": episode_index, "stats": ep_stats.result()}
        )
        self._total_frames += n
        self._global_index += n

    def close(self) -> None:
        assert self._root is not None and self._info is not None, "open() first"
        meta = self._root / "meta"
        self._write_json(meta / "info.json", self._build_info())
        self._write_jsonl(
            meta / "tasks.jsonl",
            [{"task_index": i, "task": self._tasks[i]} for i in sorted(self._tasks)],
        )
        self._write_jsonl(meta / "episodes.jsonl", self._episodes)
        self._write_jsonl(meta / "episodes_stats.jsonl", self._episodes_stats)
        self._write_json(meta / "stats.json", self._stats.result())

    # -- parquet ------------------------------------------------------------

    def _add_tabular(self, ep_stats: RunningStats, key: str, arr: np.ndarray) -> None:
        self._stats.add_tabular(key, arr)
        ep_stats.add_tabular(key, arr)

    def _write_parquet(
        self,
        episode: ObservationEpisode,
        episode_index: int,
        task_index: int,
        chunk: int,
        n: int,
        ep_stats: RunningStats,
    ) -> None:
        assert self._root is not None
        columns: dict[str, pa.Array] = {}

        # Tabular observation features (state first, then extras): a list
        # column with the declared dtype, or a scalar column for a
        # shape-(1,) feature (see tabular_column).
        state = None
        for key, fc in episode.tabular_features().items():
            arr = np.asarray(fc.data, dtype=np.dtype(fc.dtype)).reshape(n, -1)
            columns[key] = tabular_column(arr, fc.shape)
            stats_rows = arr
            if (
                key == HEAD_MARKER_POSE_KEY
                and HEAD_MARKER_VALID_KEY in episode.features
            ):
                valid = np.asarray(
                    episode.features[HEAD_MARKER_VALID_KEY].data, dtype=bool
                ).reshape(n)
                stats_rows = arr[valid]
            self._add_tabular(ep_stats, key, stats_rows)
            if key == STATE_KEY:
                state = arr

        # Synthesized action.
        if self._action is not None:
            if state is None:
                raise ValueError(f"include_action requires an {STATE_KEY!r} feature")
            act = synthesize_action(
                state, self._info.action_policy, quat_spans=self._quat_spans
            )
            columns[ACTION_KEY] = pa.array(list(act), type=pa.list_(pa.float32()))
            self._add_tabular(ep_stats, ACTION_KEY, act)

        # Bookkeeping columns.
        ts = np.asarray(episode.timestamps, dtype=np.float32)
        frame_index = np.arange(n, dtype=np.int64)
        episode_idx = np.full(n, episode_index, dtype=np.int64)
        index = np.arange(self._global_index, self._global_index + n, dtype=np.int64)
        task_idx = np.full(n, task_index, dtype=np.int64)
        columns["timestamp"] = pa.array(ts, type=pa.float32())
        columns["frame_index"] = pa.array(frame_index, type=pa.int64())
        columns["episode_index"] = pa.array(episode_idx, type=pa.int64())
        columns["index"] = pa.array(index, type=pa.int64())
        columns["task_index"] = pa.array(task_idx, type=pa.int64())
        for key, col in (
            ("timestamp", ts),
            ("frame_index", frame_index),
            ("episode_index", episode_idx),
            ("index", index),
            ("task_index", task_idx),
        ):
            self._add_tabular(ep_stats, key, col)

        out = self._root / DATA_PATH.format(
            episode_chunk=chunk, episode_index=episode_index
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out)

    # -- video --------------------------------------------------------------

    def _write_video(
        self, key: str, fc, episode_index: int, chunk: int, ep_stats
    ) -> None:
        assert self._root is not None and self._info is not None
        out = self._root / VIDEO_PATH.format(
            episode_chunk=chunk, video_key=key, episode_index=episode_index
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_episode_video(
            out,
            fc.data,
            fps=self._info.fps,
            codec=self._info.codec,
            on_sampled_frame=lambda rgb: (
                self._stats.add_image(key, rgb),
                ep_stats.add_image(key, rgb),
            ),
        )
        # Report the full frame count (not the sampled subset) as the stats
        # `count`.
        self._stats.add_image_frames(key, len(fc.data))
        ep_stats.add_image_frames(key, len(fc.data))

    # -- meta ---------------------------------------------------------------

    def _build_info(self) -> dict:
        assert self._info is not None
        info = self._info
        n_videos = len(self._episodes) * len(self._video_keys)
        return {
            "codebase_version": self.format,
            "robot_type": info.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": n_videos,
            "total_chunks": self._max_chunk + 1,
            "chunks_size": info.chunks_size,
            "fps": info.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": info_schema.build_info_features(
                info.features, fps=info.fps, action=self._action, codec=info.codec
            ),
        }

    @staticmethod
    def _write_json(path: Path, obj: dict) -> None:
        path.write_text(json.dumps(obj, indent=2) + "\n")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_episode_video(
    out: Path,
    src,
    *,
    fps: float,
    codec: str = "hevc",
    on_sampled_frame=None,
    sample_target: int = 30,
) -> None:
    """Stream one :class:`FrameSource` through a piped ffmpeg encode into ``out``.

    Shared by every writer that emits mp4 (v2.1 per-episode files, v3.0
    shards). ``on_sampled_frame`` sees ~``sample_target`` frames for stats
    accumulation, tapped off the same decode pass so pixels are only ever read
    once.
    """
    w, h = src.width, src.height
    encoder = CODEC_ENCODERS.get(codec, codec)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "hevc":
        cmd += ["-tag:v", "hvc1"]
    cmd.append(str(out))

    stride = max(1, len(src) // sample_target)
    # stderr -> temp file (not a PIPE).
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        assert proc.stdin is not None
        # An ffmpeg that dies at startup (unknown encoder, bad args, full disk)
        # breaks the pipe. BrokenPipeError is caught at BOTH the write and the
        # close (which flushes buffered bytes into the same dead pipe), so
        # control reaches the wait() below, which reports the captured stderr.
        # Any other exception out of `src` propagates unchanged.
        broke = False
        try:
            for i, frame in enumerate(src):
                rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                proc.stdin.write(rgb.tobytes())
                if on_sampled_frame is not None and i % stride == 0:
                    on_sampled_frame(rgb)
        except BrokenPipeError:
            broke = True
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                broke = True
        code = proc.wait()
        # A broken pipe with a zero exit means a truncated video; it raises
        # like a nonzero exit.
        if code != 0 or broke:
            errf.seek(0)
            raise RuntimeError(
                f"ffmpeg failed for {out} (exit {code}): "
                f"{errf.read().decode(errors='replace')}"
            )
