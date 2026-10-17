"""LeRobot ``v3.0`` dataset writer - file-based, buffered, aggregated on close.

v3.0 concatenates MANY episodes into a few large shards and recovers episode
boundaries from metadata, not filenames - so unlike v2.1's write-as-you-go,
this writer buffers tabular episodes until a shard reaches the size knob, and
keeps one ffmpeg encode open per (video key, shard), appending episodes
back-to-back::

    <root>/
      data/chunk-000/file-000.parquet         # rows from MANY episodes
      videos/<video_key>/chunk-000/file-000.mp4
      meta/info.json                          # v3 path templates + size knobs
      meta/tasks.parquet                      # task strings as the index
      meta/episodes/chunk-000/file-000.parquet
      meta/stats.json

Field spellings are matched against stock lerobot 0.4.4
(``lerobot/datasets/utils.py`` + the official v2.1->v3.0 converter): the
episode records carry ``data/chunk_index`` / ``data/file_index`` /
``dataset_from_index`` / ``dataset_to_index``, per-key
``videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}``,
``meta/episodes/{chunk_index,file_index}``, and the per-episode stats
flattened as ``stats/<feature>/<stat>`` columns.

An episode's shard indices are assigned when its shard is FLUSHED, so they
always name the file the episode actually landed in.

``fps`` is written as the true float (29.97, 119.88...).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kiwi.core.observation_episode import ObservationEpisode

from ..schema import ACTION_KEY, HEAD_MARKER_POSE_KEY, HEAD_MARKER_VALID_KEY, STATE_KEY
from ..stats import RunningStats
from . import info as info_schema
from .base import DatasetInfo, quat_spans, synthesize_action
from .lerobot_v21 import tabular_column, write_episode_video

__all__ = ["LeRobotV30Writer"]

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"

# lerobot's defaults (DEFAULT_DATA_FILE_SIZE_IN_MB / DEFAULT_VIDEO_FILE_SIZE_IN_MB).
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 200


def _flatten_stats(stats: dict) -> dict:
    """``{"observation.state": {"min": [...]}}`` -> ``{"stats/observation.state/min": [...]}``."""
    out: dict = {}
    for feature, block in stats.items():
        for stat, value in block.items():
            out[f"stats/{feature}/{stat}"] = value
    return out


class _VideoShard:
    """One (video key, shard) mp4 accumulating episodes back-to-back.

    v3 keeps a single growing file per shard; episode boundaries live in the
    episode records as ``from_timestamp`` / ``to_timestamp`` seconds within
    the file. ffmpeg cannot append to a finished mp4, so the shard holds each
    episode's lazy :class:`FrameSource` (plus that episode's stats
    accumulator, so sampled pixels attribute to the right record) and encodes
    them in ONE piped pass when the shard flushes - deferring decode, not
    buffering pixels.
    """

    def __init__(self, chunk_index: int, file_index: int) -> None:
        self.chunk_index = chunk_index
        self.file_index = file_index
        self.entries: list[tuple[object, RunningStats]] = []  # (FrameSource, ep stats)
        self.frames = 0

    class _Concat:
        """Iterate the entries as one stream, sampling stats per episode."""

        def __init__(self, entries, key: str, global_stats: RunningStats) -> None:
            self._entries = entries
            self._key = key
            self._global = global_stats
            first = entries[0][0]
            self.height = first.height
            self.width = first.width

        def __len__(self) -> int:
            return sum(len(src) for src, _ in self._entries)

        def __iter__(self):
            for src, ep_stats in self._entries:
                stride = max(1, len(src) // 30)
                for i, frame in enumerate(src):
                    if i % stride == 0:
                        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                        self._global.add_image(self._key, rgb)
                        ep_stats.add_image(self._key, rgb)
                    yield frame

    def write(
        self, out: Path, *, key: str, fps: float, codec: str, global_stats
    ) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_episode_video(
            out,
            self._Concat(self.entries, key, global_stats),
            fps=fps,
            codec=codec,
            on_sampled_frame=None,
        )


class LeRobotV30Writer:
    """Buffered writer for the LeRobot v3.0 (file-based) format."""

    format = "v3.0"

    def __init__(self) -> None:
        self._root: Path | None = None
        self._info: DatasetInfo | None = None
        self._stats = RunningStats()
        self._action = None
        self._quat_spans: tuple[int, ...] = ()
        self._video_keys: list[str] = []
        self._tasks: dict[int, str] = {}
        self._episode_records: list[dict] = []
        self._total_frames = 0
        # data shard state: buffered per-episode tables + the records that
        # will receive this shard's indices when it flushes.
        self._data_chunk = 0
        self._data_file = 0
        self._pending_tables: list[pa.Table] = []
        self._pending_records: list[dict] = []
        self._pending_bytes = 0
        # per-video-key shard state
        self._video_shards: dict[str, _VideoShard] = {}

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
        for key in self._video_keys:
            self._video_shards[key] = _VideoShard(0, 0)

    def add_episode(
        self, episode: ObservationEpisode, episode_index: int, task_index: int
    ) -> None:
        assert self._root is not None and self._info is not None, "open() first"
        episode.validate()
        n = episode.num_frames
        self._tasks[task_index] = episode.task

        ep_stats = RunningStats()
        table = self._episode_table(episode, episode_index, task_index, n, ep_stats)

        record: dict = {
            "episode_index": episode_index,
            "tasks": [episode.task],
            "length": n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "dataset_from_index": self._total_frames,
            "dataset_to_index": self._total_frames + n,
        }

        # Video: append this episode's frames to each key's current shard.
        fps = self._info.fps
        for key, fc in episode.video_features().items():
            shard = self._video_shards[key]
            record[f"videos/{key}/chunk_index"] = shard.chunk_index
            record[f"videos/{key}/file_index"] = shard.file_index
            record[f"videos/{key}/from_timestamp"] = shard.frames / fps
            record[f"videos/{key}/to_timestamp"] = (shard.frames + len(fc.data)) / fps
            # A shard is ONE ffmpeg encode sized from its first entry, and the
            # frames go in as raw bytes. An episode at another resolution is
            # refused at the append, before any encoder exists.
            if shard.entries:
                first = shard.entries[0][0]
                if (fc.data.height, fc.data.width) != (first.height, first.width):
                    raise ValueError(
                        f"episode {episode_index} feature {key!r} is "
                        f"{fc.data.width}x{fc.data.height} but the open shard "
                        f"(chunk {shard.chunk_index}, file {shard.file_index}) "
                        f"is {first.width}x{first.height}; one shard is a "
                        f"single encode at one size. Export with --resize so "
                        f"every episode matches, or export separate datasets."
                    )
            shard.entries.append((fc.data, ep_stats))
            shard.frames += len(fc.data)
            self._stats.add_image_frames(key, len(fc.data))
            ep_stats.add_image_frames(key, len(fc.data))
            # Rough size gate: raw frames compress ~50x under HEVC; flush the
            # shard when the ESTIMATED encoded size passes the knob. The knob
            # is a target, not a contract - the loader recovers boundaries
            # from metadata whatever the sharding.
            est_mb = shard.frames * fc.data.height * fc.data.width * 3 / 50 / 1e6
            if est_mb >= VIDEO_FILE_SIZE_MB:
                self._flush_video(key)

        # The record keeps its accumulator; image pixel stats arrive at shard
        # flush (encode time), so flattening happens once, at close.
        record["__ep_stats__"] = ep_stats
        self._episode_records.append(record)
        self._pending_records.append(record)
        self._pending_tables.append(table)
        self._pending_bytes += table.nbytes
        self._total_frames += n
        if self._pending_bytes / 1e6 >= DATA_FILE_SIZE_MB:
            self._flush_data()

    def close(self) -> None:
        assert self._root is not None and self._info is not None, "open() first"
        if self._pending_tables:
            self._flush_data()
        for key in self._video_keys:
            if self._video_shards[key].entries:
                self._flush_video(key)
        meta = self._root / "meta"
        (self._root / "meta").mkdir(parents=True, exist_ok=True)
        self._write_tasks()
        self._write_episodes()
        (meta / "info.json").write_text(json.dumps(self._build_info(), indent=2) + "\n")
        (meta / "stats.json").write_text(json.dumps(self._stats.result()) + "\n")

    # -- data shards ---------------------------------------------------------

    def _episode_table(
        self,
        episode: ObservationEpisode,
        episode_index: int,
        task_index: int,
        n: int,
        ep_stats: RunningStats,
    ) -> pa.Table:
        columns: dict[str, pa.Array] = {}
        state = None
        for key, fc in episode.tabular_features().items():
            arr = np.asarray(fc.data, dtype=np.dtype(fc.dtype)).reshape(n, -1)
            # list column, or a scalar column for a shape-(1,) feature - the
            # type lerobot's loader derives from the declared shape.
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
            self._stats.add_tabular(key, stats_rows)
            ep_stats.add_tabular(key, stats_rows)
            if key == STATE_KEY:
                state = arr
        if self._action is not None:
            if state is None:
                raise ValueError(f"include_action requires an {STATE_KEY!r} feature")
            act = synthesize_action(
                state, self._info.action_policy, quat_spans=self._quat_spans
            )
            columns[ACTION_KEY] = pa.array(list(act), type=pa.list_(pa.float32()))
            self._stats.add_tabular(ACTION_KEY, act)
            ep_stats.add_tabular(ACTION_KEY, act)

        ts = np.asarray(episode.timestamps, dtype=np.float32)
        frame_index = np.arange(n, dtype=np.int64)
        episode_idx = np.full(n, episode_index, dtype=np.int64)
        index = np.arange(self._total_frames, self._total_frames + n, dtype=np.int64)
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
            self._stats.add_tabular(key, col)
            ep_stats.add_tabular(key, col)
        return pa.table(columns)

    def _flush_data(self) -> None:
        assert self._root is not None
        out = self._root / DATA_PATH.format(
            chunk_index=self._data_chunk, file_index=self._data_file
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self._pending_tables), out)
        # Indices assigned at flush time: every buffered episode names the
        # shard it actually landed in.
        for record in self._pending_records:
            record["data/chunk_index"] = self._data_chunk
            record["data/file_index"] = self._data_file
        self._pending_tables = []
        self._pending_records = []
        self._pending_bytes = 0
        self._data_file += 1
        if self._data_file >= self._info.chunks_size:
            self._data_chunk += 1
            self._data_file = 0

    # -- video shards ---------------------------------------------------------

    def _flush_video(self, key: str) -> None:
        assert self._root is not None and self._info is not None
        shard = self._video_shards[key]
        out = self._root / VIDEO_PATH.format(
            video_key=key, chunk_index=shard.chunk_index, file_index=shard.file_index
        )
        shard.write(
            out,
            key=key,
            fps=self._info.fps,
            codec=self._info.codec,
            global_stats=self._stats,
        )
        file_index = shard.file_index + 1
        chunk_index = shard.chunk_index
        if file_index >= self._info.chunks_size:
            chunk_index += 1
            file_index = 0
        self._video_shards[key] = _VideoShard(chunk_index, file_index)

    # -- meta -----------------------------------------------------------------

    def _write_tasks(self) -> None:
        # lerobot reads meta/tasks.parquet as a pandas frame with the task
        # STRINGS as the index and a task_index column; spelled with pyarrow.
        items = sorted(self._tasks.items())
        table = pa.table(
            {
                "task_index": pa.array([i for i, _ in items], type=pa.int64()),
                "__index_level_0__": pa.array([t for _, t in items], type=pa.string()),
            }
        )
        meta = table.schema.metadata or {}
        pandas_meta = {
            b"pandas": json.dumps(
                {
                    "index_columns": ["__index_level_0__"],
                    "columns": [
                        {
                            "name": "task_index",
                            "field_name": "task_index",
                            "pandas_type": "int64",
                            "numpy_type": "int64",
                            "metadata": None,
                        },
                        {
                            "name": None,
                            "field_name": "__index_level_0__",
                            "pandas_type": "unicode",
                            "numpy_type": "object",
                            "metadata": None,
                        },
                    ],
                    "column_indexes": [],
                    "creator": {"library": "pyarrow"},
                    "pandas_version": "2.0.0",
                }
            ).encode()
        }
        out = self._root / TASKS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata({**meta, **pandas_meta}), out)

    def _write_episodes(self) -> None:
        # One metadata file (chunk-000/file-000), like lerobot's own
        # write_episodes; every record has every column, so missing keys
        # (an episode with no video) become nulls consistently.
        for record in self._episode_records:
            record.update(_flatten_stats(record.pop("__ep_stats__").result()))
        keys: list[str] = []
        for record in self._episode_records:
            for k in record:
                if k not in keys:
                    keys.append(k)
        columns = {k: [record.get(k) for record in self._episode_records] for k in keys}
        out = self._root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out)

    def _build_info(self) -> dict:
        assert self._info is not None
        info = self._info
        return {
            "codebase_version": self.format,
            "robot_type": info.robot_type,
            "total_episodes": len(self._episode_records),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": info.chunks_size,
            "data_files_size_in_mb": DATA_FILE_SIZE_MB,
            "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
            "fps": info.fps,
            "splits": {"train": f"0:{len(self._episode_records)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self._video_keys else None,
            "features": info_schema.build_info_features(
                info.features, fps=info.fps, action=self._action, codec=info.codec
            ),
        }
