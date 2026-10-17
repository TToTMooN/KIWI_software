"""The LeRobot v3.0 native writer: file-based shards, metadata-recovered episodes.

Spellings pinned against stock lerobot 0.4.4 (utils.py + the official
v2.1->v3.0 converter); the load-back proof runs in the isolated `lerobot`
pixi env (tests/export/loadback/, also the CI `loadback` job), so what CI
pins here is the layout and the metadata invariants a loader recovers
episodes from.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kiwi.core.observation_episode import VIDEO_DTYPE, FeatureColumn, ObservationEpisode
from kiwi.export.writers import DatasetInfo, get_writer
from kiwi.export.writers import lerobot_v30 as v30


@dataclasses.dataclass
class _FakeFrames:
    n: int
    height: int = 64
    width: int = 64

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n):
            yield np.full((self.height, self.width, 3), (i * 7) % 255, dtype=np.uint8)


def _episode(
    n: int = 10,
    dim: int = 4,
    task: str = "pick",
    video: bool = False,
    size: int = 64,
):
    state = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (dim,), [f"s{i}" for i in range(dim)], state
        )
    }
    if video:
        feats["observation.images.head"] = FeatureColumn(
            "observation.images.head",
            VIDEO_DTYPE,
            (size, size, 3),
            ["height", "width", "channels"],
            _FakeFrames(n, height=size, width=size),
        )
    return ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task=task,
        frame="marker",
        features=feats,
    )


def _info(ep) -> DatasetInfo:
    return DatasetInfo(
        repo_id="me/test",
        robot_type="portable_bimanual",
        fps=ep.fps,
        features=ep.features,
    )


def _write(tmp_path, episodes):
    w = get_writer("v3.0")
    w.open(tmp_path, _info(episodes[0]))
    tasks: dict[str, int] = {}
    for i, ep in enumerate(episodes):
        ti = tasks.setdefault(ep.task, len(tasks))
        w.add_episode(ep, i, ti)
    w.close()
    return w


def test_v30_layout_and_info(tmp_path):
    _write(tmp_path, [_episode(n=10), _episode(n=6, task="place")])

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 16
    assert info["total_tasks"] == 2
    # v3 drops v2.1's total_videos/total_chunks and adds the size knobs.
    assert "total_videos" not in info and "total_chunks" not in info
    assert info["data_files_size_in_mb"] == 100
    assert info["video_files_size_in_mb"] == 200
    assert (
        info["data_path"]
        == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    # A videoless dataset carries video_path: null (lerobot's use_videos rule).
    assert info["video_path"] is None
    assert info["splits"] == {"train": "0:2"}

    # Both episodes concatenated into ONE data shard.
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == 16
    assert table.column("episode_index").to_pylist() == [0] * 10 + [1] * 6
    assert table.column("index").to_pylist() == list(range(16))

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 4


def test_v30_shape_one_feature_is_a_scalar_column(tmp_path):
    """lerobot derives a column's type from the declared shape when it loads
    the parquet: (1,) is a scalar Value column, longer vectors are list
    columns. The default export carries two shape-(1,) features
    (observation.pose_gap.<arm>); written as list<float> they fail lerobot's
    cast ("Couldn't cast array of type list<element: float> to float")."""
    ep = _episode(n=10)
    gap = (np.arange(10, dtype=np.float32) / 100.0)[:, None]
    ep.features["observation.pose_gap.left"] = FeatureColumn(
        "observation.pose_gap.left", "float32", (1,), ["left_pose_gap_s"], gap
    )
    _write(tmp_path, [ep])
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert table.schema.field("observation.pose_gap.left").type == pa.float32()
    assert table.schema.field("observation.state").type == pa.list_(pa.float32())
    np.testing.assert_array_equal(
        table.column("observation.pose_gap.left").to_numpy(), gap[:, 0]
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["observation.pose_gap.left"]["shape"] == [1]
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert len(stats["observation.pose_gap.left"]["mean"]) == 1


def test_v30_episode_records_recover_boundaries(tmp_path):
    _write(tmp_path, [_episode(n=10), _episode(n=6, task="place")])
    eps = pq.read_table(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    rows = eps.to_pylist()
    assert [r["episode_index"] for r in rows] == [0, 1]
    assert [r["length"] for r in rows] == [10, 6]
    assert rows[0]["tasks"] == ["pick"] and rows[1]["tasks"] == ["place"]
    assert [r["dataset_from_index"] for r in rows] == [0, 10]
    assert [r["dataset_to_index"] for r in rows] == [10, 16]
    assert [r["data/chunk_index"] for r in rows] == [0, 0]
    assert [r["data/file_index"] for r in rows] == [0, 0]
    assert [r["meta/episodes/chunk_index"] for r in rows] == [0, 0]
    # Per-episode stats ride in the same records, flattened with '/'.
    assert len(rows[0]["stats/observation.state/mean"]) == 4
    assert rows[0]["stats/observation.state/count"] == [10]


def test_v30_tasks_parquet_keeps_strings_as_the_index(tmp_path):
    _write(tmp_path, [_episode(task="pick"), _episode(task="place")])
    t = pq.read_table(tmp_path / "meta" / "tasks.parquet")
    assert t.column("task_index").to_pylist() == [0, 1]
    assert t.column("__index_level_0__").to_pylist() == ["pick", "place"]
    # The pandas metadata is what makes pd.read_parquet reconstruct the
    # string index lerobot's load_tasks expects.
    meta = json.loads(t.schema.metadata[b"pandas"])
    assert meta["index_columns"] == ["__index_level_0__"]


def test_v30_data_sharding_assigns_indices_to_the_landing_file(tmp_path, monkeypatch):
    """Force tiny shards: every episode's record must name the file it is IN
    (the official converter gets this wrong for the rollover episode)."""
    monkeypatch.setattr(v30, "DATA_FILE_SIZE_MB", 1e-9)
    _write(tmp_path, [_episode(n=8), _episode(n=8), _episode(n=8)])
    eps = pq.read_table(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    for r in eps:
        shard = (
            tmp_path
            / "data"
            / f"chunk-{r['data/chunk_index']:03d}"
            / f"file-{r['data/file_index']:03d}.parquet"
        )
        table = pq.read_table(shard)
        assert r["episode_index"] in set(table.column("episode_index").to_pylist())
    # three tiny episodes -> three files
    assert [r["data/file_index"] for r in eps] == [0, 1, 2]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_v30_concatenates_video_and_records_timestamps(tmp_path):
    _write(tmp_path, [_episode(n=10, video=True), _episode(n=6, video=True)])
    shard = (
        tmp_path / "videos" / "observation.images.head" / "chunk-000" / "file-000.mp4"
    )
    assert shard.exists() and shard.stat().st_size > 0
    # No second file: both episodes share the shard.
    assert not (
        tmp_path / "videos" / "observation.images.head" / "chunk-000" / "file-001.mp4"
    ).exists()
    eps = pq.read_table(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    key = "videos/observation.images.head"
    assert eps[0][f"{key}/from_timestamp"] == 0.0
    assert eps[0][f"{key}/to_timestamp"] == pytest.approx(10 / 30.0)
    assert eps[1][f"{key}/from_timestamp"] == pytest.approx(10 / 30.0)
    assert eps[1][f"{key}/to_timestamp"] == pytest.approx(16 / 30.0)
    assert [e[f"{key}/file_index"] for e in eps] == [0, 0]
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert (
        info["video_path"]
        == "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    # Image stats attributed per episode despite the shared encode pass.
    assert np.array(eps[0]["stats/observation.images.head/mean"]).shape == (3, 1, 1)
    assert eps[0]["stats/observation.images.head/count"] == [10]
    assert eps[1]["stats/observation.images.head/count"] == [6]


def test_a_resolution_change_mid_shard_is_refused(tmp_path):
    """One shard is one ffmpeg encode sized from its first entry.

    Raw frames of another size do not fail the encoder - they misalign every
    following frame boundary and encode garbage - so the writer refuses at the
    append, before an encoder exists (which is also why this needs no ffmpeg).
    """
    w = get_writer("v3.0")
    small = _episode(video=True, size=32)
    w.open(tmp_path, _info(_episode(video=True)))
    w.add_episode(_episode(video=True), 0, 0)
    with pytest.raises(ValueError, match="single encode at one size"):
        w.add_episode(small, 1, 0)


def test_matching_sizes_append_to_one_shard(tmp_path):
    """The guard must not fire on matching sizes.

    Worth its own case because the concatenation test above needs ffmpeg, which
    CI does not install - without this, CI would only ever exercise the raise.
    """
    w = get_writer("v3.0")
    w.open(tmp_path, _info(_episode(video=True)))
    w.add_episode(_episode(n=10, video=True), 0, 0)
    w.add_episode(_episode(n=6, video=True, task="place"), 1, 0)
    shard = w._video_shards["observation.images.head"]
    assert shard.frames == 16 and len(shard.entries) == 2
