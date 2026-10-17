"""End-to-end-ish tests for the LeRobot v2.1 writer on a synthetic episode.

The tabular path needs only pyarrow (no ffmpeg); the video path is exercised
separately and skipped when ffmpeg is absent.
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
from kiwi.export.writers.base import synthesize_action


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


def _tabular_episode(n: int = 12, dim: int = 8) -> ObservationEpisode:
    state = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (dim,), [f"s{i}" for i in range(dim)], state
        )
    }
    return ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task="pick",
        frame="marker",
        features=feats,
    )


def _info(ep: ObservationEpisode) -> DatasetInfo:
    return DatasetInfo(
        repo_id="me/test",
        robot_type="portable_bimanual",
        fps=ep.fps,
        features=ep.features,
    )


def test_v21_shape_one_feature_is_a_scalar_column(tmp_path):
    """Same loader rule as v3.0: a shape-(1,) feature is a scalar column."""
    ep = _tabular_episode(n=12)
    gap = (np.arange(12, dtype=np.float32) / 100.0)[:, None]
    ep.features["observation.pose_gap.left"] = FeatureColumn(
        "observation.pose_gap.left", "float32", (1,), ["left_pose_gap_s"], gap
    )
    w = get_writer("v2.1")
    w.open(tmp_path, _info(ep))
    w.add_episode(ep, 0, 0)
    w.close()
    table = pq.read_table(next((tmp_path / "data").rglob("*.parquet")))
    assert table.schema.field("observation.pose_gap.left").type == pa.float32()
    assert table.schema.field("observation.state").type == pa.list_(pa.float32())
    np.testing.assert_array_equal(
        table.column("observation.pose_gap.left").to_numpy(), gap[:, 0]
    )


def test_writes_full_v21_layout(tmp_path):
    ep = _tabular_episode(n=12, dim=8)
    w = get_writer("v2.1")
    w.open(tmp_path, _info(ep))
    w.add_episode(ep, 0, 0)
    w.close()

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 12
    assert info["total_tasks"] == 1
    assert info["fps"] == 30.0
    assert info["splits"] == {"train": "0:1"}
    feats = info["features"]
    assert feats["observation.state"]["shape"] == [8]
    assert feats["action"]["shape"] == [8]
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert col in feats

    pq_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(pq_path)
    assert table.num_rows == 12
    assert {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    } <= set(table.column_names)

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 8
    assert np.all(np.isfinite(stats["observation.state"]["std"]))

    tasks = (tmp_path / "meta" / "tasks.jsonl").read_text().splitlines()
    assert json.loads(tasks[0]) == {"task_index": 0, "task": "pick"}
    eps = (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()
    assert json.loads(eps[0])["length"] == 12

    # v2.1 loader reads per-episode stats from episodes_stats.jsonl.
    es = (tmp_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    rec = json.loads(es[0])
    assert rec["episode_index"] == 0
    st = rec["stats"]["observation.state"]
    assert len(st["mean"]) == 8 and st["count"] == [12]
    assert np.all(np.isfinite(st["std"]))


def test_action_next_state_matches_shifted_state(tmp_path):
    ep = _tabular_episode(n=6, dim=4)
    w = get_writer("v2.1")
    w.open(tmp_path, _info(ep))
    w.add_episode(ep, 0, 0)
    w.close()
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    action = np.array(table.column("action").to_pylist(), dtype=np.float32)
    state = ep.features["observation.state"].data
    np.testing.assert_allclose(action, synthesize_action(state, "next_state"))


def test_two_episodes_global_index_continuous(tmp_path):
    w = get_writer("v2.1")
    ep = _tabular_episode(n=5, dim=4)
    w.open(tmp_path, _info(ep))
    w.add_episode(_tabular_episode(n=5, dim=4), 0, 0)
    w.add_episode(_tabular_episode(n=5, dim=4), 1, 0)
    w.close()
    t1 = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000001.parquet")
    assert t1.column("index").to_pylist() == list(range(5, 10))
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 10


def test_unknown_format_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown export format"):
        get_writer("v1.6")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_writes_video_mp4(tmp_path):
    n = 10
    state = np.zeros((n, 4), dtype=np.float32)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (4,), None, state
        ),
        "observation.images.head": FeatureColumn(
            "observation.images.head",
            VIDEO_DTYPE,
            (64, 64, 3),
            ["height", "width", "channels"],
            _FakeFrames(n),
        ),
    }
    ep = ObservationEpisode(30.0, np.arange(n) / 30.0, "pick", "marker", feats)
    w = get_writer("v2.1")
    w.open(tmp_path, _info(ep))
    w.add_episode(ep, 0, 0)
    w.close()
    mp4 = (
        tmp_path
        / "videos"
        / "chunk-000"
        / "observation.images.head"
        / "episode_000000.mp4"
    )
    assert mp4.exists() and mp4.stat().st_size > 0
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.head"]["dtype"] == "video"
    assert info["total_videos"] == 1
    # Per-episode image stats: (3,1,1) per-channel, count = frame count.
    es = json.loads(
        (tmp_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()[0]
    )
    img = es["stats"]["observation.images.head"]
    assert np.array(img["mean"]).shape == (3, 1, 1)
    assert img["count"] == [n]


def test_an_ffmpeg_that_dies_at_startup_reports_its_stderr(tmp_path, monkeypatch):
    """The encoder's stderr, captured in a temp file, is reported in the error.

    An encoder that dies before reading breaks the pipe under the first write,
    and the raw BrokenPipeError names neither ffmpeg nor the cause. Faked
    rather than driven through a real bad codec so it also runs where ffmpeg
    is absent (CI).
    """
    from kiwi.export.writers import lerobot_v21 as v21

    class _DeadStdin:
        def write(self, _b):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            raise BrokenPipeError(32, "Broken pipe")

    class _DeadProc:
        def __init__(self, *a, **k):
            self.stdin = _DeadStdin()
            # Popen writes the child's stderr straight into the temp file.
            k["stderr"].write(b"Unknown encoder 'no_such_encoder'\n")

        def wait(self):
            return 8

    monkeypatch.setattr(v21.subprocess, "Popen", _DeadProc)
    with pytest.raises(RuntimeError) as e:
        v21.write_episode_video(
            tmp_path / "out.mp4", _FakeFrames(4), fps=30.0, codec="no_such_encoder"
        )
    assert "ffmpeg failed" in str(e.value)
    assert "Unknown encoder" in str(e.value)  # the diagnosis, not just "Broken pipe"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_a_real_bad_encoder_also_reports_ffmpeg(tmp_path):
    from kiwi.export.writers import lerobot_v21 as v21

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        v21.write_episode_video(
            tmp_path / "out.mp4", _FakeFrames(4), fps=30.0, codec="no_such_encoder"
        )
