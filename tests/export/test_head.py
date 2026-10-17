"""Head support, gap sampling and format dtypes survive the artifact boundary."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kiwi.core.digest import digest_bytes, digest_npz
from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode
from kiwi.export.build import world_transform
from kiwi.export.head import sample_head
from kiwi.export.replay_state import write_replay_state
from kiwi.export.schema import (
    HEAD_MARKER_POSE_KEY,
    HEAD_MARKER_SEGMENT_ID_KEY,
    HEAD_MARKER_VALID_KEY,
    STATE_KEY,
)
from kiwi.export.source import HeadSource
from kiwi.export.writers import DatasetInfo, get_writer

WORLD = {
    "frame": "shared_rear_metric_gravity_aligned",
    "scene": "room",
    "map_set_digest": "same-maps",
}


def _head(tmp_path, *, world=WORLD):
    directory = tmp_path / "head"
    directory.mkdir()
    trajectory = directory / "head_marker.tum"
    trajectory.write_text(
        f"# t tx ty tz qx qy qz qw  (frame={WORLD['frame']} body=head_marker; t on episode global clock)\n"
        "0 1 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n3 4 0 0 0 0 0 1\n4 5 0 0 0 0 0 1\n"
    )
    diag = directory / "head_marker_diagnostics.npz"
    np.savez(diag, t=np.array([0.0, 1.0, 3.0, 4.0]), segment_id=np.array([0, 0, 1, 1]))
    report = directory / "head_marker_report.json"
    _write_report(report, world_scene=world, trajectory=trajectory, diagnostics=diag)
    return HeadSource(trajectory, report, diag)


def _write_report(report, *, world_scene, trajectory, diagnostics, **extra):
    """A report binding the trajectory and diagnostics beside it, as the stage does."""
    report.write_text(
        json.dumps(
            {
                "world_scene": world_scene,
                "output": {"path": trajectory.name, "digest": digest_bytes(trajectory)},
                "quality_outputs": {
                    "diagnostics": {
                        "path": diagnostics.name,
                        "digest": digest_npz(diagnostics),
                    }
                },
                **extra,
            }
        )
    )


def test_head_gaps_trim_and_world_transform(tmp_path):
    source = _head(tmp_path)
    grid = np.array([-1.0, 0.5, 1.0, 2.0, 3.5, 5.0])
    features, meta = sample_head(
        source,
        grid,
        world_scene=WORLD,
        transform=world_transform((0.0, 10.0, 0.0), 90.0),
    )
    pose = features[HEAD_MARKER_POSE_KEY].data
    valid = features[HEAD_MARKER_VALID_KEY].data[:, 0]
    np.testing.assert_array_equal(valid, [False, True, True, False, True, False])
    np.testing.assert_array_equal(
        features[HEAD_MARKER_SEGMENT_ID_KEY].data[:, 0], [-1, 0, 0, -1, 1, -1]
    )
    assert np.isnan(pose[~valid]).all()
    np.testing.assert_allclose(
        pose[valid, :3], [[0, 11.5, 0], [0, 12, 0], [0, 14.5, 0]], atol=1e-6
    )
    np.testing.assert_allclose(
        pose[valid, 3:], np.tile([0, 0, np.sqrt(0.5), np.sqrt(0.5)], (3, 1)), atol=1e-6
    )
    assert meta["valid_frames"] == 3


def test_mismatched_world_and_diagnostic_rows_refused(tmp_path):
    source = _head(tmp_path, world={**WORLD, "map_set_digest": "rebuilt"})
    with pytest.raises(ValueError, match="world_scene"):
        sample_head(source, np.array([0.5]), world_scene=WORLD, transform=None)
    np.savez(
        source.diagnostics,
        t=np.array([0.0, 2.0, 3.0, 4.0]),
        segment_id=np.array([0, 0, 1, 1]),
    )
    _write_report(
        source.report,
        world_scene=WORLD,
        trajectory=source.trajectory,
        diagnostics=source.diagnostics,
    )
    with pytest.raises(ValueError, match="TUM rows"):
        sample_head(source, np.array([0.5]), world_scene=WORLD, transform=None)


def test_head_without_diagnostics_or_digests_is_refused(tmp_path):
    source = _head(tmp_path)
    report = json.loads(source.report.read_text())
    stripped = {k: v for k, v in report.items() if k != "output"}
    source.report.write_text(json.dumps(stripped))
    with pytest.raises(ValueError, match="no output digest"):
        sample_head(source, np.array([0.5]), world_scene=WORLD, transform=None)
    stripped = {k: v for k, v in report.items() if k != "quality_outputs"}
    source.report.write_text(json.dumps(stripped))
    with pytest.raises(ValueError, match="no diagnostics digest"):
        sample_head(source, np.array([0.5]), world_scene=WORLD, transform=None)
    source.report.write_text(json.dumps(report))
    source.diagnostics.unlink()
    with pytest.raises(ValueError, match="diagnostics are missing"):
        sample_head(source, np.array([0.5]), world_scene=WORLD, transform=None)


def _episode(source, grid):
    features, meta = sample_head(source, grid, world_scene=WORLD, transform=None)
    state = np.zeros((len(grid), 16), dtype=np.float32)
    state[:, [6, 14]] = 1
    features[STATE_KEY] = FeatureColumn(STATE_KEY, "float32", (16,), None, state)
    return ObservationEpisode(
        2.0, grid - grid[0], "pick", WORLD["frame"], features, {"head_marker": meta}
    )


@pytest.mark.parametrize("format", ["v2.1", "v3.0"])
@pytest.mark.parametrize("missing_first", [False, True])
def test_head_dtype_nan_and_statistics_read_back(tmp_path, format, missing_first):
    source = _head(tmp_path)
    populated = _episode(source, np.array([0.0, 0.5, 1.0, 2.0]))
    missing = _episode(None, np.array([0.0, 0.5, 1.0, 2.0]))
    episodes = [missing, populated] if missing_first else [populated, missing]
    root = tmp_path / "dataset"
    writer = get_writer(format)
    writer.open(
        root,
        DatasetInfo(
            "local/test",
            "portable_bimanual",
            2.0,
            episodes[0].features,
            include_action=False,
        ),
    )
    for i, episode in enumerate(episodes):
        writer.add_episode(episode, i, 0)
    writer.close()
    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables)
    # Shape-(1,) features are scalar columns - the type lerobot derives from
    # the declared shape when it loads the parquet; the 7-vector pose stays a
    # list column.
    assert table.schema.field(HEAD_MARKER_VALID_KEY).type == pa.bool_()
    assert table.schema.field(HEAD_MARKER_SEGMENT_ID_KEY).type == pa.int64()
    assert table.schema.field(HEAD_MARKER_POSE_KEY).type == pa.list_(pa.float32())
    poses = np.array(table[HEAD_MARKER_POSE_KEY].to_pylist())
    valid = np.array(table[HEAD_MARKER_VALID_KEY].to_pylist())
    assert np.isnan(poses[~valid]).all()
    assert valid.sum() == 3
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"][STATE_KEY]["shape"] == [16]
    assert info["features"][HEAD_MARKER_VALID_KEY]["dtype"] == "bool"
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats[HEAD_MARKER_POSE_KEY]["count"] == [3]
    assert stats[HEAD_MARKER_POSE_KEY]["mean"][0] == pytest.approx(1.5)
    assert "NaN" not in (root / "meta/stats.json").read_text()
    if format == "v3.0":
        episode_table = pq.read_table(next((root / "meta/episodes").rglob("*.parquet")))
        counts = episode_table[f"stats/{HEAD_MARKER_POSE_KEY}/count"].to_pylist()
        assert counts == ([None, [3]] if missing_first else [[3], None])
    else:
        records = [
            json.loads(line)
            for line in (root / "meta/episodes_stats.jsonl").read_text().splitlines()
        ]
        assert HEAD_MARKER_POSE_KEY not in records[0 if missing_first else 1]["stats"]


@pytest.mark.parametrize("format", ["v2.1", "v3.0"])
def test_all_missing_head_stats_are_not_fabricated(tmp_path, format):
    episode = _episode(None, np.arange(3) / 2.0)
    writer = get_writer(format)
    writer.open(
        tmp_path,
        DatasetInfo(
            "local/test",
            "portable_bimanual",
            2.0,
            episode.features,
            include_action=False,
        ),
    )
    writer.add_episode(episode, 0, 0)
    writer.close()
    stats = json.loads((tmp_path / "meta/stats.json").read_text())
    assert HEAD_MARKER_POSE_KEY not in stats
    assert stats[HEAD_MARKER_VALID_KEY]["count"] == [3]
    assert episode.meta["head_marker"]["statistics_status"] == "not_computed"


def test_replay_and_umi_refuse_head_instead_of_dropping(tmp_path):
    episode = _episode(None, np.arange(3) / 2.0)
    with pytest.raises(ValueError, match="head markers"):
        write_replay_state(episode, tmp_path / "replay.npz")
    from kiwi.export.writers.umi_zarr import UmiZarrWriter

    with pytest.raises(ValueError, match="head markers"):
        UmiZarrWriter().open(
            tmp_path,
            DatasetInfo("local/test", "portable_bimanual", 2.0, episode.features),
        )


def test_diagnostic_digest_prevents_rewritten_segments_from_bridging_a_gap(tmp_path):
    source = _head(tmp_path)
    np.savez(
        source.diagnostics,
        t=np.array([0.0, 1.0, 3.0, 4.0]),
        segment_id=np.array([0, 0, 0, 0]),
    )
    with pytest.raises(ValueError, match="mismatch"):
        sample_head(source, np.array([2.0]), world_scene=WORLD, transform=None)


def test_failed_head_retains_missing_reason(tmp_path):
    source = _head(tmp_path)
    source.trajectory.unlink()
    report = json.loads(source.report.read_text())
    report.update(status="graph_failed", reason="solver did not converge")
    source.report.write_text(json.dumps(report))
    features, meta = sample_head(
        source, np.array([0.0, 1.0]), world_scene=WORLD, transform=None
    )
    assert not features[HEAD_MARKER_VALID_KEY].data.any()
    assert meta["source_status"] == "graph_failed"
    assert meta["source_reason"] == "solver did not converge"


def test_head_known_placement_mismatch_is_refused(tmp_path):
    source = _head(tmp_path, world={**WORLD, "scene_frame_digest": "old-placement"})
    with pytest.raises(ValueError, match="scene_frame_digest"):
        sample_head(
            source,
            np.array([0.5]),
            world_scene={**WORLD, "scene_frame_digest": "new-placement"},
            transform=None,
        )


@pytest.mark.parametrize("format", ["v2.1", "v3.0"])
def test_head_stats_use_validity_even_for_foreign_finite_invalid_rows(tmp_path, format):
    episode = _episode(None, np.arange(3) / 2.0)
    episode.features[HEAD_MARKER_POSE_KEY].data[:] = 10.0
    writer = get_writer(format)
    writer.open(
        tmp_path,
        DatasetInfo(
            "local/test",
            "portable_bimanual",
            2.0,
            episode.features,
            include_action=False,
        ),
    )
    writer.add_episode(episode, 0, 0)
    writer.close()
    stats = json.loads((tmp_path / "meta/stats.json").read_text())
    assert HEAD_MARKER_POSE_KEY not in stats
