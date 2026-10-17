"""Ground support: the plane and heights, NaN when absent, the transform, the archive."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from kiwi.core.digest import digest_bytes, digest_npz
from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode
from kiwi.export.archive import SourceArchive
from kiwi.export.build import world_transform
from kiwi.export.ground import sample_ground, transform_plane
from kiwi.export.replay_state import write_replay_state
from kiwi.export.schema import GROUND_PLANE_KEY, STATE_KEY, camera_height_key
from kiwi.export.source import GroundSource
from kiwi.export.writers import DatasetInfo, get_writer

WORLD = {
    "frame": "shared_rear_metric_gravity_aligned",
    "scene": "room",
    "map_set_digest": "same-maps",
}
FLOOR_Z = -1.2
GRID = np.array([0.0, 0.5, 1.0])
# Camera optical centres in the scene frame, one row per grid instant.
CAMERAS = {
    "left": np.array([[0.0, 0.0, 0.3], [0.5, 0.0, 0.4], [1.0, 2.0, 0.5]]),
    "right": np.array([[0.0, 1.0, 0.1], [0.0, 1.0, 0.2], [0.0, 1.0, 0.3]]),
}


def _ground(
    root,
    *,
    status="computed",
    floor_z=FLOOR_Z,
    world=WORLD,
    frame=None,
    observations=True,
):
    directory = Path(root) / "ground"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "ground_report.json"
    npz = directory / "ground_observations.npz"
    document = {
        "schema": "ground_v1",
        "status": status,
        "verdict": "GREEN" if status == "computed" else "RED",
        "reasons": [] if status == "computed" else ["no usable window"],
        "floor_z_m": floor_z if status == "computed" else None,
        "sigma_m": 0.006,
        "frame": frame or WORLD["frame"],
        "world_scene": world,
        "inputs": {},
        "inputs_base": ".",
    }
    if observations:
        np.savez(npz, arm=np.array(["left"]), z=np.array([floor_z]))
        document["quality_outputs"] = {
            "observations": {"path": npz.name, "digest": digest_npz(npz)}
        }
    report.write_text(json.dumps(document))
    return GroundSource(report, npz if observations else None)


def test_plane_and_heights_follow_the_hands_transform(tmp_path):
    source = _ground(tmp_path)
    transform = world_transform((1.0, 2.0, 3.0), 90.0)
    exported = {
        label: p @ transform[:3, :3].T + transform[:3, 3]
        for label, p in CAMERAS.items()
    }
    features, meta = sample_ground(
        source, GRID, exported, world_scene=WORLD, transform=transform
    )
    plane = features[GROUND_PLANE_KEY].data
    assert plane.dtype == np.float32 and plane.shape == (3, 4)
    assert features[GROUND_PLANE_KEY].names == ["nx", "ny", "nz", "d"]
    # z = floor_z lifted by the 3 m offset: n stays up, d = -floor_z - t_z
    np.testing.assert_allclose(
        plane, np.tile([0.0, 0.0, 1.0, -FLOOR_Z - 3.0], (3, 1)), atol=1e-6
    )
    for label, p in CAMERAS.items():
        height = features[camera_height_key(label)].data
        assert height.dtype == np.float32 and height.shape == (3, 1)
        # the same number as in the scene frame: the re-seating moved both sides
        np.testing.assert_allclose(height[:, 0], p[:, 2] - FLOOR_Z, atol=1e-6)
        np.testing.assert_allclose(
            exported[label] @ plane[0, :3] + plane[0, 3], height[:, 0], atol=1e-5
        )
    assert meta["status"] == "computed" and meta["valid_frames"] == 3
    assert meta["statistics_status"] == "computed"
    assert meta["floor_z_m"] == FLOOR_Z and meta["verdict"] == "GREEN"
    assert meta["placement_verification"] == "unknown"
    assert meta["observations"] == str(source.observations)


def test_transform_plane_keeps_points_on_the_plane_under_sim3():
    rng = np.random.default_rng(0)
    transform = np.eye(4)
    transform[:3, :3] = 2.0 * Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    transform[:3, 3] = [0.4, -1.0, 2.5]
    plane = np.array([0.0, 0.0, 1.0, -FLOOR_Z])
    moved = transform_plane(plane, transform)
    points = np.column_stack([rng.normal(size=(5, 2)), np.full(5, FLOOR_Z)])
    exported = points @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(exported @ moved[:3] + moved[3], 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(moved[:3]), 1.0)
    np.testing.assert_array_equal(transform_plane(plane, None), plane)


@pytest.mark.parametrize("kind", ["none", "missing", "no_floor"])
def test_absent_or_failed_ground_is_nan_with_its_status(tmp_path, kind):
    if kind == "none":
        source = None
    elif kind == "missing":
        source = GroundSource(tmp_path / "ground" / "ground_report.json")
    else:
        source = _ground(tmp_path, status="no_floor")
    features, meta = sample_ground(
        source, GRID, CAMERAS, world_scene=WORLD, transform=None
    )
    assert np.isnan(features[GROUND_PLANE_KEY].data).all()
    for label in CAMERAS:
        assert np.isnan(features[camera_height_key(label)].data).all()
    assert meta["status"] == ("no_floor" if kind == "no_floor" else "absent")
    assert meta["valid_frames"] == 0 and meta["statistics_status"] == "not_computed"
    if kind == "no_floor":
        assert meta["reasons"] == ["no usable window"] and meta["verdict"] == "RED"


def test_world_and_frame_disagreements_are_refused(tmp_path):
    source = _ground(tmp_path, world={**WORLD, "map_set_digest": "rebuilt"})
    with pytest.raises(ValueError, match="world_scene"):
        sample_ground(source, GRID, CAMERAS, world_scene=WORLD, transform=None)
    source = _ground(tmp_path, frame="another_frame")
    with pytest.raises(ValueError, match="frame"):
        sample_ground(source, GRID, CAMERAS, world_scene=WORLD, transform=None)
    source = _ground(tmp_path, world={**WORLD, "scene_frame_digest": "old"})
    with pytest.raises(ValueError, match="scene_frame_digest"):
        sample_ground(
            source,
            GRID,
            CAMERAS,
            world_scene={**WORLD, "scene_frame_digest": "new"},
            transform=None,
        )
    _, meta = sample_ground(
        source,
        GRID,
        CAMERAS,
        world_scene={**WORLD, "scene_frame_digest": "old"},
        transform=None,
    )
    assert meta["placement_verification"] == "matched"


def test_rewritten_observations_are_refused(tmp_path):
    source = _ground(tmp_path)
    np.savez(source.observations, arm=np.array(["left"]), z=np.array([0.0]))
    with pytest.raises(ValueError, match="mismatch"):
        sample_ground(source, GRID, CAMERAS, world_scene=WORLD, transform=None)


def _episode(source, grid, cameras=CAMERAS):
    features, meta = sample_ground(
        source, grid, cameras, world_scene=WORLD, transform=None
    )
    state = np.zeros((len(grid), 16), dtype=np.float32)
    state[:, [6, 14]] = 1
    features[STATE_KEY] = FeatureColumn(STATE_KEY, "float32", (16,), None, state)
    return ObservationEpisode(
        2.0, grid - grid[0], "pick", WORLD["frame"], features, {"ground": meta}
    )


@pytest.mark.parametrize("format", ["v2.1", "v3.0"])
@pytest.mark.parametrize("missing_first", [False, True])
def test_ground_dtype_nan_and_statistics_read_back(tmp_path, format, missing_first):
    populated = _episode(_ground(tmp_path), GRID)
    missing = _episode(None, GRID)
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
    assert table.schema.field(GROUND_PLANE_KEY).type == pa.list_(pa.float32())
    height_key = camera_height_key("left")
    # a shape-(1,) feature: scalar column, as lerobot reads it back
    assert table.schema.field(height_key).type == pa.float32()
    planes = np.array(table[GROUND_PLANE_KEY].to_pylist())
    assert planes.shape == (6, 4) and np.isnan(planes).all(axis=1).sum() == 3
    heights = np.array(table[height_key].to_pylist())
    np.testing.assert_allclose(
        np.sort(heights[np.isfinite(heights)]), CAMERAS["left"][:, 2] - FLOOR_Z
    )
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"][STATE_KEY]["shape"] == [16]
    assert info["features"][GROUND_PLANE_KEY]["shape"] == [4]
    assert info["features"][GROUND_PLANE_KEY]["names"] == ["nx", "ny", "nz", "d"]
    assert info["features"][camera_height_key("right")]["names"] == [
        "right_camera_height_m"
    ]
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats[GROUND_PLANE_KEY]["count"] == [3]
    assert stats[GROUND_PLANE_KEY]["mean"][3] == pytest.approx(-FLOOR_Z)
    assert stats[height_key]["count"] == [3]
    assert stats[height_key]["mean"][0] == pytest.approx(0.4 - FLOOR_Z)
    assert "NaN" not in (root / "meta/stats.json").read_text()
    if format == "v3.0":
        episode_table = pq.read_table(next((root / "meta/episodes").rglob("*.parquet")))
        counts = episode_table[f"stats/{GROUND_PLANE_KEY}/count"].to_pylist()
        assert counts == ([None, [3]] if missing_first else [[3], None])
    else:
        records = [
            json.loads(line)
            for line in (root / "meta/episodes_stats.jsonl").read_text().splitlines()
        ]
        absent = records[0 if missing_first else 1]["stats"]
        assert GROUND_PLANE_KEY not in absent and height_key not in absent


@pytest.mark.parametrize("format", ["v2.1", "v3.0"])
def test_all_missing_ground_stats_are_not_fabricated(tmp_path, format):
    episode = _episode(None, GRID)
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
    assert GROUND_PLANE_KEY not in stats and camera_height_key("left") not in stats
    assert stats[STATE_KEY]["count"] == [3]
    assert episode.meta["ground"]["statistics_status"] == "not_computed"


def test_replay_and_umi_refuse_ground_instead_of_dropping(tmp_path):
    episode = _episode(None, GRID)
    with pytest.raises(ValueError, match="ground plane"):
        write_replay_state(episode, tmp_path / "replay.npz")
    from kiwi.export.writers.umi_zarr import UmiZarrWriter

    with pytest.raises(ValueError, match="ground plane"):
        UmiZarrWriter().open(
            tmp_path,
            DatasetInfo("local/test", "portable_bimanual", 2.0, episode.features),
        )


def test_archive_copies_the_report_and_its_observations(tmp_path):
    episode = tmp_path / "episode"
    (episode / "left" / "scene").mkdir(parents=True)
    state_tum = episode / "left" / "scene" / "state.tum"
    state_tum.write_text("0 0 0 0 0 0 0 1\n")
    source = _ground(episode)
    report = json.loads(source.report.read_text())
    report["inputs"] = {
        "left": {
            "state": {
                "path": "../left/scene/state.tum",
                "digest": digest_bytes(state_tum),
            }
        }
    }
    source.report.write_text(json.dumps(report))
    archive = SourceArchive(tmp_path / "dataset")
    inputs = archive.episode(
        SimpleNamespace(
            episode_dir=episode, arms=(), cameras=(), head=None, ground=source
        ),
        SimpleNamespace(
            meta={"ground": {"status": "computed"}, "world_source": "markerless"}
        ),
        3,
    )
    assert inputs["ground.report"]["digest"].startswith("canonical_json_sha256:")
    records = {entry["original_path"]: entry for entry in archive.records}
    copied = tmp_path / "dataset" / records[str(source.report)]["archived_path"]
    assert copied.name == "ground_report.json"
    assert copied.parent.parent == tmp_path / "dataset" / "meta" / "source_records"
    assert json.loads(copied.read_text())["floor_z_m"] == FLOOR_Z
    quality = {
        Path(entry["archived_path"]).name: entry["archived_path"]
        for entry in archive.quality
        if entry.get("archived_path")
    }
    assert (
        quality["ground_observations.npz"]
        == "meta/quality/000003/ground/ground_observations.npz"
    )
    assert (tmp_path / "dataset" / quality["ground_observations.npz"]).is_file()
    # the report's own report-relative input links resolve too
    assert quality["state.tum"] == "meta/quality/000003/left/scene/state.tum"
    assert all(
        entry.get("status") != "missing" for entry in archive.records + archive.quality
    )
