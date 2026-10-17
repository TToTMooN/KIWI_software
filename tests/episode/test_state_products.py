"""The published state-estimation products, read back by the code that consumes them."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import kiwi.episode.head as head_module
import kiwi.state.head as detection_module
from kiwi.core.digest import digest_bytes
from kiwi.core.paths import repo_root
from kiwi.core.trajectory import Trajectory, read_tum_header, write_tum
from kiwi.episode.head import (
    HeadArmInput,
    _arm_marker_poses,
    _merge,
    _source_records,
    publish_head_marker,
)
from kiwi.episode.scene import init_scene
from kiwi.episode.scene_build import Unit, load_imu
from kiwi.episode.stages import run_stage
from kiwi.export.archive import SourceArchive
from kiwi.export.build import read_world_tum_header
from kiwi.export.head import sample_head
from kiwi.export.source import HeadSource
from kiwi.state import read_state_latent
from kiwi.state.geometry import se3_inv

SCENE_FRAME = "shared_rear_metric_gravity_aligned"
WORLD = {
    "frame": SCENE_FRAME,
    "scene": "room",
    "map_set_digest": "maps",
    "scene_frame_digest": "placement",
}
SYNC_OFFSETS = {"left": -0.1, "right": 0.0}
REAR_TIMESHIFTS = {"left": 0.02, "right": 0.03}


def _poses(times, x0=0.0):
    """Identity-rotation poses whose x coordinate is ``x0 + t``."""
    times = np.asarray(times, dtype=float)
    T = np.tile(np.eye(4), (len(times), 1, 1))
    T[:, 0, 3] = x0 + times
    return times, T


class _SameProcessPool:
    """Runs each submitted call at once, in the calling process."""

    created: list[_SameProcessPool] = []

    def __init__(self, max_workers=None, mp_context=None):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture
def head_episode(tmp_path, monkeypatch):
    """A two-arm episode with fused states and an ego camera carrying the marker.

    Input resolution, the worker pool and the per-arm detection are stand-ins,
    so no video is opened. ``rows[arm]`` is what the arm's worker returns.
    """
    config = tmp_path / "config"
    (config / "gripper" / "chopsticks-v2").mkdir(parents=True)
    (config / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "tool_v1",
                "name": "chopsticks-v2",
                "design": {
                    "estimator": "chopsticks",
                    "marker": {"dictionary": "DICT_4X4_50"},
                },
            }
        )
    )
    for unit in ("unit-a", "unit-b"):
        (config / "cameras" / unit).mkdir(parents=True)
    monkeypatch.setenv("KIWI_CONFIG", str(config))

    root = tmp_path / "episode"
    (root / "raw").mkdir(parents=True)
    for name in ("A.insv", "B.insv", "ego.mp4"):
        (root / "raw" / name).write_bytes(b"media")
    manifest = {
        "schema": "episode_v1",
        "name": "episode",
        "tool": "chopsticks-v2",
        "arms": {
            "left": {"camera": "unit-a", "recording": "raw/A.insv"},
            "right": {"camera": "unit-b", "recording": "raw/B.insv"},
        },
        "cameras": {
            "ego": {
                "recording": "raw/ego.mp4",
                "head_marker": {
                    "dictionary": "DICT_4X4_100",
                    "id": 50,
                    "size": 0.049,
                },
            }
        },
        "sync": {
            "method": "audio_gcc_phat",
            "origin": "right",
            "offsets": {**SYNC_OFFSETS, "ego": -0.2},
        },
    }
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))

    t = np.arange(101) * 0.1  # 0 to 10 s at 10 Hz
    inputs = []
    for label in ("left", "right"):
        scene = root / label / "scene"
        write_tum(
            Trajectory(
                t,
                np.column_stack([t, np.zeros_like(t), np.zeros_like(t)]),
                np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1)),
                SCENE_FRAME,
                "camera",
            ),
            scene / "state.tum",
            clock="episode global clock (origin=right)",
            source="kiwi.state scene fusion",
        )
        (scene / "state.inputs.json").write_text(json.dumps({"world_scene": WORLD}))
        (root / label / "lens1.mp4").write_bytes(b"video")
        calibration = tmp_path / "calibration" / label
        calibration.mkdir(parents=True)
        for name in ("kb4-back.json", "imu-back.yaml", "imu.yaml"):
            (calibration / name).write_text("{}")
        inputs.append(
            HeadArmInput(
                label=label,
                state_tum=scene / "state.tum",
                rear_video=root / label / "lens1.mp4",
                rear_times_ns=np.arange(10, dtype=np.int64),
                rear_intrinsics=calibration / "kb4-back.json",
                rear_extrinsic=calibration / "imu-back.yaml",
                front_extrinsic=calibration / "imu.yaml",
                unit=SimpleNamespace(
                    kb4={"rear": np.arange(8.0)},
                    resolution={"rear": (2880, 2880)},
                    timeshift_cam_imu={"rear": REAR_TIMESHIFTS[label]},
                    T_rear_front=lambda: np.eye(4),
                ),
                sync_offset_s=SYNC_OFFSETS[label],
            )
        )

    rows = {"left": _poses([]), "right": _poses([])}
    clock_offsets = {}

    def fake_arm_marker_poses(
        state_tum,
        rear_video,
        rear_times_ns,
        kb4,
        resolution,
        clock_offset_s,
        marker,
        T_rear_front,
    ):
        label = state_tum.parent.parent.name
        clock_offsets[label] = clock_offset_s
        return rows[label]

    monkeypatch.setattr(_SameProcessPool, "created", [])
    monkeypatch.setattr(head_module, "_resolve_inputs", lambda *_args: inputs)
    monkeypatch.setattr(head_module, "_arm_marker_poses", fake_arm_marker_poses)
    monkeypatch.setattr(head_module, "ProcessPoolExecutor", _SameProcessPool)
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        rows=rows,
        clock_offsets=clock_offsets,
        pools=_SameProcessPool.created,
    )


def test_head_stage_publishes_one_trajectory_that_export_samples(head_episode):
    root = head_episode.root
    head_episode.rows["left"] = _poses(1.0 + 0.1 * np.arange(10))
    head_episode.rows["right"] = _poses(
        np.concatenate([1.05 + 0.1 * np.arange(10), 6.0 + 0.1 * np.arange(6)])
    )

    result = run_stage("head", root)

    assert result.ok, result.summary
    assert "26 observations in 2 segments" in result.summary
    head = root / "head"
    source = HeadSource(
        head / "head_marker.tum",
        head / "head_marker_report.json",
        head / "head_marker_diagnostics.npz",
    )
    assert sorted(path.name for path in head.iterdir()) == sorted(
        path.name for path in (source.trajectory, source.report, source.diagnostics)
    )
    assert result.artifacts == [source.trajectory, source.diagnostics, source.report]

    # Each arm ran in a worker of one two-process spawn pool, on its own clock offset.
    (pool,) = head_episode.pools
    assert pool.max_workers == 2
    assert pool.mp_context.get_start_method() == "spawn"
    assert head_episode.clock_offsets == {
        label: pytest.approx(REAR_TIMESHIFTS[label] + SYNC_OFFSETS[label])
        for label in ("left", "right")
    }

    report = json.loads(source.report.read_text())
    assert report["schema"] == "head_marker_state"
    assert report["status"] == "computed_unvalidated"
    assert report["world_scene"] == WORLD
    assert report["per_arm"] == {
        "left": {"detections": 10, "rows": 10},
        "right": {"detections": 16, "rows": 16},
    }
    assert [segment["rows"] for segment in report["segments"]] == [20, 6]
    assert report["output"]["rows"] == 26
    assert set(report["input_files"]["left"]) == {
        "state",
        "rear_video",
        "rear_intrinsics",
        "rear_extrinsic",
        "front_extrinsic",
    }
    assert report["input_files"]["left"]["state"]["record"]["path"] == (
        "../left/scene/state.inputs.json"
    )

    header = read_tum_header(source.trajectory)
    assert (header.frame, header.body) == (SCENE_FRAME, "head_marker")
    with np.load(source.diagnostics) as diagnostics:
        assert diagnostics["arm"].tolist() == [0, 1] * 10 + [1] * 6
        assert diagnostics["segment_id"].tolist() == [0] * 20 + [1] * 6

    grid = np.arange(300) / 30.0
    _, metadata = sample_head(source, grid, world_scene=WORLD, transform=None)
    assert metadata["status"] == "available"
    assert metadata["valid_frames"] > 0


def test_head_merge_orders_rows_drops_equal_times_and_splits_at_gaps():
    t, T, arm, segment_id = _merge(
        [_poses([0.0, 0.1, 2.5]), _poses([0.05, 0.1, 2.6], x0=10.0)]
    )

    np.testing.assert_array_equal(t, [0.0, 0.05, 0.1, 2.5, 2.6])
    np.testing.assert_array_equal(arm, [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(segment_id, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(T[:, 0, 3], [0.0, 10.05, 0.1, 2.5, 12.6])
    assert (arm.dtype, segment_id.dtype) == (np.int8, np.int64)


def test_head_stage_without_a_detection_publishes_only_the_report(head_episode):
    root = head_episode.root
    stale = root / "head" / "head_marker.tum"
    stale.parent.mkdir()
    stale.write_text("# an earlier run\n")

    result = run_stage("head", root)

    assert not result.ok
    assert result.data == {"status": "insufficient_observations", "reused": False}
    assert "no accepted head-marker detection" in result.summary
    report_path = root / "head" / "head_marker_report.json"
    assert list((root / "head").iterdir()) == [report_path]
    report = json.loads(report_path.read_text())
    assert report["status"] == "insufficient_observations"
    assert report["output"] == {}
    assert report["segments"] == []


def test_a_rear_view_without_a_detection_contributes_no_row(head_episode, monkeypatch):
    """The per-arm worker itself, on what the detection returns for a rear
    video in which the marker is never accepted."""
    root = head_episode.root
    report_path = root / "head" / "head_marker_report.json"
    none = (np.asarray([], dtype=float), np.asarray([]))
    times = 1.0 + 0.1 * np.arange(10)
    detections = {"left": none, "right": _poses(times)}

    def detect(video, t_ns, kb4, resolution, clock_offset_s, marker):
        return detections[video.parent.name]

    monkeypatch.setattr(head_module, "_arm_marker_poses", _arm_marker_poses)
    monkeypatch.setattr(detection_module, "detect_marker_poses", detect)

    result = run_stage("head", root)

    assert result.ok, result.summary
    report = json.loads(report_path.read_text())
    assert report["per_arm"] == {
        "left": {"detections": 0, "rows": 0},
        "right": {"detections": 10, "rows": 10},
    }
    # The front camera is at x = t and the marker at x = t in the rear view;
    # the lever is the identity.
    rows = np.loadtxt(root / "head" / "head_marker.tum")
    np.testing.assert_allclose(rows[:, :2], np.column_stack([times, 2 * times]))

    # Neither wrist sees the marker: the report alone.
    detections["right"] = none
    result = run_stage("head", root)

    assert not result.ok
    assert result.data == {"status": "insufficient_observations", "reused": False}
    assert list((root / "head").iterdir()) == [report_path]


def _head_arm_inputs(tmp_path, records):
    """Two head arm inputs, each with the state record ``records`` names."""
    arms = []
    for arm in ("left", "right"):
        base = tmp_path / arm
        base.mkdir()
        (base / "state.tum").write_text("0 0 0 0 0 0 0 1\n")
        (base / "rear.mp4").write_bytes(b"video")
        (base / "intrinsics.json").write_text("{}")
        (base / "extrinsic.yaml").write_text("{}")
        (base / "extrinsic-front.yaml").write_text("{}")
        if records.get(arm) is not None:
            (base / "state.inputs.json").write_text(json.dumps(records[arm]))
        arms.append(
            HeadArmInput(
                arm,
                base / "state.tum",
                base / "rear.mp4",
                np.array([0, 1]),
                base / "intrinsics.json",
                base / "extrinsic.yaml",
                base / "extrinsic-front.yaml",
                None,
                0.0,
            )
        )
    return arms


def test_head_refuses_arm_states_with_different_scene_identities(tmp_path):
    common = {"frame": "scene", "scene": "room", "map_set_digest": "maps"}
    arms = _head_arm_inputs(
        tmp_path,
        {
            "left": {"world_scene": {**common, "scene_frame_digest": "old-placement"}},
            "right": {"world_scene": common},
        },
    )
    _, world = _source_records(arms, tmp_path / "head")
    assert world == {**common, "scene_frame_digest": None}
    (tmp_path / "right/state.inputs.json").write_text(
        json.dumps({"world_scene": {**common, "scene_frame_digest": "new-placement"}})
    )
    with pytest.raises(ValueError, match="different scene_frame_digest"):
        _source_records(arms, tmp_path / "head")


@pytest.mark.parametrize(
    ("right_record", "refusal"),
    [
        (None, "right: no scene/state.inputs.json"),
        (
            {"status": "computed"},
            "right: scene/state.inputs.json declares no world_scene",
        ),
    ],
    ids=["missing_record", "no_world_scene"],
)
def test_head_refuses_an_arm_state_without_a_scene_identity(
    tmp_path, right_record, refusal
):
    """A state the head stage cannot place in a scene is refused with the stage
    to re-run, not published as an unknown `world_scene`."""
    world = {"frame": "scene", "scene": "room", "map_set_digest": "maps"}
    arms = _head_arm_inputs(
        tmp_path, {"left": {"world_scene": world}, "right": right_record}
    )
    with pytest.raises(ValueError, match=f"{refusal}.* - re-run the state stage"):
        _source_records(arms, tmp_path / "head")


def test_an_unknown_marker_dictionary_is_refused_before_the_workers_start(
    head_episode,
):
    root, manifest = head_episode.root, head_episode.manifest
    manifest["cameras"]["ego"]["head_marker"]["dictionary"] = "DICT_UNKNOWN"
    (root / "episode.yaml").write_text(yaml.safe_dump(manifest))

    with pytest.raises(
        ValueError, match="'DICT_UNKNOWN' is not an OpenCV ArUco dictionary"
    ):
        publish_head_marker(root)

    assert head_episode.pools == []
    assert not (root / "head").exists()


# ----------------------------------------------------------------------------
# The state stage
# ----------------------------------------------------------------------------

STATE_SYNC = {"left": -0.25, "right": 0.0}
LENS_TIMESHIFT = {"extrinsic_imu": 0.013, "extrinsic_imu_back": 0.021}
IMU_PRE_ROLL_S = 0.3
FRAME_NS = 33_366_667
N_FRAMES = 60
FRONT_LOST = np.arange(N_FRAMES) % 5 == 4  # the front frames without a pose
T_FRONT_IMU = np.array(
    [
        [0.0, -1.0, 0.0, 0.02],
        [0.0, 0.0, -1.0, -0.01],
        [1.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture
def state_episode(tmp_path, monkeypatch):
    """A scene with a published map set and one synced episode that names it.

    The localization, the camera unit, the IMU extraction and the solver are
    stand-ins; ``fused`` holds what the solver was given, one entry per arm,
    and ``localized`` the streams that were localized.
    """
    cfg = tmp_path / "config"
    profiles = cfg / "scene" / "mapping-profiles"
    profiles.mkdir(parents=True)
    shutil.copyfile(
        repo_root() / "config" / "scene" / "mapping-profiles" / "default.yaml",
        profiles / "default.yaml",
    )
    (cfg / "gripper" / "chopsticks-v2").mkdir(parents=True)
    (cfg / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "tool_v1",
                "name": "chopsticks-v2",
                "design": {
                    "estimator": "chopsticks",
                    "marker": {"dictionary": "DICT_4X4_50"},
                },
            },
            sort_keys=False,
        )
    )
    monkeypatch.setenv("KIWI_CONFIG", str(cfg))

    root = tmp_path / "workbench"
    ep = root / "session-a"
    (ep / "raw").mkdir(parents=True)
    for name, label in (("A", "left"), ("B", "right")):
        (ep / "raw" / f"{name}.insv").write_bytes(f"media-{name}".encode())
        (ep / label).mkdir()
    doc = {
        "schema": "episode_v1",
        "name": "session-a",
        "tool": "chopsticks-v2",
        "arms": {
            "left": {"camera": "unit-a", "recording": "raw/A.insv"},
            "right": {"camera": "unit-b", "recording": "raw/B.insv"},
        },
        "scene": {"manifest": "../scene.yaml", "episode": "session-a"},
        "sync": {
            "method": "audio_gcc_phat",
            "origin": "right",
            "offsets": STATE_SYNC,
        },
    }
    for unit in ("unit-a", "unit-b"):
        (cfg / "cameras" / unit).mkdir(parents=True)
    (ep / "episode.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    init_scene(root=root, mapping_episode="session-a")
    # a published, complete map set + frozen scene frame
    scene_map = root / "scene_map"
    roles = {}
    for role in ("front_left", "front_right", "shared_rear"):
        d = scene_map / "maps" / f"{role}-0123456789ab"
        d.mkdir(parents=True)
        (d / "atlas.osa").write_bytes(b"ATLAS")
        roles[role] = f"maps/{role}-0123456789ab"
    (scene_map / "current.json").write_text(
        json.dumps(
            {"schema": "scene_map_pointer", "episode": "session-a", "roles": roles}
        )
    )
    (scene_map / "scene_frame.json").write_text(
        json.dumps(
            {
                "schema": "scene_frame",
                "frame": SCENE_FRAME,
                "episode": "session-a",
                "maps": {},
            }
        )
    )

    fused = []
    extractions = []
    localized = []

    def fake_localize(scene, episode_id, arm, *, lens="front"):
        localized.append((arm, lens))
        scene_dir = scene.episode_dir(episode_id) / arm / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        role = "shared_rear" if lens == "back" else f"front_{arm}"
        out = scene_dir / f"{role}_observations.txt"
        rows = [
            f"{i * FRAME_NS} 0 0 0 0 0 0 0 0"
            if lens == "front" and FRONT_LOST[i]
            else f"{i * FRAME_NS} 2 0.1 0.2 0.3 0 0 0 1"
            for i in range(N_FRAMES)
        ]
        out.write_text("\n".join(rows) + "\n")
        out.with_suffix(".inputs.json").write_text(
            json.dumps({"schema": "scene_localization_inputs", "map": roles[role]})
        )
        return out

    class FakeUnit:
        def __init__(self, base):
            self.base = base
            self.meta = {"model": "Insta360 X5"}

        def artifact(self, key):
            if key == "av_sync":
                return None
            p = self.base / f"{key}.yaml"
            if key == "imu_noise":
                p.write_text(
                    yaml.safe_dump(
                        {
                            "schema": "imu_noise_v1",
                            "noise": {
                                "accelerometer_noise_density": 4e-3,
                                "accelerometer_random_walk": 4e-4,
                                "gyroscope_noise_density": 7e-4,
                                "gyroscope_random_walk": 8e-5,
                            },
                        }
                    )
                )
            else:
                p.write_text("chain")
            return p

    def fake_read_cam_imu(path):
        return SimpleNamespace(
            T_cam_imu=T_FRONT_IMU if path.stem == "extrinsic_imu" else np.eye(4),
            timeshift_s=LENS_TIMESHIFT[path.stem],
        )

    def fake_extract(recording, out_csv):
        # extractor clock: shifted forward by the pre-roll (starts at 0)
        extractions.append(Path(recording).name)
        t = np.arange(0.0, 2.8, 0.005)
        samples = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (len(t), 1))
        np.savetxt(
            out_csv,
            np.column_stack([t, samples]),
            header="t gx gy gz ax ay az",
            comments="",
        )
        return SimpleNamespace(clock_offset_s=IMU_PRE_ROLL_S)

    def fake_fuse(
        t_nodes,
        imu_t,
        imu_gyro,
        imu_acc,
        front_obs,
        rear_obs,
        T_front_imu,
        T_rear_imu,
        noise,
    ):
        fused.append(
            SimpleNamespace(
                t_nodes=t_nodes.copy(), imu_t=imu_t, front=front_obs, rear=rear_obs
            )
        )
        return np.tile(np.eye(4), (len(t_nodes), 1, 1))

    import kiwi.core.camera.extrinsics as extr_mod
    import kiwi.core.camera.imu as imu_mod
    import kiwi.core.camera.registry as registry_mod
    import kiwi.episode.scene_build as scene_build_mod
    import kiwi.state.fusion as fusion_mod

    monkeypatch.setattr(scene_build_mod, "localize_episode", fake_localize)
    monkeypatch.setattr(
        registry_mod.CameraUnit, "load", staticmethod(lambda d: FakeUnit(tmp_path))
    )
    monkeypatch.setattr(extr_mod, "read_cam_imu", fake_read_cam_imu)
    monkeypatch.setattr(imu_mod, "extract_imu_to_csv", fake_extract)
    monkeypatch.setattr(fusion_mod, "fuse", fake_fuse)
    return SimpleNamespace(
        root=ep,
        scene_root=root,
        roles=roles,
        fused=fused,
        extractions=extractions,
        localized=localized,
    )


def test_the_state_products_pass_their_consumers(state_episode, tmp_path):
    ep, scene_root = state_episode.root, state_episode.scene_root

    result = run_stage("state", ep)

    assert result.ok, result.summary
    assert "left: 60 native frames" in result.summary
    for label in ("left", "right"):
        scene_dir = ep / label / "scene"
        tum = scene_dir / "state.tum"
        record_path = scene_dir / "state.inputs.json"
        assert result.artifacts.count(tum) == 1

        # export: the trajectory declares the scene frame, the camera body and
        # the episode global clock
        header = read_world_tum_header(tum)
        assert (header.frame, header.body) == (SCENE_FRAME, "camera")
        assert header.clock == "episode global clock"

        record = json.loads(record_path.read_text())
        assert record["schema"] == "state_trajectory_inputs"
        assert record["status"] == "computed"
        assert record["map_set"] == state_episode.roles
        assert record["diagnostics"] == {"result_status": "computed_unvalidated"}
        assert record["params"]["imu_noise_scale"] == 10.0
        assert record["params"]["gravity"] == [0.0, 0.0, -9.81]
        assert len(record["params"]) == 14
        assert record["world_scene"]["scene"] == "workbench"
        assert record["world_scene"]["scene_frame_digest"] == digest_bytes(
            scene_root / "scene_map" / "scene_frame.json"
        )
        assert record["clock_offsets_s"] == {
            "front_timeshift": LENS_TIMESHIFT["extrinsic_imu"],
            "rear_timeshift": LENS_TIMESHIFT["extrinsic_imu_back"],
            "sync_offset": STATE_SYNC[label],
        }
        assert set(record["outputs"]) == {"trajectory", "latent"}
        assert record["outputs"]["trajectory"] == {
            "path": "state.tum",
            "digest": digest_bytes(tum),
        }

        # the export archive reaches the scene manifest, the pinned profile,
        # the IMU samples and both localization records through the record
        archive = SourceArchive(tmp_path / f"dataset-{label}")
        archive.add(record_path, episode_dir=ep, episode_index=0)
        entries = {
            Path(os.path.normpath(Path(e["original_base"]) / e["original_path"])): e
            for e in (*archive.records, *archive.quality)
        }
        (imu_csv,) = scene_dir.glob("*-imu.csv")
        for source in (
            scene_root / "scene.yaml",
            scene_root / "profiles" / "default.yaml",
            imu_csv,
            imu_csv.with_suffix(".json"),
            scene_dir / f"front_{label}_observations.inputs.json",
            scene_dir / "shared_rear_observations.inputs.json",
        ):
            assert entries[source].get("status") is None, source
            assert (archive.root / entries[source]["archived_path"]).is_file()


def test_the_solve_runs_on_the_imu_clock_and_the_stamps_on_the_episode_clock(
    state_episode,
):
    ep = state_episode.root
    assert run_stage("state", ep).ok
    dt_front = LENS_TIMESHIFT["extrinsic_imu"]
    dt_rear = LENS_TIMESHIFT["extrinsic_imu_back"]
    t_ns = np.arange(N_FRAMES) * FRAME_NS

    # arms are solved in sorted order
    for label, solve in zip(("left", "right"), state_episode.fused, strict=True):
        # one node per front frame, tracked or not, on the arm's IMU clock
        assert len(solve.t_nodes) == N_FRAMES
        np.testing.assert_array_equal(
            solve.t_nodes, t_ns.astype(float) * 1e-9 + dt_front
        )
        # node and observation times are equal bit for bit, which is how a
        # tracked front frame meets its node
        t_front, T_front, run_front = solve.front
        assert (solve.t_nodes[~FRONT_LOST] == t_front).all()
        assert T_front.shape == (int((~FRONT_LOST).sum()), 4, 4)
        assert run_front.tolist() == [i // 5 for i in range(N_FRAMES) if i % 5 != 4]
        t_rear, _, run_rear = solve.rear
        np.testing.assert_array_equal(t_rear, t_ns.astype(float) * 1e-9 + dt_rear)
        assert set(run_rear.tolist()) == {0}
        # the IMU times carry the pre-roll correction and no sync offset
        assert solve.imu_t[0] == pytest.approx(-IMU_PRE_ROLL_S)

        # the sync offset is in the written timestamps only
        scene_dir = ep / label / "scene"
        rows = np.loadtxt(scene_dir / "state.tum")
        np.testing.assert_allclose(
            rows[:, 0], solve.t_nodes + STATE_SYNC[label], rtol=0, atol=1e-9
        )
        # the published pose is the front camera: T_scene_imu @ inv(T_front_imu)
        np.testing.assert_allclose(
            rows[:, 1:4], np.tile(se3_inv(T_FRONT_IMU)[:3, 3], (N_FRAMES, 1)), atol=1e-9
        )
        latent = read_state_latent(scene_dir / "state_latent.npz")
        np.testing.assert_array_equal(latent["t"], solve.t_nodes + STATE_SYNC[label])
        np.testing.assert_array_equal(latent["body_position"], np.zeros((N_FRAMES, 3)))
        np.testing.assert_array_equal(latent["front_candidate_present"], ~FRONT_LOST)
        assert latent["rear_candidate_present"].sum() == N_FRAMES - 1


def test_a_rebuilt_map_set_is_localized_and_estimated_again(state_episode):
    ep, roles = state_episode.root, state_episode.roles
    assert run_stage("state", ep).ok
    assert run_stage("state", ep).summary.startswith("0 estimated, 2 reused")
    assert len(state_episode.localized) == 4 and len(state_episode.fused) == 2

    # a rebuild gives every map directory another name
    scene_map = state_episode.scene_root / "scene_map"
    for role in list(roles):
        rebuilt = scene_map / "maps" / f"{role}-ba9876543210"
        rebuilt.mkdir()
        (rebuilt / "atlas.osa").write_bytes(b"ATLAS")
        roles[role] = f"maps/{rebuilt.name}"
    (scene_map / "current.json").write_text(
        json.dumps(
            {"schema": "scene_map_pointer", "episode": "session-a", "roles": roles}
        )
    )

    result = run_stage("state", ep)

    assert result.ok and result.summary.startswith("2 estimated, 0 reused")
    assert sorted(state_episode.localized[4:]) == [
        ("left", "back"),
        ("left", "front"),
        ("right", "back"),
        ("right", "front"),
    ]
    assert len(state_episode.fused) == 4
    for label in ("left", "right"):
        record = json.loads((ep / label / "scene" / "state.inputs.json").read_text())
        assert record["map_set"] == roles
    assert run_stage("state", ep).summary.startswith("0 estimated, 2 reused")


def test_load_imu_returns_the_samples_on_the_recording_clock(state_episode, tmp_path):
    recording = state_episode.root / "raw" / "A.insv"
    cache = tmp_path / "cache"

    t, gyro, acc = load_imu(recording, cache)

    # the extractor's file starts at zero; its recorded offset is subtracted
    assert t[0] == pytest.approx(-IMU_PRE_ROLL_S)
    assert t[1] - t[0] == pytest.approx(0.005)
    np.testing.assert_array_equal(gyro[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(acc[0], [4.0, 5.0, 6.0])
    stem = digest_bytes(recording).split(":")[1][:16]
    assert sorted(path.name for path in cache.iterdir()) == [
        f"{stem}-imu.csv",
        f"{stem}-imu.json",
    ]
    record = json.loads((cache / f"{stem}-imu.json").read_text())
    assert record["clock_offset_s"] == IMU_PRE_ROLL_S
    assert record["outputs"]["imu"]["path"] == f"{stem}-imu.csv"

    # the recording is extracted once
    np.testing.assert_array_equal(load_imu(recording, cache)[0], t)
    assert state_episode.extractions == ["A.insv"]


def test_the_lever_takes_front_camera_coordinates_to_the_rear_camera():
    T_front, T_rear = T_FRONT_IMU, np.eye(4)
    T_rear[:3, :3] = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    T_rear[:3, 3] = [-0.15, 0.05, -0.08]
    unit = Unit(
        kb4={},
        resolution={},
        T_cam_imu={"front": T_front, "rear": T_rear},
        timeshift_cam_imu={},
    )
    np.testing.assert_allclose(
        unit.T_rear_front(), T_rear @ np.linalg.inv(T_front), atol=1e-15
    )
    assert not np.allclose(unit.T_rear_front(), T_front @ np.linalg.inv(T_rear))
