"""Scene map building and localization: what is run, checked and published.

The engine, the frame export and the IMU extraction are stand-ins that follow
one synthetic motion, so everything between them runs for real: input
resolution, the calibration files, the map builds on the 30 fps subset, the
placement solve, the published observations with their records, and the
pointer.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from kiwi.core.digest import digest_bytes
from kiwi.core.paths import repo_root
from kiwi.episode import scene_build
from kiwi.episode.scene import init_scene, load_scene
from kiwi.episode.scene_build import (
    BuildOutcome,
    Unit,
    build_role,
    build_scene_maps,
    current_maps,
    frames_dir,
    is_localized,
    localize_episode,
)
from kiwi.track.frames import write_timestamps
from kiwi.track.orbslam.observations import read_observations

EPISODE = "session-a"
FPS = 30.0
GRAVITY = np.array([0.0, 0.0, -9.81])
IMU_HZ = 200.0
IMU_PRE_ROLL_S = 0.3
STREAMS = [(arm, lens) for arm in ("left", "right") for lens in ("front", "back")]
ROLES = ("front_left", "front_right", "shared_rear")
# Metres per map unit of each map, and the rotation from the world to the map.
MAP_SCALE = {"shared_rear": 0.5, "front_left": 2.0, "front_right": 3.0}
MAP_ROTATION = {
    "shared_rear": Rotation.from_euler("xyz", [0.4, -0.7, 1.1]),
    "front_left": Rotation.from_euler("xyz", [-0.2, 0.5, 0.3]),
    "front_right": Rotation.from_euler("xyz", [0.9, 0.1, -0.6]),
}
# p_cam = T_cam_imu @ p_imu and t_imu = t_cam + timeshift, per lens as the
# calibration files name it.
T_CAM_IMU = {}
for _lens, _euler, _t in (
    ("front", [0.3, -1.2, 2.0], [0.02, -0.01, 0.03]),
    ("back", [-2.6, 0.2, 0.4], [-0.15, 0.05, -0.08]),
):
    T_CAM_IMU[_lens] = np.eye(4)
    T_CAM_IMU[_lens][:3, :3] = Rotation.from_euler("xyz", _euler).as_matrix()
    T_CAM_IMU[_lens][:3, 3] = _t
TIMESHIFT = {"front": 0.01, "back": 0.04}
KB4 = {
    "front": [1000.0, 1001.0, 1440.0, 1441.0, 0.01, 0.02, 0.03, 0.04],
    "back": [990.0, 991.0, 1430.0, 1431.0, 0.05, 0.06, 0.07, 0.08],
}
RESOLUTION = [2880, 2880]


def _body_pose(t, arm):
    """A smooth IMU-body motion in a gravity-aligned world, one phase per arm."""
    t = np.asarray(t, dtype=float) + (0.0 if arm == "left" else 2.5)
    R = Rotation.from_rotvec(
        np.column_stack([0.4 * np.sin(0.5 * t), 0.03 * t, 0.5 * np.cos(0.3 * t)])
    )
    p = np.column_stack(
        [0.8 * np.sin(0.7 * t), 0.5 * np.cos(0.9 * t), 0.3 * np.sin(1.3 * t + 0.4)]
    )
    a = np.column_stack(
        [
            -0.392 * np.sin(0.7 * t),
            -0.405 * np.cos(0.9 * t),
            -0.507 * np.sin(1.3 * t + 0.4),
        ]
    )
    return R, p, a


def _camera_in_world(t_cam, arm, lens):
    """World pose (N, 4, 4) of one lens at its own frame times."""
    R, p, _ = _body_pose(np.asarray(t_cam) + TIMESHIFT[lens], arm)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = R.as_matrix()
    T[:, :3, 3] = p
    return T @ np.linalg.inv(T_CAM_IMU[lens])


def _role(arm, lens):
    return "shared_rear" if lens in ("back", "rear") else f"front_{arm}"


class _SameProcessPool:
    """Runs every call at once, in the calling process."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

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

    def map(self, fn, *iterables):
        return [fn(*args) for args in zip(*iterables)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A scene of two episodes with calibrated units, and the stand-ins.

    ``calls`` records what the stand-ins were asked for; ``failing`` names the
    engine sessions (``(mode, role)``) that end one row short, with an empty
    row file, with none, or without a tracked row.
    """
    config = tmp_path / "config"
    profiles = config / "scene" / "mapping-profiles"
    profiles.mkdir(parents=True)
    shutil.copyfile(
        repo_root() / "config" / "scene" / "mapping-profiles" / "default.yaml",
        profiles / "default.yaml",
    )
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
            },
            sort_keys=False,
        )
    )
    for unit in ("unit-a", "unit-b"):
        unit_dir = config / "cameras" / unit
        unit_dir.mkdir(parents=True)
        for lens, suffix in (("front", ""), ("back", "-back")):
            (unit_dir / f"intrinsics-kb4{suffix}.json").write_text(
                json.dumps(
                    {
                        "schema": "kb4_intrinsics",
                        "unit": unit,
                        "lens": lens,
                        "kb4": dict(
                            zip(
                                ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
                                KB4[lens],
                            )
                        ),
                        "resolution": RESOLUTION,
                    }
                )
            )
            (unit_dir / f"extrinsic-imu{suffix}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "cam0": {
                            "T_cam_imu": T_CAM_IMU[lens].tolist(),
                            "timeshift_cam_imu": TIMESHIFT[lens],
                        }
                    }
                )
            )
    monkeypatch.setenv("KIWI_CONFIG", str(config))

    root = tmp_path / "workbench"
    for name in (EPISODE, "session-b"):
        ep = root / name
        (ep / "raw").mkdir(parents=True)
        for arm, recording in (("left", "A"), ("right", "B")):
            (ep / "raw" / f"{recording}.insv").write_bytes(f"{name}-{arm}".encode())
            (ep / arm).mkdir()
            for video in ("lens0.mp4", "lens1.mp4"):
                (ep / arm / video).write_bytes(f"{name}-{arm}-{video}".encode())
        (ep / "episode.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "episode_v1",
                    "name": name,
                    "tool": "chopsticks-v2",
                    "arms": {
                        "left": {"camera": "unit-a", "recording": "raw/A.insv"},
                        "right": {"camera": "unit-b", "recording": "raw/B.insv"},
                    },
                    "scene": {"manifest": "../scene.yaml", "episode": name},
                },
                sort_keys=False,
            )
        )

    calls = SimpleNamespace(
        exports=[], engine=[], masks=[], duration_s=61.0, failing={}
    )

    def frame_times_ns(video):
        n = int(round(calls.duration_s * FPS)) + 1
        return [int(round(i * 1e9 / FPS)) for i in range(n)]

    def export_stream_frames(video, times_ns, out_dir):
        calls.exports.append((Path(video), Path(out_dir)))
        if Path(out_dir).exists():
            shutil.rmtree(out_dir)
        return write_timestamps(Path(out_dir) / "timestamps.txt", times_ns)

    def run_engine(
        mode,
        settings,
        sequence_dir,
        timestamps,
        pose_out,
        *,
        log_path,
        atlas_in=None,
        atlas_out=None,
        mask_dir=None,
    ):
        """One tracked row per listed frame: the lens pose in its role's map."""
        _, arm, lens = Path(sequence_dir).name.rsplit("-", 2)
        role = _role(arm, lens)
        calls.engine.append(
            SimpleNamespace(
                mode=mode,
                role=role,
                arm=arm,
                lens=lens,
                settings=Path(settings).read_text(),
                atlas_in=atlas_in,
                atlas_out=atlas_out,
                mask_dir=mask_dir,
            )
        )
        t_ns = np.loadtxt(timestamps, dtype=np.int64, ndmin=1)
        T = _camera_in_world(t_ns * 1e-9, arm, "back" if lens == "rear" else lens)
        q = (MAP_ROTATION[role] * Rotation.from_matrix(T[:, :3, :3])).as_quat()
        p = MAP_ROTATION[role].apply(T[:, :3, 3]) / MAP_SCALE[role]
        rows = [
            f"{t} 2 " + " ".join(f"{v:.9f}" for v in (*p[i], *q[i])) + "\n"
            for i, t in enumerate(t_ns)
        ]
        failure = calls.failing.get((mode, role))
        rows = {
            "one row short": rows[:-1],
            "no row": [],
            "untracked": [f"{t} 0 0 0 0 0 0 0 0\n" for t in t_ns],
        }.get(failure, rows)
        Path(pose_out).parent.mkdir(parents=True, exist_ok=True)
        if failure != "no row file":
            Path(pose_out).write_text("".join(rows))
        Path(log_path).write_text(f"{mode} {role}\n")
        if atlas_out is not None:
            Path(atlas_out).write_bytes(f"atlas of {role}".encode())

    def extract_imu_to_csv(recording, out_csv):
        """The arm's raw IMU on the extractor's clock, which starts at zero."""
        arm = "left" if Path(recording).name == "A.insv" else "right"
        t = np.arange(-IMU_PRE_ROLL_S, calls.duration_s + 0.5, 1.0 / IMU_HZ)
        R, _, a = _body_pose(t, arm)
        gyro = (R[:-1].inv() * R[1:]).as_rotvec() * IMU_HZ
        gyro = np.vstack([gyro, gyro[-1:]])
        acc = R.inv().apply(a - GRAVITY)
        np.savetxt(
            out_csv,
            np.column_stack([t + IMU_PRE_ROLL_S, gyro, acc]),
            header="t gx gy gz ax ay az",
            comments="",
        )
        return SimpleNamespace(clock_offset_s=IMU_PRE_ROLL_S)

    def write_front_masks(masker, frames, subset, mask_dir, kb4):
        calls.masks.append((Path(frames), len(subset), Path(mask_dir)))
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
        return mask_dir

    def stub_model():
        return "model"

    engine_file = tmp_path / "engine"
    engine_file.write_text("")
    import kiwi.core.camera.imu as imu_module

    monkeypatch.setattr(scene_build, "ProcessPoolExecutor", _SameProcessPool)
    monkeypatch.setattr(scene_build, "frame_times_ns", frame_times_ns)
    monkeypatch.setattr(scene_build, "export_stream_frames", export_stream_frames)
    monkeypatch.setattr(scene_build, "run_engine", run_engine)
    monkeypatch.setattr(scene_build, "write_front_masks", write_front_masks)
    monkeypatch.setattr(scene_build, "engine_binary", lambda: engine_file)
    monkeypatch.setattr(scene_build, "vocabulary", lambda: engine_file)
    monkeypatch.setattr(scene_build, "Masker", stub_model)
    monkeypatch.setattr(imu_module, "extract_imu_to_csv", extract_imu_to_csv)

    def scene(**init):
        options = dict(
            root=root,
            mapping_episode=EPISODE,
        )
        options.update(init)
        init_scene(**options)
        return load_scene(root / "scene.yaml")

    return SimpleNamespace(root=root, calls=calls, scene=scene)


def _observation_files(root, episode=EPISODE):
    return sorted((root / episode).glob("*/scene/*_observations.txt"))


# ----------------------------------------------------------------------------
# Calibration and one map build on resolved paths
# ----------------------------------------------------------------------------


def test_load_unit_reads_both_lenses_under_the_names_front_and_rear(world):
    from kiwi.episode.manifest import EpisodeConfig

    cfg = EpisodeConfig.load(world.root / EPISODE / "episode.yaml")
    unit = scene_build.load_unit(cfg.arms["left"])
    for klens, lens in (("front", "front"), ("rear", "back")):
        np.testing.assert_array_equal(unit.kb4[klens], KB4[lens])
        assert unit.resolution[klens] == tuple(RESOLUTION)
        np.testing.assert_array_equal(unit.T_cam_imu[klens], T_CAM_IMU[lens])
        assert unit.timeshift_cam_imu[klens] == TIMESHIFT[lens]

    (cfg.arms["left"].camera_dir / "extrinsic-imu-back.yaml").unlink()
    with pytest.raises(ValueError, match="'unit-a' publishes no extrinsic_imu_back"):
        scene_build.load_unit(cfg.arms["left"])


def test_build_role_lists_every_fourth_frame(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    t_ns = (np.arange(40) * (1e9 / 119.88)).astype(np.int64)
    write_timestamps(frames / "timestamps.txt", t_ns)
    unit = Unit(
        kb4={"rear": np.array([500.0, 500.0, 960.0, 960.0, 0.0, 0.0, 0.0, 0.0])},
        resolution={"rear": (1920, 1920)},
        T_cam_imu={},
        timeshift_cam_imu={},
    )
    sessions = []

    def run(mode, settings, sequence_dir, timestamps, pose_out, *, log_path, **kwargs):
        sessions.append((mode, sequence_dir, kwargs))
        t_ns = np.loadtxt(timestamps, dtype=np.int64)
        Path(pose_out).write_text("".join(f"{t} 2 0.1 0.2 0.3 0 0 0 1\n" for t in t_ns))
        Path(log_path).write_text("")

    monkeypatch.setattr(scene_build, "run_engine", run)
    map_dir = tmp_path / "maps" / "shared_rear"
    build_role(frames, unit, "rear", map_dir, None)

    assert len(np.loadtxt(map_dir / "timestamps.txt")) == 10
    settings = (map_dir / "settings.yaml").read_text()
    assert "Camera.fps: 30" in settings
    assert "Camera1.fx: 250.0" in settings and "Camera.width: 960" in settings
    assert sessions == [
        ("map", frames, {"atlas_out": map_dir / "atlas.osa", "mask_dir": None})
    ]


# ----------------------------------------------------------------------------
# kiwi scene build
# ----------------------------------------------------------------------------


def test_a_build_publishes_the_maps_the_scene_frame_and_the_observations(world):
    scene = world.scene()
    calls = world.calls
    scene_map = scene.root / "scene_map"

    outcome = build_scene_maps(scene)

    assert outcome.accepted and outcome.episode_id == EPISODE
    assert outcome.attempts == 1
    assert outcome.reason == f"map set built from episode {EPISODE!r}"

    # The four streams are exported once, and their frames are removed again.
    assert sorted(calls.exports) == sorted(
        (
            scene.root
            / EPISODE
            / arm
            / ("lens1.mp4" if lens == "back" else "lens0.mp4"),
            frames_dir(scene_map, EPISODE, arm, "rear" if lens == "back" else lens),
        )
        for arm, lens in STREAMS
    )
    assert not (scene_map / "frames").exists()

    # Three map builds on the 30 fps subset, then the four full streams, each
    # against its role's atlas, with masks for the two front builds.
    sessions = [(c.mode, c.role, c.arm, c.lens) for c in calls.engine]
    assert sorted(sessions[:3]) == [
        ("map", "front_left", "left", "front"),
        ("map", "front_right", "right", "front"),
        ("map", "shared_rear", "left", "rear"),
    ]
    assert sessions[3:] == [
        ("localize", "front_left", "left", "front"),
        ("localize", "shared_rear", "left", "rear"),
        ("localize", "front_right", "right", "front"),
        ("localize", "shared_rear", "right", "rear"),
    ]
    mask_dirs = {
        arm: scene_map / "masks" / f"{EPISODE}-{arm}-front" for arm in ("left", "right")
    }
    assert [(frames, mask_dir) for frames, _, mask_dir in calls.masks] == [
        (frames_dir(scene_map, EPISODE, arm, "front"), mask_dirs[arm])
        for arm in ("left", "right")
    ]
    assert {(c.mode, c.role): c.mask_dir for c in calls.engine[:3]} == {
        ("map", "shared_rear"): None,
        ("map", "front_left"): mask_dirs["left"],
        ("map", "front_right"): mask_dirs["right"],
    }
    assert all(c.mask_dir is None for c in calls.engine[3:])
    assert all("Camera.fps: 30" in c.settings for c in calls.engine)

    # Each map directory carries the first twelve hex digits of its atlas digest.
    assert set(outcome.maps) == set(ROLES)
    for role, rel in outcome.maps.items():
        atlas = scene_map / rel / "atlas.osa"
        assert rel == f"maps/{role}-{digest_bytes(atlas).split(':')[1][:12]}"
        assert sorted(path.name for path in atlas.parent.iterdir()) == [
            "atlas.osa",
            "engine.log",
            "observations.txt",
            "settings.yaml",
            "timestamps.txt",
        ]
    assert sorted(path.name for path in (scene_map / "maps").iterdir()) == sorted(
        rel.split("/")[1] for rel in outcome.maps.values()
    )
    for call in calls.engine[3:]:
        assert call.atlas_in == scene_map / outcome.maps[call.role] / "atlas.osa"

    # The pointer names the set; it is what makes the set published.
    pointer = json.loads((scene_map / "current.json").read_text())
    assert pointer == {
        "schema": "scene_map_pointer",
        "episode": EPISODE,
        "roles": outcome.maps,
    }
    assert current_maps(scene) == pointer

    # The scene frame is the shared rear map made metric: each map's scale is
    # recovered from its own rows and the arm's IMU.
    frame = json.loads((scene_map / "scene_frame.json").read_text())
    assert frame["schema"] == "scene_frame"
    assert frame["frame"] == "shared_rear_metric_gravity_aligned"
    assert frame["episode"] == EPISODE and set(frame["maps"]) == set(ROLES)
    assert frame["maps"]["shared_rear"]["translation"] == [0.0, 0.0, 0.0]
    for role in ROLES:
        assert frame["maps"][role]["scale"] == pytest.approx(MAP_SCALE[role], rel=0.005)

    # Four observation files in the scene frame, one row per frame of the
    # stream. The scene frame is the world turned about the vertical, so every
    # lens is at its own height and distance from the origin.
    assert len(_observation_files(scene.root)) == 4
    n_frames = int(round(61.0 * FPS)) + 1
    for arm, lens in STREAMS:
        role = _role(arm, lens)
        scene_dir = scene.root / EPISODE / arm / "scene"
        obs = read_observations(scene_dir / f"{role}_observations.txt")
        assert len(obs) == n_frames and obs.tracked.all()
        truth = _camera_in_world(obs.t_s, arm, lens)[:, :3, 3]
        np.testing.assert_allclose(obs.T[:, 2, 3], truth[:, 2], atol=0.005)
        np.testing.assert_allclose(
            np.linalg.norm(obs.T[:, :3, 3], axis=1),
            np.linalg.norm(truth, axis=1),
            atol=0.005,
        )

        record = json.loads(
            (scene_dir / f"{role}_observations.inputs.json").read_text()
        )
        assert record["schema"] == "scene_localization_inputs"
        assert record["map"] == outcome.maps[role]
        assert record["frame"] == "shared_rear_metric_gravity_aligned"
        assert (record["arm"], record["lens"], record["mask"]) == (arm, lens, None)
        assert (record["rows"], record["tracked_rows"]) == (n_frames, n_frames)
        assert record["published_by"] == "scene build"
        assert record["atlas_digest"] == digest_bytes(
            scene_map / outcome.maps[role] / "atlas.osa"
        )
        assert record["body"] == (
            "rear_camera_optical_centre"
            if lens == "back"
            else "front_camera_optical_centre"
        )
        assert set(record["inputs"]) == {
            "video",
            "raw_recording",
            "kb4",
            "atlas",
            "scene_frame",
        }
        # The log of a stream localized by the build stays under scene_map/.
        klens = "rear" if lens == "back" else "front"
        assert record["outputs"]["engine_log"]["path"] == (
            f"../../../scene_map/reloc/{EPISODE}-{arm}-{klens}.engine.log"
        )
        assert not (scene_dir / f"{role}.engine.log").exists()
        scale = 960 / RESOLUTION[0]
        assert record["recipe"] == {
            "image_size": [960, 960],
            "kb4_parameters": [v * scale for v in KB4[lens][:4]] + KB4[lens][4:],
            "n_features": 3500,
            "engine_threads": 10,
        }
        assert is_localized(scene, pointer, EPISODE, arm, lens)

    # The IMU of each arm is extracted once into the cache, with its record.
    cache = sorted(path.name for path in (scene_map / "cache").iterdir())
    assert len(cache) == 4 and all(
        name.endswith(("-imu.csv", "-imu.json")) for name in cache
    )

    # A published map set is reused: nothing runs, nothing is rewritten.
    stamps = {
        path: path.stat().st_mtime_ns for path in scene_map.rglob("*") if path.is_file()
    }
    again = build_scene_maps(scene)
    assert again == BuildOutcome(
        True,
        EPISODE,
        outcome.maps,
        0,
        "current map set already published and complete - reusing",
    )
    assert len(calls.engine) == 7 and len(calls.exports) == 4
    assert stamps == {
        path: path.stat().st_mtime_ns for path in scene_map.rglob("*") if path.is_file()
    }


def test_the_model_masks_the_two_front_builds(world, monkeypatch):
    scene = world.scene()
    calls = world.calls
    scene_map = scene.root / "scene_map"
    opened = []
    monkeypatch.setattr(
        scene_build, "Masker", lambda: opened.append("model") or "model"
    )

    build_scene_maps(scene)

    assert opened == ["model"]
    n_subset = int(round(61.0 * FPS)) + 1
    mask_dirs = {
        arm: scene_map / "masks" / f"{EPISODE}-{arm}-front" for arm in ("left", "right")
    }
    assert calls.masks == [
        (frames_dir(scene_map, EPISODE, arm, "front"), n_subset, mask_dirs[arm])
        for arm in ("left", "right")
    ]
    assert {(c.mode, c.role): c.mask_dir for c in calls.engine[:3]} == {
        ("map", "shared_rear"): None,
        ("map", "front_left"): mask_dirs["left"],
        ("map", "front_right"): mask_dirs["right"],
    }
    assert all(c.mask_dir is None for c in calls.engine[3:])


def test_a_profile_with_another_recipe_is_refused(tmp_path):
    for map_fps, map_image_size in ((20.0, 960), (30.0, 1920)):
        scene = SimpleNamespace(
            path=tmp_path / "scene.yaml",
            profile=SimpleNamespace(map_fps=map_fps, map_image_size=map_image_size),
        )
        with pytest.raises(
            ValueError,
            match=f"states {map_fps:g} fps at {map_image_size} px; "
            "kiwi scene build maps at 30 fps and 960 px",
        ):
            scene_build._require_recipe(scene)
    scene_build._require_recipe(
        SimpleNamespace(profile=SimpleNamespace(map_fps=30.0, map_image_size=960))
    )


def test_a_mapping_stream_shorter_than_the_floor_is_refused(world):
    scene = world.scene()
    world.calls.duration_s = 30.0
    with pytest.raises(
        ValueError,
        match=r"the left rear stream lasts 30\.0 s; map building needs 60 s "
        r"\(min_usable_duration_s\)",
    ):
        build_scene_maps(scene)
    assert world.calls.exports == [] and not (scene.root / "scene_map").exists()


def test_a_missing_engine_is_refused_with_the_build_commands(world, monkeypatch):
    scene = world.scene()
    missing = scene.root / "kiwi-orbslam-session"
    monkeypatch.setattr(scene_build, "engine_binary", lambda: missing)
    with pytest.raises(RuntimeError, match="is missing - build it first") as refusal:
        build_scene_maps(scene)
    assert "pixi run -e slam-offline build-orbslam3" in str(refusal.value)
    assert world.calls.exports == []


@pytest.mark.parametrize(
    ("session", "failure", "rows"),
    [
        (("map", "front_right"), "one row short", 1830),
        (("map", "shared_rear"), "no row file", 0),
        (("localize", "front_left"), "one row short", 1830),
        (("localize", "shared_rear"), "no row", 0),
        (("localize", "front_right"), "no row file", 0),
    ],
)
@pytest.mark.filterwarnings("ignore:loadtxt")
def test_a_session_with_fewer_rows_than_frames_publishes_nothing(
    world, session, failure, rows
):
    scene = world.scene()
    world.calls.failing[session] = failure
    with pytest.raises(
        RuntimeError,
        match=rf"the engine wrote {rows} rows for 1831 frames - see .*engine\.log",
    ):
        build_scene_maps(scene)
    scene_map = scene.root / "scene_map"
    assert current_maps(scene) is None
    assert not (scene_map / "scene_frame.json").exists()
    assert not (scene_map / "frames").exists()
    assert _observation_files(scene.root) == []


@pytest.mark.parametrize(
    ("role", "stream"),
    [("front_right", "right front"), ("shared_rear", "left rear")],
)
def test_a_localization_without_a_tracked_row_is_refused(world, role, stream):
    scene = world.scene()
    world.calls.failing[("localize", role)] = "untracked"
    log = f"{EPISODE}-{stream.replace(' ', '-')}.engine.log"
    with pytest.raises(
        RuntimeError,
        match=rf"the localization of the {stream} stream has no tracked row - "
        rf"see .*scene_map/reloc/{log}",
    ):
        build_scene_maps(scene)
    scene_map = scene.root / "scene_map"
    assert current_maps(scene) is None
    assert not (scene_map / "scene_frame.json").exists()
    assert not (scene_map / "frames").exists()
    assert _observation_files(scene.root) == []


# ----------------------------------------------------------------------------
# kiwi scene localize
# ----------------------------------------------------------------------------


def test_localization_is_bound_to_the_published_map_set(world):
    scene = world.scene()
    calls = world.calls
    with pytest.raises(ValueError, match="has no published map set"):
        localize_episode(scene, "session-b", "left")
    build_scene_maps(scene)
    published = current_maps(scene)
    scene_map = scene.root / "scene_map"
    del calls.engine[:], calls.exports[:]

    # Another episode: one session over the whole stream, against the role's atlas.
    assert not is_localized(scene, published, "session-b", "right", "back")
    out = localize_episode(scene, "session-b", "right", lens="back")
    scene_dir = scene.root / "session-b" / "right" / "scene"
    assert out == scene_dir / "shared_rear_observations.txt"
    assert is_localized(scene, published, "session-b", "right", "back")
    (call,) = calls.engine
    assert (call.mode, call.role, call.arm, call.lens) == (
        "localize",
        "shared_rear",
        "right",
        "rear",
    )
    assert call.atlas_in == scene_map / published["roles"]["shared_rear"] / "atlas.osa"
    assert call.mask_dir is None
    assert calls.exports == [
        (
            scene.root / "session-b" / "right" / "lens1.mp4",
            frames_dir(scene_map, "session-b", "right", "rear"),
        )
    ]
    assert not (scene_map / "frames" / "session-b-right-rear").exists()
    record = json.loads(out.with_suffix(".inputs.json").read_text())
    assert record["published_by"] == "scene localize"
    assert (record["arm"], record["lens"]) == ("right", "back")
    assert record["map"] == published["roles"]["shared_rear"]
    # The log is copied beside the observations.
    assert record["outputs"]["engine_log"]["path"] == "shared_rear.engine.log"
    assert (
        scene_dir / "shared_rear.engine.log"
    ).read_text() == "localize shared_rear\n"
    obs = read_observations(out)
    truth = _camera_in_world(obs.t_s, "right", "back")[:, :3, 3]
    np.testing.assert_allclose(obs.T[:, 2, 3], truth[:, 2], atol=0.005)

    # A stream localized against the published set is refused.
    with pytest.raises(FileExistsError, match="already exists - never overwritten"):
        localize_episode(scene, "session-b", "right", lens="back")
    assert len(calls.engine) == 1

    # Without its record a stream does not count as localized, nor with a
    # record that names another map; such observations are replaced.
    record_path = out.with_suffix(".inputs.json")
    record_path.unlink()
    assert not is_localized(scene, published, "session-b", "right", "back")
    record_path.write_text(
        json.dumps({**record, "map": "maps/shared_rear-000000000000"})
    )
    assert not is_localized(scene, published, "session-b", "right", "back")
    out.write_text("0 2 9 9 9 0 0 0 1\n")
    assert localize_episode(scene, "session-b", "right", lens="back") == out
    assert len(calls.engine) == 2
    assert len(read_observations(out)) == len(obs)
    assert (
        json.loads(record_path.read_text())["map"] == published["roles"]["shared_rear"]
    )
    assert is_localized(scene, published, "session-b", "right", "back")


@pytest.mark.parametrize(
    ("failure", "rows"), [("one row short", 1830), ("no row file", 0)]
)
def test_a_failed_localization_leaves_no_observations(world, failure, rows):
    scene = world.scene()
    build_scene_maps(scene)
    published = current_maps(scene)
    # The first session leaves its complete row file under scene_map/reloc/;
    # a later session that writes none must not publish those rows.
    out = localize_episode(scene, "session-b", "left")
    record_path = out.with_suffix(".inputs.json")
    record_path.write_text(
        json.dumps({"schema": "scene_localization_inputs", "map": "x"})
    )
    world.calls.failing[("localize", "front_left")] = failure

    with pytest.raises(
        RuntimeError, match=f"the engine wrote {rows} rows for 1831 frames"
    ):
        localize_episode(scene, "session-b", "left")

    assert not out.exists() and not record_path.exists()
    assert not (out.parent / "front_left.engine.log").exists()
    assert not (scene.root / "scene_map" / "frames" / "session-b-left-front").exists()
    assert not is_localized(scene, published, "session-b", "left", "front")
