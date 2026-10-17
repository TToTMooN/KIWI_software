"""The state stage: real dispatch, scene routing, and reuse on the same clock.

The estimator itself is exercised in ``test_fusion``; here the graph
is faked and what runs for real is the composer wiring - back-reference
resolution, per-arm localization routing, artifact publication and reuse.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import yaml

from kiwi.core.camera.registry import CameraUnit
from kiwi.episode.stages import REGISTRY, run_stage

#: The registry loader that parses camera.yaml, before any fixture fakes it.
_REAL_UNIT_LOAD = CameraUnit.__dict__["load"]


def test_state_stage_is_dispatched_for_real(tmp_path):
    assert REGISTRY["state"] is not None, "the state stage must not be a scaffold"
    res = run_stage("state", tmp_path)
    assert res.stage == "state" and not res.ok
    assert "kiwi init" in res.summary


def test_missing_scene_back_reference_is_actionable(tmp_path):
    (tmp_path / "episode.yaml").write_text(
        yaml.safe_dump({"schema": "episode_v1", "name": "e", "arms": {}})
    )
    res = run_stage("state", tmp_path)
    assert not res.ok and "scene" in res.summary and "back-reference" in res.summary


@pytest.fixture
def scene_episode(tmp_path, monkeypatch):
    """A scene with a published map set + one episode carrying the back-ref."""
    cfg = tmp_path / "config"
    profiles = cfg / "scene" / "mapping-profiles"
    profiles.mkdir(parents=True)
    from kiwi.core.paths import repo_root

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
            "offsets": {"left": -0.25, "right": 0.0},
        },
    }
    for unit in ("unit-a", "unit-b"):
        (cfg / "cameras" / unit).mkdir(parents=True)
    (ep / "episode.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    from kiwi.episode.scene import init_scene

    init_scene(
        root=root,
        mapping_episode="session-a",
    )
    # a published, complete map set + frozen scene frame
    scene_map = root / "scene_map"
    roles = {}
    for role in ("front_left", "front_right", "shared_rear"):
        d = scene_map / "maps" / f"0000-{role}"
        d.mkdir(parents=True)
        (d / "atlas.osa").write_bytes(b"ATLAS")
        roles[role] = f"maps/0000-{role}"
    (scene_map / "current.json").write_text(
        json.dumps(
            {"schema": "scene_map_pointer", "episode": "session-a", "roles": roles}
        )
    )
    (scene_map / "scene_frame.json").write_text(
        json.dumps(
            {
                "schema": "scene_frame",
                "frame": "shared_rear_metric_gravity_aligned",
                "episode": "session-a",
                "maps": {},
            }
        )
    )
    return ep


@pytest.fixture
def fake_estimator(tmp_path, monkeypatch):
    """Fake every heavy dependency; record the routing."""
    calls = {"localize": [], "fuse": 0, "imu_t0": []}

    def fake_localize(scene, episode_id, arm, *, lens="front", deps=None):
        calls["localize"].append((episode_id, arm, lens))
        scene_dir = scene.episode_dir(episode_id) / arm / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        role = "shared_rear" if lens == "back" else f"front_{arm}"
        out = scene_dir / f"{role}_observations.txt"
        rows = "\n".join(f"{t * 33_000_000} 2 0.1 0.2 0.3 0 0 0 1 0" for t in range(60))
        out.write_text(rows + "\n")
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
        return type("E", (), {"T_cam_imu": np.eye(4), "timeshift_s": 0.0})()

    def fake_extract(recording, out_csv):
        # extractor clock: shifted forward by the pre-roll (starts at 0)
        t = np.arange(0.0, 2.8, 0.005)
        np.savetxt(
            out_csv,
            np.column_stack([t, np.zeros((len(t), 6))]),
            header="t gx gy gz ax ay az",
            comments="",
        )
        return type("M", (), {"clock_offset_s": 0.3})()

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
        calls["fuse"] += 1
        # record the first IMU time so tests can pin the clock of the solve:
        # the extractor's pre-roll shift undone, and no sync offset
        calls["imu_t0"].append(float(imu_t[0]))
        return np.tile(np.eye(4), (len(t_nodes), 1, 1))

    import kiwi.core.camera.extrinsics as extr_mod
    import kiwi.core.camera.registry as registry_mod
    import kiwi.episode.scene_build as scene_build_mod
    import kiwi.state.fusion as fusion_mod
    import kiwi.core.camera.imu as imu_mod

    monkeypatch.setattr(scene_build_mod, "localize_episode", fake_localize)
    monkeypatch.setattr(
        registry_mod.CameraUnit, "load", staticmethod(lambda d: FakeUnit(tmp_path))
    )
    monkeypatch.setattr(extr_mod, "read_cam_imu", fake_read_cam_imu)
    monkeypatch.setattr(imu_mod, "extract_imu_to_csv", fake_extract)
    monkeypatch.setattr(fusion_mod, "fuse", fake_fuse)
    return calls


def test_state_stage_routes_each_arm_and_publishes_once(scene_episode, fake_estimator):
    res = run_stage("state", scene_episode)
    assert res.ok, res.summary
    # each arm's OWN streams were localized: front + back per arm
    assert sorted(fake_estimator["localize"]) == [
        ("session-a", "left", "back"),
        ("session-a", "left", "front"),
        ("session-a", "right", "back"),
        ("session-a", "right", "front"),
    ]
    assert fake_estimator["fuse"] == 2
    for label in ("left", "right"):
        tum = scene_episode / label / "scene" / "state.tum"
        assert tum.is_file()
        sidecar = json.loads(
            (scene_episode / label / "scene" / "state.inputs.json").read_text()
        )
        assert sidecar["schema"] == "state_trajectory_inputs"
        assert sidecar["frame"] == "shared_rear_metric_gravity_aligned"
        assert sidecar["diagnostics"]["result_status"] == "computed_unvalidated"

    again = run_stage("state", scene_episode)
    assert again.ok and "reused" in again.summary
    assert fake_estimator["fuse"] == 2, "an unchanged state clock is reused"


def test_a_malformed_camera_yaml_fails_the_arm_instead_of_raising(
    scene_episode, fake_estimator, monkeypatch
):
    from kiwi.episode.config_snapshot import snapshot_config
    from kiwi.episode.manifest import EpisodeConfig

    monkeypatch.setattr(CameraUnit, "load", _REAL_UNIT_LOAD)
    snapshot_config(scene_episode)
    cfg = EpisodeConfig.load(scene_episode / "episode.yaml", check_exists=False)
    camera_yaml = cfg.arms["left"].camera_dir / "camera.yaml"
    camera_yaml.write_text("serial: [unclosed\n")

    res = run_stage("state", scene_episode)
    assert not res.ok and "left: FAILED" in res.summary
    # The summary names the file to fix, not the parser's "<unicode string>".
    assert f"{camera_yaml}: not valid YAML" in res.summary
    record = scene_episode / "left" / "scene" / "state.inputs.json"
    assert json.loads(record.read_text())["status"] == "failed"


def test_a_malformed_scene_yaml_is_a_result_not_a_traceback(
    scene_episode, fake_estimator
):
    scene_yaml = (scene_episode.parent / "scene.yaml").resolve()
    scene_yaml.write_text(scene_yaml.read_text() + "episodes: [unclosed\n")

    res = run_stage("state", scene_episode)
    assert not res.ok
    assert res.summary.startswith(f"{scene_yaml}: not valid YAML")


def test_state_stage_without_published_maps_names_the_missing_step(
    scene_episode, fake_estimator
):
    (scene_episode.parent / "scene_map" / "current.json").unlink()
    res = run_stage("state", scene_episode)
    assert not res.ok and "kiwi scene build" in res.summary


def test_published_state_is_on_the_episode_global_clock(scene_episode, fake_estimator):
    """Both arms must publish on ONE clock: cam-IMU timeshift + the arm's
    sync offset."""
    assert run_stage("state", scene_episode).ok
    # arms are processed in sorted order; the fake extractor pre-rolls 0.3 s,
    # which the loader undoes. The solve runs on each arm's own IMU clock, so
    # the sync offset reaches the written timestamps and not the IMU times
    assert fake_estimator["imu_t0"] == pytest.approx([-0.3, -0.3])
    for label, offset in (("left", -0.25), ("right", 0.0)):
        sidecar = json.loads(
            (scene_episode / label / "scene" / "state.inputs.json").read_text()
        )
        assert sidecar["clock_offsets_s"]["sync_offset"] == offset
        header = (scene_episode / label / "scene" / "state.tum").read_text()
        assert "episode global clock" in header.splitlines()[0]


def test_the_latent_state_is_published_beside_the_trajectory(
    scene_episode, fake_estimator
):
    """The IMU-body pose the graph solves for is kept beside the published
    front-camera pose, with the nodes that had a tracked localization."""
    from kiwi.state import read_state_latent

    assert run_stage("state", scene_episode).ok
    for label in ("left", "right"):
        latent = read_state_latent(scene_episode / label / "scene" / "state_latent.npz")
        assert latent["metadata"]["frame"] == "shared_rear_metric_gravity_aligned"
        assert len(latent["body_position"]) == len(latent["t"]) == 60
        assert latent["front_candidate_present"].all()
        assert latent["rear_candidate_present"].all()


@pytest.mark.parametrize("new_left_offset", [-0.5, -0.25])
@pytest.mark.parametrize("sync_only", [False, True])
def test_cli_resync_refreshes_state_timestamps_only_when_clock_changes(
    scene_episode, fake_estimator, monkeypatch, new_left_offset, sync_only
):
    from kiwi.cli import main
    from kiwi.core.clock import GlobalClock
    from kiwi.episode import stages
    from kiwi.state import read_state_latent
    from kiwi.sync import ClockEstimate

    assert run_stage("state", scene_episode).ok
    left = scene_episode / "left" / "scene"
    right = scene_episode / "right" / "scene"
    old_times = np.loadtxt(left / "state.tum")[:, 0]
    old_right = (right / "state.tum").read_bytes()
    monkeypatch.setattr(
        "kiwi.sync.resolve_clock",
        lambda *args, **kwargs: ClockEstimate(
            clock=GlobalClock(
                origin="right", offsets_s={"left": new_left_offset, "right": 0.0}
            ).validate(),
            method="audio_gcc_phat",
            reports=("test estimate",),
            loop_residuals={},
        ),
    )
    monkeypatch.setitem(
        stages.REGISTRY,
        "convert",
        lambda directory: stages.StageResult("convert", True, "reused"),
    )
    assert (
        main(
            [
                "run",
                "--episode",
                str(scene_episode),
                "--resync",
                "--to-stage",
                "sync" if sync_only else "state",
            ]
        )
        == 0
    )
    if sync_only:
        assert fake_estimator["fuse"] == 2
        assert (
            main(
                [
                    "run",
                    "--episode",
                    str(scene_episode),
                    "--from-stage",
                    "state",
                    "--to-stage",
                    "state",
                ]
            )
            == 0
        )

    new_times = old_times + new_left_offset - (-0.25)
    assert np.loadtxt(left / "state.tum")[:, 0] == pytest.approx(new_times)
    assert read_state_latent(left / "state_latent.npz")["t"] == pytest.approx(new_times)
    record = json.loads((left / "state.inputs.json").read_text())
    assert record["clock_offsets_s"]["sync_offset"] == new_left_offset
    assert record["status"] == "computed"
    assert (right / "state.tum").read_bytes() == old_right
    assert fake_estimator["fuse"] == (3 if new_left_offset != -0.25 else 2)


def test_failed_state_refresh_is_retried_while_old_trajectory_remains(
    scene_episode, fake_estimator, monkeypatch
):
    from kiwi.episode.manifest import write_sync_block

    assert run_stage("state", scene_episode).ok
    left = scene_episode / "left" / "scene"
    old_trajectory = (left / "state.tum").read_bytes()
    write_sync_block(
        scene_episode / "episode.yaml",
        {"left": -0.5, "right": 0.0},
        method="audio_gcc_phat",
        origin="right",
    )

    def fail_fusion(*args):
        raise RuntimeError("interrupted solver")

    with monkeypatch.context() as patch:
        patch.setattr("kiwi.state.fusion.fuse", fail_fusion)
        result = run_stage("state", scene_episode)
    assert not result.ok and "interrupted solver" in result.summary
    assert (left / "state.tum").read_bytes() == old_trajectory
    assert json.loads((left / "state.inputs.json").read_text())["status"] == "failed"

    result = run_stage("state", scene_episode)
    assert result.ok, result.summary
    assert np.loadtxt(left / "state.tum")[0, 0] == pytest.approx(-0.5)
    assert json.loads((left / "state.inputs.json").read_text())["status"] == "computed"
    assert fake_estimator["fuse"] == 3
