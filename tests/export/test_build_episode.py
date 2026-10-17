"""`build_episode` end to end over synthetic artifacts.

A real (tiny) mp4, real scene-frame state tums and real `eef_state_v1` npzs - the
builder path short of a dataset writer. Skips only when cv2 has no encoder.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from kiwi.eef.completion import complete_series  # noqa: E402
from kiwi.eef.state import EefState, write_eef_state  # noqa: E402
from kiwi.export.build import build_episode  # noqa: E402
from kiwi.export.source import ArmSource, EpisodeSource  # noqa: E402

FPS = 30.0
N = 90  # 3 s clip
# The explicit synthetic calibration fixture supplies the camera-to-tip
# constant, independently of the deployed descriptions awaiting calibration.
DESCRIPTION = "test_calibrated_dc"


pytestmark = pytest.mark.usefixtures("calibrated_dc_description")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("ep") / "rec_lens0.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder available in this build of cv2")
    img = np.zeros((64, 64, 3), np.uint8)
    for i in range(N):
        img[:, :] = (i, 255 - i, i)
        vw.write(img)
    vw.release()
    return p


def _write_eef(path, n=N, angle=60.0):
    completed, audit, _ = complete_series(np.full(n, angle), FPS)
    state = EefState(
        tool="chopsticks-v3",
        estimator="chopsticks",
        unit="deg",
        fps=FPS,
        source="rec_lens0.mp4",
        pose_mode="fixed_pose",
        pose_source="config/gripper/chopsticks-v3/units/chopstick-v3-m5.yaml",
        value=np.full(n, angle),
        rejected_value=np.full(n, np.nan),
        has_pose=np.ones(n, dtype=bool),
        n_detected=np.full(n, 6, dtype=np.int32),
        n_kept=np.full(n, 6, dtype=np.int32),
        error=np.asarray([""] * n, dtype=str),
        value_completed=completed,
        completion=audit,
    )
    return write_eef_state(path, state)


SCENE_FRAME = "shared_rear_metric_gravity_aligned"
MAP_SET = {"front": "attempts/2026-09-01-front", "rear": "attempts/2026-09-01-rear"}


def _write_state_inputs(scene_dir, map_set=None):
    """The state stage's sidecar beside every state.tum: the map set it names
    is what identifies the world the trajectory lives in."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    p = scene_dir / "state.inputs.json"
    p.write_text(
        json.dumps(
            {
                "schema": "state_trajectory_inputs",
                "frame": SCENE_FRAME,
                "map_set": map_set or MAP_SET,
            }
        )
    )
    return p


def _write_tum(path, *, frame=SCENE_FRAME, x_of_t=lambda t: t):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# t tx ty tz qx qy qz qw  (frame={frame} body=camera; "
        f"t on episode global clock (origin=left))\n"
    )
    t = np.arange(0.0, 3.01, 0.1)
    path.write_text(
        header + "".join(f"{v:.9f} {x_of_t(v):.9f} 0 0 0 0 0 1\n" for v in t)
    )
    return path


def _arm(tmp_path, clip, label, *, frame=SCENE_FRAME, x_of_t=lambda t: t, sync_off=0.0):
    tum = _write_tum(
        tmp_path / label / "scene" / "state.tum", frame=frame, x_of_t=x_of_t
    )
    inputs = _write_state_inputs(tum.parent)
    eef = _write_eef(tmp_path / f"{label}_eef.npz")
    return ArmSource(
        label=label,
        lens0=clip,
        state_tum=tum,
        state_inputs=inputs,
        eef_state=eef,
        sync_off=sync_off,
        session_block={"angle_closed_deg": 20.0, "angle_open_deg": 100.0},
        tool="chopsticks-v3",
        description=DESCRIPTION,
    )


def _source(tmp_path, clip, **kw):
    # Default arms are built only when not overridden - _arm() writes the
    # world tum as a side effect, and an eager default would overwrite an
    # override's file.
    if "arms" not in kw:
        kw["arms"] = (_arm(tmp_path, clip, "left"), _arm(tmp_path, clip, "right"))
    return EpisodeSource(
        name="demo", episode_dir=tmp_path, task="stack the cubes", **kw
    )


def test_bimanual_state_layout_and_clock(tmp_path, clip):
    ep = build_episode(_source(tmp_path, clip))
    ep.validate()
    state = ep.features["observation.state"]
    assert state.shape == (16,)
    assert state.names[:8] == [
        "left_ee_x",
        "left_ee_y",
        "left_ee_z",
        "left_ee_qx",
        "left_ee_qy",
        "left_ee_qz",
        "left_ee_qw",
        "left_gripper",
    ]
    assert state.names[8].startswith("right_")
    assert ep.timestamps[0] == 0.0
    assert ep.fps == pytest.approx(FPS, rel=1e-3)
    # The state tum is a pure +x translation with identity rotation, so the
    # exported tip track is that camera track rigidly offset by the composed
    # constant's translation - the constant is applied, and applied once.
    grid = ep.timestamps + ep.meta["global_t0"]
    t_camera_tcp = np.asarray(ep.meta["extrinsics"]["left"]["T_camera_tcp"]["matrix"])
    np.testing.assert_allclose(state.data[:, 0], grid + t_camera_tcp[0, 3], atol=1e-6)
    # (60 - 20) / (100 - 20) = 0.5 everywhere, tool state and pixels same-frame.
    np.testing.assert_allclose(state.data[:, 7], 0.5, atol=1e-7)
    # one wrist video per arm, named by schema.wrist_image_key
    assert set(ep.video_features()) == {
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    # `ee_frame` is a stable npz/provenance key, always "tcp": the tool tip is
    # the only body frame this exporter produces.
    assert ep.frame == SCENE_FRAME and ep.meta["ee_frame"] == "tcp"
    assert ep.meta["tool_state"]["left"]["endpoints"] == [20.0, 100.0]
    assert ep.meta["tcp"]["left"]["measured"] is True
    # The whole chain is self-describing: both mount-frame legs with the
    # description they came from, the composed constant, and the one actually
    # applied to the trajectory.
    ex = ep.meta["extrinsics"]["left"]
    assert ex["description"]["name"] == DESCRIPTION
    for leg in ("T_mount_cam", "T_mount_tcp"):
        assert set(ex[leg]) >= {"matrix", "overridden"}
        assert ex[leg]["overridden"] is False  # no gripper unit in this fixture
        assert np.asarray(ex[leg]["matrix"]).shape == (4, 4)
    assert "override" not in ex  # nothing overrode a leg, so nothing is stamped
    assert "T_camera_tcp" in ex
    assert "T_camera_ee" in ex
    np.testing.assert_allclose(
        ex["T_camera_ee"]["matrix"], ex["T_camera_tcp"]["matrix"], atol=1e-15
    )


def test_task_window_trims_and_rebases(tmp_path, clip):
    src = _source(tmp_path, clip, task_window=(0.5, 2.0))
    ep = build_episode(src)
    assert ep.meta["window"] == [pytest.approx(0.5), pytest.approx(2.0)]
    assert ep.meta["windowed"] is True
    assert ep.timestamps[0] == 0.0
    assert ep.timestamps[-1] <= 1.5 + 1e-9
    # An explicit window overrides the spec's task_window wholesale.
    ep2 = build_episode(src, window=(1.0, None))
    assert ep2.meta["window"][0] == pytest.approx(1.0)
    assert ep2.meta["window"][1] > 2.0  # open end -> real coverage stands


def test_arms_in_different_worlds_are_refused(tmp_path, clip):
    src = _source(
        tmp_path,
        clip,
        arms=(
            _arm(tmp_path, clip, "left", frame=SCENE_FRAME),
            _arm(tmp_path, clip, "right", frame="vio_world"),
        ),
    )
    with pytest.raises(ValueError, match="disagree on the world frame"):
        build_episode(src)


def test_a_second_stream_refuses_a_defaulted_arm_offset(tmp_path, clip):
    """0.0 is only correct when there is nothing to be offset FROM.

    With two streams the offsets are what relate them, so a missing entry
    seats that arm at the global origin - silently shifted by the whole
    capture-start delta, which is what _build_camera already refuses.
    """
    src = _source(
        tmp_path,
        clip,
        arms=(
            _arm(tmp_path, clip, "left"),
            _arm(tmp_path, clip, "right", sync_off=None),
        ),
    )
    with pytest.raises(ValueError, match="no sync offset for arm"):
        build_episode(src)


def test_a_single_stream_episode_still_defaults_the_offset(tmp_path, clip):
    src = _source(tmp_path, clip, arms=(_arm(tmp_path, clip, "left", sync_off=None),))
    ep = build_episode(src)
    assert ep.meta["sync_offsets"]["left"] == 0.0


def test_colliding_camera_keys_are_refused(tmp_path, clip):
    with pytest.raises(ValueError, match="used twice"):
        build_episode(
            _source(tmp_path, clip), cameras={"left": "head", "right": "head"}
        )


def test_empty_window_is_refused(tmp_path, clip):
    with pytest.raises(ValueError, match="empty episode window"):
        build_episode(_source(tmp_path, clip), window=(2.5, 2.4))


def test_relative_state_is_placement_invariant(tmp_path, clip):
    """`--relative-state` ignores where the world anchor sat, while the
    absolute state moves with it."""
    src = _source(tmp_path, clip)
    a = build_episode(src, include_relative_state=True)
    b = build_episode(
        src,
        include_relative_state=True,
        world_yaw=-90.0,
        world_offset=(0.8, 0.1, 0.0),
    )
    key = "observation.state.relative"
    ra, rb = a.features[key], b.features[key]
    assert ra.shape == (14,)  # 7 per arm, pose only
    assert ra.names[0] == "left_rel_ee_x" and ra.names[7] == "right_rel_ee_x"
    # Row 0 is the identity pose.
    np.testing.assert_allclose(ra.data[0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.abs(ra.data[0, 3:7]), [0, 0, 0, 1], atol=1e-6)
    # The relative track is identical under the re-seating...
    np.testing.assert_allclose(ra.data[:, :3], rb.data[:, :3], atol=1e-5)
    dots = np.abs(np.sum(ra.data[:, 3:7] * rb.data[:, 3:7], axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-5)  # same rotation, either sign
    # ...while the absolute state genuinely moved, so the test bites.
    sa = a.features["observation.state"].data
    sb = b.features["observation.state"].data
    assert not np.allclose(sa[:, :2], sb[:, :2], atol=1e-3)


# --- observation.pose_gap.<arm> (per-frame validity disclosure) -------------


def test_pose_gap_is_zero_on_raw_samples_and_bounded_between(tmp_path, clip):
    from kiwi.export.schema import pose_gap_key

    oe = build_episode(_source(tmp_path, clip))
    for label in ("left", "right"):
        col = oe.features[pose_gap_key(label)]
        assert col.dtype == "float32" and tuple(col.shape) == (1,)
        assert col.names == [f"{label}_pose_gap_s"]
        gap = np.asarray(col.data)[:, 0]
        assert len(gap) == len(oe.timestamps)
        # The state tum samples every 0.1 s, so no grid instant sits
        # further than half a source period from a raw sample...
        assert float(gap.max()) <= 0.05 + 1e-9
        # ...and grid instants that COINCIDE with a raw sample read ~0
        # (t=0.0 is both a grid instant and a raw sample).
        assert gap[0] == pytest.approx(0.0, abs=1e-9)
        # The 30 Hz grid lands between the 10 Hz raw samples most of the
        # time - the disclosure must show that, not report all-zero.
        assert float(gap.max()) > 0.01


def test_pose_gap_grows_across_a_tracking_dropout(tmp_path, clip):
    from kiwi.export.schema import pose_gap_key

    def sparse_tum(path, *, frame=SCENE_FRAME, x_of_t=lambda t: t):
        # Same span as _write_tum, but the middle second of samples is gone -
        # a tracking dropout. The pose channel Slerps straight across it; the
        # gap column is what discloses that those poses are guesses.
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# t tx ty tz qx qy qz qw  (frame={frame} body=camera; "
            f"t on episode global clock (origin=left))\n"
        )
        t = [v for v in np.arange(0.0, 3.01, 0.1) if not (1.0 < v < 2.0)]
        path.write_text(
            header + "".join(f"{v:.9f} {x_of_t(v):.9f} 0 0 0 0 0 1\n" for v in t)
        )
        return path

    arms = []
    for label in ("left", "right"):
        tum = sparse_tum(tmp_path / label / "scene" / "state.tum")
        inputs = _write_state_inputs(tum.parent)
        eef = _write_eef(tmp_path / f"{label}_eef.npz")
        arms.append(
            ArmSource(
                label=label,
                lens0=clip,
                state_tum=tum,
                state_inputs=inputs,
                eef_state=eef,
                sync_off=0.0,
                session_block={"angle_closed_deg": 20.0, "angle_open_deg": 100.0},
                tool="chopsticks-v3",
                description=DESCRIPTION,
            )
        )
    oe = build_episode(_source(tmp_path, clip, arms=tuple(arms)))
    gap = np.asarray(oe.features[pose_gap_key("left")].data)[:, 0]
    t = oe.timestamps
    inside = gap[(t > 1.3) & (t < 1.7)]  # deep inside the dropout
    outside = gap[t < 0.9]
    assert float(inside.min()) > 0.2  # a guess, and visibly so
    assert float(outside.max()) <= 0.05 + 1e-9  # measured region unaffected


def test_pose_gap_can_be_dropped(tmp_path, clip):
    from kiwi.export.schema import POSE_GAP_KEY_PREFIX

    oe = build_episode(_source(tmp_path, clip), include_pose_gap=False)
    assert not [k for k in oe.features if k.startswith(POSE_GAP_KEY_PREFIX)]


@pytest.mark.parametrize("window", [None, (0.5, 2.5)])
def test_completed_output_and_fill_audit_survive_export_sampling(
    tmp_path, clip, window
):
    from kiwi.eef.completion import complete_series
    from kiwi.export.archive import SourceArchive

    source = _source(tmp_path, clip)
    raw = np.full(N, np.nan)
    raw[15], raw[75] = 40.0, 80.0
    completed, audit, _ = complete_series(raw, FPS)
    for arm in source.arms:
        # Exercise the real on-disk consumer boundary.
        with np.load(arm.eef_state) as data:
            payload = {key: data[key] for key in data.files}
        payload["value"] = raw
        payload["value_completed"] = completed
        payload.update({f"completion/{key}": value for key, value in audit.items()})
        np.savez_compressed(arm.eef_state, **payload)

    exported = build_episode(source, fps=15.0, window=window)
    if window is None:
        assert exported.meta["global_t0"] == 0.0
        assert exported.timestamps[-1] > 2.5
    samples = exported.meta["export_samples"]
    for label, column in (("left", 7), ("right", 15)):
        indices = samples[f"eef_frame_index_{label}"]
        np.testing.assert_allclose(
            exported.features["observation.state"].data[:, column],
            (completed[indices] - 20.0) / 80.0,
        )
        for key, values in audit.items():
            if np.ndim(values) == 1:
                np.testing.assert_array_equal(
                    samples[f"eef_completion_{key}_{label}"], values[indices]
                )
        summary = exported.meta["tool_state"][label]["completion"]
        assert summary["filled_frames"] == N - 2
        assert summary["policy"] == "linear_interior_nearest_edge"
        assert summary == {
            key: value.item() for key, value in audit.items() if np.ndim(value) == 0
        }
        assert "method" not in summary
        assert exported.meta["tool_state"][label]["output_channel"] == "value_completed"
    # The exported audit remains usable after the source file is replaced.
    archive = SourceArchive(tmp_path / "dataset")
    reference = archive.samples(exported, 0)
    for arm in source.arms:
        arm.eef_state.unlink()
    with np.load(archive.root / reference["path"], allow_pickle=False) as persisted:
        for key, values in samples.items():
            np.testing.assert_array_equal(persisted[key], values)


def test_optional_head_preserves_hands_and_records_actual_samples(tmp_path, clip):
    from kiwi.export.schema import HEAD_MARKER_POSE_KEY, HEAD_MARKER_VALID_KEY

    source = _source(tmp_path, clip)
    episode = build_episode(
        source, include_head_marker=True, fps=15.0, window=(0.5, 1.5)
    )
    assert episode.features["observation.state"].shape == (16,)
    assert np.isnan(episode.features[HEAD_MARKER_POSE_KEY].data).all()
    assert not episode.features[HEAD_MARKER_VALID_KEY].data.any()
    samples = episode.meta["export_samples"]
    np.testing.assert_allclose(samples["t_episode"], 0.5 + episode.timestamps)
    for label in ("left", "right"):
        np.testing.assert_array_equal(
            samples[f"video_frame_index_{label}"], 15 + 2 * np.arange(len(episode))
        )
        np.testing.assert_array_equal(
            samples[f"eef_frame_index_{label}"], samples[f"video_frame_index_{label}"]
        )


def test_head_cannot_be_combined_with_marker_world(tmp_path, clip):
    with pytest.raises(ValueError, match="markerless scene world"):
        build_episode(
            _source(tmp_path, clip), include_head_marker=True, world_source="marker"
        )


def test_changed_scene_placement_is_not_hidden_by_same_map_names(tmp_path, clip):
    source = _source(tmp_path, clip)
    for i, arm in enumerate(source.arms):
        record = json.loads(arm.state_inputs.read_text())
        record["world_scene"] = {"scene_frame_digest": f"placement-{i}"}
        arm.state_inputs.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scene_frame_digest"):
        build_episode(source)


def test_optional_ground_preserves_hands_and_is_nan_when_absent(tmp_path, clip):
    from kiwi.export.schema import GROUND_PLANE_KEY, camera_height_key, camera_pose_key

    episode = build_episode(_source(tmp_path, clip), include_ground=True)
    assert episode.features["observation.state"].shape == (16,)
    assert np.isnan(episode.features[GROUND_PLANE_KEY].data).all()
    for label in ("left", "right"):
        assert np.isnan(episode.features[camera_height_key(label)].data).all()
        # the camera track serves the height; its own feature stays opt-in
        assert camera_pose_key(label) not in episode.features
    assert episode.meta["ground"]["status"] == "absent"
    assert episode.meta["ground"]["valid_frames"] == 0


def test_optional_ground_measures_the_camera_above_the_reported_floor(tmp_path, clip):
    import hashlib

    from kiwi.core.digest import digest_npz
    from kiwi.export.schema import GROUND_PLANE_KEY, camera_height_key
    from kiwi.export.source import GroundSource

    ground = tmp_path / "ground"
    ground.mkdir()
    npz = ground / "ground_observations.npz"
    np.savez(npz, z=np.array([-1.2]))
    report = ground / "ground_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "ground_v1",
                "status": "computed",
                "verdict": "GREEN",
                "reasons": [],
                "floor_z_m": -1.2,
                "sigma_m": 0.006,
                "frame": SCENE_FRAME,
                "world_scene": {
                    "frame": SCENE_FRAME,
                    "scene": "kitchen",
                    "map_set_digest": hashlib.sha256(
                        json.dumps(
                            MAP_SET, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "scene_frame_digest": None,
                },
                "quality_outputs": {
                    "observations": {"path": npz.name, "digest": digest_npz(npz)}
                },
            }
        )
    )
    source = _source(tmp_path, clip, scene="kitchen", ground=GroundSource(report, npz))
    episode = build_episode(source, include_ground=True, world_offset=(0.0, 0.0, 0.5))
    plane = episode.features[GROUND_PLANE_KEY].data
    np.testing.assert_allclose(
        plane, np.tile([0.0, 0.0, 1.0, 1.2 - 0.5], (len(episode), 1)), atol=1e-6
    )
    for label in ("left", "right"):
        # the state tums hold the camera at z = 0: 1.2 m above the floor,
        # whatever re-seating the export bakes into both
        np.testing.assert_allclose(
            episode.features[camera_height_key(label)].data[:, 0], 1.2, atol=1e-6
        )
    assert episode.meta["ground"]["status"] == "computed"
    assert episode.meta["ground"]["placement_verification"] == "unknown"


def test_ground_cannot_be_combined_with_marker_world(tmp_path, clip):
    with pytest.raises(ValueError, match="markerless scene world"):
        build_episode(
            _source(tmp_path, clip), include_ground=True, world_source="marker"
        )
