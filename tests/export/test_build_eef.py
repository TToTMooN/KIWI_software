"""Tool state through the export: nearest-frame sampling, zero interpolation.

Export never bridges tool-state NaN gaps with unbounded `np.interp`. Gaps
are completed and audited upstream by the eef stage
(docs/estimation.md); these tests pin that the export samples the
completed series at the same frame as the video, never blends frames, and
refuses a non-finite sample instead of patching it.

Pure numpy + a tmp_path npz: no cv2, no video (VideoInfo.read is stubbed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from kiwi.core.descriptions import fixed_frame_T, load_robot_description
from kiwi.core.trajectory import PoseTrack
from kiwi.eef.completion import complete_series
from kiwi.eef.state import EefState, write_eef_state
from kiwi.export import build
from kiwi.export.build import _ArmData, _build_arm, _sample_tool_state
from kiwi.export.source import ArmSource

FPS = 30.0
N = 90
DESCRIPTION = "test_calibrated_dc"


pytestmark = pytest.mark.usefixtures("calibrated_dc_description")


def _arm_data(v: np.ndarray, sync_off: float = 0.0) -> _ArmData:
    """An _ArmData with only the fields _sample_tool_state reads."""
    t = np.array([0.0, 100.0])
    finite = np.flatnonzero(np.isfinite(v))
    return _ArmData(
        label="left",
        out_label="left",
        frame="marker",
        map_set=None,
        ee_track=PoseTrack(
            t=t, rotations=Rotation.identity(2), positions=np.zeros((2, 3))
        ),
        cam_track=None,
        grip_v=np.asarray(v, dtype=np.float64),
        grip_fps=FPS,
        grip_t0=sync_off + finite[0] / FPS,
        grip_t1=sync_off + finite[-1] / FPS,
        lens0=Path("left_lens0.mp4"),
        eef_state=Path("left_eef.npz"),
        sync_off=sync_off,
        fps=FPS,
        n_video=N,
    )


def test_samples_the_same_frame_the_video_sampler_picks():
    """Grid instants at exact frame times return exactly v[frame] - the value
    and the pixels always come from one source frame."""
    v = np.linspace(0.0, 1.0, N)
    a = _arm_data(v, sync_off=0.4)
    idx = np.array([3, 17, 42])
    grid = 0.4 + idx / FPS
    np.testing.assert_allclose(_sample_tool_state(a, grid), v[idx], atol=1e-7)


def test_no_interpolation_between_frames():
    """A grid instant between two frames gets the NEARER frame's value, never a
    blend."""
    v = np.zeros(N)
    v[10] = 1.0
    a = _arm_data(v)
    got = _sample_tool_state(a, np.array([10.4 / FPS, 10.6 / FPS]))
    np.testing.assert_array_equal(got, [1.0, 0.0])


def test_a_non_finite_sample_is_refused_not_bridged():
    """Completed output is finite by contract, so a NaN reaching the sampler
    is a broken artifact: it raises with the arm, the source and the remedy
    rather than being filled from a neighbour."""
    v = np.linspace(0.0, 1.0, N)
    v[20] = np.nan
    a = _arm_data(v)
    with pytest.raises(ValueError, match="non-finite") as e:
        _sample_tool_state(a, np.arange(N) / FPS)
    msg = str(e.value)
    assert "'left'" in msg and "left_eef.npz" in msg
    assert "re-run the eef stage" in msg
    assert "never interpolates" in msg


# --------------------------------------------------------------------------- #
# _build_arm over a real eef_state_v1 artifact
# --------------------------------------------------------------------------- #
def _write_state(tmp_path, value, fps=FPS) -> Path:
    n = len(value)
    completed, audit, _ = complete_series(np.asarray(value, dtype=np.float64), fps)
    state = EefState(
        tool="chopsticks-v3",
        estimator="chopsticks",
        unit="deg",
        fps=fps,
        source="left_lens0.mp4",
        pose_mode="fixed_pose",
        pose_source="config/gripper/chopsticks-v3/units/chopstick-v3-m5.yaml",
        value=np.asarray(value, dtype=np.float64),
        rejected_value=np.full(n, np.nan),
        has_pose=np.ones(n, dtype=bool),
        n_detected=np.full(n, 6, dtype=np.int32),
        n_kept=np.full(n, 6, dtype=np.int32),
        error=np.asarray([""] * n, dtype=str),
        value_completed=completed,
        completion=audit,
    )
    return write_eef_state(tmp_path / "rec_eef.npz", state)


SCENE_FRAME = "shared_rear_metric_gravity_aligned"
MAP_SET = {"front": "attempts/2026-09-01-front", "rear": "attempts/2026-09-01-rear"}


def _write_world_tum(tmp_path, t, xyz=0.0, frame=SCENE_FRAME) -> Path:
    """The DEFAULT world trajectory: the state stage's `<arm>/scene/state.tum`."""
    p = tmp_path / "left" / "scene" / "state.tum"
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# t tx ty tz qx qy qz qw  (frame={frame} body=camera; "
        "t on episode global clock (origin=left))\n"
    )
    p.write_text(header + "".join(f"{v:.9f} {xyz} 0 0 0 0 0 1\n" for v in t))
    _write_state_inputs(p.parent)
    return p


def _write_state_inputs(scene_dir, map_set=None) -> Path:
    """The state stage's sidecar beside every state.tum - the map set it
    names is what identifies the world the trajectory lives in."""
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


@pytest.fixture()
def stub_video(monkeypatch):
    from kiwi.core.video import VideoInfo

    monkeypatch.setattr(
        build.VideoInfo,
        "read",
        classmethod(lambda cls, path: VideoInfo(64, 64, FPS, N)),
    )


def _arm_source(tmp_path, eef, **kw) -> ArmSource:
    (tmp_path / "rec_lens0.mp4").write_bytes(b"mp4")  # VideoInfo.read is stubbed
    defaults = dict(
        label="left",
        lens0=tmp_path / "rec_lens0.mp4",
        state_tum=tmp_path / "left" / "scene" / "state.tum",
        state_inputs=tmp_path / "left" / "scene" / "state.inputs.json",
        register_tum=tmp_path / "register" / "left.tum",
        eef_state=eef,
        sync_off=0.0,
        session_block={"angle_closed_deg": 20.0, "angle_open_deg": 100.0},
        tool="chopsticks-v3",
        # The real descriptions tree, not a synthetic URDF: the camera->tip
        # constant an arm exports at is composed from it by fixed-joint FK.
        description=DESCRIPTION,
    )
    defaults.update(kw)
    return ArmSource(**defaults)


def _description_legs() -> tuple[np.ndarray, np.ndarray]:
    """``(T_mount_cam, T_mount_tcp)`` as the builder reads them off the URDF."""
    desc = load_robot_description(DESCRIPTION)
    link = {r: desc.end_effectors[r].urdf_link for r in ("default", "camera", "tcp")}
    return (
        fixed_frame_T(desc.urdf_path, link["default"], link["camera"]),
        fixed_frame_T(desc.urdf_path, link["default"], link["tcp"]),
    )


def _mount_block(key: str, t: np.ndarray, **kw) -> dict:
    """One `build.mount_*:` block as chopstick_unit_v1 spells it.

    Just the matrix and the optional `measured:` flag; the closed key set
    rejects anything else (that check is what catches a mis-pasted leg).
    """
    return {key: [[float(v) for v in row] for row in t], **kw}


def _unit_yaml(tmp_path) -> Path:
    """A stand-in gripper-unit file, so the override stamp has real content."""
    p = tmp_path / "gripper-unit.yaml"
    p.write_text(
        "schema: chopstick_unit_v1\n"
        "tool: chopsticks-v3\n"
        "build:\n"
        "  marker: {id: 5, size: 0.014}\n"
    )
    return p


def test_build_arm_normalizes_with_the_bundle_endpoints(tmp_path, stub_video):
    angles = np.full(N, 60.0)  # (60-20)/(100-20) = 0.5
    angles[0] = 10.0  # below closed -> clips to 0
    angles[1] = 140.0  # above open -> clips to 1
    eef = _write_state(tmp_path, angles)
    _write_world_tum(tmp_path, [0.0, 3.0])
    a = _build_arm(
        _arm_source(tmp_path, eef),
        include_camera_pose=False,
        repo_root=None,
    )
    np.testing.assert_allclose(a.grip_v[:3], [0.0, 1.0, 0.5], atol=1e-12)
    assert a.frame == SCENE_FRAME
    assert a.grip_fps == FPS
    # Completed output is finite throughout, so its support is the whole clip.
    assert a.grip_t0 == 0.0
    assert a.grip_t1 == pytest.approx((N - 1) / FPS)
    assert a.tool_meta["estimator"] == "chopsticks"
    assert a.tool_meta["unit"] == "deg"
    assert a.tool_meta["endpoints"] == [20.0, 100.0]


def test_build_arm_without_a_description_raises(tmp_path, stub_video):
    """No description means no camera->tip constant - refuse, never identity.

    An arm whose tool config carries no ``capture.description:`` has
    nothing to compose the constant from; the refusal names both the tool and
    the tool_v1 block to add.
    """
    eef = _write_state(tmp_path, np.full(N, 60.0))
    _write_world_tum(tmp_path, [0.0, 3.0])
    with pytest.raises(ValueError, match="`capture.description:`") as e:
        _build_arm(
            _arm_source(tmp_path, eef, description=None),
            include_camera_pose=False,
            repo_root=None,
        )
    assert "chopsticks-v3" in str(e.value)


def test_build_arm_missing_state_tum_raises(tmp_path, stub_video):
    """The default world: no scene trajectory is a refusal naming the state
    stage, never a silent fall-back onto whatever register.tum is lying
    around."""
    eef = _write_state(tmp_path, np.full(N, 60.0))
    with pytest.raises(FileNotFoundError, match="state stage"):
        _build_arm(
            _arm_source(tmp_path, eef),
            include_camera_pose=False,
            repo_root=None,
        )


def test_build_arm_missing_register_tum_raises(tmp_path, stub_video):
    eef = _write_state(tmp_path, np.full(N, 60.0))
    with pytest.raises(FileNotFoundError, match="register stage"):
        _build_arm(
            _arm_source(tmp_path, eef),
            include_camera_pose=False,
            repo_root=None,
            world_source="marker",
        )


# --------------------------------------------------------------------------- #
# per-leg gripper-unit overrides (description default < unit override)
# --------------------------------------------------------------------------- #
def test_a_mount_tcp_override_replaces_only_the_tip_leg(tmp_path, stub_video):
    """One remeasured leg, the other still the description's.

    With the constant split into two mount-frame legs, a unit whose tip was
    re-glued states only ``mount_tcp:`` and keeps the design's camera seating.
    The composed constant, both ``overridden`` flags, the override
    stamp and the unmeasured verdict all have to reflect that split.
    """
    eef = _write_state(tmp_path, np.full(N, 60.0))
    _write_world_tum(tmp_path, [0.0, 3.0])
    t_mount_cam, t_mount_tcp = _description_legs()
    # 5 mm longer than the design tip, and declared a placeholder.
    t_override = t_mount_tcp.copy()
    t_override[2, 3] += 0.005
    unit = _unit_yaml(tmp_path)

    a = _build_arm(
        _arm_source(
            tmp_path,
            eef,
            mount_tcp_override=_mount_block("T_mount_tcp", t_override, measured=False),
            override_source=unit,
        ),
        include_camera_pose=False,
        repo_root=None,
    )

    want = np.linalg.inv(t_mount_cam) @ t_override
    np.testing.assert_allclose(a.tcp_meta["t_camera_tcp_xyz"], want[:3, 3], atol=1e-12)
    assert a.tcp_meta["measured"] is False
    ex = a.extrinsics
    assert ex["T_mount_tcp"]["overridden"] is True
    assert ex["T_mount_tcp"]["measured"] is False
    assert ex["T_mount_cam"]["overridden"] is False
    # The untouched leg is still the description's, verbatim.
    np.testing.assert_allclose(ex["T_mount_cam"]["matrix"], t_mount_cam, atol=1e-12)
    np.testing.assert_allclose(ex["T_mount_tcp"]["matrix"], t_override, atol=1e-12)
    np.testing.assert_allclose(ex["T_camera_tcp"]["matrix"], want, atol=1e-12)
    assert ex["description"]["name"] == DESCRIPTION
    # The unit file that supplied the override is stamped by content.
    assert ex["override"]["sha256"]
    assert ex["override"]["path"].endswith("gripper-unit.yaml")


def test_both_legs_can_be_overridden_together(tmp_path, stub_video):
    """A fully remeasured unit: neither leg comes from the description."""
    eef = _write_state(tmp_path, np.full(N, 60.0))
    _write_world_tum(tmp_path, [0.0, 3.0])
    t_mount_cam, t_mount_tcp = _description_legs()
    cam_override = t_mount_cam.copy()
    cam_override[0, 3] += 0.002
    tcp_override = t_mount_tcp.copy()
    tcp_override[1, 3] -= 0.003

    a = _build_arm(
        _arm_source(
            tmp_path,
            eef,
            mount_cam_override=_mount_block("T_mount_cam", cam_override),
            mount_tcp_override=_mount_block("T_mount_tcp", tcp_override, measured=True),
            override_source=_unit_yaml(tmp_path),
        ),
        include_camera_pose=False,
        repo_root=None,
    )

    ex = a.extrinsics
    assert ex["T_mount_cam"]["overridden"] is True
    assert ex["T_mount_tcp"]["overridden"] is True
    # An override block with no `measured:` key is a measurement by default.
    assert a.tcp_meta["measured"] is True
    want = np.linalg.inv(cam_override) @ tcp_override
    np.testing.assert_allclose(ex["T_camera_tcp"]["matrix"], want, atol=1e-12)
    np.testing.assert_allclose(a.tcp_meta["t_camera_tcp_xyz"], want[:3, 3], atol=1e-12)
