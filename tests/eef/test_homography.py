"""Tests for kiwi.eef.homography - the fixed camera<->marker pose calibration.

aggregate_pose gets three robustness cases; calibrate_fixed_pose runs
end to end on frames with a REAL rendered ArUco tag (drawn, not footage)
warped under a known pose, through the monkeypatched rectifier/video seams;
write_fixed_pose covers the in-place chopstick_unit_v1 contract - the whole
build record (tag id + size + pose) written as one, and the refuse-overwrite
guard on an existing build.fixed_pose.

Run: pixi run test
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

from kiwi.core import rectify as rectify_mod  # noqa: E402
from kiwi.core.markers import marker_object_points  # noqa: E402
from kiwi.core.tool_config import load_gripper_unit  # noqa: E402
from kiwi.eef import homography as homog  # noqa: E402

W = H = 640
K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 320.0], [0.0, 0.0, 1.0]])
MARKER_ID = 4
MARKER_SIZE = 0.030  # ~92 px across when projected - big enough for unbiased corners
RVEC = np.array([2.7, 0.12, 0.05])
TVEC = np.array([0.0, 0.02, 0.26])


# --- aggregate_pose ----------------------------------------------------------


def test_constant_pose_recovered_with_zero_spread():
    rvecs = np.tile([0.1, -0.2, 0.3], (40, 1))
    tvecs = np.tile([0.01, 0.02, 0.30], (40, 1))
    agg = homog.aggregate_pose(rvecs, tvecs)
    np.testing.assert_allclose(agg["rvec"], [0.1, -0.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(agg["tvec"], [0.01, 0.02, 0.30], atol=1e-12)
    assert agg["rot_dev_p90"] == pytest.approx(0.0, abs=1e-9)
    assert agg["t_dev_p90_mm"] == pytest.approx(0.0, abs=1e-9)


def test_median_translation_shrugs_off_a_gross_outlier():
    tvecs = np.tile([0.0, 0.0, 0.30], (21, 1))
    tvecs[7] = [0.5, -0.5, 2.0]  # one wildly bad frame
    agg = homog.aggregate_pose(np.zeros((21, 3)), tvecs)
    np.testing.assert_allclose(agg["tvec"], [0.0, 0.0, 0.30], atol=1e-12)


def test_scattered_cluster_flags_non_rigid():
    rng = np.random.default_rng(3)
    rvecs = rng.normal(0.0, 0.15, size=(50, 3))  # ~8.6 deg per axis
    tvecs = rng.normal([0, 0, 0.3], 0.02, size=(50, 3))  # ~20 mm
    agg = homog.aggregate_pose(rvecs, tvecs)
    assert agg["rot_dev_p90"] > homog.RIGIDITY_ROT_P90_WARN_DEG
    assert agg["t_dev_p90_mm"] > homog.RIGIDITY_TRANS_P90_WARN_MM


# --- calibrate_fixed_pose, end to end ---------------------------------------


def _tag_frame() -> np.ndarray:
    """A real DICT_4X4_50 tag warped into the frame under the known pose."""
    side, pad = 120, 30
    tag = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID, side
    )
    padded = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    padded[pad : pad + side, pad : pad + side] = tag
    src = np.float32(
        [[pad, pad], [pad + side, pad], [pad + side, pad + side], [pad, pad + side]]
    )
    proj, _ = cv2.projectPoints(
        marker_object_points(MARKER_SIZE), RVEC, TVEC, K, np.zeros(5)
    )
    Hm = cv2.getPerspectiveTransform(src, proj.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(padded, Hm, (W, H))
    mask = cv2.warpPerspective(np.full_like(padded, 255), Hm, (W, H))
    frame = np.full((H, W, 3), 128, np.uint8)
    frame[mask > 127] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 127]
    return frame


class _IdentityRect:
    K_pinhole = K
    out_size = (W, H)

    @staticmethod
    def apply(frame):
        return frame


def _fake_video(monkeypatch, frames):
    info = homog.VideoInfo(width=W, height=H, fps=30.0, n_frames=len(frames))
    monkeypatch.setattr(homog.VideoInfo, "read", staticmethod(lambda p: info))
    monkeypatch.setattr(homog, "iter_frames", lambda p, stride=1: enumerate(frames))
    monkeypatch.setattr(
        rectify_mod, "load_rectifier_auto", lambda *a, **k: _IdentityRect()
    )


def test_calibrate_recovers_the_known_pose(tmp_path, monkeypatch):
    _fake_video(monkeypatch, [_tag_frame()] * 25)
    lens0 = tmp_path / "clip_lens0.mp4"
    lens0.write_bytes(b"x")
    intr = tmp_path / "intrinsics.npz"
    intr.write_bytes(b"x")
    res = homog.calibrate_fixed_pose(
        lens0, intr, marker_id=MARKER_ID, marker_size_m=MARKER_SIZE
    )
    assert res.n_used == 25
    np.testing.assert_allclose(res.rvec, RVEC, atol=0.02)
    np.testing.assert_allclose(res.tvec, TVEC, atol=0.002)
    assert res.reproj_med_px < 1.0
    assert res.rot_dev_p90_deg == pytest.approx(0.0, abs=1e-6)
    assert res.warnings == []
    fp = res.fixed_pose_block()
    assert set(fp) == {"rvec", "tvec", "fov_deg", "out_size", "lens"}
    assert fp["out_size"] == [W, H]
    assert "rigidity check" in res.summary()


def test_calibrate_refuses_too_few_detections(tmp_path, monkeypatch):
    _fake_video(monkeypatch, [np.full((H, W, 3), 128, np.uint8)] * 25)
    lens0 = tmp_path / "clip_lens0.mp4"
    lens0.write_bytes(b"x")
    intr = tmp_path / "intrinsics.npz"
    intr.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="good detections"):
        homog.calibrate_fixed_pose(
            lens0, intr, marker_id=MARKER_ID, marker_size_m=MARKER_SIZE
        )


# --- write_fixed_pose --------------------------------------------------------


def _result() -> homog.FixedPoseResult:
    return homog.FixedPoseResult(
        rvec=RVEC,
        tvec=TVEC,
        fov_deg=160.0,
        out_size=(1920, 1920),
        lens="front",
        n_used=300,
        reproj_med_px=0.57,
        rot_dev_p50_deg=0.2,
        rot_dev_p90_deg=0.49,
        t_dev_p50_mm=0.05,
        t_dev_p90_mm=0.1,
    )


def test_write_creates_a_fresh_chopstick_unit_v1_file(tmp_path):
    """Everything the calibration knows lands in the `build:` block: the tag
    it was solved against and the pose bound to it, as one record."""
    p = tmp_path / "chopstick-v3-m5.yaml"
    homog.write_fixed_pose(
        p, _result(), tool="chopsticks-v3", marker_id=5, marker_size_m=0.014
    )
    doc = yaml.safe_load(p.read_text())
    assert doc["schema"] == "chopstick_unit_v1"
    assert set(doc["build"]) == {"marker", "fixed_pose"}

    gu = load_gripper_unit(p)  # a valid unit file by construction
    assert gu.tool == "chopsticks-v3"
    # The size is persisted even when it matches the design stock: tool_v1
    # has no version-wide size to fall back on, and the tvec is linear in it.
    assert (gu.marker_id, gu.marker_size) == (5, 0.014)
    assert gu.fixed_pose["lens"] == "front"
    assert gu.fixed_pose["fov_deg"] == 160.0
    np.testing.assert_allclose(gu.fixed_pose["rvec"], RVEC)


def test_write_replaces_the_tag_identity_it_calibrated_against(tmp_path):
    """A re-sticker must not leave a stale id behind - a pose bound to a tag
    the file no longer names. The write REPLACES build.marker, and leaves
    every other block alone."""
    p = tmp_path / "u.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "schema": "chopstick_unit_v1",
                "tool": "chopsticks-v3-1",
                "build": {"marker": {"id": 4, "size": 0.0195}},
                "session": {"angle_closed_deg": 70.0, "angle_open_deg": 165.0},
            },
            sort_keys=False,
        )
    )
    homog.write_fixed_pose(p, _result(), marker_id=6, marker_size_m=0.014)

    gu = load_gripper_unit(p)
    assert (gu.marker_id, gu.marker_size) == (6, 0.014)  # both replaced
    assert gu.tool == "chopsticks-v3-1"  # the unit->design link survives
    assert gu.session == {"angle_closed_deg": 70.0, "angle_open_deg": 165.0}
    assert gu.fixed_pose["fov_deg"] == 160.0


def test_write_refuses_to_replace_a_calibrated_pose_without_force(tmp_path):
    """The old pose vanishes from the file on an overwrite (git keeps the
    history), so replacing it must be a decision, not a side effect."""
    p = tmp_path / "u.yaml"
    homog.write_fixed_pose(
        p, _result(), tool="chopsticks-v3", marker_id=6, marker_size_m=0.014
    )
    with pytest.raises(
        RuntimeError, match=r"already carries.*build\.fixed_pose.*--force"
    ):
        homog.write_fixed_pose(p, _result())
    homog.write_fixed_pose(p, _result(), force=True)  # explicit replace ok
    assert load_gripper_unit(p).marker_id == 6  # the build record stands


def test_rigidity_warning_fires_past_the_thresholds():
    import dataclasses

    bad = dataclasses.replace(_result(), rot_dev_p90_deg=4.0)
    assert any("not be rigid" in w for w in bad.warnings)
    assert "WARNING" in bad.summary()
