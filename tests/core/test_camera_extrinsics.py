"""Tests for kiwi.core.camera.extrinsics - the shared Kalibr camchain reader.

The module reads exactly one artifact: the Kalibr cam-IMU camchain. The
hand-eye and the tool tip live in the descriptions tree, and export composes
camera->tip from the rendered URDF, so nothing here composes frames.

Pure yaml/numpy logic on tmp_path fixtures, plus one read of the shipped
X5-001 chains; no hardware. The synthetic camchains are built from random
SE(3)s with scipy so the expected values are derived independently of the
code under test.

Run: pixi run test
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from kiwi.core.camera.extrinsics import load_T_cam_imu, read_cam_imu


def _se3(seed: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    T[:3, 3] = np.random.default_rng(seed).uniform(-0.5, 0.5, 3)
    return T


def _write_camchain(path, T, *, timeshift=None, directive=False, extra=""):
    body = {"cam0": {"T_cam_imu": T.tolist()}}
    if timeshift is not None:
        body["cam0"]["timeshift_cam_imu"] = timeshift
    text = yaml.safe_dump(body)
    if directive:
        text = "%YAML:1.0\n" + extra + text
    path.write_text(text)


# --- read_cam_imu / load_T_cam_imu -----------------------------------------


def test_read_cam_imu_plain_yaml_with_timeshift(tmp_path):
    T = _se3(1)
    p = tmp_path / "extrinsic-imu.yaml"
    _write_camchain(p, T, timeshift=-0.0076368120202529016)
    got = read_cam_imu(p)
    np.testing.assert_allclose(got.T_cam_imu, T, atol=1e-12)
    assert got.timeshift_s == pytest.approx(-0.0076368120202529016)


def test_read_cam_imu_timeshift_absent_is_none_not_zero(tmp_path):
    p = tmp_path / "chain.yaml"
    _write_camchain(p, _se3(2))
    assert read_cam_imu(p).timeshift_s is None


def test_read_cam_imu_strips_opencv_yaml_directive(tmp_path):
    # The OpenCV FileStorage camchain opens with %YAML:1.0,
    # which PyYAML rejects outright - the reader must strip it.
    T = _se3(3)
    p = tmp_path / "kalibr_imucam_chain.yaml"
    _write_camchain(p, T, directive=True)
    np.testing.assert_allclose(load_T_cam_imu(p), T, atol=1e-12)


def test_read_cam_imu_rejects_missing_cam0(tmp_path):
    p = tmp_path / "not-a-chain.yaml"
    p.write_text("schema: something_else\n")
    with pytest.raises(ValueError, match="cam0.T_cam_imu"):
        read_cam_imu(p)


def test_read_cam_imu_rejects_wrong_shape(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"cam0": {"T_cam_imu": [[1, 0], [0, 1]]}}))
    with pytest.raises(ValueError, match="expected \\(4, 4\\)"):
        read_cam_imu(p)


def test_the_clock_equation_direction_is_code_not_prose(tmp_path):
    """t_imu = t_camera + dt_imu_from_camera: a synthetic pulse recorded at
    camera time 10.0 with a +0.25 s shift happens at 10.25 on the IMU clock."""
    from kiwi.core.camera.extrinsics import read_cam_imu

    chain = tmp_path / "extrinsic-imu.yaml"
    chain.write_text(
        "cam0:\n"
        "  T_cam_imu:\n"
        "  - [1.0, 0.0, 0.0, 0.0]\n"
        "  - [0.0, 1.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, 1.0, 0.0]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n"
        "  timeshift_cam_imu: 0.25\n"
    )
    ext = read_cam_imu(chain)
    t_camera = 10.0
    assert t_camera + ext.dt_imu_from_camera_s == 10.25


def test_the_shipped_calibrations_carry_the_frozen_convention():
    """The real registry fixtures: both lens chains parse, expose their
    timeshift through the frozen equation accessor, and T_cam_imu acts as
    p_cam = T @ p_imu (the IMU origin lands at the translation column)."""
    import numpy as np

    from kiwi.core.camera.extrinsics import read_cam_imu
    from kiwi.core.paths import repo_root

    unit = repo_root() / "config" / "cameras" / "insta360-X5-001"
    for name in ("extrinsic-imu.yaml", "extrinsic-imu-back.yaml"):
        ext = read_cam_imu(unit / name)
        assert ext.T_cam_imu.shape == (4, 4)
        assert ext.dt_imu_from_camera_s == ext.timeshift_s is not None
        p_imu_origin = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(ext.T_cam_imu @ p_imu_origin, ext.T_cam_imu[:, 3])


def test_an_inverse_spelling_is_refused_with_instructions(tmp_path):
    chain = tmp_path / "extrinsic-imu.yaml"
    chain.write_text(
        "cam0:\n"
        "  T_imu_cam:\n"
        "  - [1.0, 0.0, 0.0, 0.0]\n"
        "  - [0.0, 1.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, 1.0, 0.0]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n"
    )
    from kiwi.core.camera.extrinsics import read_cam_imu

    with pytest.raises(ValueError, match="republish"):
        read_cam_imu(chain)
