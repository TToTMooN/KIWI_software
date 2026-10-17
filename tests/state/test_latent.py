"""The latent-state artifact: the IMU-body pose beside the front-camera pose."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.state import LATENT_SCHEMA, read_state_latent, write_state_latent

N = 7


def _poses(seed):
    """(N, 4, 4) poses with a rotation about z and random positions."""
    T = np.tile(np.eye(4), (N, 1, 1))
    angles = np.linspace(0.0, 0.3 + seed, N)[:, None]
    T[:, :3, :3] = Rotation.from_euler("z", angles).as_matrix()
    T[:, :3, 3] = np.random.default_rng(seed).normal(size=(N, 3))
    return T


def _write(path, frame="scene", **kwargs):
    return write_state_latent(
        path,
        t=np.linspace(0.0, 1.0, N),
        body=_poses(0),
        front=_poses(1),
        front_candidate_present=np.ones(N, bool),
        rear_candidate_present=np.zeros(N, bool),
        frame=frame,
        **kwargs,
    )


def test_the_latent_state_roundtrips_with_its_frame_declared(tmp_path):
    out = _write(
        tmp_path / "state_latent.npz",
        frame="shared_rear_metric_gravity_aligned",
        metadata={"clock": "episode_global_seconds"},
    )
    got = read_state_latent(out)
    meta = got.pop("metadata")
    assert meta["schema"] == LATENT_SCHEMA
    assert meta["frame"] == "shared_rear_metric_gravity_aligned"
    assert meta["latent_body"] == "imu_body_centre"
    assert meta["output_body"] == "front_camera_optical_centre"
    assert meta["clock"] == "episode_global_seconds"
    assert set(got) == {
        "t",
        "body_position",
        "body_quaternion_xyzw",
        "front_position",
        "front_quaternion_xyzw",
        "front_candidate_present",
        "rear_candidate_present",
    }
    np.testing.assert_array_equal(got["t"], np.linspace(0.0, 1.0, N))
    for name, poses in (("body", _poses(0)), ("front", _poses(1))):
        np.testing.assert_array_equal(got[f"{name}_position"], poses[:, :3, 3])
        np.testing.assert_allclose(
            Rotation.from_quat(got[f"{name}_quaternion_xyzw"]).as_matrix(),
            poses[:, :3, :3],
            atol=1e-12,
        )
    assert got["front_candidate_present"].dtype == bool
    assert got["front_candidate_present"].all()
    assert not got["rear_candidate_present"].any()


def test_a_truncated_latent_artifact_is_refused(tmp_path):
    out = _write(tmp_path / "s.npz")
    with np.load(out, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["body_position"] = arrays["body_position"][:-2]
    np.savez_compressed(out, **arrays)
    with pytest.raises(ValueError, match="one row per graph node"):
        read_state_latent(out)


def test_a_foreign_npz_is_refused(tmp_path):
    p = tmp_path / "other.npz"
    np.savez_compressed(
        p, metadata=np.asarray('{"schema": "something_else"}'), t=np.zeros(2)
    )
    with pytest.raises(ValueError, match="expected 'state_latent'"):
        read_state_latent(p)
