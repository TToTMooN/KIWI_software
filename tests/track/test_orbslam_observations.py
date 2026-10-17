"""The observation rows: nine columns per frame, written and read back."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.track.orbslam.observations import (
    TRACKED,
    Observations,
    read_localization_rows,
    read_observations,
    write_observations,
)


def _observations():
    """Three frames: tracked, lost, tracked."""
    T = np.tile(np.eye(4), (3, 1, 1))
    T[0, :3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    T[0, :3, 3] = [0.1, 0.2, 0.3]
    T[2, :3, :3] = Rotation.from_euler("xyz", [-0.4, 0.5, 0.6]).as_matrix()
    T[2, :3, 3] = [-1.0, 2.0, 0.5]
    return Observations(
        t_ns=np.array([0, 33_333_333, 66_666_666], dtype=np.int64),
        state=np.array([TRACKED, 0, TRACKED]),
        T=T,
    )


def test_observations_round_trip_in_nine_columns(tmp_path):
    obs = _observations()
    path = tmp_path / "scene" / "front_left_observations.txt"
    write_observations(path, obs)
    lines = path.read_text().splitlines()
    assert [len(line.split()) for line in lines] == [9, 9, 9]
    assert lines[1] == "33333333 0 0 0 0 0 0 0 0"
    back = read_observations(path)
    assert back.t_ns.dtype == np.int64
    np.testing.assert_array_equal(back.t_ns, obs.t_ns)
    np.testing.assert_array_equal(back.tracked, [True, False, True])
    np.testing.assert_allclose(back.T, obs.T, atol=1e-8)
    np.testing.assert_array_equal(back.t_s, obs.t_ns.astype(float) * 1e-9)


def test_localization_rows_are_the_columns_of_the_file(tmp_path):
    obs = _observations()
    path = tmp_path / "rows.txt"
    write_observations(path, obs)
    rows = read_localization_rows(path)
    assert len(rows) == 3
    assert rows.t_ns.dtype == np.int64 and rows.state.dtype == np.int64
    np.testing.assert_array_equal(rows.t_ns, obs.t_ns)
    np.testing.assert_array_equal(rows.state, [2, 0, 2])
    np.testing.assert_array_equal(rows.tracked, [True, False, True])
    np.testing.assert_allclose(rows.p, [[0.1, 0.2, 0.3], [0, 0, 0], [-1.0, 2.0, 0.5]])
    np.testing.assert_allclose(
        rows.q_xyzw[[0, 2]],
        Rotation.from_matrix(obs.T[[0, 2], :3, :3]).as_quat(),
        atol=1e-9,
    )
    np.testing.assert_array_equal(rows.q_xyzw[1], [0, 0, 0, 0])


def test_a_tenth_column_is_not_read(tmp_path):
    nine, ten = tmp_path / "nine.txt", tmp_path / "ten.txt"
    nine.write_text("0 2 0.1 0.2 0.3 0 0 0 1\n33333333 0 0 0 0 0 0 0 0\n")
    ten.write_text("0 2 0.1 0.2 0.3 0 0 0 1 0\n33333333 0 0 0 0 0 0 0 0 -1\n")
    a, b = read_localization_rows(nine), read_localization_rows(ten)
    for field in ("t_ns", "state", "p", "q_xyzw", "tracked"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(read_observations(nine).T, read_observations(ten).T)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "rows.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        read_localization_rows(path)
