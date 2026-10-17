"""Supported tracking APIs preserve clocks and units."""

import numpy as np
import pytest

from kiwi import track
from kiwi.core.camera import imu
from kiwi.core.trajectory import Trajectory, load_tum, read_tum_header


def test_tum_api_round_trips_header_and_rows(tmp_path):
    times = np.array([-1.5, -1.25, -0.85])
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.6, 0.8, 0.0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    trajectory = Trajectory(times, positions, quaternions, "map", "camera")
    path = track.write_tum(
        trajectory,
        tmp_path / "camera.tum",
        clock="episode global clock",
        source="poses",
    )
    header = read_tum_header(path)
    assert header.frame == "map" and header.body == "camera"
    assert header.clock == "episode global clock"
    t, p, q, _ = load_tum(path)
    np.testing.assert_allclose(t, times)
    np.testing.assert_allclose(p, positions)
    np.testing.assert_allclose(q, quaternions)


def test_opencv_yaml_reader_accepts_filestorage_directive():
    assert track.load_opencv_yaml("%YAML:1.0\ncamera: front\n") == {"camera": "front"}


def test_recording_input_helpers_preserve_clock_and_units(monkeypatch, tmp_path):
    import telemetry_parser

    class Parser:
        def __init__(self, path):
            pass

        def normalized_imu(self):
            return [
                {
                    "timestamp_ms": -100.0,
                    "gyro": [180.0, 0.0, 0.0],
                    "accl": [0.0, 0.0, 9.81],
                }
            ]

    monkeypatch.setattr(telemetry_parser, "Parser", Parser)
    path = tmp_path / "imu.csv"
    meta = imu.extract_imu_to_csv(tmp_path / "clip.insv", path)
    assert isinstance(meta, imu.ImuMeta)
    assert meta.n_samples == 1
    assert meta.clock_offset_s == pytest.approx(0.1)
    assert imu.compute_clock_offset_ns(-100.0) == 100_000_000
    np.testing.assert_allclose(
        np.loadtxt(path, skiprows=1), [0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 9.81]
    )
