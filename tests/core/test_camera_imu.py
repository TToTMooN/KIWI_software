"""The shared camera IMU extraction preserves units and recording clocks."""

import numpy as np
import pytest

from kiwi.core.camera import imu as imu_mod


def test_compute_clock_offset_ns_positive_for_negative_preroll():
    # IMU pre-roll ~700 ms before video t=0 -> positive offset.
    first_ts_ms = -700.123456
    got = imu_mod.compute_clock_offset_ns(first_ts_ms)
    first_t_ns = int(round(first_ts_ms * 1_000_000))
    assert got == -min(0, first_t_ns)
    assert got > 0


def test_compute_clock_offset_ns_zero_when_already_nonnegative():
    assert imu_mod.compute_clock_offset_ns(0.0) == 0
    assert imu_mod.compute_clock_offset_ns(12.5) == 0


def _fake_parser(monkeypatch, samples):
    import telemetry_parser

    class _FakeParser:
        def __init__(self, _path: str) -> None:
            pass

        def normalized_imu(self):
            return samples

    monkeypatch.setattr(telemetry_parser, "Parser", _FakeParser)


def test_extract_imu_units_and_rounding(tmp_path, monkeypatch):
    _fake_parser(
        monkeypatch,
        [
            {
                "timestamp_ms": -100.0,
                "gyro": [180.0, 0.0, -90.0],
                "accl": [0.0, 0.0, 9.81],
            },
            {"timestamp_ms": 0.0, "gyro": [0.0, 45.0, 0.0], "accl": [1.0, -2.0, 3.0]},
        ],
    )
    out = tmp_path / "imu.csv"
    meta = imu_mod.extract_imu_to_csv(tmp_path / "dummy.insv", out)
    rows = np.loadtxt(out, skiprows=1)

    assert meta.n_samples == 2
    assert meta.clock_offset_s == pytest.approx(0.1)
    # First sample lands at t=0 after the +100 ms shift.
    assert rows[0, 0] == pytest.approx(0.0, abs=1e-12)
    # gyro deg/s -> rad/s
    assert rows[0, 1] == pytest.approx(np.pi)
    assert rows[0, 3] == pytest.approx(-np.pi / 2)
    assert rows[1, 2] == pytest.approx(np.pi / 4)
    # accel unchanged
    np.testing.assert_allclose(rows[0, 4:7], [0.0, 0.0, 9.81])
    np.testing.assert_allclose(rows[1, 4:7], [1.0, -2.0, 3.0])


def test_extract_imu_no_samples_is_a_runtime_error(tmp_path, monkeypatch):
    """Zero samples raise RuntimeError, never SystemExit: inside `kiwi run` a
    SystemExit kills the process instead of producing StageResult(ok=False)."""
    _fake_parser(monkeypatch, [])
    with pytest.raises(RuntimeError, match="no samples"):
        imu_mod.extract_imu_to_csv(tmp_path / "dummy.insv", tmp_path / "imu.csv")
