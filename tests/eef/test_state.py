"""Tests for kiwi.eef.state - the eef_state_v1 npz contract.

Covers the round trip through every frame shape the estimator emits (clean
both, one-jaw, gate-rejected, post-pass-revoked, dropout), the loud
failures, and completion provenance.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kiwi.core import layout
from kiwi.eef.completion import complete_series
from kiwi.eef.state import (
    EEF_STATE_SCHEMA,
    EEF_STATE_SUFFIX,
    OVERLAY_SUFFIX,
    TRACE_SUFFIX,
    EefState,
    read_eef_state,
    write_eef_state,
)

NAN = np.nan


def test_completed_artifact_round_trip_preserves_raw_and_fill_audit(tmp_path):
    state = _state()
    completed, audit, _ = complete_series(state.value, state.fps)
    result = read_eef_state(write_eef_state(tmp_path / "completed.npz", state))
    np.testing.assert_array_equal(result.value, state.value)
    np.testing.assert_array_equal(result.output, completed)
    assert result.valid_output.all()
    for key in audit:
        np.testing.assert_array_equal(result.completion[key], audit[key])
    with np.load(tmp_path / "completed.npz", allow_pickle=False) as data:
        assert "completion/gap_duration_s" in data.files
        assert "chopsticks/gap_duration_s" not in data.files


def test_an_artifact_without_completed_output_is_refused(tmp_path):
    path = write_eef_state(tmp_path / "c_eef.npz", _state())
    with np.load(path) as data:
        payload = {
            key: data[key]
            for key in data.files
            if key != "value_completed" and not key.startswith("completion/")
        }
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="no value_completed") as e:
        read_eef_state(path)
    assert str(path) in str(e.value)
    assert "re-run the eef stage" in str(e.value)


def test_completed_artifact_refuses_inconsistent_values_or_fill_mask():
    raw = _state()
    completed, audit, _ = complete_series(raw.value, raw.fps)
    bad = completed.copy()
    bad[0] += 0.1
    with pytest.raises(ValueError, match="preserve every finite raw"):
        _state(value_completed=bad, completion=audit)
    with pytest.raises(ValueError, match="one finite value per frame"):
        _state(value_completed=raw.value, completion=audit)
    with pytest.raises(ValueError, match="exactly identify missing raw"):
        _state(
            value_completed=completed, completion={**audit, "filled": np.zeros(7, bool)}
        )


def _state(**over) -> EefState:
    """Seven representative frame results, including dropouts."""
    value = np.array([42.5, 40.0, NAN, NAN, NAN, 95.0, NAN])
    fps = 119.88011988011988
    completed, audit, _ = complete_series(value, fps)
    base = dict(
        tool="chopsticks-v3-1",
        estimator="chopsticks",
        unit="deg",
        fps=fps,
        source="VID_x_lens0.mp4",
        pose_mode="fixed_pose",
        pose_source="config/gripper/chopstick-v3-1-m5.yaml",
        value=value,
        rejected_value=np.array([NAN, NAN, 55.0, NAN, NAN, NAN, NAN]),
        has_pose=np.ones(7, dtype=bool),
        n_detected=np.array([9, 5, 8, 2, 3, 9, 5], dtype=np.int32),
        n_kept=np.array([8, 4, 7, 1, 1, 8, 4], dtype=np.int32),
        error=np.array(
            [
                "",
                "",
                "asymmetry 41.3deg > 30deg",
                "need >=2 dots on at least one side, got left=1 right=0",
                "need >=2 dots on at least one side, got left=1 right=0",
                "",
                "one-jaw 210deg > 180deg",
            ]
        ),
        detail={
            "mode": np.array(["both", "one_jaw", "both", "", "", "both", "one_jaw"]),
            "left_angle_to_normal_deg": np.array(
                [-21.0, NAN, -6.7, NAN, NAN, -47.0, NAN]
            ),
            "asymmetry_deg": np.array([0.5, NAN, 41.3, NAN, NAN, 1.0, NAN]),
            "delta_deg": np.asarray(0.31),  # a whole-series 0-d result
        },
        value_completed=completed,
        completion=audit,
    )
    base.update(over)
    return EefState(**base)


def test_round_trip_preserves_everything(tmp_path):
    s = _state()
    path = write_eef_state(tmp_path / "clip_eef.npz", s)
    assert path.name == "clip_eef.npz"
    assert not list(tmp_path.glob("*.tmp*"))  # atomic: no tmp file left behind

    r = read_eef_state(path)
    assert (r.tool, r.estimator, r.unit) == (s.tool, s.estimator, s.unit)
    assert r.fps == pytest.approx(s.fps)
    assert (r.source, r.pose_mode, r.pose_source) == (
        s.source,
        s.pose_mode,
        s.pose_source,
    )
    for name in ("value", "rejected_value"):
        np.testing.assert_array_equal(getattr(r, name), getattr(s, name))
    np.testing.assert_array_equal(r.has_pose, s.has_pose)
    np.testing.assert_array_equal(r.n_detected, s.n_detected)
    np.testing.assert_array_equal(r.n_kept, s.n_kept)
    assert list(r.error) == list(s.error)
    assert set(r.detail) == set(s.detail)
    np.testing.assert_array_equal(r.detail["mode"], s.detail["mode"])
    np.testing.assert_array_equal(r.detail["asymmetry_deg"], s.detail["asymmetry_deg"])
    assert float(r.detail["delta_deg"]) == pytest.approx(0.31)
    assert len(r) == 7


def test_validity_is_isnan_and_nothing_else(tmp_path):
    r = read_eef_state(write_eef_state(tmp_path / "c_eef.npz", _state()))
    np.testing.assert_array_equal(
        r.valid, [True, True, False, False, False, True, False]
    )
    # Frame 6 is the revoked one-jaw row: no error-based shortcut may
    # call it valid, and its diagnostics are still there.
    assert not r.valid[6] and r.detail["mode"][6] == "one_jaw"


def test_detail_keys_are_namespaced_on_disk(tmp_path):
    path = write_eef_state(tmp_path / "c_eef.npz", _state())
    with np.load(path) as data:
        assert "chopsticks/mode" in data.files
        assert "mode" not in data.files  # never in the generic namespace
        assert str(data["schema"][()]) == EEF_STATE_SCHEMA


def test_missing_file_and_foreign_schema_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="eef state"):
        read_eef_state(tmp_path / "nope_eef.npz")
    foreign = tmp_path / "other_eef.npz"
    np.savez(foreign, schema=np.asarray("imu_raw_v1"), data=np.zeros(3))
    with pytest.raises(ValueError, match="schema"):
        read_eef_state(foreign)


def test_post_init_validation():
    with pytest.raises(ValueError, match="pose_mode"):
        _state(pose_mode="live")
    with pytest.raises(ValueError, match="fps"):
        _state(fps=0.0)
    with pytest.raises(ValueError, match="length-7"):
        _state(n_kept=np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError, match="per-frame shaped"):
        _state(detail={"mode": np.array(["both"] * 3)})
    with pytest.raises(ValueError, match="0-d or 1-d"):
        _state(detail={"tilts": np.zeros((7, 2))})


def test_suffixes_are_only_the_standalone_convention():
    """Inside an episode the triplet takes layout's FIXED names in the arm's
    stream directory; these suffixes describe only a standalone run on a bare
    clip, where the stem is the namespace. Same three extensions either way.
    """
    rec = Path("/clips/VID_001_left.insv")
    assert rec.with_name(rec.stem + EEF_STATE_SUFFIX).name == "VID_001_left_eef.npz"
    assert (EEF_STATE_SUFFIX, TRACE_SUFFIX, OVERLAY_SUFFIX) == (
        "_eef.npz",
        "_eef.png",
        "_eef.mp4",
    )
    assert [
        Path(s).suffix for s in (EEF_STATE_SUFFIX, TRACE_SUFFIX, OVERLAY_SUFFIX)
    ] == [
        Path(n).suffix for n in (layout.EEF_STATE, layout.EEF_TRACE, layout.EEF_OVERLAY)
    ]
