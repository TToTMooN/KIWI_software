"""``replay_state.npz`` - the stable key set and the TCP guards.

Replay tools outside this repository read this file in another environment,
so the key set and dtypes are a contract: pinned here key by key.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode
from kiwi.export.replay_state import write_replay_state
from kiwi.export.schema import arm_state_names

NAMES = arm_state_names("left") + arm_state_names("right")


def _episode(n=6, *, measured=True):
    """An episode with meta fabricated directly, not built by ``build_episode``."""
    state = np.arange(n * 16, dtype=np.float32).reshape(n, 16)
    feats = {
        "observation.state": FeatureColumn(
            "observation.state", "float32", (16,), list(NAMES), state
        )
    }
    meta = {"ee_frame": "tcp", "arms": ["left", "right"]}
    meta["tcp"] = {
        "left": {"measured": measured, "t_camera_tcp_xyz": [0.0, 0.03, 0.17]},
        "right": {"measured": measured, "t_camera_tcp_xyz": [0.0, 0.03, 0.17]},
    }
    return ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task="stack",
        frame="marker",
        features=feats,
        meta=meta,
    )


def test_npz_carries_the_stable_key_set(tmp_path):
    out = tmp_path / "replay_state.npz"
    result = write_replay_state(_episode(), out)
    assert result.path == out and result.frames == 6 and result.dims == 16
    npz = np.load(out)
    assert set(npz.files) == {
        "state",
        "names",
        "timestamps",
        "fps",
        "frame",
        "world_scene",
        "ee_frame",
        "task",
        "tcp_measured",
        "tcp",
    }
    assert npz["state"].dtype == np.float32 and npz["state"].shape == (6, 16)
    assert list(npz["names"]) == NAMES
    assert npz["timestamps"].dtype == np.float64
    assert float(npz["fps"]) == 30.0
    assert str(npz["frame"]) == "marker"
    # The key that makes `frame` an identity: episode meta without a
    # world_scene block dumps null, and replay compares on the frame alone.
    assert json.loads(str(npz["world_scene"])) is None
    assert str(npz["ee_frame"]) == "tcp"
    assert str(npz["task"]) == "stack"
    assert bool(npz["tcp_measured"]) is True
    assert json.loads(str(npz["tcp"]))["left"]["measured"] is True


def test_unmeasured_tcp_surfaces_a_warning_not_silence(tmp_path):
    result = write_replay_state(_episode(measured=False), tmp_path / "s.npz")
    assert result.tcp_measured is False
    assert result.warning is not None and "UNMEASURED" in result.warning


def test_assert_tcp_xyz_guards_the_tip(tmp_path):
    ep = _episode()
    write_replay_state(
        ep, tmp_path / "ok.npz", assert_tcp_xyz=(0.0, 0.03, 0.17), assert_tol=0.005
    )
    with pytest.raises(ValueError, match="assert_tcp_xyz failed"):
        write_replay_state(
            ep, tmp_path / "bad.npz", assert_tcp_xyz=(0.0, 0.0, 0.0), assert_tol=0.005
        )
