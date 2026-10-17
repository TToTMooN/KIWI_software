"""Tests for the LeRobot info.json feature-schema mapping.

The mapping lives in ``kiwi.export.writers.info``; the stable NAMES live in
``kiwi.export.schema``.
"""

from __future__ import annotations

from kiwi.core.observation_episode import VIDEO_DTYPE, FeatureColumn
from kiwi.export import schema
from kiwi.export.writers import info as info_schema


def test_tabular_feature_carries_fps():
    fc = FeatureColumn("observation.state", "float32", (8,), ["a"] * 8)
    info = info_schema.feature_to_info(fc, 30.0)
    assert info == {"dtype": "float32", "shape": [8], "names": ["a"] * 8, "fps": 30.0}


def test_video_feature_carries_video_info_not_fps():
    fc = FeatureColumn("observation.images.head", VIDEO_DTYPE, (480, 640, 3), None)
    info = info_schema.feature_to_info(fc, 29.0)
    assert info["dtype"] == "video"
    assert info["shape"] == [480, 640, 3]
    assert info["names"] == ["height", "width", "channels"]
    assert "fps" not in info
    assert info["video_info"]["video.fps"] == 29.0
    assert info["video_info"]["video.codec"] == "hevc"
    assert info["video_info"]["video.pix_fmt"] == "yuv420p"


def test_build_info_features_order_and_bookkeeping():
    state = FeatureColumn("observation.state", "float32", (8,), ["a"] * 8)
    video = FeatureColumn("observation.images.head", VIDEO_DTYPE, (4, 6, 3), None)
    feats = {"observation.state": state, "observation.images.head": video}
    out = info_schema.build_info_features(
        feats, fps=30.0, action=info_schema.action_feature(state)
    )

    keys = list(out)
    assert keys[0] == "observation.state"
    assert keys[1] == "action"
    assert keys[2:7] == [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    assert keys[-1] == "observation.images.head"
    assert out["action"]["shape"] == [8]
    assert out["timestamp"] == {
        "dtype": "float32",
        "shape": [1],
        "names": None,
        "fps": 30.0,
    }
    assert out["task_index"]["dtype"] == "int64"


def test_arm_state_names_are_stable():
    """Tools outside this repository read these exact strings."""
    assert schema.arm_state_names("left") == [
        "left_ee_x",
        "left_ee_y",
        "left_ee_z",
        "left_ee_qx",
        "left_ee_qy",
        "left_ee_qz",
        "left_ee_qw",
        "left_gripper",
    ]


def test_relative_state_names_are_stable():
    assert schema.RELATIVE_STATE_KEY == "observation.state.relative"
    assert schema.arm_relative_names("left")[:2] == ["left_rel_ee_x", "left_rel_ee_y"]
    assert len(schema.arm_relative_names("left")) == 7
