"""GR00T ``meta/modality.json``: the state-slice map must match the layout exactly.

A wrong slice trains a policy on the wrong channels without any error, so the
map is derived from the same ``schema.arm_state_names`` the builder uses and
pinned here column by column.
"""

from __future__ import annotations

import json

from kiwi.export.schema import arm_state_names
from kiwi.export.writers.groot_modality import (
    build_modality,
    state_slices,
    write_modality,
)

NAMES = arm_state_names("left") + arm_state_names("right")


def test_state_slices_match_the_bimanual_layout():
    slices = state_slices(NAMES)
    assert slices == {
        "left_ee_pos": {"start": 0, "end": 3},
        "left_ee_quat": {"start": 3, "end": 7, "rotation_type": "quaternion"},
        "left_gripper": {"start": 7, "end": 8},
        "right_ee_pos": {"start": 8, "end": 11},
        "right_ee_quat": {"start": 11, "end": 15, "rotation_type": "quaternion"},
        "right_gripper": {"start": 15, "end": 16},
    }


def test_unrecognized_columns_are_kept_not_dropped():
    slices = state_slices(["odd_column"] + arm_state_names("left"))
    assert slices["odd_column"] == {"start": 0, "end": 1}
    assert slices["left_ee_pos"] == {"start": 1, "end": 4}


def test_build_modality_video_and_annotation():
    doc = build_modality(
        NAMES,
        ["observation.images.left_wrist", "observation.images.ego"],
    )
    assert doc["action"] == doc["state"]  # action shares the state layout
    assert doc["video"] == {
        "left_wrist": {"original_key": "observation.images.left_wrist"},
        "ego": {"original_key": "observation.images.ego"},
    }
    assert doc["annotation"] == {
        "human.action.task_description": {"original_key": "task_index"}
    }


def test_write_modality_lands_in_meta(tmp_path):
    p = write_modality(tmp_path, NAMES, ["observation.images.left_wrist"])
    assert p == tmp_path / "meta" / "modality.json"
    doc = json.loads(p.read_text())
    assert doc["state"]["left_ee_pos"] == {"start": 0, "end": 3}
