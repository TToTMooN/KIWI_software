"""GR00T ``meta/modality.json`` beside a LeRobot v2.1 export.

Isaac-GR00T fine-tunes on LeRobot v2.1 datasets plus one extra meta file
mapping named modalities onto slices of ``observation.state`` / ``action``,
the video keys, and the task annotation. The slices are derived from the
state column names (:func:`kiwi.export.schema.arm_state_names`), so per arm
the map is mechanical::

    "<label>_ee_pos":  {"start": i,   "end": i+3}
    "<label>_ee_quat": {"start": i+3, "end": i+7, "rotation_type": "quaternion"}
    "<label>_gripper": {"start": i+7, "end": i+8}

GR00T itself is not a dependency - the file is documented structure, and the
structural pin lives in the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import IMAGE_KEY_PREFIX

__all__ = ["build_modality", "state_slices", "write_modality"]


def state_slices(names: list[str]) -> dict[str, dict]:
    """Named modality slices from the state column names.

    Recognizes the schema's per-arm runs (``_ee_x/_ee_y/_ee_z``,
    ``_ee_qx.._ee_qw``, ``_gripper``); any column outside a recognized run
    becomes its own single-column slice under its full name, so nothing is
    ever silently dropped from the map.
    """
    out: dict[str, dict] = {}
    i = 0
    n = len(names)
    while i < n:
        name = names[i]
        if (
            name.endswith("_ee_x")
            and i + 2 < n
            and names[i + 1].endswith("_ee_y")
            and names[i + 2].endswith("_ee_z")
        ):
            label = name[: -len("_ee_x")]
            out[f"{label}_ee_pos"] = {"start": i, "end": i + 3}
            i += 3
        elif (
            name.endswith("_ee_qx")
            and i + 3 < n
            and names[i + 1].endswith("_ee_qy")
            and names[i + 2].endswith("_ee_qz")
            and names[i + 3].endswith("_ee_qw")
        ):
            label = name[: -len("_ee_qx")]
            out[f"{label}_ee_quat"] = {
                "start": i,
                "end": i + 4,
                "rotation_type": "quaternion",
            }
            i += 4
        elif name.endswith("_gripper"):
            out[name] = {"start": i, "end": i + 1}
            i += 1
        else:
            out[name] = {"start": i, "end": i + 1}
            i += 1
    return out


def build_modality(state_names: list[str], video_keys: list[str]) -> dict:
    """The full ``modality.json`` document.

    ``action`` shares ``observation.state``'s layout (the writers synthesize
    it with the same names), video modalities keep the label part of their
    LeRobot key, and the task annotation points at ``task_index`` - GR00T
    resolves the string through ``meta/tasks.jsonl``.
    """
    slices = state_slices(state_names)
    prefix = IMAGE_KEY_PREFIX
    return {
        "state": slices,
        "action": slices,
        "video": {
            (k[len(prefix) :] if k.startswith(prefix) else k): {"original_key": k}
            for k in video_keys
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def write_modality(
    dataset_dir: Path, state_names: list[str], video_keys: list[str]
) -> Path:
    """Write ``meta/modality.json`` into an exported v2.1 dataset dir."""
    doc = build_modality(state_names, video_keys)
    p = Path(dataset_dir) / "meta" / "modality.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p
