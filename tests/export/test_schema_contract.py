"""Rule 6 has teeth here: the public field names are frozen, in one place.

``kiwi/export/schema.py`` declares the names external consumers build
against. That promise needs enforcement in both directions, and this file is
it:

* the spellings themselves are pinned below as string literals, so renaming
  a schema constant fails loudly - changing a line HERE is the schema
  version bump CONTRIBUTING rule 6 requires discussing first;
* an AST guard bans ``observation.*`` string literals everywhere else in
  ``kiwi/export/``, so a writer can never drift from the schema by re-typing
  a name (an emitter spelling ``left_head`` while the schema promises
  ``left_wrist`` would otherwise go unnoticed);
* an end-to-end write pins that the names land in the artifact bytes - the
  parquet columns - not just in Python-level dicts.

``action`` is deliberately absent from the literal ban: GR00T's
``modality.json`` has a top-level section that is spelled ``"action"`` by
GR00T's own format, not by our schema. ``ACTION_KEY`` is held by the frozen
pin and the end-to-end write instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kiwi.core.observation_episode import FeatureColumn, ObservationEpisode
from kiwi.export import schema
from kiwi.export.writers import DatasetInfo, get_writer

EXPORT_DIR = Path(schema.__file__).resolve().parent


def test_public_names_frozen():
    assert schema.SCHEMA_VERSION == "kiwi_export_v1"
    assert schema.STATE_KEY == "observation.state"
    assert schema.ACTION_KEY == "action"
    assert schema.IMAGE_KEY_PREFIX == "observation.images."
    assert schema.wrist_image_key("left") == "observation.images.left_wrist"
    assert schema.RELATIVE_STATE_KEY == "observation.state.relative"
    assert schema.CAMERA_POSE_KEY_PREFIX == "observation.camera_pose."
    assert schema.camera_pose_key("left") == "observation.camera_pose.left"
    assert schema.POSE_GAP_KEY_PREFIX == "observation.pose_gap."
    assert schema.pose_gap_key("left") == "observation.pose_gap.left"
    assert schema.EE_AXES == (
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_qx",
        "ee_qy",
        "ee_qz",
        "ee_qw",
    )
    assert schema.CAMERA_HEIGHT_KEY_PREFIX == "observation.camera_height."
    assert schema.GROUND_PLANE_KEY == "observation.ground_plane"
    assert schema.HEAD_MARKER_POSE_KEY == "observation.head_marker_pose"
    assert schema.HEAD_MARKER_VALID_KEY == "observation.head_marker_valid"
    assert schema.HEAD_MARKER_SEGMENT_ID_KEY == "observation.head_marker_segment_id"


def _docstring_ids(tree: ast.Module) -> set[int]:
    """ids of Constant nodes that are module/class/function docstrings."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def test_no_observation_literals_outside_schema():
    offenders = []
    for path in sorted(EXPORT_DIR.rglob("*.py")):
        if path == EXPORT_DIR / "schema.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "observation." in node.value
                and id(node) not in docstrings
            ):
                offenders.append(
                    f"{path.relative_to(EXPORT_DIR.parent.parent)}:{node.lineno}: "
                    f"{node.value!r}"
                )
    assert not offenders, (
        "field names are spelled in kiwi/export/schema.py ONLY - import the "
        "constant instead:\n" + "\n".join(offenders)
    )


def test_v30_write_emits_the_public_spellings(tmp_path):
    # The mutation this guards against: renaming the writer's action column
    # changes the emitted dataset, and neither the frozen pin nor the literal
    # ban notices. The expected column names are deliberately literals, not
    # schema constants - this test must disagree with a rename on EITHER side.
    n, dim = 6, 4
    state = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    features = {
        schema.STATE_KEY: FeatureColumn(
            schema.STATE_KEY, "float32", (dim,), [f"s{i}" for i in range(dim)], state
        )
    }
    episode = ObservationEpisode(
        fps=30.0,
        timestamps=np.arange(n) / 30.0,
        task="pick",
        frame="marker",
        features=features,
    )
    w = get_writer("v3.0")
    w.open(tmp_path, DatasetInfo("me/x", "kiwi", 30.0, features))
    w.add_episode(episode, 0, 0)
    w.close()
    shards = list((tmp_path / "data").rglob("*.parquet"))
    assert shards, "v3.0 writer produced no data shard"
    cols = set(pq.read_table(shards[0]).column_names)
    assert {"observation.state", "action"} <= cols, cols
