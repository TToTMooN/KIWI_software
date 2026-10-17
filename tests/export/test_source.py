"""The input spec stays plain - the DAG seam, enforced.

`kiwi.export.source` is the flattened form of the manifest that the CLI (the
composition root) hands the builder. Two properties keep the boundary honest:
the module imports nothing from `kiwi` except `kiwi.core.layout` (artifact
names and path helpers: strings and Paths, never types), so no episode type
can leak into a field; and the artifact-path convention export expects from the
state and register stages is pinned.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from kiwi.export import source
from kiwi.export.source import (
    ArmSource,
    EpisodeSource,
    register_tum_path,
    state_tum_path,
)


# The one reach into kiwi the spec is allowed: the layout module's artifact
# NAME constants, so `<label>/scene/state.tum` is spelled in exactly one place.
# It defines strings, not types - everything else stays out.
ALLOWED_KIWI_IMPORTS = {"kiwi.core.layout"}


def test_source_module_imports_no_kiwi_types():
    """No import path can smuggle an episode type into the spec."""
    tree = ast.parse(Path(inspect.getfile(source)).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            assert not name.startswith("kiwi") or name in ALLOWED_KIWI_IMPORTS, (
                f"kiwi.export.source imports {name!r}; the spec must stay "
                f"plain data - only {sorted(ALLOWED_KIWI_IMPORTS)} may be reached"
            )


def test_the_world_tum_paths_are_the_stream_scoped_convention():
    assert state_tum_path(Path("/ep"), "left") == Path("/ep/left/scene/state.tum")
    assert register_tum_path(Path("/ep"), "left") == Path("/ep/left/register.tum")


def test_out_label_defaults_main_for_unlabelled_arms():
    arm = ArmSource(
        label="",
        lens0=Path("a.mp4"),
        state_tum=Path("scene/state.tum"),
        eef_state=Path("a_eef.npz"),
        sync_off=0.0,
        session_block={},
        tool="chopsticks-v3",
    )
    assert arm.out_label == "main"


def test_the_layered_recipe_field_is_spelled_session_block():
    """The spec calls the layered recipe what the YAML calls it: `session:`.

    The builder reads the normalization endpoints straight out of this
    mapping, so the field name is part of the seam.
    """
    names = {f.name for f in dataclasses.fields(ArmSource)}
    assert "session_block" in names


def test_specs_are_frozen():
    """Immutable by design: the CLI resolves once; nothing downstream mutates."""
    src = EpisodeSource(name="e", episode_dir=Path("/ep"), arms=(), task="do the thing")
    with pytest.raises(dataclasses.FrozenInstanceError):
        src.name = "other"


def test_the_world_source_is_named_never_inferred():
    from kiwi.export.source import WORLD_SOURCES

    assert WORLD_SOURCES == ("marker", "markerless")
