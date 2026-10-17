"""The shipped tool + gripper configs hold the contracts code relies on.

The YAMLs under config/gripper/ are DATA the eef stage runs
on, so this suite loads every one through the same code paths the pipeline
uses: the tool_v1 / chopstick_unit_v1 loaders, the strict per-estimator session
check, the estimator registry, the normalization endpoints, and the
tool<->unit pairing. (The SE(3) validity of the mount legs is pinned in
tests/core/test_tool_config.py, on the loaders that read them.)

It also pins the config layering on the files themselves - design facts only in
the tool file, build facts only in the unit file, every unit carrying its own
tag AND its own calibrated pose.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiwi.core.tool_config import load_gripper_unit, load_tool_config, merge_session
from kiwi.eef import ESTIMATOR_KEYS, get_estimator
from kiwi.eef.config import config_from_block, resolve_params, validate_session_block

REPO = Path(__file__).resolve().parents[2]
TOOLS = sorted((REPO / "config" / "gripper").glob("*/tool.yaml"))
UNITS = sorted((REPO / "config" / "gripper").glob("*/units/*.yaml"))


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _tool_yaml(name: str) -> Path:
    return REPO / "config" / "gripper" / name / "tool.yaml"


def test_the_expected_files_ship():
    assert [p.parent.name for p in TOOLS] == ["chopstick"]
    assert len(UNITS) == 2


@pytest.mark.parametrize("path", TOOLS, ids=lambda p: p.parent.name)
def test_tool_yaml_loads_through_the_real_code_paths(path):
    tc = load_tool_config(path)
    raw = _raw(path)
    # The design's identity is its directory; the name: key must agree.
    assert raw["name"] == path.parent.name
    assert tc.name == path.parent.name
    assert tc.estimator in ESTIMATOR_KEYS
    bundle = get_estimator(tc.estimator, source=str(path))
    assert tc.marker_dictionary == "DICT_4X4_50"
    # The replay mapping is a design fact and every shipped tool states it.
    assert tc.replay_gripper and tc.replay_description

    # The recipe must survive the strict session-key check and resolve to
    # concrete params.
    validate_session_block(tc.estimator, tc.session, str(path))
    params = resolve_params(config_from_block(tc.session, source=str(path)))
    assert params.polarity == "red"

    # The endpoints live in session: - the export reads them off the merged
    # recipe, so a tool without them cannot normalize.
    closed, opened = bundle.closed_open(tc.session)
    assert opened > closed


@pytest.mark.parametrize("path", TOOLS, ids=lambda p: p.parent.name)
def test_tool_files_state_design_facts_only(path):
    """The tool file is the DESIGN layer: no build fact may appear in it."""
    raw = _raw(path)
    assert set(raw) <= {"schema", "name", "description", "design", "session"}
    assert set(raw["design"]) <= {"estimator", "capture", "replay", "marker"}
    assert set(raw["design"]["marker"]) == {"dictionary"}  # id/size are build facts
    assert "fixed_pose" not in raw and "fixed_pose" not in raw["design"]


@pytest.mark.parametrize("path", UNITS, ids=lambda p: p.stem)
def test_unit_files_pair_with_their_tools(path):
    gu = load_gripper_unit(path)
    tool_yaml = _tool_yaml(gu.tool)
    assert tool_yaml.is_file(), f"{path.name} names unknown tool {gu.tool!r}"
    tc = load_tool_config(tool_yaml)

    # Build facts: the sticker is one physical object, so id and size live
    # together and both are required - a size matching the design stock is
    # still stated here, because the pose calibration is bound to it.
    raw_marker = _raw(path)["build"]["marker"]
    assert set(raw_marker) == {"id", "size"}
    assert isinstance(gu.marker_id, int) and gu.marker_size == 0.014

    # tool_v1 has no version-wide pose: every build carries its own, either
    # calibrated on it or explicitly SEEDED from a sibling (see the file).
    fp = gu.fixed_pose
    assert fp is not None, f"{path.name} carries no build.fixed_pose"
    assert set(fp) == {"rvec", "tvec", "fov_deg", "out_size", "lens"}
    assert len(fp["rvec"]) == 3 and len(fp["tvec"]) == 3
    # The pose is only valid at the rectifier it was solved under.
    assert fp["fov_deg"] == tc.session["fov_deg"]

    # This unit's session overrides must survive the same strict layer, and
    # the merged ladder must still resolve and still declare the endpoints.
    validate_session_block(tc.estimator, gu.session, str(path))
    merged = merge_session(tc.session, gu.session)
    resolve_params(config_from_block(merged, source=str(path)))
    closed, opened = get_estimator(tc.estimator).closed_open(merged)
    assert opened > closed
