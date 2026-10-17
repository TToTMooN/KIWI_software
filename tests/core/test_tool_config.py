"""Tests for the tool_v1 / chopstick_unit_v1 config loaders (yaml only, cv2-free).

The config schemas promise LAYERING: every key lives in the block
named for its variance class, and each block has a ceiling - the highest layer
allowed to state it. Nothing but the loader enforces that, so the rejections
here are as load-bearing as the happy paths. A fact restated in the wrong file,
an unknown key, or a mount leg pasted under the other leg's key all produce a
plausible-looking config that composes silently wrong geometry; each one must
fail at load, and a build fact placed under design: names the block it
belongs to.
"""

from __future__ import annotations

import numpy as np
import pytest

from kiwi.core import paths
from kiwi.core.tool_config import (
    GRIPPER_UNIT_SCHEMA,
    TOOL_SCHEMA,
    find_gripper_unit,
    gripper_unit_config_path,
    load_gripper_unit,
    load_T_mount_cam,
    load_T_mount_tcp,
    load_tool_config,
    merge_session,
    tool_config_path,
)

# The smallest legal pair: everything the loaders require and nothing else.
TOOL_MIN = f"""\
schema: {TOOL_SCHEMA}
design:
  estimator: chopsticks
  marker:
    dictionary: DICT_4X4_50
"""

TOOL_FULL = f"""\
schema: {TOOL_SCHEMA}
name: chopsticks-v3
description: the v3 chopstick design
design:
  estimator: chopsticks
  replay:
    gripper: chopstick_sa
    description: fr3_chopstick_sa
  marker:
    dictionary: DICT_4X4_50
session:
  polarity: red
  angle_closed_deg: 0.0
  angle_open_deg: 80.0
"""

UNIT_MIN = f"""\
schema: {GRIPPER_UNIT_SCHEMA}
tool: chopsticks-v3
build:
  marker:
    id: 4
    size: 0.014
"""

# The fully populated build: a calibrated pose, both description legs
# overridden (the remeasured unit), and this unit's recipe overrides.
UNIT_FULL = f"""\
schema: {GRIPPER_UNIT_SCHEMA}
description: the m5 build
tool: chopsticks-v3
build:
  marker:
    id: 5
    size: 0.0195
  fixed_pose:
    rvec: [0.1, 0.2, 0.3]
    tvec: [0.01, 0.02, 0.03]
    fov_deg: 160.0
    out_size: [1920, 1920]
    lens: front
  mount_tcp:
    measured: true
    T_mount_tcp: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.195], [0, 0, 0, 1]]
  mount_cam:
    T_mount_cam: [[1, 0, 0, 0.01], [0, 1, 0, 0.02], [0, 0, 1, 0.03], [0, 0, 0, 1]]
session:
  angle_closed_deg: 2.0
  angle_open_deg: 78.0
"""

# One well-formed leg per loader, plus the OTHER leg's matrix key - enough to
# drive the shared validator, and the mis-paste guard, from both sides.
MOUNT_LEGS = [
    pytest.param(load_T_mount_tcp, "T_mount_tcp", "T_mount_cam", id="mount_tcp"),
    pytest.param(load_T_mount_cam, "T_mount_cam", "T_mount_tcp", id="mount_cam"),
]


def _write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# -- path builders ----------------------------------------------------------


def test_path_builders_use_config_dir(tmp_path):
    assert (
        gripper_unit_config_path("g1", "chopsticks-v3", tmp_path)
        == tmp_path / "gripper" / "chopsticks-v3" / "units" / "g1.yaml"
    )
    assert (
        tool_config_path("chopsticks-v3", tmp_path)
        == tmp_path / "gripper" / "chopsticks-v3" / "tool.yaml"
    )


def test_path_builders_default_to_config_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.CONFIG_ENV, str(tmp_path))
    assert (
        gripper_unit_config_path("g1", "t")
        == tmp_path / "gripper" / "t" / "units" / "g1.yaml"
    )
    assert tool_config_path("t") == tmp_path / "gripper" / "t" / "tool.yaml"


def test_find_gripper_unit_scans_every_designs_units_dir(tmp_path):
    p = gripper_unit_config_path("g1", "t1", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(UNIT_MIN)
    assert find_gripper_unit("g1", tmp_path) == p


def test_find_gripper_unit_missing_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing matches"):
        find_gripper_unit("g1", tmp_path)


def test_find_gripper_unit_refuses_a_duplicate_id(tmp_path):
    """Unit ids are global vocabulary (manifests reference them bare), so the
    same id under two designs is a config bug, not a tie to break."""
    for tool in ("t1", "t2"):
        p = gripper_unit_config_path("g1", tool, tmp_path)
        p.parent.mkdir(parents=True)
        p.write_text(UNIT_MIN)
    with pytest.raises(ValueError, match="ambiguous"):
        find_gripper_unit("g1", tmp_path)


def test_load_gripper_unit_rejects_a_file_under_the_wrong_design(tmp_path):
    """A unit under design A declaring `tool:` B is a miscopy; the registry
    layout makes the parent dir authoritative, so the load refuses."""
    p = gripper_unit_config_path("g1", "not-chopsticks-v3", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(UNIT_MIN)  # declares tool: chopsticks-v3
    with pytest.raises(ValueError, match="lives under design"):
        load_gripper_unit(p)


# -- the tool file: design: facts, frozen ------------------------------------


def test_load_tool_config_flattens_the_design_block(tmp_path):
    """The design: block is the whole point of the tool file - which estimator
    bundle reads the tool, which tag family it wears, and the robot side it
    replays onto. All of it must survive the read flattened, with the session:
    recipe carried raw beside it."""
    tool = load_tool_config(_write(tmp_path, "chopsticks-v3.yaml", TOOL_FULL))
    assert tool.name == "chopsticks-v3"  # the file stem, not the `name:` field
    assert tool.path.name == "chopsticks-v3.yaml"
    assert tool.estimator == "chopsticks"
    assert tool.marker_dictionary == "DICT_4X4_50"
    assert tool.replay_gripper == "chopstick_sa"
    assert tool.replay_description == "fr3_chopstick_sa"
    assert tool.session == {
        "polarity": "red",
        "angle_closed_deg": 0.0,
        "angle_open_deg": 80.0,
    }


def test_load_tool_config_reads_a_minimal_design(tmp_path):
    """replay: and session: are optional - a design that nothing replays yet
    still resolves an estimator and a dictionary. The absent halves read as
    None / {}, never as a missing attribute."""
    tool = load_tool_config(_write(tmp_path, "t.yaml", TOOL_MIN))
    assert tool.replay_gripper is None
    assert tool.replay_description is None
    assert tool.session == {}


def test_load_tool_config_requires_a_design_block(tmp_path):
    """Without design: there is no estimator, so nothing downstream can pick a
    bundle - refuse at load rather than at the eef stage."""
    f = _write(tmp_path, "t.yaml", f"schema: {TOOL_SCHEMA}\nname: t\n")
    with pytest.raises(ValueError, match="requires a `design:` block"):
        load_tool_config(f)


def test_load_tool_config_requires_an_estimator(tmp_path):
    f = _write(
        tmp_path,
        "t.yaml",
        f"schema: {TOOL_SCHEMA}\ndesign:\n  marker: {{dictionary: DICT_4X4_50}}\n",
    )
    with pytest.raises(ValueError, match="design.estimator is required"):
        load_tool_config(f)


def test_load_tool_config_requires_a_marker_dictionary(tmp_path):
    """The dictionary is design-wide: every build of the design wears the same
    tag family, and a detector cannot be built without it."""
    f = _write(
        tmp_path, "t.yaml", f"schema: {TOOL_SCHEMA}\ndesign:\n  estimator: chopsticks\n"
    )
    with pytest.raises(ValueError, match="design.marker.dictionary is required"):
        load_tool_config(f)


def test_load_tool_config_rejects_an_unknown_schema(tmp_path):
    f = _write(tmp_path, "t.yaml", "schema: something_else_v9\n")
    with pytest.raises(ValueError, match="expected 'tool_v1'"):
        load_tool_config(f)


@pytest.mark.parametrize("key", ["polarity", "estimator", "marker", "fixed_pose"])
def test_load_tool_config_rejects_an_unknown_top_level_key(tmp_path, key):
    """Design facts live under design: and calibrations in a unit file, so
    the same names at the top level are unknown keys, never read."""
    f = _write(tmp_path, "t.yaml", TOOL_MIN + f"{key}: red\n")
    with pytest.raises(ValueError, match=f"{key!r} is not a tool_v1 top-level key"):
        load_tool_config(f)


@pytest.mark.parametrize("key", ["id", "size"])
def test_load_tool_config_rejects_build_facts_in_the_design_marker(tmp_path, key):
    """The tag id and size are BUILD facts - the sticker is one physical object
    on one physical gripper."""
    f = _write(
        tmp_path,
        "t.yaml",
        f"schema: {TOOL_SCHEMA}\n"
        "design:\n"
        "  estimator: chopsticks\n"
        "  marker:\n"
        "    dictionary: DICT_4X4_50\n"
        f"    {key}: 4\n",
    )
    with pytest.raises(ValueError, match=f"gripper unit's build.marker.{key}"):
        load_tool_config(f)


def test_load_tool_config_rejects_a_design_level_fixed_pose(tmp_path):
    """tool_v1 has no version-wide pose: a calibration is bound to one build's
    tag placement, so it can only live in that unit's build.fixed_pose."""
    f = _write(
        tmp_path,
        "t.yaml",
        f"schema: {TOOL_SCHEMA}\n"
        "design:\n"
        "  estimator: chopsticks\n"
        "  marker: {dictionary: DICT_4X4_50}\n"
        "  fixed_pose: {rvec: [0, 0, 0], tvec: [0, 0, 0]}\n",
    )
    with pytest.raises(ValueError, match="build.fixed_pose"):
        load_tool_config(f)


def test_load_tool_config_rejects_an_unknown_replay_key(tmp_path):
    f = _write(
        tmp_path,
        "t.yaml",
        f"schema: {TOOL_SCHEMA}\n"
        "design:\n"
        "  estimator: chopsticks\n"
        "  marker: {dictionary: DICT_4X4_50}\n"
        "  replay: {gripper: chopstick_sa, urdf: nope.urdf}\n",
    )
    with pytest.raises(ValueError, match="not a design.replay key"):
        load_tool_config(f)


def test_load_tool_config_rejects_a_non_mapping_session(tmp_path):
    """A scalar session: is a YAML slip (an omitted nesting level); reading it
    as empty would drop the whole recipe layer."""
    f = _write(tmp_path, "t.yaml", TOOL_MIN + "session: red\n")
    with pytest.raises(ValueError, match="session: must be a mapping"):
        load_tool_config(f)


# -- the unit file: build: facts, one physical gripper -----------------------


def test_load_gripper_unit_flattens_the_build_block(tmp_path):
    """One physical build's facts: its tag, its calibration, its measured mount
    legs, its recipe overrides. Everything the tool file cannot know."""
    unit = load_gripper_unit(_write(tmp_path, "chopstick-v3-m5.yaml", UNIT_FULL))
    assert unit.name == "chopstick-v3-m5"
    assert unit.tool == "chopsticks-v3"
    assert unit.marker_id == 5
    assert unit.marker_size == 0.0195
    assert unit.fixed_pose["rvec"] == [0.1, 0.2, 0.3]
    assert unit.fixed_pose["lens"] == "front"  # the rectifier provenance survives
    assert unit.mount_tcp["measured"] is True
    assert unit.session == {"angle_closed_deg": 2.0, "angle_open_deg": 78.0}


def test_load_gripper_unit_leaves_the_optional_build_blocks_as_none(tmp_path):
    """A unit that overrides neither leg inherits both from the robot
    description's fixed-joint FK, and one that has never been calibrated has no
    pose - so the absent blocks must read as None, not {}: an empty mapping
    would compose as an identity override."""
    unit = load_gripper_unit(_write(tmp_path, "g1.yaml", UNIT_MIN))
    assert unit.marker_id == 4
    assert unit.marker_size == 0.014
    assert unit.fixed_pose is None
    assert unit.mount_tcp is None
    assert unit.mount_cam is None
    assert unit.session == {}


def test_load_gripper_unit_requires_the_tool_link(tmp_path):
    """The unit->design link resolves the parent config and cross-checks the
    episode pairing; without it a build belongs to no design."""
    f = _write(
        tmp_path,
        "g1.yaml",
        f"schema: {GRIPPER_UNIT_SCHEMA}\nbuild:\n  marker: {{id: 4, size: 0.014}}\n",
    )
    with pytest.raises(ValueError, match="`tool:` is required"):
        load_gripper_unit(f)


def test_load_gripper_unit_requires_a_build_block(tmp_path):
    f = _write(
        tmp_path, "g1.yaml", f"schema: {GRIPPER_UNIT_SCHEMA}\ntool: chopsticks-v3\n"
    )
    with pytest.raises(ValueError, match="requires a `build:` block"):
        load_gripper_unit(f)


@pytest.mark.parametrize(
    "marker",
    ["{}", "{size: 0.014}", "{id: '4', size: 0.014}", "{id: true, size: 0.014}"],
    ids=["empty", "no-id", "string-id", "bool-id"],
)
def test_load_gripper_unit_requires_an_integer_marker_id(tmp_path, marker):
    """The id is what the detector filters on - a string or a stray `true`
    would match no tag at all, and every frame would come back empty."""
    f = _write(
        tmp_path,
        "g1.yaml",
        f"schema: {GRIPPER_UNIT_SCHEMA}\ntool: t\nbuild:\n  marker: {marker}\n",
    )
    with pytest.raises(ValueError, match="build.marker.id must be an integer"):
        load_gripper_unit(f)


@pytest.mark.parametrize(
    "size", ["19.5", "0", "-0.014", "'0.014'"], ids=["mm", "zero", "negative", "string"]
)
def test_load_gripper_unit_requires_a_marker_size_in_metres(tmp_path, size):
    """Marker size sets the metric scale of every solvePnP pose off that tag:
    a bad value does not fail, it biases the camera<->marker translation by a
    constant factor. A millimetre value reads as a 19.5 m tag, so the size is
    bounded, not coerced."""
    f = _write(
        tmp_path,
        "g1.yaml",
        f"schema: {GRIPPER_UNIT_SCHEMA}\n"
        "tool: t\n"
        f"build:\n  marker: {{id: 4, size: {size}}}\n",
    )
    with pytest.raises(ValueError, match="build.marker.size"):
        load_gripper_unit(f)


def test_load_gripper_unit_rejects_an_unknown_schema(tmp_path):
    f = _write(tmp_path, "g1.yaml", "schema: something_else_v9\n")
    with pytest.raises(ValueError, match="expected 'chopstick_unit_v1'"):
        load_gripper_unit(f)


@pytest.mark.parametrize("key", ["marker_id", "fixed_pose", "mount_tcp", "gripper"])
def test_load_gripper_unit_rejects_an_unknown_top_level_key(tmp_path, key):
    """Build facts live under build: and the recipe under session:, so the
    same names at the top level are unknown keys - a tolerated one would leave
    its layer silently absent."""
    f = _write(tmp_path, "g1.yaml", UNIT_MIN + f"{key}: 4\n")
    with pytest.raises(
        ValueError, match=f"{key!r} is not a chopstick_unit_v1 top-level key"
    ):
        load_gripper_unit(f)


def test_load_gripper_unit_rejects_an_unknown_build_key(tmp_path):
    """build: is closed too - a design key restated here (or a typo) must not
    ride along as ad-hoc metadata."""
    f = _write(
        tmp_path,
        "g1.yaml",
        f"schema: {GRIPPER_UNIT_SCHEMA}\n"
        "tool: t\n"
        "build:\n"
        "  marker: {id: 4, size: 0.014}\n"
        "  estimator: chopsticks\n",
    )
    with pytest.raises(ValueError, match="not a build: key"):
        load_gripper_unit(f)


@pytest.mark.parametrize(
    ("pose", "match"),
    [
        ("{rvec: [0, 0, 0]}", r"build.fixed_pose.tvec must be a 3-vector"),
        (
            "{rvec: [0, 0], tvec: [0, 0, 0]}",
            r"build.fixed_pose.rvec must be a 3-vector",
        ),
        (
            "{rvec: [0, 0, 0], tvec: [0, 0, 0], tvecs: [0, 0, 0]}",
            "not a build.fixed_pose key",
        ),
    ],
    ids=["missing-tvec", "short-rvec", "unknown-key"],
)
def test_load_gripper_unit_validates_the_fixed_pose_record(tmp_path, pose, match):
    """`kiwi eef calibrate` writes the pose and its rectifier provenance as ONE
    record, so a missing half or a stray key is a hand-edit gone wrong, not a
    version skew - and a half-read pose warps every plane in the clip."""
    f = _write(
        tmp_path,
        "g1.yaml",
        f"schema: {GRIPPER_UNIT_SCHEMA}\n"
        "tool: t\n"
        "build:\n"
        "  marker: {id: 4, size: 0.014}\n"
        f"  fixed_pose: {pose}\n",
    )
    with pytest.raises(ValueError, match=match):
        load_gripper_unit(f)


def test_load_gripper_unit_carries_the_two_mount_override_blocks(tmp_path):
    """A remeasured build overrides the description's legs from its unit file.

    The loader carries the raw blocks (provenance keys included); each leg then
    resolves on its own as description default < unit override, exercised end
    to end in tests/export/. What is load-bearing here is that a present block
    lands verbatim and the two legs are INDEPENDENT - a unit may override the
    tip without touching the camera seating.
    """
    both = load_gripper_unit(_write(tmp_path, "g2.yaml", UNIT_FULL))
    assert both.mount_tcp["measured"] is True  # provenance survives
    np.testing.assert_allclose(
        load_T_mount_tcp(both.mount_tcp)[:3, 3], [0.0, 0.0, 0.195]
    )
    np.testing.assert_allclose(
        load_T_mount_cam(both.mount_cam)[:3, 3], [0.01, 0.02, 0.03]
    )

    one_leg = _write(
        tmp_path,
        "g3.yaml",
        UNIT_MIN + "  mount_cam:\n"
        "    T_mount_cam: [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n",
    )
    partial = load_gripper_unit(one_leg)
    assert partial.mount_cam is not None
    assert partial.mount_tcp is None  # the tip still comes from the description


@pytest.mark.parametrize("leg", ["mount_tcp", "mount_cam"])
def test_load_gripper_unit_validates_the_mount_blocks_eagerly(tmp_path, leg):
    """A malformed mount block must fail at unit load, next to the file that
    carries it."""
    key = f"T_{leg}"
    f = _write(
        tmp_path,
        "g1.yaml",
        UNIT_MIN + f"  {leg}:\n"
        f"    {key}: [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n",
    )
    with pytest.raises(ValueError, match="not orthonormal"):
        load_gripper_unit(f)

    # The realistic hand-edit: the OTHER leg's matrix pasted under this one.
    # The closed key set catches it, and an empty block is not an identity
    # override either - it is a leg someone started and did not finish.
    other = "T_mount_cam" if leg == "mount_tcp" else "T_mount_tcp"
    mispaste = _write(
        tmp_path,
        "g2.yaml",
        UNIT_MIN + f"  {leg}:\n    {other}: {np.eye(4).tolist()}\n",
    )
    with pytest.raises(ValueError, match=f"build.{leg} has unknown key.*{other}"):
        load_gripper_unit(mispaste)
    empty = _write(tmp_path, "g3.yaml", UNIT_MIN + f"  {leg}:\n    measured: true\n")
    with pytest.raises(KeyError, match=key):
        load_gripper_unit(empty)


# -- the shared mount-leg validator -----------------------------------------


@pytest.mark.parametrize(("loader", "key", "other_key"), MOUNT_LEGS)
def test_mount_loaders_read_a_well_formed_block(loader, key, other_key):
    """The 4x4 comes back as-is, and the optional `measured:` provenance flag
    rides along without disturbing it."""
    T = loader(
        {
            "measured": True,
            key: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.195], [0, 0, 0, 1]],
        }
    )
    assert T.shape == (4, 4)
    assert T[2, 3] == 0.195
    np.testing.assert_array_equal(loader({key: np.eye(4).tolist()}), np.eye(4))


@pytest.mark.parametrize(("loader", "key", "other_key"), MOUNT_LEGS)
def test_mount_loaders_reject_bad_matrices(loader, key, other_key):
    """Every failure mode of a hand-written mount block must raise here rather
    than downstream."""
    with pytest.raises(KeyError, match=key):
        loader({"measured": True})  # no matrix at all
    with pytest.raises(KeyError, match=key):
        loader({})  # an empty block is not an identity override
    with pytest.raises(ValueError, match=r"expected \(4, 4\)"):
        loader({key: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    with pytest.raises(ValueError, match="not orthonormal"):
        loader(
            {
                key: [
                    [2, 0, 0, 0],  # scaled rotation block: not SE(3)
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
        )
    with pytest.raises(ValueError, match="bottom row"):
        loader(
            {
                key: [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0.5, 1],  # bottom row must be [0, 0, 0, 1]
                ]
            }
        )


@pytest.mark.parametrize(("loader", "key", "other_key"), MOUNT_LEGS)
@pytest.mark.parametrize("stray", ["schema", "measured_frame"])
def test_mount_loaders_reject_unknown_keys(loader, key, other_key, stray):
    with pytest.raises(ValueError, match=f"unknown key.*{stray}"):
        loader({stray: "x", key: np.eye(4).tolist()})


@pytest.mark.parametrize(("loader", "key", "other_key"), MOUNT_LEGS)
def test_mount_loaders_reject_the_other_legs_matrix_key(loader, key, other_key):
    """The two legs are the same shape, so a block pasted under the wrong key
    would compose a plausible-looking but wrong constant. The CLOSED KEY SET
    is the guard: the other leg's matrix key is simply unknown here, and the
    error names it alongside the keys that are legal."""
    with pytest.raises(ValueError, match=f"unknown key.*{other_key}"):
        loader({other_key: np.eye(4).tolist()})
    # ... and it stays a rejection when the correct key is present too, so a
    # half-finished paste cannot be read as "the right one wins".
    with pytest.raises(ValueError, match=other_key):
        loader({key: np.eye(4).tolist(), other_key: np.eye(4).tolist()})


# -- the session ladder ------------------------------------------------------


def test_merge_session_lets_the_later_layer_win():
    """The one merge of the ladder tool < unit < episode-arm: a later layer
    overrides a key it states and leaves the rest of the recipe standing."""
    tool = {"polarity": "red", "min_area": 145, "angle_open_deg": 80.0}
    unit = {"angle_open_deg": 78.0}
    arm = {"min_area": 200}
    assert merge_session(tool, unit, arm) == {
        "polarity": "red",
        "min_area": 200,
        "angle_open_deg": 78.0,
    }


def test_merge_session_skips_empty_layers_and_does_not_mutate_them():
    """Most units and arms state nothing; a None or {} layer must be a no-op,
    and the merge must never write back into the caller's dicts - the tool
    config is cached and shared across every arm of an episode."""
    tool = {"polarity": "red"}
    assert merge_session(None, tool, {}, None) == {"polarity": "red"}
    assert merge_session() == {}
    assert merge_session(tool, {"polarity": "dark"}) == {"polarity": "dark"}
    assert tool == {"polarity": "red"}  # untouched


# --- write_unit_session -------------------------------------------------------


def _unit_doc(session: str = "") -> str:
    return (
        "schema: chopstick_unit_v1\n"
        "tool: chop\n"
        "# a hand comment\n"
        "build:\n"
        "  marker: {id: 4, size: 0.014}\n"
        "  fixed_pose:\n"
        "    rvec: [1.0, 0.0, 0.0]\n"
        "    tvec: [0.0, 0.04, 0.03]\n"
        "    fov_deg: 160.0\n"
        "    out_size: [1920, 1920]\n"
        "    lens: front\n" + session
    )


def test_write_unit_session_replaces_the_override_and_keeps_the_build(tmp_path):
    from kiwi.core.tool_config import load_gripper_unit, write_unit_session

    units = tmp_path / "gripper" / "chop" / "units"
    units.mkdir(parents=True)
    p = units / "u1.yaml"
    p.write_text(_unit_doc("session:\n  clip_to_image: false\n"))
    write_unit_session(p, {"roi_radius_mm": 108.0, "angle_closed_deg": 17.0})
    gu = load_gripper_unit(p)
    assert gu.session == {"roi_radius_mm": 108.0, "angle_closed_deg": 17.0}
    assert gu.marker_id == 4 and gu.fixed_pose["fov_deg"] == 160.0
    assert not (units / "u1.tmp.yaml").exists()
    # An empty override removes the block rather than leaving `session: {}`.
    write_unit_session(p, {})
    assert "session" not in p.read_text()
    assert load_gripper_unit(p).session == {}


def test_write_unit_session_refuses_what_the_loader_would(tmp_path):
    """The write is validated before it lands: a document the loader
    rejects leaves the original file untouched and no temp file behind."""
    from kiwi.core.tool_config import write_unit_session

    units = tmp_path / "gripper" / "chop" / "units"
    units.mkdir(parents=True)
    p = units / "u1.yaml"
    p.write_text(_unit_doc())
    before = p.read_text()
    with pytest.raises(ValueError):
        write_unit_session(p, "not-a-mapping")  # type: ignore[arg-type]
    assert p.read_text() == before
    assert not (units / "u1.tmp.yaml").exists()
