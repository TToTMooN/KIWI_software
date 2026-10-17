"""The episode_v1 manifest: parsing, the validation rules, and the sync writer.

The rules pinned here include the tool_v1 config layering the manifest
resolves per arm: design facts from the tool file, build facts from the
gripper unit, the `session:` recipe layered tool < unit < arm.
Pure yaml + tmp_path; no media, no ffmpeg, no config tree outside the fixture.

Run: pixi run test
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from kiwi.episode.manifest import (
    MANIFEST_NAME,
    SCHEMA,
    EpisodeConfig,
    write_sync_block,
)

BASE = {
    "schema": SCHEMA,
    "name": "ep1",
    "tool": "chopsticks-v2",
    "arms": {
        "left": {
            "camera": "unit-a",
            "recording": "left_arm/L.insv",
        },
        "right": {
            "camera": "unit-b",
            "recording": "right_arm/R.insv",
        },
    },
}

#: Sentinel for `_tree(tool=...)`: write no tool file at all. An ABSENT tool
#: file is tolerated at load (the eef/export stages refuse actionably instead);
#: a PRESENT one must be valid tool_v1.
NO_TOOL_FILE = object()


def _tool(*, session=None, design=None):
    """A minimal valid ``tool_v1`` design file body.

    ``design:`` is the frozen half - the estimator bundle and the tag family;
    ``session:`` is the recipe defaults layer, omitted unless a test needs one.
    """
    body = {
        "schema": "tool_v1",
        "name": "chopsticks-v2",
        "design": {
            "estimator": "chopsticks",
            "marker": {"dictionary": "DICT_4X4_50"},
        },
    }
    if design:
        body["design"].update(design)
    if session is not None:
        body["session"] = session
    return body


def _unit(*, build=None, **over):
    """A minimal valid ``chopstick_unit_v1`` build file body.

    ``tool:`` and ``build.marker`` (id AND size - the sticker is one physical
    object) are the required minimum; ``build`` merges into that block so a
    test can add a ``fixed_pose``/mount leg or replace the marker.
    """
    body = {
        "schema": "chopstick_unit_v1",
        "tool": "chopsticks-v2",
        "build": {"marker": {"id": 4, "size": 0.014}},
    }
    if build:
        body["build"].update(build)
    body.update(over)
    return body


def _tree(tmp_path, doc=None, *, tool=None, units=None, clips=True):
    """Build an episode dir + a config tree; return (manifest_path, config_dir)."""
    ep = tmp_path / "ep1"
    ep.mkdir(parents=True, exist_ok=True)
    doc = BASE if doc is None else doc
    (ep / MANIFEST_NAME).write_text(yaml.safe_dump(doc, sort_keys=False))

    cfg = tmp_path / "config"
    for unit in ("unit-a", "unit-b"):
        (cfg / "cameras" / unit).mkdir(parents=True, exist_ok=True)
    (cfg / "gripper" / "chopsticks-v2").mkdir(parents=True, exist_ok=True)
    if tool is not NO_TOOL_FILE:
        (cfg / "gripper" / "chopsticks-v2" / "tool.yaml").write_text(
            yaml.safe_dump(_tool() if tool is None else tool)
        )
    for name, body in (units or {}).items():
        # A unit file's home is its declared design's units/ dir.
        unit_dir = cfg / "gripper" / body.get("tool", "chopsticks-v2") / "units"
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir / f"{name}.yaml").write_text(yaml.safe_dump(body))

    if clips:
        for sub, name in (("left_arm", "L.insv"), ("right_arm", "R.insv")):
            (ep / sub).mkdir(exist_ok=True)
            (ep / sub / name).write_bytes(b"media")
    return ep / MANIFEST_NAME, cfg


def _load(tmp_path, doc=None, **kw):
    m, cfg = _tree(
        tmp_path,
        doc,
        **{k: v for k, v in kw.items() if k in ("tool", "units", "clips")},
    )
    return EpisodeConfig.load(
        m, check_exists=kw.get("check_exists", False), config_dir=cfg
    )


def _with(**over):
    doc = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE.items()}
    doc["arms"] = {k: dict(v) for k, v in BASE["arms"].items()}
    doc.update(over)
    return doc


def _with_unit(unit="g1", **arm_over):
    """BASE with the left arm bound to a gripper unit (plus any extra keys)."""
    doc = _with()
    doc["arms"]["left"]["gripper_unit"] = unit
    doc["arms"]["left"].update(arm_over)
    return doc


# --- the schema gate --------------------------------------------------------


def test_loads_a_minimal_manifest(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.name == "ep1"
    assert sorted(cfg.arms) == ["left", "right"]
    assert cfg.arms["left"].recording.name == "L.insv"
    assert cfg.stream_labels == {"left", "right"}
    assert cfg.default_tool == "chopsticks-v2"


def test_wrong_schema_tag_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected 'episode_v1'"):
        _load(tmp_path, _with(schema="session_v1"))


def test_a_manifest_that_is_not_yaml_is_a_value_error_naming_the_file(tmp_path):
    # Every stage turns a ValueError into a result; a bare parser error would
    # escape them all as a traceback naming "<unicode string>", not the file.
    m, cfg = _tree(tmp_path)
    m.write_text("schema: episode_v1\narms: [left, right\n")
    with pytest.raises(ValueError, match=r"episode\.yaml: not valid YAML: ") as err:
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    assert str(m.resolve()) in str(err.value)
    assert isinstance(err.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize(
    "broken", ["chopsticks-v2/tool.yaml", "chopsticks-v2/units/g1.yaml"]
)
def test_a_named_config_file_that_is_not_yaml_is_a_value_error_naming_it(
    tmp_path, broken
):
    # The tool and gripper-unit files the manifest names are parsed by the
    # same load, so a hand edit that broke one is the same refusal.
    m, cfg = _tree(tmp_path, _with_unit(), units={"g1": _unit()})
    path = cfg / "gripper" / broken
    path.write_text("schema: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML: ") as err:
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    assert str(err.value).startswith(f"{path.resolve()}: ")
    assert isinstance(err.value.__cause__, yaml.YAMLError)


def test_name_defaults_to_the_directory(tmp_path):
    doc = _with()
    del doc["name"]
    assert _load(tmp_path, doc).name == "ep1"


def test_required_arm_keys(tmp_path):
    for key in ("camera", "recording"):
        doc = _with()
        del doc["arms"]["left"][key]
        with pytest.raises(ValueError, match=f"arms.left.{key} is required"):
            _load(tmp_path, doc)


def test_an_arm_with_no_tool_and_no_default_is_refused(tmp_path):
    doc = _with()
    del doc["tool"]
    with pytest.raises(ValueError, match="no tool and no top-level"):
        _load(tmp_path, doc)


# --- task_window ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expect",
    [
        (None, None),
        ([7.8, 30.0], (7.8, 30.0)),
        ([7.8, None], (7.8, None)),
        ([None, 30.0], (None, 30.0)),
    ],
)
def test_task_window_shapes(tmp_path, raw, expect):
    doc = _with() if raw is None else _with(task_window=raw)
    assert _load(tmp_path, doc).task_window == expect


def test_a_fully_open_window_is_refused_as_a_half_finished_scrub(tmp_path):
    with pytest.raises(ValueError, match="trims nothing"):
        _load(tmp_path, _with(task_window=[None, None]))


def test_task_window_end_must_be_after_start(tmp_path):
    with pytest.raises(ValueError, match="must be greater than start"):
        _load(tmp_path, _with(task_window=[30.0, 7.0]))


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], "1,2", [True, 2.0]])
def test_malformed_task_windows_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="must be \\[start_s, end_s\\]"):
        _load(tmp_path, _with(task_window=bad))


# --- eval_success -----------------------------------------------------------


def test_eval_success_defaults_to_false(tmp_path):
    assert _load(tmp_path).eval_success is False


def test_eval_success_reads_a_real_bool(tmp_path):
    assert _load(tmp_path, _with(eval_success=True)).eval_success is True


@pytest.mark.parametrize("bad", ["false", "no", 0, 1])
def test_a_stringy_eval_success_is_refused(tmp_path, bad):
    # "false" is truthy in Python.
    with pytest.raises(ValueError, match="must be a bool"):
        _load(tmp_path, _with(eval_success=bad))


# --- anchor -----------------------------------------------------------------


def test_anchor_passes_through_including_the_optional_scan_keys(tmp_path):
    doc = _with(
        anchor={
            "marker": {"id": 26, "size": 0.057, "dictionary": "DICT_4X4_50"},
            "frame": "full_6dof",
            "max_marker_dist": 1.0,
            "scan_seconds": 5,
            "fov_deg": 120.0,
            "out_size": 1280,
        }
    )
    anchor = _load(tmp_path, doc).anchor
    assert anchor["marker"]["id"] == 26
    # The register stage's knobs pass through untouched once checked.
    assert anchor["max_marker_dist"] == 1.0 and anchor["scan_seconds"] == 5
    assert anchor["fov_deg"] == 120.0 and anchor["out_size"] == 1280


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scan_seconds", "abc"),
        ("scan_seconds", None),  # a blank `scan_seconds:` line
        ("scan_seconds", [1, 2]),
        ("scan_seconds", True),
        ("scan_seconds", -5),
        ("scan_seconds", float("nan")),
        ("max_marker_dist", 0),
        ("max_marker_dist", float("inf")),
        pytest.param("max_marker_dist", 10**400, id="max_marker_dist-huge-int"),
        ("fov_deg", "wide"),
        ("fov_deg", 180),
        ("out_size", 1920.0),
        ("out_size", "1920"),
        ("out_size", False),
        ("out_size", 0),
    ],
)
def test_a_bad_anchor_knob_is_refused_with_its_key_named(tmp_path, key, value):
    # The register stage converts these without a guard.
    doc = _with(anchor={"marker": {"id": 26, "size": 0.057}, key: value})
    with pytest.raises(ValueError, match=rf"episode\.yaml: anchor\.{key} must be"):
        _load(tmp_path, doc)


def test_an_unimplemented_anchor_frame_is_refused(tmp_path):
    # gravity_yaw is not implemented.
    with pytest.raises(ValueError, match="anchor.frame must be"):
        _load(tmp_path, _with(anchor={"frame": "gravity_yaw"}))


def test_omitting_anchor_frame_is_fine(tmp_path):
    assert _load(tmp_path, _with(anchor={"marker": {"id": 1}})).anchor["marker"]


def test_a_todo_anchor_id_is_refused_by_the_manifest_parser(tmp_path):
    doc = _with(
        anchor={"marker": {"id": "TODO", "size": "TODO"}},
        cameras={
            "ego": {
                "recording": "ego/E.mp4",
                "head_marker": {
                    "dictionary": "DICT_4X4_100",
                    "id": 50,
                    "size": 0.049,
                },
            }
        },
    )
    with pytest.raises(ValueError, match=r"anchor:.*marker\.id.*'TODO'.*ArUco id"):
        _load(tmp_path, doc)


# --- the sync block ---------------------------------------------------------


def _sync(**kw):
    return _with(sync={"method": "audio_gcc_phat", **kw})


def test_a_sync_block_becomes_a_validated_global_clock(tmp_path):
    cfg = _load(tmp_path, _sync(origin="right", offsets={"left": -2.5, "right": 0.0}))
    clock = cfg.global_clock()
    assert clock.origin == "right"
    assert clock.offsets_s == {"left": -2.5, "right": 0.0}
    assert cfg.sync_method == "audio_gcc_phat"
    assert cfg.clock_covers_streams()


def test_a_sync_block_without_an_origin_is_refused(tmp_path):
    # GlobalClock cannot represent a block with no origin.
    with pytest.raises(ValueError, match="sync.origin is required"):
        _load(tmp_path, _sync(offsets={"left": 0.0, "right": 2.5}))


def test_no_sync_block_is_fine_and_yields_no_clock(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.global_clock() is None and not cfg.clock_covers_streams()


def test_sync_offsets_naming_an_undeclared_stream_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown stream 'ego'"):
        _load(
            tmp_path,
            _sync(origin="right", offsets={"left": -1.0, "right": 0.0, "ego": -2.0}),
        )


def test_a_positive_offset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="positive offset"):
        _load(tmp_path, _sync(origin="right", offsets={"left": 2.5, "right": 0.0}))


def test_the_origin_must_hold_exactly_zero(tmp_path):
    with pytest.raises(ValueError, match="must be exactly 0.0"):
        _load(tmp_path, _sync(origin="right", offsets={"left": -2.5, "right": -0.1}))


def test_the_origin_must_be_one_of_the_offsets(tmp_path):
    with pytest.raises(ValueError, match="not one of the offsets"):
        _load(tmp_path, _sync(origin="ego", offsets={"left": -2.5, "right": 0.0}))


def test_a_non_numeric_offset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be a number"):
        _load(tmp_path, _sync(origin="right", offsets={"left": "-2.5", "right": 0.0}))


def test_a_malformed_sync_block_is_refused(tmp_path):
    with pytest.raises(ValueError, match="`sync:` must be a mapping"):
        _load(tmp_path, _with(sync={"method": "audio_gcc_phat"}))


# --- extra camera streams ---------------------------------------------------


def test_an_extra_camera_stream_joins_the_clock(tmp_path):
    doc = _with(
        cameras={"ego": {"recording": "ego/E.mp4", "camera": "unit-a"}},
        sync={
            "method": "audio_gcc_phat",
            "origin": "ego",
            "offsets": {"left": -3.0, "right": -1.0, "ego": 0.0},
        },
    )
    cfg = _load(tmp_path, doc)
    assert cfg.stream_labels == {"left", "right", "ego"}
    assert cfg.cameras["ego"].camera_dir.name == "unit-a"
    # An extra stream owns the origin whenever it started last.
    assert cfg.global_clock().origin == "ego"
    assert cfg.clock_covers_streams()


def test_a_camera_added_after_the_sync_invalidates_the_clock(tmp_path):
    doc = _with(
        cameras={"ego": {"recording": "ego/E.mp4"}},
        sync={
            "method": "audio_gcc_phat",
            "origin": "right",
            "offsets": {"left": -2.5, "right": 0.0},
        },
    )
    cfg = _load(tmp_path, doc)
    # Loads fine, but the ego stream has no place on the clock - the sync stage
    # must re-derive rather than ship a video with no offset.
    assert not cfg.clock_covers_streams()


def test_a_camera_label_colliding_with_an_arm_is_refused(tmp_path):
    doc = _with(cameras={"left": {"recording": "x/L.mp4"}})
    with pytest.raises(ValueError, match="collides with arms"):
        _load(tmp_path, doc)


def test_a_camera_without_a_recording_is_refused(tmp_path):
    with pytest.raises(ValueError, match="`recording:` path"):
        _load(tmp_path, _with(cameras={"ego": {"camera": "unit-a"}}))


def test_camera_head_marker_is_parsed_in_metric_units(tmp_path):
    doc = _with(
        cameras={
            "ego": {
                "recording": "ego/E.mp4",
                "head_marker": {
                    "dictionary": "DICT_4X4_100",
                    "id": 50,
                    "size": 0.049,
                    "stationary_window": [288, 300.0],
                },
            }
        }
    )
    camera = _load(tmp_path, doc).cameras["ego"]
    assert camera.head_marker == {
        "dictionary": "DICT_4X4_100",
        "id": 50,
        "size": 0.049,
    }
    assert camera.head_stationary_window == (288.0, 300.0)


@pytest.mark.parametrize(
    "marker",
    [
        {"dictionary": "DICT_4X4_50", "id": 30},
        {"dictionary": "DICT_4X4_50", "id": True, "size": 0.05},
        {"dictionary": "DICT_4X4_50", "id": 30, "size": 50},
        {"dictionary": "", "id": 30, "size": 0.05},
    ],
)
def test_camera_head_marker_rejects_malformed_geometry(marker, tmp_path):
    doc = _with(cameras={"ego": {"recording": "ego/E.mp4", "head_marker": marker}})
    with pytest.raises(ValueError, match="head_marker"):
        _load(tmp_path, doc)


@pytest.mark.parametrize(
    "window",
    [True, [1.0], [1.0, None], [1.0, float("inf")], [2.0, 1.0]],
)
def test_camera_head_marker_rejects_a_bad_stationary_window(window, tmp_path):
    marker = {
        "dictionary": "DICT_4X4_50",
        "id": 30,
        "size": 0.05,
        "stationary_window": window,
    }
    doc = _with(cameras={"ego": {"recording": "ego/E.mp4", "head_marker": marker}})
    with pytest.raises(ValueError, match="stationary_window"):
        _load(tmp_path, doc)


def test_only_one_head_marker_frame_may_be_configured(tmp_path):
    marker = {"dictionary": "DICT_4X4_50", "id": 30, "size": 0.05}
    doc = _with(
        cameras={
            "ego": {"recording": "ego/E.mp4", "head_marker": marker},
            "helmet": {
                "recording": "ego/H.mp4",
                "head_marker": {**marker, "id": 31},
            },
        }
    )
    with pytest.raises(ValueError, match="multiple cameras"):
        _load(tmp_path, doc)


def test_head_marker_id_must_not_collide_with_an_arm_marker(tmp_path):
    doc = _with_unit()
    doc["cameras"] = {
        "ego": {
            "recording": "ego/E.mp4",
            "head_marker": {
                "dictionary": "DICT_4X4_50",
                "id": 4,
                "size": 0.05,
            },
        }
    }
    with pytest.raises(ValueError, match="collides"):
        _load(tmp_path, doc, units={"g1": _unit()})


def test_head_marker_id_must_not_collide_with_the_parsed_anchor_id(tmp_path):
    marker = {"dictionary": "DICT_4X4_50", "id": 30, "size": 0.05}
    doc = _with(
        anchor={"marker": {"id": 30, "size": 0.057}},
        cameras={"ego": {"recording": "ego/E.mp4", "head_marker": marker}},
    )
    with pytest.raises(ValueError, match="scene-anchor marker id"):
        _load(tmp_path, doc)


# --- the tool_v1 blocks: what each layer is allowed to say ------------------


def test_the_design_facts_ride_the_arm_from_the_tool_file(tmp_path):
    """`design:` is the frozen half - the estimator bundle, the tag family,
    and the replay description the export composes camera->tip from. No other
    layer has keys for these, so the ArmSpec reads them off the tool alone."""
    m, cfg = _tree(
        tmp_path,
        tool=_tool(
            design={
                "replay": {"gripper": "chopstick_sa", "description": "fr3_chopstick_sa"}
            }
        ),
    )
    arm = EpisodeConfig.load(m, check_exists=False, config_dir=cfg).arms["left"]
    assert arm.tool == "chopsticks-v2"
    assert arm.tool_config == cfg / "gripper" / "chopsticks-v2" / "tool.yaml"
    assert arm.estimator == "chopsticks"
    assert arm.description == "fr3_chopstick_sa"


def test_without_a_gripper_unit_only_the_dictionary_resolves(tmp_path):
    """Every build fact lives in the unit file, so an arm that names none has
    no tag, no calibrated pose and no mount overrides; tool_v1 has no
    version-wide default."""
    arm = _load(tmp_path).arms["left"]
    assert arm.marker == {"dictionary": "DICT_4X4_50"}
    assert arm.gripper_unit is None and arm.gripper_unit_config is None
    assert arm.fixed_pose is None and arm.fixed_pose_source == ""
    assert arm.mount_tcp is None and arm.mount_cam is None


def test_the_marker_id_and_size_come_from_the_unit(tmp_path):
    """The tag is one physical object: id and size travel together, out of the
    build file, merged onto the design's dictionary."""
    m, cfg = _tree(
        tmp_path,
        _with_unit(),
        units={"g1": _unit(build={"marker": {"id": 7, "size": 0.0195}})},
    )
    ep = EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    assert ep.arms["left"].marker == {
        "dictionary": "DICT_4X4_50",
        "id": 7,
        "size": 0.0195,
    }
    assert ep.arms["left"].gripper_unit == "g1"
    assert (
        ep.arms["left"].gripper_unit_config
        == cfg / "gripper" / "chopsticks-v2" / "units" / "g1.yaml"
    )
    # The other arm names no unit: the same tool, no tag.
    assert ep.arms["right"].marker == {"dictionary": "DICT_4X4_50"}


def test_the_fixed_pose_and_its_source_come_from_the_unit_alone(tmp_path):
    """tool_v1 has no version-wide pose - a calibration is bound to the tag
    placement of one build. `fixed_pose_source` names that file."""
    pose = {
        "rvec": [2.0, 0.0, 0.0],
        "tvec": [0.0, 0.0, 0.2],
        "fov_deg": 160.0,
        "out_size": [1920, 1920],
        "lens": "front",
    }
    m, cfg = _tree(
        tmp_path, _with_unit(), units={"g1": _unit(build={"fixed_pose": pose})}
    )
    ep = EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    assert ep.arms["left"].fixed_pose["rvec"] == [2.0, 0.0, 0.0]
    assert ep.arms["left"].fixed_pose_source == str(
        cfg / "gripper" / "chopsticks-v2" / "units" / "g1.yaml"
    )
    # An uncalibrated build resolves to no pose and no source, not a default.
    m2, cfg2 = _tree(tmp_path / "b", _with_unit(), units={"g1": _unit()})
    arm = EpisodeConfig.load(m2, check_exists=False, config_dir=cfg2).arms["left"]
    assert arm.fixed_pose is None and arm.fixed_pose_source == ""


def test_a_tool_level_fixed_pose_cannot_even_exist(tmp_path):
    """The tool layer has no fixed_pose key: a calibration lives in the unit's
    build block."""
    m, cfg = _tree(
        tmp_path,
        tool={**_tool(), "fixed_pose": {"rvec": [1, 0, 0], "tvec": [0, 0, 0.3]}},
    )
    with pytest.raises(ValueError, match="'fixed_pose' is not a tool_v1 top-level"):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


def test_the_mount_overrides_ride_the_arm_from_the_unit(tmp_path):
    """The two description legs a build may override. They ride the ArmSpec
    verbatim so the export builder composes the unit's measured geometry
    instead of the description's fixed-joint FK."""
    T = [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]
    m, cfg = _tree(
        tmp_path,
        _with_unit(),
        units={
            "g1": _unit(
                build={
                    "mount_tcp": {"T_mount_tcp": T, "measured": "2026-08-20 calipers"},
                    "mount_cam": {"T_mount_cam": T},
                }
            )
        },
    )
    arm = EpisodeConfig.load(m, check_exists=False, config_dir=cfg).arms["left"]
    assert arm.mount_tcp["T_mount_tcp"] == T
    assert arm.mount_tcp["measured"] == "2026-08-20 calipers"
    assert arm.mount_cam["T_mount_cam"] == T


@pytest.mark.parametrize(
    "leg,block,match",
    [
        # The closed key set is the mis-paste guard: the other leg's matrix
        # key is simply unknown here.
        ("mount_tcp", {"T_mount_cam": "T"}, "unknown key"),
        ("mount_tcp", {"T_mount_tcp": "T", "schema": "x"}, "unknown key"),
        ("mount_cam", {"T_mount_cam": "T", "measured_frame": "x"}, "unknown key"),
        # Provenance but no matrix, and a rotation block that is not a
        # rotation.
        ("mount_tcp", {"measured": "2026-08-20 calipers"}, "has no 'T_mount_tcp'"),
        ("mount_cam", {"T_mount_cam": "skew"}, "not orthonormal"),
    ],
)
def test_a_malformed_mount_block_fails_at_episode_load(tmp_path, leg, block, match):
    """Eagerly, next to the file - not at export time. A stray key is part of
    that: it is rejected, not silently carried."""
    known = {
        "T": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "skew": [[1, 0.3, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    body = {k: known.get(v, v) for k, v in block.items()}
    m, cfg = _tree(tmp_path, _with_unit(), units={"g1": _unit(build={leg: body})})
    with pytest.raises((ValueError, KeyError), match=match):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


# --- the session ladder -----------------------------------------------------


def test_session_layering_is_tool_then_unit_then_arm(tmp_path):
    """The one layered block: each layer restates only what it changes, and
    the later layer wins. Keys nobody restates survive from the tool."""
    doc = _with_unit(session={"angle_open_deg": 175.0})
    m, cfg = _tree(
        tmp_path,
        doc,
        tool=_tool(
            session={
                "polarity": "red",
                "angle_open_deg": 160.0,
                "angle_closed_deg": 80.0,
            }
        ),
        units={"g1": _unit(session={"polarity": "dark", "angle_open_deg": 170.0})},
    )
    ep = EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    arm = ep.arms["left"]
    assert arm.session["angle_open_deg"] == 175.0, "the arm block beats the unit"
    assert arm.session["polarity"] == "dark", "the unit beats the tool"
    assert arm.session["angle_closed_deg"] == 80.0, "the tool default survives"
    # The layers are per arm: the other arm keeps the tool's defaults alone.
    assert ep.arms["right"].session == {
        "polarity": "red",
        "angle_open_deg": 160.0,
        "angle_closed_deg": 80.0,
    }


def test_the_normalization_endpoints_are_ordinary_session_keys(tmp_path):
    """angle_closed_deg / angle_open_deg are real GripperDotConfig fields, so
    they layer and validate like every other knob."""
    m, cfg = _tree(
        tmp_path,
        _with_unit(),
        tool=_tool(session={"angle_closed_deg": 75.0, "angle_open_deg": 170.0}),
        units={"g1": _unit(session={"angle_closed_deg": 78.0})},
    )
    arm = EpisodeConfig.load(m, check_exists=False, config_dir=cfg).arms["left"]
    assert arm.session == {"angle_closed_deg": 78.0, "angle_open_deg": 170.0}


@pytest.mark.parametrize("layer", ["tool", "unit", "arm"])
def test_an_unknown_session_key_is_refused_at_the_layer_that_carries_it(
    tmp_path, layer
):
    """The session block is strict, and validated per layer rather than on the
    merged result, so the error names the file holding the typo."""
    bad = {"angle_opne_deg": 170.0}
    doc = _with_unit(**({"session": bad} if layer == "arm" else {}))
    m, cfg = _tree(
        tmp_path,
        doc,
        tool=_tool(session=bad if layer == "tool" else None),
        units={"g1": _unit(**({"session": bad} if layer == "unit" else {}))},
    )
    with pytest.raises(ValueError, match="unknown session key") as e:
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    msg = str(e.value)
    assert "did you mean 'angle_open_deg'" in msg
    assert {
        "tool": "chopsticks-v2/tool.yaml",
        "unit": "units/g1.yaml",
        "arm": "arms.left.session",
    }[layer] in msg


def test_a_tool_naming_an_unregistered_estimator_is_refused(tmp_path):
    """Which session keys exist is estimator knowledge; a tool naming a bundle
    KIWI has no validator for cannot have its recipe checked at all, so it
    fails at load instead of at the eef stage."""
    m, cfg = _tree(tmp_path, tool=_tool(design={"estimator": "parallel_jaw"}))
    with pytest.raises(ValueError, match="no session-key validator for estimator"):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


# --- the closed arm block ----------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [("gripper_marker_id", 9), ("gripper", {"polarity": "red"}), ("tol", "x")],
)
def test_an_unknown_arm_key_is_refused_not_ignored(tmp_path, key, value):
    """An arm key outside the closed set is refused."""
    doc = _with()
    doc["arms"]["left"][key] = value
    with pytest.raises(ValueError, match="arms.left has unknown key") as e:
        _load(tmp_path, doc)
    assert repr(key) in str(e.value)


# --- binding an arm to a build ---------------------------------------------


def test_a_gripper_unit_naming_no_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="names no config at"):
        _load(tmp_path, _with_unit("ghost"), units={"g1": _unit()})


def test_a_gripper_unit_built_for_another_tool_is_refused(tmp_path):
    """A unit lives under its design's units/ dir, so pairing an arm with
    another design's build cannot resolve; the refusal names where the unit
    actually lives instead of a bare not-found."""
    m, cfg = _tree(tmp_path, _with_unit(), units={"g1": _unit(tool="chopsticks-v3")})
    with pytest.raises(ValueError, match="build of a different design"):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


@pytest.mark.parametrize("bad", [19.5, 0.0, 2.0, "0.0195", True])
def test_a_unit_marker_size_outside_metres_is_refused(tmp_path, bad):
    # solvePnP translation is linear in this. 19.5 mm is 0.0195, not 19.5.
    m, cfg = _tree(
        tmp_path,
        _with_unit(),
        units={"g1": _unit(build={"marker": {"id": 4, "size": bad}})},
    )
    with pytest.raises(ValueError, match="build.marker.size"):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


@pytest.mark.parametrize("bad", ["3", 3.0, True, None])
def test_a_unit_marker_id_that_is_not_an_aruco_integer_is_refused(tmp_path, bad):
    # The id indexes a dictionary; a float or a string cannot name a tag.
    m, cfg = _tree(
        tmp_path,
        _with_unit(),
        units={"g1": _unit(build={"marker": {"id": bad, "size": 0.014}})},
    )
    with pytest.raises(ValueError, match="build.marker.id must be an integer"):
        EpisodeConfig.load(m, check_exists=False, config_dir=cfg)


def test_an_absent_tool_file_still_loads_but_carries_no_design_facts(tmp_path):
    """Loading an episode never depends on the tool file being present; the
    eef/export stages refuse actionably instead. A PRESENT file must still be
    valid tool_v1."""
    m, cfg = _tree(tmp_path, tool=NO_TOOL_FILE)
    arm = EpisodeConfig.load(m, check_exists=False, config_dir=cfg).arms["left"]
    assert arm.tool == "chopsticks-v2"
    assert arm.tool_config == cfg / "gripper" / "chopsticks-v2" / "tool.yaml"
    assert arm.estimator is None and arm.description is None
    assert arm.marker == {} and arm.session == {}


# --- operator metadata -------------------------------------------------------


def test_notes_and_review_are_kept_as_metadata(tmp_path):
    doc = _with(notes="held still ~5s", review="Processed: L/R OK")
    cfg = _load(tmp_path, doc)
    assert cfg.meta["notes"].startswith("held still")
    assert cfg.meta["review"].startswith("Processed")


# --- existence checking -----------------------------------------------------


def test_missing_paths_are_reported_together(tmp_path):
    m, cfg = _tree(tmp_path, clips=False)
    with pytest.raises(FileNotFoundError) as e:
        EpisodeConfig.load(m, check_exists=True, config_dir=cfg)
    msg = str(e.value)
    assert "arms.left.recording" in msg and "arms.right.recording" in msg


def test_existing_paths_pass_the_check(tmp_path):
    m, cfg = _tree(tmp_path)
    EpisodeConfig.load(m, check_exists=True, config_dir=cfg)


# --- write_sync_block -------------------------------------------------------


def _manifest_with_comments(tmp_path):
    p = tmp_path / MANIFEST_NAME
    p.write_text(
        textwrap.dedent(
            """\
            schema: episode_v1
            name: ep1

            # the left_arm/ directory holds the RIGHT arm's clip - verified by serial
            arms:
              left:
                camera: unit-a          # trailing comment

                recording: right_arm/R.insv

            task_window: [7.0, 30.0]  # scrubbed by hand
            """
        )
    )
    return p


def test_writes_a_block_and_keeps_every_comment(tmp_path):
    p = _manifest_with_comments(tmp_path)
    write_sync_block(
        p, {"left": -2.5, "right": 0.0}, method="audio_gcc_phat", origin="right"
    )
    text = p.read_text()
    doc = yaml.safe_load(text)
    assert doc["sync"]["origin"] == "right"
    assert doc["sync"]["offsets"] == {"left": -2.5, "right": 0.0}
    # The sync write leaves the hand-scrubbed window to the operator.
    assert doc["task_window"] == [7.0, 30.0]
    # The operator's account of the capture must survive a machine write.
    assert "holds the RIGHT arm's clip" in text
    assert "# trailing comment" in text
    assert "# scrubbed by hand" in text


def test_rewriting_replaces_the_block_in_place(tmp_path):
    p = _manifest_with_comments(tmp_path)
    write_sync_block(
        p, {"left": -2.5, "right": 0.0}, method="audio_gcc_phat", origin="right"
    )
    write_sync_block(
        p, {"left": 0.0, "right": -1.25}, method="audio_gcc_phat", origin="left"
    )
    doc = yaml.safe_load(p.read_text())
    assert doc["sync"]["origin"] == "left"
    assert doc["sync"]["offsets"] == {"left": 0.0, "right": -1.25}
    assert p.read_text().count("sync:") == 1
    assert "holds the RIGHT arm's clip" in p.read_text()


def test_negative_zero_never_ships(tmp_path):
    p = tmp_path / MANIFEST_NAME
    p.write_text("schema: episode_v1\n")
    write_sync_block(
        p, {"left": -1e-9, "right": 0.0}, method="audio_gcc_phat", origin="right"
    )
    assert "-0.000000" not in p.read_text()
    assert yaml.safe_load(p.read_text())["sync"]["offsets"]["left"] == 0.0


def test_the_writer_refuses_offsets_that_are_not_re_origined(tmp_path):
    p = tmp_path / MANIFEST_NAME
    p.write_text("schema: episode_v1\n")
    with pytest.raises(ValueError, match="positive sync offset"):
        write_sync_block(
            p, {"left": 2.5, "right": 0.0}, method="audio_gcc_phat", origin="right"
        )
    with pytest.raises(ValueError, match="must be 0.0"):
        write_sync_block(
            p, {"left": -2.5, "right": -0.5}, method="audio_gcc_phat", origin="right"
        )
    with pytest.raises(ValueError, match="not in offsets"):
        write_sync_block(p, {"left": 0.0}, method="audio_gcc_phat", origin="ego")


def test_what_the_writer_emits_loads_back_through_the_parser(tmp_path):
    m, cfg = _tree(tmp_path)
    write_sync_block(
        m, {"left": -2.5, "right": 0.0}, method="audio_gcc_phat", origin="right"
    )
    loaded = EpisodeConfig.load(m, check_exists=False, config_dir=cfg)
    assert loaded.global_clock().offsets_s == {"left": -2.5, "right": 0.0}
    assert loaded.sync_method == "audio_gcc_phat"
    assert loaded.clock_covers_streams()


# --- the resolved ArmSpec, through the real $KIWI_CONFIG route --------------


def test_arm_spec_resolves_the_build_layer_off_the_default_config_root(
    tmp_path, monkeypatch
):
    """The same resolution as above, but through `config_root()` rather than
    an injected config_dir - the path every CLI entry point actually takes.
    Binding the arm to a unit is what supplies the tag and the calibrated
    pose; without one there is no pose at all, because tool_v1 has no
    version-wide default."""
    cfg = tmp_path / "config"
    (cfg / "cameras" / "u1").mkdir(parents=True)
    (cfg / "gripper" / "chop" / "units").mkdir(parents=True)
    (cfg / "gripper" / "chop" / "tool.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "tool_v1",
                "design": {
                    "estimator": "chopsticks",
                    "marker": {"dictionary": "DICT_4X4_50"},
                    "replay": {"description": "fr3_chopstick_sa"},
                },
                "session": {"polarity": "red"},
            }
        )
    )
    (cfg / "gripper" / "chop" / "units" / "unit-a.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "chopstick_unit_v1",
                "tool": "chop",
                "build": {
                    "marker": {"id": 5, "size": 0.014},
                    "fixed_pose": {"rvec": [2, 0, 0], "tvec": [0, 0, 0.2]},
                },
            }
        )
    )
    monkeypatch.setenv("KIWI_CONFIG", str(cfg))

    ep = tmp_path / "ep"
    (ep / "left_arm").mkdir(parents=True)
    (ep / "left_arm" / "A.insv").write_bytes(b"m")
    doc = {
        "schema": "episode_v1",
        "tool": "chop",
        "arms": {
            "left": {
                "camera": "u1",
                "recording": "left_arm/A.insv",
            }
        },
    }
    (ep / MANIFEST_NAME).write_text(yaml.safe_dump(doc))
    arm = EpisodeConfig.load(ep / MANIFEST_NAME).arms["left"]
    assert arm.estimator == "chopsticks"
    assert arm.description == "fr3_chopstick_sa"
    assert arm.session == {"polarity": "red"}
    # No gripper_unit named: no build layer, so no tag and no pose.
    assert arm.marker == {"dictionary": "DICT_4X4_50"}
    assert arm.fixed_pose is None and arm.fixed_pose_source == ""

    doc["arms"]["left"]["gripper_unit"] = "unit-a"
    (ep / MANIFEST_NAME).write_text(yaml.safe_dump(doc))
    arm = EpisodeConfig.load(ep / MANIFEST_NAME).arms["left"]
    assert arm.marker == {"dictionary": "DICT_4X4_50", "id": 5, "size": 0.014}
    assert arm.fixed_pose["rvec"] == [2, 0, 0]
    assert arm.fixed_pose_source.endswith("gripper/chop/units/unit-a.yaml")


def test_capture_description_is_independent_of_replay_target(tmp_path):
    ep = _load(
        tmp_path,
        tool=_tool(
            design={
                "capture": {"description": "chopstick_dc"},
                "replay": {
                    "gripper": "chopstick_sa",
                    "description": "fr3_chopstick_sa",
                },
            }
        ),
    )
    assert ep.arms["left"].description == "chopstick_dc"
    assert ep.arms["right"].description == "chopstick_dc"
