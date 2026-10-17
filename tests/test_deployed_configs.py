"""The committed camera registry and gripper configuration stay consistent.

Each published calibration pointer resolves through CameraUnit. Front/rear
KB4 fits carry their own lens identity. Gripper designs and builds resolve
through the production loaders and robot-description registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiwi.core.camera.registry import CameraUnit
from kiwi.core.descriptions import list_descriptions, load_robot_description
from kiwi.core.tool_config import load_gripper_unit, load_tool_config

CONFIG = Path(__file__).resolve().parent.parent / "config"
UNITS = ("insta360-X5-001", "insta360-X5-002", "insta360-X5-003")
# X5-004 binds and syncs, but its front calibration/noise are not published.
REGISTRY_UNITS = UNITS + ("insta360-X5-004",)
# The tool-side deployment configs: one directory per design, holding its
# tool.yaml and its builds' unit files under units/. Collected at import so
# the walks below parametrize.
TOOL_YAMLS = sorted((CONFIG / "gripper").glob("*/tool.yaml"))
GRIPPER_YAMLS = sorted((CONFIG / "gripper").glob("*/units/*.yaml"))


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("unit", UNITS)
def test_registry_unit_is_complete(unit):
    """The quickstart's units exist, carry a serial, and every pointer resolves."""
    cam = CameraUnit.load(CONFIG / "cameras" / unit)
    assert cam.serial, f"{unit} has no serial - kiwi init cannot bind arms"
    for key in (
        "intrinsics",
        "intrinsics_kb4",
        "extrinsic_imu",
        "imu_noise",
        "av_sync",
    ):
        path = cam.artifact(key)
        assert path is not None and path.is_file(), f"{unit}: dangling {key}"


def test_registry_serials_are_distinct():
    serials = [CameraUnit.load(CONFIG / "cameras" / u).serial for u in REGISTRY_UNITS]
    assert len(set(serials)) == len(serials)


def test_x5_004_is_registry_only_by_design():
    """Serial binding + sync work; everything track-shaped is deliberately
    absent, and absent reads as None, never as a dangling pointer."""
    cam = CameraUnit.load(CONFIG / "cameras" / "insta360-X5-004")
    assert cam.serial == "IAHYA2511BS4S3"
    for key in ("intrinsics", "extrinsic_imu", "extrinsic_imu_back", "av_sync"):
        path = cam.artifact(key)
        assert path is not None and path.is_file(), f"dangling {key}"
    for key in ("imu_noise", "intrinsics_kb4"):
        assert cam.artifact(key) is None, f"{key} must be absent, not stubbed"


def test_go3_is_a_calibrated_single_lens_unit():
    """The ego camera: no serial binding (selected by name, never
    auto-matched), the av_sync table, and mode-keyed pinhole-equi intrinsics (one profile per recorded video mode)
    plus a front cam-IMU extrinsic. Absent by design: everything rear
    (single lens) and the artifacts the unit never measured (KB4 export,
    Allan noise)."""
    from kiwi.core.camera.registry import find_camera_by_serial

    d = CONFIG / "cameras" / "insta360-GO3-001"
    cam = CameraUnit.load(d)
    assert cam.serial == ""  # yaml null -> "", skipped by serial matching
    assert find_camera_by_serial(CONFIG / "cameras", "IAHYA2511BS4S3") is not None
    table = cam.artifact("av_sync")
    assert table is not None and table.name == "av-sync.yaml"

    import numpy as np
    import pytest
    import yaml

    from kiwi.core.rectify import IntrinsicsBundle

    # Mode-keyed intrinsics: one profile per recorded video mode (the crops
    # are different optical mappings), resolved only by frame size - a plain
    # artifact() call must refuse rather than guess.
    with pytest.raises(ValueError, match="artifact_for_size"):
        cam.artifact("intrinsics")
    profiles = {}
    for size in ((2688, 2688), (2720, 1536), (2560, 1440)):
        npz = cam.artifact_for_size("intrinsics", size)
        assert npz is not None and npz.is_file(), f"dangling intrinsics {size}"
        bundle = IntrinsicsBundle.load(npz)
        assert bundle.K_front is not None and bundle.D_front is not None
        assert bundle.ds_front is None  # pinhole-equi unit, not DS
        # each profile's npz records exactly the crop its mode key claims
        assert bundle.image_size == size
        profiles[size] = bundle
    with pytest.raises(ValueError, match="never calibrated"):
        cam.artifact_for_size("intrinsics", (1920, 1080))

    extr = cam.artifact("extrinsic_imu")
    assert extr is not None and extr.is_file(), "dangling extrinsic_imu"
    cam0 = yaml.safe_load(extr.read_text())["cam0"]
    assert np.asarray(cam0["T_cam_imu"]).shape == (4, 4)
    assert isinstance(cam0["timeshift_cam_imu"], float)
    assert cam0["camera_model"] == "pinhole"
    assert cam0["distortion_model"] == "equidistant"
    # The extrinsic is a physical constant with a single pointer; its
    # embedded camchain records the square-mode profile it was solved with.
    assert tuple(cam0["resolution"]) == (2688, 2688)
    assert (2688, 2688) in profiles

    for key in ("intrinsics_kb4", "imu_noise", "extrinsic_imu_back"):
        assert cam.artifact(key) is None, f"{key} must be absent, not stubbed"

    doc = yaml.safe_load(table.read_text())
    assert doc["schema"] == "av_sync_v1"
    modes = {k: v["av_delay_s"] for k, v in doc["modes"].items()}
    assert modes == {
        "2720x1536@29.97": 0.0625,
        "2560x1440@29.97": 0.0256,
        "2560x1440@50": -0.0047,
        # The mode the shipped ego clips are in.
        "2688x2688@29.97": 0.0142,
    }


# --------------------------------------------------------------------------- #
# the tool + gripper configs: schema tool_v1 / chopstick_unit_v1, and the
# pointers into the descriptions tree
# --------------------------------------------------------------------------- #
# The camera->tip constant is COMPOSED from the robot description a tool
# points at, with per-build corrections in the gripper unit files. The
# pointer must resolve: these gates keep config/ and descriptions/ consistent.
#
# tool_v1 separates files by variance class: design facts in the tool,
# build facts (the tag, calibrated fixed_pose and mount legs) in the unit,
# and session recipes layered across both. Loaders enforce every layering
# rule, so loading each deployed file checks the contract: a build fact at
# the design layer, an unknown key or an unsupported schema tag raises
# with the pointer instead of half-loading here.
@pytest.mark.parametrize("tool_yaml", TOOL_YAMLS, ids=lambda p: p.parent.name)
def test_every_deployed_tool_loads_as_tool_v1(tool_yaml):
    """Each design's shipped tool.yaml parses through the pipeline's own loader.

    The load IS the assertion: :func:`load_tool_config` requires the frozen
    design facts (estimator, marker.dictionary) and rejects anything stated
    at the wrong layer.
    """
    tc = load_tool_config(tool_yaml)
    assert tc.estimator, f"{tool_yaml.name}: no design.estimator"
    assert tc.marker_dictionary, f"{tool_yaml.name}: no design.marker.dictionary"


@pytest.mark.parametrize("tool_yaml", TOOL_YAMLS, ids=lambda p: p.parent.name)
def test_capture_and_replay_descriptions_are_explicit(tool_yaml):
    tool = load_tool_config(tool_yaml)
    desc = load_robot_description(tool.capture_description)
    # Export composes camera->tip from these three.
    for role in ("default", "tcp", "camera"):
        assert desc.end_effectors[role].urdf_link
    assert tool.replay_description in list_descriptions()


@pytest.mark.parametrize("unit_yaml", GRIPPER_YAMLS, ids=lambda p: p.stem)
def test_every_deployed_gripper_unit_loads_as_chopstick_unit_v1(unit_yaml):
    """Each shipped unit file parses, and its ``tool:`` design ships.

    :func:`load_gripper_unit` validates the whole build block eagerly - the
    marker id/size pair, the fixed_pose record, and both mount legs as SE(3) -
    loading each deployed file validates every build block it supplies.

    The ``tool:`` pointer is checked here, against the deployed tree: a unit
    naming a design that does not ship resolves to no parent config at all.
    """
    unit = load_gripper_unit(unit_yaml)
    assert CONFIG / "gripper" / unit.tool / "tool.yaml" in TOOL_YAMLS, (
        f"{unit_yaml.name}: tool {unit.tool!r} names no shipped design "
        f"(have {[p.parent.name for p in TOOL_YAMLS]})"
    )


@pytest.mark.parametrize("unit_yaml", GRIPPER_YAMLS, ids=lambda p: p.stem)
def test_every_deployed_gripper_unit_carries_its_own_fixed_pose(unit_yaml):
    """Each build ships the full camera<->marker calibration record.

    The unit file is the only layer that can carry ``fixed_pose``: a unit without it does not
    inherit a pose, it silently falls back to per-frame PnP on the detected
    tag. The provenance keys ride along in the same record because the pose
    is only valid under the rectifier it was solved with (`kiwi eef calibrate`
    writes all five).
    """
    fp = load_gripper_unit(unit_yaml).fixed_pose
    assert fp is not None, (
        f"{unit_yaml.name}: no build.fixed_pose - calibrate the unit "
        f"(`kiwi eef calibrate <clip> --gripper-unit {unit_yaml.stem}`) or "
        f"say in the file why this build runs on detected PnP"
    )
    assert set(fp) == {"rvec", "tvec", "fov_deg", "out_size", "lens"}


def test_units_of_one_design_carry_distinct_marker_ids():
    """Two builds of the same tool must not answer to the same ArUco id.

    The estimator selects its tag by id alone, so same-design duplicates are
    indistinguishable the moment both arms of an episode carry one. Ids may
    repeat ACROSS designs - an episode arm names its design, so those never
    compete.
    """
    by_tool: dict[str, dict[int, str]] = {}
    for unit_yaml in GRIPPER_YAMLS:
        unit = load_gripper_unit(unit_yaml)
        seen = by_tool.setdefault(unit.tool, {})
        clash = seen.get(unit.marker_id)
        assert clash is None, (
            f"{unit.name} and {clash} are both builds of {unit.tool} and both "
            f"carry marker id {unit.marker_id}"
        )
        seen[unit.marker_id] = unit.name


def test_every_x5_unit_resolves_its_rear_kb4():
    """All four X5 units publish a lens-aware rear KB4 fit; the front
    pointer (where published) resolves as a front fit. The lens gate is
    what makes a swapped pointer loud."""
    import pytest

    from kiwi.core.camera.kb4 import read_kb4
    from kiwi.core.camera.registry import CameraUnit
    from kiwi.core.paths import repo_root

    cameras = repo_root() / "config" / "cameras"
    for unit in (
        "insta360-X5-001",
        "insta360-X5-002",
        "insta360-X5-003",
        "insta360-X5-004",
    ):
        cam = CameraUnit.load(cameras / unit)
        rear = cam.artifact("intrinsics_kb4_back")
        assert rear is not None, f"{unit}: no rear KB4 pointer"
        params, doc = read_kb4(rear, expected_lens="back")
        assert doc["unit"] == unit and doc["lens"] == "back"
        assert params.shape == (8,)
        with pytest.raises(ValueError, match="wrong pointer|front"):
            read_kb4(rear, expected_lens="front")
        front = cam.artifact("intrinsics_kb4")
        if front is not None:  # X5-004 publishes none
            _, fdoc = read_kb4(front, expected_lens="front")
            assert fdoc["unit"] == unit and fdoc["lens"] == "front"
