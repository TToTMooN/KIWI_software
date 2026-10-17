"""The descriptions/ registry + loader: every entry loads, every mesh resolves.

Mesh-resolution and URDF-load tests need yourdfpy (the ``replay`` extra; CI
installs it so they execute there). The registry and MJCF tests are
stdlib+yaml only and always run.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from kiwi.core.paths import descriptions_root
from kiwi.core.tool_config import load_tool_config, tool_config_path
from kiwi.core.descriptions import list_descriptions, load_robot_description

ROOT = descriptions_root()
NAMES = list_descriptions()
URDF_NAMES = [n for n in NAMES if load_robot_description(n).urdf_path is not None]


def test_registry_names() -> None:
    assert "fr3_chopstick_sa" in NAMES
    assert "fr3_dual_chopstick_sa" in NAMES


def test_unknown_name_raises_with_available() -> None:
    with pytest.raises(KeyError, match="fr3_chopstick_sa"):
        load_robot_description("nope")


def test_single_arm_spec() -> None:
    d = load_robot_description("fr3_chopstick_sa")
    assert d.joint_names == tuple(f"fr3_joint{i}" for i in range(1, 8))
    assert d.home_pose is not None and len(d.home_pose) == 7
    assert d.tcp_link_name == "hand_tcp"
    assert d.ee_link_name == "gripper_origin"
    assert "camera" not in d.end_effectors
    assert d.urdf_path is not None and d.urdf_path.is_file()
    assert d.mjcf_path is not None and d.mjcf_path.is_file()
    assert d.render is not None


def test_dual_spec() -> None:
    d = load_robot_description("fr3_dual_chopstick_sa")
    assert len(d.joint_names) == 14
    assert d.joint_names[0] == "left_fr3_joint1"
    assert d.joint_names[7] == "right_fr3_joint1"
    assert d.home_pose is not None and len(d.home_pose) == 14
    # default/tcp point at the left arm, per the bimanual convention.
    assert d.tcp_link_name == "left_hand_tcp"
    assert d.end_effectors["right_tcp"].urdf_link == "right_hand_tcp"
    assert d.mjcf_path is None  # no dual MJCF exists


@pytest.mark.parametrize("name", URDF_NAMES)
def test_urdf_package_refs_resolve(name: str) -> None:
    """Every package:// mesh URI in the URDF maps to an existing file."""
    pytest.importorskip("yourdfpy")
    d = load_robot_description(name)
    assert d.urdf_path is not None
    handler = d.filename_handler
    refs = sorted(set(re.findall(r'"(package://[^"]+)"', d.urdf_path.read_text())))
    assert refs, "URDF carries no package:// refs - resolution is untested"
    for ref in refs:
        resolved = Path(handler(ref))
        assert resolved.is_file(), f"{ref} -> {resolved} does not exist"


@pytest.mark.parametrize("name", URDF_NAMES)
def test_urdf_loads_and_declares_real_frames(name: str) -> None:
    """yourdfpy parses the URDF (meshes included); declared frames exist."""
    pytest.importorskip("yourdfpy")
    d = load_robot_description(name)
    urdf = d.load_urdf()
    for ee_name, ee in d.end_effectors.items():
        if ee.urdf_link is not None:
            assert ee.urdf_link in urdf.link_map, f"{ee_name}: {ee.urdf_link}"
    actuated = set(urdf.actuated_joint_names)
    for joint in d.joint_names:
        assert joint in actuated, f"{joint} not an actuated URDF joint"


@pytest.mark.parametrize("name", ["fr3_chopstick_sa", "chopstick_sa"])
def test_source_flange_and_tcp_geometry_is_preserved(name):
    from kiwi.core.descriptions import fixed_frame_T

    source = ROOT / "chopstick_sa/ChopstickSA.urdf"
    desc = load_robot_description(name)
    actual = fixed_frame_T(desc.urdf_path, "franka_flange", "hand_tcp")
    expected = fixed_frame_T(source, "franka_flange", "hand_tcp")
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(actual[:3, 3], [0, -0.003, 0.194], atol=1e-6)
    if name.startswith("fr3_"):
        from scipy.spatial.transform import Rotation

        clocking = np.eye(4)
        clocking[:3, :3] = Rotation.from_euler("z", -135, degrees=True).as_matrix()
        np.testing.assert_allclose(
            fixed_frame_T(desc.urdf_path, "fr3_link8", "franka_flange"),
            clocking,
            atol=1e-10,
        )


@pytest.mark.parametrize("name", NAMES)
def test_only_the_capture_model_carries_a_camera_frame(name):
    """`chopstick_dc` is the capture geometry, so it alone needs an optical frame.

    Its `camera_lens0` is not measured on this design (see the provenance
    comment at the joint in ChopstickDC.urdf). The replay-side models compose
    no camera->tip constant, so a camera frame appearing on one of them is a
    generation bug.
    """
    desc = load_robot_description(name)
    roles = {role for role in desc.end_effectors if "camera" in role}
    assert roles == ({"camera"} if name == "chopstick_dc" else set())


def test_fixed_frame_T_matches_yourdfpy():
    pytest.importorskip("yourdfpy")
    from kiwi.core.descriptions import fixed_frame_T

    d = load_robot_description("fr3_chopstick_sa")
    u = d.load_urdf(load_meshes=False)
    for a, b in [
        ("gripper_origin", "hand_tcp"),
        ("fr3_link8", "franka_flange"),
        ("franka_flange", "hand_tcp"),
    ]:
        ref = np.linalg.inv(u.get_transform(a, u.base_link)) @ u.get_transform(
            b, u.base_link
        )
        np.testing.assert_allclose(fixed_frame_T(d.urdf_path, a, b), ref, atol=1e-12)
    with pytest.raises(ValueError, match="no link 'nope'"):
        fixed_frame_T(d.urdf_path, "gripper_origin", "nope")
    with pytest.raises(ValueError, match="not rigidly connected"):
        fixed_frame_T(d.urdf_path, "fr3_link0", "hand_tcp")


def _mjcf_mesh_refs(mjcf: Path) -> list[Path]:
    """Mesh files a top-level MJCF references, resolved to filesystem paths.

    ``meshdir`` is relative to the model file's directory; mesh ``file``
    attrs are relative to ``meshdir``; ``include`` paths are relative to
    the model file's directory and inherit the parent's compiler.
    """
    root = ET.parse(mjcf).getroot()
    docs = [root]
    for inc in root.iter("include"):
        docs.append(ET.parse(mjcf.parent / inc.attrib["file"]).getroot())
    meshdir = mjcf.parent
    for doc in docs:
        compiler = doc.find("compiler")
        if compiler is not None and "meshdir" in compiler.attrib:
            meshdir = mjcf.parent / compiler.attrib["meshdir"]
    out = []
    for doc in docs:
        for mesh in doc.iter("mesh"):
            if "file" in mesh.attrib:
                out.append(meshdir / mesh.attrib["file"])
    return out


@pytest.mark.parametrize(
    "mjcf",
    sorted(p for p in ROOT.glob("*/mjcf/*.xml")),
    ids=lambda p: p.name,
)
def test_mjcf_meshes_resolve(mjcf: Path) -> None:
    refs = _mjcf_mesh_refs(mjcf)
    assert refs, f"{mjcf.name} references no meshes - resolution is untested"
    for ref in refs:
        assert ref.resolve().is_file(), f"{mjcf.name}: {ref} does not exist"


@pytest.mark.parametrize(
    "tool_yaml",
    sorted(tool_config_path("x").parents[1].glob("*/tool.yaml")),
    ids=lambda p: p.stem,
)
def test_shipped_tool_replay_pointers(tool_yaml: Path) -> None:
    """Every shipped tool resolves."""
    tool = load_tool_config(tool_yaml)
    assert (ROOT / tool.replay_gripper).is_dir()
    assert tool.replay_description in NAMES
    assert tool.capture_description in NAMES
