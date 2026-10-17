"""Robot-description registry + lazy URDF loader over ``descriptions/``.

``descriptions/<family>/meta.yaml`` (schema ``robot_descriptions_v1``) is the
registry: one entry per description variant with its model paths, kinematic
groups, named end-effector frames, ``package://`` resolution, and the xacro
render source. This module reads those registries and hands replay and
export a :class:`RobotDescription` that can lazily parse the URDF with
meshes resolved - no ROS, no ament. The descriptions tree is the single
source of capture-to-replay geometry: export composes its camera->tip
constant from the rendered URDF via :func:`fixed_frame_T`.

The tree location comes from :func:`kiwi.core.paths.descriptions_root`
(``$KIWI_DESCRIPTIONS``, else ``<repo>/descriptions``). ``yourdfpy`` is
imported lazily inside :meth:`RobotDescription.load_urdf` - the ``replay``
extra ships it - so importing this module stays dependency-light;
:func:`fixed_frame_T` needs only stdlib XML parsing plus numpy.

Mesh resolution: pre-rendered URDFs embed ``package://NAME/...`` URIs
(``franka_description``, ``chopstick_sa``, ``chopstick_dc``). The registry's ``packages``
block is the only place that maps those names onto the tree;
:func:`build_filename_handler` turns the map into a yourdfpy
``filename_handler`` (with per-package path substitutions for upstream
layout renames, e.g. ``meshes/robots/fr3/`` -> ``fr3v2/``).
"""

from __future__ import annotations

import dataclasses
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import yaml

from kiwi.core.paths import descriptions_root

if TYPE_CHECKING:
    import yourdfpy

__all__ = [
    "SCHEMA",
    "EndEffector",
    "JointGroup",
    "PackageSpec",
    "RenderSource",
    "RobotDescription",
    "build_filename_handler",
    "fixed_frame_T",
    "list_descriptions",
    "load_robot_description",
]

# Schema tag every descriptions/<family>/meta.yaml must carry.
SCHEMA = "robot_descriptions_v1"


@dataclasses.dataclass(frozen=True)
class PackageSpec:
    """How to resolve one ``package://NAME/...`` URI prefix.

    ``dir`` is absolute (resolved against the descriptions root at load
    time). ``path_substitutions`` are ``{old_prefix: new_prefix}`` rewrites
    applied to the path portion before joining with ``dir`` - used when the
    URDF was rendered against an older layout of the upstream package.
    """

    dir: Path
    path_substitutions: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class JointGroup:
    """Named kinematic group (arm, left_arm, ...) within a description."""

    joints: tuple[str, ...] = ()
    home: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True)
class EndEffector:
    """Named end-effector frame (flange, tcp, camera, left_tcp, ...)."""

    urdf_link: str | None = None
    mjcf_site: str | None = None


@dataclasses.dataclass(frozen=True)
class RenderSource:
    """How to re-render the pre-rendered URDF from the xacro chain.

    Paths are relative to the descriptions root; args are passed to xacro
    as mappings (bools lowercased - see :mod:`kiwi.replay.render`).
    """

    xacro: str | None = None
    args: dict[str, object] = dataclasses.field(default_factory=dict)
    generator: str | None = None


def build_filename_handler(
    packages: dict[str, PackageSpec],
    fallback_dir: str | None = None,
) -> Callable[[str], str]:
    """Build a yourdfpy ``filename_handler`` for multi-package URDFs.

    yourdfpy's own loader takes a single mesh dir, which can't span URDFs
    that reference more than one ``package://`` root. Non-``package://``
    references fall back to yourdfpy's magic handler rooted at
    ``fallback_dir`` (pass the URDF's own directory to match yourdfpy's
    default behavior), or pass through unchanged when it is ``None``.
    """
    import yourdfpy as _yourdfpy  # noqa: PLC0415 - optional dep, replay extra

    default_handler = _yourdfpy.filename_handler_magic

    def handler(fname: str) -> str:
        if fname.startswith("package://"):
            stripped = fname[len("package://") :]
            pkg_name, _, path = stripped.partition("/")
            spec = packages.get(pkg_name)
            if spec is not None:
                for old, new in spec.path_substitutions.items():
                    if path.startswith(old):
                        path = new + path[len(old) :]
                        break
                return os.path.join(spec.dir, path)
        if fallback_dir is not None:
            return default_handler(fname=fname, dir=fallback_dir)
        return fname

    return handler


@dataclasses.dataclass(frozen=True)
class RobotDescription:
    """One registry entry: paths, groups, EE frames, mesh resolution.

    Constructed by :func:`load_robot_description`; frozen so callers can
    cache it. URDF parsing is lazy - nothing touches disk beyond the
    registry YAML until :meth:`load_urdf`.
    """

    name: str
    family: str
    root: Path
    urdf_path: Path | None
    mjcf_path: Path | None
    packages: dict[str, PackageSpec]
    groups: dict[str, JointGroup]
    end_effectors: dict[str, EndEffector]
    render: RenderSource | None
    # The registry file this entry came from, for provenance stamping
    # (export records the meta.yaml + URDF that supplied its constants).
    meta_path: Path | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        """All actuated joints across groups, in YAML insertion order.

        Single-arm: the arm's joints. Bimanual: left then right, as the
        groups appear in ``meta.yaml``.
        """
        out: list[str] = []
        for group in self.groups.values():
            out.extend(group.joints)
        return tuple(out)

    @property
    def home_pose(self) -> tuple[float, ...] | None:
        """Concatenated group homes; ``None`` if any group lacks one."""
        out: list[float] = []
        for group in self.groups.values():
            if group.home is None:
                return None
            out.extend(group.home)
        return tuple(out)

    @property
    def ee_link_name(self) -> str | None:
        """URDF link of the ``default`` end-effector (or ``None``)."""
        ee = self.end_effectors.get("default")
        return ee.urdf_link if ee else None

    @property
    def tcp_link_name(self) -> str | None:
        """URDF link of the ``tcp`` end-effector, falling back to ``default``."""
        ee = self.end_effectors.get("tcp")
        if ee and ee.urdf_link:
            return ee.urdf_link
        return self.ee_link_name

    @property
    def filename_handler(self) -> Callable[[str], str]:
        """yourdfpy ``filename_handler`` closure for this description."""
        fallback = self.urdf_path.parent.as_posix() if self.urdf_path else None
        return build_filename_handler(self.packages, fallback_dir=fallback)

    def load_urdf(
        self,
        *,
        load_meshes: bool = True,
        build_scene_graph: bool = True,
    ) -> yourdfpy.URDF:
        """Parse the URDF with this description's mesh resolution.

        ``load_meshes=False`` skips the mesh disk reads and yields a
        deepcopy-safe URDF; pass it when only kinematics are needed.

        Raises ``ValueError`` when the description has no URDF, and
        ``ImportError`` with the install hint when yourdfpy is missing.
        """
        if self.urdf_path is None:
            raise ValueError(f"description {self.name!r} has no URDF")
        try:
            import yourdfpy as _yourdfpy  # noqa: PLC0415 - optional dep
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "yourdfpy is required to load URDFs - install the replay "
                "extra (pip install -e .[replay] / pixi install -e replay)"
            ) from e
        return _yourdfpy.URDF.load(
            self.urdf_path.as_posix(),
            filename_handler=self.filename_handler,
            load_meshes=load_meshes,
            build_scene_graph=build_scene_graph,
        )


def _package_specs(raw: dict, root: Path, meta_path: Path) -> dict[str, PackageSpec]:
    out: dict[str, PackageSpec] = {}
    for name, block in (raw or {}).items():
        block = block or {}
        if "dir" not in block:
            raise ValueError(f"{meta_path}: packages.{name} is missing 'dir'")
        out[name] = PackageSpec(
            dir=root / block["dir"],
            path_substitutions=dict(block.get("path_substitutions") or {}),
        )
    return out


def _load_family(meta_path: Path, root: Path) -> dict[str, RobotDescription]:
    with open(meta_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{meta_path}: schema is {schema!r}, expected {SCHEMA!r}")
    family = raw.get("family") or meta_path.parent.name
    packages = _package_specs(raw.get("packages"), root, meta_path)
    out: dict[str, RobotDescription] = {}
    for name, entry in (raw.get("descriptions") or {}).items():
        entry = entry or {}
        groups: dict[str, JointGroup] = {}
        for gname, g in (entry.get("groups") or {}).items():
            g = g or {}
            groups[gname] = JointGroup(
                joints=tuple(g.get("joints") or ()),
                home=tuple(g["home"]) if g.get("home") is not None else None,
            )
        end_effectors = {
            ename: EndEffector(
                urdf_link=(ee or {}).get("urdf_link"),
                mjcf_site=(ee or {}).get("mjcf_site"),
            )
            for ename, ee in (entry.get("end_effectors") or {}).items()
        }
        render_raw = entry.get("render")
        render = (
            RenderSource(
                xacro=render_raw.get("xacro"),
                args=dict(render_raw.get("args") or {}),
                generator=render_raw.get("generator"),
            )
            if render_raw
            else None
        )
        out[name] = RobotDescription(
            name=name,
            family=family,
            root=root,
            urdf_path=root / entry["urdf"] if entry.get("urdf") else None,
            mjcf_path=root / entry["mjcf"] if entry.get("mjcf") else None,
            packages=packages,
            groups=groups,
            end_effectors=end_effectors,
            render=render,
            meta_path=meta_path,
        )
    return out


def _registry(root: Path | None) -> dict[str, RobotDescription]:
    base = root if root is not None else descriptions_root()
    out: dict[str, RobotDescription] = {}
    for meta_path in sorted(base.glob("*/meta.yaml")):
        for name, desc in _load_family(meta_path, base).items():
            if name in out:
                raise ValueError(
                    f"duplicate description name {name!r} "
                    f"({meta_path} vs family {out[name].family!r})"
                )
            out[name] = desc
    return out


def list_descriptions(root: Path | None = None) -> tuple[str, ...]:
    """Sorted names of every registered description under ``root``."""
    return tuple(sorted(_registry(root)))


def load_robot_description(name: str, root: Path | None = None) -> RobotDescription:
    """Look up a robot description by registry name.

    ``root`` overrides the tree location (tests); the default resolves via
    :func:`kiwi.core.paths.descriptions_root`. Raises ``KeyError`` naming
    the available descriptions when *name* isn't registered.
    """
    registry = _registry(root)
    desc = registry.get(name)
    if desc is None:
        raise KeyError(
            f"unknown robot description {name!r}; available: {sorted(registry)}"
        )
    return desc


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _urdf_joints(urdf_path: Path) -> tuple[dict, set[str]]:
    """``child_link -> (parent_link, T_parent_child, joint_name, type)`` + link names.

    Stdlib ``ElementTree``: this reader backs the export pipeline, which
    imports without the ``replay`` extra (yourdfpy).
    """
    root = ET.parse(urdf_path).getroot()
    joints: dict = {}
    for j in root.iter("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue  # <transmission><joint> stubs carry no kinematics
        origin = j.find("origin")
        t = np.eye(4)
        if origin is not None:
            xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
            t[:3, :3] = _rpy_matrix(*rpy)
            t[:3, 3] = xyz
        joints[child.get("link")] = (
            parent.get("link"),
            t,
            j.get("name"),
            j.get("type"),
        )
    links = {ln.get("name") for ln in root.iter("link")}
    return joints, links


def _fixed_ancestry(joints: dict, frame: str) -> dict[str, np.ndarray]:
    """``ancestor_link -> T_ancestor_frame`` for every ancestor reachable
    over FIXED joints (the frame itself included, at identity). Stops -
    silently - at the first non-fixed joint: frames rigid to each other are
    exactly those sharing an ancestor within this set."""
    out = {frame: np.eye(4)}
    t = np.eye(4)
    link = frame
    while link in joints:
        parent, tj, _name, jtype = joints[link]
        if jtype != "fixed":
            break
        t = tj @ t
        out[parent] = t
        link = parent
    return out


def fixed_frame_T(urdf_path: Path, frame_a: str, frame_b: str) -> np.ndarray:
    """Rigid transform between two URDF frames: ``p_a = T @ p_b``.

    Composed over FIXED joints only, by walking both frames up to their
    nearest common ancestor - so it works whether the frames are chained
    (``gripper_origin`` -> ``camera_lens0``) or siblings under a shared
    parent. Raises ``ValueError`` when a frame is not a link of the URDF or
    when the two frames are not rigidly connected (an actuated joint sits
    between them) - refused, never silently identity.

    This is the export seam: ``T_camera_tcp = inv(T(mount, camera)) @
    T(mount, tcp)`` is composed from the rendered URDF through here.
    """
    joints, links = _urdf_joints(urdf_path)
    for frame in (frame_a, frame_b):
        if frame not in links:
            raise ValueError(f"{urdf_path.name}: no link {frame!r} in the URDF")
    up_a = _fixed_ancestry(joints, frame_a)
    up_b = _fixed_ancestry(joints, frame_b)
    common = [link for link in up_b if link in up_a]
    if not common:
        raise ValueError(
            f"{urdf_path.name}: {frame_a!r} and {frame_b!r} are not rigidly "
            f"connected (no common ancestor over fixed joints)"
        )
    # up_b is ordered frame_b -> root, so the first hit is the nearest.
    anchor = common[0]
    return np.linalg.inv(up_a[anchor]) @ up_b[anchor]
