"""The dual-arm MuJoCo rig: two FR3 + chopsticks arms on live mocap bases.

Builds the solve/replay scene from KIWI's own ``descriptions/`` tree (the
``fr3_chopstick_sa`` MJCF composite, resolved through
``kiwi.core.descriptions``). The composite
already embeds the gripper with the CAD flange attachment baked in; ``build_dual``
attaches it twice under mocap mount bodies so the base placement (separation +
roll/pitch/yaw) is live-tunable at runtime via ``data.mocap_pos`` /
``data.mocap_quat`` with no scene rebuild.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco

from kiwi.replay.mount import CHOPSTICKS_BODY, TCP_SITE, Mount, apply_mount
from kiwi.replay.gripper import JOINTS, OPEN, jaw_qpos

__all__ = [
    "DEFAULT_BASE_DISTANCE",
    "FR3_CHOPSTICK_SA",
    "RobotConfig",
    "build_dual",
    "load_fr3_spec",
]

# Default base-to-base distance for the dual rig (m): the bench layout.
DEFAULT_BASE_DISTANCE = 0.805


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    """One selectable arm for the dual chopsticks rig.

    Frozen and plain-typed so it pickles into spawn-context pool workers
    (see ``kiwi.replay.batch``).
    """

    key: str
    name: str
    arm_joints: tuple[str, ...]  # ordered arm joint names (no side prefix)
    home_qpos: tuple[float, ...]  # one "ready" angle per arm joint (rad)
    # Registry name (kiwi.core.descriptions) of the composite MJCF that
    # embeds the gripper.
    composite: str
    tcp_site: str = TCP_SITE
    chopsticks_body: str = CHOPSTICKS_BODY
    gripper_joints: tuple[str, ...] = JOINTS
    gripper_open: float = OPEN

    @property
    def dof(self) -> int:
        return len(self.arm_joints)


_FR3_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
# Canonical Franka "ready" pose (joint4 negative, joint6 positive).
_FR3_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

FR3_CHOPSTICK_SA = RobotConfig(
    key="fr3_chopstick_sa",
    name="Franka FR3 + Chopstick SA",
    arm_joints=_FR3_JOINTS,
    home_qpos=_FR3_HOME,
    composite="fr3_chopstick_sa",
)


def load_fr3_spec(
    composite: str = "fr3_chopstick_sa", root: Path | None = None
) -> mujoco.MjSpec:
    """Load the generated composite with portable mesh paths resolved."""
    from kiwi.core.descriptions import load_robot_description

    desc = load_robot_description(composite, root=root)
    if desc.mjcf_path is None:
        raise ValueError(f"description {composite!r} has no MJCF")
    xml = desc.mjcf_path
    spec = mujoco.MjSpec.from_file(str(xml))
    meshdir = Path(spec.compiler.meshdir or ".")
    spec.compiler.meshdir = str((xml.parent / meshdir).resolve())
    return spec


def build_dual(
    robot: RobotConfig,
    base_distance: float = DEFAULT_BASE_DISTANCE,
    mount: Mount | None = None,
    root: Path | None = None,
) -> mujoco.MjSpec:
    """Two ``robot`` arms + chopsticks grippers, each under a *mocap* mount
    body so the base pose (separation + roll/pitch/yaw) is live-tunable at
    runtime via ``data.mocap_pos`` / ``data.mocap_quat`` - no scene rebuild.

    Bases start at +/-base_distance/2 along Y (left at +Y, right at -Y),
    facing +X on the floor. qpos layout follows attach order: left arm (dof) +
    left gripper (2), then the same for the right. ``mount`` overrides the
    gripper mount-adapter transform (flange -> gripper) on BOTH arms; ``None``
    keeps the CAD flange attachment baked into the composite.
    """
    spec = mujoco.MjSpec()
    spec.modelname = f"dual_{robot.key}"
    # Otherwise attach moves the child subtree, preventing a second attach.
    spec.copy_during_attach = True

    spec.worldbody.add_light(name="top", pos=[0, 0, 2], dir=[0, 0, -1])
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        rgba=[0.85, 0.85, 0.9, 1],
    )

    arm = load_fr3_spec(robot.composite, root=root)
    if mount is not None:
        apply_mount(arm, mount, robot.chopsticks_body)
    half = base_distance / 2.0
    for prefix, y in (("left_", half), ("right_", -half)):
        mnt = spec.worldbody.add_body(name=f"{prefix}mount", pos=[0, y, 0], mocap=True)
        spec.attach(arm, prefix=prefix, frame=mnt.add_frame())

    per_arm = list(robot.home_qpos) + jaw_qpos(robot.gripper_open).tolist()
    spec.add_key(name="home", qpos=per_arm + per_arm)
    return spec
