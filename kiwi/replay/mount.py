"""Live flange-to-CAD-root mount transforms for the imported SA gripper.

The generated composite places ``chopstick_sa_base`` under ``fr3_link8``
using the inverse of the exported ``franka_flange`` transform, preceded by
the configured -135-degree rotation about the arm flange Z axis. The fixed
``flange_site`` stays on the arm flange while the gripper and TCP move.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

# Body whose pose IS the mount-adapter transform (relative to the flange frame).
from kiwi.replay.gripper import BODY as CHOPSTICKS_BODY, TCP_SITE as TCP_SITE

# Mount-invariant flange frame, generated on ``fr3_link8``
# (the model's own ``attachment_site`` moves with the mount - see module doc).
FLANGE_SITE = "flange_site"
# Canonical grasp-point TCP site (single source of truth; ik.py re-exports it).


@dataclasses.dataclass(frozen=True)
class Mount:
    """Rigid transform of the gripper relative to the flange/attachment site."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.xyz, dtype=np.float64)

    @property
    def quat(self) -> np.ndarray:
        """wxyz quaternion for R = Rz(yaw).Ry(pitch).Rx(roll) (extrinsic xyz)."""
        x, y, z, w = Rotation.from_euler("xyz", self.rpy_deg, degrees=True).as_quat()
        return np.array([w, x, y, z], dtype=np.float64)


def apply_mount(spec, mount: Mount, body: str = CHOPSTICKS_BODY) -> None:
    """Write a mount-adapter pose onto a single-arm ``MjSpec`` in place.

    Call before ``spec.compile()`` (and, for the dual rig, before ``attach`` so
    both arms inherit it). Only touches the gripper-base ``body``, which is
    measured against the flange frame it hangs from (``fr3_link8``);
    everything inside the gripper - including its ``attachment_site`` and TCP -
    moves with it.
    """
    b = spec.body(body)
    b.pos = mount.pos
    b.quat = mount.quat


def set_model_mount(model, mount: Mount, body: str = CHOPSTICKS_BODY) -> None:
    """Write a mount-adapter pose onto a *compiled* model in place (runtime
    analogue of ``apply_mount``, which works on an MjSpec before compile).

    Mutates ``model.body_pos``/``body_quat`` for ``body`` (pass e.g.
    ``"left_chopstick_sa_base"`` for a dual rig). Call ``mujoco.mj_forward``
    afterwards to re-pose the gripper + TCP; mink's IK picks the change up on
    its next solve. This is what lets the viewer expose the mount as live
    sliders without recompiling the scene.
    """
    bid = model.body(body).id
    model.body_pos[bid] = mount.pos
    model.body_quat[bid] = mount.quat


def model_mount(model, body: str = CHOPSTICKS_BODY) -> Mount:
    """Read the current mount-adapter pose off a compiled model as a ``Mount``
    (inverse of ``set_model_mount``); used to seed the viewer sliders."""
    bid = model.body(body).id
    w, x, y, z = (float(v) for v in model.body_quat[bid])
    rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
    return Mount(
        xyz=tuple(float(v) for v in model.body_pos[bid]),
        rpy_deg=tuple(float(v) for v in rpy),
    )
