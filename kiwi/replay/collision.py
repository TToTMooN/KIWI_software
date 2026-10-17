"""Per-frame collision tally for a posed dual-arm rig.

Classifies the penetrating contacts of one both-arms configuration into three
flags the metrics report separately:

``self``      - two geoms of the SAME arm penetrate (excluding the designed
                chopstick prong pair).
``arm_cross`` - inter-arm contact where at least one geom is an arm LINK: the
                arms physically cross. Avoidable by placement -> gated hard.
``gripper``   - inter-arm contact between only gripper/chopsticks geoms: the
                two tools overlap where the human hands meet. Task-inherent ->
                soft.
"""

from __future__ import annotations

import mujoco
import numpy as np

from kiwi.replay.gripper import BODY, JAW_BODIES

SIDES = ("left", "right")

__all__ = ["build_collision_ctx", "collide"]


def _descendants(model: mujoco.MjModel, root: int) -> set[int]:
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


def geom_roles(model: mujoco.MjModel, sides=SIDES) -> dict:
    """Classify geometry by explicit gripper/jaw subtrees, never CAD names."""
    result = {}
    for side in sides:
        arm = _descendants(model, model.body(f"{side}_mount").id)
        gripper = _descendants(model, model.body(f"{side}_{BODY}").id)
        jaws = [
            _descendants(model, model.body(f"{side}_{jaw}").id) for jaw in JAW_BODIES
        ]
        result[side] = {
            "links": [
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in arm - gripper
            ],
            "gripper": [
                g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in gripper
            ],
            "jaws": [
                [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in jaw]
                for jaw in jaws
            ],
        }
    return result


def build_collision_ctx(model: mujoco.MjModel) -> dict:
    """Build contact roles and intentional paired-jaw exclusions."""
    side_of = {}
    arm_mask = np.zeros(model.ngeom, dtype=bool)
    link_mask = np.zeros(model.ngeom, dtype=bool)
    prong_pairs = set()
    for side, roles in geom_roles(model).items():
        for g in roles["links"] + roles["gripper"]:
            side_of[g] = side
            arm_mask[g] = True
        link_mask[roles["links"]] = True
        for a in roles["jaws"][0]:
            for b in roles["jaws"][1]:
                prong_pairs.update(((a, b), (b, a)))
    return {
        "side_of": side_of,
        "arm_mask": arm_mask,
        "link_mask": link_mask,
        "prong_pairs": prong_pairs,
    }


def collide(
    model: mujoco.MjModel, data: mujoco.MjData, coll: dict
) -> tuple[bool, bool, bool]:
    """(self, arm_cross, gripper) collision flags for the posed both-arm
    configuration (see the module doc for what each flag means)."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    side_of, arm_mask = coll["side_of"], coll["arm_mask"]
    link_mask, prong = coll["link_mask"], coll["prong_pairs"]
    selfc = cross = grip = False
    for i in range(data.ncon):
        con = data.contact[i]
        if con.dist >= 0.0:
            continue  # not penetrating
        g1, g2 = int(con.geom1), int(con.geom2)
        if not (arm_mask[g1] and arm_mask[g2]):
            continue  # involves a base / floor / non-arm geom
        if side_of[g1] == side_of[g2]:
            if (g1, g2) not in prong:
                selfc = True
        elif link_mask[g1] or link_mask[g2]:
            cross = True
        else:
            grip = True
        if selfc and cross and grip:
            break
    return selfc, cross, grip
