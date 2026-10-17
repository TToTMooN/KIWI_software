"""SA jaw contract used by replay, independent of capture-tool joint offsets.

Endpoints are CAD travel limits, not measured hardware closure calibration.
All recorded openness is bounded to that travel before posing the mimic pair.
"""

import numpy as np

JOINTS = ("rev_left", "rev_right")
MULTIPLIERS = (1.0, -1.0)
OPEN = 0.0
CLOSED = -0.872665
BODY = "chopstick_sa_base"
JAW_BODIES = ("chopstick_sa_left_jaw", "chopstick_sa_right_jaw")
TCP_SITE = "chopstick_sa_tcp"


def jaw_qpos(angle: float) -> np.ndarray:
    """Finite, bounded primary angle and its coupled secondary angle."""
    if not np.isfinite(angle):
        raise ValueError("gripper angle must be finite")
    return float(np.clip(angle, CLOSED, OPEN)) * np.asarray(MULTIPLIERS)
