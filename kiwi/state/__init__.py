"""Visual-inertial state estimation (component: imports ``kiwi.core`` only).

The batch estimator behind the scene-mapping feature: at every native
front-camera frame it estimates the IMU-body pose in the canonical scene
frame, from raw-IMU preintegration between adjacent frames plus absolute
front and rear map-localization poses already expressed in that one frame.

The graph lives in :mod:`kiwi.state.fusion` and takes plain arrays (node
times, IMU samples, tracked poses, calibrated transforms, noise densities);
the episode composer resolves canonical filenames, runs missing
localizations, and writes the published ``state.tum`` - never this
component. The solver backend (GTSAM) is an optional dependency loaded
lazily: the default install and every other pipeline path never import it
(``pixi install -e state`` provides it).

``STATE_PROFILES`` is the registry of named estimator recipes the scene
manifest's ``processing.state_profile`` selects from; the values of the
recipe are the constants of :mod:`kiwi.state.fusion`.

:mod:`kiwi.state.head` detects the passive marker of the head rig in the
rear images and places each detection in the scene frame through the
completed arm state.
"""

from dataclasses import dataclass

from kiwi.state.fusion import IMU_NOISE_SCALE
from kiwi.state.latent import LATENT_SCHEMA, read_state_latent, write_state_latent

__all__ = [
    "LATENT_SCHEMA",
    "STATE_PROFILES",
    "StateProfile",
    "read_state_latent",
    "write_state_latent",
]


@dataclass(frozen=True)
class StateProfile:
    """One named estimator recipe."""

    imu_noise_scale: float


#: Named estimator configurations selectable via the scene manifest's
#: ``processing.state_profile``. Site-specific tuning gets a distinct NAME
#: here, never a silent change to an existing one.
STATE_PROFILES: dict[str, StateProfile] = {
    "scene-fusion": StateProfile(imu_noise_scale=IMU_NOISE_SCALE),
}
