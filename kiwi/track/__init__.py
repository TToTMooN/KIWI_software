"""Generic trajectory helpers, scene mapping and localization.

``orbslam`` owns the ORB-SLAM3 subprocess boundary and the per-frame
observation rows. ``frames``, ``sequence`` and ``masks`` prepare the engine's
inputs: exported frames, container frame times, engine settings and keypoint
masks. ``metric_align`` fits metric scale and gravity, and ``scene_frame``
places the maps in one scene frame.

Episode orchestration publishes front/rear localization observations. The
``state`` component fuses these absolute poses with raw IMU and publishes
front-camera trajectories in the shared scene frame. Components take resolved
paths, arrays and calibration values; episode types stay in the composer.

The package-level TUM and YAML APIs are usable
independently of any estimator; their shared implementations live in core.
"""

from kiwi.core.opencv_yaml import load_opencv_yaml
from kiwi.core.trajectory import write_tum

__all__ = [
    "load_opencv_yaml",
    "write_tum",
]
