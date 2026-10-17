"""Floor height from the fused scene-frame poses (component: imports ``kiwi.core`` only).

The optional ``ground`` stage. Input: for each wrist camera the demuxed front
video, the state stage's ``scene/state.tum`` (the camera OPTICAL CENTRE,
metric, gravity-aligned z-up, both arms in the one scene frame, row i = frame
i) and the unit's KB4 fisheye fit. Output: the floor's z in that frame, one
level per episode with a calibrated sigma and a GREEN/YELLOW/RED verdict, and
the per-second window records behind it. The stage composer
(:mod:`kiwi.episode.ground`) publishes them as ``ground_report.json`` and
``ground_observations.npz``; the export turns them into
``observation.ground_plane`` and ``observation.camera_height.<arm>``.

Gravity is not estimated here: the scene frame the state stage publishes is
already gravity-aligned, so the floor has ONE unknown, its height. Two
multi-view geometry channels measure it per 1 s window and must agree:

* **plane sweep** (:mod:`kiwi.ground.sweep`) - hypothesise the camera height,
  warp every frame of the window onto that plane through the kb4 model and
  score photo-consistency; every horizontal surface (floor, tabletop, bin)
  is a minimum, the floor is the deepest one inside the plausible
  camera-height band. No learned model.
* **rectified motion stereo** (:mod:`kiwi.ground.stereo`) - two frames of the
  same camera 0.15-0.35 m apart rendered into an exactly rectified pair;
  SGBM disparity, metric depth from the pose baseline, horizontal planes
  from the lifted points, lowest survivor in the band.

Then a temporal running-median gate, a clipped weighted median per camera
and jointly, time-matched cross-camera pairs and the verdict
(:mod:`kiwi.ground.pool`).

Assumptions (the stage refuses or flags when they are visibly violated, and
cannot see them when they are not):

* one planar floor for the whole trajectory - a single level per episode,
  no stairs, ramps or platforms; the per-second series is still written, the
  headline and verdict assume one level;
* poses are the state stage's: optical centre, metric, z-up, both arms in
  one frame; row i of ``state.tum`` is frame i of ``lens0.mp4``;
* the camera looks partly downward and moves (>= 0.10 m per second, axis
  <= 55 deg from nadir) for a fair share of the episode, the floor has some
  texture, and it lies 0.8-1.8 m below the camera;
* the gripper is rigid to the camera (masked as a static envelope); held
  objects and fingers are not;
* both channels share the poses, so a common pose error is invisible.

Heavy dependencies (torch, cv2) load lazily from the submodules the stage
entry imports; ``import kiwi.ground`` itself stays cheap.
"""

from kiwi.ground.camera import Camera, kb4_scaled, load_tum
from kiwi.ground.config import GroundConfig

__all__ = ["Camera", "GroundConfig", "kb4_scaled", "load_tum"]
