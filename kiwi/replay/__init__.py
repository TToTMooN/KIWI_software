"""Kinematic replay + robot placement (a composer: may import ``kiwi.core`` + components).

Owns:

* the **robot-description registry + loader** (:mod:`kiwi.core.descriptions`):
  ``descriptions/<family>/meta.yaml`` -> lazy yourdfpy URDF with meshes
  resolved; :mod:`kiwi.replay.render` regenerates the URDF snapshots from
  their xacro chain (``python -m kiwi.replay.render [--check]``). The
  episode's tool config names the robot-side gripper via ``replay.gripper``.

* the **dual-arm batch IK solver** (``kiwi replay solve``): mink + MuJoCo
  differential IK over episode sets on the dual-arm rig. ``episodes``/``state``
  read the EE-state seam (``replay_state.npz``, written by ``kiwi export
  state``) into the (N, 7) replay contract; ``rig`` builds the dual
  ``fr3_chopstick_sa`` scene from KIWI's own ``descriptions/``;
  ``solve``/``batch`` solve it and score tracking, collision, and continuity;
  ``results`` writes the ``replay_solve.npz`` artifact.

* the **viser replay viewer** (``kiwi replay run``): playback over the solved
  trajectory with live rig tuning, joint posing, live-TCP gizmos,
  tracking-error plots, and the batch-solve controls
  (``viewer``/``batch_controls``, rendering through the vendored ``mjviser``
  bridge). Drives ``fr3_chopstick_sa`` twice on live mocap bases.

Contract notes:

* The state dump is always at the tool tip, and the reader refuses anything
  else: replay's solver drives the tip site.
* This package is kinematics-only (no real-hardware replay) and must not grow
  a ROS dependency.
"""
