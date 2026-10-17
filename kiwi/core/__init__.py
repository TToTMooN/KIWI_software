"""Shared substrate used by every component. Imports nothing else from ``kiwi``.

Owns:

* the :class:`kiwi.core.trajectory.Trajectory` contract and its machinery -
  SE(3) helpers, :class:`kiwi.core.trajectory.PoseTrack` interpolation,
  and TUM reading and writing
* the global-clock conventions (:mod:`kiwi.core.clock`)
* the config-root seam (:mod:`kiwi.core.paths`): one repo-root finder +
  the ``$KIWI_CONFIG`` override every deployment-config reader routes through
* the episode directory tree (:mod:`kiwi.core.layout`)
* ``kiwi import`` and its localhost picker (:mod:`kiwi.core.camera.ingest`,
  :mod:`kiwi.core.camera.picker`), ``.insv`` demux
  (:mod:`kiwi.core.camera.convert`) and IMU extraction
  (:mod:`kiwi.core.camera.imu`)
* serial-based camera-registry resolution (:mod:`kiwi.core.camera.registry`)
  and the readers of its calibration artifacts
  (:mod:`kiwi.core.camera.extrinsics`, :mod:`kiwi.core.camera.kb4`)
* uniform video I/O (:mod:`kiwi.core.video`); fisheye rectification +
  virtual-pinhole views (:mod:`kiwi.core.rectify`)
* marker/tag detection primitives (:mod:`kiwi.core.markers` - shared by
  calib, register, eef) and planar-geometry helpers (:mod:`kiwi.core.geometry`)
* shared frame datatypes and the tool-state estimator contract
  (:mod:`kiwi.core.pipeline`), plus preview lifecycle support
  (:mod:`kiwi.core.viz.preview`)
* the export hub datatype (:mod:`kiwi.core.observation_episode`) and
  provenance stamping (:mod:`kiwi.core.provenance`,
  :mod:`kiwi.core.config_hash`, :mod:`kiwi.core.digest`,
  :mod:`kiwi.core.generation`), with atomic small-file writes
  (:mod:`kiwi.core.atomic`)
* the robot-description registry and URDF loader
  (:mod:`kiwi.core.descriptions`)
* the OpenCV ``FileStorage`` YAML dialect reader (:mod:`kiwi.core.opencv_yaml`)
* tool / gripper-unit config resolution (:mod:`kiwi.core.tool_config`)

``kiwi.core`` never imports from any other ``kiwi`` subpackage (enforced by
``tests/test_boundaries.py``).
"""
