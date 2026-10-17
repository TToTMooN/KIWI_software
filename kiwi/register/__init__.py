"""Cross-camera registration into one world frame (component: imports ``kiwi.core`` only).

The ArUco table-marker anchor: each arm's front lens sees one marker on the
table early in the take, and re-expressing every arm's tracked camera
trajectory in that marker's own frame co-registers the arms. It is a
ground-truth and evaluation tool - the marker is an instrument put in the room
to measure the pipeline against - so no builtin pipeline profile requests it,
and export uses the markerless scene frame unless ``--world-source marker``
selects the marker frame. It runs as the on-demand ``register`` stage or
through the ``kiwi register scan|anchor`` verbs.

* **Scan** (:mod:`kiwi.register.detections`): search the early seconds of
  each arm's front lens for the episode's anchor marker; the stage caches
  every sighting as ``<arm>/detections.json``.
* **Solve** (:mod:`kiwi.register.anchor`): anchor the tracked trajectory to
  the reprojection-weighted average marker pose, gated by the two flip
  checks (:mod:`kiwi.register.health`).

Contract notes:

* Output is ``<arm>/register.tum`` - **world frame, CAMERA body, timestamps
  on the episode global clock**, all three declared in the header
  (:func:`kiwi.core.trajectory.write_tum`'s discipline). The world frame is
  the anchor marker's own 6-DoF pose (``full_6dof``, the manifest's only
  legal convention): marker centre is the origin, z along its normal.
  :func:`kiwi.export.build.read_world_tum_header` refuses a file whose
  header is missing or declares another body or clock.
* The close-marker filter defaults to ``max_marker_dist = 1.0 m``
  (:func:`kiwi.register.anchor.solve_anchor`).

This package top stays cv2-free (the ``marker_detections_v1`` schema helpers
and the flip-check math are pure numpy); the scan and the solve live in
submodules and import cv2/scipy when they actually run.
"""

from kiwi.register.detections import (
    MarkerObservation,
    load_detections,
    require_compatible,
    write_detections,
)
from kiwi.register.health import reach_verdict, spread_verdict, worst

__all__ = [
    "MarkerObservation",
    "load_detections",
    "reach_verdict",
    "require_compatible",
    "spread_verdict",
    "worst",
    "write_detections",
]
