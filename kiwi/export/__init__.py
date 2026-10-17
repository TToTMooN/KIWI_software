"""Training-dataset export (a composer: may import ``kiwi.core`` + components).

The shape is two stages with one datatype in the middle: per-episode
artifacts build a :class:`kiwi.core.observation_episode.ObservationEpisode`
(the hub), and format writers emit it - add an input by writing a builder,
add a format by writing a writer, and neither side knows about the other.

* **Inputs** (:mod:`~kiwi.export.source`, :mod:`~kiwi.export.build`): the
  CLI - the composition root - flattens each ``episode.yaml`` into a plain
  ``EpisodeSource`` (this composer never imports ``kiwi.episode``). Per arm:
  the world-frame trajectory (``<arm>/scene/state.tum`` by default,
  ``<arm>/register.tum`` under ``--world-source marker``; world frame,
  CAMERA body and episode global clock are all declared in its header and
  refused otherwise), the ``eef_state_v1`` tool-state npz (frame-indexed,
  fps inside; :mod:`kiwi.eef.state`), and the demuxed lens0 mp4. The EE body
  frame is always the tool tip (``T_world_camera @ T_camera_tcp``, the
  hand-eye cancels), and ``T_camera_tcp`` is composed from the arm's robot
  description (``inv(T_mount_cam) @ T_mount_tcp``, each leg overridable per
  gripper unit).
* **Formats** (:mod:`~kiwi.export.writers`): LeRobot **v3.0 native** (the
  default; a stock lerobot loads it), **v2.1** (+ GR00T ``modality.json``
  via ``--groot``), and the **UMI-style** ``replay_buffer.zarr.zip`` for the
  Diffusion-Policy toolchain. ``replay_state.npz``
  (:mod:`~kiwi.export.replay_state`) is the EE-state seam to ``kiwi.replay``,
  a stable format that tools outside this repository also read.
* **Curation** (:mod:`~kiwi.export.curation`): ``curation_v1`` verdicts
  re-materialize each kept window from source artifacts - never by slicing a
  previous export. **Provenance** (:mod:`~kiwi.export.write`): every dataset
  carries ``meta/export_provenance.json`` (git + pixi-lock stamp, the
  per-arm extrinsic chain, tool-state endpoints, per-episode source stamps)
  and an HF README card rendered from it.

Contract notes:

* Field names live in :mod:`kiwi.export.schema` and are stable API - external
  consumers build against them; change them only with a schema version
  bump. ``observation.state`` is absolute (8 columns per arm,
  ``left_ee_x`` ... ``left_gripper``); ``--relative-state`` adds the
  episode-relative pose feature beside it, and the ``delta`` action policy
  differences positions in the world frame while its quaternion blocks carry
  the body-frame relative rotation ``q_t^-1 * q_{t+1}``.
* **Tool state is completed upstream**: export samples ``value_completed``,
  the eef stage's finite-on-every-frame output, at the nearest EEF source
  frame, carries completion provenance, and performs no additional
  interpolation.
* ``kiwi.export.build`` / ``.write`` / ``.video_source`` import cv2 and
  pyarrow and are NOT re-exported here.
* Scope: the video stream is the demuxed lens0 fisheye as recorded (no
  per-stream projection or rectification selector), and nothing here
  uploads; ``kiwi export card`` writes the dataset ``README.md`` and
  publishing to a hub is a manual step.
"""

from kiwi.export.curation import Curation, ExportJob, expand_episodes
from kiwi.export.replay_state import ReplayStateResult, write_replay_state
from kiwi.export.source import (
    ArmSource,
    CameraSource,
    EpisodeSource,
    GroundSource,
    HeadSource,
    register_tum_path,
)

__all__ = [
    "ArmSource",
    "CameraSource",
    "Curation",
    "EpisodeSource",
    "ExportJob",
    "GroundSource",
    "HeadSource",
    "ReplayStateResult",
    "expand_episodes",
    "register_tum_path",
    "write_replay_state",
]
