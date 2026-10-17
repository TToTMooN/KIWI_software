"""The plain input spec the export builder consumes.

``kiwi.export`` is a composer and may not import ``kiwi.episode`` (composers
never import each other - ``tests/test_boundaries.py``), yet everything the
builder needs originates in ``episode.yaml``. The resolution is the stage-entry
discipline, inverted: the CALLER - the ``kiwi export`` verb in ``kiwi.cli``,
the one module allowed to import both sides - loads the manifest and flattens
it into this spec of plain paths, floats, strings and dicts. Nothing in
``kiwi/export/`` may be typed on an episode object (test-enforced), so the
builder can also be driven from a script, a notebook, or a test with no
manifest anywhere in sight.

Every value here is FINAL: the CLI resolves layering (tool < gripper unit <
episode-arm), applies flag overrides (``--task``, ``--window``), and hands
over the result. The builder never re-derives, so there is exactly one place
precedence lives.

The upstream artifacts an arm names:

* ``state_tum`` - ``<arm>/scene/state.tum``: the arm's world-frame trajectory
  and the DEFAULT one, CAMERA body, timestamps already on the episode global
  clock, all three declared in its header (see
  :func:`kiwi.export.build.read_world_tum_header`). The state stage writes it
  in the scene's own metric gravity-aligned frame, shared by every episode of
  that scene.
* ``eef_state`` - ``<arm>/eef.npz``, the ``eef_state_v1`` artifact
  (:mod:`kiwi.eef.state`): per-source-frame tool state, NaN = dropout, fps
  stored in the artifact.
* ``state_inputs`` - ``<arm>/scene/state.inputs.json``: the state stage's
  sidecar, written beside every ``state.tum``. Its ``map_set`` is the frozen
  map set the trajectory was solved against, and therefore the identity of
  the world the poses live in (see ``scene`` on :class:`EpisodeSource`).
* ``register_tum`` - ``<arm>/register.tum``: OPTIONAL, the ArUco environment
  anchor's per-episode marker frame, read only when the export names
  ``--world-source marker`` (the ground-truth path). Same header contract.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from kiwi.core import layout

__all__ = [
    "ArmSource",
    "CameraSource",
    "EpisodeSource",
    "GroundSource",
    "HeadSource",
    "WORLD_SOURCES",
    "register_tum_path",
    "state_tum_path",
]

#: How an episode's world pose was produced. An export names ONE of these
#: and every arm of every episode in it uses that one - the two are
#: different world frames (a per-episode marker pose versus a per-scene
#: map frame).
WORLD_SOURCES = ("marker", "markerless")


def state_tum_path(episode_dir: Path, label: str) -> Path:
    """Where an arm's world-frame trajectory lives:
    ``<label>/scene/state.tum``.

    The state stage writes it from the global visual-inertial estimate,
    in the scene's own metric gravity-aligned frame - shared by every
    episode of that scene, where the marker frame is per-episode. This is
    the default world an export reads.
    """
    return layout.scene_dir(Path(episode_dir), label) / layout.STATE_TUM


def register_tum_path(episode_dir: Path, label: str) -> Path:
    """Where an arm's MARKER-frame trajectory lives: ``<label>/register.tum``.

    The register stage writes it; export reads it only under
    ``--world-source marker``, the ground-truth anchor path. The spelling is
    :mod:`kiwi.core.layout`'s. Export's reader enforces the header
    contract; callers reach the path through this function.
    """
    return layout.stream_dir(Path(episode_dir), label) / layout.REGISTER_TUM


@dataclasses.dataclass(frozen=True)
class ArmSource:
    """One arm's resolved inputs. Plain data only - no episode types.

    ``session_block`` is the manifest's MERGED ``session:`` mapping (tool
    config < gripper unit < episode-arm); the tool bundle's ``closed_open``
    reads the normalization endpoints out of it. ``description`` is the
    robot-description registry name the arm's tool design declares
    (``design.capture.description``), whose URDF the builder composes the
    camera->tip constant from. ``mount_tcp_override`` / ``mount_cam_override``
    are the gripper unit's ``build:`` mount-frame blocks verbatim (``None`` =
    take that leg from the description's fixed-joint FK - builtins are never
    YAML), and ``override_source`` names the unit YAML that supplied them,
    for provenance stamping.

    ``intrinsics`` / ``intrinsics_kb4`` are the arm's camera-registry
    calibration artifacts (the DoubleSphere npz and its KB4 fit), resolved
    through ``CameraUnit.artifact`` by the caller exactly as the pipeline
    resolved them; ``camera`` is the registry unit nickname for reporting.
    The builder embeds the MODEL VALUES plus each file's content stamp into
    the episode meta, so a published dataset carries the camera model that
    made it. ``intrinsics_kb4`` may
    be None - a unit with no KB4 fit yet still exports, the DS model being
    the calibration and the KB4 a derived convenience - but a publishing
    export with ``intrinsics=None`` is REFUSED by ``build_and_write``.

    ``sync_off=None`` means the manifest's ``sync:`` block had no entry for
    this arm. It defaults to 0.0 only for a single-stream episode, whose clock
    origin IS its frame 0; with a second stream the builder refuses, since the
    offsets are what relate the streams to each other.
    """

    label: str
    lens0: Path
    #: the scene-frame trajectory (``<arm>/scene/state.tum``) - the DEFAULT
    #: world, read unless the export names ``--world-source marker``
    state_tum: Path
    eef_state: Path
    sync_off: float | None
    session_block: Mapping[str, object]
    tool: str
    #: the state stage's sidecar beside ``state_tum``
    #: (``<arm>/scene/state.inputs.json``), carrying the ``map_set`` the
    #: trajectory was solved against; read only under the markerless source
    state_inputs: Path | None = None
    #: the marker-frame trajectory (``<arm>/register.tum``), when the episode
    #: has one; only read when the export names the marker world source
    register_tum: Path | None = None
    description: str | None = None
    mount_tcp_override: Mapping[str, object] | None = None
    mount_cam_override: Mapping[str, object] | None = None
    override_source: Path | None = None
    camera: str | None = None
    intrinsics: Path | None = None
    intrinsics_kb4: Path | None = None

    @property
    def out_label(self) -> str:
        return self.label or "main"


@dataclasses.dataclass(frozen=True)
class CameraSource:
    """An extra (non-arm) stream: a video and where its frame 0 sits.

    ``sync_off=None`` means the manifest's ``sync:`` block had no entry for
    this stream; the builder refuses it.
    """

    label: str
    video: Path
    sync_off: float | None = None
    key: str | None = None
    # Same registry trio as ArmSource, best-effort: an extra stream with a
    # registry unit gets its model embedded too; one without (or with no
    # published intrinsics) is recorded as absent rather than refused.
    camera: str | None = None
    intrinsics: Path | None = None
    intrinsics_kb4: Path | None = None


@dataclasses.dataclass(frozen=True)
class HeadSource:
    """Optional marker trajectory and the report/diagnostics generated with it."""

    trajectory: Path
    report: Path
    diagnostics: Path


@dataclasses.dataclass(frozen=True)
class GroundSource:
    """Optional floor-height report and the per-window observations behind it."""

    report: Path
    observations: Path | None = None


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    """One episode's complete, resolved export input.

    ``task`` is the final language string (manifest description or CLI
    override - resolved by the caller). ``task_window`` is the manifest's
    global-clock trim, with ``None`` bounds meaning "do not trim this side";
    an explicit ``window`` argument to the builder overrides it wholesale.
    ``repo_root`` shortens provenance stamps to repo-relative paths.

    ``scene`` is the NAME of the physical scene this episode was recorded in,
    flattened by the caller out of ``episode.yaml``'s ``scene:``
    back-reference. Under the markerless world source it is half of the
    episode's world identity - the other half being the map set in each arm's
    ``state_inputs`` sidecar - and one dataset carries exactly one. ``None``
    means the caller resolved no scene (an episode with no back-reference, or
    a marker-source export, where the world is per-episode anyway).
    """

    name: str
    episode_dir: Path
    arms: tuple[ArmSource, ...]
    task: str
    cameras: tuple[CameraSource, ...] = ()
    task_window: tuple[float | None, float | None] | None = None
    repo_root: Path | None = None
    scene: str | None = None
    head: HeadSource | None = None
    ground: GroundSource | None = None
