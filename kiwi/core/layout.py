"""The episode directory tree, spelled once.

::

    <episode>/
      episode.yaml          kiwi init (sync: block rewritten by kiwi run)
      run_status.json       kiwi run
      raw/                  every recording, placed there by the operator
                            after `kiwi import` and never written to again
      <label>/              everything derived from that one stream
      synced_global.mp4     kiwi viz render     \\
      tool_state.png        kiwi viz trace       | whole-episode artifacts
      replay_state.npz      kiwi export state    |
      replay_solve.npz      kiwi replay solve   /

One rule: ``raw/`` holds recordings, a stream's directory holds everything
derived from that stream, and the episode root holds everything derived from
the episode as a whole.

Discovery (``kiwi init``) treats every media file in ``raw/`` as a
recording, so nothing may write a derived file there. :func:`refuse_raw_output`
is how a default output path honours that - callers whose default would land
in a ``raw/`` directory refuse and ask for an explicit path instead.

Stream artifacts have FIXED names (``eef.npz``, not ``<stem>_eef.npz``): the
stream directory is the namespace, so ``ls <episode>/left/`` reads the same
on every episode and consumers need only ``(episode_dir, label)`` - never the
recording's stem. Standalone verbs operating on a bare file outside any
episode keep their stem-derived beside-the-input defaults; the fixed names
apply inside an episode, where the pipeline passes these paths explicitly.

Arms' stream directories are named by SIDE (``left``/``right`` - the manifest
assigns sides by embedded serial); an extra stream's is named by its label,
which ``kiwi init`` takes from the ``_<tag>`` suffix ``kiwi import`` stamps
on the filename.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DETECTIONS",
    "EEF_OVERLAY",
    "EEF_INPUTS",
    "EEF_STATE",
    "EEF_TRACE",
    "HEAD_DIR",
    "HEAD_REPORT",
    "HEAD_TUM",
    "GROUND_DIR",
    "GROUND_OBSERVATIONS",
    "GROUND_REPORT",
    "LENS0",
    "LENS1",
    "PROXY",
    "RAW_DIR",
    "REGISTER_INPUTS",
    "REGISTER_TUM",
    "REGISTRATION_BUNDLE",
    "REPLAY_SOLVE",
    "REPLAY_STATE",
    "SCENE_DIR",
    "STATE_INPUTS",
    "STATE_LATENT",
    "STATE_DIAGNOSTICS",
    "HEAD_OBSERVATIONS",
    "HEAD_DIAGNOSTICS",
    "STATE_TUM",
    "SYNC_INPUTS",
    "ground_dir",
    "raw_dir",
    "refuse_raw_output",
    "stream_dir",
]

RAW_DIR = "raw"

# Per-stream artifact names, all inside stream_dir(episode_dir, label).
LENS0 = "lens0.mp4"  # convert: demuxed front lens
LENS1 = "lens1.mp4"  # convert: demuxed back lens
PROXY = "proxy.mp4"  # viz play: H.264 rendition of the playable video
DETECTIONS = "detections.json"  # register: marker sightings cache
REGISTER_TUM = "register.tum"  # register: trajectory in the marker frame
REGISTER_INPUTS = "register.inputs.json"  # register: method + digest sidecar
EEF_STATE = "eef.npz"  # eef: the eef_state_v1 artifact
EEF_INPUTS = "eef.inputs.json"  # eef: config-hash sidecar
EEF_TRACE = "eef.png"  # eef: raw + completed trace
EEF_OVERLAY = "eef.mp4"  # eef: annotated overlay (opt-in debug artifact)

# Scene-mapping artifacts, under <episode>/<label>/scene/.
SCENE_DIR = "scene"  # state/scene: per-stream scene-frame products
STATE_TUM = "state.tum"  # state: trajectory in the canonical scene frame
STATE_INPUTS = "state.inputs.json"  # state: profile + clock + map-set sidecar
STATE_LATENT = "state_latent.npz"  # state: IMU-body pose per node
STATE_DIAGNOSTICS = "state_diagnostics.npz"

# Optional whole-episode head-marker state, derived from both arms' fused
# states and rear images. It has its own directory because neither arm owns it.
HEAD_DIR = "head"
HEAD_TUM = "head_marker.tum"
HEAD_REPORT = "head_marker_report.json"
HEAD_OBSERVATIONS = "head_marker_observations.npz"
HEAD_DIAGNOSTICS = "head_marker_diagnostics.npz"

# Optional whole-episode floor height, derived from both arms' fused states
# and front videos. Its own directory for the same reason as head/: neither
# arm owns it. Two products, in the head stage's shape - a generation record
# and the per-window observations behind it.
GROUND_DIR = "ground"
GROUND_REPORT = (
    "ground_report.json"  # ground: ground_v1 record (level, sigma, verdict, inputs)
)
GROUND_OBSERVATIONS = "ground_observations.npz"  # ground: one row per 1 s window

# Whole-episode artifacts, at the episode root.
SYNC_INPUTS = "sync.inputs.json"  # sync: A/V-table content key behind the clock
REGISTRATION_BUNDLE = "registration_bundle.json"  # register: both-arm aggregate
REPLAY_STATE = "replay_state.npz"  # export state: the EE-state seam to replay
REPLAY_SOLVE = "replay_solve.npz"  # replay solve: solved joints + IK metrics


def raw_dir(episode_dir: Path) -> Path:
    """Where an episode's recordings live."""
    return Path(episode_dir) / RAW_DIR


def stream_dir(episode_dir: Path, label: str) -> Path:
    """Where one stream's derived artifacts live."""
    return Path(episode_dir) / label


def scene_dir(episode_dir: Path, label: str) -> Path:
    """Where one stream's scene-frame products live."""
    return stream_dir(episode_dir, label) / SCENE_DIR


def ground_dir(episode_dir: Path) -> Path:
    """Where the episode's ground-height products live."""
    return Path(episode_dir) / GROUND_DIR


def refuse_raw_output(path: Path, what: str) -> None:
    """Raise if a default output path would land inside a ``raw/`` directory.

    Nothing writes derived files into ``raw/``; a tool whose *default* would
    land there refuses and asks for an explicit output path. An explicit path
    is the operator's call and is never checked here.
    """
    if Path(path).parent.name == RAW_DIR:
        raise ValueError(
            f"the default {what} path ({path}) lands inside a raw/ directory, "
            f"which holds recordings only - every media file there is treated "
            f"as a recording by `kiwi init`. Inside an episode, run the "
            f"pipeline (`kiwi run`), which writes per-stream artifacts to "
            f"<episode>/<label>/; or pass an explicit output path."
        )
