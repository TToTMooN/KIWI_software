"""Stable export field names. External consumers build against these.

Keep this module boring: names only (plus their mechanical per-arm assembly),
no I/O and no format logic - the LeRobot ``meta/info.json`` spelling lives in
``kiwi.export.writers.info``. Changing a name here is a schema version bump,
not an edit.

This module is the ONLY place a field name is spelled. Production code under
``kiwi/export/`` imports these constants - never re-types the strings - and
``tests/export/test_schema_contract.py`` enforces both directions: the public
spellings are frozen there, and an ``observation.*`` literal anywhere else in
the package fails the suite.
"""

SCHEMA_VERSION = "kiwi_export_v1"

# Video feature keys (LeRobot ``observation.images.<label>``)
IMAGE_KEY_PREFIX = "observation.images."


def wrist_image_key(label: str) -> str:
    """One arm's wrist-stream feature key, e.g. ``observation.images.left_wrist``.

    ``label`` is the manifest's arm label - ``left`` / ``right``, or ``main``
    for a single unnamed arm. The ``_wrist`` suffix names the camera's mount
    point ("head-mounted" names the GO3 ego stream, not these). ``--cameras``
    replaces the part after ``observation.images.``, e.g.
    ``--cameras left=left_cam,right=right_cam``.
    """
    return f"{IMAGE_KEY_PREFIX}{label}_wrist"


# State/action layout: named per-field so GR00T ``modality.json`` and LeRobot
# v3 ``info.json`` schemas are mechanical to emit.
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# The per-arm ``observation.state`` column names, in order - position in
# meters, a Hamilton ``xyzw`` quaternion, then the tool state normalized to
# [0, 1] - as they land in ``meta/info.json`` ``names`` and in GR00T's
# ``modality.json`` slices. Tools outside this repository read these exact
# spellings (``left_ee_x`` ... ``left_gripper``), so they are a stable format.
EE_AXES = ("ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw")


def arm_state_names(label: str) -> list[str]:
    """The 8 state-column names one arm contributes, e.g. ``left_ee_x``."""
    return [f"{label}_{ax}" for ax in EE_AXES] + [f"{label}_gripper"]


# Optional episode-relative pose feature (--relative-state): each arm's pose
# expressed in its own first-window-frame frame - the UMI-style relative
# trajectory. 7 columns per arm
# (pose only; the tool state is already frame-free).
RELATIVE_STATE_KEY = "observation.state.relative"


def arm_relative_names(label: str) -> list[str]:
    """The 7 relative-pose column names one arm contributes."""
    return [f"{label}_rel_{ax}" for ax in EE_AXES]


# Optional per-arm camera-pose feature (--include-camera-pose): the wrist
# camera's own trajectory, 7 columns named like the arm's state pose block.
CAMERA_POSE_KEY_PREFIX = "observation.camera_pose."  # + arm label


def camera_pose_key(label: str) -> str:
    """One arm's camera-pose feature key, e.g. ``observation.camera_pose.left``."""
    return f"{CAMERA_POSE_KEY_PREFIX}{label}"


# Per-arm pose-measurement gap (on by default; --no-pose-gap drops it): one
# float32 column per arm, seconds from the grid instant to the NEAREST raw
# pose sample of that arm's trajectory. The exported pose is interpolated
# (Slerp + linear) between raw samples; the gap is ~0 at a raw sample, half
# the source period between samples, large inside a tracking dropout. The raw
# number is exported, not a thresholded valid/invalid verdict.
POSE_GAP_KEY_PREFIX = "observation.pose_gap."  # + arm label


def pose_gap_key(label: str) -> str:
    """One arm's pose-gap feature key, e.g. ``observation.pose_gap.left``."""
    return f"{POSE_GAP_KEY_PREFIX}{label}"


# Ground channels (first-class fields, not sidecars)
CAMERA_HEIGHT_KEY_PREFIX = "observation.camera_height."  # + stream label, meters
GROUND_PLANE_KEY = "observation.ground_plane"  # (4,) plane in world frame
# The plane is ``n . p + d = 0`` with ``n`` the unit normal (pointing up, out
# of the floor) in the same exported world as the hands (--include-ground).
GROUND_PLANE_NAMES = ("nx", "ny", "nz", "d")


def camera_height_key(label: str) -> str:
    """One arm's camera-height feature key, e.g. ``observation.camera_height.left``."""
    return f"{CAMERA_HEIGHT_KEY_PREFIX}{label}"


# Optional head marker: the hands retain their existing state vector.
HEAD_MARKER_POSE_KEY = "observation.head_marker_pose"
HEAD_MARKER_VALID_KEY = "observation.head_marker_valid"
HEAD_MARKER_SEGMENT_ID_KEY = "observation.head_marker_segment_id"
HEAD_MARKER_POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
