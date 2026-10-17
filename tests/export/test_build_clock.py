"""The world-tum header contract: frame, body and clock are read, never guessed.

KIWI's world tums (``scene/state.tum`` and ``register.tum``) are
global-clock-native, so there is no start-time algebra here; what these tests
pin is the refusal discipline: a file that does not DECLARE the episode global
clock and the camera body is an error - including monotonic-clock anchor
headers, whose timestamps this module does not know how to convert.
"""

from __future__ import annotations

import pytest

from kiwi.export.build import read_world_tum_header

# The header kiwi.core.trajectory.write_tum produces (np.savetxt prefixes "# ").
# Kept verbatim so a change to the writer breaks this loudly.
_GOOD = (
    "# t tx ty tz qx qy qz qw  (frame=marker body=camera; "
    "t on episode global clock (origin=right); source=detections.json)\n"
)
_GOOD_NO_SOURCE = (
    "# t tx ty tz qx qy qz qw  (frame=marker body=camera; "
    "t on episode global clock (origin=right))\n"
)
# A monotonic-clock header: it must be refused, not misread - its
# timestamps are not on the episode global clock.
_MONOTONIC = (
    "# t x y z qx qy qz qw  (marker-frame, TUM; t on monotonic (start_time=0.000))\n"
)


def _tum(tmp_path, header: str):
    p = tmp_path / "left" / "register.tum"
    p.parent.mkdir(exist_ok=True)
    p.write_text(header + "1.000000000 0 0 0 0 0 0 1\n2.000000000 0 0 0 0 0 0 1\n")
    return p


@pytest.mark.parametrize(
    "clock",
    [
        "episode global clock (origin=right)",
        "episode global clock (sync_v2, origin=right)",
    ],
)
def test_reads_frame_body_and_clock(tmp_path, clock):
    header = _GOOD.replace("episode global clock (origin=right)", clock)
    h = read_world_tum_header(_tum(tmp_path, header))
    assert h.frame == "marker"
    assert h.body == "camera"
    assert h.clock == clock


def test_sourceless_header_keeps_the_whole_clock(tmp_path):
    """The clock string contains parentheses; the parser must not truncate it."""
    h = read_world_tum_header(_tum(tmp_path, _GOOD_NO_SOURCE))
    assert h.clock == "episode global clock (origin=right)"


def test_missing_header_raises(tmp_path):
    p = tmp_path / "left.tum"
    p.write_text("1.0 0 0 0 0 0 0 1\n")
    with pytest.raises(ValueError, match="no header line"):
        read_world_tum_header(p)


def test_monotonic_clock_header_is_refused(tmp_path):
    """A monotonic clock is not the global clock; converting it needs the
    start-time algebra this module deliberately does not have."""
    with pytest.raises(ValueError, match="unrecognized header"):
        read_world_tum_header(_tum(tmp_path, _MONOTONIC))


def test_non_global_clock_is_refused(tmp_path):
    header = (
        "# t tx ty tz qx qy qz qw  (frame=vio_world body=camera; "
        "t on estimate clock; source=ov_estimate.txt)\n"
    )
    with pytest.raises(ValueError, match="not the episode global clock"):
        read_world_tum_header(_tum(tmp_path, header))


def test_non_camera_body_is_refused(tmp_path):
    """The EE composition right-multiplies a camera-frame extrinsic; an
    imu-body tum through that math is silently wrong."""
    header = (
        "# t tx ty tz qx qy qz qw  (frame=marker body=imu; "
        "t on episode global clock (origin=left))\n"
    )
    with pytest.raises(ValueError, match="expected 'camera'"):
        read_world_tum_header(_tum(tmp_path, header))


def test_an_export_never_falls_back_to_the_other_world_source(tmp_path):
    """The two sources are different world frames. A missing one is a
    refusal that names what to run - never a quiet substitution."""
    from kiwi.export.build import _world_tum
    from kiwi.export.source import ArmSource

    (tmp_path / "left").mkdir()
    marker = tmp_path / "left" / "register.tum"
    marker.write_text(_GOOD + "1.0 0 0 0 0 0 0 1\n")
    arm = ArmSource(
        label="left",
        lens0=tmp_path / "left" / "lens0.mp4",
        state_tum=tmp_path / "left" / "scene" / "state.tum",
        eef_state=tmp_path / "left" / "eef.npz",
        sync_off=0.0,
        session_block={},
        tool="t",
        register_tum=marker,
    )
    assert _world_tum(arm, "marker") == marker
    with pytest.raises(FileNotFoundError, match="never falls back"):
        _world_tum(arm, "markerless")
    with pytest.raises(ValueError, match="expected one of"):
        _world_tum(arm, "vio")


def test_the_default_world_is_the_scene_one_and_it_is_never_substituted(tmp_path):
    """The mirror case for the default markerless world: a scene
    trajectory is what an unqualified export reads, and an episode with only
    a register.tum is refused rather than silently anchored on the marker."""
    from kiwi.export.build import _world_tum
    from kiwi.export.source import ArmSource

    scene = tmp_path / "left" / "scene" / "state.tum"
    scene.parent.mkdir(parents=True)
    scene.write_text(_GOOD + "1.0 0 0 0 0 0 0 1\n")
    arm = ArmSource(
        label="left",
        lens0=tmp_path / "left" / "lens0.mp4",
        state_tum=scene,
        eef_state=tmp_path / "left" / "eef.npz",
        sync_off=0.0,
        session_block={},
        tool="t",
    )
    assert _world_tum(arm, "markerless") == scene
    # register_tum defaults to None; asking for the anchor an episode
    # never registered must name the stage, not raise a TypeError.
    with pytest.raises(FileNotFoundError, match="register stage"):
        _world_tum(arm, "marker")
