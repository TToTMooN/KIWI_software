"""The extra-``cameras`` path in kiwi.export.build.

Covers the two decisions that are easy to get silently wrong for a non-arm
video stream: how it bounds the exported window, and refusing to invent a
sync offset. The functions under test are pure float math over dataclasses -
no video is opened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.core.trajectory import PoseTrack
from kiwi.export import build
from kiwi.export.source import CameraSource


def _arm(sync_off=0.0, fps=30.0, n=3000, t0=0.0, t1=100.0, grip=(0.0, 100.0)):
    """An _ArmData with only the fields _overlap_window reads."""
    t = np.array([t0, t1], dtype=float)
    return build._ArmData(
        label="a",
        out_label="a",
        frame="marker",
        map_set=None,
        ee_track=PoseTrack(
            t=t, rotations=Rotation.identity(2), positions=np.zeros((2, 3))
        ),
        cam_track=None,
        grip_v=np.zeros(2),
        grip_fps=fps,
        grip_t0=float(grip[0]),
        grip_t1=float(grip[1]),
        lens0=Path("a_lens0.mp4"),
        eef_state=Path("a_eef.npz"),
        sync_off=sync_off,
        fps=fps,
        n_video=n,
    )


def _cam(sync_off, fps, n):
    return build._CameraData(
        label="ego",
        key="ego",
        video=Path("ego.mp4"),
        sync_off=sync_off,
        fps=fps,
        n_video=n,
    )


def test_overlap_window_unchanged_without_cameras():
    """The arms-only result is bit-identical whether or not the camera argument
    is passed - this is the guarantee every arms-only episode relies on."""
    arms = [_arm(sync_off=0.0), _arm(sync_off=1.5, t1=90.0)]
    assert build._overlap_window(arms) == build._overlap_window(arms, [])


def test_camera_starting_later_moves_lo_only():
    arms = [_arm(sync_off=0.0), _arm(sync_off=1.5)]
    lo_a, hi_a = build._overlap_window(arms)
    lo, hi = build._overlap_window(arms, [_cam(6.0, 50.0, 7971)])
    assert lo == pytest.approx(6.0)  # ego starts last -> it sets lo
    assert lo > lo_a
    assert hi == pytest.approx(hi_a)  # ego outlasts the arms -> hi untouched


def test_camera_ending_early_truncates_hi():
    """A short camera must SHORTEN the episode rather than repeat its last
    frame."""
    arms = [_arm(sync_off=0.0), _arm(sync_off=1.5)]
    lo, hi = build._overlap_window(arms, [_cam(0.0, 50.0, 2000)])
    assert hi == pytest.approx(1999 / 50.0)


def test_camera_fully_covering_changes_nothing():
    arms = [_arm(sync_off=2.0), _arm(sync_off=1.5)]
    assert build._overlap_window(
        arms, [_cam(0.0, 50.0, 100_000)]
    ) == build._overlap_window(arms)


def test_tool_state_coverage_bounds_the_window():
    """The tool-state bound: the window may not extend past the tool-state
    clip's first or last source frame."""
    arms = [_arm(grip=(4.0, 60.0)), _arm(sync_off=1.5, grip=(0.0, 100.0))]
    lo, hi = build._overlap_window(arms)
    assert lo == pytest.approx(4.0)
    assert hi == pytest.approx(60.0)


def test_build_camera_requires_a_sync_offset(tmp_path):
    """A camera with no sync offset is refused, never defaulted."""
    video = tmp_path / "ego.mp4"
    video.write_bytes(b"")
    with pytest.raises(ValueError, match="no sync offset"):
        build._build_camera(CameraSource(label="ego", video=video, sync_off=None))


def test_build_camera_missing_recording_raises(tmp_path):
    spec = CameraSource(label="ego", video=tmp_path / "absent.mp4", sync_off=0.0)
    with pytest.raises(FileNotFoundError, match="absent.mp4"):
        build._build_camera(spec)


def test_reject_duplicate_key():
    """Two streams resolving to one key would silently drop a video."""
    features = {"observation.images.head": object()}
    with pytest.raises(ValueError, match="used twice"):
        build._reject_duplicate_key(features, "observation.images.head")
    build._reject_duplicate_key(features, "observation.images.ego")  # distinct -> fine
