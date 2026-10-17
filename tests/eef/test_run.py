"""Tests for kiwi.eef.run - the tool-blind batch driver.

The video decode + rectifier seams are monkeypatched with synthetic rendered
frames (the test_kalibr_staging pattern), so the loop, the artifact assembly
and the bundle dispatch are exercised without codecs. The dummy-bundle test
drives the runner with a second bundle: none of the tool-blind layers may
reference chopsticks names for it to pass.

Run: pixi run test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import kiwi.eef  # noqa: E402
from kiwi.core import layout  # noqa: E402
from kiwi.core import rectify as rectify_mod  # noqa: E402
from kiwi.core.pipeline import GripperState  # noqa: E402
from kiwi.eef import run as run_mod  # noqa: E402
from kiwi.eef.state import read_eef_state  # noqa: E402
from tests.eef.test_chopsticks import (  # noqa: E402
    _A,
    FIXED_POSE,
    H,
    K,
    MARKER_SIZE,
    SESSION_BLOCK,
    W,
    _jaw_points,
    _make,
    _render,
)


class _IdentityRect:
    K_pinhole = K
    out_size = (W, H)

    @staticmethod
    def apply(frame):
        return frame


def _fake_video(monkeypatch, frames, fps=30.0):
    info = run_mod.VideoInfo(width=W, height=H, fps=fps, n_frames=len(frames))
    monkeypatch.setattr(run_mod.VideoInfo, "read", staticmethod(lambda p: info))
    monkeypatch.setattr(run_mod, "iter_frames", lambda p: enumerate(frames))
    monkeypatch.setattr(
        rectify_mod, "load_rectifier_auto", lambda *a, **k: _IdentityRect()
    )


def _job(tmp_path, **over) -> run_mod.ClipJob:
    """The episode case: recording flat in ``raw/``, the demuxed stream and
    every artifact in the arm's stream directory (``out_dir``)."""
    ep = tmp_path / "ep"
    rec = layout.raw_dir(ep) / "VID_001_left.insv"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_bytes(b"insv")
    sdir = layout.stream_dir(ep, "left")
    sdir.mkdir(parents=True, exist_ok=True)
    lens0 = sdir / layout.LENS0
    lens0.write_bytes(b"media")
    intr = tmp_path / "intrinsics.npz"
    intr.write_bytes(b"npz")
    base = dict(
        recording=rec,
        lens0_mp4=lens0,
        tool="chopsticks-v3-1",
        estimator_key="chopsticks",
        session_block=SESSION_BLOCK,
        intrinsics=intr,
        out_dir=sdir,
        marker_id=4,
        marker_size_m=MARKER_SIZE,
        fixed_pose=FIXED_POSE,
        fixed_pose_source="config/gripper/chopsticks-v3-1/units/test-unit.yaml",
        out_size=(W, H),
    )
    base.update(over)
    return run_mod.ClipJob(**base)


def _bare_clip(tmp_path) -> tuple[Path, Path]:
    """A clip outside any episode: ``(recording, lens0)`` side by side."""
    rec = tmp_path / "inbox" / "VID_009_left.insv"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_bytes(b"insv")
    lens0 = rec.with_name(rec.stem + "_lens0.mp4")
    lens0.write_bytes(b"media")
    return rec, lens0


def _frames(n=10, drop_at=()):
    """Synthetic clip: both-side scenes, a 1-dot dropout at the given frames."""
    _, est = _make()
    good = _render(est, np.vstack([_jaw_points(-1, -_A), _jaw_points(+1, +_A)]))
    bad = _render(est, np.array([[0.05, 0.05]]))
    return [bad if i in drop_at else good for i in range(n)]


def test_run_clip_end_to_end(tmp_path, monkeypatch):
    _fake_video(monkeypatch, _frames(10, drop_at={4}))
    ticks = []
    res = run_mod.run_clip(
        _job(tmp_path), write_trace=False, on_progress=lambda i, n: ticks.append(i)
    )

    sdir = layout.stream_dir(tmp_path / "ep", "left")
    assert res.state_path == sdir / layout.EEF_STATE
    assert res.overlay_path is None
    assert not (sdir / layout.EEF_OVERLAY).exists()
    assert res.n_valid == 9
    assert ticks  # progress surfaced

    s = read_eef_state(res.state_path)
    assert len(s) == 10
    assert (s.tool, s.estimator, s.unit) == ("chopsticks-v3-1", "chopsticks", "deg")
    assert s.pose_mode == "fixed_pose"
    assert s.pose_source == "config/gripper/chopsticks-v3-1/units/test-unit.yaml"
    assert s.fps == pytest.approx(30.0)
    np.testing.assert_array_equal(s.valid, [i != 4 for i in range(10)])
    assert s.value[0] == pytest.approx(2 * _A, abs=0.8)
    assert s.error[4].startswith("need >=2 dots")
    assert s.detail["mode"][0] == "both"
    assert "9/10 frames valid" in res.summary()
    assert "10/10 completed" in res.summary()
    assert s.completion["filled"].sum() == 1
    assert s.completion["method"][4] == "linear"
    np.testing.assert_array_equal(s.output[s.valid], s.value[s.valid])


def test_completion_fills_all_gap_levels(tmp_path, monkeypatch):
    missing = {0, 1, 4, 5, *range(8, 20), *range(23, 60), 63, 64}
    _fake_video(monkeypatch, _frames(65, drop_at=missing), fps=30.0)
    result = run_mod.run_clip(_job(tmp_path), write_trace=False)
    state = read_eef_state(result.state_path)
    assert state.valid.sum() == 65 - len(missing)
    assert state.valid_output.all()
    np.testing.assert_array_equal(state.output[state.valid], state.value[state.valid])
    assert state.completion["method"][0] == "hold_next"
    assert state.completion["method"][-1] == "hold_previous"
    assert state.completion["gap_level"][4] == "short"
    assert state.completion["gap_level"][8] == "medium"
    assert state.completion["gap_level"][23] == "long"
    assert state.completion["gap_frames"][23] == 37
    assert state.completion["gap_duration_s"][23] == pytest.approx(37 / 30)
    assert state.completion["anchor_span_s"][23] == pytest.approx(38 / 30)
    assert state.completion["filled_frames"] == len(missing)


def test_completion_single_anchor_succeeds_but_all_missing_writes_no_state(
    tmp_path, monkeypatch
):
    _fake_video(monkeypatch, _frames(5, drop_at={0, 1, 3, 4}))
    result = run_mod.run_clip(_job(tmp_path), write_trace=False)
    assert result.n_valid == 1
    assert result.state.valid_output.all()
    assert result.state.completion["filled_frames"] == 4

    _fake_video(monkeypatch, _frames(5, drop_at=set(range(5))))
    job = _job(tmp_path / "missing")
    with pytest.raises(ValueError, match="no usable EEF samples"):
        run_mod.run_clip(job, write_trace=False)
    assert not (job.out_dir / layout.EEF_STATE).exists()


def test_run_clip_trace_is_lazy(tmp_path, monkeypatch):
    _fake_video(monkeypatch, _frames(4))
    res = run_mod.run_clip(_job(tmp_path), write_trace=True)
    skipped = any("matplotlib" in n for n in res.notes)
    assert (res.trace_path is None) == skipped
    if res.trace_path is not None:
        sdir = layout.stream_dir(tmp_path / "ep", "left")
        assert res.trace_path == sdir / layout.EEF_TRACE
        assert res.trace_path.is_file()


def test_run_clip_without_out_dir_writes_the_stem_suffix_beside_the_clip(
    tmp_path, monkeypatch
):
    """The standalone convention: no ``out_dir``, so the artifacts take the
    recording's stem next to it instead of layout's fixed names."""
    _fake_video(monkeypatch, _frames(4))
    rec, lens0 = _bare_clip(tmp_path)
    res = run_mod.run_clip(
        _job(tmp_path, recording=rec, lens0_mp4=lens0, out_dir=None),
        write_trace=True,
    )

    assert res.state_path == rec.with_name("VID_009_left_eef.npz")
    assert res.state_path.is_file()
    assert read_eef_state(res.state_path).source == lens0.name
    if res.trace_path is not None:
        assert res.trace_path == rec.with_name("VID_009_left_eef.png")
    # Nothing landed in the episode tree the fixture also built.
    assert not list(layout.stream_dir(tmp_path / "ep", "left").glob("eef.*"))


def test_run_clip_missing_lens0_names_the_convert_stage(tmp_path, monkeypatch):
    _fake_video(monkeypatch, _frames(2))
    job = _job(tmp_path)
    job.lens0_mp4.unlink()
    with pytest.raises(FileNotFoundError, match="convert stage"):
        run_mod.run_clip(job)


def test_run_clip_is_tool_blind(tmp_path, monkeypatch):
    """A dummy second bundle drives the whole runner: constant width in
    metres, its own detail namespace - and nothing chopsticks-shaped."""

    class DummyEstimator:
        params = None
        pose_mode = "marker_pnp"
        pose_source = ""
        warnings = ["dummy warning"]

        def __call__(self, rectified, detections, K):
            return GripperState(value=0.012, n_detected=1, n_kept=1, has_pose=True)

    def _postprocess(states, params, fps):
        n = len(states)
        return {"width_source": np.array(["dummy"] * n)}, ["postprocessed"]

    dummy = kiwi.eef.ToolEstimator(
        name="dummy",
        unit="m",
        make_frame_estimator=lambda ctx: DummyEstimator(),
        make_detector=lambda ctx: None,
        postprocess=_postprocess,
        closed_open=lambda block: (0.0, 0.024),
    )
    monkeypatch.setitem(kiwi.eef._ESTIMATORS, "dummy", lambda: dummy)

    _fake_video(monkeypatch, [np.zeros((H, W, 3), np.uint8)] * 3)
    res = run_mod.run_clip(
        _job(tmp_path, estimator_key="dummy", tool="parallel-proto"),
        write_trace=False,
    )
    s = read_eef_state(res.state_path)
    assert (s.tool, s.estimator, s.unit) == ("parallel-proto", "dummy", "m")
    np.testing.assert_allclose(s.value, 0.012)
    assert list(s.detail) == ["width_source"]
    assert "dummy warning" in res.notes and "postprocessed" in res.notes
    with np.load(res.state_path) as data:
        assert "dummy/width_source" in data.files
        assert not any(k.startswith("chopsticks/") for k in data.files)
