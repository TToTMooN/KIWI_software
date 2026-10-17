"""Tool-blind batch driver: one clip -> ``eef_state_v1`` (+ trace, overlay).

Everything tool-specific comes through the :class:`kiwi.eef.ToolEstimator`
bundle resolved from the job's ``estimator_key``; this module never mentions
angles, dots, or jaws. The whole loop is: rectify each frame, feed the
bundle's detector + frame estimator, run the bundle's whole-series
post-passes, complete the surviving raw estimates, and write the state artifact.
Optional trace and overlay outputs accompany it. Inside an episode the paths
are ``eef.npz`` / ``eef.png`` / ``eef.mp4`` in the arm's stream directory
(the job's ``out_dir``, per :mod:`kiwi.core.layout`); standalone, with no
``out_dir``, it is ``<stem>_eef.*`` beside the recording:

- ``eef.npz`` - the contract (:mod:`kiwi.eef.state`), written on success.
- ``eef.png`` - the raw + completed trace, enabled by ``write_trace``.
  Matplotlib is loaded lazily; the trace is skipped with a note when absent.
- ``eef.mp4`` - the annotated overlay, opt-in and default OFF: it is
  a diagnostic video that no downstream computation reads.

The episode stage requests the state and trace, without enabling the overlay.

Completion preserves finite raw anchors, interpolates interior gaps and
holds the nearest anchor at the edges. A clip with no finite raw anchor
raises before writing its state artifact. Callers use :class:`ClipResult`,
which also carries the estimator's warnings and post-pass notes, including
pose-mode selection and symmetry-axis calibration results.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from kiwi.core import layout
from kiwi.core.generation import json_value
from kiwi.core.video import VideoInfo, iter_frames
from kiwi.eef import EstimatorContext, get_estimator
from kiwi.eef.completion import complete_series
from kiwi.eef.state import (
    EEF_STATE_SUFFIX,
    OVERLAY_SUFFIX,
    TRACE_SUFFIX,
    EefState,
    write_eef_state,
)

__all__ = ["ClipJob", "ClipResult", "run_clip"]


def _artifact(job: ClipJob, fixed_name: str, suffix: str) -> Path:
    """One artifact's path: fixed name in ``out_dir``, else ``<stem><suffix>``
    beside the recording (the standalone convention)."""
    if job.out_dir is not None:
        return Path(job.out_dir) / fixed_name
    return job.recording.with_name(job.recording.stem + suffix)


@dataclasses.dataclass(frozen=True)
class ClipJob:
    """A fully resolved single-clip run.

    Built by the stage entry (from the episode manifest's ``ArmSpec``, which
    has already layered tool < unit < episode-arm) or by the CLI verb (from
    the tool / unit YAMLs directly). ``out_dir`` is where the artifacts land
    (the arm's stream directory, fixed names per :mod:`kiwi.core.layout`);
    when ``None`` - a standalone run on a bare clip - they land beside the
    recording as ``<recording stem>_eef.*``. ``lens0_mp4`` is the stream
    actually decoded, already demuxed by the convert stage.
    """

    recording: Path
    lens0_mp4: Path
    tool: str
    estimator_key: str
    session_block: Mapping[str, object]
    intrinsics: Path
    out_dir: Path | None = None
    marker_id: int | None = None
    marker_size_m: float | None = None
    marker_dictionary: str | None = None
    fixed_pose: Mapping[str, object] | None = None
    fixed_pose_source: str = ""
    lens: str = "front"
    out_size: tuple[int, int] = (1920, 1920)
    live_pnp: bool = False
    overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ClipResult:
    state_path: Path
    state: EefState
    trace_path: Path | None
    overlay_path: Path | None
    notes: list[str]
    generation: dict = dataclasses.field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(self.state.valid.sum())

    def summary(self) -> str:
        s = self.state
        n = len(s)
        head = (
            f"{self.state_path.name}: {self.n_valid}/{n} frames valid, "
            f"{s.pose_mode}"
            + (f" from {s.pose_source}" if s.pose_source else "")
            + f"; {int(s.valid_output.sum())}/{n} completed"
        )
        return "\n".join([head, *self.notes]) if self.notes else head


def run_clip(
    job: ClipJob,
    *,
    write_trace: bool = True,
    write_overlay: bool = False,
    on_progress=None,
) -> ClipResult:
    """Run the job's estimator over every frame and write the artifacts.

    ``on_progress(done, total)`` (when given) is called every ~2 s of video.
    Raises for missing inputs, an unopenable stream, or no usable values.
    Completion needs at least one finite raw anchor; otherwise no new
    primary state artifact is written.
    """
    if not job.lens0_mp4.is_file():
        raise FileNotFoundError(
            f"no demuxed stream at {job.lens0_mp4} - the eef stage reads the "
            f"front-lens mp4 the convert stage produces; run `kiwi run` (or "
            f"point --lens0-mp4 at one)."
        )
    if not job.intrinsics.is_file():
        raise FileNotFoundError(f"no intrinsics npz at {job.intrinsics}")

    from kiwi.core.rectify import load_rectifier_auto

    bundle = get_estimator(job.estimator_key, source=f"tool '{job.tool}'")
    info = VideoInfo.read(job.lens0_mp4)
    # The rectifier FOV is part of the recipe (the fixed_pose is only valid
    # at the FOV it was calibrated at), so it resolves like every other knob:
    # override > gripper block > builtin.
    fov = job.overrides.get("fov_deg") or job.session_block.get("fov_deg") or 110.0
    rect = load_rectifier_auto(
        job.intrinsics,
        (info.width, info.height),
        lens=job.lens,
        out_size=job.out_size,
        fov_deg=float(fov),
    )
    ctx = EstimatorContext(
        session_block=job.session_block,
        K=rect.K_pinhole,
        image_wh=tuple(rect.out_size),
        marker_id=job.marker_id,
        marker_size_m=job.marker_size_m,
        marker_dictionary=job.marker_dictionary,
        fixed_pose=job.fixed_pose,
        fixed_pose_source=job.fixed_pose_source,
        live_pnp=job.live_pnp,
        overrides=job.overrides,
    )
    est = bundle.make_frame_estimator(ctx)
    detector = bundle.make_detector(ctx)
    notes = list(est.warnings)

    overlay_path = _artifact(job, layout.EEF_OVERLAY, OVERLAY_SUFFIX)
    writer = None
    if write_overlay:
        import cv2

        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            tuple(rect.out_size),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"overlay VideoWriter failed for {overlay_path} (mp4v); "
                f"re-run without --write-overlay"
            )

    from kiwi.core.pipeline import Detections

    empty = Detections(
        corners=np.zeros((0, 4, 2), np.float32), ids=np.zeros(0, np.int32)
    )
    states = []
    step = max(1, int(round(info.fps * 2)))
    try:
        for idx, bgr in iter_frames(job.lens0_mp4):
            rectified = rect.apply(bgr)
            dets = detector(rectified) if detector is not None else empty
            state = est(rectified, dets, rect.K_pinhole)
            states.append(state)
            if writer is not None:
                canvas = rectified.copy()
                if bundle.draw_overlay is not None:
                    bundle.draw_overlay(canvas, state)
                _overlay_header(canvas, idx, state)
                writer.write(canvas)
            if on_progress is not None and idx % step == 0:
                on_progress(idx, info.n_frames)
    finally:
        if writer is not None:
            writer.release()

    detail, post_notes = bundle.postprocess(states, est.params, info.fps)
    notes += post_notes

    # Complete the surviving raw estimates. The one-jaw delta correction and
    # the long-gap rescue have already run and remain auditable in the
    # estimator's own diagnostics. Completion adds no gates.
    raw = np.array([np.nan if s.value is None else float(s.value) for s in states])
    completed, completion, completion_notes = complete_series(
        raw,
        info.fps,
        short_max_s=getattr(est.params, "gap_short_max_s", 0.167),
        medium_max_s=getattr(est.params, "gap_medium_max_s", 1.0),
    )
    notes += completion_notes

    state = EefState(
        tool=job.tool,
        estimator=bundle.name,
        unit=bundle.unit,
        fps=info.fps,
        source=job.lens0_mp4.name,
        pose_mode=est.pose_mode,
        pose_source=est.pose_source,
        value=raw,
        rejected_value=np.array(
            [
                np.nan if s.rejected_value is None else float(s.rejected_value)
                for s in states
            ]
        ),
        has_pose=np.array([s.has_pose for s in states], dtype=bool),
        n_detected=np.array([s.n_detected for s in states], dtype=np.int32),
        n_kept=np.array([s.n_kept for s in states], dtype=np.int32),
        error=np.array([s.error or "" for s in states], dtype=str),
        detail=detail,
        value_completed=completed,
        completion=completion,
    )
    state_path = write_eef_state(
        _artifact(job, layout.EEF_STATE, EEF_STATE_SUFFIX), state
    )

    trace_path = None
    if write_trace:
        trace_path = _write_trace(
            _artifact(job, layout.EEF_TRACE, TRACE_SUFFIX),
            state,
            notes,
        )

    return ClipResult(
        state_path=state_path,
        state=state,
        trace_path=trace_path,
        overlay_path=overlay_path if writer is not None else None,
        notes=notes,
        generation={
            "params": json_value(est.params),
            "image_settings": {
                "lens": job.lens,
                "out_size": list(rect.out_size),
                "fov_deg": float(fov),
                "live_pnp": job.live_pnp,
            },
        },
    )


def _overlay_header(canvas: np.ndarray, idx: int, state) -> None:
    import cv2

    if state.value is not None:
        text, color = f"frame {idx}  value {state.value:6.2f}", (0, 255, 255)
    elif state.error:
        text, color = f"frame {idx}  {state.error}", (60, 60, 255)
    else:
        text, color = f"frame {idx}  no pose", (200, 200, 200)
    cv2.putText(
        canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def _write_trace(path: Path, state: EefState, notes: list[str]) -> Path | None:
    """The raw + output trace PNG, or None (with a note) sans matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        notes.append("(matplotlib not installed; skipping the trace png)")
        return None
    n = len(state)
    frames = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, state.value, color="C0", linewidth=1.0, label="raw")
    ax.plot(
        frames,
        state.output,
        color="C1",
        linewidth=1.0,
        alpha=0.8,
        label="completed",
    )
    ax.set_xlabel("frame")
    ax.set_ylabel(f"tool state ({state.unit})")
    ax.set_title(
        f"{state.source} - {int(state.valid.sum())}/{n} frames valid "
        f"({state.pose_mode})"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
