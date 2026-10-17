"""Generation details for the episode-stage sidecars."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

from kiwi.core import layout
from kiwi.core.generation import file_ref, json_value, producer_info


_EEF_SOURCE_DIR = Path(__file__).resolve().parents[1] / "eef"
# Every kiwi/eef module the eef stage imports, lazy imports included; a test
# holds this tuple to that import closure. homography.py stays out: only the
# fixed-pose calibration imports it, and the pose it writes enters the reuse
# key by content.
_EEF_SOURCES = (
    "__init__.py",
    "completion.py",
    "state.py",
    "run.py",
    "config.py",
    "chopsticks.py",
    "hinge.py",
    "dots.py",
)


def eef_source_fingerprint() -> dict:
    """Identify the completion policy and the kiwi/eef source by content.

    Shared ``kiwi.core`` helpers the estimator calls are not part of it.
    """
    return {
        "version": "eef_completion_v1",
        "sources": {
            name: hashlib.sha256((_EEF_SOURCE_DIR / name).read_bytes()).hexdigest()
            for name in _EEF_SOURCES
        },
    }


def eef_completion_summary(state) -> dict:
    """Report source-frame filling separately from measurement coverage."""
    import numpy as np

    audit = state.completion
    summary = {
        key: json_value(value) for key, value in audit.items() if np.ndim(value) == 0
    }
    filled = np.asarray(audit["filled"], dtype=bool)
    n_filled = int(filled.sum())
    summary.update(
        filled_frames=n_filled,
        filled_fraction=n_filled / len(state) if len(state) else 0.0,
    )
    if "gap_duration_s" in audit:
        summary["max_gap_s"] = float(np.max(audit["gap_duration_s"], initial=0.0))
    if "method" in audit:
        summary["method_counts"] = {
            str(method): int(count)
            for method, count in zip(
                *np.unique(np.asarray(audit["method"])[filled], return_counts=True)
            )
        }
    if "gap_level" in audit:
        summary["gap_level_counts"] = {
            level: int(
                np.count_nonzero(np.asarray(audit["gap_level"])[filled] == level)
            )
            for level in ("short", "medium", "long")
        }
    return summary


def sync_generation(cfg, *, estimate=None, error=None) -> dict:
    from kiwi.sync import SyncParams

    streams = {**cfg.arms, **cfg.cameras}
    doc = {
        "producer": producer_info(
            "audio_gcc_phat", dependencies=("numpy", "scipy", "ffmpeg", "ffprobe")
        ),
        "params": dataclasses.asdict(SyncParams()),
        "inputs": {
            "recordings": {
                label: file_ref(spec.recording, base=cfg.root)
                for label, spec in streams.items()
            }
        },
    }
    if error is not None:
        doc.update(
            status="failed",
            reason=str(error),
            diagnostics=getattr(error, "evidence", {}),
        )
    if estimate is not None:
        doc.update(
            status="computed",
            method=estimate.method,
            origin=estimate.clock.origin,
            offsets_s=estimate.clock.offsets_s,
            loop_residuals_s=estimate.loop_residuals,
            edges=[dataclasses.asdict(e) for e in estimate.edges],
        )
        doc["outputs"] = {"episode": file_ref(cfg.root / "episode.yaml", base=cfg.root)}
    return json_value(doc)


def conversion_generation(clip, outputs, *, base, label) -> dict:
    return {
        "schema": "conversion_inputs",
        "producer": producer_info("ffmpeg_demux", dependencies=("ffmpeg",)),
        "recipe": {
            "stream": label,
            "video_codec": "copy",
            "audio": False,
            "video_stream_indices": list(range(len(outputs))),
        },
        "inputs": {"recording": file_ref(clip, base=base)},
        "outputs": {
            f"lens{i}": file_ref(path, base=base) for i, path in enumerate(outputs)
        },
    }


def eef_generation(arm, directory, source, intrinsics, result) -> dict:
    directory = Path(directory)
    return json_value(
        {
            "producer": producer_info(
                arm.estimator or "unknown",
                dependencies=("opencv-python-headless", "numpy", "scipy"),
            ),
            **result.generation,
            "source_fingerprint": eef_source_fingerprint(),
            "completion": eef_completion_summary(result.state),
            "inputs": {
                "video": file_ref(
                    source,
                    base=directory,
                    record=directory / "convert.inputs.json"
                    if source != arm.recording
                    else None,
                ),
                "intrinsics": file_ref(intrinsics, base=directory),
                "tool": file_ref(arm.tool_config, base=directory),
                "fixed_pose": file_ref(arm.fixed_pose_source or None, base=directory),
            },
            "input_values": {
                "marker": arm.marker,
                "fixed_pose": arm.fixed_pose,
                "session": arm.session,
            },
            "outputs": {"eef": file_ref(result.state_path, base=directory)},
        }
    )


def registration_generation(
    result, *, directory, video, intrinsics, state_inputs, marker, params, error=None
) -> dict:
    from kiwi.register import health

    directory = Path(directory)
    return json_value(
        {
            "producer": producer_info(
                "marker_anchor",
                dependencies=("opencv-python-headless", "numpy", "scipy"),
            ),
            "params": {**params, "marker": marker},
            "inputs": {
                "video": file_ref(
                    video, base=directory, record=directory / "convert.inputs.json"
                ),
                "intrinsics": file_ref(intrinsics, base=directory),
                "state": file_ref(
                    directory / layout.SCENE_DIR / layout.STATE_TUM,
                    base=directory,
                    record=state_inputs,
                ),
                "detections": file_ref(directory / layout.DETECTIONS, base=directory),
            },
            "status": result.verdict[0] if result is not None else "failed",
            **({"reason": str(error)} if error is not None else {}),
            "T_track_marker": getattr(result, "T_track_marker", None),
            "diagnostics": {
                **getattr(result, "diagnostics", {}),
                **{
                    name: getattr(result, name, None)
                    for name in (
                        "n_detected",
                        "n_too_far",
                        "n_in_range",
                        "best_frame",
                        "best_reproj",
                        "spread",
                        "reach",
                    )
                },
                "verdict": result.verdict if result is not None else None,
            },
            "thresholds": {
                "displacement_warn_m": health.DISPLACEMENT_WARN_M,
                "displacement_halt_m": health.DISPLACEMENT_HALT_M,
                "reach_warn_m": health.REACH_WARN_M,
                "reach_halt_m": health.REACH_HALT_M,
                "comparison": ">",
            },
        }
    )
