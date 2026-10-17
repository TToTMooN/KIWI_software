"""The subprocess boundary around the ORB-SLAM3 session wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kiwi.core.paths import repo_root

ENGINE_THREADS = 10  # thread count of every engine session

BUILD_HELP = (
    "build it first:\n"
    "  git submodule update --init third_party/ORB_SLAM3\n"
    "  pixi install -e slam-offline\n"
    "  pixi run -e slam-offline build-orbslam3"
)


def _repo_root() -> Path:
    root = repo_root()
    if root is None:
        raise RuntimeError(
            "the ORB-SLAM3 engine is built inside a KIWI checkout, and this "
            "installation is not one - " + BUILD_HELP
        )
    return root


def engine_binary() -> Path:
    return _repo_root() / "build/orbslam3/wrapper/kiwi-orbslam-session"


def vocabulary() -> Path:
    return _repo_root() / "build/orbslam3/src/Vocabulary/ORBvoc.txt"


def _env() -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(ENGINE_THREADS)
    return env


def run_engine(
    mode: str,
    settings: Path,
    sequence_dir: Path,
    timestamps: Path,
    pose_out: Path,
    *,
    log_path: Path,
    atlas_in: Path | None = None,
    atlas_out: Path | None = None,
    mask_dir: Path | None = None,
) -> None:
    """Run one ``map`` or ``localize`` session; stdout and stderr go to ``log_path``."""
    cmd = [
        str(engine_binary()),
        "--mode",
        mode,
        "--vocabulary",
        str(vocabulary()),
        "--settings",
        str(Path(settings).resolve()),
        "--sequence-dir",
        str(Path(sequence_dir).resolve()),
        "--timestamps",
        str(Path(timestamps).resolve()),
        "--pose-observations",
        str(Path(pose_out).resolve()),
    ]
    if atlas_in is not None:
        cmd += ["--atlas-in", str(Path(atlas_in).resolve())]
    if atlas_out is not None:
        cmd += ["--atlas-out", str(Path(atlas_out).resolve())]
    if mask_dir is not None:
        cmd += ["--mask-dir", str(Path(mask_dir).resolve())]
    Path(pose_out).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as sink:
        subprocess.run(cmd, stdout=sink, stderr=subprocess.STDOUT, env=_env())
