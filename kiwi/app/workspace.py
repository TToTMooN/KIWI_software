"""Filesystem library and background jobs, independent of NiceGUI."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from kiwi.app.sources import preview_offset
from kiwi.app.worker import PROGRESS_PREFIX
from kiwi.core.atomic import atomic_write_text
from kiwi.core.camera.ingest import (
    Candidate,
    IngestError,
    copy_one,
    plan_jobs,
    resolve_unit,
)


def inside(root: Path, relative: str) -> Path:
    """Resolve a user-selected destination without escaping its data root."""
    if not relative.strip() or Path(relative).is_absolute():
        raise ValueError("Choose a relative folder inside the data root.")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("The destination must stay inside the data root.")
    return result


def scan_library(root: Path) -> tuple[list[dict], list[Path]]:
    """Read lightweight metadata; avoid walking raw media and derived trees."""
    episodes, recordings = [], []
    for folder, dirs, files in os.walk(root):
        directory = Path(folder)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        recordings.extend(directory / f for f in sorted(files) if f.endswith(".rrd"))
        manifest = directory / "episode.yaml"
        if "episode.yaml" not in files and "raw" not in dirs:
            dirs[:] = [d for d in dirs if d not in ("scene_map", "videos", "meta")]
            continue
        dirs[:] = []
        error, doc = "", {}
        if manifest.is_file():
            try:
                doc = yaml.safe_load(manifest.read_text())
                if doc is None:
                    doc = {}
                if not isinstance(doc, dict):
                    raise ValueError("Manifest must be a mapping")
                for key in ("arms", "cameras", "scene", "sync"):
                    if doc.get(key) is not None and not isinstance(doc[key], dict):
                        raise ValueError(f"Manifest {key} must be a mapping")
                for key in ("arms", "cameras"):
                    if not all(
                        isinstance(label, str) and isinstance(spec, dict)
                        for label, spec in (doc.get(key) or {}).items()
                    ):
                        raise ValueError(
                            f"Manifest {key} must map stream names to settings"
                        )
                offsets = (doc.get("sync") or {}).get("offsets")
                if offsets is not None and not isinstance(offsets, dict):
                    raise ValueError("Manifest sync.offsets must be a mapping")
            except (OSError, ValueError, yaml.YAMLError) as exc:
                error = str(exc)
                doc = {}
        streams = {**(doc.get("arms") or {}), **(doc.get("cameras") or {})}
        video_preview = any(
            (directory / label / "lens0.mp4").is_file()
            or (
                isinstance(spec.get("recording"), str)
                and (directory / spec["recording"]).is_file()
            )
            for label, spec in streams.items()
        )
        offsets = (doc.get("sync") or {}).get("offsets") or {}
        provisional = [
            name
            for name in (doc.get("cameras") or {})
            if name not in offsets and preview_offset(directory / name) is not None
        ]
        unsynced = [
            name for name in streams if name not in offsets and name not in provisional
        ]
        artifacts = []
        for label in doc.get("arms") or {}:
            for suffix, name in (("scene/state.tum", "state"), ("eef.npz", "tool")):
                if (directory / label / suffix).is_file():
                    artifacts.append(f"{label} {name}")
        for suffix, name in (
            ("head/head_marker.tum", "head"),
            ("ground/ground_report.json", "ground"),
        ):
            if (directory / suffix).is_file():
                artifacts.append(name)
        raw = directory / "raw"
        raw_count = sum(f.is_file() for f in raw.iterdir()) if raw.is_dir() else 0
        episodes.append(
            {
                "id": str(directory.relative_to(root)),
                "name": str(doc.get("name") or directory.name),
                "scene": str(
                    (doc.get("scene") or {}).get("manifest") or directory.parent.name
                ),
                "cameras": ", ".join(streams) or f"{raw_count} files",
                "available": ", ".join(artifacts) or "No derived results",
                "video_preview": video_preview,
                "status": "Manifest needs attention"
                if error
                else "Needs setup"
                if not manifest.is_file()
                else "Initialized",
                "timing": f"Missing sync: {', '.join(unsynced)}"
                if unsynced
                else f"Preview sync: {', '.join(provisional)}"
                if provisional
                else "Clock recorded"
                if streams
                else "Not initialized",
                "error": error,
            }
        )
    return episodes, recordings


@dataclasses.dataclass
class Job:
    id: str
    title: str
    status: str = "queued"
    message: str = "Waiting"
    progress: float = 0.0
    result: str = ""
    started: float = dataclasses.field(default_factory=time.time)


class Jobs:
    """One worker serializes filesystem mutations and survives page refreshes."""

    def __init__(self, cache: Path):
        self.directory = cache / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kiwi-gui")
        self.jobs: dict[str, Job] = {}
        for saved in sorted(self.directory.glob("*.json")):
            try:
                job = Job(**json.loads(saved.read_text()))
            except (ValueError, TypeError, OSError):
                continue
            self.jobs[job.id] = job
            if job.status in ("queued", "running"):
                self.update(
                    job,
                    status="interrupted",
                    message="The application stopped before this job finished.",
                )

    def update(self, job: Job, **fields):
        with self.lock:
            for name, value in fields.items():
                setattr(job, name, value)
            atomic_write_text(
                self.directory / f"{job.id}.json", json.dumps(dataclasses.asdict(job))
            )

    def submit(self, title: str, operation) -> Job:
        self.check_running()
        job = Job(uuid.uuid4().hex, title)
        with self.lock:
            self.jobs[job.id] = job
        self.update(job)

        def run():
            try:
                self.check_running()
                self.update(job, status="running", message="Starting")
                result = operation(job)
                self.update(
                    job, status="completed", progress=1.0, result=str(result or "")
                )
            except InterruptedError as exc:
                self.update(job, status="interrupted", message=str(exc))
            except Exception as exc:
                self.update(
                    job, status="failed", message=str(exc) or type(exc).__name__
                )

        self.pool.submit(run)
        return job

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [
                dataclasses.asdict(j)
                for j in sorted(
                    self.jobs.values(), key=lambda j: j.started, reverse=True
                )
            ]

    def close(self):
        self.stopping.set()
        with self.lock:
            for job in self.jobs.values():
                if job.status == "queued":
                    self.update(
                        job,
                        status="interrupted",
                        message="Application stopped before this job started.",
                    )
        self.pool.shutdown(wait=False, cancel_futures=True)

    def check_running(self):
        if self.stopping.is_set():
            raise InterruptedError("Application stopped. You can retry this job.")

    def run_process(
        self,
        command: list[str],
        log_path: Path,
        *,
        on_progress: Callable[[float, str], None] | None = None,
    ):
        """Stop the worker and any ffmpeg children when the application exits.

        The child reports progress as tagged lines on stdout, which lands in
        ``log_path``; the job's status is read back from there.
        """
        self.check_running()
        with log_path.open("w") as log, log_path.open("r") as tail:

            def drain():
                if on_progress is None:
                    return
                while True:
                    # A half-written line means the child is mid-print; rewind
                    # and take it whole on the next pass.
                    start = tail.tell()
                    line = tail.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        tail.seek(start)
                        break
                    if not line.startswith(PROGRESS_PREFIX):
                        continue
                    fraction, _, message = (
                        line[len(PROGRESS_PREFIX) :].strip().partition(" ")
                    )
                    try:
                        on_progress(float(fraction), message)
                    except ValueError:
                        continue

            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                while process.poll() is None:
                    drain()
                    if self.stopping.wait(0.25):
                        self.check_running()
                drain()
                if process.returncode:
                    raise RuntimeError(
                        log_path.read_text()[-2000:]
                        or f"Review worker exited with {process.returncode}."
                    )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def import_clips(
    clips: list[Candidate],
    destination: Path,
    cameras: Path,
    tag: str | None,
    jobs: Jobs,
    job: Job,
) -> Path:
    """Freeze and validate the whole copy plan before starting the first file."""
    if not clips:
        raise ValueError("Select at least one recording.")
    jobs.check_running()
    plan = []
    for clip in clips:
        unit = tag
        if not unit:
            try:
                unit = resolve_unit(clip.path, cameras)
            except IngestError:
                unit = None
        plan.extend(plan_jobs([clip], destination, unit))
    targets = [j.dst for j in plan]
    if len(set(targets)) != len(targets):
        raise ValueError("Selected recordings would use the same destination filename.")
    for item in plan:
        if item.src.resolve() == item.dst.resolve():
            raise ValueError("Source and destination are the same recording.")
        if item.dst.exists() and item.dst.stat().st_size != item.src.stat().st_size:
            raise ValueError(
                f"{item.dst.name} already exists with a different size. Choose another destination."
            )
    total = sum(item.src.stat().st_size for item in plan)
    finished = 0
    for item in plan:
        jobs.check_running()
        size = item.src.stat().st_size

        def tick(copied, size, elapsed):
            jobs.check_running()
            rate = copied / max(elapsed, 0.001) / 1e6
            jobs.update(
                job,
                progress=(finished + copied) / max(total, 1),
                message=f"{item.dst.name} · {rate:.1f} MB/s",
            )

        jobs.update(job, message=f"Copying {item.dst.name}")
        copy_one(item.src, item.dst, tick=tick)
        finished += size
    jobs.update(job, message=f"Imported {len(plan)} recording(s) into {destination}")
    return destination


class Files:
    """Expose only explicitly registered media, through opaque identifiers."""

    def __init__(self):
        self.paths: dict[str, Path] = {}

    def register(self, path: Path) -> str:
        path = path.resolve()
        # Rerun identifies HTTP recordings by the .rrd suffix before fetching.
        token = (
            hashlib.sha256(str(path).encode()).hexdigest()[:24] + path.suffix.lower()
        )
        self.paths[token] = path
        return f"/api/files/{token}"
