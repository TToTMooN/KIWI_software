"""Copy the generation records and numerical evidence used by an export.

The copied documents retain their original contents and producers. The index
binds their original paths/bases to archive paths, so relative links remain
resolvable after a mutable source directory is rerun.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import yaml

from kiwi.core.digest import digest_canonical_json, verify_digest
from kiwi.core.generation import file_ref

_RECORD_SUFFIXES = {".json", ".yaml", ".yml"}
_QUALITY_SUFFIXES = {".npz", ".csv", ".txt", ".tum", ".log"}


class SourceArchive:
    """One dataset's content-deduplicated record and quality archive."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.records: list[dict] = []
        self.quality: list[dict] = []
        # A reference's expectation is separate from the bytes found today.
        # Two consumers may name different generations at the same path.
        self._visited: dict[tuple[str, str, str | None], dict] = {}
        self._versions: dict[tuple[str, str], dict] = {}
        self._refs: dict[Path, dict] = {}

    def _remember_version(self, path: Path, entry: dict) -> None:
        """Resolve pending historical references when a snapshot supplies them."""
        version = dict(entry)
        version.pop("expected_digest", None)
        version.pop("status", None)
        self._versions[(str(path), entry["digest"])] = version
        for (source, _, expected), previous in self._visited.items():
            if (
                source == str(path)
                and expected == entry["digest"]
                and previous.get("status") in {"missing", "source_version_changed"}
            ):
                identity = {
                    name: previous[name]
                    for name in ("original_path", "original_base", "expected_digest")
                }
                previous.clear()
                previous.update(version)
                previous.update(identity)

    def reference(self, path: Path, *, record: Path | None = None) -> dict:
        path = Path(os.path.abspath(path))
        if path not in self._refs:
            self._refs[path] = file_ref(path)
        ref = dict(self._refs[path])
        if record is not None:
            ref["record"] = self.reference(record)
        return ref

    def add(
        self,
        path: Path,
        *,
        episode_dir: Path,
        episode_index: int,
        original_path: str | None = None,
        original_base: Path | None = None,
        expected_digest: str | None = None,
        content_base: Path | None = None,
    ) -> dict | None:
        path = Path(os.path.abspath(path))
        if path.suffix.lower() not in _RECORD_SUFFIXES | _QUALITY_SUFFIXES:
            return None  # Raw video, atlas binaries, engines and models stay at source.
        base = Path(original_base or path.parent).absolute()
        original = original_path or str(path)
        saved = (
            self._versions.get((str(path), expected_digest))
            if expected_digest
            else None
        )
        ref = saved or self.reference(path)
        digest = ref.get("digest")
        key = (str(path), str(digest), expected_digest)
        if key in self._visited:
            return self._visited[key]
        entry = {
            **(saved or {}),
            "original_path": original,
            "original_base": str(base),
            "digest": digest,
            "archived_path": saved["archived_path"] if saved else None,
            "content_base": saved["content_base"]
            if saved
            else str(Path(content_base or path.parent).absolute()),
        }
        if expected_digest is not None:
            entry["expected_digest"] = expected_digest
        self._visited[key] = entry
        is_record = path.suffix.lower() in _RECORD_SUFFIXES
        index = self.records if is_record else self.quality
        index.append(entry)
        if saved:
            return entry
        if not path.is_file():
            entry["status"] = "missing"
            return entry
        if expected_digest:
            try:
                verify_digest(expected_digest, path)
            except (ValueError, TypeError, yaml.YAMLError, BadZipFile, EOFError):
                entry["status"] = "source_version_changed"
        token = str(digest).split(":")[-1]
        if is_record:
            destination = self.root / "meta" / "source_records" / token / path.name
            entry["record_id"] = token
        else:
            try:
                relative = path.relative_to(Path(episode_dir).absolute())
                destination = (
                    self.root / "meta" / "quality" / f"{episode_index:06d}" / relative
                )
            except ValueError:
                # Shared maps/calibration evidence are stored only once, even
                # if many source episodes reference the same version.
                destination = (
                    self.root / "meta" / "quality" / "shared" / token / path.name
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        entry["archived_path"] = str(destination.relative_to(self.root))
        self._remember_version(path, entry)
        if is_record:
            try:
                content = (
                    json.loads(path.read_text())
                    if path.suffix.lower() == ".json"
                    else yaml.safe_load(path.read_text())
                )
            except (ValueError, yaml.YAMLError, UnicodeError):
                entry["references_status"] = "unreadable_record"
                return entry
            if isinstance(content, dict):
                self._walk_record(
                    content,
                    path=path,
                    base=Path(content_base or path.parent).absolute(),
                    episode_dir=episode_dir,
                    episode_index=episode_index,
                )
        return entry

    def _walk_record(
        self,
        doc: dict,
        *,
        path: Path,
        base: Path,
        episode_dir: Path,
        episode_index: int,
    ):
        snapshots = doc.get("input_snapshots") or {}
        snapshot_bases = doc.get("input_snapshot_bases") or {}
        # A scene publication can move after the consuming state was solved.
        # Prefer the producer's frozen document while keeping its original
        # path base: its own references still use that original directory.
        for name, snapshot in snapshots.items():
            input_ref = doc.get("inputs", {}).get(name, {}) or {}
            if isinstance(snapshot, dict) and "path" not in snapshot:
                origin = self._resolve(input_ref.get("path", name + ".json"), base)
                content_base = (
                    self._resolve(snapshot_bases[name], base)
                    if name in snapshot_bases
                    else origin.parent
                )
                digest = digest_canonical_json(snapshot)
                key = (str(origin), digest, input_ref.get("digest"))
                if key not in self._visited:
                    token = digest.split(":")[-1]
                    destination = (
                        self.root / "meta" / "source_records" / token / origin.name
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False)
                        + "\n"
                    )
                    entry = {
                        "record_id": token,
                        "original_path": str(origin),
                        "original_base": str(content_base),
                        "content_base": str(content_base),
                        "digest": digest,
                        "archived_path": str(destination.relative_to(self.root)),
                        "snapshot_from": str(path),
                    }
                    if input_ref.get("digest"):
                        entry["expected_digest"] = input_ref["digest"]
                        if input_ref["digest"] != digest:
                            entry["status"] = "snapshot_digest_mismatch"
                    self._visited[key] = entry
                    self.records.append(entry)
                    self._remember_version(origin, entry)
                    self._walk_record(
                        snapshot,
                        path=origin,
                        base=content_base,
                        episode_dir=episode_dir,
                        episode_index=episode_index,
                    )
                continue
            snapshot_path = (
                snapshot.get("path") if isinstance(snapshot, dict) else snapshot
            )
            if not isinstance(snapshot_path, str):
                continue
            target = self._resolve(snapshot_path, base)
            saved_base = snapshot_bases.get(name)
            self.add(
                target,
                episode_dir=episode_dir,
                episode_index=episode_index,
                original_base=base,
                original_path=(doc.get("inputs", {}).get(name, {}) or {}).get(
                    "path", str(target)
                ),
                content_base=self._resolve(saved_base, base)
                if saved_base
                else target.parent,
                expected_digest=snapshot.get("digest")
                if isinstance(snapshot, dict)
                else None,
            )

        def visit(value, field: str = "", inherited_base: Path = base):
            if field.rsplit(".", 1)[-1].lower() in {
                "model",
                "models",
                "vocabulary",
                "weights",
                "mask_model",
            }:
                return
            if isinstance(value, dict):
                explicit_base = value.get("base") or value.get("original_base")
                local_base = (
                    self._resolve(explicit_base, inherited_base)
                    if isinstance(explicit_base, str)
                    else inherited_base
                )
                location = value.get("path")
                if isinstance(location, str):
                    target = self._resolve(location, local_base)
                    self.add(
                        target,
                        episode_dir=episode_dir,
                        episode_index=episode_index,
                        original_path=location,
                        original_base=local_base,
                        expected_digest=value.get("digest")
                        or self._stamp_digest(value, target),
                    )
                for name, child in value.items():
                    if name in {
                        "path",
                        "base",
                        "original_base",
                        "input_snapshots",
                        "input_snapshot_bases",
                    }:
                        continue
                    if field == "inputs" and name in snapshots:
                        continue
                    child_base = local_base
                    if not field and name == "inputs":
                        if doc.get("input_files"):
                            continue
                        if isinstance(doc.get("inputs_base"), str):
                            child_base = self._resolve(doc["inputs_base"], base)
                    visit(child, f"{field}.{name}" if field else name, child_base)
            elif isinstance(value, list):
                for child in value:
                    visit(child, field, inherited_base)
            elif (
                isinstance(value, str)
                and Path(value).suffix.lower() in _RECORD_SUFFIXES | _QUALITY_SUFFIXES
            ):
                # Some records and configs name files by bare path (a
                # camera.yaml artifact pointer, for one).
                target = self._resolve(value, inherited_base)
                if target.is_file():
                    self.add(
                        target,
                        episode_dir=episode_dir,
                        episode_index=episode_index,
                        original_path=value,
                        original_base=inherited_base,
                    )

        visit(doc)

    @staticmethod
    def _stamp_digest(value: dict, path: Path) -> str | None:
        """Type a config_hash stamp's bare ``sha256`` by file suffix (not always a byte hash)."""
        sha = value.get("sha256")
        if not isinstance(sha, str):
            return None
        kind = {
            ".yaml": "canonical_yaml_sha256",
            ".yml": "canonical_yaml_sha256",
            ".npz": "npz_semantic_sha256",
        }.get(path.suffix.lower(), "bytes_sha256")
        return f"{kind}:{sha}"

    @staticmethod
    def _resolve(location: str, base: Path) -> Path:
        candidate = Path(location)
        return Path(
            os.path.abspath(candidate if candidate.is_absolute() else base / candidate)
        )

    def episode(self, source, observation, index: int) -> dict:
        """Archive this episode's actual selected path and return its input index."""
        episode = Path(source.episode_dir).absolute()
        inputs = {}

        def add(name, path, *, record=None):
            if path is None:
                return
            path = Path(path)
            inputs[name] = self.reference(path, record=record)
            self.add(path, episode_dir=episode, episode_index=index)
            if record is not None:
                self.add(record, episode_dir=episode, episode_index=index)

        for filename in (
            "episode.yaml",
            "sync.inputs.json",
            "run_status.json",
            "pipeline_run_status.json",
        ):
            path = episode / filename
            if path.is_file():
                add(filename, path)
        # Registry and gripper names in episode.yaml are not file references.
        # Retain their complete local config tree too, including the dated
        # artifacts those registries select and the A/V input stamps.
        for path in sorted((episode / "config").rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in _RECORD_SUFFIXES | _QUALITY_SUFFIXES
            ):
                add(str(path.relative_to(episode)), path)
        world_source = observation.meta.get("world_source", "markerless")
        for arm in source.arms:
            label = arm.out_label
            stream = episode / arm.label if arm.label else episode
            markerless = world_source == "markerless"
            pose = arm.state_tum if markerless else arm.register_tum
            record = (
                arm.state_inputs
                if markerless
                else (Path(pose).with_name("register.inputs.json") if pose else None)
            )
            add(f"{label}.pose", pose, record=record)
            add(f"{label}.eef", arm.eef_state, record=stream / "eef.inputs.json")
            add(f"{label}.video", arm.lens0, record=stream / "convert.inputs.json")
            for key in ("intrinsics", "intrinsics_kb4", "override_source"):
                add(f"{label}.{key}", getattr(arm, key))
            if markerless:
                scene_dir = Path(arm.state_tum).parent
                candidates = [
                    scene_dir / "state_latent.npz",
                    scene_dir / "state_diagnostics.npz",
                ]
                for role in (f"front_{arm.label}", "shared_rear"):
                    candidates += [
                        scene_dir / f"{role}{suffix}"
                        for suffix in (
                            "_observations.txt",
                            "_observations.inputs.json",
                            ".engine.log",
                        )
                    ]
            else:
                # The marker record recursively archives inputs.state and its
                # generation record; only the detections beside it are added here.
                candidates = [stream / "detections.json"]
            for path in candidates:
                if path.is_file():
                    self.add(path, episode_dir=episode, episode_index=index)
        selected_cameras = set(observation.meta.get("cameras", {}))
        for camera in source.cameras:
            if camera.label in selected_cameras:
                add(
                    f"{camera.label}.video",
                    camera.video,
                    record=episode / camera.label / "convert.inputs.json",
                )
                for key in ("intrinsics", "intrinsics_kb4"):
                    add(f"{camera.label}.{key}", getattr(camera, key))
        if observation.meta.get("head_marker") and source.head is not None:
            add("head.trajectory", source.head.trajectory, record=source.head.report)
            for filename in (
                "head_marker_observations.npz",
                "head_marker_diagnostics.npz",
            ):
                path = Path(source.head.report).parent / filename
                if path.is_file():
                    self.add(path, episode_dir=episode, episode_index=index)
        if observation.meta.get("ground") and source.ground is not None:
            add("ground.report", source.ground.report)
            observations = source.ground.observations
            if observations is not None and Path(observations).is_file():
                self.add(observations, episode_dir=episode, episode_index=index)
        return inputs

    def samples(self, observation, index: int) -> dict | None:
        arrays = observation.meta.get("export_samples")
        if arrays is None:
            return None
        target = self.root / "meta" / "quality" / f"{index:06d}" / "export_samples.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        ref = {**file_ref(target, base=self.root), "rows": len(observation.timestamps)}
        self.quality.append(
            {
                "original_path": None,
                "original_base": None,
                "archived_path": ref["path"],
                "digest": ref["digest"],
                "generated_by": "export",
                "episode_index": index,
            }
        )
        return ref
