# Contributing to KIWI

Thanks for contributing.
The package contracts define where code belongs and the behavior it promises.

## Where changes go

- One component per PR where possible.
  The target package's `__init__.py` docstring is the contract - read it first.

## Dev setup

```bash
pixi install                 # default env
pixi run test                # pytest (hardware-free, must stay green)
pixi install -e dev
pixi run -e dev lint         # ruff check
pixi run -e dev fmt          # ruff format
pixi run -e dev pre-commit install   # one-time; local pass == CI pass
pixi install -e docs
pixi run -e docs docs-serve           # preview the docs site
pixi run -e docs docs-build           # strict build: broken links fail; run before a docs PR
```

## PR checklist

1. **Imports** - the boundary DAG holds (`tests/test_boundaries.py` will tell you): `core` imports nothing from kiwi; components import only `core`; composers never import each other.
   If two components need the same piece, promote it into `kiwi.core`.
   The unit sets are closed: a new top-level package or module under `kiwi/` is registered in that test before anything may import it.
2. **Stage entry** - if your component owns a pipeline stage, the entry point is written *in* `kiwi/episode/stages.py`, not in your package: it returns a `StageResult` (an episode type) and lands in that module's `REGISTRY`.
   Import your component lazily inside the function body, so `import kiwi.episode` stays cheap.
   Required argument: `episode_dir: Path`; return type: `StageResult`.
   Note the direction - a composer may import a component, never the reverse, so `from kiwi.episode.stages import REGISTRY` inside `kiwi/<component>/` is a boundary violation that `test_boundary_dag` will reject.
   See `_sync_entry` for the pattern.
   The lazy spelling is tested, not just asked for: `tests/test_import_cost.py` fails if `import kiwi.cli` drags in a component or its heavy dependencies.
3. **Tests come along** - pure numpy/scipy/yaml, fixed seeds, `tmp_path`; no hardware, ROS, or CV-on-real-images in CI.
4. **Conventions** -
   - new CLIs use **tyro** dataclasses, not argparse;
   - configs are **name-resolved** (each calibration file named by an explicit `camera.yaml` pointer - see `config/README.md`; match recordings to cameras by `serial:`, never by directory name);
   - layering is per BLOCK, not per key: the ladder (CLI flag / `--set` > episode-arm `session:` > unit config > tool default > builtin) applies to `session:` recipe keys only.
     `design:` facts live in the tool file and `build:` facts in the gripper unit - one home each, nothing shadows them, and a key in the wrong file is refused (`config/README.md`).
5. **License hygiene** - GPL engines (ORB-SLAM3; Kalibr is BSD but containerized for its ROS1 stack) stay at arm's length: submodule or container, invoked as separate processes, never imported/linked into this Apache-2.0 package.
   "This package" means `kiwi/` - first-party glue that *links* an engine (e.g. `cpp/orbslam_session/`) may exist outside it, carries its own license file, and is accounted for in `third_party/NOTICE`.
   The `third_party/` vs `cpp/` split is authorship, not license: `third_party/` holds code we do not write or maintain; `cpp/` holds code we do.
6. **Export stability** - field names in `kiwi/export/schema.py` are public API; changing one is a schema version bump, discussed first.
7. **Data honesty** - tool-state gap filling belongs upstream in EEF completion, with raw `value` unchanged and an explicit per-frame and whole-clip audit.
   Preserve every finite raw anchor exactly, interpolate interior missing runs, and hold the nearest anchor at the edges.
   Reject empty or all-missing clips without a default value.
   Export samples the completed channel and carries its audit; it adds no gap interpolation of its own.
   Keep completion provenance distinct from estimator inference, and bound export windows by the completed output and other required streams.
   See [Tool state](docs/estimation.md#output) for the artifact.

## Commit style

`feat(track): ...` / `fix(sync): ...` / `docs: ...` - scope = the kiwi package.
Keep PRs reviewable; link the related issue.
