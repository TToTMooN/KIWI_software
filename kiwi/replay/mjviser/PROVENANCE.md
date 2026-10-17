# mjviser provenance

Vendored fork of [mujocolab/mjviser](https://github.com/mujocolab/mjviser)
(Apache-2.0, Copyright 2025 The mjlab Developers - the full license text is
in `LICENSE` beside this file).
Vendored on 2026-08-24 from the `robot_ik_analysis` sibling repository's fork,
whose upstream base was not recorded; the copied `conversions.py` and
`scene.py` are byte-identical to upstream at every commit from `1f0f61e`
(2026-04-02, the last change to either file before v0.0.13) through `64da06c`
(v0.0.13, 2026-04-27), so v0.0.13 is taken as the base. The local changes are
listed below.

What it is: the MuJoCo -> viser scene bridge the replay viewer renders
through. `conversions.py` extracts MuJoCo geoms/sites into trimesh meshes
(merging geoms that share a material so viser gets few large batched GLBs);
`scene.py`'s `ViserMujocoScene` builds the handles once and pushes batched
pose updates per frame (`update_from_mjdata`), plus the Scene /
Visualization / Groups overlay GUI tabs.

Vendoring notes:

- `viewer.py` and `__main__.py` (upstream's standalone simulate viewer and
  its CLI) are not carried: the replay viewer owns its own loop and drives
  `ViserMujocoScene` directly.
- Upstream's 2-space indentation is preserved; `kiwi/replay/mjviser/` is
  excluded from ruff.
- Upstream helpers nothing in KIWI calls (the batched-array update path, the
  refresh-handler hook and a few small utilities) are trimmed.
- Dependencies (mujoco, trimesh, pillow, viser) ship with the `replay`
  extra.
- KIWI preserves untextured mesh materials' MuJoCo roughness and metallic properties in PBR output, including the imported glossy black cameras.
