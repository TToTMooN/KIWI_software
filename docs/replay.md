# Replay and Gaussian scenes

## Replay

`kiwi replay` replays an episode's tool-tip trajectories kinematically on two FR3 arms (URDF/MuJoCo, no ROS) and reports whether a given robot placement can follow them.

```bash
pixi run -e replay kiwi replay solve data/ep1 data/ep2     # batch IK, metrics, <episode>/replay_solve.npz
pixi run -e replay kiwi replay run data/ep1 data/ep2       # interactive viewer
pixi run -e replay kiwi replay view fr3_chopstick_sa       # inspect a robot description
```

Each episode needs a `replay_state.npz` (`kiwi export state`); `solve` builds a missing or outdated one.
`solve` prints per-episode tracking error, IK failures, collisions and continuity, and writes the joint trajectories and settings to `replay_solve.npz`.
The default rig places the two arm bases 0.805 m apart; `--workspace-x/y/z`, `--ws-yaw/pitch/roll`, `--base-*`, `--ik-*`, `--offset` and `--traj-yaw` change it.
All episodes of one run come from the same scene and map set.

The viewer plays the solved trajectories and lets the bases, workspace, gripper mount and joints be moved, with IK re-solved on demand.

### Robot descriptions

| Model | Use |
| --- | --- |
| `chopstick_dc` | handheld capture gripper; its URDF gives export the camera-to-tip transform |
| `chopstick_sa` | servo-actuated gripper for the robot |
| `fr3_chopstick_sa`, `fr3_dual_chopstick_sa` | FR3 with the SA gripper, single and dual arm |

Sources and generated files live in `descriptions/` (see [PROVENANCE](https://github.com/TToTMooN/KIWI_software/blob/main/descriptions/PROVENANCE.md)).
After changing a source, regenerate the URDF and MuJoCo models with `pixi run -e replay python -m kiwi.replay.generate` (`--check` verifies the generated files).

The DC `camera_lens0` frame is not a calibration; export needs a calibrated `camera_lens0` in the source URDF or a `build.mount_cam` override in the gripper unit.
Replay maps the normalized opening onto the SA jaw's CAD travel `[-0.872665, 0]` rad.

## Gaussian scenes

`kiwi real2simviz` renders a static, metric Gaussian scene asset with NVIDIA's [3DGRUT](https://github.com/nv-tlabs/3dgrut) renderer and overlays the recorded wrist trajectories.

### Install

Linux x86-64 with an NVIDIA GPU, the CUDA 12.8 toolkit, a C++ compiler, Git, curl and uv:

```bash
bash third_party/setup_real2sim.sh        # --cuda-home <path> if CUDA 12.8 is not on PATH
```

The renderer is installed in `build/3dgrut/.venv` (`KIWI_3DGRUT` or `--backend-repo` selects another location).

### View

```bash
pixi run -e state -- kiwi real2simviz <asset> --port 8895
pixi run -e state -- kiwi real2simviz <asset> --episodes data/room/ep3 --port 8895   # another processed episode of the same scene
pixi run -e state -- kiwi real2simviz <asset> --views views.json --output renders    # render saved views to PNG
```

The command runs from any KIWI environment and starts the renderer inside `build/3dgrut/.venv`.

Open `http://127.0.0.1:8895`; the left hand is blue and the right green.
**Save PNG** and **Save view** write to the asset's `viewer/` directory.

### Asset

```text
<asset>/
  run.json        entry point: names the dataset, the checkpoint and the trajectories
  dataset/        images, cameras, sparse points and manifest.json, whose digest the checkpoint's training run.json must match
  training/       the 3DGRUT checkpoint and its training run.json
  trajectories/   left/right trajectories and fixed transforms
  viewer/         captures and saved views, created on first save
```

Assets are produced by KIWI's scene reconstruction pipeline, which is not part of this release. One asset is published: the scene of the public sample take from the README, with that take's wrist trajectories ([Drive](https://drive.google.com/file/d/1IlMvveFabSqR9wCnaKE8YnfTOmEG0ARk/view), 509 MB, SHA-256 `7a41144ee862a82afba5ae8da68f942301ba23d3b73942fed0d65fefdab17ee9`):

```bash
gdown 1IlMvveFabSqR9wCnaKE8YnfTOmEG0ARk -O kiwi-night-scene-asset.tar.gz
sha256sum kiwi-night-scene-asset.tar.gz
tar xzf kiwi-night-scene-asset.tar.gz
pixi run -e state -- kiwi real2simviz kiwi-night-scene-asset --port 8895
```
