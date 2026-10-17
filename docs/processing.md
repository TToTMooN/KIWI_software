# Collect and process

## Collect

### Record

1. X5 at 4K 120 fps, GO 3 at Freeform Video 50 fps, lenses clean.
2. Play a shared sound from a speaker.
3. Say "start recording", do the task, say "stop recording".

### Import

Connect each camera in file-transfer mode and run:

```bash
kiwi import --dest data/demo_session
```

A picker opens at `http://localhost:8765` with the latest clips; select the clips and press `Import`.
Each clip is copied into `--dest` and named after the camera unit its serial resolves to.
Over ssh, forward the port (`ssh -L 8765:localhost:8765 <host>`) and pass `--no-open-browser`; `--no-pick` takes the newest clip.

### Episodes

Put the clips of one take into an episode's `raw/` directory:

```
data/demo_session/
  ep1/
    raw/
      VID_20260724_173320_00_050_insta360-X5-001.insv
      VID_20260724_173318_00_070_insta360-X5-003.insv
      PRO_VID_20260820_160036_00_046_ego_video.mp4
  ep2/
    raw/
      ...
```

Then write each `episode.yaml`:

```bash
kiwi init --root data/demo_session \
  --left insta360-X5-001 --right insta360-X5-003 \
  --camera ego_video=insta360-GO3-001 \
  --head-marker-id 50 \
  --tool chopstick
```

| Option | Meaning |
| --- | --- |
| `--episode` / `--root` | one episode, or every episode under a directory |
| `--left`, `--right` | camera units of the two arms; each clip is checked against the unit's serial |
| `--camera <label>=<unit>` | unit of an extra stream, found by its `_<label>` file name tag |
| `--head-marker-id` | id of the hat marker (`--head-marker-dictionary`, `--head-marker-size` for other tags) |
| `--tool` | gripper design under `config/gripper/` |
| `--force` | overwrite an existing `episode.yaml` |

Fill in `description`, `date` and each arm's `gripper_unit:` in `episode.yaml`, then continue with the [Pipeline](#pipeline).

## Pipeline

`kiwi run` processes one episode through a chain of stages. The default profile, `global-state`, runs:

```
sync → convert → state → eef
```

`state` needs the episode's scene maps (see [Scene mapping](estimation.md#scene-mapping-and-state)); `kiwi scene init` links each episode to its scene.

| Stage | Reads | Writes |
| --- | --- | --- |
| `sync` | raw recording audio, camera A/V tables | global clock in `episode.yaml`, `sync.inputs.json` |
| `convert` | raw `.insv` recordings | `<label>/lens0.mp4`, `lens1.mp4`, `convert.inputs.json` |
| `state` | scene maps, front/rear video, raw IMU, calibration, clock | `<arm>/scene/state.tum`, `state.inputs.json`, `state_latent.npz` |
| `head` (optional) | arm states, rear videos, head-marker config | `head/head_marker.tum`, report, diagnostics |
| `eef` | front video, gripper configuration | `<arm>/eef.npz`, `eef.inputs.json` |
| `ground` (optional) | arm states, front videos | `ground/ground_report.json`, `ground/ground_observations.npz` |

| Profile | Stages beyond sync, convert and eef |
| --- | --- |
| `prepare` | none; run before the scene maps exist |
| `global-state` (default) | state |
| `global-state-head` | state, head |
| `global-state-ground` | state, ground |
| `global-state-head-ground` | state, head, ground |

```bash
pixi run -e state -- kiwi run --episode <ep>                                   # default profile
pixi run -e state -- kiwi run --episode <ep> --profile global-state-head
pixi run -e ground -- kiwi run --episode <ep> --from-stage ground --to-stage ground
```

The run stops at the first failed stage.
`run_status.json` records the stages that started; `pipeline_run_status.json` records an outcome for every stage of the requested range.

### Episode configuration

Before the first run, KIWI copies the deployment config tree (`$KIWI_CONFIG`, or the checkout's `config/`) into `<episode>/config/`.
Later runs and export read this copy; editing the checkout does not change an existing episode.

### Synchronization

Record a distinctive sound that every camera hears.
`sync` aligns the streams on that audio and converts the audio offset into a video offset with each camera's per-mode A/V table (`config/cameras/<unit>/av-sync.yaml`).
In a session with the GO 3, the X5 arms record `1920x1920@119.88`.

A run reuses the clock saved in `episode.yaml`.
Pass `--resync` to estimate a new one after changing recordings, camera assignments or calibration:

```bash
kiwi run --episode <ep> --resync
```

Task and curation windows are not moved when the clock changes.

## Inspect

`kiwi viz` shows the artifacts of a processed episode on the episode's global clock.

```bash
pixi run -e viz kiwi viz play <episode>                       # Rerun viewer
pixi run -e viz kiwi viz play <episode> --save episode.rrd    # reopen with `rerun episode.rrd`
pixi run -e viz kiwi viz render <episode>                     # <episode>/synced_global.mp4
pixi run -e viz kiwi viz trace <episode>                      # <episode>/tool_state.png
```

`play` shows each stream's video, each arm's `state.tum` path and moving front camera, and the raw and completed tool state from `eef.npz`.
HEVC streams get a 480 px H.264 rendition at `<label>/proxy.mp4` on first use (`--no-proxies` skips them, `--force-proxies` re-encodes).

`render` writes all streams side by side with the global time and a tool-state strip; `--streams` picks and orders the panels, `--t-start` / `--t-end` narrow the window.
`trace` plots every arm's tool state on the global clock with the `task_window` shaded; it accepts several episodes and `--output-dir`.
