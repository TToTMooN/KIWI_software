# Description provenance

The active models are Chopstick SA, Chopstick DC, and FR3 assemblies using SA.

## Chopstick grippers

`chopstick_sa` (servo actuation) and `chopstick_dc` (data collection) are URDF exports of two Onshape assemblies.
Their original meshes and exporter configuration are kept, and the URDFs are kept as exported except for the single `camera_lens0` addition to the DC file described below; the source Onshape document and assembly URLs are recorded in each original `config.json`.
The two assemblies and these exports are the KIWI project's own work, released under Apache-2.0 with the rest of this repository; the source CAD is published in [KIWI_hardware](https://github.com/TToTMooN/KIWI_hardware) under the same license. The Franka assets' license does not cover them.

| Source | SHA-256 of the export |
|---|---|
| `chopstick_sa/ChopstickSA.urdf` | `1eb1caebf796498f756b1f134716374e885f7c309a9c45fff3d39c240a456661` |
| `chopstick_dc/ChopstickDC.urdf` | `416beeab76d33e3f861e3c5428e44af736d6975b1e52add970b3dbe9c33077ed` (the Onshape export; the shipped file, with the `camera_lens0` link added, hashes to `03ea6017f36b9a37fae3ce84a40df2400312036a76ac0d632be907f960d6afac`) |

`python -m kiwi.replay.generate` derives normalized URDFs, standalone and FR3 MuJoCo models, registries, and glossy camera GLBs.
`--check` verifies byte-for-byte reproducibility of every output.
Camera GLBs keep the exported STL geometry and use RGB `[10, 10, 10]`, roughness `0.15`, and metallic factor `0`.
MuJoCo uses the original STL with a dark material, specular `0.8`, and shininess `0.85`.

The sources supply TCP and SA flange transforms, which are CAD facts rather than physical calibration measurements.
Calibrated `camera_lens0` frames added to the sources are kept and registered by generation.

### DC camera frame

`chopstick_dc/ChopstickDC.urdf` carries a `camera_lens0` link that the CAD export does not contain, so the file differs from the export hash above; that link is the only edit.
The frame is not a calibration of the DC design or of any physical build.
Exports stamp this leg `measured: true` because a description leg is recorded as a design fact; a build can carry a `build.mount_cam:` override with `measured: false` instead.
Replace it with an optical calibration of the handheld before using exported TCP tracks.

## Shared FR3 assets

`fr3/` contains KIWI composition sources, derived assemblies, and shared FR3 mesh geometry.
The reusable `fr3/mjcf/components/fr3_arm.xml` holds the arm geometry, inertias, and actuators.
Its flange is the URDF's `fr3_link8`.
The assembly applies a -135-degree yaw about that flange's Z axis.
This assembly rotation precedes the inverse CAD flange transform; it does not modify the imported standalone models.
KIWI-authored FR3 composition code is Apache-2.0.
The `.obj` visual and `.stl` collision meshes under `fr3/meshes/` are byte-identical copies of MuJoCo Menagerie's `franka_fr3/assets/` at commit `7876dcde` (Apache-2.0), which Menagerie derived from `franka_description`'s DAE and STL meshes with Blender and obj2mjcf; `fr3_common.xml` and `fr3_arm.xml` are edited from Menagerie's `fr3.xml`.

## Upstream Franka description

Source: `frankarobotics/franka_description` (formerly `frankaemika/`), version 2.6.0, Apache-2.0; the vendored `LICENSE`, `NOTICE` and `package.xml` record these facts.
`upstream/franka_description/` is pruned to FR3v2 visual/collision meshes for links 0-7 and the xacro sources/parameters needed to render the arm.
Do not patch vendored sources; re-snapshot and update the recorded version.
Registry substitution maps the arm's `meshes/robots/fr3/` references to the vendored `meshes/robots/fr3v2/` tree.
`third_party/NOTICE` points here for license accounting.
