# ORB-SLAM3: local patch series

KIWI's scene-mapping feature uses [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
(GPL-3.0) as a separate engine process. The engine source is the pristine
upstream repository, pinned as the `third_party/ORB_SLAM3` submodule at the
commit named in `BASE_COMMIT.txt`; every local change lives in this
directory as an auditable patch series instead of a fork.

## Contents

- `0001-keypoint-mask-localization.patch` — one diff against the
  pinned base: per-frame keypoint exclude masks (`System::SetNextKpMask`;
  masked keypoints keep their descriptors, so they still count in place
  recognition, and are skipped in the matching and triangulation searches),
  localization against a loaded atlas (in localization mode the tracker
  relocalizes in the largest map of the atlas instead of initializing a
  new map, and retries relocalization after losing track: upstream retries
  only from LOST, which the localization mode never reaches from
  RECENTLY_LOST), and two atlas-save fixes (shutdown waits for local mapping and
  loop closing before the atlas is written; the pre-save pass iterates a
  copy of the map-point set it erases from).
- `BASE_COMMIT.txt` — the upstream commit the patch applies onto. The
  submodule gitlink pins the same commit; the build refuses to run if the
  two ever disagree.

## Build

Linux only. The pixi `slam-offline` environment carries the complete
toolchain and C++ dependencies (compiler, cmake, ninja/make, pkg-config,
OpenCV 4, Eigen, GLEW, Boost, OpenSSL, X11) - no system packages, no sudo:

```bash
git submodule update --init third_party/ORB_SLAM3
pixi install -e slam-offline
pixi run -e slam-offline build-orbslam3
```

The build never touches the submodule. `third_party/build_orbslam3.sh`
archives the pinned tree into `build/orbslam3/src/`, applies the patch
there (failing loudly if the base commit or the patch do not match), clones
[Pangolin](https://github.com/stevenlovegrove/Pangolin) v0.6
(`dd801d24`, MIT; a link-time dependency only - the session wrapper disables
the viewer) into the gitignored build tree, and extracts the ORB vocabulary
from the upstream archive. It builds the ORB-SLAM3 library without compiling
the upstream example executables. `git status` stays clean throughout.

The build also compiles `cpp/orbslam_session/` - the first-party
`kiwi-orbslam-session` wrapper (GPL-3.0 combined work, LICENSE beside its
source), which is the ONLY process contract the pipeline invokes: every
input and output is a named argument (vocabulary, settings, sequence,
timestamps, atlas in/out, pose observations, mask directory). The wrapper
runs the engine in the atlas directory with a copy of the settings file
that names the atlas through the upstream keys `System.SaveAtlasToFile` and
`System.LoadAtlasFromFile`. A source stamp over the
upstream commit + this patch directory + the wrapper sources re-stages
the build from scratch whenever any of them changes.

Artifacts:

```text
build/orbslam3/wrapper/kiwi-orbslam-session
build/orbslam3/src/lib/libORB_SLAM3.so
build/orbslam3/src/Vocabulary/ORBvoc.txt
```

## Boundary

ORB-SLAM3 remains GPL-3.0 and is never imported or linked into the
Apache-2.0 `kiwi` package: KIWI invokes the session wrapper as a separate
process over an argv/file contract (see `third_party/NOTICE`). The
default KIWI installation, `import kiwi`, and ordinary CI need neither the
submodule nor this build. The Python module that runs the engine process
(`kiwi.track.orbslam.engine` - invocation and logs) uses the executable and
vocabulary paths documented above.
