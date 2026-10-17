#!/usr/bin/env bash
# Build ORB-SLAM3 (headless use) + Pangolin inside the pixi `slam-offline` env.
#
# Everything is self-contained: toolchain and C++ deps come from the env, no
# system packages, no sudo. Run from the repository root:
#
#     git submodule update --init third_party/ORB_SLAM3
#     pixi install -e slam-offline
#     pixi run -e slam-offline build-orbslam3
#
# The third_party/ORB_SLAM3 submodule is NEVER modified. The build works on
# an archived copy under build/orbslam3/src/ with the local patch series
# (third_party/patches/orbslam3/) applied there, so `git status` stays clean
# before, during, and after a build. If the submodule is not at the patch's
# base commit, or the patch does not apply, this script fails immediately.
#
# Pinned sources:
#   ORB_SLAM3  upstream V1.0, commit in patches/orbslam3/BASE_COMMIT.txt -
#              carried as the third_party/ORB_SLAM3 submodule.
#   Pangolin   v0.6 dd801d2 (GLEW-based; avoids the libepoxy dependency of
#              newer Pangolin). Build-only dependency: cloned on demand into
#              the gitignored build tree, never a submodule. It is a
#              link-time dependency only - the session wrapper constructs
#              the SLAM system with the viewer disabled.
#
# Artifacts (all under the gitignored build/ tree):
#   build/orbslam3/wrapper/kiwi-orbslam-session   THE process contract the
#                                                 driver invokes (explicit
#                                                 arguments only; source in
#                                                 cpp/orbslam_session/)
#   build/orbslam3/src/lib/libORB_SLAM3.so
#   build/orbslam3/src/Vocabulary/ORBvoc.txt      (extracted from the
#                                                  upstream archive)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
SUB="$ROOT/third_party/ORB_SLAM3"
PATCHES="$ROOT/third_party/patches/orbslam3"
BUILD="$ROOT/build/orbslam3"
REL_SRC="build/orbslam3/src"
SRC="$ROOT/$REL_SRC"
INSTALL="$BUILD/install"
J="$(nproc)"

PANGOLIN_COMMIT=dd801d244db3a8e27b7fe8020cd751404aa818fd  # v0.6
BASE_COMMIT="$(cut -d' ' -f2 "$PATCHES/BASE_COMMIT.txt")"
PATCH="$PATCHES/0001-keypoint-mask-localization.patch"
WRAPPER="$ROOT/cpp/orbslam_session"
STAMP="$BUILD/.source-stamp"
# The stamp covers EVERY source that shapes the built engine: the pinned
# upstream commit, the whole patch directory, and the wrapper sources. Any
# change re-stages from scratch - a stale staging must never be reused.
STAMP_WANT="base=$BASE_COMMIT
$(cd "$ROOT" && sha256sum third_party/patches/orbslam3/* cpp/orbslam_session/*)"

echo "=== env: CONDA_PREFIX=${CONDA_PREFIX:?run via 'pixi run -e slam-offline build-orbslam3'}  jobs=$J"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$INSTALL${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# Upstream V1.0 (and Pangolin v0.6) predate today's toolchain: CMake >= 4
# refuses their old cmake_minimum_required (POLICY_VERSION_MINIMUM), current
# libstdc++ no longer includes <cstdint>/<limits> transitively, and newer gcc
# promotes array-bounds warnings in upstream code to errors.
COMPAT_CXX_FLAGS="-include cstdint -include limits -Wno-error=array-bounds -Wno-array-bounds -Wno-error"

echo "=== source staging (submodule stays pristine) ==="
if [ ! -e "$SUB/.git" ]; then
    echo "ERROR: third_party/ORB_SLAM3 is not initialized." >&2
    echo "Run: git submodule update --init third_party/ORB_SLAM3" >&2
    exit 1
fi
HEAD_COMMIT="$(git -C "$SUB" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$BASE_COMMIT" ]; then
    echo "ERROR: third_party/ORB_SLAM3 is at $HEAD_COMMIT," >&2
    echo "but the patch series applies onto $BASE_COMMIT" >&2
    echo "(third_party/patches/orbslam3/BASE_COMMIT.txt)." >&2
    echo "Run: git submodule update --checkout third_party/ORB_SLAM3" >&2
    exit 1
fi
if [ -d "$SRC" ] && [ "$(cat "$STAMP" 2>/dev/null)" != "$STAMP_WANT" ]; then
    echo "source stamp changed - re-staging from scratch"
    rm -rf "$SRC" "$BUILD/wrapper" "$STAMP"
fi
if [ -d "$SRC" ]; then
    echo "reusing staged source (stamp matches; incremental rebuild)"
else
    mkdir -p "$SRC"
    # Archive the pinned tree - never a working-tree copy, so local edits or
    # stray files in the submodule can not leak into the build.
    git -C "$SUB" archive "$BASE_COMMIT" | tar -x -C "$SRC"
    # Apply from the repo root with an explicit --directory: run from inside
    # a subdirectory, git apply silently skips every patched path outside
    # that subdirectory (the staging tree lives inside this repository) and
    # still exits 0 - a no-op that looks like success.
    git -C "$ROOT" apply --check --directory="$REL_SRC" "$PATCH"
    git -C "$ROOT" apply --directory="$REL_SRC" "$PATCH"
    # The wrapper sets keypoint masks through this patched method.
    # Its absence means the patch application above had no effect.
    if ! grep -q "SetNextKpMask" "$SRC/include/System.h"; then
        echo "ERROR: patch applied without effect - keypoint-mask API missing" >&2
        exit 1
    fi
    echo "$STAMP_WANT" > "$STAMP"
fi

echo "=== Pangolin ==="
if [ ! -d "$BUILD/Pangolin/.git" ]; then
    git clone https://github.com/stevenlovegrove/Pangolin.git "$BUILD/Pangolin"
    git -C "$BUILD/Pangolin" checkout --quiet "$PANGOLIN_COMMIT"
elif [ "$(git -C "$BUILD/Pangolin" rev-parse HEAD)" != "$PANGOLIN_COMMIT" ]; then
    echo "ERROR: $BUILD/Pangolin is not at the pinned commit $PANGOLIN_COMMIT." >&2
    echo "Remove it and re-run: rm -rf $BUILD/Pangolin" >&2
    exit 1
fi
cmake -S "$BUILD/Pangolin" -B "$BUILD/Pangolin/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="-include cstdint -include limits" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF \
    -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_LIBREALSENSE2=OFF \
    -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF \
    -DBUILD_PANGOLIN_LIBUVC=OFF -DBUILD_PANGOLIN_UVC_MEDIAFOUNDATION=OFF \
    -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_VIDEO=OFF \
    -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LZ4=OFF
cmake --build "$BUILD/Pangolin/build" -j"$J" --target install

echo "=== ORB_SLAM3 Thirdparty ==="
# g2o and ORB share Eigen-allocated graph objects. Keep dependencies
# pre-C++17 like ORB so Eigen uses matching aligned new/delete paths.
for lib in DBoW2 g2o Sophus; do
    cmake -S "$SRC/Thirdparty/$lib" -B "$SRC/Thirdparty/$lib/build" \
        -DCMAKE_CXX_STANDARD=14 \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_CXX_FLAGS="$COMPAT_CXX_FLAGS" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SRC/Thirdparty/$lib/build" -j"$J"
done

echo "=== ORB vocabulary ==="
if [ ! -f "$SRC/Vocabulary/ORBvoc.txt" ]; then
    tar -xf "$SRC/Vocabulary/ORBvoc.txt.tar.gz" -C "$SRC/Vocabulary"
fi

echo "=== ORB_SLAM3 ==="
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="$COMPAT_CXX_FLAGS"
cmake --build "$SRC/build" -j"$J" --target ORB_SLAM3

echo "=== session wrapper ==="
cmake -S "$WRAPPER" -B "$BUILD/wrapper" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="$COMPAT_CXX_FLAGS" \
    -DORBSLAM_SRC="$SRC"
cmake --build "$BUILD/wrapper" -j"$J"

echo "=== done ==="
ls -la "$SRC/lib/" "$BUILD/wrapper/kiwi-orbslam-session" \
    "$SRC/Vocabulary/ORBvoc.txt"
