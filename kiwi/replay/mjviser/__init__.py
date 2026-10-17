"""MuJoCo -> viser scene bridge (batched meshes, live pose sync, overlay GUI).

Vendored from https://github.com/mujocolab/mjviser
(Apache-2.0, Copyright 2025 The mjlab Developers; see LICENSE and
PROVENANCE.md in this directory). Local edits should keep upstream's 2-space
indentation style; ruff excludes this directory for that reason.

Upstream's standalone ``viewer.py`` + ``__main__.py`` (a generic simulate
loop) were dropped in the vendoring; the replay viewer drives
``ViserMujocoScene`` directly.
"""

from .scene import ViserMujocoScene as ViserMujocoScene
