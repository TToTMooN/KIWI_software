"""The laziness promise is a test, not a convention - CLI startup stays light.

``kiwi/episode/stages.py`` keeps every component import inside the stage
function bodies so ``import kiwi.episode`` stays cheap and the CLI starts
fast (CONTRIBUTING, checklist item 2).

This pins the promise end to end: importing ``kiwi.cli`` (the module behind
the ``kiwi`` executable, which eagerly imports ``kiwi.episode.stages``) in a
fresh interpreter must load no component and none of the heavy third-party
stacks the components and the viz/export/replay composers pull in.
"""

import json
import subprocess
import sys

# What `import kiwi.cli` may load from kiwi: the composition surface only.
# Components (and the heavy composers viz/export/replay) arrive lazily,
# inside a function body, when their stage or verb actually runs.
LIGHT_UNITS = {"cli", "core", "episode"}

# Heavy third-party stacks `import kiwi.cli` may not load. numpy and yaml
# are cheap and expected; these are not.
HEAVY_TOP_LEVEL = {
    "cv2",
    "scipy",
    "matplotlib",
    "rerun",
    "viser",
    "yourdfpy",
    "nicegui",
}


def test_cli_import_stays_light():
    code = "import json, sys\nimport kiwi.cli\nprint(json.dumps(sorted(sys.modules)))\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    mods = json.loads(out.stdout)
    kiwi_units = {m.split(".")[1] for m in mods if m.startswith("kiwi.")}
    heavy = {m.split(".")[0] for m in mods} & HEAVY_TOP_LEVEL
    assert kiwi_units <= LIGHT_UNITS, (
        f"import kiwi.cli eagerly loads kiwi unit(s) "
        f"{sorted(kiwi_units - LIGHT_UNITS)}; component imports belong inside "
        f"the stage function bodies (see kiwi/episode/stages.py)"
    )
    assert not heavy, (
        f"import kiwi.cli pulls in {sorted(heavy)} at startup; move the "
        f"import that drags it in into a function body"
    )
