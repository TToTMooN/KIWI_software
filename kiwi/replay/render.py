#!/usr/bin/env python3
"""Regenerate URDF snapshots from each registry entry's declared source.

Imported SA/DC models use ``kiwi.replay.generate``; the reusable FR3 arm
uses xacro. This command updates URDFs only. Run ``python -m
kiwi.replay.generate`` for a complete update of URDF, MJCF, registries,
and materials. Both commands support ``--check`` to detect output drift.

No ROS: xacro's package lookup is resolved through the local descriptions
registry by the ament stub below.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import tyro

from kiwi.core.paths import descriptions_root
from kiwi.core.descriptions import RobotDescription, _registry

# $(find NAME) -> directory relative to the descriptions root. The names are
# the ones embedded in the xacro sources.
PACKAGE_LAYOUT = {
    "kiwi_descriptions": "",
    "franka_description": "upstream/franka_description",
}


def install_ament_stub(root: Path) -> None:
    """Point ``ament_index_python`` at the descriptions tree, ROS-free.

    Installs stub modules into ``sys.modules`` BEFORE xacro is imported, so
    ``$(find ...)`` resolves through :data:`PACKAGE_LAYOUT`. Always
    overrides an existing entry, so a machine with real ROS renders against
    this tree, not its workspace.
    """
    packages = {name: str(root / rel) for name, rel in PACKAGE_LAYOUT.items()}

    class _PackageNotFoundError(LookupError):
        pass

    def get_package_share_directory(name: str) -> str:
        try:
            return packages[name]
        except KeyError:
            raise _PackageNotFoundError(name) from None

    mod = types.ModuleType("ament_index_python")
    sub = types.ModuleType("ament_index_python.packages")
    for m in (mod, sub):
        m.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
        m.PackageNotFoundError = _PackageNotFoundError  # type: ignore[attr-defined]
    mod.packages = sub  # type: ignore[attr-defined]
    sys.modules["ament_index_python"] = mod
    sys.modules["ament_index_python.packages"] = sub


def render_description(desc: RobotDescription, root: Path) -> str:
    """Render one registry entry's URDF text from its xacro source.

    The caller must have installed the ament stub. Boolean args are
    lowercased (YAML ``false`` must reach xacro as ``"false"``, not
    ``"False"``); the absolute repo prefix is stripped from xacro's
    auto-generated banner so the output is checkout-independent.
    """
    import xacro  # noqa: PLC0415 - optional dep, replay extra

    if desc.render is None:
        raise ValueError(f"description {desc.name!r} declares no render source")
    if desc.render.generator == "imported_grippers":
        from kiwi.replay.generate import urdf

        return urdf(root, desc.name)
    if desc.render.xacro is None:
        raise ValueError(f"description {desc.name!r} has no supported generator")
    mappings = {
        k: (str(v).lower() if isinstance(v, bool) else str(v))
        for k, v in desc.render.args.items()
    }
    doc = xacro.process_file(str(root / desc.render.xacro), mappings=mappings)
    text = doc.toprettyxml(indent="  ")
    return text.replace(str(root.parent) + "/", "")


@dataclasses.dataclass
class Args:
    """Regenerate (or verify) the pre-rendered URDF snapshots."""

    # Compare against the committed snapshots instead of rewriting; exit 1
    # and name each drifted file when they differ.
    check: bool = False
    # Render only this registry entry (default: every entry with a render block).
    only: str | None = None


def main(args: Args) -> int:
    root = descriptions_root()
    install_ament_stub(root)
    targets = [
        d
        for d in _registry(None).values()
        if d.render is not None and (args.only is None or d.name == args.only)
    ]
    if not targets:
        print(
            f"render: no renderable description matches {args.only!r}", file=sys.stderr
        )
        return 2
    drifted: list[Path] = []
    for desc in targets:
        assert desc.urdf_path is not None  # render implies a urdf output
        text = render_description(desc, root)
        if args.check:
            on_disk = desc.urdf_path.read_text(encoding="utf-8")
            if on_disk != text:
                drifted.append(desc.urdf_path)
                print(f"render --check: DRIFT {desc.urdf_path}", file=sys.stderr)
            else:
                print(f"render --check: ok {desc.name}")
        else:
            desc.urdf_path.write_text(text, encoding="utf-8")
            print(f"render: wrote {desc.urdf_path}")
    if drifted:
        print(
            "render --check: snapshots drifted from their xacro sources - "
            "re-run `python -m kiwi.replay.render` and commit the result",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
