"""Enforce the boundary DAG - the architecture is a test, not a convention.

    core <- {calib, track, sync, register, state, eef, ground, perception}
         <- {episode, viz, export, replay, real2sim} <- app <- cli

Rules checked over every module's imports (via ast, no execution):

* ``kiwi.core`` imports nothing else from ``kiwi``.
* Components import at most ``kiwi.core`` (and themselves).
* Composers import at most core + components - never each other, never
  ``kiwi.cli``.
* ``kiwi.app`` composes services for the CLI and GUI; lower layers cannot import it.
* ``kiwi.cli`` (top-level composition) may import anything.

The world is closed: every file under ``kiwi/`` must map to a unit listed
here, and every ``kiwi.*`` import must name one. A new top-level package
or module registers in these sets before anything may import it, so it can
never become shared surface with no rule applied.

A cross-component import is a regression: promote the shared piece into
``kiwi.core`` instead.
"""

import ast
from pathlib import Path

KIWI_DIR = Path(__file__).resolve().parent.parent / "kiwi"

CORE = {"core"}
COMPONENTS = {
    "calib",
    "track",
    "sync",
    "register",
    "state",
    "eef",
    "ground",
    "perception",
}
COMPOSERS = {"episode", "viz", "export", "replay", "real2sim"}
TOP_MODULES = {"cli"}
APPLICATIONS = {"app"}
ALL_UNITS = CORE | COMPONENTS | COMPOSERS | TOP_MODULES | APPLICATIONS


def _unit_of(path: Path, kiwi_dir: Path = KIWI_DIR) -> str:
    """Map a file under kiwi/ to its architectural unit."""
    rel = path.relative_to(kiwi_dir)
    if len(rel.parts) == 1:  # kiwi/<module>.py
        return "__root__" if rel.name == "__init__.py" else rel.stem
    return rel.parts[0]  # kiwi/<pkg>/...


def _kiwi_imports(
    path: Path, pkg: tuple[str, ...] | None = None, kiwi_dir: Path = KIWI_DIR
) -> set[str]:
    """Names under kiwi that this file imports (absolute AND relative forms).

    ``pkg`` is the file's package as dotted parts (``("kiwi", "episode")``
    for ``kiwi/episode/stages.py``); derived from ``path`` when omitted.
    Relative imports resolve against it, so ``from ..sync import x`` inside
    a composer is caught exactly like ``from kiwi.sync import x``.

    Records the segment after ``kiwi.`` whether or not it is a registered
    unit - judging it is the caller's job, so an import of an unregistered
    module is a reportable violation instead of invisible.
    """
    if pkg is None:
        pkg = ("kiwi", *path.relative_to(kiwi_dir).parts[:-1])
    tree = ast.parse(path.read_text(), filename=str(path))
    units: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "kiwi" and len(parts) > 1:
                    units.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against this package
                if node.level > len(pkg):
                    continue  # escapes the kiwi tree; broken either way
                base = pkg[: len(pkg) - node.level + 1]
                parts = [*base, *(node.module.split(".") if node.module else [])]
            elif node.module is None:
                continue
            else:
                parts = node.module.split(".")
            if parts[0] != "kiwi":
                continue
            if len(parts) > 1:
                units.add(parts[1])
            else:  # ``from kiwi import X`` (or a relative path landing on kiwi)
                for alias in node.names:
                    # A unit iff it exists as a submodule; ``from kiwi
                    # import __version__`` names an attribute, not a unit.
                    target = kiwi_dir / alias.name
                    if target.is_dir() or target.with_suffix(".py").is_file():
                        units.add(alias.name)
    return units


ALLOWED = {
    **{u: CORE for u in CORE},  # core may see itself only
    **{u: CORE | {u} for u in COMPONENTS},
    **{u: CORE | COMPONENTS | {u} for u in COMPOSERS},
    "cli": ALL_UNITS,
    "app": CORE | COMPONENTS | COMPOSERS | APPLICATIONS,
    "__root__": set(),  # kiwi/__init__.py stays import-free
}


def _violations(kiwi_dir: Path = KIWI_DIR) -> list[str]:
    violations = []
    for path in sorted(kiwi_dir.rglob("*.py")):
        rel = path.relative_to(kiwi_dir.parent)
        unit = _unit_of(path, kiwi_dir)
        allowed = ALLOWED.get(unit)
        if allowed is None:
            violations.append(
                f"{rel}: '{unit}' is not a registered unit - a new package "
                f"or module under kiwi/ joins the graph by being added to "
                f"CORE/COMPONENTS/COMPOSERS/TOP_MODULES in this test first"
            )
            continue  # no allowed-set to judge its imports against
        for imported in sorted(_kiwi_imports(path, kiwi_dir=kiwi_dir)):
            if imported not in ALL_UNITS:
                violations.append(
                    f"{rel}: unit '{unit}' imports 'kiwi.{imported}', "
                    f"which is not a registered unit"
                )
            elif imported not in allowed:
                violations.append(
                    f"{rel}: unit '{unit}' imports 'kiwi.{imported}' "
                    f"(allowed: {sorted(allowed) or 'nothing'})"
                )
    return violations


def test_boundary_dag():
    violations = _violations()
    assert not violations, "boundary DAG violations:\n" + "\n".join(violations)


def test_checker_resolves_relative_imports(tmp_path):
    # The DAG guarantee rests on this function - it must see the relative
    # spellings too, or a composer could smuggle in `from ..viz import x`.
    # `from .. import __version__` names an attribute of kiwi/__init__.py,
    # not a unit, and must stay unrecorded.
    f = tmp_path / "x.py"
    f.write_text(
        "from ..sync import audio\nfrom .. import track\nfrom . import stages\n"
        "from .. import __version__\n"
    )
    assert _kiwi_imports(f, pkg=("kiwi", "episode")) == {"sync", "track", "episode"}


def test_unregistered_units_fail_closed(tmp_path):
    # The world is closed: a new module or package under kiwi/ and any
    # import of it must each fail with a registration message - and the
    # scan must continue past them, so real violations elsewhere are still
    # listed instead of masked.
    k = tmp_path / "kiwi"
    (k / "newpkg").mkdir(parents=True)
    (k / "newpkg" / "__init__.py").write_text("from kiwi.viz import player\n")
    (k / "new.py").write_text("")
    (k / "sync").mkdir()
    (k / "sync" / "__init__.py").write_text("from kiwi.new import shared\n")
    v = _violations(k)
    assert any("'new' is not a registered unit" in m for m in v)
    assert any("'newpkg' is not a registered unit" in m for m in v)
    assert any("unit 'sync' imports 'kiwi.new'" in m for m in v)


def test_every_unit_exists():
    for unit in CORE | COMPONENTS | COMPOSERS | APPLICATIONS:
        assert (KIWI_DIR / unit / "__init__.py").is_file(), f"missing package {unit}"
    for mod in TOP_MODULES:
        assert (KIWI_DIR / f"{mod}.py").is_file(), f"missing module {mod}"
