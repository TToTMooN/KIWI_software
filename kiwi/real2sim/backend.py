"""Locate NVIDIA's 3DGRUT backend and run the renderer in its environment."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path


def resolve_backend_repo(repo: Path | None = None) -> Path:
    """Explicit path, KIWI_3DGRUT, then the repository-local installation."""
    selected = (
        repo
        or os.environ.get("KIWI_3DGRUT")
        or Path(__file__).resolve().parents[2] / "build/3dgrut"
    )
    return Path(selected).expanduser().resolve()


def _backend(repo: Path | None, python: Path | None, script: str):
    repo = resolve_backend_repo(repo)
    if not (repo / script).is_file():
        raise FileNotFoundError(
            f"{repo / script}: run third_party/setup_real2sim.sh or set KIWI_3DGRUT"
        )
    executable = python or (repo / ".venv" / "bin" / "python")
    if not executable.is_file():
        raise FileNotFoundError(
            f"{executable}: run third_party/setup_real2sim.sh or pass the backend Python"
        )
    # The interpreter symlink is not resolved: the supplied virtual
    # environment's pyvenv.cfg selects the environment.
    return repo, executable.absolute()


#: Written by third_party/setup_real2sim.sh beside the backend venv: the CUDA
#: toolkit it validated, as plain ``export NAME=value`` lines.
INSTALLER_ENVIRONMENT = "kiwi-cuda-env.sh"

_VARIABLE = re.compile(r"\$(\w+)|\$\{(\w+)\}")


def installer_environment(venv: Path, base: dict[str, str]) -> dict[str, str]:
    """The toolkit variables the installer validated, read from the
    ``kiwi-cuda-env.sh`` it wrote beside the backend venv.

    The installer pairs the PyTorch cu128 wheels with the CUDA 12.8 toolkit
    it found, writes that pairing to the file, and sources the file from the
    venv's ``activate``. The backend is launched from KIWI's own environment,
    and the pairing is read here: the
    lines are ``export NAME=value`` with the installer's ``%q`` quoting, and
    ``$NAME`` / ``${NAME}`` references resolve against the lines above, then
    ``base`` (``VIRTUAL_ENV`` is the venv). A missing file is an empty
    mapping, so a backend installed some other way keeps the caller's
    environment; empty values are dropped, so an unset variable at install
    time stays unset.
    """
    path = venv / INSTALLER_ENVIRONMENT
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    lookup = {**base, "VIRTUAL_ENV": str(venv)}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        name, _, raw = line[len("export ") :].partition("=")
        name = name.strip()
        words = shlex.split(raw)
        value = words[0] if words else ""
        value = _VARIABLE.sub(lambda m: lookup.get(m.group(1) or m.group(2), ""), value)
        if value:
            found[name] = value
            lookup[name] = value
    return found


def backend_environment(python: Path) -> dict[str, str]:
    """The backend process environment: the caller's, with the toolkit the
    installer validated applied over it, and the backend-local build tools
    (notably Slang) visible to CUDA JIT.

    The installer's toolkit takes precedence over a ``CUDA_HOME`` or ``PATH``
    in the caller's environment. ``MAX_JOBS`` set by the caller is kept.
    """
    env = os.environ.copy()
    python = python.absolute()
    for key, value in installer_environment(python.parent.parent, env).items():
        if key == "MAX_JOBS":
            env.setdefault(key, value)
        else:
            env[key] = value
    env["PATH"] = os.pathsep.join((str(python.parent), env.get("PATH", "")))
    env.setdefault("MAX_JOBS", "6")
    return env


def run_worker(python: Path, module: str, *args) -> None:
    """Run ``python -m module`` in the backend environment with KIWI importable."""
    env = backend_environment(python)
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (package_root, env.get("PYTHONPATH")))
    )
    subprocess.run(
        [str(python), "-u", "-m", module, *map(str, args)], env=env, check=True
    )
