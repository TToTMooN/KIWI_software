"""Reader for the ``kb4_intrinsics`` registry artifact (KB4 lens fits).

The rule this module follows: **artifact readers live in kiwi.core;
writers live in kiwi.calib** (the fit itself is calibration work). Scene
mapping, state estimation, ground and ``real2sim`` read lens models through
here, so none of them needs ``kiwi.calib`` to do it.

One canonical schema: a ``kb4_intrinsics`` document records which
physical ``unit`` and which ``lens`` it models, its 8 KB4 parameters, and
the fitted resolution. The lens gate refuses a front fit resolved through
the rear pointer (or vice versa).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from kiwi.core.camera.registry import ARTIFACT_SCHEMAS

__all__ = ["KB4_SCHEMA", "LENSES", "read_kb4"]

KB4_SCHEMA = ARTIFACT_SCHEMAS["intrinsics_kb4"]
LENSES = ("front", "back")

_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")


def read_kb4(
    path: Path, *, expected_lens: str | None = None
) -> tuple[np.ndarray, dict]:
    """Read one KB4 artifact: ``((8,) params, full_document)``, validated.

    Checks the schema, the recorded ``unit`` and ``lens``, the parameter
    set (present and finite), and the resolution (a pair of positive
    integers). ``expected_lens`` is the gate a pointer resolver passes:
    a mismatch is refused.
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != KB4_SCHEMA:
        raise ValueError(
            f"{path}: schema is {doc.get('schema')!r}, expected {KB4_SCHEMA!r}"
        )
    lens = doc.get("lens")
    if lens not in LENSES:
        raise ValueError(f"{path}: lens is {lens!r}, expected one of {LENSES}")
    if not doc.get("unit"):
        raise ValueError(f"{path}: a {KB4_SCHEMA} artifact must record its unit")
    if expected_lens is not None and lens != expected_lens:
        raise ValueError(
            f"{path}: this is a {lens}-lens fit, but the {expected_lens} lens "
            f"was requested - a fit attached to the wrong pointer projects "
            f"through the wrong model"
        )
    kb4 = doc.get("kb4")
    if not isinstance(kb4, dict):
        raise ValueError(f"{path}: missing kb4 parameter block")
    missing = [f for f in _PARAMS if f not in kb4]
    if missing:
        raise ValueError(f"{path}: kb4 block missing {missing}")
    params = np.array([float(kb4[f]) for f in _PARAMS], dtype=np.float64)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"{path}: kb4 parameters must all be finite")
    res = doc.get("resolution")
    ok = (
        isinstance(res, (list, tuple))
        and len(res) == 2
        and all(
            isinstance(v, int)
            and not isinstance(v, bool)
            and v > 0
            and math.isfinite(v)
            for v in res
        )
    )
    if not ok:
        raise ValueError(
            f"{path}: resolution must be a pair of positive integers, got {res!r}"
        )
    return params, doc
