"""Read the OpenCV FileStorage YAML dialect with PyYAML.

OpenCV writes %YAML:1.0 rather than a standard YAML directive. Camera
extrinsics and engine settings may carry it; all YAML readers route through
this module to remove that directive before parsing. Import-light: re and
yaml only, never cv2.
"""

from __future__ import annotations

import re

import yaml

__all__ = ["load_opencv_yaml", "strip_opencv_directive"]

# Consume the whole directive line: OpenCV-dialect configs may carry a
# trailing comment on it (`%YAML:1.0 # need to specify the file type at the top!`).
_OPENCV_YAML_DIRECTIVE = re.compile(r"^%YAML:[\d.]+[^\n]*\r?\n?", re.MULTILINE)


def strip_opencv_directive(text: str) -> str:
    """``text`` minus a leading ``%YAML:1.0`` directive line, if present.

    For callers that need the *text* in standard YAML, not a parsed document -
    :func:`kiwi.core.config_hash.hash_yaml` hashes whatever the file parses to
    (``null`` included), so it cannot go through :func:`load_opencv_yaml`'s
    dict coercion.
    """
    return _OPENCV_YAML_DIRECTIVE.sub("", text, count=1)


def load_opencv_yaml(text: str) -> dict:
    """PyYAML-load OpenCV FileStorage YAML text.

    Read-only convenience for inspecting a config. Do NOT round-trip a config
    through this to write one back: OpenCV's reader is quoting-sensitive
    (``model: "kalibr"`` is treated differently from ``model: kalibr``);
    writers of this dialect render it from text templates.
    """
    return yaml.safe_load(strip_opencv_directive(text)) or {}
