"""Shared presentation: one colour per stream label, everywhere.

The player, the synced render, and the trace png all draw the same arms, so
a label keeps its colour across all three.

The palette is Okabe-Ito.
"""

from __future__ import annotations

__all__ = ["FALLBACK", "PALETTE", "bgr", "stream_color"]

PALETTE: dict[str, tuple[int, int, int]] = {
    "left": (0, 114, 178),  # blue
    "right": (213, 94, 0),  # vermillion
    "ego": (0, 158, 115),  # bluish green
}
FALLBACK = (86, 180, 233)  # sky blue, for any label not spelled above


def stream_color(label: str) -> tuple[int, int, int]:
    """RGB for a stream label."""
    return PALETTE.get(label, FALLBACK)


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """RGB -> BGR, for the cv2-drawn artifacts."""
    r, g, b = rgb
    return (b, g, r)
