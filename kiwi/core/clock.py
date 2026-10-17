"""Global-clock conventions: every per-stream offset in an episode speaks these.

The episode's global clock:

* **t = 0 is frame 0 of whichever video stream started LAST** (arm or extra
  camera). All per-stream offsets are therefore <= 0, and the named origin
  stream holds exactly 0.0.
* Offsets are **per stream**, keyed by stream label (``"left"``, ``"right"``,
  ``"ego"``, ...), and live in the episode manifest's ``sync:`` block - written by
  the sync stage, never hand-rolled.
* Sign convention (stated on every offset-bearing signature): a positive
  pairwise offset means *other started later*; on the shared clock,
  ``global_off(other) = global_off(ref) + offset_s``.
* **Audio time is not video time.** A camera stamps audio and video with a
  constant relative delay that depends on its exact recording mode, so raw
  audio offsets must pass through a per-mode A/V correction before becoming
  video offsets (owned by :mod:`kiwi.sync`).
* Explicit re-sync replaces the saved offsets. Task and curation window bounds
  remain unchanged and must be reviewed against the new clock.

Offset estimation belongs to :mod:`kiwi.sync`; :mod:`kiwi.episode` reads and
writes the manifest's sync block. This module holds their shared conventions
and clock type without importing either component.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class GlobalClock:
    """Resolved per-stream offsets on the episode's global clock.

    ``origin`` is the stream label holding t=0 (the last video stream to
    start); every value in ``offsets_s`` is <= 0 and ``offsets_s[origin]``
    is exactly 0.0.
    """

    origin: str
    offsets_s: dict[str, float] = field(default_factory=dict)

    def validate(self) -> "GlobalClock":
        if self.origin not in self.offsets_s:
            raise ValueError(f"origin {self.origin!r} missing from offsets")
        for label, value in self.offsets_s.items():
            try:
                finite = not isinstance(value, bool) and math.isfinite(value)
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(
                    f"clock offset for {label!r} must be a finite number, got {value!r}"
                )
        if abs(self.offsets_s[self.origin]) > 1e-9:
            raise ValueError("origin stream must sit at exactly 0.0")
        late = {k: v for k, v in self.offsets_s.items() if v > 1e-9}
        if late:
            raise ValueError(f"global clock requires offsets <= 0, got {late}")
        return self
