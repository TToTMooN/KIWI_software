"""Keep the replay viewer and gripper tuner serving until Ctrl-C.

Callers create and manage their own Viser server. :func:`hold_until_interrupt`
keeps the process alive after setup or playback and returns on KeyboardInterrupt.
"""

from __future__ import annotations

import time

__all__ = ["hold_until_interrupt"]

DEFAULT_HOLD_MSG = "viser still serving; press Ctrl-C to exit."


def hold_until_interrupt(msg: str | None = DEFAULT_HOLD_MSG) -> None:
    """Block, keeping the viser server alive, until the operator hits Ctrl-C.

    Prints ``msg`` (skipped if ``None``/empty) then sleeps until
    ``KeyboardInterrupt``, which it swallows so the caller returns normally.
    """
    if msg:
        print(msg)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
