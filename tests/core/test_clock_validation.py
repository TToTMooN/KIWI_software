"""A global clock never validates unusable timestamps."""

import pytest

from kiwi.core.clock import GlobalClock


@pytest.mark.parametrize("label", ["origin", "other"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), None, True, "0", []]
)
def test_every_clock_offset_must_be_a_finite_number(label, value):
    offsets = {"origin": 0.0, "other": -1.0}
    offsets[label] = value
    with pytest.raises(ValueError, match=f"{label!r}.*finite number"):
        GlobalClock("origin", offsets).validate()


def test_finite_offsets_keep_the_global_clock_contract():
    clock = GlobalClock("left", {"left": 0.0, "right": -1.5})
    assert clock.validate() is clock
