import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zeroproof.bridge.ieee_tr import from_ieee, to_ieee


@given(st.floats(allow_nan=False, allow_infinity=False, width=64))
def test_roundtrip_finite(x: float):
    tr = from_ieee(x)
    y = to_ieee(tr)
    assert y == x


@given(st.sampled_from([float("inf"), float("-inf")]))
def test_roundtrip_infinities(x: float):
    tr = from_ieee(x)
    y = to_ieee(tr)
    assert math.isinf(y) and (y > 0) == (x > 0)


def test_roundtrip_nan():
    tr = from_ieee(float("nan"))
    y = to_ieee(tr)
    assert math.isnan(y)


def test_signed_zero_preserved():
    for x in (+0.0, -0.0):
        tr = from_ieee(x)
        y = to_ieee(tr)
        # signbit: True for -0.0
        import struct

        def signbit(f: float) -> bool:
            b = struct.pack(">d", f)
            return (b[0] & 0x80) != 0

        assert signbit(y) == signbit(x)
