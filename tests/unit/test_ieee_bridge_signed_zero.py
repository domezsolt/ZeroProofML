"""Tests for IEEE↔TR bridge hygiene, especially signed zeros and specials."""

import math

from zeroproof.bridge import from_ieee, to_ieee
from zeroproof.core import TRTag, real
from zeroproof.core import tr_div


def test_signed_zero_roundtrip_and_division():
    # Round-trip signed zeros
    zpos_tr = from_ieee(0.0)
    zneg_tr = from_ieee(-0.0)
    assert zpos_tr.tag == TRTag.REAL
    assert zneg_tr.tag == TRTag.REAL

    zpos_ieee = to_ieee(zpos_tr)
    zneg_ieee = to_ieee(zneg_tr)
    assert math.copysign(1.0, zpos_ieee) > 0.0
    assert math.copysign(1.0, zneg_ieee) < 0.0

    # Division by ±0 follows sign conventions
    one = real(1.0)
    mone = real(-1.0)
    pos_inf = tr_div(one, zpos_tr)
    neg_inf = tr_div(one, zneg_tr)
    pos_inf2 = tr_div(mone, zneg_tr)
    neg_inf2 = tr_div(mone, zpos_tr)
    assert pos_inf.tag == TRTag.PINF
    assert neg_inf.tag == TRTag.NINF
    assert pos_inf2.tag == TRTag.PINF
    assert neg_inf2.tag == TRTag.NINF


def test_special_values_mapping():
    # NaN ↔ PHI
    phi = from_ieee(float('nan'))
    assert phi.tag == TRTag.PHI
    back_nan = to_ieee(phi)
    assert math.isnan(back_nan)

    # +∞ and -∞ mapping
    p = from_ieee(float('inf'))
    n = from_ieee(float('-inf'))
    assert p.tag == TRTag.PINF
    assert n.tag == TRTag.NINF
    assert math.isinf(to_ieee(p)) and to_ieee(p) > 0
    assert math.isinf(to_ieee(n)) and to_ieee(n) < 0

