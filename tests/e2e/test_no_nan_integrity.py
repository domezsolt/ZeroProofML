"""E2E tests that should never produce NaN (PHI) values.

These are intentionally lightweight so they can run in CI quickly.
"""

import numpy as np
import pytest

from zeroproof import TRTag, real, to_ieee
from zeroproof.autodiff import TRNode, gradient_tape, tr_add, tr_div, tr_mul
from zeroproof.core import TRScalar


def _is_finite_tr(x: TRScalar) -> bool:
    """Return True if TRScalar is a REAL and finite when converted to IEEE."""
    if x.tag != TRTag.REAL:
        return False
    v = to_ieee(x)
    return np.isfinite(v)


def test_no_nan_basic_operations():
    """Simple arithmetic on safe inputs must not yield PHI/NaN."""
    vals = [0.5, 1.0, 2.0, 10.0]
    for v in vals:
        x = real(v)
        # y = (x + 1) * x / 2
        y = tr_add(x, real(1.0))
        y = tr_mul(y, x)
        y = tr_div(y, real(2.0))
        assert y.tag != TRTag.PHI
        assert _is_finite_tr(y)


def test_no_nan_autodiff_gradient():
    """Gradients on safe expressions must be finite and REAL."""
    x = TRNode.parameter(real(2.0))
    with gradient_tape() as tape:
        tape.watch(x)
        # f(x) = x^2 + x
        y = tr_add(tr_mul(x, x), x)
    grad = tape.gradient(y, [x])[0]
    assert grad.value.tag == TRTag.REAL
    assert np.isfinite(grad.value.value)
