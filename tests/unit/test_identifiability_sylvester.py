"""Tests for identifiability diagnostics via Sylvester s_min surrogate."""

import math

from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.metrics.identifiability import compute_sylvester_smin


def _set_coeffs(model: TRRational, theta: list[float], phi: list[float]) -> None:
    # theta: length d_p+1, phi: length d_q
    for i, v in enumerate(theta):
        model.theta[i]._value = real(float(v))
    for j, v in enumerate(phi):
        model.phi[j]._value = real(float(v))


def test_sylvester_small_for_shared_factor():
    # P(x) = (x-1)^2 = 1 - 2x + x^2
    # Q(x) = (x-1)^2 = 1 - 2x + x^2
    m = TRRational(d_p=2, d_q=2, basis=MonomialBasis())
    _set_coeffs(m, theta=[1.0, -2.0, 1.0], phi=[-2.0, 1.0])

    smin = compute_sylvester_smin(m)
    assert math.isfinite(smin)
    # Exact common factor should drive s_min ~ 0
    assert smin < 1e-10


def test_sylvester_larger_for_coprime():
    # P(x) = x (theta=[0,1,0]), Q(x) = 1 + x^2 (phi=[0,1])
    m = TRRational(d_p=2, d_q=2, basis=MonomialBasis())
    _set_coeffs(m, theta=[0.0, 1.0, 0.0], phi=[0.0, 1.0])

    smin = compute_sylvester_smin(m)
    assert math.isfinite(smin)
    assert smin > 1e-6
