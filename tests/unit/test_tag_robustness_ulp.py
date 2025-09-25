"""
Test tag robustness to small (≈ULP) perturbations outside guard bands.

For inputs where |Q| ≫ tau_Q_off, tags should remain REAL and invariant under
±1 ULP perturbations to the input.
"""

import math
import numpy as np

from zeroproof.policy import TRPolicy, TRPolicyConfig
from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.core import real, TRTag


def setup_module(module):
    # Install a conservative policy with tiny guard band
    pol = TRPolicy(
        tau_Q_on=1e-9,
        tau_Q_off=2e-9,
        tau_P_on=1e-12,
        tau_P_off=2e-12,
        deterministic_reduction=True,
        keep_signed_zero=True,
    )
    TRPolicyConfig.set_policy(pol)


def teardown_module(module):
    TRPolicyConfig.set_policy(None)


def _nextafter_f64(x: float, toward: float) -> float:
    return float(np.nextafter(np.float64(x), np.float64(toward)))


def test_tag_invariant_under_ulps_outside_band():
    layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    # Q(x)=1+0.01x keeps |Q| ≫ tau for |x|≤1
    layer.theta[0]._value = real(0.0)
    layer.theta[1]._value = real(1.0)
    layer.phi[0]._value = real(0.01)

    xs = [0.0, 0.3, -0.7, 1.0]
    for x in xs:
        # Baseline tag
        _, tag0 = layer.forward(real(x))
        assert tag0 == TRTag.REAL
        # +1 ULP and -1 ULP around x
        xp = _nextafter_f64(x, math.inf)
        xm = _nextafter_f64(x, -math.inf)
        _, tagp = layer.forward(real(xp))
        _, tagm = layer.forward(real(xm))
        assert tagp == TRTag.REAL
        assert tagm == TRTag.REAL

