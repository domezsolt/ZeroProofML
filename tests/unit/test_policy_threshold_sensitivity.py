"""
Tests for TRPolicy.resolve_thresholds scaling with local sensitivities.

Verifies that increasing local_scale_q/p increases the derived tau thresholds.
Also verifies that TRRational.estimate_local_scales produces larger scales when
coefficient norms grow.
"""

from zeroproof.policy import TRPolicy
from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.core import real


def test_resolve_thresholds_scales_with_local_sensitivity():
    pol = TRPolicy()
    ulp = 1e-6
    # Baseline
    pol.resolve_thresholds(ulp, local_scale_q=1.0, local_scale_p=1.0)
    tau_q_on_base = pol.tau_Q_on
    tau_p_on_base = pol.tau_P_on

    # Larger local scales → larger thresholds
    pol.resolve_thresholds(ulp, local_scale_q=10.0, local_scale_p=5.0)
    assert pol.tau_Q_on > tau_q_on_base
    assert pol.tau_P_on > tau_p_on_base


def test_estimate_local_scales_increases_with_coeff_norms():
    basis = MonomialBasis()
    layer_small = TRRational(d_p=2, d_q=2, basis=basis)
    layer_big = TRRational(d_p=2, d_q=2, basis=basis)

    # Small coefficients
    for t in layer_small.theta:
        t._value = real(0.01)
    for p in layer_small.phi:
        p._value = real(0.01)

    # Larger coefficients
    for t in layer_big.theta:
        t._value = real(0.5)
    for p in layer_big.phi:
        p._value = real(0.5)

    q_small, p_small = layer_small.estimate_local_scales()
    q_big, p_big = layer_big.estimate_local_scales()

    assert q_big > q_small
    assert p_big > p_small

