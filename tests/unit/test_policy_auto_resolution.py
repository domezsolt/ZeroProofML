"""Tests for policy threshold auto-resolution from model local scales."""

from zeroproof.training import enable_policy_from_model
from zeroproof.policy import TRPolicyConfig
from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.core import TRTag, real


def _make_model(phi_val: float, theta_val: float = 0.0) -> TRRational:
    m = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    # Set controllable parameter magnitudes
    m.theta[0]._value = real(theta_val)
    m.theta[1]._value = real(theta_val)
    m.phi[0]._value = real(phi_val)
    return m


def test_auto_thresholds_scale_with_model_sensitivities():
    # Small-denominator model vs large-denominator model
    small = _make_model(phi_val=0.01)
    large = _make_model(phi_val=10.0)

    # Enable policy from small model
    pol_small = enable_policy_from_model(small, ulp_scale=4.0)
    tau_q_on_small = pol_small.tau_Q_on
    tau_q_off_small = pol_small.tau_Q_off

    # Enable policy from large model (should scale thresholds up)
    pol_large = enable_policy_from_model(large, ulp_scale=4.0)
    tau_q_on_large = pol_large.tau_Q_on
    tau_q_off_large = pol_large.tau_Q_off

    assert tau_q_on_large > tau_q_on_small
    assert tau_q_off_large > tau_q_off_small


def test_determinism_outside_guard_bands_with_auto_policy():
    model = _make_model(phi_val=1.0)
    pol = enable_policy_from_model(model, ulp_scale=4.0)
    # Choose x far from poles so |Q| >> tau_Q_off
    x = 0.5
    y1, tag1 = model.forward(real(x))
    y2, tag2 = model.forward(real(x))

    assert tag1 == TRTag.REAL
    assert tag2 == TRTag.REAL
    # Policy currently active should be the same instance
    assert TRPolicyConfig.get_policy() is pol

