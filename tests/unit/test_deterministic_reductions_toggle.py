"""Deterministic reduction toggle tests.

Verifies that enabling/disabling policy.deterministic_reduction (pairwise vs
sequential sums) preserves REAL tags and yields numerically close outputs for
fixed models and inputs.
"""

from typing import List

from zeroproof.policy import TRPolicy, TRPolicyConfig
from zeroproof.layers import TRRational, MonomialBasis, tr_softmax
from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real


def _build_rational(dp: int = 8, dq: int = 8) -> TRRational:
    layer = TRRational(d_p=dp, d_q=dq, basis=MonomialBasis())
    # Deterministic coefficient pattern (no RNG): alternating small magnitudes
    for i, th in enumerate(layer.theta):
        sign = 1.0 if (i % 2 == 0) else -1.0
        th._value = real(sign * (0.01 + 0.001 * i))
    for j, ph in enumerate(layer.phi):
        sign = -1.0 if (j % 2 == 0) else 1.0
        ph._value = real(sign * (0.005 + 0.0005 * j))
    return layer


def _eval_rational(layer: TRRational, xs: List[float]) -> List[float]:
    outs: List[float] = []
    for x in xs:
        y, tag = layer.forward(real(float(x)))
        assert tag == TRTag.REAL
        outs.append(float(y.value.value))
    return outs


def test_pairwise_toggle_preserves_outputs_rational():
    layer = _build_rational(10, 10)
    xs = [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]

    # Policy with deterministic reductions ON
    pol_on = TRPolicy(
        tau_Q_on=1e-12, tau_Q_off=2e-12,
        tau_P_on=1e-15, tau_P_off=2e-15,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol_on)
    outs_on = _eval_rational(layer, xs)

    # Policy with deterministic reductions OFF
    pol_off = TRPolicy(
        tau_Q_on=1e-12, tau_Q_off=2e-12,
        tau_P_on=1e-15, tau_P_off=2e-15,
        deterministic_reduction=False,
    )
    TRPolicyConfig.set_policy(pol_off)
    outs_off = _eval_rational(layer, xs)

    # Compare elementwise (numerically close)
    for a, b in zip(outs_on, outs_off):
        assert abs(a - b) <= 1e-10

    # Cleanup
    TRPolicyConfig.set_policy(None)


def test_pairwise_toggle_preserves_outputs_softmax():
    # Build logits as TRNodes with values that exercise normalization
    logits = [
        TRNode.constant(real(-2.0)),
        TRNode.constant(real(0.0)),
        TRNode.constant(real(3.0)),
        TRNode.constant(real(1.5)),
    ]

    pol_on = TRPolicy(deterministic_reduction=True)
    TRPolicyConfig.set_policy(pol_on)
    probs_on_nodes = tr_softmax(logits)
    probs_on = [float(p.value.value) for p in probs_on_nodes]

    pol_off = TRPolicy(deterministic_reduction=False)
    TRPolicyConfig.set_policy(pol_off)
    probs_off_nodes = tr_softmax(logits)
    probs_off = [float(p.value.value) for p in probs_off_nodes]

    # Both should sum to ~1 and be numerically close elementwise
    assert abs(sum(probs_on) - 1.0) < 1e-9
    assert abs(sum(probs_off) - 1.0) < 1e-9
    for a, b in zip(probs_on, probs_off):
        assert abs(a - b) <= 1e-10

    # Cleanup
    TRPolicyConfig.set_policy(None)

