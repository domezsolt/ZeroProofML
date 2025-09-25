"""Tag robustness tests: invariance to tiny FP perturbations outside guard bands.

We assert that when |Q| >= tau_Q_off by a wide margin, small (≤ a few ULP)
perturbations to inputs do not change tag classification (stay REAL).
"""

import math

from zeroproof.policy import TRPolicy, TRPolicyConfig
from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.core import real, TRTag


def test_tags_invariant_to_ulps_outside_guard_bands():
    # Install a policy with small thresholds
    pol = TRPolicy(
        tau_Q_on=1e-9,
        tau_Q_off=2e-9,
        tau_P_on=1e-12,
        tau_P_off=2e-12,
        keep_signed_zero=True,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)

    # Simple rational: P(x) = a0 + a1 x, Q(x) = 1 + b1 x
    layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    layer.theta[0]._value = real(0.2)
    layer.theta[1]._value = real(-0.05)
    layer.phi[0]._value = real(0.01)  # Q(x) ~ 1 + 0.01 x

    # Choose x where |Q| is comfortably above tau_Q_off; Q(0.1) = 1.001
    x0 = 0.1
    # Compute a small perturbation ~ a few ULP around x0 as float64
    # ULP at 0.1 is roughly 2**-53 scaled; use a fixed epsilon
    ulp = 2**-52  # generous bound
    deltas = [0.0, ulp, -ulp, 2*ulp, -2*ulp]

    tags = []
    for d in deltas:
        y, tag = layer.forward(real(x0 + d))
        tags.append(tag)

    # All tags should be REAL and identical
    assert all(t == TRTag.REAL for t in tags)

