"""Unit tests for second-order safeguards utilities.

Covers:
 - Monotonicity of surrogate bounds vs. saturation bound
 - Finite curvature bounds near poles using curvature_bound_for_batch
"""

import math

from zeroproof.autodiff.grad_mode import GradientModeConfig
from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.optim_utils_second_order import (
    curvature_bound_for_batch,
    saturating_surrogate_bounds,
)


def test_saturating_surrogate_bounds_monotone():
    # Smaller bound -> larger (G_max, H_max); Larger bound -> closer to 1
    b_small = 1e-2
    b_medium = 1e-1
    b_large = 1.0

    Gs, Hs = saturating_surrogate_bounds(b_small)
    Gm, Hm = saturating_surrogate_bounds(b_medium)
    Gl, Hl = saturating_surrogate_bounds(b_large)

    assert Gs >= Gm >= Gl >= 1.0
    assert Hs >= Hm >= Hl >= 1.0


def test_curvature_bound_finite_near_pole():
    # Configure a simple rational: y = (theta0 + theta1 x) / (1 + phi1 x)
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    # Place a pole near x ≈ -0.1 by setting phi1 = 10
    model.theta[0]._value = real(0.1)
    model.theta[1]._value = real(0.2)
    model.phi[0]._value = real(10.0)

    # Set a finite saturation bound and expose via grad mode config
    GradientModeConfig.set_saturation_bound(0.1)

    # Choose inputs near the pole; |Q| is small
    xs = [-0.099, -0.101]

    cb = curvature_bound_for_batch(model, xs)

    # Ensure outputs are present and finite
    assert "curvature_bound" in cb
    assert "G_max" in cb and "H_max" in cb
    assert all(k in cb for k in ("B_k", "H_k", "depth_hint", "q_min"))

    for key in ("curvature_bound", "G_max", "H_max", "B_k", "H_k"):
        val = float(cb[key])
        assert math.isfinite(val)
        assert val >= 0.0

    # With bound=0.1, G_max ≈ 10 or greater (clamped ≥1)
    assert cb["G_max"] >= 10.0
