"""
Unit tests for the coprime surrogate regularizer on TRRational.

Checks that:
- The surrogate increases when P and Q share a zero near sampled points.
- When disabled, regularization_loss reduces to α/2 * ||φ||^2.
"""

from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.core import real


def _get_scalar(x):
    # Helper to unwrap TRScalar nested value
    try:
        return float(x.value.value)
    except Exception:
        return float('nan')


def test_coprime_surrogate_increases_when_zeros_align():
    """Surrogate term should be larger when P and Q share a zero near the sampled grid."""
    basis = MonomialBasis()

    # Aligned zeros: choose a≈0.3 so sampled grid in surrogate includes 0.3
    a = 0.3
    phi1_aligned = -1.0 / a  # Q(x)=1+phi1*x -> zero at x=a

    # Layer A: aligned P zero at x=a and Q zero at x=a
    layer_aligned = TRRational(
        d_p=1, d_q=1, basis=basis,
        enable_coprime_regularizer=True,
        lambda_coprime=1.0,
        alpha_phi=0.0,
    )
    # P(x) = θ0 + θ1 x with zero at x=a ⇒ choose θ1=1, θ0 = -a
    layer_aligned.theta[0]._value = real(-a)
    layer_aligned.theta[1]._value = real(1.0)
    # Q(x) = 1 + φ1 x with φ1 chosen above
    layer_aligned.phi[0]._value = real(phi1_aligned)

    # Layer B: separated zeros (P zero at x=a, Q has small slope so no zero near grid)
    layer_sep = TRRational(
        d_p=1, d_q=1, basis=basis,
        enable_coprime_regularizer=True,
        lambda_coprime=1.0,
        alpha_phi=0.0,
    )
    layer_sep.theta[0]._value = real(-a)
    layer_sep.theta[1]._value = real(1.0)
    layer_sep.phi[0]._value = real(0.05)  # Q ≈ 1 + 0.05 x (no zero near sampled grid)

    reg_aligned = layer_aligned.regularization_loss()
    reg_sep = layer_sep.regularization_loss()

    val_aligned = _get_scalar(reg_aligned)
    val_sep = _get_scalar(reg_sep)

    assert val_aligned > val_sep, f"Expected aligned surrogate > separated (got {val_aligned} vs {val_sep})"


def test_regularization_no_surrogate_matches_alpha_half_phi_norm():
    """When disabled, reg loss equals α/2 * ||φ||^2 (within small tolerance)."""
    basis = MonomialBasis()
    alpha = 0.2
    layer = TRRational(
        d_p=1, d_q=2, basis=basis,
        enable_coprime_regularizer=False,
        lambda_coprime=0.0,
        alpha_phi=alpha,
    )
    # Set denominator coefficients φ1, φ2
    layer.phi[0]._value = real(0.3)
    layer.phi[1]._value = real(-0.4)

    reg = layer.regularization_loss()
    val = _get_scalar(reg)
    expected = 0.5 * alpha * ((0.3 ** 2) + ((-0.4) ** 2))

    assert abs(val - expected) < 1e-8, f"Expected {expected}, got {val}"

