"""
Unit tests for coprime surrogate on TRRationalMulti with shared Q.
"""

from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRationalMulti


def _as_float(x):
    try:
        return float(x.value.value)
    except Exception:
        return float("nan")


def test_coprime_surrogate_multi_shared_q_aligned_vs_separated():
    basis = MonomialBasis()
    n_outputs = 2
    # Aligned zeros near x=a
    a = 0.3
    phi1_aligned = -1.0 / a

    # Model A: enable surrogate, shared Q zero near a, both numerators zero at a
    multi_a = TRRationalMulti(
        d_p=1,
        d_q=1,
        n_outputs=n_outputs,
        basis=basis,
        shared_Q=True,
        enable_coprime_regularizer=True,
        lambda_coprime=1.0,
        alpha_phi=0.0,
    )
    # Set shared denominator Q(x)=1+phi1*x
    shared_phi = multi_a.layers[0].phi
    shared_phi[0]._value = real(phi1_aligned)
    # Set numerators P_i(x) = x - a
    for head in multi_a.layers:
        head.theta[0]._value = real(-a)
        head.theta[1]._value = real(1.0)

    # Model B: enable surrogate, shared Q has no zero near a, same numerators
    multi_b = TRRationalMulti(
        d_p=1,
        d_q=1,
        n_outputs=n_outputs,
        basis=basis,
        shared_Q=True,
        enable_coprime_regularizer=True,
        lambda_coprime=1.0,
        alpha_phi=0.0,
    )
    shared_phi_b = multi_b.layers[0].phi
    shared_phi_b[0]._value = real(0.05)
    for head in multi_b.layers:
        head.theta[0]._value = real(-a)
        head.theta[1]._value = real(1.0)

    reg_a = multi_a.regularization_loss()
    reg_b = multi_b.regularization_loss()

    assert _as_float(reg_a) > _as_float(reg_b)
