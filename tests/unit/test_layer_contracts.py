"""Tests for layer contract publishing (B_k, H_k, G_max, H_max)."""

from zeroproof.layers import TRRational, TRMultiInputRational, MonomialBasis
from zeroproof.core import real


def test_trrational_layer_contract_keys_and_values():
    m = TRRational(d_p=2, d_q=2, basis=MonomialBasis())
    m.theta[0]._value = real(0.2)
    m.theta[1]._value = real(0.1)
    m.theta[2]._value = real(-0.05)
    m.phi[0]._value = real(0.5)
    m.phi[1]._value = real(-0.25)
    c = m.get_layer_contract()
    for k in ('B_k', 'H_k', 'G_max', 'H_max', 'depth_hint'):
        assert k in c
    assert c['B_k'] >= 1.0 and c['H_k'] >= 1.0
    assert c['G_max'] >= 1.0 and c['H_max'] >= 1.0
    assert c['depth_hint'] >= 4


def test_trmultiinput_layer_contract_aggregates():
    multi = TRMultiInputRational(input_dim=4, n_outputs=2, d_p=1, d_q=1, basis=MonomialBasis(), shared_Q=True)
    # Set some coefficients to non-trivial values
    for head in multi.heads:
        head.theta[0]._value = real(0.1)
        head.theta[1]._value = real(-0.05)
    multi.heads[0].phi[0]._value = real(0.2)
    c = multi.get_layer_contract()
    for k in ('B_k', 'H_k', 'G_max', 'H_max', 'depth_hint'):
        assert k in c
    assert c['B_k'] >= 1.0
