"""Unit tests for TR softmax surrogate."""

import math

from zeroproof.layers import tr_softmax
from zeroproof.autodiff import TRNode
from zeroproof.core import real, TRTag


def _to_vals(nodes):
    vals = []
    for n in nodes:
        tag = getattr(n, 'tag', None)
        if tag == TRTag.REAL:
            vals.append(float(n.value.value))
        else:
            vals.append(float('nan'))
    return vals


def test_tr_softmax_basic_properties():
    logits = [TRNode.constant(real(0.0)), TRNode.constant(real(1.0)), TRNode.constant(real(-1.0))]
    probs = tr_softmax(logits)
    assert isinstance(probs, list) and len(probs) == 3
    # All REAL in this simple case
    for p in probs:
        assert p.tag == TRTag.REAL
        assert float(p.value.value) >= 0.0
    s = sum(_to_vals(probs))
    assert abs(s - 1.0) < 1e-6
    # Monotonic: softmax(1) > softmax(0) > softmax(-1)
    vals = _to_vals(probs)
    assert vals[1] > vals[0] > vals[2]


def test_tr_softmax_shift_invariance():
    base = [0.1, 1.2, -0.7]
    logits1 = [TRNode.constant(real(x)) for x in base]
    logits2 = [TRNode.constant(real(x + 10.0)) for x in base]
    p1 = _to_vals(tr_softmax(logits1))
    p2 = _to_vals(tr_softmax(logits2))
    # Allow small approximation error due to rational exp
    for a, b in zip(p1, p2):
        assert abs(a - b) < 1e-2


def test_tr_softmax_nonreal_tolerated():
    from zeroproof.core import pinf
    logits = [TRNode.constant(real(0.0)), TRNode.constant(pinf())]
    probs = tr_softmax(logits)
    assert len(probs) == 2
    # Ensure tags are present and no crash; at least one may be non-REAL
    assert all(hasattr(p, 'tag') for p in probs)


def test_tr_softmax_extreme_large_and_small():
    # Very large/small logits should not crash and should produce sensible probabilities
    # Case 1: large positive dominates
    logits1 = [TRNode.constant(real(1000.0)), TRNode.constant(real(-1000.0)), TRNode.constant(real(0.0))]
    p1 = tr_softmax(logits1)
    vals1 = _to_vals(p1)
    s1 = sum(vals1)
    assert abs(s1 - 1.0) < 1e-3
    assert vals1[0] > 0.99  # dominant class near 1
    # Case 2: large negative but one less negative dominates
    logits2 = [TRNode.constant(real(-1000.0)), TRNode.constant(real(-1001.0)), TRNode.constant(real(-999.0))]
    p2 = tr_softmax(logits2)
    vals2 = _to_vals(p2)
    s2 = sum(vals2)
    assert abs(s2 - 1.0) < 1e-3
    # argmax should be index 2 (-999)
    assert vals2[2] == max(vals2)


def test_tr_softmax_with_infinities_and_phi():
    # +inf present — current implementation may yield non-REALs after shift-by-max
    from zeroproof.core import pinf, ninf, phi as tr_phi
    logits_inf = [TRNode.constant(real(0.0)), TRNode.constant(pinf()), TRNode.constant(real(1.0))]
    probs_inf = tr_softmax(logits_inf)
    assert len(probs_inf) == 3
    # Ensure tags exist and API is stable
    assert all(hasattr(p, 'tag') for p in probs_inf)
    # Include PHI explicitly
    logits_phi = [TRNode.constant(tr_phi()), TRNode.constant(real(0.0))]
    probs_phi = tr_softmax(logits_phi)
    assert len(probs_phi) == 2
    assert all(hasattr(p, 'tag') for p in probs_phi)


def test_tr_softmax_one_hot_policy_infinity():
    # Enable one-hot policy and ensure +INF yields deterministic one-hot
    from zeroproof.policy import TRPolicy, TRPolicyConfig
    pol = TRPolicy(softmax_one_hot_infinity=True)
    TRPolicyConfig.set_policy(pol)
    try:
        from zeroproof.core import pinf
        logits = [TRNode.constant(real(0.0)), TRNode.constant(pinf()), TRNode.constant(real(1.0))]
        probs = tr_softmax(logits)
        vals = _to_vals(probs)
        assert probs[1].tag == TRTag.REAL and abs(vals[1] - 1.0) < 1e-12
        assert probs[0].tag == TRTag.REAL and abs(vals[0]) < 1e-12
        assert probs[2].tag == TRTag.REAL and abs(vals[2]) < 1e-12

        # Multiple +INFs -> pick first index deterministically
        logits2 = [TRNode.constant(pinf()), TRNode.constant(pinf()), TRNode.constant(real(0.0))]
        probs2 = tr_softmax(logits2)
        vals2 = _to_vals(probs2)
        assert abs(vals2[0] - 1.0) < 1e-12
        assert abs(vals2[1]) < 1e-12 and abs(vals2[2]) < 1e-12
    finally:
        # Reset policy
        TRPolicyConfig.set_policy(None)
