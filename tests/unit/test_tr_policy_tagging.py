"""
Unit tests for TRPolicy-based tagging with guard bands and hysteresis.

Covers:
- Determinism outside guard bands (REAL tags stable across runs)
- Hysteresis behavior (enter SAT band → remain non-REAL until OFF threshold)
- Shared-Q concordance for multi-output rationals
"""

from zeroproof.policy import TRPolicy, TRPolicyConfig
from zeroproof.layers import TRRational, TRRationalMulti, MonomialBasis
from zeroproof.autodiff import TRNode
from zeroproof.core import real, TRTag
from zeroproof.layers.hybrid_rational import HybridTRRational
from zeroproof.autodiff.hybrid_gradient import HybridGradientSchedule, HybridGradientContext
from zeroproof.autodiff.grad_mode import GradientModeConfig, GradientMode


def setup_module(module):
    # Install a default policy with explicit thresholds
    pol = TRPolicy(
        tau_Q_on=1e-6,   # tiny thresholds for determinism test
        tau_Q_off=2e-6,
        tau_P_on=1e-9,
        tau_P_off=2e-9,
        keep_signed_zero=True,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)


def teardown_module(module):
    # Remove policy to avoid side-effects on other tests
    TRPolicyConfig.set_policy(None)


def test_determinism_outside_guard_bands():
    """Tags should be REAL and stable when |Q| >> tau_Q_off."""
    layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    # Set parameters so Q(x)=1+phi1*x with small magnitude change
    # Keep phi small so |Q| stays near 1 for tested x
    layer.theta[0]._value = real(0.5)  # P(0)=0.5 but irrelevant — outside band must be REAL
    layer.theta[1]._value = real(0.1)
    layer.phi[0]._value = real(0.01)   # Q ≈ 1 + 0.01 x

    xs = [real(-0.5), real(0.0), real(0.5), real(1.0)]
    tags_first = []
    tags_second = []

    # First pass
    for x in xs:
        y, tag = layer.forward(x)
        tags_first.append(tag)

    # Second pass (ensure state doesn't drift)
    for x in xs:
        y, tag = layer.forward(x)
        tags_second.append(tag)

    assert all(t == TRTag.REAL for t in tags_first)
    assert tags_first == tags_second


def test_hysteresis_band_behavior():
    """Entering ON band sets non-REAL; mid-band keeps previous tag; OFF returns REAL."""
    # Stronger thresholds for this test to simplify x selection
    pol = TRPolicy(
        tau_Q_on=1e-3,
        tau_Q_off=2e-3,
        tau_P_on=1e-6,
        tau_P_off=2e-6,
        deterministic_reduction=True,
        keep_signed_zero=True,
    )
    TRPolicyConfig.set_policy(pol)

    layer = TRRational(d_p=0, d_q=1, basis=MonomialBasis())
    # P(x) = 1 (so |P| >= tau_P_on)
    layer.theta[0]._value = real(1.0)
    # Q(x) = 1 - x (choose phi1 = -1.0)
    layer.phi[0]._value = real(-1.0)

    # |Q| = |1 - x|
    x_on = real(1.0 - 5e-4)   # |Q| = 5e-4 < tau_Q_on → non-REAL
    x_mid = real(1.0 - 1.5e-3)  # |Q| = 1.5e-3 ∈ (tau_on, tau_off) → keep previous
    x_off = real(1.0 - 3e-3)  # |Q| = 3e-3 ≥ tau_Q_off → REAL

    _, tag_on = layer.forward(x_on)
    _, tag_mid = layer.forward(x_mid)
    _, tag_off = layer.forward(x_off)

    assert tag_on in (TRTag.PINF, TRTag.NINF, TRTag.PHI)
    # Mid-band should keep non-REAL due to hysteresis
    assert tag_mid == tag_on
    # Outside OFF threshold returns REAL
    assert tag_off == TRTag.REAL


def test_shared_q_concordance_multi_output():
    """Multi-output with shared Q should yield concordant tags across outputs."""
    basis = MonomialBasis()
    multi = TRRationalMulti(d_p=1, d_q=1, n_outputs=2, basis=basis, shared_Q=True)
    # Share denominator is already wired via TRRationalMulti
    # Configure numerators differently but concordance relies on Q only
    # Set phi1 to place a zero near x=1 → small |Q|
    shared_phi = multi.layers[0].phi  # shared reference
    shared_phi[0]._value = real(-1.0)  # Q(x)=1 - x
    # Make sure numerators are non-zero so non-REAL becomes INF
    for layer in multi.layers:
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(0.0)

    x = TRNode.constant(real(1.0 - 5e-4))
    outs = multi.forward(x)
    tags = [tag for (_, tag) in outs]
    assert len(set(tags)) == 1, "Tags should be concordant across outputs with shared Q"


def test_determinism_multi_output_outside_guard_bands():
    """TRRationalMulti should be REAL and stable when |Q| >> tau_Q_off."""
    basis = MonomialBasis()
    multi = TRRationalMulti(d_p=1, d_q=1, n_outputs=3, basis=basis, shared_Q=True)
    # Configure shared denominator Q(x) = 1 + 0.01 x (far from zero over test domain)
    shared_phi = multi.layers[0].phi
    shared_phi[0]._value = real(0.01)
    # Different numerators per output (should not affect REAL tagging outside band)
    for i, layer in enumerate(multi.layers):
        layer.theta[0]._value = real(0.1 * (i + 1))
        layer.theta[1]._value = real(-0.05 * (i + 1))

    xs = [TRNode.constant(real(t)) for t in (-0.7, -0.2, 0.0, 0.4, 0.9)]
    first_pass = []
    second_pass = []

    for x in xs:
        outs = multi.forward(x)
        first_pass.append([tag for (_, tag) in outs])

    for x in xs:
        outs = multi.forward(x)
        second_pass.append([tag for (_, tag) in outs])

    # All REAL and stable
    for tags in first_pass:
        assert all(t == TRTag.REAL for t in tags)
    assert first_pass == second_pass


def test_determinism_hybrid_tr_rational_outside_guard_bands():
    """HybridTRRational delegates to TRRational; tags remain REAL and stable outside band."""
    layer = HybridTRRational(d_p=1, d_q=1, basis=MonomialBasis(), hybrid_schedule=None, track_Q_values=False)
    # Q(x) = 1 + 0.01 x keeps |Q| far above tau
    layer.theta[0]._value = real(0.25)
    layer.theta[1]._value = real(0.0)
    layer.phi[0]._value = real(0.01)

    xs = [real(-0.5), real(0.0), real(0.5), real(1.0)]
    tags_first = []
    tags_second = []
    for x in xs:
        _, tag = layer.forward(x)
        tags_first.append(tag)
    for x in xs:
        _, tag = layer.forward(x)
        tags_second.append(tag)

    assert all(t == TRTag.REAL for t in tags_first)
    assert tags_first == tags_second


def test_determinism_hybrid_with_schedule_outside_band():
    """With an active hybrid schedule, policy tagging remains REAL and stable outside band."""
    # Set up a trivial schedule (no warmup, no transition specifics)
    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(enable=True, warmup_epochs=0, transition_epochs=0,
                                      delta_init=1e-2, delta_final=1e-2)
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    layer = HybridTRRational(d_p=1, d_q=1, basis=MonomialBasis(), hybrid_schedule=schedule, track_Q_values=False)
    # Q(x) = 1 + 0.02 x stays far above tiny tau thresholds across test inputs
    layer.theta[0]._value = real(0.1)
    layer.theta[1]._value = real(-0.05)
    layer.phi[0]._value = real(0.02)

    xs = [real(-0.8), real(-0.3), real(0.0), real(0.6), real(0.9)]
    tags_first = []
    tags_second = []

    for x in xs:
        _, tag = layer.forward(x)
        tags_first.append(tag)
    for x in xs:
        _, tag = layer.forward(x)
        tags_second.append(tag)

    assert all(t == TRTag.REAL for t in tags_first)
    assert tags_first == tags_second

    # Clean up hybrid context
    HybridGradientContext.reset()


def test_policy_flip_counts_across_batches():
    """Flip count increases when batches cross ON/OFF thresholds."""
    # Install policy with distinct ON/OFF
    pol = TRPolicy(
        tau_Q_on=0.1,
        tau_Q_off=0.2,
        tau_P_on=1e-6,
        tau_P_off=2e-6,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)

    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(enable=True, warmup_epochs=0, transition_epochs=0,
                                      delta_init=0.0, delta_final=0.0)
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    # Batch 1: small q values → enter SAT (flip 0->1)
    for q in (0.05, 0.08, 0.09, 0.03):
        HybridGradientContext.update_q_value(q)
    HybridGradientContext.end_batch_policy_update()

    # Batch 2: large q values → exit to MR (flip 1->0)
    for q in (0.25, 0.3, 0.5):
        HybridGradientContext.update_q_value(q)
    HybridGradientContext.end_batch_policy_update()

    stats = HybridGradientContext.get_statistics()
    assert stats.get('policy_flip_count', 0) >= 2
    flip_rate = stats.get('flip_rate', 0.0)
    assert flip_rate > 0.0

    # Clean up
    HybridGradientContext.reset()
    TRPolicyConfig.set_policy(None)


def test_local_hybrid_saturation_triggers_with_large_delta():
    """Local HYBRID saturation should activate when delta is large relative to |Q|.

    Builds a small TRRational, sets a HYBRID schedule with a large local threshold,
    computes a simple loss over a batch, and asserts saturating activations > 0.
    """
    # No policy needed for this test
    TRPolicyConfig.set_policy(None)

    # Reset hybrid context and enable HYBRID mode
    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(enable=True, warmup_epochs=0, transition_epochs=0,
                                      delta_init=10.0, delta_final=10.0)
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)
    GradientModeConfig.set_mode(GradientMode.HYBRID)

    # Simple rational: P(x)=a0+a1 x, Q(x)=1+b1 x
    layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    layer.theta[0]._value = real(0.2)
    layer.theta[1]._value = real(-0.1)
    layer.phi[0]._value = real(0.2)

    # Build a tiny batch and loss L = (1/N) Σ 0.5 * y^2
    xs = [real(-1.0), real(-0.5), real(0.0), real(0.5), real(1.0)]
    half = TRNode.constant(real(0.5))
    loss_node = None
    for x in xs:
        y, _ = layer.forward(x)
        term = half * y * y
        loss_node = term if loss_node is None else (loss_node + term)
    inv_n = TRNode.constant(real(1.0 / float(len(xs))))
    loss_node = loss_node * inv_n

    # Backward pass to invoke DIV gradients with HYBRID logic
    loss_node.backward()

    stats = HybridGradientContext.get_statistics()
    sat = stats.get('saturating_activations', 0)
    total = stats.get('total_gradient_calls', 0)
    assert total > 0
    assert sat > 0, f"Expected saturating activations > 0, got {sat} (total={total})"

    # Cleanup
    HybridGradientContext.reset()
    GradientModeConfig.reset()
