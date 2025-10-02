"""
Tests for finite/low-density switching in the hybrid policy controller.

Simulates batches with small-|Q| (enter SAT) then large-|Q| (exit MR) and
checks that flip counts are finite and flip_rate remains bounded.
"""

from zeroproof.autodiff.hybrid_gradient import HybridGradientContext, HybridGradientSchedule
from zeroproof.policy import TRPolicy, TRPolicyConfig


def setup_module(module):
    # Enable a reasonable policy band for testing hysteresis
    pol = TRPolicy(
        tau_Q_on=0.1,
        tau_Q_off=0.2,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)


def teardown_module(module):
    TRPolicyConfig.set_policy(None)


def test_finite_switching_over_batches():
    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(
        enable=True, warmup_epochs=0, transition_epochs=0, delta_init=0.0, delta_final=0.0
    )
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    # Batch 1–2: Near-pole → enter SAT
    for _ in range(2):
        for q in (0.05, 0.08, 0.03, 0.09):
            HybridGradientContext.update_q_value(q)
        HybridGradientContext.end_batch_policy_update()

    # Batch 3–10: Far from pole → exit to MR and remain
    for _ in range(3, 11):
        for q in (0.25, 0.4, 0.7, 1.2):
            HybridGradientContext.update_q_value(q)
        HybridGradientContext.end_batch_policy_update()

    stats = HybridGradientContext.get_statistics()
    flips = stats.get("policy_flip_count", 0)
    flip_rate = stats.get("flip_rate", 0.0)

    # Expect at most 2 flips (MR->SAT, SAT->MR) across these batches
    assert flips <= 2
    # Flip density should be low across 10 batches
    assert 0.0 <= flip_rate <= 0.5

    HybridGradientContext.reset()
