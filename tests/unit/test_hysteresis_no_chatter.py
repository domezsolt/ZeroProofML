"""Finite/low-density switching (no-chatter) tests for policy-driven hysteresis.

We simulate batches with policy hysteresis ON/OFF thresholds and verify that:
 - With q values well outside the guard band, there are no flips.
 - Occasional excursions cause a finite number of flips.
 - Over many stable batches, the flip density remains low.
"""

from zeroproof.autodiff.hybrid_gradient import HybridGradientContext, HybridGradientSchedule
from zeroproof.policy import TRPolicy, TRPolicyConfig


def test_no_chatter_with_stable_q_values():
    # Install policy with clear ON/OFF hysteresis thresholds
    pol = TRPolicy(
        tau_Q_on=1e-3,
        tau_Q_off=2e-3,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)

    # Hybrid schedule enabled (thresholds come from policy; schedule delta may be None)
    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(
        enable=True, warmup_epochs=0, transition_epochs=0, delta_init=0.0, delta_final=0.0
    )
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    # Batch loop with q values well above tau_Q_off (stable MR region)
    for _ in range(50):
        for q in (0.01, 0.02, 0.03):  # >> tau_Q_off
            HybridGradientContext.update_q_value(q)
        HybridGradientContext.end_batch_policy_update()

    stats = HybridGradientContext.get_statistics()
    assert stats.get("policy_flip_count", 0) == 0
    assert stats.get("flip_rate", 0.0) == 0.0


def test_finite_flips_with_rare_excursions():
    # Install policy with hysteresis
    pol = TRPolicy(
        tau_Q_on=1e-3,
        tau_Q_off=2e-3,
        deterministic_reduction=True,
    )
    TRPolicyConfig.set_policy(pol)

    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(
        enable=True, warmup_epochs=0, transition_epochs=0, delta_init=0.0, delta_final=0.0
    )
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    # Mostly stable MR batches with occasional near-pole batches
    flips = 0
    batches = 0
    for i in range(100):
        if i in (10, 50):
            # Excursions near/on the ON threshold (enter SAT then leave)
            for q in (5e-4, 8e-4, 1.5e-3):
                HybridGradientContext.update_q_value(q)
        else:
            # Stable MR
            for q in (0.01, 0.02, 0.03):
                HybridGradientContext.update_q_value(q)
        HybridGradientContext.end_batch_policy_update()
        batches += 1

    stats = HybridGradientContext.get_statistics()
    flips = stats.get("policy_flip_count", 0) or 0
    flip_rate = stats.get("flip_rate", 0.0) or 0.0
    # Finite flips and low flip density
    assert flips > 0 and flips < 10
    assert flip_rate < 0.1
