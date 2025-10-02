"""Bounded-update (hybrid safety) tests across MR↔SAT switches.

We verify that per-batch parameter updates are bounded by lr * ||g||,
both in a near-pole (SAT) batch and a far-from-pole (MR) batch.
"""

import math

from zeroproof.autodiff.grad_mode import GradientMode, GradientModeConfig
from zeroproof.autodiff.hybrid_gradient import HybridGradientContext, HybridGradientSchedule
from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.training.coverage import CoverageTracker


def _trscalar_list(vals):
    return [real(float(v)) for v in vals]


def _param_vector(model):
    vec = []
    if hasattr(model, "parameters"):
        for p in model.parameters():
            if p.value.tag.name == "REAL":
                vec.append(float(p.value.value))
            else:
                vec.append(0.0)
    return vec


def _l2_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def test_bounded_update_near_and_far_batches():
    # Model: y = (θ0 + θ1 x) / (1 + φ1 x)
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    model.theta[0]._value = real(0.1)
    model.theta[1]._value = real(0.2)
    model.phi[0]._value = real(10.0)  # Pole near x ≈ -0.1

    opt = Optimizer(model.parameters(), learning_rate=0.01)
    cfg = HybridTrainingConfig(max_epochs=1, batch_size=2, verbose=False)
    trainer = HybridTRTrainer(model=model, optimizer=opt, config=cfg)

    # Hybrid setup: fixed local threshold so near-pole triggers SAT
    HybridGradientContext.reset()
    schedule = HybridGradientSchedule(
        enable=True, warmup_epochs=0, transition_epochs=0, delta_init=0.05, delta_final=0.05
    )
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)
    # Ensure hybrid mode active
    GradientModeConfig.set_mode(GradientMode.HYBRID)
    GradientModeConfig.set_saturation_bound(1.0)

    # Helper: run a single batch via internal API to get per-batch metrics
    def run_batch(xs, ys):
        coverage_tracker = CoverageTracker()
        return trainer._train_batch(_trscalar_list(xs), _trscalar_list(ys), coverage_tracker)

    # Batch 1: near pole (SAT expected)
    params_before = _param_vector(model)
    metrics1 = run_batch([-0.099, -0.101], [0.0, 0.0])
    params_after = _param_vector(model)
    # Compute update norm
    delta = [a - b for a, b in zip(params_after, params_before)]
    upd_norm = _l2_norm(delta)
    # Bound: lr * sqrt(gn_proxy)
    lr = opt.learning_rate
    gn = float(metrics1.get("gn_proxy", 0.0))
    bound = lr * math.sqrt(max(gn, 0.0)) + 1e-12
    assert upd_norm <= bound

    # Batch 2: far from pole (MR expected)
    params_before = _param_vector(model)
    metrics2 = run_batch([0.5, 0.7], [0.0, 0.0])
    params_after = _param_vector(model)
    delta = [a - b for a, b in zip(params_after, params_before)]
    upd_norm = _l2_norm(delta)
    gn = float(metrics2.get("gn_proxy", 0.0))
    bound = lr * math.sqrt(max(gn, 0.0)) + 1e-12
    assert upd_norm <= bound


def test_contract_safe_lr_clamps():
    # Model with non-trivial contract
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    model.theta[0]._value = real(0.1)
    model.theta[1]._value = real(0.2)
    model.phi[0]._value = real(1.5)

    # Large starting LR to force clamp
    opt = Optimizer(model.parameters(), learning_rate=10.0)
    cfg = HybridTrainingConfig(
        max_epochs=1,
        batch_size=2,
        verbose=False,
        use_contract_safe_lr=True,
        contract_c=1.0,
        loss_smoothness_beta=1.0,
    )
    trainer = HybridTRTrainer(model=model, optimizer=opt, config=cfg)

    def run_batch(xs, ys):
        coverage_tracker = CoverageTracker()
        return trainer._train_batch(_trscalar_list(xs), _trscalar_list(ys), coverage_tracker)

    # One batch triggers clamp
    contract = model.get_layer_contract()
    B_k = float(contract.get("B_k", 1.0))
    G_max = float(contract.get("G_max", 1.0))
    depth = int(contract.get("depth_hint", 4))
    eta_expected = 1.0 / (max(B_k, G_max) ** max(1, depth))

    run_batch([0.1, 0.2], [0.0, 0.0])

    # Allow small numerical differences due to contract estimation rounding
    assert trainer.optimizer.learning_rate <= eta_expected * 1.05
