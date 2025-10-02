"""
Minimal demo: policy-driven hybrid controller with TRRational.

Creates a simple rational layer with a denominator root near x≈1 (Q=1-x),
enables a default TRPolicy, and runs a few batches to show policy-mode and
|Q| quantiles reported by the hybrid controller.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import zeroproof as zp
from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.autodiff.hybrid_gradient import HybridGradientContext
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import enable_default_tr_policy


def make_dataset(
    n: int = 128, center: float = 1.0, span: float = 2e-4
) -> List[Tuple[List[zp.TRScalar], List[zp.TRScalar]]]:
    """Create a 1D dataset clustered around a pole at x≈center.

    Args:
        n: number of samples (denser helps quantiles)
        center: pole location (Q≈0 when x≈center)
        span: total window width around center
    """
    low = center - span / 2.0
    high = center + span / 2.0
    xs = [low + (high - low) * i / max(1, n - 1) for i in range(n)]
    # Single batch: inputs as TRScalars, targets all zeros (focus on controller metrics)
    inputs = [zp.real(float(x)) for x in xs]
    targets = [zp.real(0.0) for _ in xs]
    return [(inputs, targets)]


def build_model() -> TRRational:
    # Degree-2 numerator, degree-1 denominator; monomial basis
    model = TRRational(d_p=2, d_q=1, basis=MonomialBasis())
    # Force a denominator root near x=1 by setting φ1 ≈ -1 (Q(x) = 1 + φ1*x ≈ 1 - x)
    try:
        model.phi[0]._value = zp.real(-1.0)
    except Exception:
        pass
    return model


def main() -> None:
    random.seed(0)

    # Enable a default TR policy (or let the trainer do it via config)
    # Use a large ulp_scale so the guard band is visible in this tiny demo
    enable_default_tr_policy(ulp_scale=1e12, deterministic_reduction=True)

    # Build model
    model = build_model()

    # Prepare data (single batch repeated per epoch)
    # Dense cluster around x≈1 so q_p10 falls under tau_Q_on
    data = make_dataset(n=512, center=1.0, span=2e-4)

    print("Starting policy-driven hybrid demo...")
    for epoch in range(3):
        # Configure hybrid epoch state
        HybridGradientContext.update_epoch(epoch)
        GradientModeConfig.set_mode(GradientMode.HYBRID)
        GradientModeConfig.set_local_threshold(None)  # Prefer policy τ for decisions

        # One batch per epoch in this minimal demo
        inputs, targets = data[0]

        # Forward pass and policy-based tag classification occurs inside model.forward
        ys: list[TRNode] = []
        for x in inputs:
            y, _ = model.forward(x)
            ys.append(y)

        # Build a TRNode loss that preserves the graph: L = (1/N) Σ 0.5 * y^2
        half = TRNode.constant(zp.real(0.5))
        loss_node = None
        for y in ys:
            term = half * y * y
            loss_node = term if loss_node is None else (loss_node + term)
        if loss_node is None:
            return
        inv_n = TRNode.constant(zp.real(1.0 / float(len(ys))))
        loss_node = loss_node * inv_n

        # Backward to populate hybrid stats (DIV gradients consult the controller)
        loss_node.backward()

        # Collect stats
        stats = HybridGradientContext.get_statistics()
        print(
            f"Epoch {epoch+1}: loss={float(loss_node.value.value):.4f}, "
            f"near_pole_ratio={stats.get('near_pole_ratio', 0.0):.3f}, "
            f"policy_mode={stats.get('policy_mode', 'MR')}, "
            f"q_p10={stats.get('q_p10')}, q_p50={stats.get('q_p50')}, q_p90={stats.get('q_p90')}, "
            f"tau_q_on={stats.get('tau_q_on')}, tau_q_off={stats.get('tau_q_off')}, "
            f"sat_calls={stats.get('saturating_activations', 0)}/{stats.get('total_gradient_calls', 0)}"
        )

        # Update policy hysteresis at end of batch
        HybridGradientContext.end_batch_policy_update()

    print("Done.")


if __name__ == "__main__":
    main()
