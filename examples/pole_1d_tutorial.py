"""
1D Pole Tutorial

Train a TR-Rational layer on the function y = 1/(x - a) on [-2, 2],
excluding a small window around the pole at x=a. Demonstrates stable
training near singularities without epsilon hacks.

Usage:
  python examples/pole_1d_tutorial.py --center 0.5 --epochs 50 --seed 123
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import numpy as np

import zeroproof as zp


def make_dataset(
    center: float,
    n_train: int = 512,
    n_val: int = 128,
    exclude_radius: float = 0.05,
    seed: int | None = 123,
) -> Tuple[
    List[Tuple[List[zp.TRScalar], List[zp.TRScalar]]],
    List[Tuple[List[zp.TRScalar], List[zp.TRScalar]]],
]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def sample_points(n: int) -> np.ndarray:
        xs = []
        while len(xs) < n:
            x = float(np.random.uniform(-2.0, 2.0))
            if abs(x - center) >= exclude_radius:
                xs.append(x)
        return np.array(xs, dtype=float)

    def f(x: float) -> float:
        # y = 1/(x-a)
        return 1.0 / (x - center)

    def to_batches(xs: np.ndarray, batch_size: int = 64):
        batches: List[Tuple[List[zp.TRScalar], List[zp.TRScalar]]] = []
        for i in range(0, len(xs), batch_size):
            xb = xs[i : i + batch_size]
            inputs = [zp.real(float(v)) for v in xb]
            targets = [zp.real(float(f(v))) for v in xb]
            batches.append((inputs, targets))
        return batches

    x_train = sample_points(n_train)
    x_val = sample_points(n_val)
    train_batches = to_batches(x_train)
    val_batches = to_batches(x_val)
    return train_batches, val_batches


def main() -> None:
    ap = argparse.ArgumentParser(description="1D TR pole tutorial: y = 1/(x-a)")
    ap.add_argument("--center", type=float, default=0.5, help="Pole location a")
    ap.add_argument("--exclude", type=float, default=0.05, help="Exclusion radius around pole")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--pairwise", action="store_true", help="Enable deterministic reductions")
    ap.add_argument(
        "--eval-out", type=str, default="", help="Optional JSON path to save evaluation metrics"
    )
    ap.add_argument("--no-viz", action="store_true", help="Disable plots during evaluation")
    args = ap.parse_args()

    # Optional deterministic reductions
    if args.pairwise:
        zp.TRPolicyConfig.set_policy(zp.TRPolicy(deterministic_reduction=True))

    # Build dataset
    train_data, val_data = make_dataset(args.center, exclude_radius=args.exclude, seed=args.seed)

    # Model: simple rational with d_p=2, d_q=1 on Chebyshev basis
    basis = zp.layers.ChebyshevBasis()
    model = zp.layers.TRRational(d_p=2, d_q=1, basis=basis, alpha_phi=1e-3)

    # Trainer config
    cfg = zp.training.TrainingConfig(
        learning_rate=0.01,
        batch_size=64,
        max_epochs=args.epochs,
        use_adaptive_loss=True,
        target_coverage=0.95,
        lambda_learning_rate=0.05,
        verbose=True,
    )
    trainer = zp.training.TRTrainer(model, config=cfg)

    # Train
    history = trainer.train(train_data, val_data)

    # Quick report
    final_loss = history["loss"][-1]
    cov_hist = history.get("coverage", [])
    final_cov = cov_hist[-1] if cov_hist else float("nan")
    print(f"Final loss: {final_loss:.6f}; coverage: {final_cov:.3f}")

    # Evaluate near the pole and across a wider grid to verify behavior and log metrics
    xs_small = np.linspace(args.center - 0.1, args.center + 0.1, 11)
    real_count = 0
    for x in xs_small:
        _, tag = model.forward(zp.real(float(x)))
        if tag == zp.TRTag.REAL:
            real_count += 1
    print(f"Near-pole REAL ratio in small window: {real_count}/{len(xs_small)}")

    # Integrated evaluator over [-2, 2]
    xs = list(np.linspace(-2.0, 2.0, 201))
    evaluator = zp.utils.evaluation_api.create_evaluator(
        true_poles=[args.center],
        enable_viz=not args.no_viz,
        save_plots=not args.no_viz,
        plot_dir="evaluation_plots",
    )
    evaluator.evaluate_model(model, xs)
    print("Evaluator summary:", evaluator.get_summary_statistics())
    if args.eval_out:
        evaluator.export_metrics(args.eval_out)
        print(f"Saved evaluation metrics to {args.eval_out}")


if __name__ == "__main__":
    main()
