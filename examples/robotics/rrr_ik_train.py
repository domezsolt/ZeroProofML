"""
RRR-arm inverse kinematics training with ZeroProofML (3R; multi-output; shared-Q).

Setup: input [theta1, theta2, theta3, dx, dy] -> output [dtheta1, dtheta2, dtheta3].
Model: TRMultiInputRational(shared_Q=True) with P3/Q2 heads.
Data: stratify buckets by manipulability σ1·σ2 = sqrt(det(J J^T)).
Metrics: per-bucket MSE, PLE to 3R singular sets, sign consistency (θ2, θ3),
         residual consistency, coverage.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis, TRMultiInputRational
from zeroproof.metrics.pole_3r import compute_pole_metrics_3r
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.utils.config import DEFAULT_BUCKET_EDGES
from zeroproof.utils.seeding import set_global_seed

try:
    # Relative + script import compatibility
    from .rrr_ik_dataset import IK3RSample, RRRDatasetGenerator
except Exception:
    from rrr_ik_dataset import IK3RSample, RRRDatasetGenerator


def _prepare_data(samples: List[IK3RSample]) -> Tuple[List, List, List[float]]:
    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    detj: List[float] = []
    for s in samples:
        inputs.append([float(s.theta1), float(s.theta2), float(s.theta3), float(s.dx), float(s.dy)])
        targets.append([float(s.dtheta1), float(s.dtheta2), float(s.dtheta3)])
        detj.append(abs(float(s.det_J)))
    return inputs, targets, detj


def _bucketize_mse(
    mse_list: List[float], keys: List[float], edges: List[float]
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[float]] = {
        f"({edges[i]:.0e},{edges[i+1]:.0e}]": [] for i in range(len(edges) - 1)
    }
    counts: Dict[str, int] = {k: 0 for k in buckets}
    for mse, dj in zip(mse_list, keys):
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (dj > lo) and (dj <= hi):
                k = f"({lo:.0e},{hi:.0e}]"
                buckets[k].append(mse)
                counts[k] += 1
                break
    agg: Dict[str, Dict[str, Any]] = {}
    for k, vals in buckets.items():
        if vals:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "count": int(counts[k]),
            }
        else:
            agg[k] = {"mean": None, "std": None, "count": int(counts[k])}
    return agg


def _plot_per_bucket_bars(agg: Dict[str, Dict[str, Any]], outpath: str) -> None:
    try:
        import matplotlib.pyplot as plt

        labels = list(agg.keys())
        means = [agg[k]["mean"] if agg[k]["mean"] is not None else 0.0 for k in labels]
        stds = [agg[k]["std"] if agg[k]["std"] is not None else 0.0 for k in labels]
        counts = [agg[k]["count"] for k in labels]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, means, yerr=stds, alpha=0.8, capsize=3, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for xi, c in zip(x, counts):
            ax.text(
                xi,
                max(0.0, means[int(xi)] if xi < len(means) else 0.0) * 1.02 + 1e-6,
                str(c),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylabel("MSE (mean ± std)")
        ax.set_title("Per-bucket MSE by manipulability (3R)")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"Saved per-bucket MSE bars to {outpath}")
    except Exception as e:
        print(f"Plotting skipped ({e})")


def train_and_evaluate(
    dataset_file: str,
    output_dir: str,
    epochs: int = 60,
    learning_rate: float = 0.01,
    batch_size: int = 256,
    degree_p: int = 3,
    degree_q: int = 2,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    # Load dataset
    with open(dataset_file, "r") as f:
        data = json.load(f)
    samples_raw = data["samples"]
    # Reconstruct for typing clarity (keep dicts for speed)
    samples = samples_raw

    # Train/test split (80/20 by order; dataset JSON may already be stratified)
    n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]
    train_inputs, train_targets, _ = _prepare_data([IK3RSample(**s) for s in train_samples])
    test_inputs, test_targets, test_detj = _prepare_data([IK3RSample(**s) for s in test_samples])

    # Model
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    model = TRMultiInputRational(
        input_dim=5,
        n_outputs=3,
        d_p=degree_p,
        d_q=degree_q,
        basis=MonomialBasis(),
        hidden_dims=[16],
        shared_Q=True,
        enable_pole_head=True,
    )
    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=learning_rate),
        config=HybridTrainingConfig(
            learning_rate=learning_rate,
            max_epochs=epochs,
            use_hybrid_gradient=True,
            use_tag_loss=True,
            lambda_tag=0.05,
            use_pole_head=True,
            lambda_pole=0.1,
            enable_anti_illusion=True,
            lambda_residual=0.02,
            log_interval=1,
            enable_structured_logging=False,
            save_plots=False,
        ),
    )

    # Mini-batch training loop
    set_global_seed(seed)
    history: List[float] = []
    for epoch in range(epochs):
        epoch_losses: List[float] = []
        for i in range(0, len(train_inputs), batch_size):
            batch_in = train_inputs[i : i + batch_size]
            batch_tg = train_targets[i : i + batch_size]
            tr_inputs = [[TRNode.constant(real(x)) for x in inp] for inp in batch_in]
            result = trainer._train_batch_multi(
                tr_inputs, [[float(y) for y in tgt] for tgt in batch_tg]
            )
            epoch_losses.append(result.get("loss", float("inf")))
        if epoch_losses:
            avg_loss = float(np.mean(epoch_losses))
            history.append(avg_loss)
            if (epoch % 5) == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    # Evaluation
    per_sample_mse: List[float] = []
    predictions: List[List[float]] = []
    tags_all: List[List[TRTag]] = []
    for inp, tgt in zip(test_inputs, test_targets):
        tr_inp = [TRNode.constant(real(x)) for x in inp]
        outs = model.forward(tr_inp)
        pred_vec: List[float] = []
        tags_vec: List[TRTag] = []
        for y, tag in outs:
            if tag == TRTag.REAL:
                pred_vec.append(float(y.value.value))
            else:
                pred_vec.append(0.0)
            tags_vec.append(tag)
        predictions.append(pred_vec)
        tags_all.append(tags_vec)
        per_sample_mse.append(float(np.mean([(pv - tv) ** 2 for pv, tv in zip(pred_vec, tgt)])))

    # Coverage
    total_outputs = len(tags_all) * 3
    real_outputs = sum(1 for row in tags_all for t in row if t == TRTag.REAL)
    coverage = float(real_outputs / max(1, total_outputs))

    # Per-bucket MSE
    edges = data.get("metadata", {}).get("bucket_edges", DEFAULT_BUCKET_EDGES)
    agg = _bucketize_mse(per_sample_mse, test_detj, edges)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    _plot_per_bucket_bars(agg, os.path.join(output_dir, "figures", "e3r_per_bucket_bars.png"))

    # 3R pole metrics
    pole_metrics = compute_pole_metrics_3r(test_inputs, predictions)

    results = {
        "model": "TRMultiInputRational_sharedQ",
        "params": len(model.parameters()),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "final_train_loss": history[-1] if history else None,
        "test_mse_mean": float(np.mean(per_sample_mse)) if per_sample_mse else None,
        "coverage_outputs": coverage,
        "per_bucket": agg,
        "bucket_edges": edges,
        "pole_metrics_3r": pole_metrics,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "e3r_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved 3R results to {out_json}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Train 3R IK with TR (shared-Q; multi-output)")
    ap.add_argument("--dataset", type=str, required=True, help="Path to 3R IK dataset JSON")
    ap.add_argument("--output_dir", type=str, default="results/robotics/e3r")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--learning_rate", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--degree_p", type=int, default=3)
    ap.add_argument("--degree_q", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    set_global_seed(args.seed)
    train_and_evaluate(
        dataset_file=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        degree_p=args.degree_p,
        degree_q=args.degree_q,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
