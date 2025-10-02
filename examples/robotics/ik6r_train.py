"""
Train TR model on synthetic 6R IK dataset and report per-bin (d1) metrics.

Inputs: JSON from examples/robotics/ik6r_dataset.py
Model: TRMultiInputRational (input 12: q(6)+twist(6); output 6: dq)
Metrics: overall/test MSE, per-bin MSE by d1 edges, multi-singularity subset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis, TRMultiInputRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.utils.seeding import set_global_seed
from zeroproof.utils.serialization import to_builtin


def _prepare(
    data: Dict[str, Any]
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[int], List[float]]:
    samples = data["samples"]
    meta = data.get("metadata", {})
    n_total = len(samples)
    n_train = int(meta.get("n_train", int(0.8 * n_total)))
    # Build arrays
    inputs = []
    targets = []
    d1_vals = []
    bins = []
    for s in samples:
        q = [float(x) for x in s["q"]]
        twist = [float(x) for x in s["twist"]]
        dq = [float(x) for x in s["dq_target"]]
        x = q + twist
        inputs.append(x)
        targets.append(dq)
        d1_vals.append(float(s.get("d1", float("nan"))))
        bins.append(int(s.get("bin_idx", 0)))
    train = (inputs[:n_train], targets[:n_train])
    test = (inputs[n_train:], targets[n_train:])
    test_bins = bins[n_train:]
    test_d1 = d1_vals[n_train:]
    return train[0], train[1], test[0], test_bins, test_d1


def _bucketize_mse(
    per_sample_mse: List[float], test_bins: List[int], edges: List[float]
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, List[float]] = {
        f"({edges[i]:.0e},{edges[i+1]:.0e}]": [] for i in range(len(edges) - 1)
    }
    for mse, b in zip(per_sample_mse, test_bins):
        key = (
            f"({edges[b]:.0e},{edges[b+1]:.0e}]" if b < len(edges) - 1 else f"({edges[-2]:.0e},inf]"
        )
        agg.setdefault(key, []).append(float(mse))
    out: Dict[str, Dict[str, float]] = {}
    for k, xs in agg.items():
        if xs:
            mu = float(np.mean(xs))
            sd = float(np.std(xs))
            out[k] = {"mean_mse": mu, "std_mse": sd, "n": int(len(xs))}
        else:
            out[k] = {"mean_mse": None, "std_mse": None, "n": 0}
    return out


def train_and_eval(
    dataset_file: str, output_dir: str, epochs: int, lr: float, batch_size: int, seed: int | None
) -> Dict[str, Any]:
    with open(dataset_file, "r") as fh:
        data = json.load(fh)
    raw_edges = data.get("metadata", {}).get(
        "bucket_edges", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]
    )
    # Normalize edges to floats for formatting and comparisons
    edges = []
    for e in raw_edges:
        try:
            edges.append(float(e))
        except Exception:
            s = str(e).strip().lower()
            edges.append(float("inf") if s in ("inf", "+inf", "infinity") else float(e))
    tr_in, tr_tg, te_in, te_bins, _ = _prepare(data)

    set_global_seed(seed)
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    model = TRMultiInputRational(
        input_dim=12,
        n_outputs=6,
        d_p=3,
        d_q=2,
        basis=MonomialBasis(),
        hidden_dims=[16],
        shared_Q=True,
    )
    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=lr),
        config=HybridTrainingConfig(
            learning_rate=lr,
            max_epochs=epochs,
            use_hybrid_gradient=True,
            use_tag_loss=True,
            lambda_tag=0.05,
            use_pole_head=True,
            lambda_pole=0.1,
            enable_anti_illusion=False,
            log_interval=1,
            enable_structured_logging=False,
            save_plots=False,
        ),
    )

    # Mini-batch training
    history: List[float] = []
    for ep in range(epochs):
        losses_ep: List[float] = []
        for i in range(0, len(tr_in), batch_size):
            batch_in = tr_in[i : i + batch_size]
            batch_tg = tr_tg[i : i + batch_size]
            t_inputs = [[TRNode.constant(real(x)) for x in row] for row in batch_in]
            result = trainer._train_batch_multi(
                t_inputs, [[float(y) for y in row] for row in batch_tg]
            )
            if "loss" in result:
                losses_ep.append(float(result["loss"]))
        if losses_ep:
            history.append(float(np.mean(losses_ep)))
            if ep % max(1, epochs // 10) == 0:
                print(f"Epoch {ep}: loss={history[-1]:.6f}")

    # Evaluation
    per_mse: List[float] = []
    predictions: List[List[float]] = []
    for row in te_in:
        ts = [TRNode.constant(real(x)) for x in row]
        outs = model.forward(ts)
        pred = []
        for y, tag in outs:
            pred.append(float(y.value.value) if tag == TRTag.REAL else 0.0)
        predictions.append(pred)
    # Pair with available targets
    test_targets = data["samples"][len(tr_in) :]
    for p, s in zip(predictions, test_targets):
        tgt = [float(x) for x in s["dq_target"]]
        per_mse.append(float(np.mean([(pv - tv) ** 2 for pv, tv in zip(p, tgt)])))

    overall_mse = float(np.mean(per_mse)) if per_mse else float("inf")
    per_bucket = _bucketize_mse(per_mse, te_bins, edges)

    results = {
        "model": "TRMultiInputRational_sharedQ",
        "params": len(model.parameters()),
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "final_train_loss": history[-1] if history else None,
        "test_mse_mean": overall_mse,
        "per_bucket": per_bucket,
        "bucket_edges": edges,
        "test_size": len(te_in),
        "predictions": predictions,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "ik6r_results.json")
    with open(out_json, "w") as fh:
        json.dump(to_builtin(results), fh, indent=2)
    print(f"Saved 6R results to {out_json}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TR model on 6R IK dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", default="results/robotics/ik6r")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--learning_rate", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    train_and_eval(
        dataset_file=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
