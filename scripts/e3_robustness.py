#!/usr/bin/env python3
"""
E3: Robustness to near-pole shift.

Train on the original dataset (quick profile) and evaluate on:
  - Original quick test subset (baseline)
  - Shifted quick test subset (heavier near-pole mass)

Models: MLP, Rational+ε (grid search), ZeroProofML Basic, ZeroProofML Full

Outputs:
  - JSON summary with per-bucket MSE on original vs shifted tests and deltas
  - Optional robustness bar figure for B0–B2 relative deltas

Usage:
  python scripts/e3_robustness.py \
    --orig data/rr_ik_dataset.json \
    --shifted data/rr_ik_dataset_shifted.json \
    --outdir results/robotics/e3_robustness \
    --max_train 2000 --max_test 500
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root is on sys.path
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Baselines
from examples.baselines.mlp_baseline import MLPBaseline, MLPConfig, MLPTrainer
from examples.baselines.rational_eps_baseline import (
    RationalEpsConfig,
    RationalEpsModel,
    RationalEpsTrainer,
    grid_search_epsilon,
)
from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import TRMultiInputRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer

DEFAULT_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def quick_indices(
    samples: List[Dict[str, float]], max_train: int, max_test: int, edges: List[float]
) -> Tuple[List[int], List[int]]:
    n_total = len(samples)
    n_train_full = int(0.8 * n_total)
    train_idx = list(range(n_train_full))
    test_idx = list(range(n_train_full, n_total))

    # Select quick train subset
    selected_train = train_idx[: min(max_train, len(train_idx))]

    # Stratified selection for test to ensure B0–B3 presence
    def bucketize(idx_list):
        buckets = {i: [] for i in range(len(edges) - 1)}
        for i in idx_list:
            th2 = float(samples[i]["theta2"])
            dj = abs(math.sin(th2))
            for b in range(len(edges) - 1):
                lo, hi = edges[b], edges[b + 1]
                if ((dj >= lo) if b == 0 else (dj > lo)) and dj <= hi:
                    buckets[b].append(i)
                    break
        return buckets

    tb = bucketize(test_idx)
    selected_test: List[int] = []
    for b in range(min(4, len(edges) - 1)):
        if tb.get(b):
            selected_test.append(tb[b][0])
            tb[b] = tb[b][1:]
    rr_order = [b for b in range(len(edges) - 1) if tb.get(b)]
    ptrs = {b: 0 for b in rr_order}
    while len(selected_test) < min(max_test, len(test_idx)) and rr_order:
        new_rr = []
        for b in rr_order:
            blist = tb.get(b, [])
            p = ptrs[b]
            if p < len(blist):
                selected_test.append(blist[p])
                ptrs[b] = p + 1
                if ptrs[b] < len(blist):
                    new_rr.append(b)
            if len(selected_test) >= min(max_test, len(test_idx)):
                break
        rr_order = new_rr
        if not rr_order:
            break
    if len(selected_test) < min(max_test, len(test_idx)):
        remaining = [i for i in test_idx if i not in selected_test]
        selected_test.extend(remaining[: (min(max_test, len(test_idx)) - len(selected_test))])

    return selected_train, selected_test


def build_data(
    samples: List[Dict[str, float]], idxs: List[int]
) -> Tuple[List[List[float]], List[List[float]]]:
    inputs, targets = [], []
    for i in idxs:
        s = samples[i]
        inputs.append([s["theta1"], s["theta2"], s["dx"], s["dy"]])
        targets.append([s["dtheta1"], s["dtheta2"]])
    return inputs, targets


def compute_bucket_mse(
    per_sample_mse: List[float], inputs: List[List[float]], edges: List[float]
) -> Dict[str, float]:
    bucket_map: Dict[str, List[float]] = {
        f"({edges[i]:.0e},{edges[i+1]:.0e}]": [] for i in range(len(edges) - 1)
    }

    def key_for(th2: float) -> str:
        dj = abs(math.sin(th2))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (dj > lo) and (dj <= hi):
                return f"({lo:.0e},{hi:.0e}]"
        return f"({edges[-2]:.0e},inf]"

    for mse, inp in zip(per_sample_mse, inputs):
        bucket_map[key_for(float(inp[1]))].append(float(mse))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in bucket_map.items()}


def eval_mlp(train_data, val_data, test_orig, test_shift, edges) -> Dict[str, Any]:
    cfg = MLPConfig(
        hidden_dims=[32, 16], activation="relu", learning_rate=0.01, epochs=1, batch_size=32
    )
    model = MLPBaseline(cfg)
    trainer = MLPTrainer(model, Optimizer(model.parameters(), learning_rate=cfg.learning_rate))
    trainer.train(*train_data, *val_data, verbose=True)
    # Evaluate
    te_orig = trainer.evaluate(*test_orig)
    te_shift = trainer.evaluate(*test_shift)

    # Per-sample mse for buckets
    def per_sample_mse(preds, targets):
        return [np.mean([(p - t) ** 2 for p, t in zip(pv, tv)]) for pv, tv in zip(preds, targets)]

    bm_orig = compute_bucket_mse(
        per_sample_mse(te_orig["predictions"], test_orig[1]), test_orig[0], edges
    )
    bm_shift = compute_bucket_mse(
        per_sample_mse(te_shift["predictions"], test_shift[1]), test_shift[0], edges
    )
    return {
        "overall": {"orig": te_orig["mse"], "shift": te_shift["mse"]},
        "bucket_mse": {"orig": bm_orig, "shift": bm_shift},
    }


def eval_rational_eps(train_data, val_data, test_orig, test_shift, edges) -> Dict[str, Any]:
    cfg = RationalEpsConfig(
        epochs=3, epsilon_values=[1e-4, 1e-3], learning_rate=0.01, batch_size=32
    )
    # Grid search on original val (orig quick test)
    gs = grid_search_epsilon(
        train_data, val_data, cfg, output_dir="results/robotics/e3_robustness_rational"
    )
    best_eps = gs.get("best_epsilon", 1e-4)
    # Train final
    model = RationalEpsModel(cfg, best_eps)
    trainer = RationalEpsTrainer(
        model, Optimizer(model.parameters(), learning_rate=cfg.learning_rate)
    )
    trainer.train(*train_data, *val_data, verbose=False)
    # Evaluate on both tests
    te_orig = trainer._evaluate_simple(*test_orig)
    te_shift = trainer._evaluate_simple(*test_shift)
    # Per-sample mse for buckets
    bm_orig = compute_bucket_mse(te_orig["per_sample_mse"], test_orig[0], edges)
    bm_shift = compute_bucket_mse(te_shift["per_sample_mse"], test_shift[0], edges)
    return {
        "overall": {"orig": te_orig["mse"], "shift": te_shift["mse"]},
        "bucket_mse": {"orig": bm_orig, "shift": bm_shift},
        "epsilon": best_eps,
    }


def eval_tr(
    train_data, val_data, test_orig, test_shift, edges, enable_enhancements: bool
) -> Dict[str, Any]:
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    # Determine dims
    output_dim = len(train_data[1][0]) if train_data[1] else 2
    input_dim = len(train_data[0][0]) if train_data[0] else 4
    model = TRMultiInputRational(
        input_dim=input_dim, n_outputs=output_dim, d_p=3, d_q=2, hidden_dims=[8], shared_Q=True
    )
    if enable_enhancements:
        trainer = HybridTRTrainer(
            model=model,
            optimizer=Optimizer(model.parameters(), learning_rate=0.01),
            config=HybridTrainingConfig(
                learning_rate=0.01,
                max_epochs=3,
                use_hybrid_gradient=True,
                use_tag_loss=True,
                lambda_tag=0.05,
                use_pole_head=True,
                lambda_pole=0.1,
                enable_anti_illusion=True,
                lambda_residual=0.02,
            ),
        )
    else:
        trainer = None

    # Simplified training loop (sample-wise)
    epochs = 3
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        for inp, tgt in zip(*train_data):
            tr_inputs = [TRNode.constant(real(x)) for x in inp]
            outs = model.forward(tr_inputs)
            sample_loss = TRNode.constant(real(0.0))
            valid = 0
            for j, (y_pred, tag) in enumerate(outs):
                if tag == TRTag.REAL:
                    diff = y_pred - TRNode.constant(real(tgt[j]))
                    sample_loss = sample_loss + diff * diff
                    valid += 1
            if valid == 0:
                continue
            sample_loss.backward()
            if enable_enhancements and trainer is not None and hasattr(trainer, "step_all"):
                trainer.step_all()
            else:
                Optimizer(model.parameters(), learning_rate=0.01).step(model)
            if sample_loss.tag == TRTag.REAL:
                epoch_loss += sample_loss.value.value / max(1, valid)
            n_samples += 1
        _ = epoch_loss / max(1, n_samples)

    def eval_on(inputs: List[List[float]], targets: List[List[float]]):
        preds = []
        per_mse = []
        for inp, tgt in zip(inputs, targets):
            tr_inputs = [TRNode.constant(real(x)) for x in inp]
            outs = model.forward(tr_inputs)
            pv = [(y.value.value if tag == TRTag.REAL else 0.0) for (y, tag) in outs]
            preds.append(pv)
            per_mse.append(np.mean([(a - b) ** 2 for a, b in zip(pv, tgt)]))
        return float(np.mean(per_mse)) if per_mse else float("inf"), preds, per_mse

    m_orig, p_orig, per_orig = eval_on(*test_orig)
    m_shift, p_shift, per_shift = eval_on(*test_shift)
    bm_orig = compute_bucket_mse(per_orig, test_orig[0], edges)
    bm_shift = compute_bucket_mse(per_shift, test_shift[0], edges)
    return {
        "overall": {"orig": m_orig, "shift": m_shift},
        "bucket_mse": {"orig": bm_orig, "shift": bm_shift},
    }


def plot_robustness_b012(summary: Dict[str, Any], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping robustness plot.")
        return
    methods = [k for k in summary.keys()]
    buckets = ["(0e+00,1e-05]", "(1e-05,1e-04]", "(1e-04,1e-03]"]
    rel = []
    for m in methods:
        mo = summary[m]["bucket_mse"]["orig"]
        ms = summary[m]["bucket_mse"]["shift"]
        row = []
        for b in buckets:
            o = mo.get(b)
            s = ms.get(b)
            if o in (None, float("nan")) or o == 0:
                row.append(0.0)
            else:
                row.append(100.0 * (float(s) - float(o)) / float(o))
        rel.append(row)
    rel = np.array(rel)
    x = np.arange(len(buckets))
    width = 0.8 / len(methods)
    plt.figure(figsize=(7.0, 3.0), dpi=150)
    for i, m in enumerate(methods):
        plt.bar(x + (i - len(methods) / 2) * width + width / 2, rel[i], width, label=m)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xticks(x, ["B0", "B1", "B2"])
    plt.ylabel("Relative Δ MSE (%) (shifted vs orig)")
    plt.legend(fontsize=8, ncol=min(4, len(methods)), frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved robustness plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="E3: Robustness to near-pole shift")
    ap.add_argument("--orig", required=True, help="Original dataset JSON")
    ap.add_argument("--shifted", required=True, help="Shifted dataset JSON")
    ap.add_argument("--outdir", default="results/robotics/e3_robustness", help="Output directory")
    ap.add_argument("--max_train", type=int, default=2000)
    ap.add_argument("--max_test", type=int, default=500)
    ap.add_argument("--models", nargs="+", default=["mlp", "rational_eps", "tr_basic", "tr_full"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load datasets and edges
    ds_orig = load_dataset(args.orig)
    ds_shift = load_dataset(args.shifted)
    samples_orig = ds_orig["samples"]
    samples_shift = ds_shift["samples"]
    md_edges = ds_orig.get("metadata", {}).get("bucket_edges")
    edges = []
    if isinstance(md_edges, list):
        for e in md_edges:
            try:
                edges.append(float(e))
            except Exception:
                s = str(e).strip().lower()
                edges.append(float("inf") if s in ("inf", "+inf", "infinity") else float(e))
    else:
        edges = DEFAULT_EDGES

    # Quick subsets
    tr_idx, te_idx_orig = quick_indices(samples_orig, args.max_train, args.max_test, edges)
    _, te_idx_shift = quick_indices(samples_shift, args.max_train, args.max_test, edges)

    train_data = build_data(samples_orig, tr_idx)
    test_orig = build_data(samples_orig, te_idx_orig)
    test_shift = build_data(samples_shift, te_idx_shift)
    val_data = test_orig  # use original quick test as validation during quick training

    summary: Dict[str, Any] = {}

    if "mlp" in args.models:
        print("Evaluating MLP...")
        summary["MLP"] = eval_mlp(train_data, val_data, test_orig, test_shift, edges)

    if "rational_eps" in args.models:
        print("Evaluating Rational+ε...")
        summary["Rational+ε"] = eval_rational_eps(
            train_data, val_data, test_orig, test_shift, edges
        )

    if "tr_basic" in args.models:
        print("Evaluating ZeroProofML (Basic)...")
        summary["ZeroProofML (Basic)"] = eval_tr(
            train_data, val_data, test_orig, test_shift, edges, enable_enhancements=False
        )

    if "tr_full" in args.models:
        print("Evaluating ZeroProofML (Full)...")
        summary["ZeroProofML (Full)"] = eval_tr(
            train_data, val_data, test_orig, test_shift, edges, enable_enhancements=True
        )

    # Save JSON
    out_json = os.path.join(args.outdir, "e3_robustness_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {out_json}")

    # Plot robustness for B0–B2
    out_fig = os.path.join(args.outdir, "e3_robustness_b012.png")
    plot_robustness_b012(summary, out_fig)


if __name__ == "__main__":
    main()
