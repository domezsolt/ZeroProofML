#!/usr/bin/env python3
"""
Compute sign consistency across θ2=0 crossings for E1 quick runs (5 seeds),
using saved predictions from each method. Produces a compact bar figure and
prints mean±std per method.

Usage:
  python scripts/eval_sign_consistency.py \
    --dataset data/rr_ik_dataset.json \
    --runs_glob 'results/robotics/quick_s*' \
    --outjson results/robotics/quick_sign_consistency.json \
    --outfig results/robotics/figures/e1_sign_consistency.png \
    --n_paths 12 --th1_tol 0.15 --th2_window 0.30 --min_dtheta2 1e-3

Notes:
  - Reconstructs the E1-quick test subset order (stratified by |sin θ2|) to
    align saved predictions with the dataset samples.
  - Methods: MLP, Rational+ε (best-ε baseline), ZeroProofML-Basic, ZeroProofML-Full.
  - Uses compute_sign_consistency_rate with a small magnitude filter on |Δθ2|
    to ignore near-zero noise near the crossing.
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Metric helper
from zeroproof.metrics.pole_2d import compute_sign_consistency_rate

DEFAULT_BUCKET_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]


def load_dataset(dataset_path: str) -> Dict:
    with open(dataset_path, "r") as fh:
        return json.load(fh)


def reconstruct_quick_test_indices(
    samples: List[Dict], max_test: int = 500, edges: Optional[List[float]] = None
) -> List[int]:
    """Match the quick profile selection used in examples/baselines/compare_all.py.

    - Base split: first 80% train, last 20% test by contiguous indexing.
    - Bucketize test indices by |sin(theta2)| into DEFAULT_BUCKET_EDGES.
    - Preselect one sample from B0–B3 (if available), then round-robin fill
      remaining slots across buckets in order.
    """
    edges = edges or DEFAULT_BUCKET_EDGES
    n_total = len(samples)
    n_train_full = int(0.8 * n_total)
    test_idx = list(range(n_train_full, n_total))

    # Bucketize by |sin(theta2)|
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
    selected = []
    # Preselect one from B0–B3
    for b in range(min(4, len(edges) - 1)):
        if tb.get(b):
            selected.append(tb[b][0])
            tb[b] = tb[b][1:]
    # Round-robin fill
    rr_order = [b for b in range(len(edges) - 1) if tb.get(b)]
    ptrs = {b: 0 for b in rr_order}
    while len(selected) < min(max_test, len(test_idx)) and rr_order:
        new_rr = []
        for b in rr_order:
            blist = tb.get(b, [])
            p = ptrs[b]
            if p < len(blist):
                selected.append(blist[p])
                ptrs[b] = p + 1
                if ptrs[b] < len(blist):
                    new_rr.append(b)
            if len(selected) >= min(max_test, len(test_idx)):
                break
        rr_order = new_rr
        if not rr_order:
            break
    # Fallback: append remaining in order
    if len(selected) < min(max_test, len(test_idx)):
        remaining = [i for i in test_idx if i not in selected]
        need = min(max_test, len(test_idx)) - len(selected)
        selected.extend(remaining[:need])
    return selected


def load_predictions(run_dir: str) -> Dict[str, List[List[float]]]:
    """Load predictions for each method from a quick run directory."""
    preds: Dict[str, List[List[float]]] = {}

    # MLP
    mlp_file = os.path.join(run_dir, "mlp", "mlp_baseline_results.json")
    try:
        with open(mlp_file, "r") as fh:
            data = json.load(fh)
        preds["MLP"] = data.get("test_metrics", {}).get("predictions", [])
    except Exception:
        pass

    # Rational+ε (use chosen ε baseline)
    rat_dir = os.path.join(run_dir, "rational_eps")
    try:
        # Prefer baseline file; fallback to any *_baseline_eps_*.json
        candidates = glob.glob(os.path.join(rat_dir, "rational_eps_baseline_eps_*.json"))
        if candidates:
            rat_file = sorted(candidates)[0]
            with open(rat_file, "r") as fh:
                data = json.load(fh)
            preds["Rational+ε"] = data.get("test_metrics", {}).get("predictions", [])
    except Exception:
        pass

    # ZeroProofML Basic
    trb_file = os.path.join(run_dir, "zeroproof_basic", "tr_rational_basic_results.json")
    try:
        with open(trb_file, "r") as fh:
            data = json.load(fh)
        preds["ZeroProofML-Basic"] = data.get("predictions", []) or data.get(
            "test_metrics", {}
        ).get("predictions", [])
    except Exception:
        pass

    # ZeroProofML Full
    trf_file = os.path.join(run_dir, "zeroproof_full", "zeroproof_full_results.json")
    try:
        with open(trf_file, "r") as fh:
            data = json.load(fh)
        preds["ZeroProofML-Full"] = data.get("predictions", []) or data.get("test_metrics", {}).get(
            "predictions", []
        )
    except Exception:
        pass

    return preds


def apply_min_dtheta2_mask(preds: List[List[float]], tau: float) -> List[List[float]]:
    out = []
    for p in preds:
        if not p or len(p) < 2:
            out.append(p)
            continue
        d1, d2 = float(p[0]), float(p[1])
        if abs(d2) <= tau:
            out.append([d1, 0.0])
        else:
            out.append([d1, d2])
    return out


def aggregate_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate sign consistency across θ2=0 for E1 quick runs"
    )
    ap.add_argument("--dataset", default="data/rr_ik_dataset.json", help="Path to dataset JSON")
    ap.add_argument(
        "--runs_glob", default="results/robotics/quick_s*", help="Glob for quick run directories"
    )
    ap.add_argument("--outjson", default=None, help="Optional path to write aggregated JSON")
    ap.add_argument(
        "--outfig",
        default="results/robotics/figures/e1_sign_consistency.png",
        help="Output figure path",
    )
    ap.add_argument("--n_paths", type=int, default=12)
    ap.add_argument("--th1_tol", type=float, default=0.15)
    ap.add_argument("--th2_window", type=float, default=0.30)
    ap.add_argument("--min_dtheta2", type=float, default=1e-3)
    args = ap.parse_args()

    # Load dataset and reconstruct quick test inputs
    dset = load_dataset(args.dataset)
    samples = dset.get("samples", [])
    if not samples:
        raise SystemExit("Dataset missing or empty")
    edges = None
    md = dset.get("metadata", {})
    if isinstance(md, dict) and "bucket_edges" in md:
        try:
            edges = [
                float(e)
                if not (isinstance(e, str) and e.lower() in ("inf", "+inf", "infinity"))
                else float("inf")
                for e in md["bucket_edges"]
            ]
        except Exception:
            edges = None
    selected = reconstruct_quick_test_indices(samples, max_test=500, edges=edges)
    test_inputs = [
        [samples[i]["theta1"], samples[i]["theta2"], samples[i]["dx"], samples[i]["dy"]]
        for i in selected
    ]

    # Scan runs and compute per-run sign consistency
    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise SystemExit(f"No runs found matching {args.runs_glob}")

    per_method: Dict[str, List[float]] = {
        "MLP": [],
        "Rational+ε": [],
        "ZeroProofML-Basic": [],
        "ZeroProofML-Full": [],
    }

    for rd in run_dirs:
        preds_map = load_predictions(rd)
        for method, preds in preds_map.items():
            if method not in per_method:
                continue
            # Align length
            n = min(len(preds), len(test_inputs))
            if n == 0:
                continue
            preds_f = apply_min_dtheta2_mask(preds[:n], args.min_dtheta2)
            score = compute_sign_consistency_rate(
                test_inputs[:n],
                preds_f,
                n_paths=args.n_paths,
                th1_tol=args.th1_tol,
                th2_window=args.th2_window,
            )
            per_method[method].append(score)

    # Aggregate and print
    summary = {
        "methods": {},
        "params": {
            "n_paths": args.n_paths,
            "th1_tol": args.th1_tol,
            "th2_window": args.th2_window,
            "min_dtheta2": args.min_dtheta2,
        },
    }
    print("Sign consistency across θ2=0 (mean±std over runs):")
    for m, xs in per_method.items():
        if not xs:
            continue
        mu, sd = aggregate_mean_std(xs)
        print(f"  {m:18s}: {mu*100:.2f}% ± {sd*100:.2f}% (n={len(xs)})")
        summary["methods"][m] = {"mean": mu, "std": sd, "n": len(xs), "values": xs}

    # Save JSON if requested
    if args.outjson:
        os.makedirs(os.path.dirname(args.outjson), exist_ok=True)
        with open(args.outjson, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved summary to {args.outjson}")

    # Plot bar figure
    try:
        import matplotlib.pyplot as plt

        methods = [m for m in per_method.keys() if per_method[m]]
        if methods:
            means = [summary["methods"][m]["mean"] * 100.0 for m in methods]
            stds = [summary["methods"][m]["std"] * 100.0 for m in methods]
            plt.figure(figsize=(5.5, 3.0), dpi=150)
            x = np.arange(len(methods))
            plt.bar(methods, means, yerr=stds, capsize=3)
            plt.ylabel("Sign Consistency (%)")
            plt.ylim(0, 100)
            plt.tight_layout()
            os.makedirs(os.path.dirname(args.outfig), exist_ok=True)
            plt.savefig(args.outfig)
            plt.close()
            print(f"Saved sign consistency figure to {args.outfig}")
    except Exception as e:
        print(f"Figure not generated ({e})")


if __name__ == "__main__":
    main()
