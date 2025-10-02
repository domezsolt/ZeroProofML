#!/usr/bin/env python3
"""
Crossing-sweep sign consistency without retraining.

Builds a mini-set from the quick test subset by:
 - fixing a displacement direction angle phi (degrees) with tolerance
 - sweeping near-crossing points |theta2| <= window for multiple theta1 anchors
 - pairing closest negative/positive theta2 samples and counting sign flips

Uses saved predictions from quick runs (results/robotics/quick_s*/...).

Outputs:
 - JSON summary with mean±std sign consistency and contributing pair counts
 - Bar plot figure with error bars and pair-count annotations

Usage example:
  python scripts/crossing_sweep_eval.py \
    --dataset data/rr_ik_dataset.json \
    --runs_glob 'results/robotics/quick_s*' \
    --outjson results/robotics/quick_sign_consistency_sweep.json \
    --outfig results/robotics/figures/e1_sign_consistency_sweep.png \
    --phi_deg 0 --phi_tol_deg 15 --n_paths 12 --th1_tol 0.15 --th2_window 0.30 \
    --k_pairs 4 --min_dtheta2 1e-3
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_BUCKET_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]


def load_dataset(dataset_path: str) -> Dict:
    with open(dataset_path, "r") as fh:
        return json.load(fh)


def reconstruct_quick_test_indices(
    samples: List[Dict], max_test: int = 500, edges: Optional[List[float]] = None
) -> List[int]:
    edges = edges or DEFAULT_BUCKET_EDGES
    n_total = len(samples)
    n_train_full = int(0.8 * n_total)
    test_idx = list(range(n_train_full, n_total))

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
    for b in range(min(4, len(edges) - 1)):
        if tb.get(b):
            selected.append(tb[b][0])
            tb[b] = tb[b][1:]
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
    if len(selected) < min(max_test, len(test_idx)):
        remaining = [i for i in test_idx if i not in selected]
        selected.extend(remaining[: (min(max_test, len(test_idx)) - len(selected))])
    return selected


def load_predictions(run_dir: str) -> Dict[str, List[List[float]]]:
    preds: Dict[str, List[List[float]]] = {}
    try:
        with open(os.path.join(run_dir, "mlp", "mlp_baseline_results.json")) as fh:
            preds["MLP"] = json.load(fh).get("test_metrics", {}).get("predictions", [])
    except Exception:
        pass
    rat_dir = os.path.join(run_dir, "rational_eps")
    try:
        candidates = glob.glob(os.path.join(rat_dir, "rational_eps_baseline_eps_*.json"))
        if candidates:
            with open(sorted(candidates)[0]) as fh:
                preds["Rational+ε"] = json.load(fh).get("test_metrics", {}).get("predictions", [])
    except Exception:
        pass
    try:
        with open(os.path.join(run_dir, "zeroproof_basic", "tr_rational_basic_results.json")) as fh:
            data = json.load(fh)
        preds["ZeroProofML-Basic"] = data.get("predictions", []) or data.get(
            "test_metrics", {}
        ).get("predictions", [])
    except Exception:
        pass
    try:
        with open(os.path.join(run_dir, "zeroproof_full", "zeroproof_full_results.json")) as fh:
            data = json.load(fh)
        preds["ZeroProofML-Full"] = data.get("predictions", []) or data.get("test_metrics", {}).get(
            "predictions", []
        )
    except Exception:
        pass
    return preds


def paired_consistency_with_direction(
    test_inputs, preds, n_paths, th1_tol, th2_window, k_pairs, min_dtheta2, phi_deg, phi_tol_deg
) -> Tuple[float, int]:
    if not test_inputs or not preds:
        return 0.0, 0
    n = min(len(test_inputs), len(preds))
    th1 = np.array([float(x[0]) for x in test_inputs[:n]])
    th2 = np.array([float(x[1]) for x in test_inputs[:n]])
    dx = np.array([float(x[2]) for x in test_inputs[:n]])
    dy = np.array([float(x[3]) for x in test_inputs[:n]])
    ang = np.degrees(np.arctan2(dy, dx))
    d2 = np.array([float(p[1]) if len(p) > 1 else 0.0 for p in preds[:n]])
    anchors = np.linspace(th1.min(), th1.max(), num=n_paths)
    total_pairs = 0
    flips = 0

    # Normalize angles to [-180,180]
    def ang_diff(a, b):
        d = a - b
        d = (d + 180.0) % 360.0 - 180.0
        return np.abs(d)

    for a in anchors:
        mask = (
            (np.abs(th1 - a) <= th1_tol)
            & (np.abs(th2) <= th2_window)
            & (ang_diff(ang, phi_deg) <= phi_tol_deg)
        )
        idx = np.where(mask)[0]
        if idx.size < 2:
            continue
        neg = idx[th2[idx] < 0.0]
        pos = idx[th2[idx] > 0.0]
        if neg.size == 0 or pos.size == 0:
            continue
        neg_sorted = neg[np.argsort(np.abs(th2[neg]))][:k_pairs]
        pos_sorted = pos[np.argsort(np.abs(th2[pos]))][:k_pairs]
        K = int(min(len(neg_sorted), len(pos_sorted)))
        for i in range(K):
            b_idx = neg_sorted[i]
            a_idx = pos_sorted[i]
            b_sign = np.sign(d2[b_idx]) if abs(d2[b_idx]) > min_dtheta2 else 0.0
            a_sign = np.sign(d2[a_idx]) if abs(d2[a_idx]) > min_dtheta2 else 0.0
            if b_sign == 0.0 or a_sign == 0.0:
                continue
            total_pairs += 1
            if b_sign != a_sign:
                flips += 1
    return (float(flips / total_pairs) if total_pairs > 0 else 0.0, int(total_pairs))


def main():
    ap = argparse.ArgumentParser(
        description="Direction-fixed crossing sweep consistency (no retraining)"
    )
    ap.add_argument("--dataset", default="data/rr_ik_dataset.json")
    ap.add_argument("--runs_glob", default="results/robotics/quick_s*")
    ap.add_argument("--outjson", default="results/robotics/quick_sign_consistency_sweep.json")
    ap.add_argument("--outfig", default="results/robotics/figures/e1_sign_consistency_sweep.png")
    ap.add_argument(
        "--phi_deg", type=float, default=0.0, help="Desired displacement direction angle (deg)"
    )
    ap.add_argument("--phi_tol_deg", type=float, default=15.0, help="Angle tolerance (deg)")
    ap.add_argument("--n_paths", type=int, default=12)
    ap.add_argument("--th1_tol", type=float, default=0.15)
    ap.add_argument("--th2_window", type=float, default=0.30)
    ap.add_argument("--k_pairs", type=int, default=4)
    ap.add_argument("--min_dtheta2", type=float, default=1e-3)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    samples = ds.get("samples", [])
    if not samples:
        raise SystemExit("Dataset missing or empty")
    edges = None
    md = ds.get("metadata", {})
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

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise SystemExit(f"No runs found matching {args.runs_glob}")

    per_method_scores: Dict[str, List[float]] = {
        "MLP": [],
        "Rational+ε": [],
        "ZeroProofML-Basic": [],
        "ZeroProofML-Full": [],
    }
    per_method_pairs: Dict[str, List[int]] = {k: [] for k in per_method_scores.keys()}

    for rd in run_dirs:
        preds_map = load_predictions(rd)
        for method, preds in preds_map.items():
            if method not in per_method_scores:
                continue
            score, count = paired_consistency_with_direction(
                test_inputs,
                preds,
                args.n_paths,
                args.th1_tol,
                args.th2_window,
                args.k_pairs,
                args.min_dtheta2,
                args.phi_deg,
                args.phi_tol_deg,
            )
            per_method_scores[method].append(score)
            per_method_pairs[method].append(count)

    summary = {"methods": {}, "params": vars(args)}
    print("Direction-fixed paired consistency (mean±std over runs):")
    for m, xs in per_method_scores.items():
        if not xs:
            continue
        arr = np.array(xs, dtype=float)
        mu, sd = float(arr.mean()), float(arr.std())
        c_arr = np.array(per_method_pairs[m], dtype=float)
        summary["methods"][m] = {
            "mean": mu,
            "std": sd,
            "n": len(xs),
            "values": per_method_scores[m],
            "pairs_mean": float(c_arr.mean()) if c_arr.size else 0.0,
            "pairs_min": int(c_arr.min()) if c_arr.size else 0,
            "pairs_max": int(c_arr.max()) if c_arr.size else 0,
        }
        print(
            f"  {m:18s}: {mu*100:.2f}% ± {sd*100:.2f}% (pairs~{summary['methods'][m]['pairs_mean']:.1f})"
        )

    os.makedirs(os.path.dirname(args.outjson), exist_ok=True)
    with open(args.outjson, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {args.outjson}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        methods = [m for m in per_method_scores.keys() if per_method_scores[m]]
        if methods:
            means = [summary["methods"][m]["mean"] * 100.0 for m in methods]
            stds = [summary["methods"][m]["std"] * 100.0 for m in methods]
            plt.figure(figsize=(6.2, 3.2), dpi=150)
            bars = plt.bar(methods, means, yerr=stds, capsize=3)
            plt.ylabel("Sign Consistency (%)")
            plt.ylim(0, 100)
            for i, m in enumerate(methods):
                pairs = summary["methods"][m].get("pairs_mean", 0.0)
                plt.text(
                    i,
                    means[i] + max(1.0, stds[i]) + 1.0,
                    f"pairs~{pairs:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            plt.tight_layout()
            os.makedirs(os.path.dirname(args.outfig), exist_ok=True)
            plt.savefig(args.outfig)
            plt.close()
            print(f"Saved figure to {args.outfig}")
    except Exception as e:
        print(f"Figure not generated ({e})")


if __name__ == "__main__":
    main()
