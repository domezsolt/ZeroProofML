#!/usr/bin/env python3
"""
Plot E2 ablation figures:
 - Bucket-wise MSE bars (Hybrid vs Mask-REAL)
 - PLE trajectory over epochs
 - Bench avg_step_ms bars

Usage:
  python scripts/plot_ablation.py \
    --mask_dir results/robotics/ablation_mask_real \
    --hybrid_dir results/robotics/ablation_hybrid \
    --outdir results/robotics/figures

Inputs expected:
  <dir>/bucket_metrics.json
  <dir>/results_tr_rat.json
"""

import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def plot_bucket_bars(
    mask_dir: str, hybrid_dir: str, outdir: str, fname: str = "e2_ablation_buckets.png"
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping bucket bar plot.")
        return ""
    bm_mask = load_json(os.path.join(mask_dir, "bucket_metrics.json"))
    bm_hyb = load_json(os.path.join(hybrid_dir, "bucket_metrics.json"))
    # stable order by edges
    edges = bm_mask.get("bucket_edges") or bm_hyb.get("bucket_edges")

    def fmt_edge(x):
        try:
            v = float(x)
            return "inf" if (v == float("inf")) else f"{v:.0e}"
        except Exception:
            s = str(x).strip().lower()
            return "inf" if s in ("inf", "+inf", "infinity") else s

    keys = [f"({fmt_edge(edges[i])},{fmt_edge(edges[i+1])}]" for i in range(len(edges) - 1)]
    mvals = [bm_mask["per_bucket"].get(k, {}).get("mean_mse", 0.0) for k in keys]
    hvals = [bm_hyb["per_bucket"].get(k, {}).get("mean_mse", 0.0) for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    plt.figure(figsize=(7.0, 3.0), dpi=150)
    plt.bar(x - w / 2, mvals, width=w, label="Mask-REAL")
    plt.bar(x + w / 2, hvals, width=w, label="Hybrid")
    plt.xticks(x, keys)
    plt.ylabel("MSE")
    plt.xlabel("Buckets by |det(J)|")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved ablation bucket bars to {out_path}")
    return out_path


def plot_ple_trajectory(
    mask_dir: str, hybrid_dir: str, outdir: str, fname: str = "e2_ple_history.png"
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping PLE trajectory plot.")
        return ""
    rt_mask = load_json(os.path.join(mask_dir, "results_tr_rat.json"))
    rt_hyb = load_json(os.path.join(hybrid_dir, "results_tr_rat.json"))
    ph_mask: List[float] = rt_mask.get("training_summary", {}).get("ple_history", [])
    ph_hyb: List[float] = rt_hyb.get("training_summary", {}).get("ple_history", [])
    if not ph_mask and not ph_hyb:
        print("No PLE histories found; skipping plot.")
        return ""
    plt.figure(figsize=(6.0, 3.0), dpi=150)
    if ph_mask:
        plt.plot(range(len(ph_mask)), ph_mask, label="Mask-REAL")
    if ph_hyb:
        plt.plot(range(len(ph_hyb)), ph_hyb, label="Hybrid")
    plt.xlabel("Epoch")
    plt.ylabel("PLE (lower is better)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PLE trajectory to {out_path}")
    return out_path


def plot_bench(mask_dir: str, hybrid_dir: str, outdir: str, fname: str = "e2_bench.png") -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping bench plot.")
        return ""
    rt_mask = load_json(os.path.join(mask_dir, "results_tr_rat.json"))
    rt_hyb = load_json(os.path.join(hybrid_dir, "results_tr_rat.json"))
    bh_mask = rt_mask.get("training_summary", {}).get("bench_history", [])
    bh_hyb = rt_hyb.get("training_summary", {}).get("bench_history", [])

    def avg_step_ms(bh):
        if not bh:
            return 0.0
        vals = [float(x.get("avg_step_ms", 0.0)) for x in bh]
        return sum(vals) / len(vals) if vals else 0.0

    bars = {"Mask-REAL": avg_step_ms(bh_mask), "Hybrid": avg_step_ms(bh_hyb)}
    plt.figure(figsize=(4.0, 3.0), dpi=150)
    plt.bar(list(bars.keys()), list(bars.values()))
    plt.ylabel("avg_step_ms (per epoch)")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved bench bars to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot ablation figures for E2")
    ap.add_argument("--mask_dir", required=True, help="Path to Mask-REAL ablation dir")
    ap.add_argument("--hybrid_dir", required=True, help="Path to Hybrid ablation dir")
    ap.add_argument(
        "--outdir", default="results/robotics/figures", help="Output directory for figures"
    )
    args = ap.parse_args()

    plot_bucket_bars(args.mask_dir, args.hybrid_dir, args.outdir)
    plot_ple_trajectory(args.mask_dir, args.hybrid_dir, args.outdir)
    plot_bench(args.mask_dir, args.hybrid_dir, args.outdir)


if __name__ == "__main__":
    main()
