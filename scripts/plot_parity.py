#!/usr/bin/env python3
"""
Plot compact figures from aggregated parity JSON:

- Grouped per-bucket MSE bar chart with error bars
- PLE histogram over seeds (per method)

Usage:
  python scripts/plot_parity.py \
    --summary results/robotics/quick_aggregate.json \
    --outdir results/robotics/figures

Notes:
  - The summary JSON can be generated via:
      python scripts/aggregate_parity.py --json_out results/robotics/quick_aggregate.json
  - Requires matplotlib. If missing, install with: pip install matplotlib
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List


def _fmt_edge(v) -> str:
    try:
        x = float(v)
    except Exception:
        s = str(v).strip().lower()
        if s in ("inf", "+inf", "infinity"):
            x = float("inf")
        else:
            raise
    if math.isinf(x):
        return "inf"
    return f"{x:.0e}"


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def make_bucket_keys(edges: List[Any]) -> List[str]:
    keys = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        keys.append(f"({_fmt_edge(lo)},{_fmt_edge(hi)}]")
    return keys


def plot_per_bucket_bars(
    summary: Dict[str, Any], outdir: str, filename: str = "e1_per_bucket_bars.png"
):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping per-bucket bar plot.")
        return None

    methods = summary.get("methods", ["MLP", "Rational+ε", "ZeroProofML-Basic", "ZeroProofML-Full"])
    edges = summary.get("bucket_edges") or [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]
    bucket_keys = make_bucket_keys(edges)

    # Collect data
    means = []
    stds = []
    labels = []
    for m in methods:
        pb = summary.get("per_bucket_mse", {}).get(m) or {}
        m_means = [pb.get(k, {}).get("mean") for k in bucket_keys]
        m_stds = [pb.get(k, {}).get("std") for k in bucket_keys]
        # Skip methods with no bucket data
        if all(v is None for v in m_means):
            continue
        labels.append(m)
        means.append([0.0 if v is None else float(v) for v in m_means])
        stds.append([0.0 if v is None else float(v) for v in m_stds])

    if not means:
        print("No per-bucket data found in summary; skipping bar plot.")
        return None

    means = np.array(means)
    stds = np.array(stds)

    n_methods, n_buckets = means.shape
    x = np.arange(n_buckets)
    width = 0.8 / n_methods

    plt.figure(figsize=(7.0, 3.2), dpi=150)
    for i in range(n_methods):
        plt.bar(
            x + (i - n_methods / 2) * width + width / 2,
            means[i],
            width,
            yerr=stds[i],
            capsize=2,
            label=labels[i],
        )

    plt.xticks(x, bucket_keys, rotation=0)
    plt.ylabel("MSE (lower is better)")
    plt.xlabel("Buckets by |det(J)|")
    plt.legend(fontsize=8, ncol=min(n_methods, 4), frameon=False)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved per-bucket bar figure to {out_path}")
    return out_path


def plot_ple_hist(summary: Dict[str, Any], outdir: str, filename: str = "e1_ple_hist.png"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping PLE histogram.")
        return None

    methods = summary.get("methods", ["MLP", "Rational+ε", "ZeroProofML-Basic", "ZeroProofML-Full"])
    raw = summary.get("pole_metrics_raw", {})

    plt.figure(figsize=(5.8, 3.0), dpi=150)
    colors = ["#4e79a7", "#f28e2c", "#59a14f", "#e15759", "#b6992d"]
    plotted = 0
    for i, m in enumerate(methods):
        arr = raw.get(m, {}).get("ple")
        if not arr:
            continue
        vals = np.array(arr, dtype=float)
        bins = max(3, min(10, len(vals)))
        plt.hist(
            vals, bins=bins, alpha=0.45, label=m, color=colors[i % len(colors)], edgecolor="white"
        )
        plotted += 1

    if plotted == 0:
        print("No raw PLE arrays in summary; skipping PLE histogram.")
        return None

    plt.xlabel("PLE (lower is better)")
    plt.ylabel("Count")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PLE histogram to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Plot per-bucket bars and PLE hist from aggregated parity summary"
    )
    ap.add_argument(
        "--summary",
        required=True,
        help="Path to aggregated JSON (from aggregate_parity.py --json_out)",
    )
    ap.add_argument(
        "--outdir", default="results/robotics/figures", help="Directory to save figures"
    )
    args = ap.parse_args()

    summary = load_summary(args.summary)
    plot_per_bucket_bars(summary, args.outdir)
    plot_ple_hist(summary, args.outdir)


if __name__ == "__main__":
    main()
