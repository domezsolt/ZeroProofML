#!/usr/bin/env python3
"""
Plot B0–B2 bar chart from the aggregated per-bucket CSV.

Reads the CSV produced by scripts/aggregate_buckets.py and renders a grouped
bar chart for buckets B0, B1, B2 across selected methods.

Usage (typical):
  python scripts/plot_b012_bars.py \
    --csv results/robotics/aggregated/buckets_table.csv \
    --out results/robotics/figures/b012_bars.png

Optional: choose methods and title
  python scripts/plot_b012_bars.py \
    --csv results/robotics/aggregated/buckets_table.csv \
    --methods "ZeroProofML (Basic)" "ZeroProofML (Full)" \
             "Rational+ε (ε=1e-2, no clip)" "Rational+ε (ε=1e-3, no clip)" "Rational+ε (ε=1e-4, no clip)" \
    --title "B0–B2 MSE by method" \
    --out results/robotics/figures/b012_bars.png
"""

import argparse
import csv
import os
from typing import Dict, List


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def default_method_order(rows: List[Dict[str, str]]) -> List[str]:
    """Produce a reasonable default ordering for methods if none provided."""
    names = [row["Method"] for row in rows]
    # Prefer TR variants first, then ε grid (no clip), then ε+clip
    priority = []
    for n in names:
        key = (0, n)
        low = n.lower()
        if "zeroproofml (full)" in low:
            key = (0, n)
        elif "zeroproofml (basic)" in low:
            key = (1, n)
        elif "no clip" in low:
            # sort by epsilon magnitude if present
            key = (2, n)
        elif "+clip" in low or "clip" in low:
            key = (3, n)
        priority.append((key, n))
    priority.sort(key=lambda x: x[0])
    ordered = [n for (_, n) in priority]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for n in ordered:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot B0–B2 bar chart from aggregated per-bucket CSV")
    ap.add_argument(
        "--csv", required=True, help="Path to aggregated CSV (from aggregate_buckets.py)"
    )
    ap.add_argument(
        "--out", default="results/robotics/figures/b012_bars.png", help="Output figure path"
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Explicit list of method names to include, in order",
    )
    ap.add_argument("--title", default="B0–B2 MSE by method", help="Title for the plot")
    ap.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale (default: linear)",
    )
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print(f"No rows found in {args.csv}")
        return

    # Determine method order
    methods = args.methods or default_method_order(rows)
    # Build a map for quick lookup
    row_by_method = {row["Method"]: row for row in rows}
    selected = [m for m in methods if m in row_by_method]
    if not selected:
        print("No matching methods found; check --methods names or CSV content.")
        return

    # Extract B0–B2 means
    buckets = ["B0", "B1", "B2"]
    data = []  # shape: len(selected) x len(buckets)
    for m in selected:
        row = row_by_method[m]
        vals = []
        for b in buckets:
            v = row.get(b)
            try:
                vals.append(float(v) if v not in (None, "", "None") else float("nan"))
            except Exception:
                vals.append(float("nan"))
        data.append(vals)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping plot.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_methods = len(selected)
    n_buckets = len(buckets)
    x = np.arange(n_buckets)
    width = 0.8 / max(1, n_methods)

    plt.figure(figsize=(7.0, 3.0), dpi=150)
    for i, (m, vals) in enumerate(zip(selected, data)):
        offset = (i - n_methods / 2) * width + width / 2
        plt.bar(x + offset, vals, width=width, label=m)
    plt.xticks(x, buckets)
    plt.ylabel("MSE")
    plt.yscale(args.yscale)
    plt.title(args.title)
    plt.legend(fontsize=8, ncol=min(3, n_methods), frameon=False)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"Saved B0–B2 bars to {args.out}")


if __name__ == "__main__":
    main()
