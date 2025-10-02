#!/usr/bin/env python3
"""
Aggregate across-seed TR 6R results (ik6r_results.json) into a CSV and optional LaTeX.

Usage:
  python scripts/aggregate_ik6r_seeds.py \
    --glob 'results/robotics/ik6r_s*/ik6r_results.json' \
    --out results/robotics/ik6r_agg/summary.csv \
    --latex results/robotics/ik6r_agg/summary.tex
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Tuple


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def agg_mean_std(xs: List[float]) -> Tuple[float, float, int]:
    vals = [safe_float(x) for x in xs if isinstance(x, (int, float)) or (isinstance(x, str) and x)]
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    return (mu, math.sqrt(var), n)


def collect(paths: List[str]) -> Tuple[Dict[str, Tuple[float, float, int]], List[str]]:
    overall: List[float] = []
    per_bucket: Dict[str, List[float]] = {}
    all_keys: List[str] = []
    for p in sorted(paths):
        try:
            with open(p, "r") as fh:
                res = json.load(fh)
        except Exception:
            continue
        overall.append(safe_float(res.get("test_mse_mean")))
        pb = res.get("per_bucket", {})
        for k, v in pb.items():
            all_keys.append(k)
            per_bucket.setdefault(k, []).append(safe_float(v.get("mean_mse")))

    # stable order of keys by numeric edges if possible
    def _key_order(k: str) -> Tuple[int, str]:
        # Expect like '(0e+00,1e-05]'
        return (0, k)

    keys = sorted(set(all_keys), key=_key_order)
    out: Dict[str, Tuple[float, float, int]] = {}
    out["overall_mse"] = agg_mean_std(overall)
    for k in keys:
        out[f"bucket_{k}"] = agg_mean_std(per_bucket.get(k, []))
    return out, keys


def write_csv(path: str, agg: Dict[str, Tuple[float, float, int]], keys: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("Metric,mean,std,n\n")
        mu, sd, n = agg.get("overall_mse", (float("nan"), float("nan"), 0))
        fh.write(f"overall_mse,{mu},{sd},{n}\n")
        for k in keys:
            m, s, nn = agg.get(f"bucket_{k}", (float("nan"), float("nan"), 0))
            fh.write(f"{k},{m},{s},{nn}\n")


def write_latex(path: str, agg: Dict[str, Tuple[float, float, int]], keys: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{l" + "c" * (min(5, len(keys)) + 1) + "}")
    lines.append("    \\toprule")
    header = ["Metric", "Overall"] + [keys[i] for i in range(min(5, len(keys)))]
    lines.append("    " + " & ".join(header) + " \\\\")
    lines.append("    \\midrule")
    mu, sd, n = agg.get("overall_mse", (float("nan"), float("nan"), 0))
    row = ["TR 6R", f"{mu:.6f} $\\pm$ {sd:.6f} (n={n})"]
    for i in range(min(5, len(keys))):
        m, s, nn = agg.get(f"bucket_{keys[i]}", (float("nan"), float("nan"), 0))
        row.append(f"{m:.6f} $\\pm$ {s:.6f} (n={nn})" if nn else "-")
    lines.append("    " + " & ".join(row) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(
        "  \\caption{Across-seed mean$\\pm$std for overall and per-bin (d1) MSE on 6R synthetic dataset.}"
    )
    lines.append("  \\label{tab:ik6r_agg}")
    lines.append("\\end{table}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate TR 6R results across seeds")
    ap.add_argument("--glob", default="results/robotics/ik6r_s*/ik6r_results.json")
    ap.add_argument("--out", default="results/robotics/ik6r_agg/summary.csv")
    ap.add_argument("--latex", default=None)
    args = ap.parse_args()
    paths = glob.glob(args.glob)
    if not paths:
        print(f"No files matched: {args.glob}")
        return
    agg, keys = collect(paths)
    write_csv(args.out, agg, keys)
    print(f"Saved 6R aggregate CSV to {args.out}")
    if args.latex:
        write_latex(args.latex, agg, keys)
        print(f"Saved 6R aggregate LaTeX to {args.latex}")


if __name__ == "__main__":
    main()
