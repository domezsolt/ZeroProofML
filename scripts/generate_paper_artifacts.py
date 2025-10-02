#!/usr/bin/env python3
"""
Generate LaTeX-ready artifacts (tables and figure includes) from the paper_suite outputs.

Reads per-seed comprehensive JSONs and aggregated CSVs under an output root
and writes compact LaTeX tables for inclusion in paper.tex and theory appendix.

Examples:
  python scripts/generate_paper_artifacts.py \
    --root results/robotics/paper_suite \
    --methods "ZeroProofML-Full" "Rational+ε" "Smooth" "Learnable-ε" "EpsEnsemble" "MLP" \
    --with-rollout --with-3r --with-6r

Outputs (under <root>/latex/ by default):
  - overall_table.tex        (overall MSE mean±std across seeds)
  - near_pole_table.tex      (B0–B3 mean±std across seeds)
  - rollout_table.tex        (closed-loop summary)            [optional]
  - e3r_table.tex            (3R TR-only metrics)             [optional]
  - ik6r_table.tex           (6R seed aggregate)              [optional]
  - includes.tex             (small \\input scaffold)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple


def _load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _mean_std(xs: List[float]) -> Tuple[float, float, int]:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    return mu, math.sqrt(var), n


def _norm_key(s: str) -> str:
    """Normalize method labels to improve matching across variants.

    Removes spaces, punctuation and lowercases; keeps only alphanumerics.
    E.g., 'ZeroProofML (Full)' == 'ZeroProofML-Full' == 'zeroproofmlfull'.
    """
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_key(requested: str, available: List[str]) -> str | None:
    """Find best matching key from available for the requested label using normalization."""
    target = _norm_key(requested)
    for k in available:
        if _norm_key(k) == target:
            return k
    return None


def collect_overall(root: str, methods: List[str]) -> Dict[str, Tuple[float, float, int]]:
    pattern = os.path.join(root, "seed_*", "comprehensive_comparison.json")
    files = sorted(glob.glob(pattern))
    per_method: Dict[str, List[float]] = {m: [] for m in methods}
    for p in files:
        d = _load_json(p)
        table = d.get("comparison_table", [])
        # Build name->Test_MSE map
        m2 = {}
        for row in table:
            name = str(row.get("Method"))
            val = row.get("Test_MSE")
            if isinstance(val, (int, float)):
                m2[name] = float(val)
        for m in methods:
            key = _find_key(m, list(m2.keys())) or (m if m in m2 else None)
            if key is not None:
                per_method[m].append(m2[key])
    return {m: _mean_std(per_method[m]) for m in methods}


def collect_bins(
    root: str, methods: List[str]
) -> Tuple[List[str], Dict[str, Dict[str, Tuple[float, float, int]]]]:
    pattern = os.path.join(root, "seed_*", "comprehensive_comparison.json")
    files = sorted(glob.glob(pattern))
    # Determine bucket keys order from first file
    bucket_keys: List[str] = []
    per_method_bins: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    for p in files:
        d = _load_json(p)
        indiv = d.get("individual_results", {})
        if not bucket_keys:
            # Take from any available method
            for v in indiv.values():
                nb = v.get("near_pole_bucket_mse")
                if isinstance(nb, dict):
                    bm = nb.get("bucket_mse", {})
                    if isinstance(bm, dict) and bm:
                        bucket_keys = list(bm.keys())
                        break
        # Fill per method
        for m in methods:
            # match individual_results key by normalized label
            key = _find_key(m, list(indiv.keys())) or m
            v = indiv.get(key)
            if not isinstance(v, dict):
                continue
            nb = v.get("near_pole_bucket_mse", {})
            bm = nb.get("bucket_mse", {})
            if not isinstance(bm, dict):
                continue
            for k in bucket_keys:
                if k not in per_method_bins[m]:
                    per_method_bins[m][k] = []
                val = bm.get(k)
                if isinstance(val, (int, float)):
                    per_method_bins[m][k].append(float(val))
    # Reduce
    reduced: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for m in methods:
        reduced[m] = {}
        for k in bucket_keys:
            xs = per_method_bins[m].get(k, [])
            reduced[m][k] = _mean_std(xs)
    return bucket_keys, reduced


def write_overall_table(path: str, stats: Dict[str, Tuple[float, float, int]]) -> None:
    def _fmt_method(name: str) -> str:
        # Minimal LaTeX-friendly method names
        return name.replace("ε", "$\\varepsilon$")

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{lc}")
    lines.append("    \\toprule")
    lines.append("    Method & Overall MSE \\\\")
    lines.append("    \\midrule")
    for m, (mu, sd, n) in stats.items():
        if n:
            lines.append(f"    {_fmt_method(m)} & {mu:.6f} $\\pm$ {sd:.6f} (n={n}) \\\\")
        else:
            lines.append(f"    {_fmt_method(m)} & - \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Across-seed overall MSE (lower is better).}")
    lines.append("  \\label{tab:overall_mse}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_bins_table(
    path: str,
    keys: List[str],
    stats: Dict[str, Dict[str, Tuple[float, float, int]]],
    methods: List[str],
) -> None:
    def _fmt_method(name: str) -> str:
        return name.replace("ε", "$\\varepsilon$")

    # Show B0–B2 columns for compactness
    shown = keys[:3] if len(keys) >= 3 else keys
    hdr = " & ".join(["Method"] + shown)
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{l" + "c" * len(shown) + "}")
    lines.append("    \\toprule")
    lines.append(f"    {hdr} \\\\")
    lines.append("    \\midrule")
    for m in methods:
        row = [m]
        for k in shown:
            mu, sd, n = stats.get(m, {}).get(k, (float("nan"), float("nan"), 0))
            if n:
                row.append(f"{mu:.6f} $\\pm$ {sd:.6f} (n={n})")
            else:
                row.append("-")
        lines.append("    " + " & ".join([_fmt_method(row[0])] + row[1:]) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Near-pole per-bucket MSE (B0–B2), mean$\\pm$std across seeds.}")
    lines.append("  \\label{tab:near_pole_bins}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_rollout_table(path: str, rollout_json: str) -> None:
    if not os.path.exists(rollout_json):
        return
    d = _load_json(rollout_json)

    def fmt(m):
        if m not in d:
            return None
        x = d[m]
        return (x.get("mean_tracking_error"), x.get("max_joint_step"), x.get("failure_rate"))

    def _fmt_method(name: str) -> str:
        return name.replace("ε", "$\\varepsilon$")

    methods = ["MLP", "Rational+ε", "ZeroProofML-Basic", "ZeroProofML-Full"]
    rows = [(m, fmt(m)) for m in methods if fmt(m) is not None]
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{lccc}")
    lines.append("    \\toprule")
    lines.append("    Method & Mean Track Err & Max $\\|\\Delta\\theta\\|$ & Failure \\\\")
    lines.append("    \\midrule")
    for m, vals in rows:
        er, mx, fr = vals
        lines.append(f"    {_fmt_method(m)} & {er:.4f} & {mx:.4f} & {fr:.2f}\\% \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Closed-loop tracking near poles (lower is better).}")
    lines.append("  \\label{tab:rollout}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_e3r_table(path: str, e3r_json: str) -> None:
    if not os.path.exists(e3r_json):
        return
    d = _load_json(e3r_json)
    pole = d.get("pole_metrics_3r", {})
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\toprule")
    lines.append("    Metric & Value & Notes \\\\")
    lines.append("    \\midrule")
    lines.append(f'    Test MSE (mean) & {d.get("test_mse_mean"):.6f} & TR-Full (3R) \\\\')
    if "ple" in pole:
        lines.append(f'    PLE (rad) & {pole.get("ple"):.6f} & \\\\')
    if "sign_consistency_theta2" in pole:
        lines.append(
            f'    Sign consistency ($\\theta_2$) & {pole.get("sign_consistency_theta2"):.3f} & fraction of anchors \\\\'
        )
    if "sign_consistency_theta3" in pole:
        lines.append(
            f'    Sign consistency ($\\theta_3$) & {pole.get("sign_consistency_theta3"):.3f} & fraction of anchors \\\\'
        )
    if "residual_consistency" in pole:
        lines.append(
            f'    Residual consistency & {pole.get("residual_consistency"):.6f} & FK error \\\\'
        )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{3R near-pole metrics (TR-Full).}")
    lines.append("  \\label{tab:e3r}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_ik6r_table(path: str, ik6r_csv: str) -> None:
    if not os.path.exists(ik6r_csv):
        return
    # Simple passthrough with minimal formatting note
    with open(ik6r_csv, "r") as fh:
        header = fh.readline().strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]
    # Build LaTeX table with a few key rows
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering\\small")
    lines.append("  \\begin{tabular}{lcccc}")
    lines.append("    \\toprule")
    lines.append("    Metric & Mean & Std & n & Notes \\\\")
    lines.append("    \\midrule")
    # Expect first data row to be 'overall_mse'
    for r in rows:
        if len(r) < 4:
            continue
        name, mu, sd, n = r[0], r[1], r[2], r[3]
        if name == "overall_mse":
            lines.append(
                f"    Overall MSE & {float(mu):.6f} & {float(sd):.6f} & {int(float(n))} & 6R TR \\\\"
            )
        # Show a couple of bins
        if name in ("(0e+00,1e-05]", "(1e-05,1e-04]", "(1e-04,1e-03]"):
            lines.append(
                f"    {name} & {float(mu):.6f} & {float(sd):.6f} & {int(float(n))} & 6R bins \\\\"
            )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{6R synthetic TR results (overall and selected bins).}")
    lines.append("  \\label{tab:ik6r}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LaTeX artifacts from paper_suite outputs")
    ap.add_argument(
        "--root", required=True, help="Output root (e.g., results/robotics/paper_suite)"
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=["ZeroProofML-Full", "Rational+ε", "Smooth", "Learnable-ε", "EpsEnsemble", "MLP"],
    )
    ap.add_argument(
        "--latex-dir", default="latex", help="Subdirectory under root to write LaTeX files"
    )
    ap.add_argument("--with-rollout", action="store_true")
    ap.add_argument("--with-3r", action="store_true")
    ap.add_argument("--with-6r", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.join(root, args.latex_dir)
    os.makedirs(outdir, exist_ok=True)

    # Overall
    overall = collect_overall(root, args.methods)
    write_overall_table(os.path.join(outdir, "overall_table.tex"), overall)

    # Near-pole bins
    keys, bins = collect_bins(root, args.methods)
    write_bins_table(os.path.join(outdir, "near_pole_table.tex"), keys, bins, args.methods)

    # Rollout
    if args.with_rollout:
        write_rollout_table(
            os.path.join(outdir, "rollout_table.tex"), os.path.join(root, "rollout_summary.json")
        )

    # 3R
    if args.with_3r:
        write_e3r_table(
            os.path.join(outdir, "e3r_table.tex"), os.path.join(root, "e3r", "e3r_results.json")
        )

    # 6R
    if args.with_6r:
        write_ik6r_table(
            os.path.join(outdir, "ik6r_table.tex"),
            os.path.join(root, "aggregated", "ik6r_summary.csv"),
        )

    # Small includes scaffold
    includes = [
        "% Auto-generated LaTeX includes",
        "% Overall MSE (across seeds):",
        f'\\input{{{os.path.join(args.latex_dir, "overall_table.tex")}}}',
        "% Near-pole bins (B0–B2):",
        f'\\input{{{os.path.join(args.latex_dir, "near_pole_table.tex")}}}',
    ]
    if args.with_rollout:
        includes += [
            "% Closed-loop tracking summary:",
            f'\\input{{{os.path.join(args.latex_dir, "rollout_table.tex")}}}',
        ]
    if args.with_3r:
        includes += [
            "% 3R TR metrics:",
            f'\\input{{{os.path.join(args.latex_dir, "e3r_table.tex")}}}',
        ]
    if args.with_6r:
        includes += [
            "% 6R TR metrics:",
            f'\\input{{{os.path.join(args.latex_dir, "ik6r_table.tex")}}}',
        ]
    with open(os.path.join(outdir, "includes.tex"), "w") as fh:
        fh.write("\n".join(includes) + "\n")
    print(f"Wrote LaTeX artifacts to {outdir}")


if __name__ == "__main__":
    main()
