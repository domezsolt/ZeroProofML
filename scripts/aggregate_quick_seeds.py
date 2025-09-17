#!/usr/bin/env python3
"""
Aggregate across-seed results from experiments/robotics/run_all.py quick runs.

Reads comprehensive_comparison.json files under a glob (e.g., results/robotics/quick_s*/)
and computes mean±std across seeds for each method's overall MSE and, if available,
near-pole buckets (B0–B2 or B0–B4 if present).

Outputs a CSV and optional LaTeX table.

Usage:
  python scripts/aggregate_quick_seeds.py \
    --glob 'results/robotics/quick_s*/comprehensive_comparison.json' \
    --out results/robotics/aggregated/seed_summary.csv \
    --latex results/robotics/aggregated/seed_summary.tex
"""

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Tuple


def load_comp(path: str) -> Dict[str, Any]:
    with open(path, 'r') as fh:
        return json.load(fh)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def agg_mean_std(xs: List[float]) -> Tuple[float, float, int]:
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    n = len(vals)
    if n == 0:
        return (float('nan'), float('nan'), 0)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    return (mu, math.sqrt(var), n)


def collect(paths: List[str], methods: List[str]) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    # Returns: method -> metric_name -> (mean, std, n)
    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    # Per-seed collection
    per_seed: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    for p in sorted(paths):
        try:
            comp = load_comp(p)
        except Exception:
            continue
        indiv = comp.get('individual_results', {})
        for m in methods:
            res = indiv.get(m)
            if not isinstance(res, dict):
                continue
            # Overall test MSE
            overall = res.get('final_mse') or res.get('final_val_mse') or res.get('test_mse')
            if overall is not None:
                per_seed[m].setdefault('overall_mse', []).append(safe_float(overall))
            # Near-pole buckets if present
            npb = res.get('near_pole_bucket_mse', {})
            mse_map = npb.get('bucket_mse') if isinstance(npb, dict) else None
            if isinstance(mse_map, dict):
                for key, val in mse_map.items():
                    metric_name = f"bucket_{key}"
                    per_seed[m].setdefault(metric_name, []).append(safe_float(val))
    # Aggregate
    for m in methods:
        out[m] = {}
        metrics = per_seed.get(m, {})
        for name, vals in metrics.items():
            out[m][name] = agg_mean_std(vals)
    return out


def write_csv(path: str, agg: Dict[str, Dict[str, Tuple[float, float, int]]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Collect all metric names
    metric_names: List[str] = []
    for m in agg.values():
        for k in m.keys():
            if k not in metric_names:
                metric_names.append(k)
    # Order metrics: overall first, then B0..B4 if present
    def metric_key(n: str) -> Tuple[int, str]:
        if n == 'overall_mse':
            return (0, n)
        if n.startswith('bucket_'):
            return (1, n)
        return (2, n)
    metric_names.sort(key=metric_key)
    with open(path, 'w') as fh:
        # Header
        fh.write('Method')
        for n in metric_names:
            fh.write(f',{n}_mean,{n}_std,{n}_n')
        fh.write('\n')
        # Rows
        for method, metrics in agg.items():
            fh.write(method)
            for n in metric_names:
                mu, sd, nn = metrics.get(n, (float('nan'), float('nan'), 0))
                fh.write(f',{mu},{sd},{nn}')
            fh.write('\n')


def write_latex(path: str, agg: Dict[str, Dict[str, Tuple[float, float, int]]], methods: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Select columns
    cols = ['overall_mse', '(0e+00,1e-05]', '(1e-05,1e-04]', '(1e-04,1e-03]']
    # Build table
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('  \\centering\\small')
    lines.append('  \\begin{tabular}{lcccc}')
    lines.append('    \\toprule')
    lines.append('    Method & Overall MSE & B0 & B1 & B2 \\\\')
    lines.append('    \\midrule')
    for m in methods:
        row = [m]
        metrics = agg.get(m, {})
        # overall
        mu, sd, n = metrics.get('overall_mse', (float('nan'), float('nan'), 0))
        row.append(f"{mu:.6f} $\\pm$ {sd:.6f} (n={n})" if n else '-')
        # buckets
        for key in cols[1:]:
            name = f'bucket_{key}'
            mu, sd, n = metrics.get(name, (float('nan'), float('nan'), 0))
            row.append(f"{mu:.6f} $\\pm$ {sd:.6f} (n={n})" if n else '-')
        lines.append('    ' + ' & '.join(row) + ' \\\\')
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('  \\caption{Across-seed mean$\\pm$std (3 seeds) for overall MSE and near-pole buckets on the exact dataset.}')
    lines.append('  \\label{tab:seed_agg}')
    lines.append('\\end{table}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Aggregate across-seed quick runs (run_all.py)')
    ap.add_argument('--glob', default='results/robotics/quick_s*/comprehensive_comparison.json')
    ap.add_argument('--methods', nargs='*', default=['MLP', 'Rational+ε'])
    ap.add_argument('--out', default='results/robotics/aggregated/seed_summary.csv')
    ap.add_argument('--latex', default=None)
    args = ap.parse_args()

    paths = glob.glob(args.glob)
    if not paths:
        print(f'No files matched: {args.glob}')
        return
    agg = collect(paths, args.methods)
    write_csv(args.out, agg)
    print(f'Saved seed aggregate CSV to {args.out}')
    if args.latex:
        write_latex(args.latex, agg, args.methods)
        print(f'Saved seed aggregate LaTeX to {args.latex}')


if __name__ == '__main__':
    main()

