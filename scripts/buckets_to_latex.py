#!/usr/bin/env python3
"""
Convert aggregated per-bucket CSV to a LaTeX table.

Reads the CSV produced by scripts/aggregate_buckets.py and emits a LaTeX
tabular with selected methods and buckets. Bucket cells can include counts.

Usage (typical):
  python scripts/buckets_to_latex.py \
    --csv results/robotics/aggregated/buckets_table.csv \
    --methods "ZeroProofML (Basic)" "ZeroProofML (Full)" \
             "Rational+ε (ε=1e-2, no clip)" "Rational+ε (ε=1e-3, no clip)" \
             "Rational+ε (ε=1e-4, no clip)" "Rational+ε+Clip (ε=1e-3)" \
    --buckets B0 B1 B2 B3 B4 \
    --caption "Per-bucket MSE (mean) with counts (n) on exact-pole dataset" \
    --label tab:buckets_mse \
    --out results/robotics/aggregated/buckets_table.tex

Defaults:
  - If --methods omitted, uses a heuristic order (TR Full, TR Basic, ε (no clip), ε+Clip).
  - If --buckets omitted, uses B0 B1 B2 B3 B4.
  - Cells formatted as "{mse:.6f} (n={count})".
"""

import argparse
import csv
import os
from typing import List, Dict


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        return list(r)


def default_method_order(rows: List[Dict[str, str]]) -> List[str]:
    names = [row['Method'] for row in rows]
    priority = []
    for n in names:
        key = (4, n)
        low = n.lower()
        if 'zeroproofml (full)' in low:
            key = (0, n)
        elif 'zeroproofml (basic)' in low:
            key = (1, n)
        elif 'no clip' in low:
            key = (2, n)
        elif '+clip' in low or 'clip' in low:
            key = (3, n)
        priority.append((key, n))
    priority.sort(key=lambda x: x[0])
    ordered = [n for (_, n) in priority]
    # Deduplicate
    seen = set()
    out = []
    for n in ordered:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_table(rows: List[Dict[str, str]], methods: List[str], buckets: List[str], caption: str, label: str) -> str:
    # Header row
    headers = ['Method'] + buckets
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('  \\centering\\small')
    lines.append('  \\begin{tabular}{l' + 'c' * len(buckets) + '}')
    lines.append('    \\toprule')
    lines.append('    ' + ' & '.join(headers) + ' \\\\')
    lines.append('    \\midrule')

    # Map rows by method
    by_method = {row['Method']: row for row in rows}
    for m in methods:
        if m not in by_method:
            continue
        row = by_method[m]
        cells = [m]
        for b in buckets:
            mse = row.get(b)
            std = row.get(f'{b}_std')
            n = row.get(f'{b}_n')
            try:
                mse_val = float(mse) if mse not in (None, '', 'None') else float('nan')
                std_val = float(std) if std not in (None, '', 'None') else None
                if std_val is not None and std_val == std_val:  # not NaN
                    cell = f"{mse_val:.6f} $\\pm$ {std_val:.6f} (n={n})"
                else:
                    cell = f"{mse_val:.6f} (n={n})"
            except Exception:
                cell = f"- (n={n})"
            cells.append(cell)
        lines.append('    ' + ' & '.join(cells) + ' \\\\')
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    if caption:
        lines.append(f'  \\caption{{{caption}}}')
    if label:
        lines.append(f'  \\label{{{label}}}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Convert aggregated buckets CSV to LaTeX table')
    ap.add_argument('--csv', required=True, help='Path to aggregated CSV (from aggregate_buckets.py)')
    ap.add_argument('--methods', nargs='*', default=None, help='List of method names to include (in order)')
    ap.add_argument('--buckets', nargs='*', default=None, help='Buckets to include (default: B0 B1 B2 B3 B4)')
    ap.add_argument('--caption', default='Per-bucket MSE (mean) with counts (n). Lower is better.', help='LaTeX caption')
    ap.add_argument('--label', default='tab:buckets_mse', help='LaTeX label')
    ap.add_argument('--out', default='results/robotics/aggregated/buckets_table.tex', help='Output .tex file')
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print(f'No rows in {args.csv}')
        return
    methods = args.methods or default_method_order(rows)
    buckets = args.buckets or ['B0', 'B1', 'B2', 'B3', 'B4']

    table = build_table(rows, methods, buckets, args.caption, args.label)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(table + '\n')
    print(f'Saved LaTeX table to {args.out}')


if __name__ == '__main__':
    main()
