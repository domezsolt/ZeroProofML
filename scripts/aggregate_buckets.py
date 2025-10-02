#!/usr/bin/env python3
"""
Aggregate per-bucket metrics across multiple runs into a single CSV.

Scans run directories for `buckets.json` (produced by scripts/evaluate_trainer_buckets.py)
and emits a wide CSV with columns:
  Method, B0, B1, B2, B3, B4, B0_n, B1_n, B2_n, B3_n, B4_n

Usage (typical):
  python scripts/aggregate_buckets.py \
    --scan runs \
    --out results/robotics/aggregated/buckets_table.csv

You can also pass explicit paths to buckets files:
  python scripts/aggregate_buckets.py --files runs/ik_tr_basic/buckets.json runs/ik_tr_full/buckets.json
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple


def _label_from_path(path: str, results_path: str) -> str:
    dname = os.path.basename(os.path.dirname(path))
    # Heuristics based on directory name
    name = dname
    low = dname.lower()
    if "tr_basic" in low:
        return "ZeroProofML (Basic)"
    if "tr_full" in low:
        return "ZeroProofML (Full)"
    if "noclip" in low:
        # Try to extract epsilon from results
        try:
            with open(results_path, "r") as fh:
                res = json.load(fh)
            eps = res.get("config", {}).get("epsilon")
            if eps is None:
                # parse from dirname
                if "1e-2" in low or "1e2" in low:
                    eps = "1e-2"
                elif "1e-3" in low or "1e3" in low:
                    eps = "1e-3"
                elif "1e-4" in low or "1e4" in low:
                    eps = "1e-4"
            return f"Rational+ε (ε={eps}, no clip)"
        except Exception:
            return "Rational+ε (no clip)"
    if "ik_eps_" in low and "noclip" not in low:
        try:
            with open(results_path, "r") as fh:
                res = json.load(fh)
            eps = res.get("config", {}).get("epsilon")
            return f"Rational+ε+Clip (ε={eps})"
        except Exception:
            return "Rational+ε+Clip"
    # Fallback to model_type label
    try:
        with open(results_path, "r") as fh:
            res = json.load(fh)
        mt = res.get("config", {}).get("model_type", "unknown")
        return mt
    except Exception:
        return name


def _bucket_order(edges: List[float]) -> List[str]:
    # Produce canonical keys like "(0e+00,1e-05]" ... using edges order
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]

        def fmt(x):
            if x == float("inf"):
                return "inf"
            try:
                return f"{x:.0e}"
            except Exception:
                return str(x)

        out.append(f"({fmt(lo)},{fmt(hi)}]")
    return out


def _load_buckets_file(path: str) -> Tuple[Dict[str, Dict[str, float]], List[str], str]:
    with open(path, "r") as fh:
        data = json.load(fh)
    edges = data.get("bucket_edges") or []
    # Normalize edges to floats
    e2 = []
    for e in edges:
        try:
            e2.append(float(e))
        except Exception:
            s = str(e).strip().lower()
            e2.append(float("inf") if s in ("inf", "+inf", "infinity") else float(e))
    order = _bucket_order(e2) if e2 else []
    per_bucket = data.get("per_bucket", {})
    results_path = data.get("results", "")
    return per_bucket, order, results_path


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-bucket MSE across runs into a CSV table"
    )
    ap.add_argument("--scan", default="runs", help="Root directory to scan for buckets.json files")
    ap.add_argument("--files", nargs="*", default=None, help="Explicit list of buckets.json files")
    ap.add_argument(
        "--out", default="results/robotics/aggregated/buckets_table.csv", help="Output CSV path"
    )
    args = ap.parse_args()

    if args.files:
        files = args.files
    else:
        files = glob.glob(os.path.join(args.scan, "**", "buckets.json"), recursive=True)
        files.sort()
    if not files:
        print("No buckets.json files found")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    bucket_headers = ["B0", "B1", "B2", "B3", "B4"]
    count_headers = [f"{b}_n" for b in bucket_headers]
    std_headers = [f"{b}_std" for b in bucket_headers]

    for path in files:
        try:
            per_bucket, order, results_path = _load_buckets_file(path)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
            continue
        label = _label_from_path(path, results_path)
        # Map in order
        means = {}
        counts = {}
        stds = {}
        # Build key mapping from per_bucket keys to order index
        # per_bucket[k] has keys: mean_mse, std_mse, n
        for idx, key in enumerate(order):
            means[idx] = None
            counts[idx] = 0
            stds[idx] = None
        for k, v in per_bucket.items():
            try:
                idx = order.index(k)
            except ValueError:
                continue
            means[idx] = v.get("mean_mse")
            counts[idx] = v.get("n", 0)
            stds[idx] = v.get("std_mse")
        row = {"Method": label}
        for i, h in enumerate(bucket_headers):
            row[h] = means.get(i)
        for i, h in enumerate(std_headers):
            row[h] = stds.get(i)
        for i, h in enumerate(count_headers):
            row[h] = counts.get(i)
        row["Source"] = os.path.dirname(path)
        rows.append(row)

    # Write CSV
    headers = ["Method"] + bucket_headers + std_headers + count_headers + ["Source"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved aggregated table to {args.out}")


if __name__ == "__main__":
    main()
