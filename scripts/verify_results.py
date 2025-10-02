#!/usr/bin/env python3
"""
Verify paper-parity metrics from run JSONs.

Checks:
- PLE (Pole Localization Error) is under a max threshold across runs
- Near-pole bucket MSE bounds for B0 and B1 (per-seed or percentile across seeds)
- Optional: Non-empty B0–B3 bucket counts (guardrail promoted to failure)

Inputs can be:
- A single comprehensive JSON file (e.g., results/.../comprehensive_comparison.json)
- A directory containing seed_* or quick_* subfolders with comprehensive JSONs
- A glob pattern via --glob

Exit codes:
- 0: all checks pass
- 1: one or more checks failed
- 2: invalid input / no files found
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


def _is_json_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".json")


def _find_jsons(path_or_glob: str) -> List[str]:
    # Glob pattern support
    if any(ch in path_or_glob for ch in ["*", "?", "["]):
        return sorted([p for p in _glob.glob(path_or_glob, recursive=True) if _is_json_file(p)])
    # Single file
    if _is_json_file(path_or_glob):
        return [path_or_glob]
    # Directory scan: look for seed_*/comprehensive_comparison.json and quick_*/comprehensive_comparison.json
    if os.path.isdir(path_or_glob):
        candidates: List[str] = []
        for subpat in ["seed_*", "quick_*"]:
            candidates.extend(
                _glob.glob(os.path.join(path_or_glob, subpat, "comprehensive_comparison.json"))
            )
        # Also allow direct comprehensive_comparison.json at root
        direct = os.path.join(path_or_glob, "comprehensive_comparison.json")
        if os.path.exists(direct):
            candidates.append(direct)
        return sorted([p for p in candidates if _is_json_file(p)])
    return []


def _to_float_edges(edges: List[Any]) -> List[float]:
    out: List[float] = []
    for e in edges:
        if isinstance(e, (int, float)):
            out.append(float(e))
        else:
            s = str(e).strip().lower()
            if s in ("inf", "+inf", "infinity"):
                out.append(float("inf"))
            else:
                try:
                    out.append(float(s))
                except Exception:
                    out.append(float("nan"))
    return out


def _bucket_key(edges: List[float], i: int) -> str:
    lo = edges[i]
    hi = edges[i + 1]
    lo_s = f"{lo:.0e}" if math.isfinite(lo) else "inf"
    hi_s = f"{hi:.0e}" if math.isfinite(hi) else "inf"
    return f"({lo_s},{hi_s}]"


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def verify_files(
    files: List[str],
    method_key: str,
    max_ple: Optional[float],
    max_b0: Optional[float],
    max_b1: Optional[float],
    percentile: Optional[float],
    require_nonempty_b03: bool,
    verbose: bool,
) -> int:
    ple_vals: List[float] = []
    b0_vals: List[float] = []
    b1_vals: List[float] = []

    nonempty_failures: List[Tuple[str, List[str]]] = []
    missing: List[str] = []

    for fp in files:
        try:
            with open(fp, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[ERROR] Failed to read {fp}: {e}")
            return 2

        indiv = data.get("individual_results") or {}
        if method_key not in indiv:
            if verbose:
                print(f"[WARN] Method '{method_key}' not in {fp} -> keys: {list(indiv.keys())}")
            missing.append(fp)
            continue

        m = indiv[method_key]
        # PLE
        ple = None
        if isinstance(m.get("pole_metrics"), dict):
            ple = m["pole_metrics"].get("ple")
        if isinstance(ple, (int, float)):
            ple_vals.append(float(ple))
        elif verbose:
            print(f"[INFO] No PLE found in {fp} for '{method_key}'")

        # Bucket MSEs
        nb = m.get("near_pole_bucket_mse")
        if isinstance(nb, dict):
            edges = _to_float_edges(
                nb.get("edges") or data.get("dataset_info", {}).get("bucket_edges") or []
            )
            if len(edges) >= 3:
                k0 = _bucket_key(edges, 0)
                k1 = _bucket_key(edges, 1)
                bmap = nb.get("bucket_mse", {}) or {}
                v0 = bmap.get(k0)
                v1 = bmap.get(k1)
                if isinstance(v0, (int, float)):
                    b0_vals.append(float(v0))
                if isinstance(v1, (int, float)):
                    b1_vals.append(float(v1))
                if require_nonempty_b03:
                    cm = nb.get("bucket_counts", {}) or {}
                    empty: List[str] = []
                    for i in range(min(4, len(edges) - 1)):
                        ki = _bucket_key(edges, i)
                        cnt = cm.get(ki)
                        if not isinstance(cnt, (int, float)) or int(cnt) <= 0:
                            empty.append(ki)
                    if empty:
                        nonempty_failures.append((fp, empty))
        elif verbose:
            print(f"[INFO] No near_pole_bucket_mse found in {fp} for '{method_key}'")

    failures: List[str] = []

    # Non-empty guardrail
    if require_nonempty_b03 and nonempty_failures:
        for fp, empties in nonempty_failures:
            failures.append(f"Empty near-pole buckets in {fp}: {', '.join(empties)}")

    # Aggregate checks (percentile or per-run)
    def check_metric(name: str, vals: List[float], bound: Optional[float]) -> None:
        if bound is None or not vals:
            return
        if percentile is None:
            # Require all runs <= bound
            bad = [
                (i, v)
                for i, v in enumerate(vals)
                if not (isinstance(v, (int, float)) and v <= bound)
            ]
            if bad:
                details = ", ".join([f"run{i}={v:.6f}" for i, v in bad])
                failures.append(f"{name} exceeded bound {bound:.6f}: {details}")
        else:
            pval = _percentile(vals, float(percentile))
            if not (isinstance(pval, (int, float)) and pval <= bound):
                failures.append(
                    f"{name} {percentile}th percentile {pval:.6f} exceeded bound {bound:.6f}"
                )

    check_metric("PLE", ple_vals, max_ple)
    check_metric("B0 MSE", b0_vals, max_b0)
    check_metric("B1 MSE", b1_vals, max_b1)

    # Report summary
    print("Verification summary:")
    print(f"  Files checked: {len(files)}")
    print(f"  Method: {method_key}")
    if ple_vals:
        print(f"  PLE: n={len(ple_vals)} min={min(ple_vals):.6f} max={max(ple_vals):.6f}")
    if b0_vals:
        print(f"  B0 MSE: n={len(b0_vals)} min={min(b0_vals):.6f} max={max(b0_vals):.6f}")
    if b1_vals:
        print(f"  B1 MSE: n={len(b1_vals)} min={min(b1_vals):.6f} max={max(b1_vals):.6f}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if missing and len(missing) == len(files):
        print("[ERROR] No files contained the requested method; nothing verified.")
        return 2
    print("\nAll checks passed.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify paper-parity metrics from comprehensive run JSONs"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--path", help="Path to a JSON file or a directory containing seed_* subfolders"
    )
    src.add_argument(
        "--glob",
        dest="globpat",
        help="Glob pattern to JSON files (e.g., results/.../seed_*/comprehensive_comparison.json)",
    )
    ap.add_argument(
        "--method", default="ZeroProofML-Full", help="Method key in 'individual_results' to verify"
    )
    ap.add_argument(
        "--max-ple", type=float, default=None, help="Max allowed PLE (fail if exceeded)"
    )
    ap.add_argument("--max-b0", type=float, default=None, help="Max allowed MSE for B0 bucket")
    ap.add_argument("--max-b1", type=float, default=None, help="Max allowed MSE for B1 bucket")
    ap.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Aggregate percentile across seeds (set to None/omit to require per-run bounds)",
    )
    ap.add_argument(
        "--no-percentile",
        action="store_true",
        help="Disable percentile aggregation; enforce per-run bounds",
    )
    ap.add_argument(
        "--require-nonempty-b03",
        action="store_true",
        help="Fail when any of B0–B3 bucket counts are zero",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = ap.parse_args(argv)

    files: List[str] = []
    if args.globpat:
        files = _find_jsons(args.globpat)
    elif args.path:
        files = _find_jsons(args.path)

    if not files:
        print("[ERROR] No JSON files found to verify.")
        return 2

    perc = (
        None
        if args.no_percentile
        else (None if args.percentile is None else float(args.percentile))
    )
    return verify_files(
        files,
        method_key=args.method,
        max_ple=args.max_ple,
        max_b0=args.max_b0,
        max_b1=args.max_b1,
        percentile=perc,
        require_nonempty_b03=bool(args.require_nonempty_b03),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    raise SystemExit(main())
