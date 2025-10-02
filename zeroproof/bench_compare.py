# MIT License
# See LICENSE file in the project root for full license text.
"""
Benchmark Results Comparator

Compare two ZeroProof benchmark JSON files and report regressions.

Usage:
  python -m zeroproof.bench_compare --baseline pathA.json --candidate pathB.json \
    --max-slowdown 1.20

The comparator checks suites and tries to compare common entries by
`mean_time` (lower is better) and `operations_per_second` (higher is better).
If either indicates a slowdown beyond the threshold, it reports and exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _collect_flat(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    flat: Dict[str, Dict[str, Any]] = {}
    for suite, entries in results.items():
        if isinstance(entries, dict):
            for name, obj in entries.items():
                if isinstance(obj, dict):
                    flat[f"{suite}.{name}"] = obj
    return flat


def compare(baseline: Dict[str, Any], candidate: Dict[str, Any], max_slowdown: float) -> int:
    base_flat = _collect_flat(baseline.get("results", {}))
    cand_flat = _collect_flat(candidate.get("results", {}))
    keys = sorted(set(base_flat.keys()) & set(cand_flat.keys()))
    if not keys:
        print("No common benchmark entries to compare.")
        return 0

    failures = 0
    for key in keys:
        b = base_flat[key]
        c = cand_flat[key]
        b_time = float(b.get("mean_time", 0.0))
        c_time = float(c.get("mean_time", 0.0))
        b_ops = float(b.get("operations_per_second", 0.0))
        c_ops = float(c.get("operations_per_second", 0.0))
        # Prefer mean_time; fallback to ops/sec if unavailable
        slowdown = None
        if b_time > 0 and c_time > 0:
            slowdown = c_time / b_time
        elif b_ops > 0 and c_ops > 0:
            slowdown = b_ops / c_ops  # invert so >1 means slower
        if slowdown is None:
            continue
        status = "OK"
        if slowdown > max_slowdown:
            status = "REGRESSION"
            failures += 1
        print(f"{key}: slowdown_x={slowdown:.3f} [{status}]")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare ZeroProof benchmarks")
    ap.add_argument("--baseline", required=True, help="Baseline JSON file")
    ap.add_argument("--candidate", required=True, help="Candidate JSON file")
    ap.add_argument("--max-slowdown", type=float, default=1.20, help="Max allowed slowdown factor")
    args = ap.parse_args(argv)

    base = _load(args.baseline)
    cand = _load(args.candidate)
    return compare(base, cand, args.max_slowdown)


if __name__ == "__main__":
    sys.exit(main())
