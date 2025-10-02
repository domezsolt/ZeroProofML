# MIT License
# See LICENSE file in the project root for full license text.
"""
Benchmark Summary CLI

Render a compact summary from a ZeroProof benchmark JSON file.

Usage:
  python -m zeroproof.bench_summary path/to/bench.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _format_s(val: float) -> str:
    return f"{val*1000:.3f} ms"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ZeroProof benchmark summary")
    ap.add_argument("path", help="Benchmark JSON file")
    args = ap.parse_args(argv)
    data = _load(args.path)
    results = data.get("results", {})

    print("ZeroProof Bench Summary")
    print("=" * 28)
    for suite, entries in results.items():
        if not isinstance(entries, dict):
            continue
        print(f"\n[{suite}]")
        for name, obj in entries.items():
            if not isinstance(obj, dict):
                continue
            mt = obj.get("mean_time")
            ops = obj.get("operations_per_second")
            if mt is not None:
                print(f"- {name}: {_format_s(float(mt))}")
            elif ops is not None:
                print(f"- {name}: {float(ops):.0f} ops/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
