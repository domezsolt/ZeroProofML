"""
Update CI benchmark baseline.

Copies the newest JSON from a benchmark results directory to
benchmarks/baseline.json.

Usage:
  python scripts/update_benchmark_baseline.py --src benchmark_results
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from typing import Optional


def latest_json(src_dir: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    return files[-1] if files else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Update benchmarks/baseline.json from latest result")
    ap.add_argument("--src", default="benchmark_results", help="Source directory with benchmark JSONs")
    args = ap.parse_args()

    src = latest_json(args.src)
    if not src:
        print(f"No JSON files found in {args.src}")
        return 1
    os.makedirs("benchmarks", exist_ok=True)
    dst = os.path.join("benchmarks", "baseline.json")
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

