# MIT License
# See LICENSE file in the project root for full license text.
"""
ZeroProof Evaluator CLI

Run integrated evaluation metrics on a model over a 1D grid.

Usage:
  python -m zeroproof.eval --xmin -2 --xmax 2 --n 201 --true-pole 0.5 --out metrics.json

If no checkpoint is provided, evaluates a small TR-Rational baseline.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from . import layers, utils


def _make_grid(xmin: float, xmax: float, n: int) -> List[float]:
    if n <= 1:
        return [xmin]
    step = (xmax - xmin) / (n - 1)
    return [xmin + i * step for i in range(n)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ZeroProof evaluator CLI")
    ap.add_argument("--xmin", type=float, default=-2.0)
    ap.add_argument("--xmax", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=201)
    ap.add_argument("--true-pole", type=float, default=0.5)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    args = ap.parse_args(argv)

    # Build model (default small TR-Rational)
    model = layers.TRRational(d_p=2, d_q=1, basis=layers.ChebyshevBasis())

    xs = _make_grid(args.xmin, args.xmax, args.n)

    evaluator = utils.evaluation_api.create_evaluator(
        true_poles=[args.true_pole],
        enable_viz=not args.no_viz,
        save_plots=not args.no_viz,
        plot_dir="evaluation_plots",
    )

    evaluator.evaluate_model(model, xs)
    print("Evaluation summary:", evaluator.get_summary_statistics())

    if args.out:
        evaluator.export_metrics(args.out)
        print(f"Saved metrics to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
