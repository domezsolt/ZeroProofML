"""
Evaluator CLI demo

Run integrated evaluation metrics on a simple TR-Rational model over a 1D grid.

Usage:
  python examples/evaluator_cli.py --true-pole 0.5 --xmin -2 --xmax 2 --n 200 \
    --no-viz --out metrics.json

This avoids optional backends and runs purely on the NumPy/TR path.
"""

from __future__ import annotations

import argparse
from typing import List

import zeroproof as zp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run integrated evaluator on a TR model")
    ap.add_argument("--true-pole", type=float, default=0.5, help="Ground-truth pole location")
    ap.add_argument("--xmin", type=float, default=-2.0, help="X range min")
    ap.add_argument("--xmax", type=float, default=2.0, help="X range max")
    ap.add_argument("--n", type=int, default=200, help="Number of evaluation points")
    ap.add_argument("--no-viz", action="store_true", help="Disable visualization")
    ap.add_argument("--out", type=str, default="", help="Path to save metrics JSON (optional)")
    return ap.parse_args()


def build_model() -> zp.layers.TRRational:
    # Simple, small rational model
    return zp.layers.TRRational(d_p=2, d_q=1, basis=zp.layers.ChebyshevBasis())


def make_grid(xmin: float, xmax: float, n: int) -> List[float]:
    if n <= 1:
        return [xmin]
    step = (xmax - xmin) / (n - 1)
    return [xmin + i * step for i in range(n)]


def main() -> None:
    args = parse_args()

    model = build_model()
    xs = make_grid(args.xmin, args.xmax, args.n)

    evaluator = zp.utils.evaluation_api.create_evaluator(
        true_poles=[args.true_pole],
        enable_viz=not args.no_viz,
        save_plots=not args.no_viz,
        plot_dir="evaluation_plots",
    )

    metrics = evaluator.evaluate_model(model, xs)
    summary = evaluator.get_summary_statistics()
    print("Evaluation summary:")
    print(summary)

    if args.out:
        evaluator.export_metrics(args.out)
        print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
