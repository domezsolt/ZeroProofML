# MIT License
# See LICENSE file in the project root for full license text.
"""
ZeroProof Hybrid Overhead CLI

Compare Mask‑REAL baseline vs Hybrid configuration on a tiny synthetic task.

Usage:
  python -m zeroproof.overhead_cli --out runs/overhead.json

Notes:
  - Uses a small 1D regression task y = 1/(x-a) with a TR‑Rational model.
  - Runs a single epoch through a minimal trainer to estimate per‑batch timing.
  - Does not require optional backends.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import zeroproof as zp
from zeroproof.autodiff import TRNode
from zeroproof.utils.overhead import overhead_report


def make_dataset(
    center: float = 0.5, n: int = 256, exclude: float = 0.05
) -> List[Tuple[List[zp.TRScalar], List[zp.TRScalar]]]:
    # Simple 1D grid excluding a small window around the pole
    xs: List[float] = []
    lo, hi = -2.0, 2.0
    step = (hi - lo) / max(1, n - 1)
    for i in range(n):
        x = lo + i * step
        if abs(x - center) >= exclude:
            xs.append(x)

    def f(x: float) -> float:
        return 1.0 / (x - center)

    inputs = [zp.real(float(x)) for x in xs]
    targets = [zp.real(float(f(x))) for x in xs]
    # Single batch loader
    return [(inputs, targets)]


class _MiniOptimizer:
    def __init__(self, params: List[TRNode], lr: float = 0.01) -> None:
        self.params = params
        self.learning_rate = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        if self.learning_rate == 0.0:
            return
        from zeroproof.core import tr_mul, tr_sub

        lr_node = TRNode.constant(zp.real(self.learning_rate))
        for p in self.params:
            if p.gradient is not None and p.gradient.tag == zp.TRTag.REAL:
                update = tr_mul(lr_node, p.gradient)
                p._value = tr_sub(p.value, update)


class _MiniTrainer:
    """Tiny trainer with batch APIs to satisfy overhead_report."""

    def __init__(self, model: Any, lr: float = 0.01) -> None:
        self.model = model
        self.optimizer = _MiniOptimizer(model.parameters(), lr)
        self.hybrid_schedule = None
        self.epoch = 0

    def _train_batch(
        self, inputs: List[zp.TRScalar], targets: List[zp.TRScalar], _unused=None
    ) -> Dict[str, float]:
        import time

        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        # Forward and loss accumulation (balanced sum)
        losses: List[TRNode] = []
        for x_sc, t_sc in zip(inputs, targets):
            y, tag = self.model.forward(x_sc)
            if tag == zp.TRTag.REAL:
                diff = y - TRNode.constant(t_sc)
                losses.append(diff * diff)
        if not losses:
            return {"loss": float("inf"), "optim_ms": 0.0}

        def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
            if not nodes:
                return TRNode.constant(zp.real(0.0))
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            return _pairwise_sum(nodes[:mid]) + _pairwise_sum(nodes[mid:])

        total = _pairwise_sum(losses) / TRNode.constant(zp.real(float(len(losses))))
        total.backward()
        self.optimizer.step()
        t1 = time.perf_counter()
        return {
            "loss": total.value.value if total.value.tag == zp.TRTag.REAL else float("inf"),
            "optim_ms": (t1 - t0) * 1000.0,
        }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ZeroProof Hybrid Overhead")
    ap.add_argument("--out", default="runs/overhead.json", help="Output JSON path")
    ap.add_argument("--center", type=float, default=0.5, help="Pole center a in 1/(x-a)")
    ap.add_argument("--n", type=int, default=256, help="Number of samples")
    ap.add_argument("--exclude", type=float, default=0.05, help="Exclusion radius around pole")
    args = ap.parse_args(argv)

    # Model
    model = zp.layers.TRRational(d_p=2, d_q=1, basis=zp.layers.ChebyshevBasis())
    trainer = _MiniTrainer(model, lr=0.01)
    data = make_dataset(center=args.center, n=args.n, exclude=args.exclude)

    rep = overhead_report(trainer, data)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Overhead report saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
