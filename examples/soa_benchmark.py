"""
SoA vs regular forward micro-benchmark for TRRational value evaluation.

This benchmark compares the regular TRNode-based forward (no gradients)
to the value-only SoA fast path in TRRational.value_only().

Run:
  python examples/soa_benchmark.py --n 20000 --dp 3 --dq 3
"""

from __future__ import annotations

import argparse
import random
import time

from zeroproof.autodiff import TRNode
from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational


def bench(n: int, d_p: int, d_q: int, seed: int = 0):
    random.seed(seed)
    layer = TRRational(d_p=d_p, d_q=d_q, basis=MonomialBasis(), enable_soa_value_only=True)
    # Randomize parameters
    for th in layer.theta:
        th._value = real(random.uniform(-0.5, 0.5))
    for ph in layer.phi:
        ph._value = real(random.uniform(-0.1, 0.1))

    xs = [random.uniform(-1.0, 1.0) for _ in range(n)]

    # Regular forward (no grads): build nodes and compute
    t0 = time.perf_counter()
    s = 0.0
    for x in xs:
        y_node, _ = layer.forward(TRNode.constant(real(x)))
        if y_node.tag == y_node.value.tag:  # always true; force access
            s += float(y_node.value.value) if y_node.tag.name == "REAL" else 0.0
    t1 = time.perf_counter()
    t_regular = t1 - t0

    # SoA fast path (value-only)
    t2 = time.perf_counter()
    s2 = 0.0
    for x in xs:
        y = layer.value_only(x)
        s2 += float(y.value) if y.tag.name == "REAL" else 0.0
    t3 = time.perf_counter()
    t_soa = t3 - t2

    speedup = t_regular / t_soa if t_soa > 0 else float("inf")
    print(f"N={n} d_p={d_p} d_q={d_q}")
    print(
        f"Regular forward: {t_regular:.4f}s  | SoA (value-only): {t_soa:.4f}s  | speedup: {speedup:.2f}x"
    )
    # Prevent optimizing away sums
    print(f"checksum regular={s:.6f} soa={s2:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--dp", type=int, default=3)
    p.add_argument("--dq", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    bench(args.n, args.dp, args.dq, seed=args.seed)


if __name__ == "__main__":
    main()
