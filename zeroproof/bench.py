# MIT License
# See LICENSE file in the project root for full license text.
"""
ZeroProof Benchmarks CLI

Run a small suite of microbenchmarks to characterize performance.

Usage:
  python -m zeroproof.bench --suite all --out benchmark_results

Suites:
  - arithmetic: core TR ops
  - autodiff: simple gradients, chain rule, multi-parameter
  - layers: TR-Rational and TR-Norm forward
  - scaling: graph depth and batch size scaling
  - parallel: sequential vs threaded map speedup
  - memory: simple memory usage estimator on large graphs
  - caching: fibonacci with/without memoization
  - all: run all of the above
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import zeroproof as zp
from zeroproof.autodiff import TRNode, gradient_tape, tr_add, tr_mul
from zeroproof.core import real as tr_real
from zeroproof.core import tr_add as core_add
from zeroproof.core import tr_div as core_div
from zeroproof.core import tr_mul as core_mul
from zeroproof.core.reductions import set_deterministic_reduction
from zeroproof.core.reductions import tr_sum as core_sum
from zeroproof.layers import TRNorm, TRRational
from zeroproof.utils import (
    OperationBenchmark,
    TRBenchmark,
    create_scaling_benchmark,
    memoize_tr,
    parallel_map,
)


class MiniBench:
    def __init__(self, out_dir: str, iterations: int = 1000, samples: int = 5) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.iterations = iterations
        self.samples = samples

    def suite_arithmetic(self) -> Dict[str, Any]:
        bench = OperationBenchmark()
        res = bench.benchmark_arithmetic()
        self.results["arithmetic"] = res
        return res

    def suite_autodiff(self) -> Dict[str, Any]:
        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        def simple_derivative():
            x = TRNode.parameter(zp.real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = tr_mul(x, x)
            return tape.gradient(y, x)

        res["simple_derivative"] = bench.benchmark(
            simple_derivative,
            name="simple_derivative",
            iterations=self.iterations,
            samples=self.samples,
        )

        def chain():
            x = TRNode.parameter(zp.real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = tr_mul(x, x)
                z = tr_add(y, x)
                w = tr_mul(z, z)
            return tape.gradient(w, x)

        res["chain_rule"] = bench.benchmark(
            chain, name="chain_rule", iterations=max(1, self.iterations // 5), samples=self.samples
        )
        self.results["autodiff"] = res
        return res

    def suite_layers(self) -> Dict[str, Any]:
        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        rat = TRRational(d_p=3, d_q=2)

        def rat_forward():
            return rat.forward(TRNode.constant(zp.real(1.25)))

        res["tr_rational_forward"] = bench.benchmark(
            rat_forward,
            name="tr_rational_forward",
            iterations=self.iterations,
            samples=self.samples,
        )

        norm = TRNorm(num_features=8)

        def norm_forward():
            batch = [[TRNode.constant(zp.real(float(i + j))) for j in range(8)] for i in range(16)]
            return norm.forward(batch)

        res["tr_norm_forward_b16_f8"] = bench.benchmark(
            norm_forward,
            name="tr_norm_forward_b16_f8",
            iterations=max(1, self.iterations // 20),
            samples=self.samples,
        )

        self.results["layers"] = res
        return res

    def suite_scaling(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}

        def deep_graph(depth: int):
            x = TRNode.parameter(zp.real(1.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = x
                for _ in range(depth):
                    y = zp.tr_add(zp.tr_mul(y, x), zp.real(1.0))
            return tape.gradient(y, x)

        res["graph_depth"] = create_scaling_benchmark(
            deep_graph, sizes=[10, 20, 40, 80], name="graph_depth"
        )

        def batch_op(n: int):
            outs = []
            for i in range(n):
                r = tr_add(
                    tr_mul(TRNode.constant(zp.real(float(i))), TRNode.constant(zp.real(1.01))),
                    TRNode.constant(zp.real(0.1)),
                )
                outs.append(r)
            return outs

        res["batch_size"] = create_scaling_benchmark(
            batch_op, sizes=[10, 100, 1000], name="batch_size"
        )
        self.results["scaling"] = res
        return res

    def suite_parallel(self) -> Dict[str, Any]:
        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        def work(x: zp.TRScalar):
            r = x
            for _ in range(100):
                r = tr_add(r, TRNode.constant(zp.real(0.1)))
                r = tr_mul(r, TRNode.constant(zp.real(1.001)))
            return r

        inputs = [zp.real(float(i)) for i in range(400)]

        def seq():
            return [work(x) for x in inputs]

        res["sequential"] = bench.benchmark(
            seq, name="sequential_400", iterations=1, samples=self.samples
        )

        # Keep parallel modest for portability
        for workers in [2, 4]:

            def par():
                return parallel_map(work, inputs)

            res[f"parallel_{workers}"] = bench.benchmark(
                par, name=f"parallel_{workers}", iterations=1, samples=self.samples
            )
        self.results["parallel"] = res
        return res

    def suite_overhead(self) -> Dict[str, Any]:
        """Compare basic TR core ops vs IEEE float ops."""
        import time

        res: Dict[str, Any] = {}

        def bench_pair(tr_op, py_op, a_vals, b_vals, iters=3000):
            # TR path
            t0 = time.perf_counter()
            _acc_tr = tr_real(0.0)
            for i in range(iters):
                _acc_tr = tr_op(a_vals[i % len(a_vals)], b_vals[i % len(b_vals)])
            t1 = time.perf_counter()
            tr_time = t1 - t0
            # IEEE float path
            t0 = time.perf_counter()
            _acc_f = 0.0
            for i in range(iters):
                _acc_f = py_op(float(i % 7), float((i + 1) % 11))
            t1 = time.perf_counter()
            py_time = t1 - t0
            return {
                "tr_sec": tr_time,
                "py_sec": py_time,
                "slowdown_x": tr_time / py_time if py_time > 0 else float("inf"),
            }

        vals_a = [tr_real(float(i)) for i in range(1, 8)]
        vals_b = [tr_real(float(i)) for i in range(1, 12)]
        res["add"] = bench_pair(core_add, lambda x, y: x + y, vals_a, vals_b)
        res["mul"] = bench_pair(core_mul, lambda x, y: x * y, vals_a, vals_b)
        res["div"] = bench_pair(core_div, lambda x, y: x / (y if y != 0 else 1.0), vals_a, vals_b)
        self.results["overhead"] = res
        return res

    def suite_reductions(self) -> Dict[str, Any]:
        """Benchmark sequential vs pairwise reductions over TRScalars."""
        import random
        import time

        res: Dict[str, Any] = {}

        def make_values(n: int):
            random.seed(123)
            return [tr_real(random.uniform(-1.0, 1.0)) for _ in range(n)]

        def seq_sum(vals):
            acc = tr_real(0.0)
            for v in vals:
                acc = core_add(acc, v)
            return acc

        def pair_sum(vals):
            def _pair(xs):
                if not xs:
                    return tr_real(0.0)
                if len(xs) == 1:
                    return xs[0]
                mid = len(xs) // 2
                return core_add(_pair(xs[:mid]), _pair(xs[mid:]))

            return _pair(vals)

        for n in [100, 1000, 3000]:
            vals = make_values(n)
            # Sequential
            t0 = time.perf_counter()
            s1 = seq_sum(vals)
            t1 = time.perf_counter()
            # Pairwise
            t2 = time.perf_counter()
            s2 = pair_sum(vals)
            t3 = time.perf_counter()
            # Record
            res[f"n{n}"] = {
                "sequential_s": (t1 - t0),
                "pairwise_s": (t3 - t2),
                "slowdown_x": ((t1 - t0) / (t3 - t2)) if (t3 - t2) > 0 else float("inf"),
                "equal": (s1.value == s2.value and s1.tag == s2.tag),
            }
        self.results["reductions"] = res
        return res

    def suite_tag_overhead(self) -> Dict[str, Any]:
        """Compare a composite expression with tags vs float fallback.

        Expression: f(x) = (x^2 + x) / (x + 1)
        Uses a set of x values including ones that cause denominator=0.
        """
        # Keep these imports local to minimize module import overhead in users
        # (Ruff: they are intentionally unused within this block, see utils.overhead)
        res: Dict[str, Any] = {}

        xs_tr = [tr_real(v) for v in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]
        xs_fl = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

        def tr_once():
            acc = tr_real(0.0)
            for a in xs_tr:
                num = core_add(core_mul(a, a), a)
                den = core_add(a, tr_real(1.0))
                acc = core_add(acc, core_div(num, den))
            return acc

        def fl_once():
            acc = 0.0
            for a in xs_fl:
                num = a * a + a
                den = a + 1.0
                if den == 0.0:
                    # mimic TR: skip or treat as 0 contribution
                    continue
                acc += num / den
            return acc

        from zeroproof.utils.overhead import compare_tr_vs_float

        res["composite"] = compare_tr_vs_float(
            "composite",
            tr_once,
            fl_once,
            iterations=max(500, self.iterations),
            repeats=self.samples,
        )
        self.results["tag_overhead"] = res
        return res

    def suite_reductions_policy(self) -> Dict[str, Any]:
        """Benchmark core reductions with deterministic_reduction toggle.

        Uses zeroproof.core.reductions.tr_sum and set_deterministic_reduction.
        """
        import time

        res: Dict[str, Any] = {}
        vals = [tr_real(float(i % 7 - 3)) for i in range(2000)]
        # Disabled
        set_deterministic_reduction(False)
        t0 = time.perf_counter()
        _ = core_sum(vals)
        t1 = time.perf_counter()
        # Enabled
        set_deterministic_reduction(True)
        t2 = time.perf_counter()
        _ = core_sum(vals)
        t3 = time.perf_counter()
        # Reset to default off
        set_deterministic_reduction(False)
        res["tr_sum"] = {
            "disabled_s": (t1 - t0),
            "enabled_s": (t3 - t2),
            "slowdown_x": ((t3 - t2) / (t1 - t0)) if (t1 - t0) > 0 else float("inf"),
            "n": len(vals),
        }
        self.results["reductions_policy"] = res
        return res

    def suite_near_pole(self) -> Dict[str, Any]:
        """Compare throughput near a pole vs far from a pole.

        Uses a simple composite expression:
            f(x) = (x^2 + x) / (x - c)
        and measures runtime over values near the pole (x≈c) vs far away.
        """
        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        center = 0.5
        # Precompute TRScalars to avoid allocation cost in the timed region
        # Near-pole set: small offsets from the center, excluding exact pole
        near_offsets = [
            1e-9,
            5e-9,
            1e-8,
            5e-8,
            1e-7,
            5e-7,
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
        ]
        xs_near = []
        for d in near_offsets:
            xs_near.append(tr_real(center + d))
            xs_near.append(tr_real(center - d))

        # Far-from-pole set: values at least 0.5 away from the pole
        xs_far = [
            tr_real(v)
            for v in (-2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0)
            if abs(v - center) >= 0.5
        ]

        # Center is implicit in expr via constant subtraction

        def expr(a):
            # (x^2 + x) / (x - c)
            num = core_add(core_mul(a, a), a)
            den = core_add(a, tr_real(-float(center)))
            return core_div(num, den)

        def near_once():
            acc = tr_real(0.0)
            for a in xs_near:
                acc = core_add(acc, expr(a))
            return acc

        def far_once():
            acc = tr_real(0.0)
            for a in xs_far:
                acc = core_add(acc, expr(a))
            return acc

        res["near_pole"] = bench.benchmark(
            near_once,
            name="near_pole",
            iterations=max(100, self.iterations // 2),
            samples=self.samples,
        )
        res["far_from_pole"] = bench.benchmark(
            far_once,
            name="far_from_pole",
            iterations=max(100, self.iterations // 2),
            samples=self.samples,
        )

        self.results["near_pole"] = res
        return res

    def suite_caching(self) -> Dict[str, Any]:
        """Benchmark caching effectiveness with a simple Fibonacci example.

        Compares no-cache recursion vs memoized recursion for small n.
        """
        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        def fib_no_cache(n: int):
            if n <= 1:
                return tr_real(float(n))
            return core_add(fib_no_cache(n - 1), fib_no_cache(n - 2))

        @memoize_tr()
        def fib_cached(n: int):
            if n <= 1:
                return tr_real(float(n))
            return core_add(fib_cached(n - 1), fib_cached(n - 2))

        # Keep n modest for portability
        for n in [10, 15, 20]:
            # No cache: fewer iterations to avoid long runtimes
            res[f"fib_{n}_no_cache"] = bench.benchmark(
                lambda n=n: fib_no_cache(n),
                name=f"fibonacci_{n}_no_cache",
                iterations=max(1, self.iterations // 50),
                samples=self.samples,
            )
            # Cached: clear and run more iterations
            fib_cached.cache_clear()
            res[f"fib_{n}_cached"] = bench.benchmark(
                lambda n=n: fib_cached(n),
                name=f"fibonacci_{n}_cached",
                iterations=max(1, self.iterations // 5),
                samples=self.samples,
            )
        self.results["caching"] = res
        return res

    def suite_torch(self) -> Dict[str, Any]:
        """Optional PyTorch backend microbench (skips if torch not installed)."""
        try:
            import torch  # type: ignore
        except Exception:
            self.results["backend_torch"] = {"skipped": True}
            return self.results["backend_torch"]

        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        def torch_op():
            x = torch.randn(10000, dtype=torch.float64)
            y = torch.randn(10000, dtype=torch.float64)
            z = x * x + y
            return float(z.sum().item())

        res["torch_vec_op"] = bench.benchmark(
            torch_op,
            name="torch_vec_op",
            iterations=max(50, self.iterations // 10),
            samples=self.samples,
        )
        self.results["backend_torch"] = res
        return res

    def suite_jax(self) -> Dict[str, Any]:
        """Optional JAX backend microbench (skips if jax not installed)."""
        try:
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore
        except Exception:
            self.results["backend_jax"] = {"skipped": True}
            return self.results["backend_jax"]

        bench = TRBenchmark()
        res: Dict[str, Any] = {}

        def jax_op():
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (10000,), dtype=jnp.float64)
            y = jax.random.normal(key, (10000,), dtype=jnp.float64)
            z = x * x + y
            return float(jnp.sum(z).item())

        res["jax_vec_op"] = bench.benchmark(
            jax_op,
            name="jax_vec_op",
            iterations=max(50, self.iterations // 10),
            samples=self.samples,
        )
        self.results["backend_jax"] = res
        return res

    def suite_memory(self) -> Dict[str, Any]:
        from zeroproof.utils import profile_memory_usage

        res: Dict[str, Any] = {}

        def make_graph(n: int):
            x = TRNode.constant(zp.real(0.0))
            for i in range(n):
                x = tr_add(x, TRNode.constant(zp.real(float(i))))
            return x

        for n in [100, 1000, 5000]:
            _, mem = profile_memory_usage(make_graph, n)
            res[f"graph_{n}"] = {"memory_mb": mem}
        self.results["memory"] = res
        return res

    def save(self) -> str:
        # Use an ISO timestamp in the JSON content, but a filesystem-safe
        # variant for the filename (colons are not allowed by artifact upload).
        ts_iso = datetime.now().isoformat()
        ts_safe = ts_iso.replace(":", "-")
        out = {
            "timestamp": ts_iso,
            "results": self.results,
            "system_info": TRBenchmark()._collect_system_info(),
        }
        path = os.path.join(self.out_dir, f"bench_{ts_safe}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return path

    def write_summary(self, json_path: str) -> str:
        """Render a compact text summary next to the JSON file."""
        from zeroproof.bench_summary import _load  # reuse loader

        data = _load(json_path)
        results = data.get("results", {})
        lines = ["ZeroProof Bench Summary", "=" * 28, ""]
        for suite, entries in results.items():
            if not isinstance(entries, dict):
                continue
            lines.append(f"[{suite}]")
            for name, obj in entries.items():
                if not isinstance(obj, dict):
                    continue
                mt = obj.get("mean_time")
                ops = obj.get("operations_per_second")
                if mt is not None:
                    lines.append(f"- {name}: {float(mt)*1000:.3f} ms")
                elif ops is not None:
                    lines.append(f"- {name}: {float(ops):.0f} ops/s")
            lines.append("")
        summary_path = os.path.join(self.out_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        return summary_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ZeroProof mini benchmarks")
    ap.add_argument("--out", default="benchmark_results", help="Output directory")
    ap.add_argument(
        "--suite",
        nargs="+",
        choices=[
            "arithmetic",
            "autodiff",
            "layers",
            "scaling",
            "parallel",
            "memory",
            "caching",
            "overhead",
            "reductions",
            "tag_overhead",
            "reductions_policy",
            "near_pole",
            "torch",
            "jax",
            "all",
        ],
        default=["all"],
    )
    ap.add_argument(
        "--iterations", type=int, default=1000, help="Iterations per sample for micro-benchmarks"
    )
    ap.add_argument("--samples", type=int, default=5, help="Number of samples per benchmark")
    args = ap.parse_args(argv)

    b = MiniBench(args.out, iterations=max(1, args.iterations), samples=max(1, args.samples))
    suites = {
        "arithmetic": b.suite_arithmetic,
        "autodiff": b.suite_autodiff,
        "layers": b.suite_layers,
        "scaling": b.suite_scaling,
        "parallel": b.suite_parallel,
        "memory": b.suite_memory,
        "overhead": b.suite_overhead,
        "reductions": b.suite_reductions,
        "tag_overhead": b.suite_tag_overhead,
        "reductions_policy": b.suite_reductions_policy,
        "near_pole": b.suite_near_pole,
        "caching": b.suite_caching,
        "torch": b.suite_torch,
        "jax": b.suite_jax,
    }
    # Define a conservative default set for "all" to avoid long/fragile suites
    default_all = [
        suites["arithmetic"],
        suites["autodiff"],
        suites["layers"],
        suites["parallel"],
        suites["memory"],
        suites["overhead"],
        suites["reductions"],
        suites["tag_overhead"],
        suites["reductions_policy"],
        suites["near_pole"],
        suites["caching"],
    ]
    to_run = default_all if "all" in args.suite else [suites[s] for s in args.suite]
    for fn in to_run:
        fn()
    json_path = b.save()
    print(f"Benchmark results saved to {json_path}")
    summary_path = b.write_summary(json_path)
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
