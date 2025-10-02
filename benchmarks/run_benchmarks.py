"""
Comprehensive benchmark suite for ZeroProof.

This script runs various benchmarks to measure performance characteristics
of transreal arithmetic operations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from datetime import datetime

import numpy as np

import zeroproof as zp
from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.core import TRScalar
from zeroproof.layers import TRNorm, TRRational
from zeroproof.utils import (
    OperationBenchmark,
    ParallelConfig,
    TRBenchmark,
    create_scaling_benchmark,
    memoize_tr,
    parallel_map,
)


class ComprehensiveBenchmarks:
    """Run comprehensive benchmarks for ZeroProof."""

    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def run_arithmetic_benchmarks(self):
        """Benchmark basic arithmetic operations."""
        print("\n=== Arithmetic Benchmarks ===")

        benchmark = OperationBenchmark()
        results = benchmark.benchmark_arithmetic()

        # Additional special case benchmarks
        special_benchmark = TRBenchmark()

        # Overflow handling
        def overflow_test():
            return zp.tr_mul(zp.real(1e200), zp.real(1e200))

        results["overflow_handling"] = special_benchmark.benchmark(
            overflow_test, name="overflow_handling", iterations=10000
        )

        # Underflow handling
        def underflow_test():
            return zp.tr_div(zp.real(1e-200), zp.real(1e200))

        results["underflow_handling"] = special_benchmark.benchmark(
            underflow_test, name="underflow_handling", iterations=10000
        )

        self.results["arithmetic"] = results
        return results

    def run_autodiff_benchmarks(self):
        """Benchmark autodiff operations."""
        print("\n=== Autodiff Benchmarks ===")

        benchmark = TRBenchmark()
        results = {}

        # Simple derivative
        def simple_derivative():
            x = TRNode.parameter(zp.real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = zp.tr_mul(x, x)  # x^2
            return tape.gradient(y, x)

        results["simple_derivative"] = benchmark.benchmark(
            simple_derivative, name="simple_derivative", iterations=1000
        )

        # Chain rule
        def chain_rule():
            x = TRNode.parameter(zp.real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = zp.tr_mul(x, x)  # x^2
                z = zp.tr_add(y, x)  # x^2 + x
                w = zp.tr_mul(z, z)  # (x^2 + x)^2
            return tape.gradient(w, x)

        results["chain_rule"] = benchmark.benchmark(chain_rule, name="chain_rule", iterations=500)

        # Multiple parameters
        def multi_param():
            x = TRNode.parameter(zp.real(2.0))
            y = TRNode.parameter(zp.real(3.0))
            with gradient_tape() as tape:
                tape.watch(x)
                tape.watch(y)
                z = zp.tr_add(zp.tr_mul(x, y), zp.tr_div(x, y))
            return tape.gradient(z, [x, y])

        results["multi_param_grad"] = benchmark.benchmark(
            multi_param, name="multi_param_grad", iterations=500
        )

        self.results["autodiff"] = results
        return results

    def run_layer_benchmarks(self):
        """Benchmark neural network layers."""
        print("\n=== Layer Benchmarks ===")

        benchmark = TRBenchmark()
        results = {}

        # TR-Rational layer
        rational = TRRational(d_p=3, d_q=2)

        def rational_forward():
            x = TRNode.constant(zp.real(1.5))
            return rational.forward(x)

        results["rational_forward"] = benchmark.benchmark(
            rational_forward, name="rational_forward", iterations=1000
        )

        # TR-Norm layer
        norm = TRNorm(num_features=10)

        def norm_forward():
            batch = [[TRNode.constant(zp.real(float(i + j))) for j in range(10)] for i in range(32)]
            return norm.forward(batch)

        results["norm_forward"] = benchmark.benchmark(
            norm_forward, name="norm_forward_batch32", iterations=100
        )

        self.results["layers"] = results
        return results

    def run_scaling_benchmarks(self):
        """Benchmark scaling characteristics."""
        print("\n=== Scaling Benchmarks ===")

        results = {}

        # Graph depth scaling
        def deep_graph(depth):
            x = TRNode.parameter(zp.real(1.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = x
                for _ in range(depth):
                    y = zp.tr_add(zp.tr_mul(y, x), zp.real(1.0))
            return tape.gradient(y, x)

        results["graph_depth"] = create_scaling_benchmark(
            deep_graph, sizes=[10, 20, 50, 100], name="graph_depth"
        )

        # Batch size scaling
        def batch_operation(size):
            inputs = [zp.real(float(i)) for i in range(size)]
            outputs = []
            for x in inputs:
                y = zp.tr_mul(x, x)
                z = zp.tr_add(y, zp.real(1.0))
                outputs.append(z)
            return outputs

        results["batch_size"] = create_scaling_benchmark(
            batch_operation, sizes=[10, 100, 1000, 10000], name="batch_size"
        )

        self.results["scaling"] = results
        return results

    def run_parallel_benchmarks(self):
        """Benchmark parallel processing."""
        print("\n=== Parallel Processing Benchmarks ===")

        benchmark = TRBenchmark()
        results = {}

        def work_function(x):
            # Simulate some work
            result = x
            for _ in range(100):
                result = zp.tr_add(result, zp.real(0.1))
                result = zp.tr_mul(result, zp.real(1.001))
            return result

        inputs = [zp.real(float(i)) for i in range(1000)]

        # Sequential
        def sequential():
            return [work_function(x) for x in inputs]

        results["sequential"] = benchmark.benchmark(
            sequential, name="sequential_1000", iterations=1, samples=5
        )

        # Parallel with different worker counts
        for num_workers in [2, 4, 8]:
            config = ParallelConfig(backend="thread", num_workers=num_workers)

            def parallel():
                return parallel_map(work_function, inputs, config)

            results[f"parallel_{num_workers}"] = benchmark.benchmark(
                parallel, name=f"parallel_{num_workers}_workers", iterations=1, samples=5
            )

        self.results["parallel"] = results
        return results

    def run_memory_benchmarks(self):
        """Benchmark memory usage patterns."""
        print("\n=== Memory Benchmarks ===")

        from zeroproof.utils import profile_memory_usage

        results = {}

        # Large graph creation
        def create_large_graph(n):
            nodes = []
            x = TRNode.parameter(zp.real(1.0))
            nodes.append(x)

            for i in range(n):
                y = zp.tr_add(x, TRNode.constant(zp.real(float(i))))
                nodes.append(y)
                x = y

            return nodes

        for size in [100, 1000, 10000]:
            _, memory = profile_memory_usage(create_large_graph, size)
            results[f"graph_{size}_nodes"] = {
                "memory_mb": memory,
                "nodes": size,
                "mb_per_node": memory / size if size > 0 else 0,
            }

        self.results["memory"] = results
        return results

    def run_caching_benchmarks(self):
        """Benchmark caching effectiveness."""
        print("\n=== Caching Benchmarks ===")

        benchmark = TRBenchmark()
        results = {}

        # Fibonacci without cache
        def fib_no_cache(n):
            if n <= 1:
                return zp.real(float(n))
            return zp.tr_add(fib_no_cache(n - 1), fib_no_cache(n - 2))

        # Fibonacci with cache
        @memoize_tr()
        def fib_cached(n):
            if n <= 1:
                return zp.real(float(n))
            return zp.tr_add(fib_cached(n - 1), fib_cached(n - 2))

        # Benchmark different sizes
        for n in [10, 15, 20, 25]:
            # Without cache
            results[f"fib_{n}_no_cache"] = benchmark.benchmark(
                lambda: fib_no_cache(n), name=f"fibonacci_{n}_no_cache", iterations=1, samples=3
            )

            # With cache (clear cache first)
            fib_cached.cache_clear()
            results[f"fib_{n}_cached"] = benchmark.benchmark(
                lambda: fib_cached(n), name=f"fibonacci_{n}_cached", iterations=10, samples=3
            )

        self.results["caching"] = results
        return results

    def save_results(self):
        """Save all benchmark results."""
        timestamp = datetime.now().isoformat()

        output = {
            "timestamp": timestamp,
            "results": self.results,
            "system_info": TRBenchmark()._collect_system_info(),
        }

        filename = os.path.join(self.output_dir, f"benchmarks_{timestamp}.json")
        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to {filename}")

        # Also save a summary
        self._save_summary()

    def _save_summary(self):
        """Save a human-readable summary."""
        summary_file = os.path.join(self.output_dir, "summary.txt")

        with open(summary_file, "w") as f:
            f.write("ZeroProof Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")

            # Arithmetic operations
            if "arithmetic" in self.results:
                f.write("Arithmetic Operations (ops/sec):\n")
                for name, result in self.results["arithmetic"].items():
                    if hasattr(result, "operations_per_second"):
                        f.write(f"  {name}: {result.operations_per_second:,.0f}\n")
                f.write("\n")

            # Autodiff operations
            if "autodiff" in self.results:
                f.write("Autodiff Operations (ms/op):\n")
                for name, result in self.results["autodiff"].items():
                    if hasattr(result, "mean_time"):
                        f.write(f"  {name}: {result.mean_time * 1000:.3f}\n")
                f.write("\n")

            # Parallel speedup
            if "parallel" in self.results:
                seq = self.results["parallel"].get("sequential")
                if seq and hasattr(seq, "mean_time"):
                    f.write("Parallel Speedup:\n")
                    seq_time = seq.mean_time
                    for name, result in self.results["parallel"].items():
                        if "parallel" in name and hasattr(result, "mean_time"):
                            speedup = seq_time / result.mean_time
                            f.write(f"  {name}: {speedup:.2f}x\n")
                    f.write("\n")

        print(f"Summary saved to {summary_file}")


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="Run ZeroProof benchmarks")
    parser.add_argument(
        "--output", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
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
            "all",
        ],
        default=["all"],
        help="Benchmark suites to run",
    )

    args = parser.parse_args()

    print("ZeroProof Comprehensive Benchmarks")
    print("==================================")

    benchmarks = ComprehensiveBenchmarks(args.output)

    suites = {
        "arithmetic": benchmarks.run_arithmetic_benchmarks,
        "autodiff": benchmarks.run_autodiff_benchmarks,
        "layers": benchmarks.run_layer_benchmarks,
        "scaling": benchmarks.run_scaling_benchmarks,
        "parallel": benchmarks.run_parallel_benchmarks,
        "memory": benchmarks.run_memory_benchmarks,
        "caching": benchmarks.run_caching_benchmarks,
    }

    if "all" in args.suite:
        to_run = list(suites.values())
    else:
        to_run = [suites[name] for name in args.suite]

    for func in to_run:
        func()

    benchmarks.save_results()

    print("\n==================================")
    print("Benchmarking complete!")


if __name__ == "__main__":
    main()
