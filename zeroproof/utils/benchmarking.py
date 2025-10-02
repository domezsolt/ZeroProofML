"""
Benchmarking utilities for transreal computations.

This module provides tools for performance measurement and comparison
of transreal arithmetic operations.
"""

import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..autodiff import TRNode
from ..core import TRScalar, TRTag, ninf, phi, pinf, real

BENCHMARKING_AVAILABLE = True


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    mean_time: float  # seconds
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    samples: int
    iterations_per_sample: int
    total_iterations: int
    memory_usage_mb: Optional[float] = None
    tag_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        return self.iterations_per_sample / self.mean_time if self.mean_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "median_time": self.median_time,
            "samples": self.samples,
            "iterations_per_sample": self.iterations_per_sample,
            "total_iterations": self.total_iterations,
            "operations_per_second": self.operations_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "tag_distribution": self.tag_distribution,
            "metadata": self.metadata,
        }


class TRBenchmark:
    """Benchmark suite for transreal operations."""

    def __init__(self, warmup_iterations: int = 10):
        """
        Initialize benchmark suite.

        Args:
            warmup_iterations: Number of warmup iterations
        """
        self.warmup_iterations = warmup_iterations
        self._results: Dict[str, BenchmarkResult] = {}
        self._system_info = self._collect_system_info()

    def benchmark(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        name: Optional[str] = None,
        iterations: int = 1000,
        samples: int = 10,
        track_memory: bool = True,
    ) -> BenchmarkResult:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            name: Benchmark name
            iterations: Iterations per sample
            samples: Number of samples
            track_memory: Whether to track memory usage

        Returns:
            Benchmark result
        """
        kwargs = kwargs or {}
        name = name or func.__name__

        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)

        # Collect samples
        times = []
        tag_counts = {}

        # Memory tracking
        if PSUTIL_AVAILABLE and track_memory:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            memory_before = 0

        for _ in range(samples):
            # Time iterations
            start = time.perf_counter()

            for _ in range(iterations):
                result = func(*args, **kwargs)

                # Track tags
                if hasattr(result, "value") and hasattr(result.value, "tag"):
                    tag = result.value.tag.name
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            end = time.perf_counter()
            times.append(end - start)

        # Memory after
        if PSUTIL_AVAILABLE and track_memory:
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
        else:
            memory_used = None

        # Compute statistics
        result = BenchmarkResult(
            name=name,
            mean_time=statistics.mean(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            median_time=statistics.median(times),
            samples=samples,
            iterations_per_sample=iterations,
            total_iterations=samples * iterations,
            memory_usage_mb=memory_used,
            tag_distribution=tag_counts,
            metadata={
                "function": func.__name__,
                "args_length": len(args),
            },
        )

        self._results[name] = result
        return result

    def compare(
        self, *funcs: Callable, args: tuple = (), iterations: int = 1000, samples: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance of multiple functions.

        Args:
            funcs: Functions to compare
            args: Common arguments
            iterations: Iterations per sample
            samples: Number of samples

        Returns:
            Dictionary of results
        """
        results = {}

        for func in funcs:
            name = func.__name__
            result = self.benchmark(
                func, args=args, name=name, iterations=iterations, samples=samples
            )
            results[name] = result

        return results

    def generate_report(self) -> str:
        """Generate a text report of all benchmarks."""
        lines = ["Transreal Benchmark Report", "=" * 50, ""]

        # System info
        lines.append("System Information:")
        for key, value in self._system_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Results table
        lines.append(
            f"{'Benchmark':<30} {'Mean(ms)':>12} {'Std(ms)':>12} {'Ops/sec':>15} {'Memory(MB)':>12}"
        )
        lines.append("-" * 95)

        # Sort by mean time
        sorted_results = sorted(self._results.values(), key=lambda r: r.mean_time)

        for result in sorted_results:
            mem_str = f"{result.memory_usage_mb:.2f}" if result.memory_usage_mb else "N/A"
            lines.append(
                f"{result.name:<30} "
                f"{result.mean_time*1000:>12.3f} "
                f"{result.std_time*1000:>12.3f} "
                f"{result.operations_per_second:>15.0f} "
                f"{mem_str:>12}"
            )

        # Tag distribution
        lines.extend(["", "Tag Distribution:", "-" * 30])
        for result in sorted_results:
            if result.tag_distribution:
                lines.append(f"\n{result.name}:")
                total = sum(result.tag_distribution.values())
                for tag, count in sorted(result.tag_distribution.items()):
                    percent = count / total * 100 if total > 0 else 0
                    lines.append(f"  {tag}: {count} ({percent:.1f}%)")

        return "\n".join(lines)

    def save_results(self, filename: str):
        """Save benchmark results to JSON."""
        data = {
            "system_info": self._system_info,
            "results": {name: result.to_dict() for name, result in self._results.items()},
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def plot_comparison(self, names: Optional[List[str]] = None, output_file: Optional[str] = None):
        """
        Plot comparison of benchmark results.

        Args:
            names: Benchmark names to plot (all if None)
            output_file: Save plot to file if provided
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            import logging

            logging.getLogger(__name__).warning(
                "Plotting requires matplotlib and numpy. Install with: pip install matplotlib numpy"
            )
            return

        if not self._results:
            import logging

            logging.getLogger(__name__).info("No benchmark results to plot")
            return

        # Select results
        if names:
            results = {n: r for n, r in self._results.items() if n in names}
        else:
            results = self._results

        # Prepare data
        names = list(results.keys())
        mean_times = [r.mean_time * 1000 for r in results.values()]  # ms
        std_times = [r.std_time * 1000 for r in results.values()]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Time comparison
        x = np.arange(len(names))
        ax1.bar(x, mean_times, yerr=std_times, capsize=5)
        ax1.set_xlabel("Benchmark")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Execution Time Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")

        # Operations per second
        ops_per_sec = [r.operations_per_second for r in results.values()]
        ax2.bar(x, ops_per_sec)
        ax2.set_xlabel("Benchmark")
        ax2.set_ylabel("Operations/second")
        ax2.set_title("Throughput Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    def _collect_system_info(self) -> Dict[str, str]:
        """Collect system information."""
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu": platform.processor() or "Unknown",
        }

        if PSUTIL_AVAILABLE:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
        else:
            import os

            info["cpu_count"] = os.cpu_count() or "Unknown"
            info["memory_gb"] = "Unknown"

        return info

    def _format_time(self, seconds: float) -> str:
        """Format time for display."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:.1f}ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.1f}µs"
        elif seconds < 1:
            return f"{seconds * 1e3:.1f}ms"
        else:
            return f"{seconds:.3f}s"


class OperationBenchmark:
    """Specialized benchmarks for transreal operations."""

    def __init__(self):
        self.benchmark = TRBenchmark()

    def benchmark_arithmetic(self) -> Dict[str, BenchmarkResult]:
        """Benchmark basic arithmetic operations."""
        from ..core import tr_add, tr_div, tr_mul, tr_sub

        # Test values
        a = real(3.14159)
        b = real(2.71828)

        results = {}

        # Addition
        results["add_real"] = self.benchmark.benchmark(
            lambda: tr_add(a, b), name="add_real", iterations=10000
        )

        # Subtraction
        results["sub_real"] = self.benchmark.benchmark(
            lambda: tr_sub(a, b), name="sub_real", iterations=10000
        )

        # Multiplication
        results["mul_real"] = self.benchmark.benchmark(
            lambda: tr_mul(a, b), name="mul_real", iterations=10000
        )

        # Division
        results["div_real"] = self.benchmark.benchmark(
            lambda: tr_div(a, b), name="div_real", iterations=10000
        )

        # Special cases
        inf = pinf()
        zero = real(0.0)

        results["mul_zero_inf"] = self.benchmark.benchmark(
            lambda: tr_mul(zero, inf), name="mul_zero_inf", iterations=10000
        )

        results["div_by_zero"] = self.benchmark.benchmark(
            lambda: tr_div(a, zero), name="div_by_zero", iterations=10000
        )

        return results

    def benchmark_autodiff(self) -> Dict[str, BenchmarkResult]:
        """Benchmark autodiff operations."""
        from ..autodiff import gradient_tape
        from ..core import tr_add, tr_mul

        results = {}

        # Simple gradient
        def simple_gradient():
            x = TRNode.parameter(real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = tr_mul(x, x)
            return tape.gradient(y, x)

        results["simple_gradient"] = self.benchmark.benchmark(
            simple_gradient, name="simple_gradient", iterations=1000
        )

        # Chain of operations
        def chain_gradient():
            x = TRNode.parameter(real(2.0))
            with gradient_tape() as tape:
                tape.watch(x)
                y = x
                for _ in range(10):
                    y = tr_add(tr_mul(y, x), real(1.0))
            return tape.gradient(y, x)

        results["chain_gradient"] = self.benchmark.benchmark(
            chain_gradient, name="chain_gradient", iterations=100
        )

        return results

    def benchmark_special_values(self) -> Dict[str, BenchmarkResult]:
        """Benchmark operations with special values."""
        from ..core import tr_add, tr_div, tr_mul

        results = {}

        # Create special values
        real_val = real(1.0)
        pos_inf = pinf()
        neg_inf = ninf()
        null = phi()

        # Test combinations
        special_ops = [
            ("inf_plus_inf", lambda: tr_add(pos_inf, pos_inf)),
            ("inf_minus_inf", lambda: tr_add(pos_inf, neg_inf)),
            ("inf_times_zero", lambda: tr_mul(pos_inf, real(0.0))),
            ("inf_div_inf", lambda: tr_div(pos_inf, pos_inf)),
            ("phi_propagation", lambda: tr_add(null, real_val)),
        ]

        for name, op in special_ops:
            results[name] = self.benchmark.benchmark(op, name=name, iterations=10000)

        return results


def create_scaling_benchmark(
    operation: Callable, sizes: List[int], name: str = "scaling"
) -> Dict[int, BenchmarkResult]:
    """
    Benchmark operation scaling with input size.

    Args:
        operation: Operation to benchmark (takes size as argument)
        sizes: List of input sizes
        name: Base name for benchmarks

    Returns:
        Dictionary mapping size to result
    """
    benchmark = TRBenchmark()
    results = {}

    for size in sizes:
        result = benchmark.benchmark(
            lambda: operation(size),
            name=f"{name}_{size}",
            iterations=max(1, 1000 // size),  # Adjust iterations
            samples=10,
        )
        results[size] = result

    return results


def profile_memory_usage(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Profile memory usage of a function.

    Args:
        func: Function to profile
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (result, memory_used_mb)
    """
    import gc

    gc.collect()

    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
    else:
        result = func(*args, **kwargs)
        memory_used = 0.0  # Can't measure without psutil

    return result, memory_used
