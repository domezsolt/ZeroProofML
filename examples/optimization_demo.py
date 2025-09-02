"""
Demonstration of optimization tools for transreal computations.

This example shows how to use profiling, caching, parallel processing,
and other optimization features.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import zeroproof as zp
from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.utils import (
    TROptimizer, OptimizationConfig,
    TRProfiler, profile_tr_operation,
    memoize_tr, TRCache,
    parallel_map, ParallelConfig,
    TRBenchmark, OperationBenchmark,
)


def demonstrate_profiling():
    """Show profiling capabilities."""
    print("=== Profiling Demonstration ===\n")
    
    profiler = TRProfiler()
    
    @profiler.profile_operation("polynomial_eval")
    def evaluate_polynomial(x, coeffs):
        """Evaluate polynomial using Horner's method."""
        result = zp.real(0.0)
        for coeff in reversed(coeffs):
            result = zp.tr_mul(result, x)
            result = zp.tr_add(result, coeff)
        return result
    
    @profiler.profile_operation("naive_polynomial")
    def naive_polynomial(x, coeffs):
        """Evaluate polynomial naively."""
        result = zp.real(0.0)
        for i, coeff in enumerate(coeffs):
            power = x
            for _ in range(i):
                power = zp.tr_mul(power, x)
            term = zp.tr_mul(coeff, power) if i > 0 else coeff
            result = zp.tr_add(result, term)
        return result
    
    # Test with various polynomials
    with profiler:
        x = zp.real(2.0)
        coeffs = [zp.real(float(i)) for i in range(10)]  # 0 + 1x + 2x^2 + ... + 9x^9
        
        # Compare methods
        for _ in range(100):
            result1 = evaluate_polynomial(x, coeffs)
            result2 = naive_polynomial(x, coeffs)
    
    # Generate report
    print(profiler.generate_report())
    print()


def demonstrate_caching():
    """Show caching benefits."""
    print("=== Caching Demonstration ===\n")
    
    # Create cache
    cache = TRCache(max_size=1000)
    
    @memoize_tr(cache=cache)
    def fibonacci_tr(n):
        """Compute Fibonacci number using TR arithmetic."""
        if n <= 1:
            return zp.real(float(n))
        
        return zp.tr_add(fibonacci_tr(n-1), fibonacci_tr(n-2))
    
    # First computation (builds cache)
    start = time.time()
    result1 = fibonacci_tr(30)
    time1 = time.time() - start
    print(f"First computation: {time1:.4f}s")
    
    # Second computation (uses cache)
    start = time.time()
    result2 = fibonacci_tr(30)
    time2 = time.time() - start
    print(f"Cached computation: {time2:.4f}s")
    
    # Cache statistics
    stats = cache.get_statistics()
    print(f"\nCache statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Time saved: {stats['compute_time_saved']:.4f}s")
    print()


def demonstrate_parallel_processing():
    """Show parallel processing capabilities."""
    print("=== Parallel Processing Demonstration ===\n")
    
    def complex_computation(x):
        """Simulate complex computation."""
        result = x
        for i in range(1000):
            result = zp.tr_add(result, zp.real(0.001))
            result = zp.tr_mul(result, zp.real(1.0001))
        return result
    
    # Generate test data
    inputs = [zp.real(float(i)) for i in range(100)]
    
    # Sequential processing
    start = time.time()
    sequential_results = [complex_computation(x) for x in inputs]
    sequential_time = time.time() - start
    
    # Parallel processing
    config = ParallelConfig(backend='thread', num_workers=4)
    start = time.time()
    parallel_results = parallel_map(complex_computation, inputs, config)
    parallel_time = time.time() - start
    
    print(f"Sequential time: {sequential_time:.4f}s")
    print(f"Parallel time: {parallel_time:.4f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    print()


def demonstrate_graph_optimization():
    """Show computational graph optimization."""
    print("=== Graph Optimization Demonstration ===\n")
    
    # Create optimizer
    optimizer = TROptimizer(OptimizationConfig(
        constant_folding=True,
        common_subexpression_elimination=True,
        fuse_operations=True
    ))
    
    # Build a graph with optimization opportunities
    x = TRNode.parameter(zp.real(2.0))
    
    with gradient_tape() as tape:
        tape.watch(x)
        
        # Redundant operations
        a = zp.tr_add(x, TRNode.constant(zp.real(0.0)))  # x + 0
        b = zp.tr_mul(a, TRNode.constant(zp.real(1.0)))  # x * 1
        
        # Common subexpression
        c = zp.tr_mul(x, x)  # x^2
        d = zp.tr_mul(x, x)  # x^2 again
        
        # Combine
        e = zp.tr_add(c, d)  # 2x^2
        f = zp.tr_mul(e, b)   # 2x^2 * x = 2x^3
    
    # Optimize
    print("Before optimization:")
    print(f"  Graph nodes: {_count_nodes(f)}")
    
    optimized = optimizer.optimize(f)
    
    print("\nAfter optimization:")
    print(f"  Graph nodes: {_count_nodes(optimized)}")
    print(f"  Statistics: {optimizer.get_statistics()}")
    
    # Verify result is correct
    print(f"\nResult: {optimized.value.value} (expected: {2 * 2**3})")
    print()


def demonstrate_benchmarking():
    """Show benchmarking capabilities."""
    print("=== Benchmarking Demonstration ===\n")
    
    # Create benchmark suite
    benchmark = TRBenchmark()
    
    # Define operations to benchmark
    def add_chain(n):
        result = zp.real(0.0)
        for i in range(n):
            result = zp.tr_add(result, zp.real(1.0))
        return result
    
    def mul_chain(n):
        result = zp.real(1.0)
        for i in range(n):
            result = zp.tr_mul(result, zp.real(1.001))
        return result
    
    def mixed_ops(n):
        result = zp.real(1.0)
        for i in range(n):
            if i % 2 == 0:
                result = zp.tr_add(result, zp.real(0.1))
            else:
                result = zp.tr_mul(result, zp.real(1.01))
        return result
    
    # Benchmark operations
    n = 1000
    results = benchmark.compare(
        add_chain, mul_chain, mixed_ops,
        args=(n,),
        iterations=100,
        samples=10
    )
    
    # Print results
    print(benchmark.generate_report())
    
    # Save results
    benchmark.save_results("benchmark_results.json")
    print("\nResults saved to benchmark_results.json")
    print()


def demonstrate_special_value_optimization():
    """Show optimization with special TR values."""
    print("=== Special Value Optimization ===\n")
    
    # Operations that produce special values
    operations = [
        ("Division by zero", lambda: zp.tr_div(zp.real(1.0), zp.real(0.0))),
        ("0 * infinity", lambda: zp.tr_mul(zp.real(0.0), zp.pinf())),
        ("inf - inf", lambda: zp.tr_add(zp.pinf(), zp.ninf())),
        ("log(negative)", lambda: zp.tr_log(zp.real(-1.0))),
    ]
    
    # Profile special value handling
    profiler = TRProfiler()
    
    with profiler:
        for name, op in operations:
            @profiler.profile_operation(name)
            def wrapped_op():
                return op()
            
            result = wrapped_op()
            print(f"{name}: {result}")
    
    print(f"\nTag distribution in special operations:")
    results = profiler.get_results()
    for name, profile in results.items():
        print(f"  {name}: {profile.tag_distribution}")
    print()


def _count_nodes(root):
    """Count nodes in a computational graph."""
    visited = set()
    stack = [root]
    
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        
        visited.add(id(node))
        
        if hasattr(node, '_grad_info') and node._grad_info and hasattr(node._grad_info, 'inputs'):
            for inp_ref in node._grad_info.inputs:
                inp = inp_ref()
                if inp is not None:
                    stack.append(inp)
    
    return len(visited)


if __name__ == "__main__":
    print("ZeroProof: Optimization Tools Demo")
    print("==================================\n")
    
    demonstrate_profiling()
    demonstrate_caching()
    demonstrate_parallel_processing()
    demonstrate_graph_optimization()
    demonstrate_benchmarking()
    demonstrate_special_value_optimization()
    
    print("==================================")
    print("Optimization tools help you build")
    print("efficient transreal applications!")
