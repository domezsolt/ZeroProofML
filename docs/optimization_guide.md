# Optimization Guide for ZeroProof

## Overview

ZeroProof provides comprehensive optimization tools to help you build high-performance transreal applications. This guide covers profiling, caching, parallel processing, graph optimization, and benchmarking.

## Table of Contents

1. [Profiling](#profiling)
2. [Caching](#caching)
3. [Parallel Processing](#parallel-processing)
4. [Graph Optimization](#graph-optimization)
5. [Benchmarking](#benchmarking)
6. [Best Practices](#best-practices)

## Profiling

### Basic Profiling

Use the `TRProfiler` to measure performance:

```python
from zeroproof.utils import TRProfiler

profiler = TRProfiler(trace_memory=True)

@profiler.profile_operation("my_computation")
def compute(x, y):
    return tr_mul(tr_add(x, y), tr_sub(x, y))

with profiler:
    result = compute(real(3.0), real(2.0))

print(profiler.generate_report())
```

### Profile Decorators

Quick profiling with decorators:

```python
from zeroproof.utils import profile_tr_operation, memory_profile

@profile_tr_operation("matrix_multiply")
def matrix_op(a, b):
    # Your computation
    pass

@memory_profile
def memory_intensive():
    # Will print memory usage
    pass
```

### Tag Statistics

Analyze tag distributions in computations:

```python
from zeroproof.utils import tag_statistics

nodes = [...]  # Your computation nodes
stats = tag_statistics(nodes)

print(f"Total nodes: {stats['total']}")
print(f"Tag distribution: {stats['by_tag']}")
print(f"Non-REAL operations: {len(stats['non_real_operations'])}")
```

## Caching

### Basic Memoization

Cache computation results:

```python
from zeroproof.utils import memoize_tr

@memoize_tr()
def expensive_computation(x, n):
    result = x
    for i in range(n):
        result = tr_add(tr_mul(result, real(1.01)), real(i))
    return result

# First call computes
result1 = expensive_computation(real(1.0), 100)

# Second call uses cache
result2 = expensive_computation(real(1.0), 100)
```

### Custom Cache Configuration

```python
from zeroproof.utils import TRCache, memoize_tr

# Configure cache
cache = TRCache(
    max_size=10000,         # Maximum entries
    max_memory_mb=100,      # Maximum memory usage
    eviction_policy='lru',  # or 'lfu', 'fifo'
    ttl_seconds=3600        # Time to live
)

@memoize_tr(cache=cache)
def my_function(x):
    return expensive_operation(x)

# Get cache statistics
stats = cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory used: {stats['memory_used_mb']:.2f} MB")
```

### Operation-Specific Caching

```python
from zeroproof.utils import get_operation_cache

op_cache = get_operation_cache()

# Cache specific operations
a, b = real(2.0), real(3.0)
result = tr_add(a, b)
op_cache.cache_add(a, b, result)

# Retrieve from cache
cached = op_cache.get_add(a, b)
```

## Parallel Processing

### Basic Parallel Map

```python
from zeroproof.utils import parallel_map, ParallelConfig

def process_item(x):
    # Complex computation
    return tr_mul(x, x)

inputs = [real(float(i)) for i in range(1000)]

# Configure parallel execution
config = ParallelConfig(
    num_workers=4,
    backend='thread',  # or 'process'
    chunk_size=100
)

results = parallel_map(process_item, inputs, config)
```

### Thread vs Process Pools

```python
from zeroproof.utils import TRThreadPool, TRProcessPool

# Thread pool (shared memory, good for I/O bound)
with TRThreadPool(num_workers=4) as pool:
    results = pool.map(compute, inputs)

# Process pool (separate memory, good for CPU bound)
with TRProcessPool(num_workers=4) as pool:
    results = pool.map(compute, inputs)
```

### Parallel Computation with Dependencies

```python
from zeroproof.utils import ParallelTRComputation

comp = ParallelTRComputation()

# Add tasks with dependencies
comp.add_task('load_data', load_data)
comp.add_task('preprocess', preprocess, depends_on=['load_data'])
comp.add_task('compute_a', compute_a, depends_on=['preprocess'])
comp.add_task('compute_b', compute_b, depends_on=['preprocess'])
comp.add_task('combine', combine, depends_on=['compute_a', 'compute_b'])

# Execute respecting dependencies
results = comp.execute()
```

### Vectorized Operations

```python
from zeroproof.utils import vectorize_operation

@vectorize_operation
def my_operation(x):
    return tr_add(tr_mul(x, x), real(1.0))

# Works with single values
result = my_operation(real(2.0))

# Automatically parallelizes for lists
results = my_operation([real(i) for i in range(100)])

# Works with NumPy arrays if available
import numpy as np
arr = np.array([real(i) for i in range(100)])
results = my_operation(arr)
```

## Graph Optimization

### Automatic Optimization

```python
from zeroproof.utils import optimize_tr_graph, OptimizationConfig

config = OptimizationConfig(
    constant_folding=True,
    common_subexpression_elimination=True,
    fuse_operations=True,
    eliminate_dead_code=True
)

# Build computation graph
x = TRNode.parameter(real(2.0))
with gradient_tape() as tape:
    tape.watch(x)
    y = tr_add(x, real(0))  # x + 0
    z = tr_mul(y, real(1))  # x * 1
    w = tr_add(z, z)        # 2x

# Optimize
optimized = optimize_tr_graph(w, config)
```

### Custom Optimization Rules

```python
from zeroproof.utils import GraphOptimizer

optimizer = GraphOptimizer()

# Add custom rule
def is_identity_add(node):
    # Check if node is x + 0
    return (node._grad_info and 
            node._grad_info.op_type == OpType.ADD and
            any(inp.value == real(0.0) for inp in get_inputs(node)))

def optimize_identity_add(node):
    # Return non-zero input
    for inp in get_inputs(node):
        if inp.value != real(0.0):
            return inp
    return node

optimizer.add_rule(is_identity_add, optimize_identity_add)
```

### Operation Fusion

```python
from zeroproof.utils import OperationFuser

fuser = OperationFuser()

# Add fusion pattern for linear operations
def detect_linear(nodes):
    # Detect a*x + b pattern
    return len(nodes) == 2 and is_mul(nodes[0]) and is_add(nodes[1])

def fuse_linear(nodes):
    # Create fused linear operation
    return LinearOp(nodes[0], nodes[1])

fuser.add_pattern('linear', detect_linear, fuse_linear)

# Fuse operations
fused_nodes = fuser.fuse(computation_nodes)
```

## Benchmarking

### Basic Benchmarking

```python
from zeroproof.utils import TRBenchmark

benchmark = TRBenchmark()

# Benchmark a function
result = benchmark.benchmark(
    my_function,
    args=(arg1, arg2),
    iterations=1000,
    samples=10
)

print(f"Mean time: {result.mean_time:.4f}s")
print(f"Ops/sec: {result.operations_per_second:,.0f}")
```

### Training Bench Metrics

Hybrid training records per‑epoch timing metrics that help you pinpoint bottlenecks without custom instrumentation.

- Metrics per epoch:
  - `avg_step_ms`: average time per batch (ms)
  - `data_time_ms`: average time spent on data preparation per batch (ms)
  - `optim_time_ms`: average optimizer/step time per batch (ms)
  - `batches`: number of batches in the epoch
- Access: present in training summaries as `bench_history` when using `HybridTRTrainer` (and surfaced by higher‑level drivers).
- Logging cadence: controlled by `log_interval` in trainer config (robotics example exposes `--log_every`).

### Comparing Implementations

```python
# Compare different implementations
results = benchmark.compare(
    implementation1,
    implementation2,
    implementation3,
    args=(test_input,),
    iterations=1000
)

# Generate comparison report
print(benchmark.generate_report())

# Plot results
benchmark.plot_comparison(output_file="comparison.png")
```

### Scaling Analysis

```python
from zeroproof.utils import create_scaling_benchmark

def algorithm(n):
    # Algorithm that scales with n
    result = real(0)
    for i in range(n):
        result = tr_add(result, real(i))
    return result

# Test scaling
results = create_scaling_benchmark(
    algorithm,
    sizes=[10, 100, 1000, 10000],
    name="sum_scaling"
)

# Analyze scaling behavior
for size, result in results.items():
    print(f"n={size}: {result.mean_time:.4f}s")
```

### Memory Profiling

```python
from zeroproof.utils import profile_memory_usage

def memory_intensive():
    nodes = []
    for i in range(10000):
        nodes.append(TRNode.constant(real(i)))
    return nodes

result, memory_mb = profile_memory_usage(memory_intensive)
print(f"Memory used: {memory_mb:.2f} MB")
```

## Best Practices

### 1. Profile Before Optimizing

Always profile first to identify bottlenecks:

```python
with TRProfiler() as profiler:
    # Your application code
    pass

# Analyze where time is spent
report = profiler.generate_report()
```

### 2. Use Caching Wisely

- Cache expensive, deterministic computations
- Monitor cache hit rates
- Set appropriate cache sizes and TTLs
- Clear caches when needed

```python
@memoize_tr()
def expensive_but_deterministic(x):
    # Good candidate for caching
    pass

# Monitor cache performance
print(expensive_but_deterministic.cache_info())
```

### 3. Choose the Right Parallelization

- **Thread pool**: Good for I/O-bound tasks, shared memory
- **Process pool**: Good for CPU-bound tasks, true parallelism
- **Vectorization**: Best for element-wise operations

```python
# I/O bound - use threads
config = ParallelConfig(backend='thread')

# CPU bound - use processes
config = ParallelConfig(backend='process')

# Element-wise - use vectorization
@vectorize_operation
def element_op(x):
    return tr_mul(x, x)
```

### 4. Optimize Computational Graphs

For complex computations with autodiff:

```python
# Build graph once
optimizer = TROptimizer()
optimized_graph = optimizer.optimize(original_graph)

# Reuse optimized graph
for data in dataset:
    result = evaluate(optimized_graph, data)
```

### 5. Monitor Special Values

Track the distribution of special values:

```python
stats = tag_statistics(computation_nodes)
if stats['percentages']['PHI'] > 10:
    print("Warning: High percentage of PHI values")
```

### 6. Batch Operations

Process data in batches for better performance:

```python
from zeroproof.utils import batch_tr_operation

# Process in batches of 1000
results = batch_tr_operation(
    operation,
    inputs,
    batch_size=1000,
    parallel=True
)

### 7. Prefer Pairwise (Tree) Reductions

When aggregating many terms (e.g., per‑sample losses), avoid linear left‑fold
accumulation like `total = total + term`. This builds deep computation graphs
that can degrade numerical behavior and stress the backward pass (Python
recursion limits).

Instead, reduce via a balanced pairwise tree:

```python
def pairwise_sum(nodes: list[TRNode]) -> TRNode:
    if not nodes:
        return TRNode.constant(real(0.0))
    if len(nodes) == 1:
        return nodes[0]
    mid = len(nodes) // 2
    return pairwise_sum(nodes[:mid]) + pairwise_sum(nodes[mid:])

avg_loss = pairwise_sum(sample_losses) / TRNode.constant(real(float(len(sample_losses))))
```

ZeroProof’s trainer uses a pairwise reduction for loss aggregation and honors
policy‑controlled deterministic reductions for numeric determinism.
```

### 7. Use Mixed Precision

For large computations, consider precision trade-offs:

```python
from zeroproof.bridge import MixedPrecisionStrategy

strategy = MixedPrecisionStrategy(
    compute_precision=Precision.FLOAT32,     # Fast
    accumulate_precision=Precision.FLOAT64,  # Accurate
    output_precision=Precision.FLOAT32       # Compact
)
```

## Performance Tips

1. **Minimize graph construction overhead**: Build graphs once and reuse
2. **Use operation fusion**: Combine multiple operations into one
3. **Leverage parallelism**: Use all available cores
4. **Cache aggressively**: But monitor memory usage
5. **Profile regularly**: Performance characteristics can change
6. **Handle special values early**: Prevent PHI propagation
7. **Batch small operations**: Reduce function call overhead

## Example: Optimized Application

Here's an example combining multiple optimization techniques:

```python
from zeroproof.utils import (
    TRProfiler, memoize_tr, parallel_map,
    optimize_tr_graph, TRBenchmark
)

# Profile the entire application
profiler = TRProfiler()

# Cache expensive computations
@memoize_tr()
@profiler.profile_operation("compute_features")
def compute_features(data):
    # Feature computation
    pass

# Optimize computational graphs
@profiler.profile_operation("build_model")
def build_optimized_model():
    with gradient_tape() as tape:
        # Build graph
        pass
    
    # Optimize
    return optimize_tr_graph(model)

# Parallel data processing
@profiler.profile_operation("process_batch")
def process_batch(batch):
    config = ParallelConfig(backend='thread', num_workers=4)
    return parallel_map(compute_features, batch, config)

# Main application
with profiler:
    model = build_optimized_model()
    
    for batch in data_loader:
        features = process_batch(batch)
        results = model(features)

# Analyze performance
print(profiler.generate_report())

# Benchmark different configurations
benchmark = TRBenchmark()
benchmark.compare(
    lambda: process_batch(test_batch),
    lambda: process_batch_v2(test_batch),
    iterations=100
)
```

This comprehensive approach ensures your transreal applications run efficiently while maintaining numerical stability and correctness.
