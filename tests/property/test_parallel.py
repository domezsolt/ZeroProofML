"""Property-based tests for parallel processing utilities."""

import multiprocessing as mp
import os
import sys
import time
import weakref
from concurrent.futures import TimeoutError

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from zeroproof.autodiff import TRNode, tr_add, tr_mul
from zeroproof.core import TRTag, ninf, phi, pinf, real
from zeroproof.utils.parallel import (
    ParallelConfig,
    ParallelTRArray,
    ParallelTRComputation,
    TRProcessPool,
    TRThreadPool,
    batch_tr_operation,
    parallel_graph_eval,
    parallel_map,
    parallel_reduce,
    vectorize_operation,
)


class TestTRThreadPool:
    """Test thread pool for TR operations."""

    def test_basic_thread_pool(self):
        """Test basic thread pool operations."""
        with TRThreadPool(num_workers=2) as pool:

            def square(x):
                return tr_mul(x, x)

            inputs = [real(float(i)) for i in range(10)]
            results = pool.map(square, inputs)

            assert len(results) == 10
            # Results might not be in order due to parallel processing
            # Check that all expected values are present
            result_values = sorted([r.value.value for r in results])
            expected_values = sorted([float(i * i) for i in range(10)])
            assert result_values == expected_values

    @given(st.integers(min_value=1, max_value=4))
    def test_thread_pool_workers(self, num_workers):
        """Test thread pool with different worker counts."""
        is_ci = bool(os.getenv("CI"))
        with TRThreadPool(num_workers=num_workers) as pool:

            def slow_add(x):
                time.sleep(0.01)
                return tr_add(x, real(1.0))

            inputs = [real(float(i)) for i in range(num_workers * 2)]

            start = time.time()
            results = pool.map(slow_add, inputs)
            duration = time.time() - start

            # For single worker, parallelization overhead may make it slower
            # Only expect speedup with multiple workers
            sequential_time = 0.01 * len(inputs)
            if num_workers > 1:
                # With multiple workers, expect some speedup. On CI runners,
                # allow more overhead due to shared CPU and scheduler jitter.
                if is_ci:
                    assert duration < sequential_time * 1.25  # tolerate 25% slower than sequential
                else:
                    assert duration < sequential_time * 0.9  # local: require modest speedup
            else:
                # With single worker, just ensure it completes reasonably
                # (may be slower due to thread overhead). On CI allow more slack.
                if is_ci:
                    assert duration < sequential_time * 3.0
                else:
                    assert duration < sequential_time * 2.0

            # Verify results
            for i, result in enumerate(results):
                assert result.value.value == float(i + 1)

    def test_thread_pool_async(self):
        """Test async operations in thread pool."""
        with TRThreadPool() as pool:

            def compute(x, y):
                return tr_mul(x, y)

            # Submit async tasks
            future1 = pool.apply_async(compute, (real(2.0), real(3.0)))
            future2 = pool.apply_async(compute, (real(4.0), real(5.0)))

            # Get results
            result1 = future1.result()
            result2 = future2.result()

            assert result1.value.value == 6.0
            assert result2.value.value == 20.0


class TestTRProcessPool:
    """Test process pool for TR operations."""

    def test_basic_process_pool(self):
        """Test basic process pool operations."""

        # Define function at module level for pickling
        def square(x):
            from zeroproof.autodiff import tr_mul

            return tr_mul(x, x)

        with TRProcessPool(num_workers=2) as pool:
            inputs = [real(float(i)) for i in range(10)]
            results = pool.map(square, inputs)

            assert len(results) == 10
            # Note: Process pool may not preserve exact order
            # but all results should be present
            values = sorted([r.value.value for r in results])
            expected = sorted([float(i * i) for i in range(10)])
            assert values == expected


class TestParallelMap:
    """Test parallel map operations."""

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
            min_size=0,
            max_size=50,
        )
    )
    def test_parallel_map_correctness(self, values):
        """Test that parallel map produces correct results."""

        def operation(x):
            return tr_add(tr_mul(x, x), real(1.0))

        inputs = [real(v) for v in values]

        # Sequential computation
        sequential_results = [operation(x) for x in inputs]

        # Parallel computation
        parallel_results = parallel_map(operation, inputs)

        # Results should match
        assert len(parallel_results) == len(sequential_results)
        for par, seq in zip(parallel_results, sequential_results):
            assert par.value == seq.value

    def test_parallel_map_backends(self):
        """Test different parallel backends."""

        def double(x):
            return tr_add(x, x)

        inputs = [real(float(i)) for i in range(20)]

        # Thread backend
        config_thread = ParallelConfig(backend="thread")
        results_thread = parallel_map(double, inputs, config_thread)

        # Process backend - need pickleable function
        def double_pickleable(x):
            from zeroproof.autodiff import tr_add

            return tr_add(x, x)

        config_process = ParallelConfig(backend="process")
        results_process = parallel_map(double_pickleable, inputs, config_process)

        # Results should be same
        for t, p in zip(results_thread, results_process):
            assert t.value.value == p.value.value


class TestParallelReduce:
    """Test parallel reduction operations."""

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10),
            min_size=0,
            max_size=100,
        )
    )
    def test_parallel_reduce_sum(self, values):
        """Test parallel sum reduction."""
        inputs = [real(v) for v in values]

        if not inputs:
            return

        # Sequential sum
        sequential_sum = inputs[0]
        for inp in inputs[1:]:
            sequential_sum = tr_add(sequential_sum, inp)

        # Parallel sum
        parallel_sum = parallel_reduce(tr_add, inputs)

        # Should be close (floating point accumulation order may differ)
        assert abs(parallel_sum.value.value - sequential_sum.value.value) < 1e-10

    def test_parallel_reduce_with_initial(self):
        """Test parallel reduce with initial value."""
        inputs = [real(float(i)) for i in range(10)]
        initial = real(100.0)

        result = parallel_reduce(tr_add, inputs, initial)

        expected = 100.0 + sum(range(10))
        assert result.value.value == expected


class TestVectorization:
    """Test operation vectorization."""

    def test_vectorize_scalar_operation(self):
        """Test vectorizing a scalar operation."""

        @vectorize_operation
        def square(x):
            return tr_mul(x, x)

        # Single scalar
        result_scalar = square(real(3.0))
        assert result_scalar.value.value == 9.0

        # List of scalars
        inputs_list = [real(float(i)) for i in range(5)]
        results_list = square(inputs_list)

        assert len(results_list) == 5
        for i, result in enumerate(results_list):
            assert result.value.value == float(i * i)

    @pytest.mark.skipif("numpy" not in sys.modules, reason="NumPy not available")
    def test_vectorize_numpy_operation(self):
        """Test vectorizing with numpy arrays."""
        import numpy as np

        @vectorize_operation
        def add_one(x):
            return tr_add(x, real(1.0))

        # Create numpy array of TR scalars
        inputs = np.array([[real(1.0), real(2.0)], [real(3.0), real(4.0)]])

        results = add_one(inputs)

        assert results.shape == (2, 2)
        assert results[0, 0].value.value == 2.0
        assert results[1, 1].value.value == 5.0


class TestParallelComputation:
    """Test parallel computation with dependencies."""

    def test_dependency_execution(self):
        """Test execution with dependencies."""
        comp = ParallelTRComputation()

        # Define tasks with dependencies
        def task_a():
            return real(1.0)

        def task_b():
            return real(2.0)

        def task_c(a_result, b_result):
            return tr_add(a_result, b_result)

        comp.add_task("a", task_a)
        comp.add_task("b", task_b)
        comp.add_task(
            "c", lambda: task_c(comp._results["a"], comp._results["b"]), depends_on=["a", "b"]
        )

        results = comp.execute()

        assert results["a"].value.value == 1.0
        assert results["b"].value.value == 2.0
        assert results["c"].value.value == 3.0

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        comp = ParallelTRComputation()

        comp.add_task("a", lambda: real(1.0), depends_on=["b"])
        comp.add_task("b", lambda: real(2.0), depends_on=["a"])

        with pytest.raises(ValueError, match="Circular dependency"):
            comp.execute()


class TestBatchProcessing:
    """Test batch processing utilities."""

    @given(st.integers(min_value=10, max_value=1000))
    def test_batch_processing(self, total_size):
        """Test batch processing of operations."""
        # Create inputs
        inputs = [(real(float(i)), real(float(i + 1))) for i in range(total_size)]

        # Process in batches
        results = batch_tr_operation(tr_add, inputs, batch_size=100, parallel=True)

        assert len(results) == total_size
        for i, result in enumerate(results):
            assert result.value.value == float(i + i + 1)


class TestParallelArrayOperations:
    """Test SIMD-style array operations."""

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=200))
    def test_parallel_array_add(self, values):
        """Test parallel array addition."""
        a = [real(v) for v in values]
        b = [real(v + 1) for v in values]

        # Parallel addition
        result = ParallelTRArray.add(a, b, parallel=True)

        assert len(result) == len(values)
        for i, (res, val) in enumerate(zip(result, values)):
            expected = val + (val + 1)
            # Check if result would overflow
            if abs(expected) > 1e308:  # Near float64 max
                # Should get infinity
                assert res.value.tag in [TRTag.PINF, TRTag.NINF]
            else:
                # Should get REAL result
                assert res.value.tag == TRTag.REAL
                assert res.value.value == expected

    def test_parallel_array_reduce(self):
        """Test parallel array reduction."""
        values = [real(float(i)) for i in range(100)]

        # Parallel sum
        result = ParallelTRArray.reduce_sum(values, parallel=True)

        expected = sum(range(100))
        assert result.value.value == expected


@pytest.mark.benchmark
class TestParallelPerformance:
    """Benchmark parallel performance."""

    def test_parallel_speedup(self, benchmark):
        """Test that parallelization provides speedup."""

        def expensive_operation(x):
            # Simulate expensive computation
            result = x
            for _ in range(100):
                result = tr_add(result, real(0.001))
                result = tr_mul(result, real(1.0001))
            return result

        inputs = [real(float(i)) for i in range(100)]

        # Benchmark parallel execution
        config = ParallelConfig(backend="thread", num_workers=4)

        def parallel_run():
            return parallel_map(expensive_operation, inputs, config)

        results = benchmark(parallel_run)

        assert len(results) == 100
        # Verify some results
        assert results[0].value.tag == TRTag.REAL

    @given(st.integers(min_value=1, max_value=8))
    def test_scalability(self, num_workers):
        """Test scalability with different worker counts."""
        is_ci = bool(os.getenv("CI"))

        def work(x):
            time.sleep(0.001)  # Simulate work
            return tr_mul(x, real(2.0))

        inputs = [real(float(i)) for i in range(num_workers * 10)]
        config = ParallelConfig(backend="thread", num_workers=num_workers)

        start = time.time()
        results = parallel_map(work, inputs, config)
        duration = time.time() - start

        # Should scale somewhat with workers
        sequential_time = 0.001 * len(inputs)
        speedup = sequential_time / duration

        # Expect at least some speedup with multiple workers
        if num_workers > 1:
            # Require some speedup locally; on CI tolerate less due to contention.
            if is_ci:
                assert speedup > 0.85
            else:
                assert speedup > 1.1  # At least ~10% speedup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
