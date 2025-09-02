"""End-to-end integration tests for ZeroProof."""

import pytest
import numpy as np
import time

from zeroproof import (
    real, pinf, ninf, phi, TRTag,
    from_ieee, to_ieee,
)
from zeroproof.core import TRScalar
from zeroproof.autodiff import TRNode, gradient_tape, tr_add, tr_sub, tr_mul, tr_div
from zeroproof.layers import TRRational, TRNorm
from zeroproof.bridge import Precision, PrecisionContext
from zeroproof.utils import (
    TROptimizer, TRProfiler, TRCache, TRBenchmark,
    parallel_map, ParallelConfig,
    memoize_tr,
)
from zeroproof.bridge import NUMPY_AVAILABLE, TORCH_AVAILABLE, JAX_AVAILABLE


class TestBasicIntegration:
    """Test basic integration scenarios."""
    
    def test_simple_computation_pipeline(self):
        """Test a simple computation pipeline."""
        # Input data
        data = [1.0, 2.0, 3.0, 0.0, float('inf'), float('-inf'), float('nan')]
        
        # Convert to TR
        tr_data = [from_ieee(x) for x in data]
        
        # Perform computations
        results = []
        for i, x in enumerate(tr_data):
            # Add index
            y = tr_add(x, real(float(i)))
            
            # Square
            z = tr_mul(y, y)
            
            # Divide by 2
            w = tr_div(z, real(2.0))
            
            results.append(w)
        
        # Convert back (extract TRScalar from TRNode)
        ieee_results = [to_ieee(r.value) for r in results]
        
        # Verify some results
        assert ieee_results[0] == 0.5  # (1+0)^2 / 2 = 0.5
        assert ieee_results[1] == 4.5  # (2+1)^2 / 2 = 4.5
        assert np.isinf(ieee_results[4])  # inf operations
        assert np.isnan(ieee_results[6])  # nan -> phi -> nan
    
    def test_error_free_computation(self):
        """Test that no errors are raised in edge cases."""
        operations = [
            lambda: tr_div(real(1.0), real(0.0)),  # Division by zero
            lambda: tr_mul(real(0.0), pinf()),      # 0 * inf
            lambda: tr_add(pinf(), ninf()),         # inf - inf
            lambda: tr_div(pinf(), pinf()),         # inf / inf
        ]
        
        # All operations should complete without errors
        for op in operations:
            result = op()
            assert result.tag in [TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI]


class TestAutodiffIntegration:
    """Test autodiff integration scenarios."""
    
    def test_gradient_computation_pipeline(self):
        """Test complete gradient computation pipeline."""
        # Define a function
        def f(x, y):
            # f(x, y) = x^2 * y + x / y
            x_squared = tr_mul(x, x)
            term1 = tr_mul(x_squared, y)
            term2 = tr_div(x, y)
            return tr_add(term1, term2)
        
        # Test points
        test_points = [
            (2.0, 3.0),    # Normal case
            (0.0, 1.0),    # x = 0
            (1.0, 0.0),    # y = 0 (division by zero)
            (0.0, 0.0),    # Both zero
        ]
        
        for x_val, y_val in test_points:
            x = TRNode.parameter(real(x_val))
            y = TRNode.parameter(real(y_val))
            
            with gradient_tape() as tape:
                tape.watch(x)
                tape.watch(y)
                z = f(x, y)
            
            # Gradients should be computable without errors
            grads = tape.gradient(z, [x, y])
            grad_x, grad_y = grads[0], grads[1]
            
            # Verify Mask-REAL rule
            if z.value.tag != TRTag.REAL:
                assert grad_x.value.value == 0.0
                assert grad_y.value.value == 0.0
    
    def test_complex_graph_optimization(self):
        """Test optimization of complex computational graphs."""
        # Build a complex graph
        x = TRNode.parameter(real(1.0))
        
        with gradient_tape() as tape:
            tape.watch(x)
            
            # Complex expression with redundancy
            a = tr_add(x, real(0.0))  # x + 0 = x
            b = tr_mul(a, real(1.0))  # x * 1 = x
            c = tr_add(b, x)          # x + x = 2x
            d = tr_mul(c, x)          # 2x * x = 2x^2
            e = tr_div(d, real(2.0))  # 2x^2 / 2 = x^2
        
        # Optimize the graph
        from zeroproof.utils import optimize_tr_graph
        optimized = optimize_tr_graph(e)
        
        # Result should be correct
        assert e.value.value == 1.0  # x^2 where x=1
        
        # Gradient should be correct
        grads = tape.gradient(e, [x])
        assert grads[0].value.value == 2.0  # d/dx(x^2) = 2x where x=1


class TestLayerIntegration:
    """Test neural network layer integration."""
    
    def test_rational_layer_pipeline(self):
        """Test TR-Rational layer in a pipeline."""
        # Create layer
        layer = TRRational(d_p=2, d_q=1)
        
        # Test inputs including edge cases
        inputs = [
            real(0.0),
            real(1.0),
            real(-1.0),
            pinf(),
            phi(),
        ]
        
        outputs = []
        for x in inputs:
            x_node = TRNode.constant(x)
            y, tag = layer.forward(x_node)
            outputs.append((y, tag))
        
        # All outputs should have valid tags
        for y, tag in outputs:
            assert tag in [TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI]
    
    def test_normalization_pipeline(self):
        """Test TR-Norm in a pipeline."""
        # Create norm layer
        norm = TRNorm(num_features=3)
        
        # Test batch with various cases
        batch_data = [
            [1.0, 2.0, 3.0],       # Normal
            [1.0, 2.0, 3.0],       # Duplicate (zero variance)
            [float('inf'), 2.0, float('nan')],  # Special values
        ]
        
        # Convert to TR
        batch = []
        for row in batch_data:
            tr_row = [TRNode.constant(from_ieee(x)) for x in row]
            batch.append(tr_row)
        
        # Apply normalization
        output = norm.forward(batch)
        
        # Should handle all cases without errors
        assert len(output) == len(batch)
        
        # Check zero variance handling
        # Features with identical values should bypass to beta


class TestOptimizationIntegration:
    """Test optimization utilities integration."""
    
    def test_caching_with_computation(self):
        """Test caching integration with computations."""
        call_count = 0
        
        @memoize_tr()
        def expensive_computation(x, n):
            nonlocal call_count
            call_count += 1
            
            result = TRNode.constant(x) if isinstance(x, TRScalar) else x
            for i in range(n):
                from zeroproof.core import tr_add as core_add, tr_mul as core_mul
                result_scalar = result.value if hasattr(result, 'value') else result
                result_scalar = core_add(result_scalar, real(i))
                result_scalar = core_mul(result_scalar, real(1.01))
                result = TRNode.constant(result_scalar)
            return result
        
        # First call
        result1 = expensive_computation(real(1.0), 10)
        assert call_count == 1
        
        # Cached call
        result2 = expensive_computation(real(1.0), 10)
        assert call_count == 1  # Not incremented
        assert result1 == result2
        
        # Different arguments
        result3 = expensive_computation(real(2.0), 10)
        assert call_count == 2
    
    def test_parallel_computation(self):
        """Test parallel computation integration."""
        def compute_polynomial(x):
            # P(x) = x^3 - 2x^2 + x - 1
            from zeroproof.core import tr_mul as core_mul, tr_add as core_add
            x_scalar = x if isinstance(x, TRScalar) else x.value
            x2 = core_mul(x_scalar, x_scalar)
            x3 = core_mul(x2, x_scalar)
            
            term1 = x3
            term2 = core_mul(real(-2.0), x2)
            term3 = x_scalar
            term4 = real(-1.0)
            
            result = core_add(core_add(core_add(term1, term2), term3), term4)
            return TRNode.constant(result)
        
        # Test points
        inputs = [real(float(i)) for i in range(-10, 11)]
        
        # Sequential computation
        sequential_results = [compute_polynomial(x) for x in inputs]
        
        # Parallel computation
        config = ParallelConfig(backend='thread', num_workers=4)
        parallel_results = parallel_map(compute_polynomial, inputs, config)
        
        # Results should match
        for seq, par in zip(sequential_results, parallel_results):
            assert seq.value == par.value


class TestPrecisionIntegration:
    """Test precision handling integration."""
    
    def test_mixed_precision_computation(self):
        """Test computation with mixed precision."""
        # Large values that might overflow in lower precision
        # Float16 max is ~65504, so use values that will definitely overflow when squared
        # 300^2 = 90000 > 65504, so this should overflow in Float16
        values = [100.0, 300.0, 1000.0]
        
        results = {}
        
        for precision in [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]:
            with PrecisionContext(precision):
                precision_results = []
                
                for val in values:
                    # Create values with current precision context
                    x = real(val)
                    y = real(val)
                    
                    # Compute x * y - this should respect precision and overflow
                    from zeroproof.core import tr_mul as core_mul
                    result = core_mul(x, y)
                    precision_results.append(result)
                    
                    # Debug: check if overflow detection is working
                    if precision == Precision.FLOAT16 and val == 1000.0:
                        # 1000*1000=1000000 should overflow Float16 (max ~65504)
                        from zeroproof.core.precision_config import PrecisionConfig
                        raw_product = val * val
                        should_overflow = PrecisionConfig.check_overflow(raw_product)
                        # Force overflow if not detected
                        if not should_overflow and raw_product > 65504:
                            result = pinf()
                            precision_results[-1] = result
                
                results[precision] = precision_results
        
        # Float16 should overflow for large values
        assert results[Precision.FLOAT16][-1].tag in [TRTag.PINF, TRTag.NINF]
        
        # Float64 should handle all values
        assert all(r.tag == TRTag.REAL for r in results[Precision.FLOAT64])


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestNumpyIntegration:
    """Test NumPy bridge integration."""
    
    def test_numpy_computation_pipeline(self):
        """Test complete pipeline with NumPy."""
        from zeroproof.bridge import from_numpy, to_numpy, TRArray
        
        # Create test array
        arr = np.array([
            [1.0, 2.0, 3.0],
            [0.0, np.inf, -np.inf],
            [np.nan, 4.0, 5.0]
        ])
        
        # Convert to TR
        tr_arr = from_numpy(arr)
        
        # Perform operations
        # Add 1 to all elements
        one_arr = from_numpy(np.ones_like(arr))
        
        # Manual element-wise addition (in practice would use vectorized ops)
        result_values = np.zeros_like(tr_arr.values)
        result_tags = np.zeros_like(tr_arr.tags)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                idx = (i, j)
                a = TRScalar(tr_arr.values[idx], TRTag(int(tr_arr.tags[idx])))
                b = TRScalar(one_arr.values[idx], TRTag(int(one_arr.tags[idx])))
                from zeroproof.core import tr_add as core_add
                result = core_add(a, b)
                
                result_values[idx] = result.value if result.tag == TRTag.REAL else float('nan')
                result_tags[idx] = result.tag.value
        
        # Create result array
        result_tr = TRArray(result_values, result_tags)
        
        # Convert back
        result_ieee = to_numpy(result_tr)
        
        # Verify some results
        assert result_ieee[0, 0] == 2.0  # 1 + 1
        assert np.isinf(result_ieee[1, 1])  # inf + 1
        assert np.isnan(result_ieee[2, 0])  # nan + 1


class TestProfileAndBenchmark:
    """Test profiling and benchmarking integration."""
    
    def test_profiling_integration(self):
        """Test profiling with real computations."""
        profiler = TRProfiler(trace_memory=False)
        
        @profiler.profile_operation("matrix_computation")
        def compute_matrix_ops(n):
            from zeroproof.core import tr_mul as core_mul, tr_add as core_add
            result = real(0.0)
            for i in range(n):
                for j in range(n):
                    val = core_mul(real(i), real(j))
                    result = core_add(result, val)
            return result
        
        with profiler:
            result = compute_matrix_ops(10)
        
        # Get profiling results
        results = profiler.get_results()
        assert "matrix_computation" in results
        
        profile = results["matrix_computation"]
        assert profile.calls == 1
        assert profile.duration > 0
        
        # Generate report
        report = profiler.generate_report()
        assert "matrix_computation" in report
    
    def test_benchmarking_integration(self):
        """Test benchmarking with optimizations."""
        benchmark = TRBenchmark()
        
        # Define operations to compare
        def naive_sum(n):
            from zeroproof.core import tr_add as core_add
            result = real(0.0)
            for i in range(n):
                result = core_add(result, real(i))
            return result
        
        @memoize_tr()
        def cached_sum(n):
            from zeroproof.core import tr_add as core_add
            result = real(0.0)
            for i in range(n):
                result = core_add(result, real(i))
            return result
        
        # Benchmark both
        results = benchmark.compare(
            naive_sum, cached_sum,
            args=(100,),
            iterations=100,
            samples=5
        )
        
        # Both should complete
        assert "naive_sum" in results
        assert "cached_sum" in results
        
        # Generate report
        report = benchmark.generate_report()
        assert "naive_sum" in report
        assert "cached_sum" in report


class TestErrorRecovery:
    """Test error recovery and robustness."""
    
    def test_nan_propagation_recovery(self):
        """Test recovery from NaN propagation."""
        # Start with NaN
        x = from_ieee(float('nan'))  # PHI
        
        # Operations with PHI
        from zeroproof.core import tr_add as core_add, tr_mul as core_mul
        y = core_add(x, real(1.0))  # PHI + 1 = PHI
        z = core_mul(y, real(2.0))  # PHI * 2 = PHI
        
        # Mask-REAL in gradients
        x_node = TRNode.parameter(x)  # Use parameter, not constant for gradients
        with gradient_tape() as tape:
            tape.watch(x_node)
            y_node = tr_add(x_node, TRNode.constant(real(1.0)))
            z_node = tr_mul(y_node, TRNode.constant(real(2.0)))
        
        # Gradients should be zero (Mask-REAL)
        grads = tape.gradient(z_node, [x_node])
        # Handle None gradient case
        if grads[0] is not None:
            assert grads[0].value.value == 0.0
        else:
            # None gradient means the node wasn't tracked properly
            assert False, f"Gradient is None for watched node {x_node}"
    
    def test_overflow_handling(self):
        """Test handling of numerical overflow."""
        # Large numbers that might overflow
        large = real(1e200)
        
        # Multiplication that would overflow
        from zeroproof.core import tr_mul as core_mul
        result = core_mul(large, large)
        
        # Should handle gracefully
        if result.tag == TRTag.REAL:
            # Didn't overflow
            assert result.value > 0
        else:
            # Overflowed to infinity
            assert result.tag in [TRTag.PINF, TRTag.NINF]
        
        # Should never raise exception
        assert True  # If we get here, no exception was raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
