"""Extended tests for IEEE bridge functionality."""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from zeroproof.bridge import (
    JAX_AVAILABLE,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    Precision,
    PrecisionContext,
    from_ieee,
    to_ieee,
    with_precision,
)
from zeroproof.core import TRTag, ninf, phi, pinf, real


class TestPrecision:
    """Test precision handling."""

    def test_precision_context(self):
        """Test precision context manager."""
        # Default is float64
        assert PrecisionContext.get_current_precision() == Precision.FLOAT64

        # Test context
        with PrecisionContext(Precision.FLOAT32):
            assert PrecisionContext.get_current_precision() == Precision.FLOAT32

            # Nested context
            with PrecisionContext(Precision.FLOAT16):
                assert PrecisionContext.get_current_precision() == Precision.FLOAT16

            # Back to float32
            assert PrecisionContext.get_current_precision() == Precision.FLOAT32

        # Back to default
        assert PrecisionContext.get_current_precision() == Precision.FLOAT64

    def test_precision_casting(self):
        """Test precision casting effects."""
        # Value that fits in all precisions
        tr_val = real(1.5)

        # Should be unchanged in any precision
        for prec in [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]:
            result = with_precision(tr_val, prec)
            assert result.tag == TRTag.REAL
            assert result.value == 1.5

        # Value that overflows float16
        large_val = real(100000.0)
        result_f16 = with_precision(large_val, Precision.FLOAT16)
        assert result_f16.tag == TRTag.PINF  # Overflow to infinity

        # But fits in float32
        result_f32 = with_precision(large_val, Precision.FLOAT32)
        assert result_f32.tag == TRTag.REAL
        assert result_f32.value == 100000.0

    def test_precision_with_special_values(self):
        """Test that special values are unaffected by precision."""
        special_values = [pinf(), ninf(), phi()]

        for val in special_values:
            for prec in [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]:
                result = with_precision(val, prec)
                assert result.tag == val.tag

    def test_mixed_precision_strategy(self):
        """Test mixed precision computation strategy."""
        from zeroproof.bridge import MixedPrecisionStrategy

        strategy = MixedPrecisionStrategy(
            compute_precision=Precision.FLOAT16,
            accumulate_precision=Precision.FLOAT32,
            output_precision=Precision.FLOAT16,
        )

        # Test accumulation in higher precision
        values = [real(0.001) for _ in range(1000)]
        result = strategy.accumulate(values)

        # Should accumulate to approximately 1.0
        assert result.tag == TRTag.REAL
        assert abs(result.value - 1.0) < 0.01

        # Finalize to output precision
        final = strategy.finalize(result)
        assert final.tag == TRTag.REAL


class TestNumpyBridge:
    """Test NumPy bridge functionality."""

    def test_numpy_import_handling(self):
        """Test graceful handling when NumPy not available."""
        try:
            import numpy as np

            numpy_available = True
        except ImportError:
            numpy_available = False

        from zeroproof.bridge import NUMPY_AVAILABLE

        assert NUMPY_AVAILABLE == numpy_available

        if not numpy_available:
            with pytest.raises(ImportError):
                from zeroproof.bridge import TRArray

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")
    def test_numpy_array_conversion(self):
        """Test NumPy array conversions."""
        import numpy as np

        from zeroproof.bridge import TRArray, from_numpy, to_numpy

        # Test scalar conversion
        scalar_result = from_numpy(3.14)
        assert hasattr(scalar_result, "tag")
        assert scalar_result.tag == TRTag.REAL
        assert scalar_result.value == 3.14

        # Test array conversion
        arr = np.array([1.0, float("inf"), float("-inf"), float("nan"), 0.0])
        tr_arr = from_numpy(arr)

        assert isinstance(tr_arr, TRArray)
        assert tr_arr.shape == (5,)

        # Check tags
        assert tr_arr.is_real()[0]  # 1.0
        assert tr_arr.is_pinf()[1]  # inf
        assert tr_arr.is_ninf()[2]  # -inf
        assert tr_arr.is_phi()[3]  # nan
        assert tr_arr.is_real()[4]  # 0.0

        # Convert back
        ieee_arr = to_numpy(tr_arr)
        assert np.array_equal(arr, ieee_arr, equal_nan=True)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")
    def test_numpy_tag_counting(self):
        """Test tag counting utilities."""
        import numpy as np

        from zeroproof.bridge import count_tags, from_numpy

        arr = np.array([1.0, 2.0, float("inf"), float("nan"), -3.0, float("-inf")])
        tr_arr = from_numpy(arr)

        counts = count_tags(tr_arr)
        assert counts["REAL"] == 3
        assert counts["PINF"] == 1
        assert counts["NINF"] == 1
        assert counts["PHI"] == 1


class TestPyTorchBridge:
    """Test PyTorch bridge functionality."""

    def test_torch_import_handling(self):
        """Test graceful handling when PyTorch not available."""
        try:
            import torch

            torch_available = True
        except ImportError:
            torch_available = False

        from zeroproof.bridge import TORCH_AVAILABLE

        assert TORCH_AVAILABLE == torch_available

        if not torch_available:
            with pytest.raises(ImportError):
                from zeroproof.bridge import TRTensor

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_torch_tensor_conversion(self):
        """Test PyTorch tensor conversions."""
        import torch

        from zeroproof.bridge import TRTensor, from_torch, to_torch

        # Create test tensor
        tensor = torch.tensor([1.0, float("inf"), float("-inf"), float("nan"), 0.0])
        tr_tensor = from_torch(tensor)

        assert isinstance(tr_tensor, TRTensor)
        assert tr_tensor.shape == torch.Size([5])

        # Check tags
        assert tr_tensor.is_real()[0]  # 1.0
        assert tr_tensor.is_pinf()[1]  # inf
        assert tr_tensor.is_ninf()[2]  # -inf
        assert tr_tensor.is_phi()[3]  # nan
        assert tr_tensor.is_real()[4]  # 0.0

        # Convert back
        ieee_tensor = to_torch(tr_tensor)
        assert torch.allclose(tensor, ieee_tensor, equal_nan=True)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_torch_device_handling(self):
        """Test device movement for TRTensor."""
        import torch

        from zeroproof.bridge import TRTensor, from_torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create CPU tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        tr_tensor = from_torch(tensor)

        # Move to CUDA
        cuda_tensor = tr_tensor.cuda()
        assert cuda_tensor.device.type == "cuda"

        # Move back to CPU
        cpu_tensor = cuda_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"


class TestJAXBridge:
    """Test JAX bridge functionality."""

    def test_jax_import_handling(self):
        """Test graceful handling when JAX not available."""
        try:
            import jax

            jax_available = True
        except ImportError:
            jax_available = False

        from zeroproof.bridge import JAX_AVAILABLE

        assert JAX_AVAILABLE == jax_available

        if not jax_available:
            with pytest.raises(ImportError):
                from zeroproof.bridge import TRJaxArray

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_array_conversion(self):
        """Test JAX array conversions."""
        import jax.numpy as jnp

        from zeroproof.bridge import TRJaxArray, from_jax, to_jax

        # Create test array
        arr = jnp.array([1.0, jnp.inf, -jnp.inf, jnp.nan, 0.0])
        tr_arr = from_jax(arr)

        assert isinstance(tr_arr, TRJaxArray)
        assert tr_arr.shape == (5,)

        # Check tags
        assert tr_arr.is_real()[0]  # 1.0
        assert tr_arr.is_pinf()[1]  # inf
        assert tr_arr.is_ninf()[2]  # -inf
        assert tr_arr.is_phi()[3]  # nan
        assert tr_arr.is_real()[4]  # 0.0

        # Convert back
        ieee_arr = to_jax(tr_arr)
        assert jnp.allclose(arr, ieee_arr, equal_nan=True)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_operations(self):
        """Test JAX TR operations."""
        import jax.numpy as jnp

        from zeroproof.bridge import from_jax, tr_add_jax, tr_mul_jax

        # Test addition
        a = from_jax(jnp.array([1.0, 2.0, jnp.inf]))
        b = from_jax(jnp.array([3.0, jnp.inf, -jnp.inf]))

        result = tr_add_jax(a, b)

        # Check results
        assert result.is_real()[0]  # 1 + 3 = 4
        assert result.is_pinf()[1]  # 2 + inf = inf
        assert result.is_phi()[2]  # inf + (-inf) = PHI


class TestCrossBridgeCompatibility:
    """Test compatibility between different bridges."""

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")
    def test_scalar_consistency(self):
        """Test that scalar conversions are consistent across bridges."""
        import numpy as np

        from zeroproof.bridge import from_numpy, to_numpy

        test_values = [0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")]

        for val in test_values:
            # Through NumPy
            tr_scalar = from_numpy(val)
            back_val = to_numpy(tr_scalar)

            # Direct IEEE
            tr_direct = from_ieee(val)
            back_direct = to_ieee(tr_direct)

            # Should have same tags
            assert tr_scalar.tag == tr_direct.tag

            # And same round-trip values
            if math.isnan(val):
                assert math.isnan(back_val) and math.isnan(back_direct)
            else:
                assert back_val == back_direct == val


@pytest.mark.property
class TestPrecisionProperties:
    """Property-based tests for precision handling."""

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_precision_preserves_finite_values(self, value):
        """Test that finite values are preserved or overflow consistently."""
        from zeroproof.bridge import analyze_precision_requirements

        analysis = analyze_precision_requirements([value])
        min_prec = analysis["min_precision"]

        # If value fits in precision, it should be preserved
        tr_val = real(value)
        result = with_precision(tr_val, min_prec)

        if result.tag == TRTag.REAL:
            # Value was preserved
            assert abs(result.value - value) <= abs(value) * 1e-6
        else:
            # Value overflowed - check it was too large
            from zeroproof.bridge import get_precision_info

            info = get_precision_info(min_prec)
            assert abs(value) > info["max_value"]

    @given(st.sampled_from(list(Precision)))
    def test_precision_context_nesting(self, precision):
        """Test that precision contexts nest properly."""
        original = PrecisionContext.get_current_precision()

        with PrecisionContext(precision):
            assert PrecisionContext.get_current_precision() == precision

            # Nested same precision
            with PrecisionContext(precision):
                assert PrecisionContext.get_current_precision() == precision

        assert PrecisionContext.get_current_precision() == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
