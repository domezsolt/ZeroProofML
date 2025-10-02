"""Unit tests for precision enforcement in ZeroProof."""

import numpy as np
import pytest

from zeroproof import (
    PrecisionConfig,
    PrecisionMode,
    from_ieee,
    ninf,
    phi,
    pinf,
    precision_context,
    real,
    to_ieee,
    tr_add,
    tr_div,
    tr_log,
    tr_mul,
    tr_sqrt,
)


class TestPrecisionConfig:
    """Test precision configuration functionality."""

    def test_default_precision(self):
        """Test that default precision is float64."""
        assert PrecisionConfig.get_precision() == PrecisionMode.FLOAT64
        assert PrecisionConfig.get_dtype() == np.float64

    def test_set_precision(self):
        """Test setting different precision modes."""
        original = PrecisionConfig.get_precision()

        try:
            # Test setting via enum
            PrecisionConfig.set_precision(PrecisionMode.FLOAT32)
            assert PrecisionConfig.get_precision() == PrecisionMode.FLOAT32
            assert PrecisionConfig.get_dtype() == np.float32

            # Test setting via string
            PrecisionConfig.set_precision("float16")
            assert PrecisionConfig.get_precision() == PrecisionMode.FLOAT16
            assert PrecisionConfig.get_dtype() == np.float16

            # Test invalid precision
            with pytest.raises(ValueError):
                PrecisionConfig.set_precision("float128")
        finally:
            PrecisionConfig.set_precision(original)

    def test_precision_context(self):
        """Test precision context manager."""
        original = PrecisionConfig.get_precision()
        assert original == PrecisionMode.FLOAT64

        with precision_context("float32"):
            assert PrecisionConfig.get_precision() == PrecisionMode.FLOAT32
            x = real(1.0)
            # Value should be Python float (precision is enforced internally)
            assert isinstance(x.value, float)

        # Should revert to original
        assert PrecisionConfig.get_precision() == original

    def test_enforce_precision(self):
        """Test precision enforcement."""
        # Test float64 (default)
        value64 = PrecisionConfig.enforce_precision(3.14159265358979323846)
        assert isinstance(value64, float)

        # Test float32
        with precision_context("float32"):
            value32 = PrecisionConfig.enforce_precision(3.14159265358979323846)
            assert isinstance(value32, float)
            # Float32 has less precision - the value will be rounded
            # when converted through numpy float32
            assert abs(value32 - value64) > 0  # They should differ due to precision

    def test_overflow_detection(self):
        """Test overflow detection for different precisions."""
        # Float64 can handle very large values
        assert not PrecisionConfig.check_overflow(1e100)
        assert PrecisionConfig.check_overflow(1e400)

        # Float32 has smaller range
        with precision_context("float32"):
            assert not PrecisionConfig.check_overflow(1e20)
            assert PrecisionConfig.check_overflow(1e50)

        # Float16 has very limited range
        with precision_context("float16"):
            assert not PrecisionConfig.check_overflow(100.0)
            assert PrecisionConfig.check_overflow(100000.0)


class TestPrecisionInOperations:
    """Test that precision is maintained in operations."""

    def test_real_factory_precision(self):
        """Test that real() enforces precision."""
        # Default float64
        x = real(3.14159265358979323846)
        assert isinstance(x.value, float)

        # Float32
        with precision_context("float32"):
            y = real(3.14159265358979323846)
            assert isinstance(y.value, float)
            # Precision should be limited due to float32 conversion
            assert y.value != x.value

    def test_arithmetic_precision(self):
        """Test precision in arithmetic operations."""
        with precision_context("float32"):
            x = real(1.0)
            y = real(2.0)

            # Addition
            z = tr_add(x, y)
            assert z.tag.name == "REAL"
            assert isinstance(z.value, float)

            # Multiplication
            w = tr_mul(x, y)
            assert isinstance(w.value, float)

            # Division
            v = tr_div(x, y)
            assert isinstance(v.value, float)

    def test_overflow_to_infinity(self):
        """Test that overflow produces infinity tags."""
        # Float16 has max ~65504
        with precision_context("float16"):
            x = real(1000.0)
            y = real(1000.0)

            # Should overflow to infinity
            z = tr_mul(x, y)
            assert z.tag.name == "PINF"

            # Negative overflow
            w = tr_mul(real(-1000.0), y)
            assert w.tag.name == "NINF"

    def test_ieee_bridge_precision(self):
        """Test precision in IEEE bridge conversions."""
        with precision_context("float32"):
            # From IEEE
            x = from_ieee(3.14159265358979323846)
            assert isinstance(x.value, float)

            # To IEEE
            y = real(2.718281828459045)
            ieee_val = to_ieee(y)
            # Should maintain float32 precision
            assert abs(ieee_val - 2.7182817) < 1e-6

    def test_unary_ops_precision(self):
        """Test precision in unary operations."""
        with precision_context("float32"):
            x = real(2.0)

            # Square root
            sqrt_x = tr_sqrt(x)
            assert isinstance(sqrt_x.value, float)

            # Logarithm
            log_x = tr_log(x)
            assert isinstance(log_x.value, float)

    def test_precision_consistency(self):
        """Test that operations maintain consistent precision."""
        with precision_context("float32"):
            # Create a chain of operations
            x = real(1.1)
            y = real(2.2)
            z = real(3.3)

            # Complex expression
            result = tr_div(tr_add(tr_mul(x, y), z), tr_sqrt(tr_add(x, y)))

            # All intermediate results should maintain float32 precision
            assert isinstance(result.value, float)


class TestPrecisionEdgeCases:
    """Test edge cases in precision handling."""

    def test_subnormal_handling(self):
        """Test handling of subnormal numbers."""
        # Float32 smallest normal is ~1.18e-38
        with precision_context("float32"):
            tiny = real(1e-40)
            # Should be subnormal or zero in float32
            assert tiny.value == 0.0 or abs(tiny.value) < 1e-38

    def test_precision_epsilon(self):
        """Test machine epsilon for different precisions."""
        # Float64 epsilon
        eps64 = PrecisionConfig.get_epsilon()
        assert abs(eps64 - 2.220446049250313e-16) < 1e-20

        # Float32 epsilon
        with precision_context("float32"):
            eps32 = PrecisionConfig.get_epsilon()
            assert abs(eps32 - 1.1920929e-07) < 1e-10

        # Float16 epsilon
        with precision_context("float16"):
            eps16 = PrecisionConfig.get_epsilon()
            assert abs(eps16 - 0.00097656) < 1e-6

    def test_precision_limits(self):
        """Test max and min values for different precisions."""
        # Float64
        assert PrecisionConfig.get_max() > 1e300
        assert PrecisionConfig.get_min() < 1e-300

        # Float32
        with precision_context("float32"):
            assert 1e38 < PrecisionConfig.get_max() < 1e39
            assert 1e-45 < PrecisionConfig.get_min() < 1e-37

        # Float16
        with precision_context("float16"):
            assert 60000 < PrecisionConfig.get_max() < 70000
            assert 1e-8 < PrecisionConfig.get_min() < 1e-4
