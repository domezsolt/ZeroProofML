"""Unit tests for TRScalar type."""

import math
import pytest
from hypothesis import given, strategies as st

from zeroproof.core import TRScalar, TRTag, real, pinf, ninf, phi
from zeroproof.core import tr_add, tr_mul, tr_div, tr_neg, tr_abs, tr_sign
from zeroproof.core import tr_log, tr_sqrt, tr_pow_int
from zeroproof.bridge import from_ieee, to_ieee


class TestTRScalarCreation:
    """Test creation of TRScalar values."""
    
    def test_real_creation(self):
        """Test creating REAL tagged scalars."""
        x = real(3.14)
        assert x.tag == TRTag.REAL
        assert x.value == 3.14
        
        # Test with various values
        assert real(0.0).value == 0.0
        assert real(-42.5).value == -42.5
        assert real(1e100).value == 1e100
    
    def test_real_validation(self):
        """Test that real() rejects non-finite values."""
        with pytest.raises(ValueError):
            real(float('inf'))
        with pytest.raises(ValueError):
            real(float('-inf'))
        with pytest.raises(ValueError):
            real(float('nan'))
    
    def test_infinity_creation(self):
        """Test creating infinity scalars."""
        pos_inf = pinf()
        neg_inf = ninf()
        assert pos_inf.tag == TRTag.PINF
        assert neg_inf.tag == TRTag.NINF
        
        # Value field is not meaningful for infinities
        assert math.isnan(pos_inf.value)
        assert math.isnan(neg_inf.value)
    
    def test_phi_creation(self):
        """Test creating PHI (nullity) scalar."""
        null = phi()
        assert null.tag == TRTag.PHI
        assert math.isnan(null.value)


class TestTRScalarArithmetic:
    """Test arithmetic operations on TRScalar values."""
    
    def test_division_by_zero(self):
        """Test that division by zero returns appropriate infinities."""
        x = real(3.0)
        y = real(0.0)
        z = real(-2.0)
        
        assert (x / y).tag == TRTag.PINF  # 3/0 → +∞
        assert (z / y).tag == TRTag.NINF  # -2/0 → -∞
        assert (y / y).tag == TRTag.PHI   # 0/0 → Φ
    
    def test_infinity_arithmetic(self):
        """Test arithmetic with infinities."""
        inf = pinf()
        x = real(5.0)
        
        assert (inf + x).tag == TRTag.PINF     # ∞ + 5 → ∞
        assert (inf - inf).tag == TRTag.PHI    # ∞ - ∞ → Φ
        assert (real(0.0) * inf).tag == TRTag.PHI  # 0 × ∞ → Φ
        
        # More infinity tests
        assert (inf * inf).tag == TRTag.PINF   # ∞ × ∞ → ∞
        assert (inf * ninf()).tag == TRTag.NINF  # ∞ × -∞ → -∞
        assert (inf / inf).tag == TRTag.PHI    # ∞ / ∞ → Φ
        assert (x / inf).tag == TRTag.REAL     # finite / ∞ → 0
        assert (x / inf).value == 0.0
    
    def test_phi_propagation(self):
        """Test that PHI propagates through operations."""
        p = phi()
        x = real(5.0)
        
        # PHI propagates through all operations
        assert (p + x).tag == TRTag.PHI
        assert (x + p).tag == TRTag.PHI
        assert (p * x).tag == TRTag.PHI
        assert (p / x).tag == TRTag.PHI
        assert (x / p).tag == TRTag.PHI
        assert tr_neg(p).tag == TRTag.PHI
        assert tr_abs(p).tag == TRTag.PHI
    
    @pytest.mark.property
    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
    def test_real_arithmetic_preservation(self, value):
        """Property: arithmetic on REAL values stays REAL when no overflow."""
        x = real(value)
        y = real(2.0)
        
        # Test addition
        result_add = x + y
        expected_add = value + 2.0
        if math.isfinite(expected_add):
            assert result_add.tag == TRTag.REAL
            assert result_add.value == expected_add
        else:
            # Overflow should produce infinity
            assert result_add.tag in (TRTag.PINF, TRTag.NINF)
        
        # Test subtraction
        result_sub = x - y
        assert result_sub.tag == TRTag.REAL  # Unlikely to overflow with these bounds
        
        # Test multiplication
        result_mul = x * y
        expected_mul = value * 2.0
        if math.isfinite(expected_mul):
            assert result_mul.tag == TRTag.REAL
            assert result_mul.value == expected_mul
        else:
            # Overflow should produce infinity
            assert result_mul.tag in (TRTag.PINF, TRTag.NINF)
        
        # Test division
        if value != 0:
            result_div = y / x
            expected_div = 2.0 / value
            if math.isfinite(expected_div):
                assert result_div.tag == TRTag.REAL
                assert result_div.value == expected_div
            else:
                # Overflow should produce infinity
                assert result_div.tag in (TRTag.PINF, TRTag.NINF)


class TestTRScalarProperties:
    """Test mathematical properties of TRScalar."""
    
    @pytest.mark.property
    @given(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100)
    )
    def test_commutativity(self, a, b):
        """Property: addition and multiplication are commutative."""
        x = real(a)
        y = real(b)
        
        # Addition commutativity
        sum_xy = x + y
        sum_yx = y + x
        assert sum_xy.tag == sum_yx.tag
        if sum_xy.tag == TRTag.REAL:
            assert sum_xy.value == pytest.approx(sum_yx.value)
        
        # Multiplication commutativity
        mul_xy = x * y
        mul_yx = y * x
        assert mul_xy.tag == mul_yx.tag
        if mul_xy.tag == TRTag.REAL:
            assert mul_xy.value == pytest.approx(mul_yx.value)
    
    @pytest.mark.property
    def test_totality(self):
        """Property: all operations are total (never raise exceptions)."""
        # This should never raise an exception
        _ = real(1.0) / real(0.0)
        _ = pinf() - pinf()
        _ = real(0.0) * pinf()
        _ = ninf() / ninf()
        
        # Domain-aware operations also don't raise
        _ = tr_log(real(-1.0))
        _ = tr_sqrt(real(-1.0))
        _ = tr_pow_int(real(0.0), 0)
    
    def test_special_values(self):
        """Test special mathematical values."""
        # Test signed zeros
        pos_zero = real(0.0)
        neg_zero = real(-0.0)
        assert pos_zero.value == 0.0
        assert neg_zero.value == -0.0
        
        # Division by signed zeros
        x = real(1.0)
        assert (x / pos_zero).tag == TRTag.PINF
        assert (x / neg_zero).tag == TRTag.NINF


class TestIEEEBridge:
    """Test IEEE-754 bridge functionality."""
    
    def test_ieee_to_tr_conversion(self):
        """Test converting IEEE floats to TR values."""
        assert from_ieee(3.14).tag == TRTag.REAL
        assert from_ieee(3.14).value == 3.14
        assert from_ieee(float('inf')).tag == TRTag.PINF
        assert from_ieee(float('-inf')).tag == TRTag.NINF
        assert from_ieee(float('nan')).tag == TRTag.PHI
        
        # Test special float values
        assert from_ieee(0.0).tag == TRTag.REAL
        assert from_ieee(-0.0).tag == TRTag.REAL
        assert from_ieee(1e-300).tag == TRTag.REAL  # Subnormal
    
    def test_tr_to_ieee_conversion(self):
        """Test converting TR values to IEEE floats."""
        assert to_ieee(real(3.14)) == 3.14
        assert to_ieee(pinf()) == float('inf')
        assert to_ieee(ninf()) == float('-inf')
        assert math.isnan(to_ieee(phi()))
        
        # Test edge cases
        assert to_ieee(real(0.0)) == 0.0
        assert to_ieee(real(-0.0)) == -0.0
    
    @pytest.mark.property
    @given(st.floats(width=64))
    def test_round_trip_conversion(self, value):
        """Property: IEEE → TR → IEEE preserves values."""
        tr_val = from_ieee(value)
        ieee_val = to_ieee(tr_val)
        
        if math.isfinite(value):
            assert ieee_val == value
        elif math.isinf(value):
            assert math.isinf(ieee_val) and (value > 0) == (ieee_val > 0)
        else:  # NaN
            assert math.isnan(ieee_val)


class TestUnaryOperations:
    """Test unary operations on transreal values."""
    
    def test_negation(self):
        """Test transreal negation."""
        assert tr_neg(real(5.0)).value == -5.0
        assert tr_neg(real(-3.0)).value == 3.0
        assert tr_neg(pinf()).tag == TRTag.NINF
        assert tr_neg(ninf()).tag == TRTag.PINF
        assert tr_neg(phi()).tag == TRTag.PHI
    
    def test_absolute_value(self):
        """Test transreal absolute value."""
        assert tr_abs(real(5.0)).value == 5.0
        assert tr_abs(real(-5.0)).value == 5.0
        assert tr_abs(real(0.0)).value == 0.0
        assert tr_abs(pinf()).tag == TRTag.PINF
        assert tr_abs(ninf()).tag == TRTag.PINF
        assert tr_abs(phi()).tag == TRTag.PHI
    
    def test_sign(self):
        """Test transreal sign function."""
        assert tr_sign(real(5.0)).value == 1.0
        assert tr_sign(real(-5.0)).value == -1.0
        assert tr_sign(real(0.0)).value == 0.0
        assert tr_sign(pinf()).value == 1.0
        assert tr_sign(ninf()).value == -1.0
        assert tr_sign(phi()).tag == TRTag.PHI
    
    def test_logarithm(self):
        """Test transreal logarithm."""
        # Valid domain
        assert abs(tr_log(real(1.0)).value - 0.0) < 1e-10
        assert abs(tr_log(real(math.e)).value - 1.0) < 1e-10
        
        # Invalid domain returns PHI
        assert tr_log(real(0.0)).tag == TRTag.PHI
        assert tr_log(real(-1.0)).tag == TRTag.PHI
        assert tr_log(ninf()).tag == TRTag.PHI
        
        # Special cases
        assert tr_log(pinf()).tag == TRTag.PINF
        assert tr_log(phi()).tag == TRTag.PHI
    
    def test_square_root(self):
        """Test transreal square root."""
        # Valid domain
        assert tr_sqrt(real(4.0)).value == 2.0
        assert tr_sqrt(real(0.0)).value == 0.0
        
        # Invalid domain returns PHI
        assert tr_sqrt(real(-1.0)).tag == TRTag.PHI
        assert tr_sqrt(ninf()).tag == TRTag.PHI
        
        # Special cases
        assert tr_sqrt(pinf()).tag == TRTag.PINF
        assert tr_sqrt(phi()).tag == TRTag.PHI
    
    def test_integer_power(self):
        """Test transreal integer power."""
        # Regular cases
        assert tr_pow_int(real(2.0), 3).value == 8.0
        assert tr_pow_int(real(2.0), -2).value == 0.25
        assert tr_pow_int(real(2.0), 0).value == 1.0
        
        # Special cases that return PHI
        assert tr_pow_int(real(0.0), 0).tag == TRTag.PHI
        assert tr_pow_int(pinf(), 0).tag == TRTag.PHI
        assert tr_pow_int(ninf(), 0).tag == TRTag.PHI
        
        # Powers of infinity
        assert tr_pow_int(pinf(), 2).tag == TRTag.PINF
        assert tr_pow_int(pinf(), -2).tag == TRTag.REAL
        assert tr_pow_int(ninf(), 2).tag == TRTag.PINF  # Even power
        assert tr_pow_int(ninf(), 3).tag == TRTag.NINF  # Odd power


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
