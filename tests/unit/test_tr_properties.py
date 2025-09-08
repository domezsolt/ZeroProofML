"""
Property-based tests for TR operations.

Tests totality, closure, embedding, and algebraic properties.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite

from zeroproof.core import (
    TRScalar, TRTag,
    real, pinf, ninf, phi, bottom,
    tr_add, tr_sub, tr_mul, tr_div,
    tr_abs, tr_sign, tr_neg,
    tr_log, tr_sqrt, tr_pow_int,
    is_real, is_pinf, is_ninf, is_phi,
)


# ============================================================================
# Hypothesis Strategies
# ============================================================================

@composite
def finite_reals(draw, min_value=-1e6, max_value=1e6):
    """Generate finite real TR values."""
    val = draw(st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False
    ))
    return real(val)


@composite
def tr_scalars(draw):
    """Generate arbitrary TR scalars."""
    tag = draw(st.sampled_from([
        TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI
    ]))
    
    if tag == TRTag.REAL:
        val = draw(st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False
        ))
        return real(val)
    elif tag == TRTag.PINF:
        return pinf()
    elif tag == TRTag.NINF:
        return ninf()
    else:
        return phi()


@composite
def tr_scalars_no_phi(draw):
    """Generate TR scalars excluding PHI."""
    tag = draw(st.sampled_from([
        TRTag.REAL, TRTag.PINF, TRTag.NINF
    ]))
    
    if tag == TRTag.REAL:
        val = draw(st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            min_value=-1e6,
            max_value=1e6
        ))
        return real(val)
    elif tag == TRTag.PINF:
        return pinf()
    else:
        return ninf()


# ============================================================================
# Totality Tests
# ============================================================================

class TestTotality:
    """Test that all TR operations are total (never throw exceptions)."""
    
    @given(tr_scalars(), tr_scalars())
    @settings(max_examples=200)
    def test_add_totality(self, a, b):
        """Addition is total on TR."""
        result = tr_add(a, b)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    @given(tr_scalars(), tr_scalars())
    @settings(max_examples=200)
    def test_mul_totality(self, a, b):
        """Multiplication is total on TR."""
        result = tr_mul(a, b)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    @given(tr_scalars(), tr_scalars())
    @settings(max_examples=200)
    def test_div_totality(self, a, b):
        """Division is total on TR."""
        result = tr_div(a, b)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    @given(tr_scalars())
    @settings(max_examples=200)
    def test_unary_totality(self, x):
        """Unary operations are total on TR."""
        # All unary ops should work without throwing
        results = [
            tr_neg(x),
            tr_abs(x),
            tr_sign(x),
            tr_log(x),
            tr_sqrt(x),
        ]
        
        for result in results:
            assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    def test_division_by_zero(self):
        """Division by zero produces appropriate TR values."""
        # Positive / 0 = PINF
        result = tr_div(real(5.0), real(0.0))
        assert is_pinf(result)
        
        # Negative / 0 = NINF
        result = tr_div(real(-5.0), real(0.0))
        assert is_ninf(result)
        
        # 0 / 0 = PHI
        result = tr_div(real(0.0), real(0.0))
        assert is_phi(result)


# ============================================================================
# Closure Tests
# ============================================================================

class TestClosure:
    """Test that TR operations are closed (results stay in TR)."""
    
    @given(tr_scalars(), tr_scalars())
    @settings(max_examples=200)
    def test_add_closure(self, a, b):
        """Addition is closed over TR."""
        result = tr_add(a, b)
        assert isinstance(result, TRScalar)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    @given(tr_scalars(), tr_scalars(), tr_scalars())
    @settings(max_examples=200)
    def test_composition_closure(self, a, b, c):
        """Composed operations stay in TR."""
        # (a + b) * c
        intermediate = tr_add(a, b)
        result = tr_mul(intermediate, c)
        
        assert isinstance(result, TRScalar)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    @given(tr_scalars())
    @settings(max_examples=200)
    def test_repeated_ops_closure(self, x):
        """Repeated operations stay in TR."""
        result = x
        for _ in range(5):
            result = tr_add(result, result)  # Double 5 times
        
        assert isinstance(result, TRScalar)
        assert result.tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}


# ============================================================================
# Embedding Tests
# ============================================================================

class TestEmbedding:
    """Test that R embeds correctly into TR."""
    
    @given(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False))
    @settings(max_examples=200)
    def test_real_embedding_preserves_value(self, x):
        """Embedding preserves real values."""
        tr_val = real(x)
        assert tr_val.tag == TRTag.REAL
        assert tr_val.value == x
    
    @given(
        st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
        st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False)
    )
    @settings(max_examples=200)
    def test_real_operations_preserve_embedding(self, x, y):
        """Operations on embedded reals match real arithmetic."""
        # Embed
        tr_x = real(x)
        tr_y = real(y)
        
        # Add
        tr_sum = tr_add(tr_x, tr_y)
        if is_real(tr_sum):  # May overflow to infinity
            assert abs(tr_sum.value - (x + y)) < 1e-10
        
        # Multiply
        tr_prod = tr_mul(tr_x, tr_y)
        if is_real(tr_prod):  # May overflow to infinity
            assert abs(tr_prod.value - (x * y)) < 1e-10
        
        # Divide (if y != 0)
        if abs(y) > 1e-10:
            tr_quot = tr_div(tr_x, tr_y)
            if is_real(tr_quot):
                assert abs(tr_quot.value - (x / y)) < 1e-10
    
    def test_special_values_embedding(self):
        """Special values embed correctly."""
        # Zero
        zero = real(0.0)
        assert zero.tag == TRTag.REAL
        assert zero.value == 0.0
        
        # One
        one = real(1.0)
        assert one.tag == TRTag.REAL
        assert one.value == 1.0
        
        # Negative one
        neg_one = real(-1.0)
        assert neg_one.tag == TRTag.REAL
        assert neg_one.value == -1.0


# ============================================================================
# Algebraic Properties (REAL slice)
# ============================================================================

class TestAlgebraicProperties:
    """Test algebraic properties on the REAL slice."""
    
    @given(finite_reals(), finite_reals())
    @settings(max_examples=200)
    def test_add_commutative(self, a, b):
        """Addition is commutative on REAL."""
        result1 = tr_add(a, b)
        result2 = tr_add(b, a)
        
        if is_real(result1) and is_real(result2):
            assert abs(result1.value - result2.value) < 1e-10
        else:
            assert result1.tag == result2.tag
    
    @given(finite_reals(), finite_reals())
    @settings(max_examples=200)
    def test_mul_commutative(self, a, b):
        """Multiplication is commutative on REAL."""
        result1 = tr_mul(a, b)
        result2 = tr_mul(b, a)
        
        if is_real(result1) and is_real(result2):
            assert abs(result1.value - result2.value) < 1e-10
        else:
            assert result1.tag == result2.tag
    
    @given(finite_reals(), finite_reals(), finite_reals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_add_associative(self, a, b, c):
        """Addition is associative on REAL."""
        # Limit values to prevent overflow differences
        assume(abs(a.value) < 1e3)
        assume(abs(b.value) < 1e3)
        assume(abs(c.value) < 1e3)
        
        # (a + b) + c
        result1 = tr_add(tr_add(a, b), c)
        # a + (b + c)
        result2 = tr_add(a, tr_add(b, c))
        
        if is_real(result1) and is_real(result2):
            assert abs(result1.value - result2.value) < 1e-10
    
    @given(finite_reals())
    @settings(max_examples=200)
    def test_add_identity(self, a):
        """Zero is the additive identity."""
        zero = real(0.0)
        
        result1 = tr_add(a, zero)
        result2 = tr_add(zero, a)
        
        assert is_real(result1)
        assert is_real(result2)
        assert abs(result1.value - a.value) < 1e-10
        assert abs(result2.value - a.value) < 1e-10
    
    @given(finite_reals())
    @settings(max_examples=200)
    def test_mul_identity(self, a):
        """One is the multiplicative identity."""
        one = real(1.0)
        
        result1 = tr_mul(a, one)
        result2 = tr_mul(one, a)
        
        assert is_real(result1)
        assert is_real(result2)
        assert abs(result1.value - a.value) < 1e-10
        assert abs(result2.value - a.value) < 1e-10


# ============================================================================
# Special Cases and Edge Cases
# ============================================================================

class TestSpecialCases:
    """Test special cases and edge behaviors."""
    
    def test_infinity_arithmetic(self):
        """Test infinity arithmetic rules."""
        # inf + inf = inf
        assert is_pinf(tr_add(pinf(), pinf()))
        assert is_ninf(tr_add(ninf(), ninf()))
        
        # inf + (-inf) = phi
        assert is_phi(tr_add(pinf(), ninf()))
        assert is_phi(tr_add(ninf(), pinf()))
        
        # 0 * inf = phi
        assert is_phi(tr_mul(real(0.0), pinf()))
        assert is_phi(tr_mul(real(0.0), ninf()))
        assert is_phi(tr_mul(pinf(), real(0.0)))
        assert is_phi(tr_mul(ninf(), real(0.0)))
        
        # inf / inf = phi
        assert is_phi(tr_div(pinf(), pinf()))
        assert is_phi(tr_div(ninf(), ninf()))
        assert is_phi(tr_div(pinf(), ninf()))
        assert is_phi(tr_div(ninf(), pinf()))
    
    def test_phi_propagation(self):
        """Test that PHI propagates through operations."""
        # PHI + anything = PHI
        assert is_phi(tr_add(phi(), real(5.0)))
        assert is_phi(tr_add(phi(), pinf()))
        assert is_phi(tr_add(phi(), ninf()))
        assert is_phi(tr_add(phi(), phi()))
        
        # PHI * anything = PHI
        assert is_phi(tr_mul(phi(), real(5.0)))
        assert is_phi(tr_mul(phi(), pinf()))
        assert is_phi(tr_mul(phi(), ninf()))
        assert is_phi(tr_mul(phi(), phi()))
        
        # PHI / anything = PHI
        assert is_phi(tr_div(phi(), real(5.0)))
        assert is_phi(tr_div(phi(), pinf()))
        
        # anything / PHI = PHI
        assert is_phi(tr_div(real(5.0), phi()))
        assert is_phi(tr_div(pinf(), phi()))
    
    def test_power_special_cases(self):
        """Test power operation special cases."""
        # 0^0 = PHI
        assert is_phi(tr_pow_int(real(0.0), 0))
        
        # inf^0 = PHI
        assert is_phi(tr_pow_int(pinf(), 0))
        assert is_phi(tr_pow_int(ninf(), 0))
        
        # x^0 = 1 for x != 0, inf
        assert tr_pow_int(real(5.0), 0).value == 1.0
        assert tr_pow_int(real(-3.0), 0).value == 1.0
        
        # 0^n = 0 for n > 0
        assert tr_pow_int(real(0.0), 5).value == 0.0
        
        # 0^n = inf for n < 0
        assert is_pinf(tr_pow_int(real(0.0), -1))
    
    def test_log_sqrt_domains(self):
        """Test log and sqrt domain restrictions."""
        # log(negative) = PHI
        assert is_phi(tr_log(real(-1.0)))
        assert is_phi(tr_log(real(-5.0)))
        assert is_phi(tr_log(real(0.0)))
        
        # log(positive) = REAL
        result = tr_log(real(2.718281828))
        assert is_real(result)
        assert abs(result.value - 1.0) < 0.01
        
        # sqrt(negative) = PHI
        assert is_phi(tr_sqrt(real(-1.0)))
        assert is_phi(tr_sqrt(real(-5.0)))
        
        # sqrt(non-negative) = REAL
        result = tr_sqrt(real(4.0))
        assert is_real(result)
        assert abs(result.value - 2.0) < 1e-10
        
        result = tr_sqrt(real(0.0))
        assert is_real(result)
        assert result.value == 0.0


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Test that TR operations are deterministic."""
    
    @given(tr_scalars(), tr_scalars())
    @settings(max_examples=100)
    def test_operation_determinism(self, a, b):
        """Same inputs produce same outputs."""
        # Run operation multiple times
        results = []
        for _ in range(5):
            results.append(tr_add(a, b))
        
        # All results should be identical
        first_tag = results[0].tag
        for result in results[1:]:
            assert result.tag == first_tag
            if first_tag == TRTag.REAL:
                assert result.value == results[0].value
    
    def test_zero_division_determinism(self):
        """Division by zero is deterministic."""
        # Same sign divisions
        for _ in range(5):
            assert is_pinf(tr_div(real(1.0), real(0.0)))
            assert is_ninf(tr_div(real(-1.0), real(0.0)))
            assert is_phi(tr_div(real(0.0), real(0.0)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
