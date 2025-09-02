"""Unit tests for transreal reduction operations."""

import pytest
from zeroproof.core import (
    real, pinf, ninf, phi, TRTag,
    tr_sum, tr_mean, tr_prod, tr_min, tr_max,
    ReductionMode
)


class TestSumReduction:
    """Test transreal sum reduction."""
    
    def test_sum_strict_mode(self):
        """Test sum with STRICT mode."""
        # All REAL values
        values = [real(1.0), real(2.0), real(3.0)]
        assert tr_sum(values).value == 6.0
        
        # With PHI - should return PHI
        values = [real(1.0), phi(), real(3.0)]
        assert tr_sum(values).tag == TRTag.PHI
        
        # With infinities
        values = [real(1.0), pinf(), real(3.0)]
        assert tr_sum(values).tag == TRTag.PINF
        
        # Conflicting infinities
        values = [pinf(), ninf()]
        assert tr_sum(values).tag == TRTag.PHI
    
    def test_sum_drop_null_mode(self):
        """Test sum with DROP_NULL mode."""
        # Mixed with PHI
        values = [real(1.0), phi(), real(3.0)]
        result = tr_sum(values, ReductionMode.DROP_NULL)
        assert result.value == 4.0
        
        # All PHI
        values = [phi(), phi()]
        assert tr_sum(values, ReductionMode.DROP_NULL).tag == TRTag.PHI
        
        # Empty after dropping PHI
        values = []
        assert tr_sum(values, ReductionMode.DROP_NULL).value == 0.0


class TestMeanReduction:
    """Test transreal mean reduction."""
    
    def test_mean_strict_mode(self):
        """Test mean with STRICT mode."""
        # All REAL
        values = [real(1.0), real(2.0), real(3.0)]
        assert tr_mean(values).value == 2.0
        
        # With PHI
        values = [real(1.0), phi(), real(3.0)]
        assert tr_mean(values).tag == TRTag.PHI
    
    def test_mean_drop_null_mode(self):
        """Test mean with DROP_NULL mode."""
        # Mixed with PHI
        values = [real(2.0), phi(), real(4.0)]
        result = tr_mean(values, ReductionMode.DROP_NULL)
        assert result.value == 3.0  # (2+4)/2
        
        # Division by zero case
        values = [real(0.0)]
        result = tr_mean(values)
        assert result.value == 0.0


class TestMinMaxReduction:
    """Test min/max reductions."""
    
    def test_min_reduction(self):
        """Test minimum reduction."""
        # All REAL
        values = [real(3.0), real(1.0), real(2.0)]
        assert tr_min(values).value == 1.0
        
        # With NINF
        values = [real(1.0), ninf(), real(2.0)]
        assert tr_min(values).tag == TRTag.NINF
        
        # With PHI in strict mode
        values = [real(1.0), phi(), real(2.0)]
        assert tr_min(values).tag == TRTag.PHI
        
        # Drop null mode
        values = [real(3.0), phi(), real(1.0)]
        assert tr_min(values, ReductionMode.DROP_NULL).value == 1.0
    
    def test_max_reduction(self):
        """Test maximum reduction."""
        # All REAL
        values = [real(1.0), real(3.0), real(2.0)]
        assert tr_max(values).value == 3.0
        
        # With PINF
        values = [real(1.0), pinf(), real(2.0)]
        assert tr_max(values).tag == TRTag.PINF
        
        # Order test: NINF < REAL < PINF
        values = [ninf(), real(0.0), pinf()]
        assert tr_max(values).tag == TRTag.PINF
        assert tr_min(values).tag == TRTag.NINF


class TestProductReduction:
    """Test product reduction."""
    
    def test_product_reduction(self):
        """Test product with various cases."""
        # All REAL
        values = [real(2.0), real(3.0), real(4.0)]
        assert tr_prod(values).value == 24.0
        
        # With zero and infinity
        values = [real(0.0), pinf()]
        assert tr_prod(values).tag == TRTag.PHI  # 0 × ∞ = PHI
        
        # Empty list
        assert tr_prod([]).value == 1.0  # Multiplicative identity
