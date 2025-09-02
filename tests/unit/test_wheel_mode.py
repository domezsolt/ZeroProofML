"""Unit tests for wheel mode."""

import pytest
import zeroproof as zp
from zeroproof import real, pinf, ninf, phi, bottom, TRTag
from zeroproof.core import tr_add, tr_mul, tr_div, ArithmeticMode, wheel_mode, use_wheel, use_transreal


class TestWheelMode:
    """Test wheel mode arithmetic behavior."""
    
    def setup_method(self):
        """Reset to transreal mode before each test."""
        use_transreal()
    
    def test_mode_switching(self):
        """Test switching between transreal and wheel modes."""
        # Default should be transreal
        from zeroproof.core import WheelModeConfig
        assert WheelModeConfig.is_transreal()
        assert not WheelModeConfig.is_wheel()
        
        # Switch to wheel mode
        use_wheel()
        assert WheelModeConfig.is_wheel()
        assert not WheelModeConfig.is_transreal()
        
        # Switch back
        use_transreal()
        assert WheelModeConfig.is_transreal()
        assert not WheelModeConfig.is_wheel()
    
    def test_context_manager(self):
        """Test wheel mode context manager."""
        # Start in transreal mode
        assert zp.tr_mul(real(0.0), pinf()).tag == TRTag.PHI
        
        # Use wheel mode temporarily
        with wheel_mode():
            # 0 × ∞ = ⊥ in wheel mode
            result = zp.tr_mul(real(0.0), pinf())
            assert result.tag == TRTag.BOTTOM
        
        # Back to transreal mode
        assert zp.tr_mul(real(0.0), pinf()).tag == TRTag.PHI
    
    def test_zero_times_infinity(self):
        """Test 0 × ∞ in different modes."""
        # Transreal mode: 0 × ∞ = Φ
        result_tr = tr_mul(real(0.0), pinf())
        assert result_tr.tag == TRTag.PHI
        
        result_tr2 = tr_mul(ninf(), real(0.0))
        assert result_tr2.tag == TRTag.PHI
        
        # Wheel mode: 0 × ∞ = ⊥
        with wheel_mode():
            result_w = tr_mul(real(0.0), pinf())
            assert result_w.tag == TRTag.BOTTOM
            
            result_w2 = tr_mul(ninf(), real(0.0))
            assert result_w2.tag == TRTag.BOTTOM
    
    def test_infinity_plus_infinity(self):
        """Test ∞ + ∞ in different modes."""
        # Transreal mode: +∞ + +∞ = +∞
        result_tr1 = tr_add(pinf(), pinf())
        assert result_tr1.tag == TRTag.PINF
        
        # Transreal mode: -∞ + -∞ = -∞
        result_tr2 = tr_add(ninf(), ninf())
        assert result_tr2.tag == TRTag.NINF
        
        # Transreal mode: +∞ + -∞ = Φ
        result_tr3 = tr_add(pinf(), ninf())
        assert result_tr3.tag == TRTag.PHI
        
        # Wheel mode: ∞ + ∞ = ⊥
        with wheel_mode():
            result_w1 = tr_add(pinf(), pinf())
            assert result_w1.tag == TRTag.BOTTOM
            
            result_w2 = tr_add(ninf(), ninf())
            assert result_w2.tag == TRTag.BOTTOM
            
            result_w3 = tr_add(pinf(), ninf())
            assert result_w3.tag == TRTag.BOTTOM
    
    def test_infinity_minus_infinity(self):
        """Test ∞ - ∞ in different modes."""
        # In both modes, this is like +∞ + -∞
        # Transreal: Φ
        result_tr = tr_add(pinf(), tr_neg(pinf()))
        assert result_tr.tag == TRTag.PHI
        
        # Wheel: ⊥
        with wheel_mode():
            result_w = tr_add(pinf(), tr_neg(pinf()))
            assert result_w.tag == TRTag.BOTTOM
    
    def test_infinity_div_infinity(self):
        """Test ∞ / ∞ in different modes."""
        # Transreal mode: ∞ / ∞ = Φ
        result_tr = tr_div(pinf(), pinf())
        assert result_tr.tag == TRTag.PHI
        
        result_tr2 = tr_div(ninf(), pinf())
        assert result_tr2.tag == TRTag.PHI
        
        # Wheel mode: ∞ / ∞ = ⊥
        with wheel_mode():
            result_w = tr_div(pinf(), pinf())
            assert result_w.tag == TRTag.BOTTOM
            
            result_w2 = tr_div(ninf(), pinf())
            assert result_w2.tag == TRTag.BOTTOM
    
    def test_bottom_propagation(self):
        """Test that BOTTOM propagates through operations."""
        with wheel_mode():
            # Create a bottom value
            b = bottom()
            assert b.tag == TRTag.BOTTOM
            
            # BOTTOM + anything = BOTTOM
            assert tr_add(b, real(5.0)).tag == TRTag.BOTTOM
            assert tr_add(real(5.0), b).tag == TRTag.BOTTOM
            assert tr_add(b, pinf()).tag == TRTag.BOTTOM
            
            # BOTTOM × anything = BOTTOM
            assert tr_mul(b, real(5.0)).tag == TRTag.BOTTOM
            assert tr_mul(real(5.0), b).tag == TRTag.BOTTOM
            assert tr_mul(b, pinf()).tag == TRTag.BOTTOM
            
            # BOTTOM ÷ anything = BOTTOM
            assert tr_div(b, real(5.0)).tag == TRTag.BOTTOM
            assert tr_div(real(5.0), b).tag == TRTag.BOTTOM
            
            # Unary operations on BOTTOM = BOTTOM
            assert tr_neg(b).tag == TRTag.BOTTOM
            assert tr_abs(b).tag == TRTag.BOTTOM
            assert tr_sign(b).tag == TRTag.BOTTOM
            assert tr_log(b).tag == TRTag.BOTTOM
            assert tr_sqrt(b).tag == TRTag.BOTTOM
            assert tr_pow_int(b, 2).tag == TRTag.BOTTOM
    
    def test_mixed_operations(self):
        """Test complex expressions in wheel mode."""
        with wheel_mode():
            # (0 × ∞) + 5 = ⊥ + 5 = ⊥
            result1 = tr_add(tr_mul(real(0.0), pinf()), real(5.0))
            assert result1.tag == TRTag.BOTTOM
            
            # 1 / (∞ - ∞) = 1 / ⊥ = ⊥
            result2 = tr_div(real(1.0), tr_add(pinf(), ninf()))
            assert result2.tag == TRTag.BOTTOM
            
            # sqrt(0 × ∞) = sqrt(⊥) = ⊥
            result3 = tr_sqrt(tr_mul(real(0.0), pinf()))
            assert result3.tag == TRTag.BOTTOM
    
    def test_ieee_bridge(self):
        """Test IEEE conversion with BOTTOM values."""
        with wheel_mode():
            # Create a bottom value
            b = bottom()
            
            # BOTTOM converts to NaN
            ieee_val = zp.to_ieee(b)
            assert math.isnan(ieee_val)
            
            # NaN still converts to PHI (not BOTTOM)
            # because from_ieee doesn't know about wheel mode
            tr_val = zp.from_ieee(float('nan'))
            assert tr_val.tag == TRTag.PHI
    
    def test_display(self):
        """Test string representation of BOTTOM."""
        b = bottom()
        assert str(b) == "⊥"
        assert "BOTTOM" in repr(b)


class TestWheelModeVsTransreal:
    """Compare behavior between wheel and transreal modes."""
    
    def test_comparison_table(self):
        """Test key differences between modes."""
        test_cases = [
            # (operation, expected_transreal, expected_wheel)
            (lambda: tr_mul(real(0.0), pinf()), TRTag.PHI, TRTag.BOTTOM),
            (lambda: tr_mul(real(0.0), ninf()), TRTag.PHI, TRTag.BOTTOM),
            (lambda: tr_add(pinf(), pinf()), TRTag.PINF, TRTag.BOTTOM),
            (lambda: tr_add(ninf(), ninf()), TRTag.NINF, TRTag.BOTTOM),
            (lambda: tr_add(pinf(), ninf()), TRTag.PHI, TRTag.BOTTOM),
            (lambda: tr_div(pinf(), pinf()), TRTag.PHI, TRTag.BOTTOM),
            (lambda: tr_div(ninf(), ninf()), TRTag.PHI, TRTag.BOTTOM),
        ]
        
        for operation, expected_tr, expected_wheel in test_cases:
            # Test in transreal mode
            use_transreal()
            result_tr = operation()
            assert result_tr.tag == expected_tr
            
            # Test in wheel mode
            use_wheel()
            result_wheel = operation()
            assert result_wheel.tag == expected_wheel
        
        # Reset to transreal
        use_transreal()
    
    def test_normal_operations_unchanged(self):
        """Test that normal operations work the same in both modes."""
        operations = [
            lambda: tr_add(real(2.0), real(3.0)),
            lambda: tr_mul(real(2.0), real(3.0)),
            lambda: tr_div(real(6.0), real(2.0)),
            lambda: tr_add(real(5.0), pinf()),
            lambda: tr_mul(real(5.0), pinf()),
            lambda: tr_div(real(5.0), pinf()),
            lambda: tr_log(real(2.718)),
            lambda: tr_sqrt(real(4.0)),
        ]
        
        for op in operations:
            # Get result in transreal mode
            use_transreal()
            result_tr = op()
            
            # Get result in wheel mode
            use_wheel()
            result_wheel = op()
            
            # Should be the same
            assert result_tr.tag == result_wheel.tag
            if result_tr.tag == TRTag.REAL:
                assert abs(result_tr.value - result_wheel.value) < 1e-10
        
        # Reset
        use_transreal()


# Import necessary functions
from zeroproof.core import tr_neg, tr_abs, tr_sign, tr_log, tr_sqrt, tr_pow_int
import math
