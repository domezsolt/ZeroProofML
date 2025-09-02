"""
Focused tests for the Mask-REAL gradient rule.

These tests verify that the Mask-REAL rule correctly zeros gradients
when the forward pass produces non-REAL tags (PINF, NINF, PHI).
"""

import pytest
from zeroproof.core import real, pinf, ninf, phi, TRTag
from zeroproof.autodiff import TRNode, gradient_tape, tr_grad


class TestMaskREALBasics:
    """Basic tests for Mask-REAL gradient behavior."""
    
    def test_pinf_forward_zeros_gradient(self):
        """Test that +∞ forward values produce zero gradients."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # 1/0 → +∞
            y = TRNode.constant(real(1.0)) / x
            
        assert y.tag == TRTag.PINF
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
        assert grads[0].tag == TRTag.REAL
    
    def test_ninf_forward_zeros_gradient(self):
        """Test that -∞ forward values produce zero gradients."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # -1/0 → -∞
            y = TRNode.constant(real(-1.0)) / x
            
        assert y.tag == TRTag.NINF
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_phi_forward_zeros_gradient(self):
        """Test that Φ forward values produce zero gradients."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # 0/0 → Φ
            y = x / x
            
        assert y.tag == TRTag.PHI
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_real_forward_nonzero_gradient(self):
        """Test that REAL forward values can have non-zero gradients."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            tape.watch(x)
            
            # x² has REAL output for finite x
            y = x * x
            
        assert y.tag == TRTag.REAL
        
        grads = tape.gradient(y, [x])
        # dy/dx = 2x = 4
        assert grads[0].value.value == 4.0


class TestMaskREALPropagation:
    """Test gradient propagation through computation graphs."""
    
    def test_non_real_intermediate_blocks_gradient_flow(self):
        """Test that non-REAL intermediate values block gradient flow."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # Create path: x → 1/x (∞) → result
            intermediate = TRNode.constant(real(1.0)) / x  # +∞
            result = intermediate + TRNode.constant(real(5.0))  # +∞
            
        assert intermediate.tag == TRTag.PINF
        assert result.tag == TRTag.PINF
        
        grads = tape.gradient(result, [x])
        assert grads[0].value.value == 0.0
    
    def test_real_to_non_real_transition(self):
        """Test gradient at the boundary where computation becomes non-REAL."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            y = TRNode.parameter(real(1.0))
            tape.watch(x)
            tape.watch(y)
            
            # x - y = 0 at x=1, y=1
            diff = x - y
            # 1/(x-y) → ∞ at x=y
            result = TRNode.constant(real(1.0)) / diff
            
        assert result.tag == TRTag.PINF
        
        grads = tape.gradient(result, [x, y])
        # Both gradients should be zero due to non-REAL result
        assert grads[0].value.value == 0.0
        assert grads[1].value.value == 0.0
    
    def test_multiple_paths_with_mixed_tags(self):
        """Test gradient computation with multiple paths of different tags."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            y = TRNode.parameter(real(0.0))
            tape.watch(x)
            tape.watch(y)
            
            # Path 1: x² (REAL)
            path1 = x * x
            
            # Path 2: 1/y (PINF)
            path2 = TRNode.constant(real(1.0)) / y
            
            # Sum: REAL + PINF = PINF
            result = path1 + path2
            
        assert path1.tag == TRTag.REAL
        assert path2.tag == TRTag.PINF
        assert result.tag == TRTag.PINF
        
        grads = tape.gradient(result, [x, y])
        # Both gradients zero because result is non-REAL
        assert grads[0].value.value == 0.0
        assert grads[1].value.value == 0.0


class TestMaskREALDomainOperations:
    """Test Mask-REAL with domain-aware operations."""
    
    def test_log_invalid_domain(self):
        """Test gradient when log receives invalid input."""
        from zeroproof.autodiff import tr_log
        
        with gradient_tape() as tape:
            x = TRNode.parameter(real(-1.0))
            tape.watch(x)
            
            # log(-1) → Φ
            y = tr_log(x)
            
        assert y.tag == TRTag.PHI
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_sqrt_invalid_domain(self):
        """Test gradient when sqrt receives invalid input."""
        from zeroproof.autodiff import tr_sqrt
        
        with gradient_tape() as tape:
            x = TRNode.parameter(real(-4.0))
            tape.watch(x)
            
            # sqrt(-4) → Φ
            y = tr_sqrt(x)
            
        assert y.tag == TRTag.PHI
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_pow_zero_to_zero(self):
        """Test gradient of 0^0 which produces Φ."""
        from zeroproof.autodiff import tr_pow_int
        
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # 0^0 → Φ
            y = tr_pow_int(x, 0)
            
        assert y.tag == TRTag.PHI
        
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0


class TestMaskREALComplexScenarios:
    """Test Mask-REAL in complex computational scenarios."""
    
    def test_nested_non_real_operations(self):
        """Test deeply nested operations with non-REAL values."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # Build nested structure that produces non-REAL
            y1 = TRNode.constant(real(1.0)) / x  # +∞
            y2 = y1 * y1  # +∞ * +∞ = +∞
            y3 = y2 + TRNode.constant(real(1.0))  # +∞ + 1 = +∞
            y4 = TRNode.constant(real(2.0)) / y3  # 2/+∞ = 0 (REAL!)
            
        assert y1.tag == TRTag.PINF
        assert y2.tag == TRTag.PINF
        assert y3.tag == TRTag.PINF
        assert y4.tag == TRTag.REAL  # Back to REAL
        
        grads = tape.gradient(y4, [x])
        # Gradient should still be zero due to non-REAL intermediate
        assert grads[0].value.value == 0.0
    
    def test_function_with_singularity(self):
        """Test gradient of function with removable singularity."""
        def f(x):
            # f(x) = sin(x)/x has removable singularity at x=0
            # We approximate with (x² + 1)/x which has pole at x=0
            numerator = x * x + TRNode.constant(real(1.0))
            return numerator / x
        
        # Test at singularity
        x0 = TRNode.parameter(real(0.0))
        with gradient_tape() as tape:
            tape.watch(x0)
            y = f(x0)
        
        assert y.tag == TRTag.PINF  # (0² + 1)/0 = 1/0 = +∞
        
        grads = tape.gradient(y, [x0])
        assert grads[0].value.value == 0.0
        
        # Test near singularity (should have large gradient)
        x1 = TRNode.parameter(real(0.01))
        with gradient_tape() as tape:
            tape.watch(x1)
            y = f(x1)
        
        assert y.tag == TRTag.REAL
        grads = tape.gradient(y, [x1])
        # Gradient exists and is large near pole
        assert grads[0].tag == TRTag.REAL
        assert abs(grads[0].value.value) > 100  # Large gradient near pole
    
    def test_gradient_accumulation_with_mask_real(self):
        """Test that gradient accumulation respects Mask-REAL."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # Multiple uses of x, some producing non-REAL
            y1 = x + TRNode.constant(real(1.0))  # 1 (REAL)
            y2 = TRNode.constant(real(1.0)) / x  # +∞ (PINF)
            y3 = x * TRNode.constant(real(2.0))  # 0 (REAL)
            
            # Sum them: 1 + ∞ + 0 = ∞
            result = y1 + y2 + y3
            
        assert result.tag == TRTag.PINF
        
        grads = tape.gradient(result, [x])
        # All paths blocked due to non-REAL result
        assert grads[0].value.value == 0.0


class TestMaskREALEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_infinity_arithmetic_gradients(self):
        """Test gradients involving infinity arithmetic."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            tape.watch(x)
            
            # Create ∞ - ∞ = Φ
            inf1 = TRNode.constant(real(1.0)) / (x - TRNode.constant(real(1.0)))  # 1/0 = +∞
            inf2 = TRNode.constant(real(2.0)) / (x - TRNode.constant(real(1.0)))  # 2/0 = +∞
            result = inf1 - inf2  # ∞ - ∞ = Φ
            
        assert result.tag == TRTag.PHI
        
        grads = tape.gradient(result, [x])
        assert grads[0].value.value == 0.0
    
    def test_zero_times_infinity(self):
        """Test gradient of 0 * ∞ = Φ."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            y = TRNode.parameter(real(0.0))
            tape.watch(x)
            tape.watch(y)
            
            # 0 * ∞ = Φ
            zero = x
            infinity = TRNode.constant(real(1.0)) / y
            result = zero * infinity
            
        assert result.tag == TRTag.PHI
        
        grads = tape.gradient(result, [x, y])
        assert grads[0].value.value == 0.0
        assert grads[1].value.value == 0.0
    
    def test_signed_zero_division(self):
        """Test gradients with signed zero division."""
        # Division by +0
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))  # +0
            tape.watch(x)
            y = TRNode.constant(real(1.0)) / x
        
        assert y.tag == TRTag.PINF
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
        
        # Division by -0
        with gradient_tape() as tape:
            x = TRNode.parameter(real(-0.0))  # -0
            tape.watch(x)
            y = TRNode.constant(real(1.0)) / x
        
        assert y.tag == TRTag.NINF
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
