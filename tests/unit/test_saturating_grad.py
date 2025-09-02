"""Unit tests for saturating gradient mode."""

import pytest
import numpy as np
from zeroproof import real, pinf, ninf, phi, TRTag
from zeroproof.autodiff import (
    TRNode, gradient_tape, GradientMode, gradient_mode,
    use_mask_real, use_saturating, GradientModeConfig
)
from zeroproof.autodiff.saturating_ops import (
    saturate_value, saturating_reciprocal,
    saturating_div_grad, saturating_log_grad
)
from zeroproof.layers import SaturatingTRRational, create_saturating_rational


class TestSaturatingOperations:
    """Test saturating gradient operations."""
    
    def test_saturate_value(self):
        """Test value saturation."""
        # Small values should pass through mostly unchanged
        x = real(0.1)
        saturated = saturate_value(x, bound=1.0)
        assert saturated.tag == TRTag.REAL
        assert abs(saturated.value - 0.1 / 1.1) < 1e-10
        
        # Large values should saturate towards ±1
        x = real(100.0)
        saturated = saturate_value(x, bound=1.0)
        assert saturated.tag == TRTag.REAL
        assert abs(saturated.value - 100.0 / 101.0) < 1e-10
        assert saturated.value < 1.0
        
        # Infinities saturate to ±1
        assert saturate_value(pinf()).value == 1.0
        assert saturate_value(ninf()).value == -1.0
        assert saturate_value(phi()).value == 0.0
    
    def test_saturating_reciprocal(self):
        """Test saturating reciprocal."""
        # Normal reciprocal for large values
        x = real(10.0)
        recip = saturating_reciprocal(x, bound=0.1)
        assert recip.tag == TRTag.REAL
        # Should be close to 1/10
        assert abs(recip.value - 0.1) < 0.01
        
        # Saturated reciprocal near zero
        x = real(0.01)
        recip = saturating_reciprocal(x, bound=1.0)
        assert recip.tag == TRTag.REAL
        # Should be bounded, not exploding to 100
        assert abs(recip.value) < 1.0
        
        # Exact zero should give bounded result
        x = real(0.0)
        recip = saturating_reciprocal(x, bound=1.0)
        assert recip.tag == TRTag.REAL
        assert recip.value == 0.0  # sign(0) / 1 = 0
    
    def test_gradient_mode_switching(self):
        """Test switching between gradient modes."""
        # Default should be MASK_REAL
        assert GradientModeConfig.get_mode() == GradientMode.MASK_REAL
        
        # Switch to saturating
        use_saturating(bound=2.0)
        assert GradientModeConfig.get_mode() == GradientMode.SATURATING
        assert GradientModeConfig.get_saturation_bound() == 2.0
        
        # Switch back
        use_mask_real()
        assert GradientModeConfig.get_mode() == GradientMode.MASK_REAL
        
        # Context manager
        with gradient_mode(GradientMode.SATURATING, saturation_bound=5.0):
            assert GradientModeConfig.get_mode() == GradientMode.SATURATING
            assert GradientModeConfig.get_saturation_bound() == 5.0
        
        # Should revert
        assert GradientModeConfig.get_mode() == GradientMode.MASK_REAL


class TestSaturatingGradients:
    """Test gradient computation in saturating mode."""
    
    def test_division_gradients(self):
        """Test division gradients with saturation."""
        # Create nodes
        x = TRNode.parameter(real(1.0))
        y = TRNode.parameter(real(0.1))  # Small denominator
        
        # Compute in MASK_REAL mode
        with gradient_mode(GradientMode.MASK_REAL):
            z = x / y
            z.backward()
            mask_grad_x = x.gradient.value if x.gradient else 0.0
            mask_grad_y = y.gradient.value if y.gradient else 0.0
        
        # Reset gradients
        x.zero_grad()
        y.zero_grad()
        
        # Compute in SATURATING mode
        with gradient_mode(GradientMode.SATURATING, saturation_bound=1.0):
            z = x / y
            z.backward()
            sat_grad_x = x.gradient.value if x.gradient else 0.0
            sat_grad_y = y.gradient.value if y.gradient else 0.0
        
        # Saturating should have bounded gradients
        assert abs(mask_grad_x) > abs(sat_grad_x)  # 1/0.1 = 10 vs bounded
        assert abs(mask_grad_y) > abs(sat_grad_y)  # -1/0.01 = -100 vs bounded
    
    def test_singularity_handling(self):
        """Test gradient behavior at singularities."""
        # Division by zero
        x = TRNode.parameter(real(1.0))
        y = TRNode.parameter(real(0.0))
        
        # MASK_REAL mode - should zero gradients
        with gradient_mode(GradientMode.MASK_REAL):
            z = x / y  # Will be PINF
            assert z.tag == TRTag.PINF
            z.backward()
            assert x.gradient.value == 0.0  # Masked
            assert y.gradient.value == 0.0  # Masked
        
        # Reset
        x.zero_grad()
        y.zero_grad()
        
        # SATURATING mode - should give bounded gradients
        with gradient_mode(GradientMode.SATURATING, saturation_bound=1.0):
            z = x / y  # Still PINF
            assert z.tag == TRTag.PINF
            z.backward()
            # Gradients should be computed even for non-REAL
            assert x.gradient is not None
            assert y.gradient is not None
    
    def test_log_gradient_saturation(self):
        """Test log gradient saturation near zero."""
        # Log near zero
        x = TRNode.parameter(real(0.01))
        
        # Standard gradient
        with gradient_mode(GradientMode.MASK_REAL):
            y = TRNode.constant(real(0.0))
            for ref in x._gradient_tape_stack:
                ref._watch(x)
            from zeroproof.autodiff import tr_log
            z = tr_log(x)
            z.backward()
            standard_grad = x.gradient.value  # Should be 1/0.01 = 100
        
        # Reset
        x.zero_grad()
        
        # Saturating gradient
        with gradient_mode(GradientMode.SATURATING, saturation_bound=1.0):
            y = TRNode.constant(real(0.0))
            for ref in x._gradient_tape_stack:
                ref._watch(x)
            z = tr_log(x)
            z.backward()
            saturating_grad = x.gradient.value  # Should be bounded
        
        # Saturating should be smaller
        assert abs(saturating_grad) < abs(standard_grad)


class TestSaturatingRational:
    """Test saturating rational layer."""
    
    def test_layer_creation(self):
        """Test creating saturating rational layer."""
        # Default mask-real
        layer1 = create_saturating_rational(3, 2, mode="mask-real")
        assert layer1.gradient_mode == GradientMode.MASK_REAL
        
        # Saturating mode
        layer2 = create_saturating_rational(3, 2, mode="saturating", saturation_bound=2.0)
        assert layer2.gradient_mode == GradientMode.SATURATING
        assert layer2.saturation_bound == 2.0
    
    def test_forward_with_mode(self):
        """Test forward pass with different modes."""
        layer = SaturatingTRRational(2, 2, gradient_mode=GradientMode.MASK_REAL)
        
        # Test with layer's default mode
        x = real(0.5)
        y1, tag1 = layer.forward(x)
        
        # Test with override
        y2, tag2 = layer.forward_with_mode(x, mode=GradientMode.SATURATING)
        
        # Results should be the same (forward pass unchanged)
        assert tag1 == tag2
        assert abs(y1.value.value - y2.value.value) < 1e-10
    
    def test_gradient_comparison(self):
        """Test comparing gradients between modes."""
        layer = SaturatingTRRational(2, 1, saturation_bound=1.0)
        
        # Create test batch near a potential singularity
        x_batch = [real(0.0), real(0.1), real(0.5), real(1.0)]
        
        # Compare modes
        results = layer.compare_gradient_modes(x_batch)
        
        # Check structure
        assert 'mask_real' in results
        assert 'saturating' in results
        assert len(results['mask_real']['gradients']) == 4
        assert len(results['saturating']['gradients']) == 4
        
        # At x=0 (potential singularity), gradients should differ
        mask_grads = results['mask_real']['gradients'][0]
        sat_grads = results['saturating']['gradients'][0]
        
        # If output was non-REAL, mask-real should have zeros
        if results['mask_real']['tags'][0] != TRTag.REAL:
            assert all(g == 0.0 for g in mask_grads)


class TestIntegration:
    """Integration tests for saturating gradients."""
    
    def test_training_with_saturating(self):
        """Test training behavior with saturating gradients."""
        from zeroproof.training import Optimizer
        
        # Create layer with saturating mode
        layer = SaturatingTRRational(
            d_p=2, d_q=1,
            gradient_mode=GradientMode.SATURATING,
            saturation_bound=5.0
        )
        
        # Simple optimizer
        optimizer = Optimizer(layer.parameters(), learning_rate=0.01)
        
        # Training step near singularity
        x = real(0.01)  # Near zero, where Q might be small
        target = real(2.0)
        
        # Forward
        y, _ = layer.forward(x)
        
        # Loss
        loss = (y - TRNode.constant(target)) ** 2
        
        # Backward
        loss.backward()
        
        # Check gradients are bounded
        for param in layer.parameters():
            if param.gradient is not None and param.gradient.tag == TRTag.REAL:
                # Gradients should be reasonable, not exploding
                assert abs(param.gradient.value) < 100.0
        
        # Update should succeed
        optimizer.step()
