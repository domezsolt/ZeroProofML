"""
Tests for L1 projection in TR-Rational layers.

This module tests the L1 projection functionality that ensures
denominator coefficients stay within a bounded region for stability.
"""

import pytest
import math
from typing import List

from zeroproof.core import TRScalar, TRTag, real, pinf, ninf, phi
from zeroproof.autodiff import TRNode
from zeroproof.layers import TRRational, TRRationalMulti
from zeroproof.training import Optimizer


class TestL1Projection:
    """Test L1 projection functionality."""
    
    def test_no_projection_when_disabled(self):
        """Test that no projection occurs when l1_projection is None."""
        layer = TRRational(d_p=2, d_q=2, l1_projection=None)
        
        # Set large phi values
        for phi_k in layer.phi:
            phi_k._value = real(10.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Values should remain unchanged
        for phi_k in layer.phi:
            assert phi_k.value.value == 10.0
    
    def test_projection_when_norm_exceeds_bound(self):
        """Test that projection occurs when L1 norm exceeds bound."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Set phi values that exceed L1 bound
        # φ = [2.0, 3.0], ||φ||₁ = 5.0 > 1.0
        layer.phi[0]._value = real(2.0)
        layer.phi[1]._value = real(3.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Check that L1 norm is now at the bound
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert abs(l1_norm - bound) < 1e-10
        
        # Check that ratios are preserved
        # After projection: φ = [0.4, 0.6]
        assert abs(layer.phi[0].value.value - 0.4) < 1e-10
        assert abs(layer.phi[1].value.value - 0.6) < 1e-10
    
    def test_no_projection_when_norm_within_bound(self):
        """Test that no projection occurs when L1 norm is within bound."""
        bound = 10.0
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Set phi values within L1 bound
        layer.phi[0]._value = real(1.0)
        layer.phi[1]._value = real(2.0)
        
        # Store original values
        orig_values = [phi_k.value.value for phi_k in layer.phi]
        
        # Apply projection
        layer._project_phi_l1()
        
        # Values should remain unchanged
        for i, phi_k in enumerate(layer.phi):
            assert phi_k.value.value == orig_values[i]
    
    def test_projection_with_negative_values(self):
        """Test projection with negative coefficient values."""
        bound = 2.0
        layer = TRRational(d_p=2, d_q=3, l1_projection=bound)
        
        # Set mixed positive/negative values
        # φ = [3.0, -2.0, 1.0], ||φ||₁ = 6.0 > 2.0
        layer.phi[0]._value = real(3.0)
        layer.phi[1]._value = real(-2.0)
        layer.phi[2]._value = real(1.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Check L1 norm
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert abs(l1_norm - bound) < 1e-10
        
        # Check that signs are preserved and ratios maintained
        # Scale factor = 2.0 / 6.0 = 1/3
        assert abs(layer.phi[0].value.value - 1.0) < 1e-10
        assert abs(layer.phi[1].value.value - (-2.0/3)) < 1e-10
        assert abs(layer.phi[2].value.value - (1.0/3)) < 1e-10
    
    def test_projection_with_non_real_values(self):
        """Test that non-REAL values are ignored in projection."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=3, l1_projection=bound)
        
        # Set mixed REAL and non-REAL values
        layer.phi[0]._value = real(2.0)
        layer.phi[1]._value = pinf()
        layer.phi[2]._value = real(3.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Only REAL values should be projected
        # L1 norm of REAL values = 5.0, scale = 1.0/5.0 = 0.2
        assert abs(layer.phi[0].value.value - 0.4) < 1e-10
        assert layer.phi[1].value.tag == TRTag.PINF  # Unchanged
        assert abs(layer.phi[2].value.value - 0.6) < 1e-10
    
    def test_projection_in_forward_pass(self):
        """Test that projection is applied during forward pass."""
        bound = 0.5
        layer = TRRational(d_p=1, d_q=2, l1_projection=bound)
        
        # Set phi values that exceed bound
        layer.phi[0]._value = real(1.0)
        layer.phi[1]._value = real(1.0)
        
        # Forward pass should apply projection
        x = real(1.0)
        y, tag = layer.forward(x)
        
        # Check that projection was applied
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert abs(l1_norm - bound) < 1e-10
    
    def test_gradient_scaling_during_projection(self):
        """Test that gradients are also scaled during projection."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Set phi values and gradients
        layer.phi[0]._value = real(2.0)
        layer.phi[1]._value = real(3.0)
        layer.phi[0]._gradient = TRNode.constant(real(0.5))
        layer.phi[1]._gradient = TRNode.constant(real(1.0))
        
        # Apply projection
        layer._project_phi_l1()
        
        # Check that gradients are scaled by same factor as values
        # Scale factor = 1.0 / 5.0 = 0.2
        assert abs(layer.phi[0].gradient.value.value - 0.1) < 1e-10  # 0.5 * 0.2
        assert abs(layer.phi[1].gradient.value.value - 0.2) < 1e-10  # 1.0 * 0.2
    
    def test_projection_with_optimizer_integration(self):
        """Test L1 projection integration with optimizer."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Create optimizer
        optimizer = Optimizer(layer.parameters(), learning_rate=0.1)
        
        # Set gradients that would push phi outside L1 ball
        for phi_k in layer.phi:
            phi_k._gradient = TRNode.constant(real(-5.0))
        
        # Optimization step with model reference
        optimizer.step(model=layer)
        
        # Check that L1 constraint is satisfied after update
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert l1_norm <= bound + 1e-10
    
    def test_multi_output_shared_denominator_projection(self):
        """Test L1 projection with multi-output layer and shared denominator."""
        bound = 1.5
        multi_layer = TRRationalMulti(
            d_p=2, d_q=2, n_outputs=3, 
            shared_Q=True
        )
        
        # Manually set l1_projection for the shared layer
        multi_layer.layers[0].l1_projection = bound
        
        # Set large phi values in shared denominator
        shared_phi = multi_layer.layers[0].phi
        shared_phi[0]._value = real(3.0)
        shared_phi[1]._value = real(4.0)
        
        # Apply projection through first layer
        multi_layer.layers[0]._project_phi_l1()
        
        # Check that all layers see the projected values
        for layer in multi_layer.layers:
            l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
            assert abs(l1_norm - bound) < 1e-10
    
    def test_projection_preserves_zero_coefficients(self):
        """Test that zero coefficients remain zero after projection."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=3, l1_projection=bound)
        
        # Set some zero and non-zero values
        layer.phi[0]._value = real(3.0)
        layer.phi[1]._value = real(0.0)
        layer.phi[2]._value = real(2.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Check that zero remains zero
        assert layer.phi[1].value.value == 0.0
        
        # Check L1 norm
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert abs(l1_norm - bound) < 1e-10
    
    def test_projection_with_very_small_bound(self):
        """Test projection with very small L1 bound."""
        bound = 1e-6
        layer = TRRational(d_p=1, d_q=2, l1_projection=bound)
        
        # Set normal-sized phi values
        layer.phi[0]._value = real(1.0)
        layer.phi[1]._value = real(2.0)
        
        # Apply projection
        layer._project_phi_l1()
        
        # Check that values are scaled down significantly
        l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        assert abs(l1_norm - bound) < 1e-15
        
        # Values should be very small
        assert abs(layer.phi[0].value.value) < 1e-6
        assert abs(layer.phi[1].value.value) < 1e-6
    
    def test_projection_stability_over_iterations(self):
        """Test that repeated projections maintain stability."""
        bound = 1.0
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Simulate multiple optimization steps
        for _ in range(10):
            # Randomly perturb phi values
            for phi_k in layer.phi:
                phi_k._value = real(phi_k.value.value + 0.5)
            
            # Apply projection
            layer._project_phi_l1()
            
            # Check L1 constraint
            l1_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
            assert l1_norm <= bound + 1e-10
    
    def test_compute_q_min_with_projection(self):
        """Test that Q minimum tracking works with L1 projection."""
        bound = 0.5
        layer = TRRational(d_p=2, d_q=2, l1_projection=bound)
        
        # Set phi values
        layer.phi[0]._value = real(0.3)
        layer.phi[1]._value = real(0.2)
        
        # Create batch of inputs
        x_batch = [real(0.0), real(1.0), real(-1.0), real(2.0)]
        
        # Compute minimum |Q(x)|
        q_min = layer.compute_q_min(x_batch)
        
        # With small phi values, Q should stay away from zero
        assert q_min > 0.4  # Q(x) = 1 + small terms
    
    def test_invalid_projection_bound(self):
        """Test that invalid projection bounds are handled correctly."""
        # Test with zero bound (should be ignored)
        layer = TRRational(d_p=2, d_q=2, l1_projection=0.0)
        
        # Set phi values
        layer.phi[0]._value = real(1.0)
        layer.phi[1]._value = real(2.0)
        
        orig_values = [phi_k.value.value for phi_k in layer.phi]
        
        # Apply projection (should do nothing)
        layer._project_phi_l1()
        
        # Values should be unchanged
        for i, phi_k in enumerate(layer.phi):
            assert phi_k.value.value == orig_values[i]
        
        # Test with negative bound (should be ignored)
        layer.l1_projection = -1.0
        layer._project_phi_l1()
        
        # Values should still be unchanged
        for i, phi_k in enumerate(layer.phi):
            assert phi_k.value.value == orig_values[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
