"""Unit tests for TR-Norm layers."""

import pytest
import math
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from zeroproof.core import real, pinf, ninf, phi, TRTag
from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.layers import TRNorm, TRLayerNorm


class TestTRNorm:
    """Test TR-Norm (batch normalization) functionality."""
    
    def test_initialization(self):
        """Test layer initialization."""
        # Basic initialization
        norm = TRNorm(num_features=3)
        assert norm.num_features == 3
        assert norm.affine
        assert len(norm.gamma) == 3
        assert len(norm.beta) == 3
        
        # Check parameters
        for i in range(3):
            assert norm.gamma[i].value.value == 1.0  # γ initialized to 1
            assert norm.beta[i].value.value == 0.0   # β initialized to 0
        
        # Without affine parameters
        norm = TRNorm(num_features=2, affine=False)
        assert norm.gamma is None
        assert norm.beta is None
    
    def test_forward_normal_case(self):
        """Test forward pass with normal variance."""
        norm = TRNorm(num_features=2)
        
        # Input batch: 3 samples, 2 features
        x = [
            [real(1.0), real(4.0)],
            [real(2.0), real(5.0)],
            [real(3.0), real(6.0)],
        ]
        
        output = norm(x)
        
        # Feature 0: mean=2, var=2/3, std=sqrt(2/3)
        # Feature 1: mean=5, var=2/3, std=sqrt(2/3)
        
        # Check output shape
        assert len(output) == 3
        assert len(output[0]) == 2
        
        # Check normalization
        # x[0,0] normalized: (1-2)/sqrt(2/3) = -sqrt(3/2) ≈ -1.225
        expected = -math.sqrt(3/2)
        assert output[0][0].value.value == pytest.approx(expected, rel=1e-5)
    
    def test_forward_zero_variance_bypass(self):
        """Test bypass when variance is zero."""
        norm = TRNorm(num_features=2)
        
        # Set beta values for testing
        norm.beta[0]._value = real(5.0)
        norm.beta[1]._value = real(-3.0)
        
        # Input with zero variance (all same values)
        x = [
            [real(2.0), real(7.0)],
            [real(2.0), real(7.0)],
            [real(2.0), real(7.0)],
        ]
        
        output = norm(x)
        
        # With zero variance, output should be β
        for i in range(3):
            assert output[i][0].value.value == 5.0   # β₀
            assert output[i][1].value.value == -3.0  # β₁

    def test_running_stats_tracking(self):
        """Ensure running stats tracking does not raise and updates values."""
        norm = TRNorm(num_features=2, track_running_stats=True, momentum=0.5)
        
        # Two small batches
        x1 = [
            [real(1.0), real(4.0)],
            [real(3.0), real(6.0)],
        ]
        x2 = [
            [real(2.0), real(5.0)],
            [real(4.0), real(7.0)],
        ]
        
        # Forward passes should update running stats
        out1 = norm(x1)
        out2 = norm(x2)
        
        # Check counters and that stats moved away from initial values
        assert norm.num_batches_tracked is not None
        assert norm.num_batches_tracked >= 1
        assert norm.running_mean is not None and norm.running_var is not None
        # Means should be between initial (0.0) and batch means (~2.0, ~5.0)
        assert 0.0 <= norm.running_mean[0] <= 3.0
        assert 0.0 <= norm.running_mean[1] <= 7.0
        # Vars should be positive and finite
        assert norm.running_var[0] > 0.0
        assert norm.running_var[1] > 0.0
    
    def test_forward_with_non_real_values(self):
        """Test normalization with non-REAL values (drop-null)."""
        norm = TRNorm(num_features=1)
        
        # Mix of REAL and non-REAL values
        inf_node = TRNode.constant(pinf())
        phi_node = TRNode.constant(phi())
        
        x = [
            [real(1.0)],
            [inf_node],   # Will be dropped
            [real(3.0)],
            [phi_node],   # Will be dropped
            [real(5.0)],
        ]
        
        output = norm(x)
        
        # Stats computed only over REAL values: 1, 3, 5
        # mean = 3, var = 8/3
        
        # Check that all outputs have correct shape
        assert len(output) == 5
        
        # REAL inputs should be normalized
        assert output[0][0].tag == TRTag.REAL
        assert output[2][0].tag == TRTag.REAL
        assert output[4][0].tag == TRTag.REAL
        
        # Non-REAL inputs still produce outputs (but may be non-REAL)
        assert output[1][0].tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
        assert output[3][0].tag in {TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI}
    
    def test_all_non_real_triggers_bypass(self):
        """Test that all non-REAL values trigger bypass."""
        norm = TRNorm(num_features=1)
        norm.beta[0]._value = real(42.0)
        
        # All non-REAL
        x = [
            [TRNode.constant(pinf())],
            [TRNode.constant(ninf())],
            [TRNode.constant(phi())],
        ]
        
        output = norm(x)
        
        # Should bypass to beta since no REAL values
        for i in range(3):
            if output[i][0].tag == TRTag.REAL:
                assert output[i][0].value.value == 42.0
    
    def test_gradient_flow_normal_case(self):
        """Test gradient flow through normalization."""
        norm = TRNorm(num_features=1)
        
        with gradient_tape() as tape:
            # Create input
            x_vals = []
            for val in [1.0, 2.0, 3.0]:
                node = TRNode.parameter(real(val))
                tape.watch(node)
                x_vals.append(node)
            
            x = [[x_vals[0]], [x_vals[1]], [x_vals[2]]]
            
            # Forward pass
            output = norm(x)
            
            # Sum outputs for scalar loss
            loss = output[0][0] + output[1][0] + output[2][0]
        
        # Compute gradients
        grads = tape.gradient(loss, x_vals)
        
        # Gradients should be computed
        for grad in grads:
            assert grad is not None
            assert grad.tag == TRTag.REAL
    
    def test_gradient_flow_bypass_case(self):
        """Test gradient flow when bypass is triggered."""
        norm = TRNorm(num_features=1)
        
        with gradient_tape() as tape:
            # Watch beta parameter
            tape.watch(norm.beta[0])
            
            # Input with zero variance
            x = [[real(5.0)], [real(5.0)], [real(5.0)]]
            
            # Forward pass
            output = norm(x)
            
            # Loss
            loss = output[0][0] + output[1][0] + output[2][0]
        
        # Gradient w.r.t beta
        grads = tape.gradient(loss, [norm.beta[0]])
        
        # In bypass mode, d(loss)/d(beta) = 3 (batch size)
        assert grads[0].value.value == 3.0


class TestTRLayerNorm:
    """Test TR Layer Normalization."""
    
    def test_initialization(self):
        """Test layer norm initialization."""
        ln = TRLayerNorm(normalized_shape=4)
        assert ln.num_features == 4
        assert ln.elementwise_affine
        assert len(ln.gamma) == 4
        assert len(ln.beta) == 4
    
    def test_forward_single_sample(self):
        """Test layer norm on single sample."""
        ln = TRLayerNorm(4)
        
        # Input: one sample with 4 features
        x = [real(1.0), real(2.0), real(3.0), real(4.0)]
        
        output = ln(x)
        
        # mean = 2.5, var = 1.25
        assert len(output) == 4
        
        # Check first element: (1 - 2.5) / sqrt(1.25)
        expected = -1.5 / math.sqrt(1.25)
        assert output[0].value.value == pytest.approx(expected, rel=1e-5)
    
    def test_zero_variance_bypass(self):
        """Test bypass with constant features."""
        ln = TRLayerNorm(3)
        ln.beta[0]._value = real(10.0)
        ln.beta[1]._value = real(20.0)
        ln.beta[2]._value = real(30.0)
        
        # All features have same value
        x = [real(7.0), real(7.0), real(7.0)]
        
        output = ln(x)
        
        # Should bypass to beta values
        assert output[0].value.value == 10.0
        assert output[1].value.value == 20.0
        assert output[2].value.value == 30.0
    
    def test_mixed_real_non_real(self):
        """Test with mixed REAL and non-REAL features."""
        ln = TRLayerNorm(4)
        
        x = [
            real(1.0),
            TRNode.constant(pinf()),
            real(3.0),
            TRNode.constant(phi()),
        ]
        
        output = ln(x)
        
        # Stats computed over REAL values only: 1.0, 3.0
        # mean = 2, var = 1
        
        assert len(output) == 4
        # First REAL: (1-2)/1 = -1
        assert output[0].value.value == pytest.approx(-1.0)
        # Third REAL: (3-2)/1 = 1
        assert output[2].value.value == pytest.approx(1.0)


class TestNormalizationProperties:
    """Test mathematical properties of normalization."""
    
    def test_output_statistics(self):
        """Test that normalized outputs have mean≈0, var≈1."""
        norm = TRNorm(num_features=1, affine=False)
        
        # Large batch for better statistics
        x = [[real(float(i))] for i in range(100)]
        
        output = norm(x)
        
        # Compute output statistics
        values = [out[0].value.value for out in output]
        mean = sum(values) / len(values)
        var = sum((v - mean)**2 for v in values) / len(values)
        
        # Should be approximately standard normal
        assert abs(mean) < 0.01
        assert abs(var - 1.0) < 0.01
    
    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(st.lists(st.floats(min_value=-10, max_value=10, 
                             allow_nan=False, allow_infinity=False),
                   min_size=2, max_size=20))
    def test_invariance_to_affine_transform(self, values):
        """Test that normalization is invariant to affine input transforms."""
        assume(len(set(values)) > 1)  # Need variance > 0
        
        norm = TRNorm(num_features=1, affine=False)
        
        # Original input
        x1 = [[real(v)] for v in values]
        output1 = norm(x1)
        
        # Affine transformed input: 2x + 3
        x2 = [[real(2*v + 3)] for v in values]
        output2 = norm(x2)
        
        # Outputs should be identical
        for i in range(len(values)):
            if output1[i][0].tag == TRTag.REAL and output2[i][0].tag == TRTag.REAL:
                assert output1[i][0].value.value == pytest.approx(
                    output2[i][0].value.value, rel=1e-7, abs=1e-10
                )
    
    def test_gradient_preserves_zero_mean(self):
        """Test that gradients preserve zero mean property."""
        norm = TRNorm(num_features=1, affine=False)
        
        with gradient_tape() as tape:
            # Create batch
            x_nodes = []
            for i in range(5):
                node = TRNode.parameter(real(float(i)))
                tape.watch(node)
                x_nodes.append(node)
            
            x = [[node] for node in x_nodes]
            
            # Forward
            output = norm(x)
            
            # Loss that treats all outputs equally
            loss = output[0][0]
            for i in range(1, 5):
                loss = loss + output[i][0]
        
        # Get gradients
        grads = tape.gradient(loss, x_nodes)
        
        # Sum of gradients should be near zero (preservation of translation invariance)
        grad_sum = sum(g.value.value for g in grads if g.tag == TRTag.REAL)
        assert abs(grad_sum) < 1e-10


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_sample_batch(self):
        """Test normalization with batch size 1."""
        norm = TRNorm(num_features=2)
        norm.beta[0]._value = real(7.0)
        norm.beta[1]._value = real(-2.0)
        
        x = [[real(3.0), real(5.0)]]
        
        output = norm(x)
        
        # With single sample, variance = 0, should bypass
        assert output[0][0].value.value == 7.0
        assert output[0][1].value.value == -2.0
    
    def test_empty_batch(self):
        """Test with empty batch."""
        norm = TRNorm(num_features=1)
        
        x = []
        output = norm(x)
        
        assert len(output) == 0
    
    def test_very_small_variance(self):
        """Test with very small but non-zero variance."""
        norm = TRNorm(num_features=1)
        
        # Create values with tiny variance
        epsilon = 1e-8
        x = [
            [real(1.0)],
            [real(1.0 + epsilon)],
            [real(1.0 - epsilon)],
        ]
        
        output = norm(x)
        
        # Should still normalize (not bypass)
        # Check that outputs are different
        vals = [out[0].value.value for out in output]
        assert len(set(vals)) > 1  # Not all the same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
