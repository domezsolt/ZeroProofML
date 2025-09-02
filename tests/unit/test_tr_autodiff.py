"""Unit tests for transreal autodifferentiation with Mask-REAL rule."""

import math
import pytest
from hypothesis import given, strategies as st

from zeroproof.core import TRScalar, TRTag, real, pinf, ninf, phi
from zeroproof.autodiff import (
    TRNode, TRGradientTape, gradient_tape,
    tr_grad, tr_value_and_grad, check_gradient,
    tr_add, tr_sub, tr_mul, tr_div,
    tr_neg, tr_abs, tr_sign,
    tr_log, tr_sqrt, tr_pow_int,
)


class TestTRNode:
    """Test TRNode functionality."""
    
    def test_node_creation(self):
        """Test creating TRNode instances."""
        # Constant node
        x = TRNode.constant(real(3.14))
        assert x.value.value == 3.14
        assert x.tag == TRTag.REAL
        assert not x.requires_grad
        assert x.gradient is None
        
        # Parameter node
        y = TRNode.parameter(real(2.0))
        assert y.value.value == 2.0
        assert y.requires_grad
        assert y.gradient is None
    
    def test_node_operations(self):
        """Test basic operations on nodes."""
        x = TRNode.parameter(real(3.0))
        y = TRNode.parameter(real(2.0))
        
        # Addition
        z = x + y
        assert z.value.value == 5.0
        assert z.requires_grad
        
        # Multiplication
        w = x * y
        assert w.value.value == 6.0
        assert w.requires_grad
        
        # Mixed with constants
        c = x + 1.0
        assert c.value.value == 4.0
        assert c.requires_grad


class TestMaskREALRule:
    """Test the Mask-REAL gradient rule."""
    
    def test_real_path_gradients(self):
        """Test gradients on REAL paths match classical calculus."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            tape.watch(x)
            
            # y = x^2 + 3x + 1
            y = x * x + 3 * x + 1
        
        # dy/dx = 2x + 3 = 7 at x=2
        grads = tape.gradient(y, [x])
        assert len(grads) == 1
        assert grads[0].value.value == 7.0
    
    def test_non_real_forward_zeros_gradient(self):
        """Test that non-REAL forward values produce zero gradients."""
        # Test division by zero producing infinity
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # y = 1/x → +∞
            y = TRNode.constant(real(1.0)) / x
        
        assert y.tag == TRTag.PINF
        
        # Gradient should be zero due to Mask-REAL
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
        assert grads[0].tag == TRTag.REAL
    
    def test_phi_forward_zeros_gradient(self):
        """Test that PHI forward values produce zero gradients."""
        # Test 0/0 producing PHI
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # y = x/x → Φ at x=0
            y = x / x
        
        assert y.tag == TRTag.PHI
        
        # Gradient should be zero due to Mask-REAL
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_intermediate_non_real_zeros_path(self):
        """Test that non-REAL intermediate values zero the entire path."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(0.0))
            tape.watch(x)
            
            # Create a path with non-REAL intermediate
            # y = 1/x → +∞
            y = TRNode.constant(real(1.0)) / x
            # z = y + 1 → +∞
            z = y + 1
            # w = 1/z → 0 (would be REAL, but path has non-REAL)
            w = TRNode.constant(real(1.0)) / z
        
        assert y.tag == TRTag.PINF  # Intermediate is non-REAL
        assert z.tag == TRTag.PINF
        assert w.tag == TRTag.REAL  # Final value is REAL
        
        # But gradient should still be zero due to non-REAL in path
        grads = tape.gradient(w, [x])
        assert grads[0].value.value == 0.0
    
    def test_mixed_paths_real_and_non_real(self):
        """Test computation with both REAL and non-REAL paths."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            y = TRNode.parameter(real(0.0))
            tape.watch(x)
            tape.watch(y)
            
            # Branch 1: REAL path
            branch1 = x * x  # 4
            
            # Branch 2: non-REAL path (division by zero)
            branch2 = TRNode.constant(real(1.0)) / y  # +∞
            
            # Combine branches
            result = branch1 + branch2  # 4 + ∞ = ∞
        
        assert result.tag == TRTag.PINF
        
        # Gradients
        grads = tape.gradient(result, [x, y])
        
        # Both gradients should be zero because result is non-REAL
        assert grads[0].value.value == 0.0  # ∂result/∂x = 0
        assert grads[1].value.value == 0.0  # ∂result/∂y = 0


class TestGradientFunctions:
    """Test high-level gradient computation functions."""
    
    def test_tr_grad_simple(self):
        """Test tr_grad on simple functions."""
        def f(x):
            return x * x + 2 * x + 1
        
        grad_f = tr_grad(f)
        
        # Test at x = 3
        x = TRNode.parameter(real(3.0))
        df_dx = grad_f(x)
        
        # f'(x) = 2x + 2, so f'(3) = 8
        assert df_dx.value.value == 8.0
    
    def test_tr_grad_multiple_args(self):
        """Test tr_grad with multiple arguments."""
        def f(x, y):
            return x * x + y * y + x * y
        
        # Gradient with respect to both arguments
        grad_f = tr_grad(f, argnums=[0, 1])
        
        x = TRNode.parameter(real(2.0))
        y = TRNode.parameter(real(3.0))
        grads = grad_f(x, y)
        
        # ∂f/∂x = 2x + y = 7
        # ∂f/∂y = 2y + x = 8
        assert grads[0].value.value == 7.0
        assert grads[1].value.value == 8.0
    
    def test_tr_value_and_grad(self):
        """Test computing both value and gradient."""
        def f(x):
            return x * x * x  # x^3
        
        value_and_grad_f = tr_value_and_grad(f)
        
        x = TRNode.parameter(real(2.0))
        val, grad = value_and_grad_f(x)
        
        # f(2) = 8, f'(2) = 3*2^2 = 12
        assert val.value.value == 8.0
        assert grad.value.value == 12.0
    
    def test_check_gradient(self):
        """Test gradient checking with finite differences."""
        def f(x):
            return x * x + tr_log(x)
        
        x = TRNode.parameter(real(2.0))
        analytical, numerical, error = check_gradient(f, x)
        
        # f'(x) = 2x + 1/x, so f'(2) = 4 + 0.5 = 4.5
        assert analytical.value.value == pytest.approx(4.5, rel=1e-5)
        assert numerical == pytest.approx(4.5, rel=1e-3)
        assert error < 1e-5


class TestDomainAwareGradients:
    """Test gradients of domain-aware operations."""
    
    def test_log_gradient(self):
        """Test gradient of logarithm."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            tape.watch(x)
            y = tr_log(x)
        
        # d/dx[log(x)] = 1/x = 0.5
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.5
    
    def test_log_invalid_domain(self):
        """Test gradient when log domain is violated."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(-1.0))
            tape.watch(x)
            y = tr_log(x)  # log(-1) = PHI
        
        assert y.tag == TRTag.PHI
        
        # Gradient should be zero due to Mask-REAL
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0
    
    def test_sqrt_gradient(self):
        """Test gradient of square root."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(4.0))
            tape.watch(x)
            y = tr_sqrt(x)
        
        # d/dx[sqrt(x)] = 1/(2*sqrt(x)) = 1/4 = 0.25
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.25
    
    def test_pow_int_gradient(self):
        """Test gradient of integer power."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(3.0))
            tape.watch(x)
            y = tr_pow_int(x, 4)  # x^4
        
        # d/dx[x^4] = 4*x^3 = 4*27 = 108
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 108.0
    
    def test_pow_zero_special_case(self):
        """Test gradient of x^0."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(5.0))
            tape.watch(x)
            y = tr_pow_int(x, 0)  # x^0 = 1
        
        # d/dx[x^0] = 0 (except at x=0)
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0


class TestGradientTape:
    """Test gradient tape functionality."""
    
    def test_persistent_tape(self):
        """Test using a persistent tape multiple times."""
        x = TRNode.parameter(real(2.0))
        
        with gradient_tape(persistent=True) as tape:
            tape.watch(x)
            y = x * x
        
        # First use
        grads1 = tape.gradient(y, [x])
        assert grads1[0].value.value == 4.0
        
        # Second use (should work with persistent=True)
        grads2 = tape.gradient(y, [x])
        assert grads2[0].value.value == 4.0
    
    def test_non_persistent_tape_error(self):
        """Test that non-persistent tape can only be used once."""
        x = TRNode.parameter(real(2.0))
        
        with gradient_tape(persistent=False) as tape:
            tape.watch(x)
            y = x * x
        
        # First use should work
        grads1 = tape.gradient(y, [x])
        assert grads1[0].value.value == 4.0
        
        # Second use should raise error
        with pytest.raises(RuntimeError):
            tape.gradient(y, [x])
    
    def test_nested_operations(self):
        """Test deeply nested operations."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            tape.watch(x)
            
            # y = ((x + 1) * 2 - 3) / 4
            y = (x + 1) * 2
            y = y - 3
            y = y / 4
        
        # Simplified: y = (2x + 2 - 3) / 4 = (2x - 1) / 4
        # dy/dx = 2/4 = 0.5
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.5


class TestComplexGradientFlows:
    """Test complex gradient flow scenarios."""
    
    def test_rational_function_gradient(self):
        """Test gradient of a rational function P(x)/Q(x)."""
        def rational(x):
            # f(x) = (x^2 + 1) / (x + 2)
            p = x * x + 1
            q = x + 2
            return p / q
        
        grad_f = tr_grad(rational)
        
        # Test at x = 1
        x = TRNode.parameter(real(1.0))
        df_dx = grad_f(x)
        
        # f'(x) = [(2x)(x+2) - (x^2+1)(1)] / (x+2)^2
        # At x=1: [2*3 - 2*1] / 9 = 4/9
        assert df_dx.value.value == pytest.approx(4/9, rel=1e-10)
    
    def test_gradient_at_near_singularity(self):
        """Test gradient computation near a singularity."""
        eps = 1e-8
        
        with gradient_tape() as tape:
            x = TRNode.parameter(real(eps))
            tape.watch(x)
            
            # y = 1/x - very large but still REAL
            y = TRNode.constant(real(1.0)) / x
        
        assert y.tag == TRTag.REAL  # Not quite at singularity
        
        # dy/dx = -1/x^2 - very large negative
        grads = tape.gradient(y, [x])
        assert grads[0].tag == TRTag.REAL
        assert grads[0].value.value == pytest.approx(-1/(eps*eps), rel=1e-5)
    
    @pytest.mark.property
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_chain_rule_property(self, x_val):
        """Property test: chain rule for composite functions."""
        def f(x):
            # Composite function: sqrt(log(x^2 + 1))
            return tr_sqrt(tr_log(x * x + 1))
        
        x = TRNode.parameter(real(x_val))
        
        # Compute gradient using our autodiff
        grad_f = tr_grad(f)
        auto_grad = grad_f(x)
        
        # Compute expected gradient analytically
        # Let u = x^2 + 1, v = log(u), f = sqrt(v)
        # df/dx = df/dv * dv/du * du/dx
        #       = 1/(2*sqrt(v)) * 1/u * 2x
        u = x_val * x_val + 1
        v = math.log(u)
        expected = (1 / (2 * math.sqrt(v))) * (1 / u) * (2 * x_val)
        
        assert auto_grad.tag == TRTag.REAL
        assert auto_grad.value.value == pytest.approx(expected, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
