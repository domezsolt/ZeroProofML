"""
Property-based tests for transreal autodifferentiation.

These tests use Hypothesis to verify that autodiff maintains
expected mathematical properties across random inputs.
"""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from zeroproof.autodiff import TRNode, gradient_tape, tr_add, tr_grad, tr_mul
from zeroproof.core import TRTag, real

# Strategies for generating test data
finite_reals = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, allow_subnormal=False
)

small_positive_reals = st.floats(min_value=0.1, max_value=10.0)

safe_reals = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


class TestGradientProperties:
    """Test mathematical properties of gradients."""

    @given(finite_reals)
    def test_constant_function_zero_gradient(self, x_val):
        """Gradient of constant function is zero."""

        def constant_func(x):
            return TRNode.constant(real(42.0))

        grad_f = tr_grad(constant_func)
        grad = grad_f(real(x_val))

        assert grad.value.value == 0.0

    @given(finite_reals)
    def test_identity_function_unit_gradient(self, x_val):
        """Gradient of identity function is 1."""

        def identity(x):
            return x

        grad_f = tr_grad(identity)
        grad = grad_f(real(x_val))

        assert grad.value.value == 1.0

    @given(finite_reals, finite_reals)
    def test_linearity_of_gradient(self, x_val, a):
        """Test that gradient is linear: ∇(af) = a∇f."""

        def f(x):
            return tr_mul(x, x)

        def scaled_f(x):
            return tr_mul(TRNode.constant(real(a)), f(x))

        # Gradient of f
        grad_f = tr_grad(f)
        g1 = grad_f(real(x_val))

        # Gradient of a*f
        grad_scaled = tr_grad(scaled_f)
        g2 = grad_scaled(real(x_val))

        if g1.tag == TRTag.REAL and g2.tag == TRTag.REAL:
            expected = a * g1.value.value
            assert g2.value.value == pytest.approx(expected, rel=1e-10)

    @given(safe_reals)
    def test_chain_rule(self, x_val):
        """Test chain rule: (f∘g)' = f'(g)·g'."""

        # Inner function g(x) = x + 1
        def g(x):
            return tr_add(x, TRNode.constant(real(1.0)))

        # Outer function f(y) = y²
        def f(y):
            return tr_mul(y, y)

        # Composition h(x) = f(g(x)) = (x+1)²
        def h(x):
            return f(g(x))

        # Direct gradient of composition
        grad_h = tr_grad(h)
        dh_dx = grad_h(real(x_val))

        # Expected: h'(x) = 2(x+1)·1 = 2x + 2
        expected = 2 * x_val + 2

        if dh_dx.tag == TRTag.REAL:
            assert dh_dx.value.value == pytest.approx(expected, rel=1e-10)


class TestMaskREALProperties:
    """Test properties specific to Mask-REAL rule."""

    @given(finite_reals)
    def test_non_real_forward_implies_zero_gradient(self, x_val):
        """Any non-REAL forward value produces zero gradient."""
        # Create scenarios that produce non-REAL values
        functions = [
            lambda x: TRNode.constant(real(1.0)) / TRNode.constant(real(0.0)),  # +∞
            lambda x: TRNode.constant(real(-1.0)) / TRNode.constant(real(0.0)),  # -∞
            lambda x: TRNode.constant(real(0.0)) / TRNode.constant(real(0.0)),  # Φ
        ]

        for f in functions:
            with gradient_tape() as tape:
                x = TRNode.parameter(real(x_val))
                tape.watch(x)
                y = f(x)

                # Verify non-REAL output
                assert y.tag != TRTag.REAL

            grads = tape.gradient(y, [x])
            assert grads[0].value.value == 0.0

    @given(st.floats(min_value=0.01, max_value=100.0))
    def test_gradient_continuity_near_singularity(self, epsilon):
        """Test gradient behavior approaching a singularity."""

        def f(x):
            # f(x) = 1/(x-1) has singularity at x=1
            return TRNode.constant(real(1.0)) / (x - TRNode.constant(real(1.0)))

        # Test just before singularity
        x_before = TRNode.parameter(real(1.0 - epsilon))
        grad_f = tr_grad(f)
        grad_before = grad_f(x_before)

        # Test just after singularity
        x_after = TRNode.parameter(real(1.0 + epsilon))
        grad_after = grad_f(x_after)

        # Both should be REAL with opposite signs
        assert grad_before.tag == TRTag.REAL
        assert grad_after.tag == TRTag.REAL
        assert grad_before.value.value * grad_after.value.value < 0  # Opposite signs

    @given(finite_reals)
    def test_mask_real_composition_property(self, x_val):
        """If any intermediate is non-REAL, final gradient is zero."""

        def create_non_real_intermediate(x):
            # First create a non-REAL value
            non_real = TRNode.constant(real(1.0)) / TRNode.constant(real(0.0))  # +∞
            # Then bring it back to REAL
            return TRNode.constant(real(1.0)) / non_real  # 1/∞ = 0

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)
            y = create_non_real_intermediate(x)

            # Final value is REAL
            assert y.tag == TRTag.REAL
            assert y.value.value == 0.0

        # But gradient should still be zero due to non-REAL intermediate
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0


class TestGradientAccumulation:
    """Test gradient accumulation for shared nodes."""

    @given(safe_reals)
    def test_multiple_paths_accumulate(self, x_val):
        """Gradients accumulate when variable is used multiple times."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)

            # Use x in multiple paths
            # y = x² + 2x = x·x + x + x
            path1 = x * x  # Contributes 2x to gradient
            path2 = x  # Contributes 1 to gradient
            path3 = x  # Contributes 1 to gradient
            y = path1 + path2 + path3

        grads = tape.gradient(y, [x])

        # Total gradient: 2x + 1 + 1 = 2x + 2
        expected = 2 * x_val + 2
        assert grads[0].value.value == pytest.approx(expected, rel=1e-10)

    @given(safe_reals, safe_reals)
    def test_independent_variables_independent_gradients(self, x_val, y_val):
        """Independent variables have independent gradients."""
        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            y = TRNode.parameter(real(y_val))
            tape.watch(x)
            tape.watch(y)

            # z = x² + y²
            z = x * x + y * y

        grads = tape.gradient(z, [x, y])

        # ∂z/∂x = 2x, ∂z/∂y = 2y
        assert grads[0].value.value == pytest.approx(2 * x_val, rel=1e-10)
        assert grads[1].value.value == pytest.approx(2 * y_val, rel=1e-10)


class TestDomainAwareGradients:
    """Test gradients of domain-aware operations."""

    @given(small_positive_reals)
    def test_log_gradient_valid_domain(self, x_val):
        """Test log gradient in valid domain."""
        from zeroproof.autodiff import tr_log

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)
            y = tr_log(x)

        grads = tape.gradient(y, [x])

        # d/dx[log(x)] = 1/x
        expected = 1.0 / x_val
        assert grads[0].value.value == pytest.approx(expected, rel=1e-10)

    @given(st.floats(max_value=-0.1, allow_infinity=False))
    def test_log_gradient_invalid_domain(self, x_val):
        """Test log gradient in invalid domain produces zero."""
        from zeroproof.autodiff import tr_log

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)
            y = tr_log(x)  # log(negative) = Φ

            assert y.tag == TRTag.PHI

        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0

    @given(st.floats(min_value=0.0, max_value=100.0))
    def test_sqrt_gradient_valid_domain(self, x_val):
        """Test sqrt gradient in valid domain."""
        from zeroproof.autodiff import tr_sqrt

        assume(x_val > 0.01)  # Avoid near-zero for numerical stability

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)
            y = tr_sqrt(x)

        grads = tape.gradient(y, [x])

        # d/dx[sqrt(x)] = 1/(2*sqrt(x))
        expected = 1.0 / (2.0 * math.sqrt(x_val))
        assert grads[0].value.value == pytest.approx(expected, rel=1e-5)


class TestNumericalStability:
    """Test numerical stability of gradient computations."""

    @given(st.floats(min_value=1e-10, max_value=1e-5))
    def test_gradient_near_zero(self, epsilon):
        """Test gradient computation near zero."""

        # Function with potential issues near zero
        def f(x):
            # f(x) = x² / (x² + ε²)
            x_squared = x * x
            eps_squared = TRNode.constant(real(epsilon * epsilon))
            return x_squared / (x_squared + eps_squared)

        x = TRNode.parameter(real(epsilon))
        grad_f = tr_grad(f)
        grad = grad_f(x)

        # Gradient should be well-defined and finite
        assert grad.tag == TRTag.REAL
        assert math.isfinite(grad.value.value)

    @given(st.floats(min_value=1e5, max_value=1e10))
    def test_gradient_large_values(self, large_val):
        """Test gradient computation with large values."""

        def f(x):
            # Normalize to prevent overflow
            normalized = x / TRNode.constant(real(1e6))
            return normalized * normalized

        x = TRNode.parameter(real(large_val))
        grad_f = tr_grad(f)
        grad = grad_f(x)

        # Should handle large values gracefully
        if grad.tag == TRTag.REAL:
            assert math.isfinite(grad.value.value)


class TestComplexGradientFlows:
    """Test complex gradient flow scenarios."""

    @settings(max_examples=50)
    @given(safe_reals, safe_reals, safe_reals)
    def test_multivariate_polynomial(self, x_val, y_val, z_val):
        """Test gradients of multivariate polynomial."""

        def f(x, y, z):
            # f(x,y,z) = x²y + yz² + x
            term1 = x * x * y
            term2 = y * z * z
            term3 = x
            return term1 + term2 + term3

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            y = TRNode.parameter(real(y_val))
            z = TRNode.parameter(real(z_val))
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)

            result = f(x, y, z)

        grads = tape.gradient(result, [x, y, z])

        # Expected gradients
        # ∂f/∂x = 2xy + 1
        # ∂f/∂y = x² + z²
        # ∂f/∂z = 2yz
        expected_x = 2 * x_val * y_val + 1
        expected_y = x_val * x_val + z_val * z_val
        expected_z = 2 * y_val * z_val

        assert grads[0].value.value == pytest.approx(expected_x, rel=1e-10)
        assert grads[1].value.value == pytest.approx(expected_y, rel=1e-10)
        assert grads[2].value.value == pytest.approx(expected_z, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
