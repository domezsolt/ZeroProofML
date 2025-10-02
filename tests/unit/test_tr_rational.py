"""Unit tests for TR-Rational layers."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.core import TRTag, ninf, phi, pinf, real
from zeroproof.layers import ChebyshevBasis, MonomialBasis, TRRational, TRRationalMulti


class TestTRRational:
    """Test TR-Rational layer functionality."""

    def test_initialization(self):
        """Test layer initialization."""
        # Basic initialization
        layer = TRRational(d_p=2, d_q=2)
        assert layer.d_p == 2
        assert layer.d_q == 2
        assert len(layer.theta) == 3  # θ_0, θ_1, θ_2
        assert len(layer.phi) == 2  # φ_1, φ_2 (φ_0 = 1 implicit)

        # Check parameters are nodes with gradients
        for param in layer.parameters():
            assert isinstance(param, TRNode)
            assert param.requires_grad

        # Test with custom basis
        chebyshev = ChebyshevBasis()
        layer = TRRational(d_p=3, d_q=1, basis=chebyshev)
        assert layer.basis.name == "chebyshev"

    def test_forward_real_output(self):
        """Test forward pass producing REAL output."""
        layer = TRRational(d_p=1, d_q=1)

        # Set simple coefficients: P(x) = 1 + x, Q(x) = 1 + 0.5x
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(1.0)
        layer.phi[0]._value = real(0.5)

        # Test at x = 1: P(1) = 2, Q(1) = 1.5, y = 2/1.5 = 4/3
        x = TRNode.constant(real(1.0))
        y, tag = layer.forward(x)

        assert tag == TRTag.REAL
        assert y.value.value == pytest.approx(4 / 3, rel=1e-10)

    def test_forward_pole_infinity(self):
        """Test forward pass at a pole producing infinity."""
        layer = TRRational(d_p=1, d_q=1)

        # Set coefficients: P(x) = 1, Q(x) = 1 - x
        # This has a pole at x = 1
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(0.0)
        layer.phi[0]._value = real(-1.0)

        # Test at pole x = 1: Q(1) = 0, P(1) = 1 → +∞
        x = TRNode.constant(real(1.0))
        y, tag = layer.forward(x)

        assert tag == TRTag.PINF

    def test_forward_indeterminate(self):
        """Test forward pass producing PHI (0/0)."""
        layer = TRRational(d_p=1, d_q=1)

        # Set coefficients: P(x) = x - 1, Q(x) = x - 1
        # Both are zero at x = 1
        layer.theta[0]._value = real(-1.0)
        layer.theta[1]._value = real(1.0)
        layer.phi[0]._value = real(-1.0)

        # Test at x = 1: P(1) = 0, Q(1) = 0 → Φ
        x = TRNode.constant(real(1.0))
        y, tag = layer.forward(x)

        assert tag == TRTag.PHI

    def test_gradient_real_path(self):
        """Test gradients when output is REAL."""
        layer = TRRational(d_p=1, d_q=1)

        # Simple function: y = x / (1 + x)
        layer.theta[0]._value = real(0.0)
        layer.theta[1]._value = real(1.0)
        layer.phi[0]._value = real(1.0)

        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            tape.watch(x)
            y = layer(x)

            # At x=1: y = 1/2
            assert y.tag == TRTag.REAL
            assert y.value.value == pytest.approx(0.5)

        # Compute gradient
        # dy/dx = 1/(1+x)² = 1/4 at x=1
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == pytest.approx(0.25)

    def test_gradient_mask_real_at_pole(self):
        """Test that gradients are zero at poles (Mask-REAL)."""
        layer = TRRational(d_p=1, d_q=1)

        # P(x) = 1, Q(x) = 1 - x (pole at x = 1)
        # Setting theta[0]=1, theta[1]=0 gives P(x) = 1
        # Setting phi[0]=-1 gives Q(x) = 1 - x
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(0.0)
        layer.phi[0]._value = real(-1.0)

        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            tape.watch(x)
            y = layer(x)

            # At x=1: 1/(1-1) = 1/0 = +∞
            assert y.tag == TRTag.PINF

        # Gradient should be zero due to Mask-REAL
        grads = tape.gradient(y, [x])
        assert grads[0].value.value == 0.0

    def test_regularization_loss(self):
        """Test L2 regularization on denominator."""
        layer = TRRational(d_p=2, d_q=2, alpha_phi=0.1)

        # Set some phi values
        layer.phi[0]._value = real(2.0)
        layer.phi[1]._value = real(-1.0)

        # Regularization = α/2 * (φ₁² + φ₂²) = 0.1/2 * (4 + 1) = 0.25
        reg_loss = layer.regularization_loss()
        assert reg_loss.value.value == pytest.approx(0.25)

    def test_q_min_computation(self):
        """Test computation of minimum |Q(x)|."""
        layer = TRRational(d_p=1, d_q=1)

        # Q(x) = 1 + 0.5x
        layer.phi[0]._value = real(0.5)

        # Test on batch
        x_batch = [real(-1.0), real(0.0), real(1.0), real(2.0)]
        q_min = layer.compute_q_min(x_batch)

        # Q values: 0.5, 1.0, 1.5, 2.0
        # Minimum is 0.5
        assert q_min == pytest.approx(0.5)


class TestTRRationalMulti:
    """Test multi-output rational layer."""

    def test_shared_denominator(self):
        """Test that denominator is shared across outputs."""
        layer = TRRationalMulti(d_p=1, d_q=1, n_outputs=3, shared_Q=True)

        # Check that phi parameters are the same objects
        phi_0 = layer.layers[0].phi
        phi_1 = layer.layers[1].phi
        phi_2 = layer.layers[2].phi

        assert phi_0 is phi_1
        assert phi_1 is phi_2

        # But theta parameters are different
        theta_0 = layer.layers[0].theta
        theta_1 = layer.layers[1].theta
        assert theta_0 is not theta_1

    def test_independent_outputs(self):
        """Test independent rational functions."""
        layer = TRRationalMulti(d_p=1, d_q=1, n_outputs=2, shared_Q=False)

        # Set different coefficients for each output
        layer.layers[0].theta[0]._value = real(1.0)
        layer.layers[0].theta[1]._value = real(0.0)
        layer.layers[0].phi[0]._value = real(1.0)

        layer.layers[1].theta[0]._value = real(0.0)
        layer.layers[1].theta[1]._value = real(1.0)
        layer.layers[1].phi[0]._value = real(-1.0)

        x = TRNode.constant(real(0.5))
        outputs = layer(x)

        # First output: 1/(1 + 0.5) = 2/3
        # Second output: 0.5/(1 - 0.5) = 1
        assert outputs[0].value.value == pytest.approx(2 / 3)
        assert outputs[1].value.value == pytest.approx(1.0)


class TestBasisFunctions:
    """Test basis function implementations."""

    def test_monomial_basis(self):
        """Test monomial basis evaluation."""
        basis = MonomialBasis(domain=(-1, 1))
        x = real(2.0)

        # Evaluate up to degree 3
        psi = basis(x, 3)

        assert len(psi) == 4
        assert psi[0].value == 1.0  # x^0
        assert psi[1].value == 2.0  # x^1
        assert psi[2].value == 4.0  # x^2
        assert psi[3].value == 8.0  # x^3

    def test_chebyshev_basis(self):
        """Test Chebyshev basis evaluation."""
        basis = ChebyshevBasis(domain=(-1, 1))
        x = real(0.5)

        # Evaluate up to degree 3
        psi = basis(x, 3)

        # Chebyshev polynomials at x=0.5:
        # T_0(0.5) = 1
        # T_1(0.5) = 0.5
        # T_2(0.5) = 2(0.5)² - 1 = -0.5
        # T_3(0.5) = 4(0.5)³ - 3(0.5) = -1
        assert len(psi) == 4
        assert psi[0].value == pytest.approx(1.0)
        assert psi[1].value == pytest.approx(0.5)
        assert psi[2].value == pytest.approx(-0.5)
        assert psi[3].value == pytest.approx(-1.0)

    def test_chebyshev_domain_transform(self):
        """Test Chebyshev basis with non-standard domain."""
        basis = ChebyshevBasis(domain=(0, 2))

        # x=1 in [0,2] maps to t=0 in [-1,1]
        x = real(1.0)
        psi = basis(x, 2)

        # At t=0: T_0=1, T_1=0, T_2=-1
        assert psi[0].value == pytest.approx(1.0)
        assert psi[1].value == pytest.approx(0.0)
        assert psi[2].value == pytest.approx(-1.0)


class TestGradientFlowProperties:
    """Test gradient flow through rational layers."""

    @given(st.floats(min_value=-2.0, max_value=2.0))
    def test_gradient_continuity_away_from_poles(self, x_val):
        """Test smooth gradients away from singularities."""
        layer = TRRational(d_p=1, d_q=1)

        # y = x / (2 + x) - smooth everywhere (no poles in [-2, 2])
        layer.theta[0]._value = real(0.0)
        layer.theta[1]._value = real(1.0)
        # Q(x) = 1 + phi[0]*x, so for Q(x) = 2 + x, we need phi[0] = 1
        # But Q(x) = 1 + x is also fine, just avoid x = -1
        layer.phi[0]._value = real(0.5)

        with gradient_tape() as tape:
            x = TRNode.parameter(real(x_val))
            tape.watch(x)
            y = layer(x)

        if y.tag == TRTag.REAL:
            grads = tape.gradient(y, [x])
            # Gradient should be finite
            assert grads[0].tag == TRTag.REAL
            assert math.isfinite(grads[0].value.value)

    def test_parameter_gradient_accumulation(self):
        """Test gradient accumulation for layer parameters."""
        layer = TRRational(d_p=1, d_q=1)

        with gradient_tape() as tape:
            # Watch parameters
            for param in layer.parameters():
                tape.watch(param)

            # Multiple forward passes
            x1 = TRNode.constant(real(1.0))
            y1 = layer(x1)

            x2 = TRNode.constant(real(2.0))
            y2 = layer(x2)

            # Combined loss
            loss = y1 * y1 + y2 * y2

        # Compute gradients w.r.t parameters
        grads = tape.gradient(loss, layer.parameters())

        # All parameter gradients should be computed
        for grad in grads:
            assert grad is not None
            if grad.tag == TRTag.REAL:
                assert math.isfinite(grad.value.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
