"""Tests for pole detection head and loss."""

from typing import List

import numpy as np
import pytest

from zeroproof.autodiff import TRNode
from zeroproof.core import TRScalar, TRTag, ninf, phi, pinf, real
from zeroproof.layers.basis import MonomialBasis
from zeroproof.layers.pole_aware_rational import FullyIntegratedRational, PoleAwareRational
from zeroproof.training.pole_detection import (
    DomainSpecificPoleDetector,
    PoleDetectionConfig,
    PoleDetectionHead,
    binary_cross_entropy,
    compute_pole_loss,
    compute_pole_metrics,
    sigmoid,
    tanh_activation,
)


class TestPoleDetectionHead:
    """Test pole detection head functionality."""

    def test_initialization(self):
        """Test pole head initialization."""
        config = PoleDetectionConfig(hidden_dim=8)
        head = PoleDetectionHead(input_dim=3, config=config)

        assert len(head.W1) == 8
        assert len(head.b1) == 8
        assert len(head.W_out) == 8
        assert head.b_out is not None

    def test_forward_pass(self):
        """Test forward pass through pole head."""
        config = PoleDetectionConfig(hidden_dim=4, normalize_output=True)
        head = PoleDetectionHead(input_dim=2, config=config)

        x = TRNode.constant(real(0.5))
        score = head.forward(x)

        # Check output is valid
        assert score.tag == TRTag.REAL
        # Sigmoid output should be in [0, 1]
        assert 0.0 <= score.value.value <= 1.0

    def test_activations(self):
        """Test different activation functions."""
        x = TRNode.constant(real(0.5))

        # Test sigmoid
        sig = sigmoid(x)
        assert sig.tag == TRTag.REAL
        assert 0.5 < sig.value.value < 0.7  # sigmoid(0.5) ≈ 0.62

        # Test tanh
        tanh = tanh_activation(x)
        assert tanh.tag == TRTag.REAL
        assert 0.4 < tanh.value.value < 0.6  # tanh(0.5) ≈ 0.46

        # Test with extreme values
        x_large = TRNode.constant(real(10.0))
        sig_large = sigmoid(x_large)
        assert sig_large.value.value > 0.98

        x_ninf = TRNode.constant(ninf())
        sig_ninf = sigmoid(x_ninf)
        assert sig_ninf.value.value == 0.0

    def test_pole_probability(self):
        """Test pole probability prediction."""
        config = PoleDetectionConfig(hidden_dim=4)
        head = PoleDetectionHead(input_dim=2, config=config)

        x = TRNode.constant(real(0.0))
        prob = head.predict_pole_probability(x)

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_parameters(self):
        """Test parameter collection."""
        config = PoleDetectionConfig(hidden_dim=3, use_residual=True)
        head = PoleDetectionHead(input_dim=2, config=config)

        params = head.parameters()

        # Should have W1, b1, W2, b2, W_out, b_out
        # W1: 3x3, b1: 3, W2: 3x3, b2: 3, W_out: 3, b_out: 1
        expected_count = 9 + 3 + 9 + 3 + 3 + 1  # 28
        assert len(params) == expected_count


class TestPoleLoss:
    """Test pole detection loss computation."""

    def test_binary_cross_entropy(self):
        """Test BCE loss computation."""
        pred = TRNode.constant(real(0.7))

        # Test with target = 1
        loss1 = binary_cross_entropy(pred, 1.0)
        assert loss1.tag == TRTag.REAL
        assert loss1.value.value > 0  # Should have positive loss

        # Test with target = 0
        loss0 = binary_cross_entropy(pred, 0.0)
        assert loss0.tag == TRTag.REAL
        assert loss0.value.value > 0

        # Perfect prediction should have near-zero loss
        pred_perfect = TRNode.constant(real(0.99))
        loss_perfect = binary_cross_entropy(pred_perfect, 1.0)
        assert loss_perfect.value.value < 0.1

    def test_compute_pole_loss_with_teacher(self):
        """Test pole loss with teacher labels."""
        predictions = [TRNode.constant(real(1.0)) for _ in range(3)]
        pole_scores = [
            TRNode.constant(real(0.8)),
            TRNode.constant(real(0.2)),
            TRNode.constant(real(0.6)),
        ]
        teacher_labels = [1.0, 0.0, 1.0]

        config = PoleDetectionConfig(teacher_weight=1.0)
        loss = compute_pole_loss(predictions, pole_scores, None, teacher_labels, config)

        assert loss is not None
        assert loss.tag == TRTag.REAL
        assert loss.value.value > 0

    def test_compute_pole_loss_with_Q_values(self):
        """Test pole loss with Q-value self-supervision."""
        predictions = [TRNode.constant(real(1.0)) for _ in range(3)]
        pole_scores = [
            TRNode.constant(real(0.8)),
            TRNode.constant(real(0.2)),
            TRNode.constant(real(0.6)),
        ]
        Q_values = [0.05, 0.5, 0.01]  # First and last are near poles

        config = PoleDetectionConfig(proximity_threshold=0.1)
        loss = compute_pole_loss(predictions, pole_scores, Q_values, None, config)

        assert loss is not None
        assert loss.tag == TRTag.REAL

    def test_compute_pole_loss_with_tags(self):
        """Test pole loss with tag-based weak supervision."""
        predictions = [TRNode.constant(real(1.0)), TRNode.constant(pinf()), TRNode.constant(phi())]
        pole_scores = [
            TRNode.constant(real(0.2)),
            TRNode.constant(real(0.8)),
            TRNode.constant(real(0.9)),
        ]

        loss = compute_pole_loss(predictions, pole_scores, None, None, None)

        assert loss is not None
        assert loss.tag == TRTag.REAL

    def test_compute_pole_metrics(self):
        """Test pole detection metrics computation."""
        pole_scores = [
            TRNode.constant(real(0.8)),
            TRNode.constant(real(0.2)),
            TRNode.constant(real(0.7)),
            TRNode.constant(real(0.3)),
        ]
        true_poles = [True, False, True, False]

        metrics = compute_pole_metrics(pole_scores, true_poles)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Perfect predictions
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestPoleAwareRational:
    """Test pole-aware rational layer."""

    def test_initialization(self):
        """Test layer initialization."""
        basis = MonomialBasis()
        layer = PoleAwareRational(d_p=2, d_q=2, basis=basis, enable_pole_head=True)

        assert layer.pole_head is not None
        assert layer.enable_pole_head
        assert hasattr(layer, "pole_predictions")

    def test_forward_with_pole_score(self):
        """Test forward pass with pole score."""
        basis = MonomialBasis()
        layer = PoleAwareRational(d_p=2, d_q=1, basis=basis, enable_pole_head=True)

        x = real(0.5)
        y, tag, pole_score = layer.forward_with_pole_score(x)

        assert tag in [TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI]
        assert pole_score is not None
        assert pole_score.tag == TRTag.REAL

    def test_Q_value_tracking(self):
        """Test Q value tracking."""
        basis = MonomialBasis()
        layer = PoleAwareRational(d_p=2, d_q=1, basis=basis, track_Q_values=True)

        x = real(0.5)
        y, tag = layer.forward(x)

        q_val = layer.get_Q_value()
        assert q_val is not None
        assert q_val >= 0  # Absolute value

    def test_pole_detection_evaluation(self):
        """Test pole detection evaluation."""
        basis = MonomialBasis()
        layer = PoleAwareRational(d_p=2, d_q=1, basis=basis, enable_pole_head=True)

        inputs = [0.0, 0.5, 1.0]
        true_poles = [True, False, False]

        metrics = layer.evaluate_pole_detection(inputs, true_poles)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert len(layer.pole_metrics_history) == 1


class TestFullyIntegratedRational:
    """Test fully integrated rational layer."""

    def test_initialization(self):
        """Test fully integrated layer initialization."""
        basis = MonomialBasis()
        layer = FullyIntegratedRational(
            d_p=2, d_q=1, basis=basis, enable_tag_head=True, enable_pole_head=True
        )

        assert layer.tag_head is not None
        assert layer.pole_head is not None

    def test_forward_fully_integrated(self):
        """Test fully integrated forward pass."""
        basis = MonomialBasis()
        layer = FullyIntegratedRational(
            d_p=2, d_q=1, basis=basis, enable_tag_head=True, enable_pole_head=True
        )

        x = real(0.5)
        result = layer.forward_fully_integrated(x)

        assert "output" in result
        assert "tag" in result
        assert "tag_logits" in result
        assert "pole_score" in result
        assert "pole_probability" in result

        # Check types
        assert result["tag"] in [TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI]
        assert isinstance(result["pole_probability"], float)
        assert 0.0 <= result["pole_probability"] <= 1.0

    def test_integration_summary(self):
        """Test integration summary."""
        basis = MonomialBasis()
        layer = FullyIntegratedRational(
            d_p=2, d_q=1, basis=basis, enable_tag_head=True, enable_pole_head=True
        )

        summary = layer.get_integration_summary()

        assert "tag_prediction_enabled" in summary
        assert "pole_detection_enabled" in summary
        assert "total_parameters" in summary

        assert summary["tag_prediction_enabled"] == True
        assert summary["pole_detection_enabled"] == True
        assert summary["total_parameters"] > 0


class TestDomainSpecificDetector:
    """Test domain-specific pole detection."""

    def test_robotics_singularity(self):
        """Test robotics singularity detection."""
        detector = DomainSpecificPoleDetector(domain="robotics")

        # Test RR robot singularity
        is_singular = detector.get_robotics_singularity([0.0, 0.0], robot_type="RR")
        assert is_singular == True  # θ2=0 is singular

        is_not_singular = detector.get_robotics_singularity([0.0, np.pi / 2], robot_type="RR")
        assert is_not_singular == False

    def test_custom_teacher_function(self):
        """Test custom teacher function."""
        detector = DomainSpecificPoleDetector()

        # Set custom function
        def near_zero(x):
            return abs(x) < 0.1

        detector.set_teacher_function(near_zero)

        labels = detector.generate_labels([0.05, 0.5, -0.08, 1.0])
        assert labels == [1.0, 0.0, 1.0, 0.0]

    def test_control_poles(self):
        """Test control system pole detection."""
        detector = DomainSpecificPoleDetector(domain="control")

        # Simple 2x2 system with eigenvalues near imaginary axis
        A = np.array([[0.05, 1], [-1, 0.05]])
        poles = detector.get_control_poles(A, threshold=0.1)

        assert len(poles) == 2  # Both eigenvalues have small real part
