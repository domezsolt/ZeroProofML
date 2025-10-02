"""
Unit tests for tag-loss functionality.

Tests the tag classification loss, tag prediction head,
and integration with rational layers.
"""

from typing import List

import numpy as np
import pytest

from zeroproof.autodiff import TRNode
from zeroproof.core import TRScalar, TRTag, ninf, phi, pinf, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.tag_aware_rational import TagAwareMultiRational, TagAwareRational
from zeroproof.training.tag_loss import (
    TagClass,
    TagPredictionHead,
    compute_tag_accuracy,
    compute_tag_confusion_matrix,
    compute_tag_loss,
    cross_entropy_loss,
    softmax,
)


class TestTagClass:
    """Test tag class enumeration and conversion."""

    def test_from_tag_conversion(self):
        """Test conversion from TRTag to TagClass."""
        assert TagClass.from_tag(TRTag.REAL) == TagClass.REAL
        assert TagClass.from_tag(TRTag.PINF) == TagClass.PINF
        assert TagClass.from_tag(TRTag.NINF) == TagClass.NINF
        assert TagClass.from_tag(TRTag.PHI) == TagClass.PHI

    def test_onehot_encoding(self):
        """Test one-hot encoding of tag classes."""
        assert TagClass.REAL.to_onehot() == [1.0, 0.0, 0.0, 0.0]
        assert TagClass.PINF.to_onehot() == [0.0, 1.0, 0.0, 0.0]
        assert TagClass.NINF.to_onehot() == [0.0, 0.0, 1.0, 0.0]
        assert TagClass.PHI.to_onehot() == [0.0, 0.0, 0.0, 1.0]


class TestSoftmax:
    """Test softmax computation."""

    def test_softmax_basic(self):
        """Test basic softmax functionality."""
        # Equal logits should give equal probabilities
        logits = [
            TRNode.constant(real(0.0)),
            TRNode.constant(real(0.0)),
            TRNode.constant(real(0.0)),
            TRNode.constant(real(0.0)),
        ]

        probs = softmax(logits)
        assert len(probs) == 4

        # Check that probabilities sum to 1
        total = probs[0]
        for p in probs[1:]:
            from zeroproof.core import tr_add

            total = tr_add(total.value, p.value)
            total = TRNode.constant(total)

        assert abs(total.value.value - 1.0) < 0.1  # Approximate due to Taylor series

    def test_softmax_with_temperature(self):
        """Test temperature scaling in softmax."""
        logits = [
            TRNode.constant(real(1.0)),
            TRNode.constant(real(0.0)),
            TRNode.constant(real(-1.0)),
            TRNode.constant(real(0.0)),
        ]

        # High temperature (more uniform)
        probs_high_temp = softmax(logits, temperature=10.0)

        # Low temperature (more peaked)
        probs_low_temp = softmax(logits, temperature=0.1)

        # First probability should be higher with low temperature
        assert probs_low_temp[0].value.value > probs_high_temp[0].value.value


class TestCrossEntropyLoss:
    """Test cross-entropy loss computation."""

    def test_perfect_prediction(self):
        """Test loss for perfect prediction."""
        # Perfect prediction: [1, 0, 0, 0] for class 0
        probs = [
            TRNode.constant(real(0.99)),  # Nearly 1
            TRNode.constant(real(0.003)),
            TRNode.constant(real(0.003)),
            TRNode.constant(real(0.004)),
        ]

        loss = cross_entropy_loss(probs, TagClass.REAL)

        # Loss should be very small
        assert loss.value.tag == TRTag.REAL
        assert loss.value.value < 0.1

    def test_wrong_prediction(self):
        """Test loss for wrong prediction."""
        # Wrong prediction: [0.1, 0.7, 0.1, 0.1] for class 0
        probs = [
            TRNode.constant(real(0.1)),
            TRNode.constant(real(0.7)),
            TRNode.constant(real(0.1)),
            TRNode.constant(real(0.1)),
        ]

        loss = cross_entropy_loss(probs, TagClass.REAL)

        # Loss should be high
        assert loss.value.tag == TRTag.REAL
        assert loss.value.value > 0.5


class TestTagPredictionHead:
    """Test the tag prediction head network."""

    def test_initialization(self):
        """Test head initialization."""
        head = TagPredictionHead(input_dim=3, hidden_dim=8, basis=MonomialBasis())

        # Check parameters are created
        params = head.parameters()
        expected_params = (
            (3 + 1) * 8
            + 8  # W1: (input_dim+1) x hidden_dim
            + 8 * 4  # b1: hidden_dim
            + 4  # W2: hidden_dim x 4  # b2: 4
        )
        assert len(params) == expected_params

    def test_forward_pass(self):
        """Test forward pass produces logits."""
        head = TagPredictionHead(input_dim=2, hidden_dim=4)

        x = TRNode.constant(real(0.5))
        logits = head.forward(x)

        # Should produce 4 logits
        assert len(logits) == 4

        # All should be TRNodes with REAL values
        for logit in logits:
            assert isinstance(logit, TRNode)
            assert logit.value.tag == TRTag.REAL

    def test_predict_tag(self):
        """Test tag prediction."""
        head = TagPredictionHead(input_dim=2, hidden_dim=4)

        x = TRNode.constant(real(0.5))
        pred_class, probs = head.predict_tag(x)

        # Should return a valid class
        assert pred_class in [TagClass.REAL, TagClass.PINF, TagClass.NINF, TagClass.PHI]

        # Probabilities should sum to approximately 1
        assert abs(sum(probs) - 1.0) < 0.2  # Loose bound due to approximation


class TestTagLossIntegration:
    """Test tag loss integration with predictions."""

    def test_compute_tag_loss_with_mixed_tags(self):
        """Test tag loss computation with mixed tags."""
        # Create predictions with different tags
        predictions = [
            TRNode.constant(real(1.0)),  # REAL
            TRNode.constant(pinf()),  # PINF
            TRNode.constant(ninf()),  # NINF
            TRNode.constant(phi()),  # PHI
        ]

        # Create mock logits
        tag_logits = [
            [
                TRNode.constant(real(1.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
            ],  # Predict REAL
            [
                TRNode.constant(real(0.0)),
                TRNode.constant(real(1.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
            ],  # Predict PINF
            [
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(1.0)),
                TRNode.constant(real(0.0)),
            ],  # Predict NINF
            [
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(0.0)),
                TRNode.constant(real(1.0)),
            ],  # Predict PHI
        ]

        loss = compute_tag_loss(predictions, tag_logits, weight=0.1)

        # Loss should be computed only for non-REAL samples
        assert loss.value.tag == TRTag.REAL
        assert loss.value.value >= 0.0

    def test_tag_accuracy_computation(self):
        """Test accuracy computation."""
        predictions = [
            TRNode.constant(real(1.0)),  # REAL
            TRNode.constant(pinf()),  # PINF
            TRNode.constant(ninf()),  # NINF
            TRNode.constant(phi()),  # PHI
        ]

        # Mix of correct and incorrect predictions
        tag_predictions = [
            TagClass.REAL,  # Correct
            TagClass.PINF,  # Correct
            TagClass.PINF,  # Wrong (should be NINF)
            TagClass.PHI,  # Correct
        ]

        accuracy = compute_tag_accuracy(predictions, tag_predictions)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        predictions = [
            TRNode.constant(real(1.0)),  # REAL
            TRNode.constant(real(2.0)),  # REAL
            TRNode.constant(pinf()),  # PINF
            TRNode.constant(ninf()),  # NINF
        ]

        tag_predictions = [
            TagClass.REAL,  # Correct
            TagClass.PINF,  # Wrong (REAL predicted as PINF)
            TagClass.PINF,  # Correct
            TagClass.NINF,  # Correct
        ]

        matrix = compute_tag_confusion_matrix(predictions, tag_predictions)

        # Check specific entries
        assert matrix["REAL"]["REAL"] == 1
        assert matrix["REAL"]["PINF"] == 1
        assert matrix["PINF"]["PINF"] == 1
        assert matrix["NINF"]["NINF"] == 1


class TestTagAwareRational:
    """Test tag-aware rational layer."""

    def test_layer_initialization(self):
        """Test layer creation with tag head."""
        layer = TagAwareRational(
            d_p=2, d_q=2, basis=MonomialBasis(), enable_tag_head=True, tag_head_hidden_dim=4
        )

        assert layer.enable_tag_head == True
        assert layer.tag_head is not None

    def test_forward_with_tag_prediction(self):
        """Test forward pass with tag prediction."""
        layer = TagAwareRational(d_p=1, d_q=1, basis=MonomialBasis(), enable_tag_head=True)

        # Set parameters
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(0.0)
        layer.phi[0]._value = real(-2.0)  # Q = 1 - 2x

        x = TRNode.constant(real(0.3))
        y, tag, tag_logits = layer.forward_with_tag_pred(x)

        # Should return output, tag, and logits
        assert y.value.tag == TRTag.REAL
        assert tag == TRTag.REAL
        assert tag_logits is not None
        assert len(tag_logits) == 4  # 4 classes

    def test_parameters_include_tag_head(self):
        """Test that parameters include tag head weights."""
        layer = TagAwareRational(d_p=1, d_q=1, enable_tag_head=True, tag_head_hidden_dim=2)

        params_without_head = 2 + 1  # theta + phi
        params_with_head = layer.num_parameters()

        # Should have more parameters with tag head
        assert params_with_head > params_without_head


class TestTagAwareMultiRational:
    """Test multi-output tag-aware rational layer."""

    def test_shared_tag_head(self):
        """Test that tag head is shared across outputs."""
        layer = TagAwareMultiRational(
            d_p=2, d_q=2, n_outputs=3, shared_Q=True, enable_tag_head=True
        )

        # Should have single shared tag head
        assert layer.tag_head is not None

        x = TRNode.constant(real(0.5))
        outputs, tag_logits = layer.forward_with_tag_pred(x)

        # Should get multiple outputs but single tag prediction
        assert len(outputs) == 3
        assert tag_logits is not None
        assert len(tag_logits) == 4  # 4 classes


def test_gradient_flow_through_tag_loss():
    """Test that gradients flow through tag loss."""
    # Create a simple model with tag head
    layer = TagAwareRational(d_p=1, d_q=1, enable_tag_head=True, tag_head_hidden_dim=2)

    # Forward pass
    x = TRNode.parameter(real(0.5))
    y, tag, tag_logits = layer.forward_with_tag_pred(x)

    # Create mock predictions for tag loss
    predictions = [y]
    all_logits = [tag_logits]

    # Compute tag loss
    loss = compute_tag_loss(predictions, all_logits, weight=1.0)

    # If tag is non-REAL, loss should be computed
    if tag != TRTag.REAL:
        assert loss.value.value > 0

        # Backward pass
        loss.backward()

        # Check gradients exist for tag head parameters
        for param in layer.tag_head.parameters():
            if param.requires_grad:
                assert param.gradient is not None
