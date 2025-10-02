"""
Tag-loss implementation for non-REAL outputs.

This module enables non-REAL samples (PINF/NINF/PHI) to contribute
supervision through auxiliary classification losses. Instead of being
completely ignored, these samples help the model learn the geometry
and types of singularities.
"""

import math
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from ..autodiff import TRNode, tr_abs, tr_add, tr_div, tr_mul, tr_pow_int, tr_sub
from ..core import TRScalar, TRTag, ninf, phi, pinf, real


class TagClass(IntEnum):
    """Integer encoding for tag classification."""

    REAL = 0
    PINF = 1
    NINF = 2
    PHI = 3

    @classmethod
    def from_tag(cls, tag: TRTag) -> "TagClass":
        """Convert TRTag to TagClass."""
        if tag == TRTag.REAL:
            return cls.REAL
        elif tag == TRTag.PINF:
            return cls.PINF
        elif tag == TRTag.NINF:
            return cls.NINF
        elif tag == TRTag.PHI:
            return cls.PHI
        else:
            raise ValueError(f"Unknown tag: {tag}")

    def to_onehot(self) -> List[float]:
        """Convert to one-hot encoding."""
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[self.value] = 1.0
        return vec


def softmax(logits: List[TRNode], temperature: float = 1.0) -> List[TRNode]:
    """
    Compute softmax probabilities from logits.

    Uses the numerically stable version: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        logits: List of logit values
        temperature: Temperature parameter for sharpness

    Returns:
        List of probability values
    """
    if not logits:
        return []

    # Ensure TRNode inputs
    logits = [
        logit if isinstance(logit, TRNode) else TRNode.constant(real(float(logit)))
        for logit in logits
    ]

    # Find max for numerical stability (prefer REAL)
    max_logit = logits[0]
    for logit in logits[1:]:
        if max_logit.tag != TRTag.REAL and logit.tag == TRTag.REAL:
            max_logit = logit
        elif logit.tag == TRTag.REAL and max_logit.tag == TRTag.REAL:
            if logit.value.value > max_logit.value.value:
                max_logit = logit

    # Compute exp(x - max) / temp
    temp_node = TRNode.constant(real(temperature))
    exp_values = []

    for logit in logits:
        # x - max
        shifted = tr_sub(logit, max_logit)
        # (x - max) / temp
        scaled = tr_div(shifted, temp_node)

        # Simplified exp approximation (for TR compatibility)
        # Using Taylor series: exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
        # This is sufficient for demonstration; full exp would need proper TR implementation
        x = scaled
        x2 = tr_mul(x, x)
        x3 = tr_mul(x2, x)

        half = TRNode.constant(real(0.5))
        sixth = TRNode.constant(real(1.0 / 6.0))

        exp_x = TRNode.constant(real(1.0))
        exp_x = tr_add(exp_x, x)
        exp_x = tr_add(exp_x, tr_mul(half, x2))
        exp_x = tr_add(exp_x, tr_mul(sixth, x3))

        exp_values.append(exp_x)

    # Compute sum over REALs to avoid PHI/infs, and ensure positivity
    exp_sum = TRNode.constant(real(0.0))
    for exp_val in exp_values:
        if exp_val.tag == TRTag.REAL and exp_val.value.value > 0.0:
            exp_sum = tr_add(exp_sum, exp_val)

    # If sum invalid, fallback to uniform distribution in [0,1]
    if exp_sum.tag != TRTag.REAL or exp_sum.value.value <= 0.0:
        uniform = TRNode.constant(real(1.0 / float(len(exp_values))))
        return [uniform for _ in exp_values]

    # Normalize
    probabilities = []
    for exp_val in exp_values:
        if exp_val.tag == TRTag.REAL and exp_val.value.value > 0.0:
            prob = tr_div(exp_val, exp_sum)
        else:
            prob = TRNode.constant(real(0.0))
        probabilities.append(prob)

    return probabilities


def cross_entropy_loss(
    pred_probs: List[TRNode], true_class: TagClass, epsilon: float = 1e-7
) -> TRNode:
    """
    Compute cross-entropy loss for tag classification.

    CE = -log(p_true_class)

    Args:
        pred_probs: Predicted probabilities for each class
        true_class: True tag class
        epsilon: Small value to avoid log(0)

    Returns:
        Cross-entropy loss
    """
    if len(pred_probs) != 4:
        raise ValueError("Expected 4 probability values for 4 classes")

    # Get probability for true class
    true_prob = pred_probs[true_class.value]

    # Clamp probability into [epsilon, 1] in TR domain
    eps_node = TRNode.constant(real(epsilon))
    one_node = TRNode.constant(real(1.0))
    safe_prob = tr_add(true_prob, eps_node)
    # Ensure upper bound by simple min approximation: p <= 1
    if safe_prob.tag == TRTag.REAL and safe_prob.value.value > 1.0:
        safe_prob = one_node

    # -log(p) using Taylor approximation for small deviations from 1
    # log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
    # For general x, we use -log(x) ≈ large_penalty * (1 - x) as approximation

    one = TRNode.constant(real(1.0))
    deviation = tr_sub(one, safe_prob)

    # Simple approximation: -log(p) ≈ -log(1 - (1-p)) ≈ (1-p) + (1-p)²/2
    dev2 = tr_mul(deviation, deviation)
    half = TRNode.constant(real(0.5))

    loss = deviation
    loss = tr_add(loss, tr_mul(half, dev2))

    return loss


def compute_tag_loss(
    predictions: List[TRNode],
    tag_logits: List[List[TRNode]],
    weight: float = 0.05,
    adaptive_weight: bool = False,
    coverage: Optional[float] = None,
) -> TRNode:
    """
    Compute auxiliary tag classification loss.

    Args:
        predictions: Main model predictions (used to extract true tags)
        tag_logits: Predicted tag logits [batch_size, 4]
        weight: Base weight for tag loss
        adaptive_weight: Whether to adapt weight based on coverage
        coverage: Current coverage (required if adaptive_weight=True)

    Returns:
        Weighted tag loss
    """
    if not predictions or not tag_logits:
        return TRNode.constant(real(0.0))

    if len(predictions) != len(tag_logits):
        raise ValueError("Batch size mismatch between predictions and tag logits")

    # Adapt weight if coverage is too high
    effective_weight = weight
    if adaptive_weight and coverage is not None:
        # When coverage > 98%, increase weight to encourage exploration
        if coverage > 0.98:
            # Scale up to 4x the base weight when coverage = 100%
            scale = (coverage - 0.98) / 0.02
            effective_weight = weight * (1 + 3 * scale)  # 1x to 4x scaling

    losses = []
    for pred, logits in zip(predictions, tag_logits):
        # Get true tag
        true_tag = pred.tag
        true_class = TagClass.from_tag(true_tag)

        # Compute loss for all outputs (not just non-REAL)
        # This helps the model learn to predict REAL tags correctly too
        # Compute softmax probabilities
        probs = softmax(logits)

        # Compute cross-entropy loss
        ce_loss = cross_entropy_loss(probs, true_class)

        # Weight non-REAL predictions more heavily
        if true_tag != TRTag.REAL:
            # Double weight for non-REAL to encourage learning these cases
            ce_loss = tr_mul(TRNode.constant(real(2.0)), ce_loss)

        losses.append(ce_loss)

    if not losses:
        return TRNode.constant(real(0.0))

    # Average loss
    total = losses[0]
    for loss in losses[1:]:
        total = tr_add(total, loss)

    avg_loss = tr_div(total, TRNode.constant(real(float(len(losses)))))

    # Apply effective weight
    weighted_loss = tr_mul(TRNode.constant(real(effective_weight)), avg_loss)

    return weighted_loss


class TagPredictionHead:
    """
    Auxiliary head for tag prediction.

    This can be attached to a rational layer to predict
    the tag (REAL/PINF/NINF/PHI) of the output.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8, basis=None):
        """
        Initialize tag prediction head.

        Args:
            input_dim: Input dimension (basis degree)
            hidden_dim: Hidden layer dimension
            basis: Basis functions to use
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.basis = basis

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize network parameters."""

        # Hidden layer: input_dim+1 (bias feature) -> hidden_dim
        self.W1 = []
        scale1 = math.sqrt(2.0 / max(1, self.input_dim))
        for i in range(self.hidden_dim):
            row = []
            for j in range(self.input_dim + 1):  # +1 for constant basis term
                val = (2 * ((i + j) % 2) - 1) * scale1 * 0.1
                row.append(TRNode.parameter(real(val), name=f"W1_{i}_{j}"))
            self.W1.append(row)

        # Bias for hidden layer
        self.b1 = []
        for i in range(self.hidden_dim):
            self.b1.append(TRNode.parameter(real(0.0), name=f"b1_{i}"))

        # Output layer: hidden_dim -> 4 (classes)
        self.W2 = []
        scale2 = math.sqrt(2.0 / max(1, self.hidden_dim))
        for i in range(4):  # 4 tag classes
            row = []
            for j in range(self.hidden_dim):
                val = (2 * ((i + j) % 2) - 1) * scale2 * 0.1
                row.append(TRNode.parameter(real(val), name=f"W2_{i}_{j}"))
            self.W2.append(row)

        # Bias for output layer
        self.b2 = []
        for i in range(4):
            self.b2.append(TRNode.parameter(real(0.0), name=f"b2_{i}"))

    def forward(self, x: TRNode) -> List[TRNode]:
        """
        Forward pass to predict tag logits.

        Args:
            x: Input value

        Returns:
            List of 4 logits for tag classes
        """
        # Evaluate basis if provided
        if self.basis is not None:
            features = self.basis(x, self.input_dim)
        else:
            # Simple polynomial features: [1, x, x^2, ...]
            features = [TRNode.constant(real(1.0))]
            x_power = x
            for _ in range(self.input_dim - 1):
                features.append(x_power)
                x_power = tr_mul(x_power, x)

        # Hidden layer: ReLU(W1 @ features + b1)
        hidden = []
        for i in range(self.hidden_dim):
            # Compute W1[i] @ features
            activation = self.b1[i]
            for j, feat in enumerate(features[: len(self.W1[i])]):
                activation = tr_add(activation, tr_mul(self.W1[i][j], feat))

            # Simple ReLU: max(0, x)
            # For TR, approximate with half-wave linear: (x + |x|)/2
            abs_act = tr_abs(activation)
            relu = tr_mul(TRNode.constant(real(0.5)), tr_add(activation, abs_act))
            hidden.append(relu)

        # Output layer: W2 @ hidden + b2
        logits = []
        for i in range(4):
            # Compute W2[i] @ hidden
            output = self.b2[i]
            for j, h in enumerate(hidden):
                output = tr_add(output, tr_mul(self.W2[i][j], h))
            logits.append(output)

        return logits

    def parameters(self) -> List[TRNode]:
        """Get all trainable parameters."""
        params = []

        # Hidden layer parameters
        for row in self.W1:
            params.extend(row)
        params.extend(self.b1)

        # Output layer parameters
        for row in self.W2:
            params.extend(row)
        params.extend(self.b2)

        return params

    def predict_tag(self, x: TRNode) -> Tuple[TagClass, List[float]]:
        """
        Predict tag class from input.

        Args:
            x: Input value

        Returns:
            Tuple of (predicted_class, probabilities)
        """
        logits = self.forward(x)
        probs = softmax(logits)

        # Extract probability values
        prob_values = []
        for p in probs:
            if p.tag == TRTag.REAL:
                prob_values.append(p.value.value)
            else:
                prob_values.append(0.0)

        # Find argmax
        max_idx = 0
        max_prob = prob_values[0]
        for i, p in enumerate(prob_values[1:], 1):
            if p > max_prob:
                max_prob = p
                max_idx = i

        predicted_class = TagClass(max_idx)

        return predicted_class, prob_values


def compute_tag_accuracy(predictions: List[TRNode], tag_predictions: List[TagClass]) -> float:
    """
    Compute accuracy of tag predictions.

    Args:
        predictions: Model outputs (for true tags)
        tag_predictions: Predicted tag classes

    Returns:
        Accuracy score
    """
    if len(predictions) != len(tag_predictions):
        raise ValueError("Length mismatch")

    correct = 0
    total = 0

    for pred, tag_pred in zip(predictions, tag_predictions):
        true_tag = pred.tag
        true_class = TagClass.from_tag(true_tag)

        if true_class == tag_pred:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compute_tag_confusion_matrix(
    predictions: List[TRNode], tag_predictions: List[TagClass]
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix for tag predictions.

    Args:
        predictions: Model outputs (for true tags)
        tag_predictions: Predicted tag classes

    Returns:
        Confusion matrix as nested dict
    """
    matrix = {}
    for true_cls in TagClass:
        matrix[true_cls.name] = {}
        for pred_cls in TagClass:
            matrix[true_cls.name][pred_cls.name] = 0

    for pred, tag_pred in zip(predictions, tag_predictions):
        true_tag = pred.tag
        true_class = TagClass.from_tag(true_tag)

        matrix[true_class.name][tag_pred.name] += 1

    return matrix
