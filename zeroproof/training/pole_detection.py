"""
Pole detection head and loss for Q≈0 localization.

This module implements an auxiliary network that learns to predict where
the denominator Q(x) approaches zero, enabling explicit pole localization
rather than just inferring from outputs.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..autodiff import TRNode
from ..autodiff.tr_ops_grad import tr_add, tr_div, tr_mul, tr_neg, tr_sub
from ..core import TRScalar, TRTag, ninf, phi, pinf, real


@dataclass
class PoleDetectionConfig:
    """Configuration for pole detection head."""

    hidden_dim: int = 16
    use_basis: bool = True
    activation: str = "tanh"  # "tanh", "sigmoid", "relu"
    dropout_rate: float = 0.0
    teacher_weight: float = 0.5  # Weight for teacher signal vs self-supervised
    proximity_threshold: float = 0.1  # |Q| threshold for pole proximity
    use_residual: bool = False  # Add residual connection
    normalize_output: bool = True  # Apply sigmoid to output


def sigmoid(x: TRNode) -> TRNode:
    """
    Compute sigmoid activation: 1/(1 + exp(-x)).

    Uses Taylor approximation for TR compatibility.

    Args:
        x: Input node

    Returns:
        Sigmoid output
    """
    # For TR compatibility, use approximation
    # sigmoid(x) ≈ 0.5 + 0.25*x - 0.03125*x³ for |x| < 2
    # For larger |x|, saturate to 0 or 1

    half = TRNode.constant(real(0.5))
    quarter = TRNode.constant(real(0.25))

    # Access tag properly through the TRNode's value
    x_val = x.value if hasattr(x, "value") else x
    if hasattr(x_val, "tag"):
        x_tag = x_val.tag
        x_real_value = x_val.value if x_tag == TRTag.REAL else 0.0
    else:
        # Assume it's a regular TRScalar
        x_tag = TRTag.REAL
        x_real_value = float(x_val) if hasattr(x_val, "__float__") else 0.0

    if x_tag != TRTag.REAL:
        # Non-REAL inputs saturate
        if x_tag == TRTag.PINF:
            return TRNode.constant(real(1.0))
        elif x_tag == TRTag.NINF:
            return TRNode.constant(real(0.0))
        else:
            return half

    # Check magnitude for saturation
    if x_real_value > 4:
        return TRNode.constant(real(0.99))
    elif x_real_value < -4:
        return TRNode.constant(real(0.01))

    # Taylor approximation for small values
    x2 = tr_mul(x, x)
    x3 = tr_mul(x2, x)
    coeff3 = TRNode.constant(real(-0.03125))

    result = half
    result = tr_add(result, tr_mul(quarter, x))
    result = tr_add(result, tr_mul(coeff3, x3))

    return result


def tanh_activation(x: TRNode) -> TRNode:
    """
    Compute tanh activation.

    Uses approximation: tanh(x) ≈ x - x³/3 for small x,
    saturates to ±1 for large |x|.

    Args:
        x: Input node

    Returns:
        Tanh output
    """
    # Access tag properly through the TRNode's value
    x_val = x.value if hasattr(x, "value") else x
    if hasattr(x_val, "tag"):
        x_tag = x_val.tag
        x_real_value = x_val.value if x_tag == TRTag.REAL else 0.0
    else:
        # Assume it's a regular TRScalar
        x_tag = TRTag.REAL
        x_real_value = float(x_val) if hasattr(x_val, "__float__") else 0.0

    if x_tag != TRTag.REAL:
        if x_tag == TRTag.PINF:
            return TRNode.constant(real(1.0))
        elif x_tag == TRTag.NINF:
            return TRNode.constant(real(-1.0))
        else:
            return TRNode.constant(real(0.0))

    # Saturation
    if x_real_value > 3:
        return TRNode.constant(real(0.99))
    elif x_real_value < -3:
        return TRNode.constant(real(-0.99))

    # Taylor approximation
    x2 = tr_mul(x, x)
    x3 = tr_mul(x2, x)
    third = TRNode.constant(real(1.0 / 3.0))

    result = x
    result = tr_sub(result, tr_mul(third, x3))

    return result


def relu_activation(x: TRNode) -> TRNode:
    """
    Compute ReLU activation: max(0, x).

    Args:
        x: Input node

    Returns:
        ReLU output
    """
    # Access tag properly through the TRNode's value
    x_val = x.value if hasattr(x, "value") else x
    if hasattr(x_val, "tag"):
        x_tag = x_val.tag
        x_real_value = x_val.value if x_tag == TRTag.REAL else 0.0
    else:
        # Assume it's a regular TRScalar
        x_tag = TRTag.REAL
        x_real_value = float(x_val) if hasattr(x_val, "__float__") else 0.0

    if x_tag == TRTag.REAL and x_real_value > 0:
        return x
    elif x_tag == TRTag.PINF:
        return x
    else:
        return TRNode.constant(real(0.0))


class PoleDetectionHead:
    """
    Auxiliary head for pole detection.

    Predicts probability that Q(x) ≈ 0 at given input x.
    Can be trained with teacher signals (where available) or
    self-supervised from model outputs.
    """

    def __init__(self, input_dim: int, config: Optional[PoleDetectionConfig] = None, basis=None):
        """
        Initialize pole detection head.

        Args:
            input_dim: Input dimension (basis degree)
            config: Pole detection configuration
            basis: Optional basis functions
        """
        self.input_dim = input_dim
        self.config = config or PoleDetectionConfig()
        self.basis = basis

        # Initialize network parameters
        self._initialize_parameters()

        # Tracking
        self.prediction_history = []
        self.teacher_accuracy = None

    def _initialize_parameters(self):
        """Initialize network parameters."""
        hidden = self.config.hidden_dim

        # Input layer: input_dim -> hidden_dim
        self.W1 = []
        self.b1 = []

        scale1 = math.sqrt(2.0 / (self.input_dim + 1))
        for i in range(hidden):
            row = []
            for j in range(self.input_dim + 1):  # +1 for bias term in basis
                val = np.random.randn() * scale1
                row.append(TRNode.parameter(real(val), name=f"pole_W1_{i}_{j}"))
            self.W1.append(row)

            self.b1.append(TRNode.parameter(real(0.0), name=f"pole_b1_{i}"))

        # Hidden layer: hidden_dim -> hidden_dim (optional second layer)
        if self.config.use_residual:
            self.W2 = []
            self.b2 = []

            scale2 = math.sqrt(2.0 / hidden)
            for i in range(hidden):
                row = []
                for j in range(hidden):
                    val = np.random.randn() * scale2
                    row.append(TRNode.parameter(real(val), name=f"pole_W2_{i}_{j}"))
                self.W2.append(row)

                self.b2.append(TRNode.parameter(real(0.0), name=f"pole_b2_{i}"))

        # Output layer: hidden_dim -> 1
        self.W_out = []
        scale_out = math.sqrt(2.0 / hidden)
        for j in range(hidden):
            val = np.random.randn() * scale_out
            self.W_out.append(TRNode.parameter(real(val), name=f"pole_Wout_{j}"))

        self.b_out = TRNode.parameter(real(0.0), name="pole_bout")

    def forward(self, x: TRNode) -> TRNode:
        """
        Forward pass to predict pole score.

        Args:
            x: Input value

        Returns:
            Pole score (before sigmoid if normalize_output=False)
        """
        # Get basis features
        if self.basis is not None:
            features = self.basis(x, self.input_dim)
        else:
            # Simple polynomial features
            features = [TRNode.constant(real(1.0))]  # Bias
            x_power = x
            for _ in range(self.input_dim):
                features.append(x_power)
                x_power = tr_mul(x_power, x)

        # First hidden layer
        hidden1 = []
        for i in range(self.config.hidden_dim):
            # Linear combination
            act = self.b1[i]
            for j, feat in enumerate(features[: len(self.W1[i])]):
                act = tr_add(act, tr_mul(self.W1[i][j], feat))

            # Apply activation
            if self.config.activation == "tanh":
                act = tanh_activation(act)
            elif self.config.activation == "sigmoid":
                act = sigmoid(act)
            elif self.config.activation == "relu":
                act = relu_activation(act)

            hidden1.append(act)

        # Optional second hidden layer with residual
        if self.config.use_residual:
            hidden2 = []
            for i in range(self.config.hidden_dim):
                # Linear combination
                act = self.b2[i]
                for j, h in enumerate(hidden1):
                    act = tr_add(act, tr_mul(self.W2[i][j], h))

                # Residual connection
                act = tr_add(act, hidden1[i])

                # Apply activation
                if self.config.activation == "tanh":
                    act = tanh_activation(act)
                elif self.config.activation == "sigmoid":
                    act = sigmoid(act)
                elif self.config.activation == "relu":
                    act = relu_activation(act)

                hidden2.append(act)

            final_hidden = hidden2
        else:
            final_hidden = hidden1

        # Output layer
        output = self.b_out
        for j, h in enumerate(final_hidden):
            output = tr_add(output, tr_mul(self.W_out[j], h))

        # Apply sigmoid normalization if requested
        if self.config.normalize_output:
            output = sigmoid(output)

        return output

    def parameters(self) -> List[TRNode]:
        """Get all trainable parameters."""
        params = []

        # First layer
        for row in self.W1:
            params.extend(row)
        params.extend(self.b1)

        # Optional second layer
        if self.config.use_residual:
            for row in self.W2:
                params.extend(row)
            params.extend(self.b2)

        # Output layer
        params.extend(self.W_out)
        params.append(self.b_out)

        return params

    def predict_pole_probability(self, x: TRNode) -> float:
        """
        Get pole probability as scalar.

        Args:
            x: Input value

        Returns:
            Probability that Q(x) ≈ 0
        """
        score = self.forward(x)

        if not self.config.normalize_output:
            score = sigmoid(score)

        # Access value properly
        score_val = score.value if hasattr(score, "value") else score
        if hasattr(score_val, "tag") and score_val.tag == TRTag.REAL:
            return float(score_val.value)
        else:
            return 0.5  # Default for non-REAL


def binary_cross_entropy(pred: TRNode, target: float, epsilon: float = 1e-7) -> TRNode:
    """
    Binary cross-entropy loss.

    BCE = -[t*log(p) + (1-t)*log(1-p)]

    Args:
        pred: Predicted probability (0 to 1)
        target: True label (0 or 1)
        epsilon: Small value for numerical stability

    Returns:
        BCE loss
    """
    eps = TRNode.constant(real(epsilon))
    one = TRNode.constant(real(1.0))
    target_node = TRNode.constant(real(target))

    # Clip prediction for stability
    safe_pred = pred
    pred_val = pred.value if hasattr(pred, "value") else pred
    if hasattr(pred_val, "tag") and pred_val.tag == TRTag.REAL:
        if pred_val.value < epsilon:
            safe_pred = eps
        elif pred_val.value > 1 - epsilon:
            safe_pred = TRNode.constant(real(1 - epsilon))

    # BCE using approximation for TR compatibility
    # Use squared error as simplified loss
    # loss = (pred - target)²

    diff = tr_add(safe_pred, TRNode.constant(real(-target)))
    loss = tr_mul(diff, diff)

    return loss


def compute_pole_loss(
    predictions: List[TRNode],
    pole_scores: List[TRNode],
    Q_values: Optional[List[float]] = None,
    teacher_labels: Optional[List[float]] = None,
    config: Optional[PoleDetectionConfig] = None,
) -> TRNode:
    """
    Compute loss for pole detection.

    Can use either:
    1. Teacher signals (if available)
    2. Self-supervised from |Q| values
    3. Weak supervision from output tags

    Args:
        predictions: Model outputs (for tag-based supervision)
        pole_scores: Predicted pole probabilities
        Q_values: Optional |Q(x)| values for self-supervision
        teacher_labels: Optional teacher pole labels
        config: Pole detection configuration

    Returns:
        Pole detection loss
    """
    if not pole_scores:
        return TRNode.constant(real(0.0))

    config = config or PoleDetectionConfig()
    losses = []

    for i, score in enumerate(pole_scores):
        target = None

        # Priority 1: Teacher labels
        if teacher_labels and i < len(teacher_labels):
            target = teacher_labels[i]
            weight = config.teacher_weight

        # Priority 2: Self-supervised from Q values
        elif Q_values and i < len(Q_values):
            q_val = Q_values[i]
            # Near pole if |Q| < threshold
            target = 1.0 if abs(q_val) < config.proximity_threshold else 0.0
            weight = 1.0 - config.teacher_weight

        # Priority 3: Weak supervision from tags
        elif predictions and i < len(predictions):
            tag = predictions[i].tag
            # Non-REAL suggests near pole
            target = 0.0 if tag == TRTag.REAL else 1.0
            weight = 0.5  # Lower confidence

        if target is not None:
            loss = binary_cross_entropy(score, target)
            weighted_loss = tr_mul(TRNode.constant(real(weight)), loss)
            losses.append(weighted_loss)

    if not losses:
        return TRNode.constant(real(0.0))

    # Average loss
    total = losses[0]
    for loss in losses[1:]:
        total = tr_add(total, loss)

    avg_loss = tr_div(total, TRNode.constant(real(float(len(losses)))))

    return avg_loss


def compute_pole_metrics(
    pole_scores: List[TRNode], true_poles: List[bool], threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics for pole detection.

    Args:
        pole_scores: Predicted pole probabilities
        true_poles: True pole indicators
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    if len(pole_scores) != len(true_poles):
        raise ValueError("Length mismatch")

    # Convert predictions to binary
    predictions = []
    for score in pole_scores:
        score_val = score.value if hasattr(score, "value") else score
        if hasattr(score_val, "tag") and score_val.tag == TRTag.REAL:
            prob = score_val.value
            predictions.append(prob > threshold)
        else:
            predictions.append(False)

    # Compute metrics
    tp = sum(1 for pred, true in zip(predictions, true_poles) if pred and true)
    fp = sum(1 for pred, true in zip(predictions, true_poles) if pred and not true)
    tn = sum(1 for pred, true in zip(predictions, true_poles) if not pred and not true)
    fn = sum(1 for pred, true in zip(predictions, true_poles) if not pred and true)

    metrics = {}

    # Accuracy
    total = tp + fp + tn + fn
    metrics["accuracy"] = (tp + tn) / total if total > 0 else 0

    # Precision
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"])
        )
    else:
        metrics["f1"] = 0

    # Counts
    metrics["true_positives"] = tp
    metrics["false_positives"] = fp
    metrics["true_negatives"] = tn
    metrics["false_negatives"] = fn

    return metrics


class DomainSpecificPoleDetector:
    """
    Domain-specific pole detector with teacher signals.

    Provides analytical pole indicators for specific domains
    like robotics (Jacobian singularities) or control (unstable poles).
    """

    def __init__(self, domain: str = "general"):
        """
        Initialize domain-specific detector.

        Args:
            domain: Application domain ("robotics", "control", "general")
        """
        self.domain = domain
        self.teacher_fn = None

    def set_teacher_function(self, fn: Callable[[float], bool]):
        """
        Set custom teacher function for pole detection.

        Args:
            fn: Function that returns True if input is near pole
        """
        self.teacher_fn = fn

    def get_robotics_singularity(self, joint_angles: List[float], robot_type: str = "RR") -> bool:
        """
        Check for kinematic singularity in robot.

        Args:
            joint_angles: Current joint configuration
            robot_type: Type of robot ("RR", "RP", "3R")

        Returns:
            True if configuration is singular
        """
        if robot_type == "RR":
            # 2-link planar robot
            # Singular when links are aligned (θ2 = 0 or π)
            if len(joint_angles) >= 2:
                theta2 = joint_angles[1]
                return abs(math.sin(theta2)) < 0.1

        elif robot_type == "RP":
            # Rotation-prismatic robot
            # Singular when prismatic joint at zero
            if len(joint_angles) >= 2:
                d = joint_angles[1]
                return abs(d) < 0.1

        return False

    def get_control_poles(self, system_matrix: np.ndarray, threshold: float = 0.1) -> List[complex]:
        """
        Find poles of control system.

        Args:
            system_matrix: System A matrix
            threshold: Distance threshold for pole proximity

        Returns:
            List of pole locations
        """
        # Eigenvalues are poles of the system
        eigenvalues = np.linalg.eigvals(system_matrix)

        # Filter near-imaginary axis (marginally stable)
        poles = []
        for eig in eigenvalues:
            if abs(eig.real) < threshold:
                poles.append(eig)

        return poles

    def generate_labels(self, inputs: List[float], **kwargs) -> List[float]:
        """
        Generate teacher labels for inputs.

        Args:
            inputs: Input values
            **kwargs: Domain-specific parameters

        Returns:
            Binary labels (1.0 for near-pole, 0.0 otherwise)
        """
        labels = []

        for x in inputs:
            if self.teacher_fn:
                is_pole = self.teacher_fn(x)
            elif self.domain == "robotics":
                is_pole = self.get_robotics_singularity([x], **kwargs)
            else:
                # Default: no teacher signal
                is_pole = False

            labels.append(1.0 if is_pole else 0.0)

        return labels
