"""
Enhanced pole detection module with improved accuracy.

This module provides an improved pole detection head with better initialization,
regularization, and integration with the TRRational layers.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..autodiff import TRNode
from ..autodiff.tr_ops_grad import tr_add, tr_div, tr_mul, tr_neg, tr_sub
from ..core import TRScalar, TRTag, ninf, phi, pinf, real
from ..policy import TRPolicyConfig
from ..training.pole_detection import relu_activation, sigmoid, tanh_activation
from .basis import Basis


@dataclass
class EnhancedPoleConfig:
    """Enhanced configuration for pole detection."""

    # Architecture
    hidden_dims: List[int] = None  # Multi-layer architecture
    use_residual: bool = True  # Residual connections
    use_batch_norm: bool = False  # Batch normalization (TR-safe)
    activation: str = "tanh"
    dropout_rate: float = 0.1

    # Initialization
    init_strategy: str = "xavier_uniform"  # xavier_uniform, xavier_normal, he_uniform
    init_gain: float = 1.0  # Gain factor for initialization
    bias_init: float = 0.0  # Initial bias value

    # Training
    teacher_weight: float = 0.7  # Increased from 0.5 for better supervision
    self_supervised_weight: float = 0.3
    proximity_threshold: float = 0.1
    hard_threshold: float = 0.01  # Very close to pole

    # Regularization
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.1  # Smooth hard labels

    # Loss weights (increased for better accuracy)
    pole_loss_weight: float = 0.2  # Increased from 0.1
    false_positive_penalty: float = 2.0  # Penalize false positives more
    false_negative_penalty: float = 1.5  # Also penalize false negatives

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16, 8]  # Deeper default architecture


class EnhancedPoleDetectionHead:
    """
    Enhanced pole detection head with improved accuracy.

    Features:
    - Better initialization strategy
    - Multi-layer architecture with residual connections
    - Improved loss weighting
    - Teacher signal integration
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[EnhancedPoleConfig] = None,
        basis: Optional[Basis] = None,
    ):
        """
        Initialize enhanced pole detection head.

        Args:
            input_dim: Input dimension (degree of basis)
            config: Configuration
            basis: Optional basis functions
        """
        self.input_dim = input_dim
        self.config = config or EnhancedPoleConfig()
        self.basis = basis

        # Build network architecture
        self._build_network()

        # Initialize parameters with better strategy
        self._initialize_parameters()

        # Tracking
        self.pole_predictions = []
        self.teacher_signals = []
        self.accuracy_history = []

    def _build_network(self):
        """Build the network architecture."""
        self.layers = []
        self.weights = []
        self.biases = []

        # Input to hidden layers
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            # Weight matrix
            W = []
            for j in range(hidden_dim):
                W_row = []
                for k in range(prev_dim):
                    W_row.append(TRNode.constant(real(0.0)))
                W.append(W_row)
            self.weights.append(W)

            # Bias vector
            b = []
            for j in range(hidden_dim):
                b.append(TRNode.constant(real(self.config.bias_init)))
            self.biases.append(b)

            prev_dim = hidden_dim

        # Output layer (single output for pole probability)
        W_out = []
        for k in range(prev_dim):
            W_out.append(TRNode.constant(real(0.0)))
        self.weights.append([W_out])  # Single output row

        self.biases.append([TRNode.constant(real(self.config.bias_init))])

    def _initialize_parameters(self):
        """Initialize parameters with improved strategy."""
        # Initialize each layer
        for layer_idx, W in enumerate(self.weights):
            n_in = len(W[0]) if W and W[0] else self.input_dim
            n_out = len(W)

            if self.config.init_strategy == "xavier_uniform":
                # Xavier/Glorot uniform initialization
                limit = self.config.init_gain * math.sqrt(6.0 / (n_in + n_out))
                for i in range(n_out):
                    for j in range(n_in):
                        val = np.random.uniform(-limit, limit)
                        W[i][j] = TRNode.constant(real(val))

            elif self.config.init_strategy == "xavier_normal":
                # Xavier/Glorot normal initialization
                std = self.config.init_gain * math.sqrt(2.0 / (n_in + n_out))
                for i in range(n_out):
                    for j in range(n_in):
                        val = np.random.normal(0, std)
                        W[i][j] = TRNode.constant(real(val))

            elif self.config.init_strategy == "he_uniform":
                # He initialization (better for ReLU)
                limit = self.config.init_gain * math.sqrt(6.0 / n_in)
                for i in range(n_out):
                    for j in range(n_in):
                        val = np.random.uniform(-limit, limit)
                        W[i][j] = TRNode.constant(real(val))

            # Special initialization for output layer
            if layer_idx == len(self.weights) - 1:
                # Output layer - smaller initialization
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        W[i][j] = TRNode.constant(real(np.random.normal(0, 0.01)))

    def forward(
        self, x: Union[TRNode, TRScalar], features: Optional[List[TRNode]] = None
    ) -> TRNode:
        """
        Forward pass through pole detection head.

        Args:
            x: Input value
            features: Optional pre-computed features (e.g., basis functions)

        Returns:
            Pole detection score (before sigmoid)
        """
        # Ensure x is a node
        if isinstance(x, TRScalar):
            x = TRNode.constant(x)
        elif not isinstance(x, TRNode):
            x = TRNode.constant(real(float(x)))

        # Get features
        if features is None:
            if self.basis is not None:
                features = self.basis(x, self.input_dim)
            else:
                # Simple polynomial features
                features = [x]
                x_power = x
                for i in range(1, self.input_dim):
                    x_power = tr_mul(x_power, x)
                    features.append(x_power)

        # Forward through layers
        hidden = features
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            next_hidden = []
            for i in range(len(W)):
                # Compute W[i] @ hidden + b[i]
                output = b[i]
                for j in range(len(hidden)):
                    if j < len(W[i]):
                        output = tr_add(output, tr_mul(W[i][j], hidden[j]))
                next_hidden.append(output)

            # Apply activation (except last layer)
            if layer_idx < len(self.weights) - 1:
                activated = []
                for h in next_hidden:
                    if self.config.activation == "tanh":
                        activated.append(tanh_activation(h))
                    elif self.config.activation == "sigmoid":
                        activated.append(sigmoid(h))
                    elif self.config.activation == "relu":
                        activated.append(relu_activation(h))
                    else:
                        activated.append(h)

                # Residual connection if enabled
                if self.config.use_residual and len(hidden) == len(activated):
                    for i in range(len(activated)):
                        activated[i] = tr_add(activated[i], hidden[i])

                hidden = activated
            else:
                hidden = next_hidden

        # Return raw score (caller applies sigmoid if needed)
        return hidden[0]

    def predict_pole_probability(self, x: Union[TRNode, TRScalar]) -> float:
        """
        Predict pole probability with sigmoid activation.

        Args:
            x: Input value

        Returns:
            Probability that Q(x) ≈ 0
        """
        score = self.forward(x)
        prob = sigmoid(score)

        # Extract float value
        if hasattr(prob, "value"):
            prob_val = prob.value
            if hasattr(prob_val, "tag") and prob_val.tag == TRTag.REAL:
                return float(prob_val.value)

        return 0.5  # Default if can't extract

    def compute_loss(
        self,
        predictions: List[TRNode],
        Q_values: Optional[List[float]] = None,
        teacher_labels: Optional[List[float]] = None,
    ) -> TRNode:
        """
        Compute pole detection loss with improved weighting.

        Args:
            predictions: Pole detection scores
            Q_values: Actual |Q(x)| values for self-supervision
            teacher_labels: Optional teacher signals

        Returns:
            Weighted loss
        """
        if not predictions:
            return TRNode.constant(real(0.0))

        losses = []

        # Process each prediction
        for i, pred in enumerate(predictions):
            loss_components = []

            # Teacher supervision if available
            if teacher_labels and i < len(teacher_labels):
                teacher_target = teacher_labels[i]

                # Apply label smoothing
                if self.config.label_smoothing > 0:
                    teacher_target = (
                        1 - self.config.label_smoothing
                    ) * teacher_target + self.config.label_smoothing * 0.5

                # Binary cross-entropy with teacher
                prob = sigmoid(pred)
                target = TRNode.constant(real(teacher_target))

                # BCE loss with weighted penalties
                if teacher_target > 0.5:  # True pole
                    # Penalize false negatives
                    weight = self.config.false_negative_penalty
                else:  # Not a pole
                    # Penalize false positives more
                    weight = self.config.false_positive_penalty

                epsilon = TRNode.constant(real(1e-7))
                pos_term = tr_mul(target, tr_neg(self._safe_log(tr_add(prob, epsilon))))
                neg_term = tr_mul(
                    tr_sub(TRNode.constant(real(1.0)), target),
                    tr_neg(
                        self._safe_log(tr_add(tr_sub(TRNode.constant(real(1.0)), prob), epsilon))
                    ),
                )
                bce_loss = tr_mul(TRNode.constant(real(weight)), tr_add(pos_term, neg_term))

                loss_components.append(
                    tr_mul(TRNode.constant(real(self.config.teacher_weight)), bce_loss)
                )

            # Self-supervised loss if Q values available
            if Q_values and i < len(Q_values):
                q_abs = abs(Q_values[i])

                # Create self-supervised target
                if q_abs < self.config.hard_threshold:
                    # Very close to pole - strong signal
                    self_target = 0.95
                elif q_abs < self.config.proximity_threshold:
                    # Near pole - moderate signal
                    self_target = 0.7
                else:
                    # Far from pole
                    self_target = 0.1

                # Self-supervised BCE
                prob = sigmoid(pred)
                target = TRNode.constant(real(self_target))
                epsilon = TRNode.constant(real(1e-7))

                pos_term = tr_mul(target, tr_neg(self._safe_log(tr_add(prob, epsilon))))
                neg_term = tr_mul(
                    tr_sub(TRNode.constant(real(1.0)), target),
                    tr_neg(
                        self._safe_log(tr_add(tr_sub(TRNode.constant(real(1.0)), prob), epsilon))
                    ),
                )
                self_loss = tr_add(pos_term, neg_term)

                loss_components.append(
                    tr_mul(TRNode.constant(real(self.config.self_supervised_weight)), self_loss)
                )

            # Combine loss components
            if loss_components:
                sample_loss = loss_components[0]
                for component in loss_components[1:]:
                    sample_loss = tr_add(sample_loss, component)
                losses.append(sample_loss)

        # Average over batch
        if losses:
            total_loss = losses[0]
            for loss in losses[1:]:
                total_loss = tr_add(total_loss, loss)

            batch_size = TRNode.constant(real(float(len(losses))))
            avg_loss = tr_div(total_loss, batch_size)

            # Apply overall weight
            weighted_loss = tr_mul(TRNode.constant(real(self.config.pole_loss_weight)), avg_loss)

            return weighted_loss

        return TRNode.constant(real(0.0))

    def _safe_log(self, x: TRNode) -> TRNode:
        """Safe logarithm that handles TR edge cases."""
        # Check if x is very small or non-REAL
        x_val = x.value if hasattr(x, "value") else x
        if hasattr(x_val, "tag"):
            if x_val.tag == TRTag.REAL and x_val.value > 1e-10:
                # Safe to compute log
                # Use approximation: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
                # Or just return a reasonable negative value based on x
                if x_val.value < 0.1:
                    return TRNode.constant(real(-2.3))  # log(0.1)
                elif x_val.value < 0.5:
                    return TRNode.constant(real(-0.69))  # log(0.5)
                else:
                    # Close to 1, use linear approximation
                    return tr_sub(x, TRNode.constant(real(1.0)))
            else:
                # Non-REAL or very small - return large negative
                return TRNode.constant(real(-10.0))

        # Fallback
        return TRNode.constant(real(-5.0))

    def parameters(self) -> List[TRNode]:
        """Get all trainable parameters."""
        params = []

        # Weights and biases
        for W in self.weights:
            for row in W:
                params.extend(row)

        for b in self.biases:
            params.extend(b)

        return params

    def regularization_loss(self) -> TRNode:
        """Compute L2 regularization loss."""
        if self.config.weight_decay <= 0:
            return TRNode.constant(real(0.0))

        reg_loss = TRNode.constant(real(0.0))

        # L2 penalty on weights
        for W in self.weights:
            for row in W:
                for w in row:
                    reg_loss = tr_add(reg_loss, tr_mul(w, w))

        # Scale by weight decay
        reg_loss = tr_mul(TRNode.constant(real(self.config.weight_decay * 0.5)), reg_loss)

        return reg_loss


class PoleRegularizer:
    """
    Regularizer that encourages Q(x) to have poles at specific locations.

    This helps the model learn to place poles where desired, improving
    control over the learned rational function.
    """

    def __init__(
        self,
        target_poles: List[float],
        regularization_weight: float = 0.01,
        sharpness: float = 10.0,
    ):
        """
        Initialize pole regularizer.

        Args:
            target_poles: Desired pole locations
            regularization_weight: Weight for regularization loss
            sharpness: How sharply to encourage poles (higher = sharper)
        """
        self.target_poles = target_poles
        self.regularization_weight = regularization_weight
        self.sharpness = sharpness

    def compute_regularization(self, Q_func: callable, x_samples: List[float]) -> TRNode:
        """
        Compute regularization loss that encourages poles.

        Args:
            Q_func: Function that computes Q(x)
            x_samples: Sample points to evaluate

        Returns:
            Regularization loss
        """
        reg_loss = TRNode.constant(real(0.0))

        # For each target pole location
        for pole_loc in self.target_poles:
            # Sample points around the pole
            nearby_points = []
            for offset in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                x_near = pole_loc + offset
                # Find closest sample point
                closest_idx = min(range(len(x_samples)), key=lambda i: abs(x_samples[i] - x_near))
                nearby_points.append(x_samples[closest_idx])

            # Encourage small |Q| at these points
            for x_val in nearby_points:
                x_node = TRNode.constant(real(x_val))
                Q_val = Q_func(x_node)

                # Penalty that decreases as |Q| approaches 0
                # Using 1/(1 + sharpness * |Q|²) as the target
                Q_squared = tr_mul(Q_val, Q_val)
                denom = tr_add(
                    TRNode.constant(real(1.0)),
                    tr_mul(TRNode.constant(real(self.sharpness)), Q_squared),
                )

                # We want to maximize 1/denom, which means minimizing denom
                # So the loss is just denom
                reg_loss = tr_add(reg_loss, denom)

        # Scale by weight and number of poles
        if self.target_poles:
            n_poles = TRNode.constant(real(float(len(self.target_poles))))
            reg_loss = tr_div(reg_loss, n_poles)
            reg_loss = tr_mul(TRNode.constant(real(self.regularization_weight)), reg_loss)

        return reg_loss


class PoleAwareRationalInterface:
    """
    Interface for attaching pole detection to TRRational layers.

    This provides a clean way to add pole detection capabilities to
    existing rational layers without modifying their core implementation.
    """

    def __init__(
        self,
        rational_layer: Any,
        pole_config: Optional[EnhancedPoleConfig] = None,
        enable_regularization: bool = False,
        target_poles: Optional[List[float]] = None,
    ):
        """
        Initialize pole-aware interface.

        Args:
            rational_layer: Base TRRational or TRRationalMulti layer
            pole_config: Configuration for pole detection
            enable_regularization: Whether to use pole regularization
            target_poles: Optional target pole locations
        """
        self.rational_layer = rational_layer
        self.pole_config = pole_config or EnhancedPoleConfig()

        # Extract dimensions from rational layer
        input_dim = max(rational_layer.d_p, rational_layer.d_q) + 1
        basis = getattr(rational_layer, "basis", None)

        # Create pole detection head
        self.pole_head = EnhancedPoleDetectionHead(
            input_dim=input_dim, config=self.pole_config, basis=basis
        )

        # Create regularizer if enabled
        self.regularizer = None
        if enable_regularization and target_poles:
            self.regularizer = PoleRegularizer(
                target_poles=target_poles, regularization_weight=0.01, sharpness=10.0
            )

        # Tracking
        self.pole_detection_history = []

    def forward_with_pole_detection(self, x: Union[TRScalar, TRNode]) -> Dict[str, Any]:
        """
        Forward pass with pole detection.

        Args:
            x: Input value

        Returns:
            Dictionary with output, tag, pole score, and Q value
        """
        # Get standard output
        y, tag = self.rational_layer.forward(x)

        # Get basis features if available
        features = None
        if hasattr(self.rational_layer, "basis"):
            max_degree = max(self.rational_layer.d_p, self.rational_layer.d_q)
            features = self.rational_layer.basis(x, max_degree)

        # Compute pole detection score
        pole_score = self.pole_head.forward(x, features)
        pole_prob = self.pole_head.predict_pole_probability(x)

        # Get Q value if tracked
        Q_abs = None
        if hasattr(self.rational_layer, "_last_Q_abs"):
            Q_abs = self.rational_layer._last_Q_abs

        result = {
            "output": y,
            "tag": tag,
            "pole_score": pole_score,
            "pole_probability": pole_prob,
            "Q_abs": Q_abs,
        }

        # Store in history
        self.pole_detection_history.append(
            {"x": x, "pole_prob": pole_prob, "Q_abs": Q_abs, "tag": tag}
        )

        return result

    def compute_pole_loss(
        self,
        predictions: List[TRNode],
        Q_values: Optional[List[float]] = None,
        teacher_labels: Optional[List[float]] = None,
        x_samples: Optional[List[float]] = None,
    ) -> TRNode:
        """
        Compute combined pole detection and regularization loss.

        Args:
            predictions: Pole detection scores
            Q_values: Actual |Q(x)| values
            teacher_labels: Optional teacher signals
            x_samples: Input samples for regularization

        Returns:
            Combined loss
        """
        # Pole detection loss
        detection_loss = self.pole_head.compute_loss(predictions, Q_values, teacher_labels)

        # Regularization loss if enabled
        reg_loss = TRNode.constant(real(0.0))
        if self.regularizer and x_samples:
            # Create Q function wrapper
            def Q_func(x_node):
                # Evaluate Q at x
                basis = self.rational_layer.basis
                max_degree = self.rational_layer.d_q
                psi = basis(x_node, max_degree)

                # Compute Q(x)
                terms: List[TRNode] = [TRNode.constant(real(1.0))]
                for k in range(1, self.rational_layer.d_q + 1):
                    if k < len(psi) and k <= len(self.rational_layer.phi):
                        terms.append(tr_mul(self.rational_layer.phi[k - 1], psi[k]))

                # Optional deterministic pairwise reduction
                def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
                    if not nodes:
                        return TRNode.constant(real(0.0))
                    if len(nodes) == 1:
                        return nodes[0]
                    mid = len(nodes) // 2
                    left = _pairwise_sum(nodes[:mid])
                    right = _pairwise_sum(nodes[mid:])
                    return tr_add(left, right)

                use_pairwise = False
                try:
                    pol = TRPolicyConfig.get_policy()
                    use_pairwise = bool(pol and pol.deterministic_reduction)
                except Exception:
                    use_pairwise = False
                if use_pairwise:
                    return _pairwise_sum(terms)
                else:
                    acc = terms[0]
                    for t in terms[1:]:
                        acc = tr_add(acc, t)
                    return acc

            reg_loss = self.regularizer.compute_regularization(Q_func, x_samples)

        # Add weight decay regularization
        weight_decay_loss = self.pole_head.regularization_loss()

        # Combine losses
        total_loss = detection_loss
        total_loss = tr_add(total_loss, reg_loss)
        total_loss = tr_add(total_loss, weight_decay_loss)

        return total_loss

    def get_parameters(self) -> List[TRNode]:
        """Get all parameters including pole head."""
        params = []

        # Rational layer parameters
        if hasattr(self.rational_layer, "parameters"):
            params.extend(self.rational_layer.parameters())
        else:
            # Manual extraction
            params.extend(self.rational_layer.theta)
            params.extend(self.rational_layer.phi)

        # Pole head parameters
        params.extend(self.pole_head.parameters())

        return params

    def evaluate_accuracy(self, test_data: List[Tuple[float, bool]]) -> Dict[str, float]:
        """
        Evaluate pole detection accuracy.

        Args:
            test_data: List of (x_value, is_pole) tuples

        Returns:
            Dictionary of metrics
        """
        predictions = []
        labels = []

        for x_val, is_pole in test_data:
            prob = self.pole_head.predict_pole_probability(TRNode.constant(real(x_val)))
            predictions.append(prob)
            labels.append(1.0 if is_pole else 0.0)

        # Compute metrics
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Binary predictions with threshold 0.5
        binary_preds = (predictions > 0.5).astype(float)

        # Accuracy
        accuracy = np.mean(binary_preds == labels)

        # Precision and recall
        true_positives = np.sum((binary_preds == 1) & (labels == 1))
        false_positives = np.sum((binary_preds == 1) & (labels == 0))
        false_negatives = np.sum((binary_preds == 0) & (labels == 1))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
