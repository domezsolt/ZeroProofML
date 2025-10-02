"""
Tag-aware rational layer with auxiliary tag prediction.

This module extends the rational layer with tag classification capabilities,
enabling non-REAL outputs to contribute supervision through auxiliary losses.
"""

from typing import List, Optional, Tuple, Union

from ..autodiff import TRNode
from ..autodiff.hybrid_gradient import HybridGradientSchedule
from ..core import TRScalar, TRTag, real
from ..training.tag_loss import TagClass, TagPredictionHead
from .basis import Basis
from .hybrid_rational import HybridTRRational


class TagAwareRational(HybridTRRational):
    """
    Rational layer with tag prediction capability.

    This layer includes an auxiliary head that predicts the output tag
    (REAL/PINF/NINF/PHI), enabling supervision from non-REAL samples.
    """

    def __init__(
        self,
        d_p: int,
        d_q: int,
        basis: Optional[Basis] = None,
        shared_Q: bool = False,
        lambda_rej: float = 0.0,
        alpha_phi: float = 1e-3,
        l1_projection: Optional[float] = None,
        adaptive_loss_policy=None,
        projection_index: Optional[int] = None,
        hybrid_schedule: Optional[HybridGradientSchedule] = None,
        track_Q_values: bool = False,
        enable_tag_head: bool = True,
        tag_head_hidden_dim: int = 8,
    ):
        """
        Initialize tag-aware rational layer.

        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q
            basis: Basis functions to use
            shared_Q: If True, share Q across multiple outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominator
            l1_projection: Optional L1 bound for φ
            adaptive_loss_policy: Optional adaptive loss policy
            projection_index: Index for input projection
            hybrid_schedule: Hybrid gradient schedule
            track_Q_values: Whether to track Q values
            enable_tag_head: Whether to enable tag prediction head
            tag_head_hidden_dim: Hidden dimension for tag head
        """
        super().__init__(
            d_p,
            d_q,
            basis,
            shared_Q,
            lambda_rej,
            alpha_phi,
            l1_projection,
            adaptive_loss_policy,
            projection_index,
            hybrid_schedule,
            track_Q_values,
        )

        self.enable_tag_head = enable_tag_head

        # Initialize tag prediction head if enabled
        if enable_tag_head:
            # Use same basis degree as denominator
            self.tag_head = TagPredictionHead(
                input_dim=d_q, hidden_dim=tag_head_hidden_dim, basis=basis
            )
        else:
            self.tag_head = None

        # Track tag prediction statistics
        self.tag_prediction_history = []
        self.tag_accuracy_history = []

    def forward_with_tag_pred(
        self, x: Union[TRScalar, TRNode]
    ) -> Tuple[TRNode, TRTag, Optional[List[TRNode]]]:
        """
        Forward pass with tag prediction.

        Args:
            x: Input value

        Returns:
            Tuple of (output_node, output_tag, tag_logits)
        """
        # Standard rational forward pass
        y, tag = self.forward(x)

        # Predict tag if head is enabled
        tag_logits = None
        if self.enable_tag_head and self.tag_head is not None:
            # Ensure x is a node
            if isinstance(x, TRScalar):
                x = TRNode.constant(x)
            elif not isinstance(x, TRNode):
                x = TRNode.constant(real(float(x)))

            tag_logits = self.tag_head.forward(x)

            # Track prediction if in evaluation mode
            if not x.requires_grad:
                pred_class, probs = self.tag_head.predict_tag(x)
                true_class = TagClass.from_tag(tag)

                self.tag_prediction_history.append(
                    {"true": true_class, "pred": pred_class, "probs": probs}
                )

        return y, tag, tag_logits

    def parameters(self) -> List[TRNode]:
        """Get all parameters including tag head."""
        params = super().parameters()

        if self.enable_tag_head and self.tag_head is not None:
            params.extend(self.tag_head.parameters())

        return params

    def num_parameters(self) -> int:
        """Total number of parameters including tag head when enabled."""
        base = super().num_parameters()
        if self.enable_tag_head and self.tag_head is not None:
            return base + len(self.tag_head.parameters())
        return base

    def get_tag_statistics(self) -> dict:
        """Get tag prediction statistics."""
        stats = {}

        if not self.tag_prediction_history:
            return stats

        # Compute accuracy
        correct = sum(1 for h in self.tag_prediction_history if h["true"] == h["pred"])
        total = len(self.tag_prediction_history)
        stats["tag_accuracy"] = correct / total if total > 0 else 0.0

        # Compute per-class accuracy
        for cls in TagClass:
            class_preds = [h for h in self.tag_prediction_history if h["true"] == cls]
            if class_preds:
                class_correct = sum(1 for h in class_preds if h["pred"] == cls)
                stats[f"accuracy_{cls.name}"] = class_correct / len(class_preds)

        # Add confusion matrix
        from ..training.tag_loss import compute_tag_confusion_matrix

        predictions = [TRNode.constant(real(0.0)) for _ in self.tag_prediction_history]
        for i, h in enumerate(self.tag_prediction_history):
            # Set tag based on true class
            if h["true"] == TagClass.PINF:
                predictions[i]._value = real(float("inf"))
                predictions[i]._value._tag = TRTag.PINF
            elif h["true"] == TagClass.NINF:
                predictions[i]._value = real(float("-inf"))
                predictions[i]._value._tag = TRTag.NINF
            elif h["true"] == TagClass.PHI:
                predictions[i]._value = real(0.0)
                predictions[i]._value._tag = TRTag.PHI

        pred_classes = [h["pred"] for h in self.tag_prediction_history]
        stats["confusion_matrix"] = compute_tag_confusion_matrix(predictions, pred_classes)

        return stats

    def reset_tag_statistics(self):
        """Reset tag prediction statistics."""
        self.tag_prediction_history.clear()
        self.tag_accuracy_history.clear()


class TagAwareMultiRational:
    """
    Multi-output rational layer with shared tag prediction.

    All outputs share a single tag prediction head for efficiency.
    """

    def __init__(
        self,
        d_p: int,
        d_q: int,
        n_outputs: int,
        basis: Optional[Basis] = None,
        shared_Q: bool = True,
        lambda_rej: float = 0.0,
        alpha_phi: float = 1e-3,
        hybrid_schedule: Optional[HybridGradientSchedule] = None,
        enable_tag_head: bool = True,
        tag_head_hidden_dim: int = 8,
    ):
        """
        Initialize multi-output tag-aware rational layer.

        Args:
            d_p: Degree of numerator polynomials
            d_q: Degree of denominator polynomial(s)
            n_outputs: Number of outputs
            basis: Basis functions to use
            shared_Q: If True, share denominator across outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominators
            hybrid_schedule: Hybrid gradient schedule
            enable_tag_head: Whether to enable tag prediction
            tag_head_hidden_dim: Hidden dimension for tag head
        """
        self.n_outputs = n_outputs
        self.shared_Q = shared_Q
        self.enable_tag_head = enable_tag_head

        # Create individual layers
        self.layers = []
        for i in range(n_outputs):
            layer = TagAwareRational(
                d_p,
                d_q,
                basis,
                shared_Q,
                lambda_rej,
                alpha_phi,
                hybrid_schedule=hybrid_schedule,
                enable_tag_head=False,  # We'll use a shared head
            )
            self.layers.append(layer)

        # Share denominator if requested
        if shared_Q and n_outputs > 1:
            shared_phi = self.layers[0].phi
            for layer in self.layers[1:]:
                layer.phi = shared_phi

        # Create shared tag head if enabled
        if enable_tag_head:
            self.tag_head = TagPredictionHead(
                input_dim=max(d_p, d_q), hidden_dim=tag_head_hidden_dim, basis=basis
            )
        else:
            self.tag_head = None

    def forward(self, x: Union[TRScalar, TRNode]) -> List[Tuple[TRNode, TRTag]]:
        """
        Forward pass for all outputs.

        Args:
            x: Input value

        Returns:
            List of (output_node, output_tag) tuples
        """
        return [layer.forward(x) for layer in self.layers]

    def forward_with_tag_pred(
        self, x: Union[TRScalar, TRNode]
    ) -> Tuple[List[Tuple[TRNode, TRTag]], Optional[List[TRNode]]]:
        """
        Forward pass with shared tag prediction.

        Args:
            x: Input value

        Returns:
            Tuple of (outputs, tag_logits)
        """
        # Get all outputs
        outputs = self.forward(x)

        # Get tag prediction if enabled
        tag_logits = None
        if self.enable_tag_head and self.tag_head is not None:
            if isinstance(x, TRScalar):
                x = TRNode.constant(x)
            elif not isinstance(x, TRNode):
                x = TRNode.constant(real(float(x)))

            tag_logits = self.tag_head.forward(x)

        return outputs, tag_logits

    def parameters(self) -> List[TRNode]:
        """Get all parameters including shared tag head."""
        params = []

        # Collect unique parameters from layers
        seen = set()
        for layer in self.layers:
            for param in layer.parameters():
                param_id = id(param)
                if param_id not in seen:
                    params.append(param)
                    seen.add(param_id)

        # Add tag head parameters
        if self.enable_tag_head and self.tag_head is not None:
            params.extend(self.tag_head.parameters())

        return params

    def regularization_loss(self) -> TRNode:
        """Compute total regularization loss."""
        if self.shared_Q:
            # Only regularize once for shared denominator
            return self.layers[0].regularization_loss()
        else:
            # Sum regularization across all layers
            from ..core import tr_add

            total_reg = TRNode.constant(real(0.0))
            for layer in self.layers:
                total_reg = tr_add(total_reg, layer.regularization_loss())
            return total_reg
