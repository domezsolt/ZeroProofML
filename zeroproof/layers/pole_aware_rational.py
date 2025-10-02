"""
Pole-aware rational layer with Q≈0 detection.

This module extends the rational layer with a pole detection head that
explicitly learns to predict where the denominator Q approaches zero.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..autodiff import TRNode
from ..autodiff.hybrid_gradient import HybridGradientSchedule
from ..core import TRScalar, TRTag, real
from ..policy import TRPolicyConfig, classify_tag_with_policy
from ..training.pole_detection import PoleDetectionConfig, PoleDetectionHead, compute_pole_metrics
from .basis import Basis
from .hybrid_rational import HybridTRRational
from .tag_aware_rational import TagAwareRational


class PoleAwareRational(HybridTRRational):
    """
    Rational layer with pole detection capability.

    This layer includes an auxiliary head that predicts where Q(x) ≈ 0,
    enabling explicit pole localization learning with optional teacher signals.
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
        track_Q_values: bool = True,
        enable_pole_head: bool = True,
        pole_config: Optional[PoleDetectionConfig] = None,
    ):
        """
        Initialize pole-aware rational layer.

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
            enable_pole_head: Whether to enable pole detection head
            pole_config: Configuration for pole detection
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

        self.enable_pole_head = enable_pole_head

        # Initialize pole detection head if enabled
        if enable_pole_head:
            pole_config = pole_config or PoleDetectionConfig()
            self.pole_head = PoleDetectionHead(
                input_dim=max(d_p, d_q), config=pole_config, basis=basis
            )
        else:
            self.pole_head = None

        # Track last Q value for self-supervision
        self._last_Q_value = None
        self._last_Q_abs = None

        # Tracking
        self.pole_predictions = []
        self.true_poles = []
        self.pole_metrics_history = []

    def forward(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag]:
        """
        Forward pass with Q-value tracking for pole detection.

        Args:
            x: Input value

        Returns:
            Tuple of (output_node, output_tag)
        """
        # Ensure x is a node
        if isinstance(x, TRScalar):
            x = TRNode.constant(x)
        elif not isinstance(x, TRNode):
            x = TRNode.constant(real(float(x)))

        # Evaluate basis functions
        psi = self.basis(x, max(self.d_p, self.d_q))

        # Compute P(x) with optional deterministic pairwise reduction
        def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
            if not nodes:
                return TRNode.constant(real(0.0))
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = _pairwise_sum(nodes[:mid])
            right = _pairwise_sum(nodes[mid:])
            return left + right

        P_terms: List[TRNode] = []
        for k in range(0, self.d_p + 1):
            if k < len(psi):
                P_terms.append(self.theta[k] * psi[k])
        use_pairwise = False
        try:
            from ..policy import TRPolicyConfig

            pol = TRPolicyConfig.get_policy()
            use_pairwise = bool(pol and pol.deterministic_reduction)
        except Exception:
            use_pairwise = False
        if use_pairwise:
            P = _pairwise_sum(P_terms)
        else:
            P = P_terms[0]
            for term in P_terms[1:]:
                P = P + term

        # Compute Q(x) with optional deterministic pairwise reduction
        Q_terms: List[TRNode] = [TRNode.constant(real(1.0))]
        for k in range(1, self.d_q + 1):
            if k < len(psi) and k <= len(self.phi):
                Q_terms.append(self.phi[k - 1] * psi[k])
        if use_pairwise:
            Q = _pairwise_sum(Q_terms)
        else:
            Q = Q_terms[0]
            for term in Q_terms[1:]:
                Q = Q + term

        # Store Q value for pole detection
        self._last_Q_value = Q.value
        if Q.value.tag == TRTag.REAL:
            self._last_Q_abs = abs(Q.value.value)
        else:
            self._last_Q_abs = float("inf")

        # Update hybrid controller Q-tracking for quantile stats
        try:
            if Q.tag == TRTag.REAL:
                from ..autodiff.hybrid_gradient import HybridGradientContext

                HybridGradientContext.update_q_value(abs(Q.value.value))
        except Exception:
            pass

        # Compute rational function
        y = P / Q

        # Determine tag (policy-aware if configured)
        tag = y.tag
        try:
            policy = TRPolicyConfig.get_policy()
            if policy is not None:
                prev = getattr(self, "_last_policy_tag", None)
                tag = classify_tag_with_policy(
                    policy, P.value, Q.value, y.tag, prev_policy_tag=prev
                )
                self._last_policy_tag = tag
            else:
                # Fallback: simple TR-based classification near exact pole
                if Q.value.tag == TRTag.REAL and Q.value.value != 0:
                    tag = TRTag.REAL
                elif Q.value.tag == TRTag.REAL and Q.value.value == 0:
                    if P.value.tag == TRTag.REAL and P.value.value != 0:
                        tag = TRTag.PINF if P.value.value > 0 else TRTag.NINF
                    else:
                        tag = TRTag.PHI
                else:
                    tag = y.tag
        except Exception:
            # Any error in policy classification: use TR tag
            tag = y.tag

        # Track Q values if enabled
        if self.track_Q_values and self._last_Q_abs is not None:
            if not hasattr(self, "q_min_history"):
                self.q_min_history = []
                self.q_max_history = []

            self.q_min_history.append(self._last_Q_abs)
            self.q_max_history.append(self._last_Q_abs)

        return y, tag

    def forward_with_pole_score(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag, TRNode]:
        """
        Forward pass with pole detection score.

        Args:
            x: Input value

        Returns:
            Tuple of (output_node, output_tag, pole_score)
        """
        # Standard forward pass
        y, tag = self.forward(x)

        # Compute pole detection score
        pole_score = None
        if self.enable_pole_head and self.pole_head is not None:
            if isinstance(x, TRScalar):
                x = TRNode.constant(x)
            elif not isinstance(x, TRNode):
                x = TRNode.constant(real(float(x)))

            pole_score = self.pole_head.forward(x)

            # Track prediction if not training
            if not x.requires_grad:
                prob = self.pole_head.predict_pole_probability(x)
                self.pole_predictions.append(prob)

        return y, tag, pole_score

    def get_Q_value(self) -> Optional[float]:
        """
        Get the last computed |Q| value.

        Returns:
            Absolute value of Q or None
        """
        return self._last_Q_abs

    def parameters(self) -> List[TRNode]:
        """Get all parameters including pole head."""
        params = super().parameters()

        if self.enable_pole_head and self.pole_head is not None:
            params.extend(self.pole_head.parameters())

        return params

    def evaluate_pole_detection(
        self, inputs: List[float], true_poles: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate pole detection accuracy.

        Args:
            inputs: Input values
            true_poles: True pole indicators

        Returns:
            Dictionary of metrics
        """
        if not self.enable_pole_head or self.pole_head is None:
            return {}

        # Get predictions
        pole_scores = []
        for x_val in inputs:
            x = TRNode.constant(real(x_val))
            score = self.pole_head.forward(x)
            pole_scores.append(score)

        # Compute metrics
        metrics = compute_pole_metrics(pole_scores, true_poles)

        # Store in history
        self.pole_metrics_history.append(metrics)

        return metrics

    def reset_pole_statistics(self):
        """Reset pole detection statistics."""
        self.pole_predictions.clear()
        self.true_poles.clear()
        self.pole_metrics_history.clear()


class FullyIntegratedRational(TagAwareRational):
    """
    Fully integrated rational layer with all enhancements.

    Combines:
    - Hybrid gradient schedule
    - Tag-loss for non-REAL outputs
    - Pole detection head
    - Coverage tracking
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
        track_Q_values: bool = True,
        enable_tag_head: bool = True,
        tag_head_hidden_dim: int = 8,
        enable_pole_head: bool = True,
        pole_config: Optional[PoleDetectionConfig] = None,
    ):
        """
        Initialize fully integrated rational layer.

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
            enable_tag_head: Whether to enable tag prediction
            tag_head_hidden_dim: Hidden dimension for tag head
            enable_pole_head: Whether to enable pole detection
            pole_config: Configuration for pole detection
        """
        # Initialize tag-aware layer
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
            enable_tag_head,
            tag_head_hidden_dim,
        )

        self.enable_pole_head = enable_pole_head

        # Add pole detection head
        if enable_pole_head:
            pole_config = pole_config or PoleDetectionConfig()
            self.pole_head = PoleDetectionHead(
                input_dim=max(d_p, d_q), config=pole_config, basis=basis
            )
        else:
            self.pole_head = None

        # Tracking for full integration
        self._last_Q_abs = None
        self.integration_metrics = {}
        self.pole_metrics_history = []

    def forward_fully_integrated(
        self, x: Union[TRScalar, TRNode]
    ) -> Dict[str, Union[TRNode, TRTag, float]]:
        """
        Forward pass with all enhancements.

        Args:
            x: Input value

        Returns:
            Dictionary with all outputs
        """
        # Ensure x is a node
        if isinstance(x, TRScalar):
            x = TRNode.constant(x)
        elif not isinstance(x, TRNode):
            x = TRNode.constant(real(float(x)))

        result = {}

        # Get standard output with tag prediction
        y, tag, tag_logits = self.forward_with_tag_pred(x)
        result["output"] = y
        result["tag"] = tag
        result["tag_logits"] = tag_logits

        # Get pole detection score
        if self.enable_pole_head and self.pole_head is not None:
            pole_score = self.pole_head.forward(x)
            pole_prob = self.pole_head.predict_pole_probability(x)
            result["pole_score"] = pole_score
            result["pole_probability"] = pole_prob

        # Track Q value
        if hasattr(self, "_last_Q_value"):
            result["Q_value"] = self._last_Q_value
            if self._last_Q_value and self._last_Q_value.tag == TRTag.REAL:
                result["Q_abs"] = abs(self._last_Q_value.value)

        return result

    def parameters(self) -> List[TRNode]:
        """Get all parameters from all heads."""
        params = super().parameters()  # Gets tag head params too

        if self.enable_pole_head and self.pole_head is not None:
            params.extend(self.pole_head.parameters())

        return params

    def get_integration_summary(self) -> Dict[str, Any]:
        """
        Get summary of all integrated features.

        Returns:
            Dictionary with feature statistics
        """
        summary: Dict[str, Any] = {}

        # Hybrid gradient status
        if self.hybrid_schedule:
            summary["hybrid_enabled"] = True
            summary["hybrid_mode"] = self.hybrid_schedule.get_mode_description(0)
        else:
            summary["hybrid_enabled"] = False

        # Tag prediction status
        if self.enable_tag_head:
            summary["tag_prediction_enabled"] = True
            tag_stats = self.get_tag_statistics()
            summary["tag_accuracy"] = tag_stats.get("tag_accuracy", 0.0)
        else:
            summary["tag_prediction_enabled"] = False

        # Pole detection status
        if self.enable_pole_head:
            summary["pole_detection_enabled"] = True
            if self.pole_metrics_history:
                latest = self.pole_metrics_history[-1]
                summary["pole_f1_score"] = latest.get("f1", 0.0)
        else:
            summary["pole_detection_enabled"] = False

        # Q value tracking
        if self.track_Q_values and hasattr(self, "q_min_history"):
            summary["q_min_observed"] = min(self.q_min_history) if self.q_min_history else None
            summary["q_max_observed"] = max(self.q_max_history) if self.q_max_history else None

        # Parameter count
        summary["total_parameters"] = len(self.parameters())

        return summary
