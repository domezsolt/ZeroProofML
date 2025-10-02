"""
TR-Rational layer with hybrid gradient support.

This module extends the TR-Rational layer to support the hybrid gradient
schedule, enabling sophisticated near-pole learning strategies.
"""

from typing import Optional, Tuple, Union

from ..autodiff import TRNode
from ..autodiff.grad_mode import GradientMode, gradient_mode
from ..autodiff.hybrid_gradient import HybridGradientContext, HybridGradientSchedule
from ..core import TRScalar, TRTag, real
from .basis import Basis
from .tr_rational import TRRational


class HybridTRRational(TRRational):
    """
    TR-Rational layer with hybrid gradient schedule support.

    This layer automatically detects when it's operating near poles
    and can switch between Mask-REAL and Saturating gradients based
    on the hybrid schedule configuration.

    Attributes:
        hybrid_schedule: Optional hybrid gradient schedule
        track_Q_values: Whether to track Q values for analysis
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
    ):
        """
        Initialize hybrid TR-Rational layer.

        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q (must be ≥ 1)
            basis: Basis functions to use (default: MonomialBasis)
            shared_Q: If True, share Q across multiple outputs
            lambda_rej: Penalty for non-REAL outputs in loss
            alpha_phi: L2 regularization coefficient for φ (denominator)
            l1_projection: Optional L1 bound for φ to ensure stability
            adaptive_loss_policy: Optional adaptive loss policy
            projection_index: Index for input projection if needed
            hybrid_schedule: Hybrid gradient schedule configuration
            track_Q_values: Whether to track Q values for analysis
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
        )

        self.hybrid_schedule = hybrid_schedule
        self.track_Q_values = track_Q_values

        # Statistics tracking
        self.q_min_history = []
        self.q_max_history = []
        self.near_pole_counts = []

        # Register with global context if schedule provided
        if hybrid_schedule is not None:
            HybridGradientContext.set_schedule(hybrid_schedule)

    def forward(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag]:
        """
        Forward pass with Q-value tracking for hybrid gradients.

        Args:
            x: Input value (scalar)

        Returns:
            Tuple of (output_node, output_tag)
        """
        # Ensure we're using hybrid mode if schedule is active
        if self.hybrid_schedule is not None and self.hybrid_schedule.enable:
            # Temporarily switch to hybrid mode for this forward pass
            from ..autodiff.grad_mode import GradientModeConfig

            old_mode = GradientModeConfig.get_mode()
            GradientModeConfig.set_mode(GradientMode.HYBRID)

            try:
                # Perform standard forward computation
                result = super().forward(x)

                # Track Q value if enabled
                if self.track_Q_values and hasattr(self, "_last_Q_value"):
                    Q_val = getattr(self, "_last_Q_value", None)
                    if Q_val is not None and Q_val.tag == TRTag.REAL:
                        abs_Q = abs(Q_val.value)

                        # Update min/max tracking
                        if not self.q_min_history or abs_Q < min(self.q_min_history):
                            self.q_min_history.append(abs_Q)
                        if not self.q_max_history or abs_Q > max(self.q_max_history):
                            self.q_max_history.append(abs_Q)

                        # Check if near pole
                        threshold = HybridGradientContext._local_threshold
                        if threshold is not None and abs_Q <= threshold:
                            self.near_pole_counts.append(1)
                        else:
                            self.near_pole_counts.append(0)

                return result

            finally:
                # Restore previous mode
                GradientModeConfig.set_mode(old_mode)
        else:
            # No hybrid schedule, use standard forward
            return super().forward(x)

    def update_epoch(self, epoch: int) -> None:
        """
        Update the hybrid schedule for a new epoch.

        Args:
            epoch: Current training epoch
        """
        if self.hybrid_schedule is not None:
            HybridGradientContext.update_epoch(epoch)

            # Reset per-epoch statistics
            HybridGradientContext.reset_statistics()
            self.near_pole_counts.clear()

    def get_hybrid_statistics(self) -> dict:
        """
        Get statistics about hybrid gradient usage.

        Returns:
            Dictionary with statistics
        """
        stats = {}

        # Get global context statistics
        if self.hybrid_schedule is not None:
            stats.update(HybridGradientContext.get_statistics())

        # Add layer-specific statistics
        if self.q_min_history:
            stats["q_min"] = min(self.q_min_history)
            stats["q_max"] = max(self.q_max_history)
            stats["q_min_current"] = self.q_min_history[-1] if self.q_min_history else None

        if self.near_pole_counts:
            total = len(self.near_pole_counts)
            near_pole = sum(self.near_pole_counts)
            stats["near_pole_ratio_layer"] = near_pole / total if total > 0 else 0.0

        # Add schedule description
        if self.hybrid_schedule is not None:
            epoch = HybridGradientContext._current_epoch
            stats["mode_description"] = self.hybrid_schedule.get_mode_description(epoch)

        return stats

    def reset_statistics(self) -> None:
        """Reset all tracking statistics."""
        self.q_min_history.clear()
        self.q_max_history.clear()
        self.near_pole_counts.clear()
        if self.hybrid_schedule is not None:
            HybridGradientContext.reset_statistics()


class HybridRationalWithPoleHead(HybridTRRational):
    """
    Hybrid rational layer with auxiliary pole detection head.

    This extends the hybrid rational layer with an additional
    network that predicts where Q(x) ≈ 0, enabling explicit
    pole localization learning.
    """

    def __init__(
        self,
        d_p: int,
        d_q: int,
        d_pole: int = 3,
        basis: Optional[Basis] = None,
        shared_Q: bool = False,
        lambda_rej: float = 0.0,
        alpha_phi: float = 1e-3,
        l1_projection: Optional[float] = None,
        adaptive_loss_policy=None,
        projection_index: Optional[int] = None,
        hybrid_schedule: Optional[HybridGradientSchedule] = None,
        track_Q_values: bool = True,
        lambda_pole: float = 0.1,
    ):
        """
        Initialize hybrid rational with pole head.

        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q
            d_pole: Degree of pole detection polynomial
            basis: Basis functions to use
            shared_Q: If True, share Q across multiple outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominator
            l1_projection: Optional L1 bound for φ
            adaptive_loss_policy: Optional adaptive loss policy
            projection_index: Index for input projection
            hybrid_schedule: Hybrid gradient schedule
            track_Q_values: Whether to track Q values
            lambda_pole: Weight for pole detection loss
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

        self.d_pole = d_pole
        self.lambda_pole = lambda_pole

        # Initialize pole detection parameters
        self._initialize_pole_head()

    def _initialize_pole_head(self):
        """Initialize parameters for pole detection head."""
        import math

        self.pole_params = []
        for i in range(self.d_pole + 1):
            # Small random initialization
            val = (2 * (i % 2) - 1) * math.sqrt(2.0 / (self.d_pole + 1)) * 0.1
            self.pole_params.append(TRNode.parameter(real(val), name=f"pole_{i}"))

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
        if isinstance(x, TRScalar):
            x = TRNode.constant(x)
        elif not isinstance(x, TRNode):
            x = TRNode.constant(real(float(x)))

        # Evaluate basis for pole detection
        psi = self.basis(x, self.d_pole)

        # Compute pole score: Σ pole_k * ψ_k(x)
        pole_score = self.pole_params[0] * psi[0]
        for k in range(1, min(self.d_pole + 1, len(psi))):
            pole_score = pole_score + self.pole_params[k] * psi[k]

        return y, tag, pole_score

    def pole_parameters(self) -> list:
        """Get pole detection head parameters."""
        return self.pole_params

    def all_parameters(self) -> list:
        """Get all parameters including pole head."""
        params = self.parameters()  # Base rational parameters
        params.extend(self.pole_params)
        return params
