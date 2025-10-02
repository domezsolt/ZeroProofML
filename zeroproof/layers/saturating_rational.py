"""
TR-Rational layer with saturating gradient support.

This module extends the TR-Rational layer to support both Mask-REAL
and saturating gradient modes for research and ablation studies.
"""

from typing import List, Optional, Tuple, Union

from ..autodiff import TRNode, gradient_tape
from ..autodiff.grad_mode import GradientMode, gradient_mode
from ..core import TRScalar, TRTag, real
from .basis import Basis
from .tr_rational import TRRational


class SaturatingTRRational(TRRational):
    """
    TR-Rational layer with selectable gradient mode.

    This extends the standard TR-Rational layer to support both:
    - Mask-REAL: Zero gradients for non-REAL outputs (default)
    - Saturating: Bounded gradients near singularities
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
        gradient_mode: GradientMode = GradientMode.MASK_REAL,
        saturation_bound: float = 1.0,
    ):
        """
        Initialize saturating TR-Rational layer.

        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q (must be ≥ 1)
            basis: Basis functions to use (default: MonomialBasis)
            shared_Q: If True, share Q across multiple outputs
            lambda_rej: Penalty for non-REAL outputs in loss
            alpha_phi: L2 regularization coefficient for φ (denominator)
            l1_projection: Optional L1 bound for φ to ensure stability
            adaptive_loss_policy: Optional adaptive loss policy
            gradient_mode: Gradient computation mode (MASK_REAL or SATURATING)
            saturation_bound: Bound for saturating gradients
        """
        super().__init__(
            d_p, d_q, basis, shared_Q, lambda_rej, alpha_phi, l1_projection, adaptive_loss_policy
        )

        self.gradient_mode = gradient_mode
        self.saturation_bound = saturation_bound

    def forward(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag]:
        """
        Forward pass with gradient mode context.

        Args:
            x: Input value (scalar)

        Returns:
            Tuple of (output_node, output_tag)
        """
        # Use gradient mode context for this forward pass
        with gradient_mode(self.gradient_mode, self.saturation_bound):
            return super().forward(x)

    def forward_with_mode(
        self,
        x: Union[TRScalar, TRNode],
        mode: Optional[GradientMode] = None,
        bound: Optional[float] = None,
    ) -> Tuple[TRNode, TRTag]:
        """
        Forward pass with temporary gradient mode override.

        Args:
            x: Input value
            mode: Gradient mode to use (None uses layer's default)
            bound: Saturation bound (None uses layer's default)

        Returns:
            Tuple of (output_node, output_tag)
        """
        if mode is None:
            mode = self.gradient_mode
        if bound is None:
            bound = self.saturation_bound

        with gradient_mode(mode, bound):
            return super().forward(x)

    def compare_gradient_modes(self, x_batch: List[Union[TRScalar, TRNode]]) -> dict:
        """
        Compare gradients between Mask-REAL and saturating modes.

        Useful for ablation studies and understanding the effect
        of different gradient computation strategies.

        Args:
            x_batch: Batch of input values

        Returns:
            Dictionary with gradient statistics for each mode
        """
        results = {
            "mask_real": {"gradients": [], "tags": []},
            "saturating": {"gradients": [], "tags": []},
        }

        # Store original parameters
        theta_orig = [p.value for p in self.theta]
        phi_orig = [p.value for p in self.phi]

        # Test Mask-REAL mode
        with gradient_mode(GradientMode.MASK_REAL):
            for x in x_batch:
                # Reset parameters
                for p, v in zip(self.theta, theta_orig):
                    p._value = v
                for p, v in zip(self.phi, phi_orig):
                    p._value = v

                # Zero gradients
                for p in self.parameters():
                    p.zero_grad()

                # Forward pass
                y, tag = self.forward_with_mode(x, GradientMode.MASK_REAL)
                results["mask_real"]["tags"].append(tag)

                # Backward pass
                y.backward()

                # Collect gradients
                grads = []
                for p in self.parameters():
                    if p.gradient is not None:
                        grads.append(
                            p.gradient.value if p.gradient.tag == TRTag.REAL else float("nan")
                        )
                    else:
                        grads.append(0.0)
                results["mask_real"]["gradients"].append(grads)

        # Test saturating mode
        with gradient_mode(GradientMode.SATURATING, self.saturation_bound):
            for x in x_batch:
                # Reset parameters
                for p, v in zip(self.theta, theta_orig):
                    p._value = v
                for p, v in zip(self.phi, phi_orig):
                    p._value = v

                # Zero gradients
                for p in self.parameters():
                    p.zero_grad()

                # Forward pass
                y, tag = self.forward_with_mode(x, GradientMode.SATURATING)
                results["saturating"]["tags"].append(tag)

                # Backward pass
                y.backward()

                # Collect gradients
                grads = []
                for p in self.parameters():
                    if p.gradient is not None:
                        grads.append(
                            p.gradient.value if p.gradient.tag == TRTag.REAL else float("nan")
                        )
                    else:
                        grads.append(0.0)
                results["saturating"]["gradients"].append(grads)

        # Restore original mode
        for p, v in zip(self.theta, theta_orig):
            p._value = v
        for p, v in zip(self.phi, phi_orig):
            p._value = v

        return results


def create_saturating_rational(
    d_p: int, d_q: int, mode: str = "mask-real", saturation_bound: float = 1.0, **kwargs
) -> SaturatingTRRational:
    """
    Create a TR-Rational layer with specified gradient mode.

    Args:
        d_p: Numerator degree
        d_q: Denominator degree
        mode: "mask-real" or "saturating"
        saturation_bound: Bound for saturating mode
        **kwargs: Additional arguments for TRRational

    Returns:
        Configured SaturatingTRRational layer
    """
    if mode == "mask-real":
        grad_mode = GradientMode.MASK_REAL
    elif mode == "saturating":
        grad_mode = GradientMode.SATURATING
    else:
        raise ValueError(f"Unknown gradient mode: {mode}")

    return SaturatingTRRational(
        d_p, d_q, gradient_mode=grad_mode, saturation_bound=saturation_bound, **kwargs
    )
