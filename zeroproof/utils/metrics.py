# MIT License
# See LICENSE file in the project root for full license text.
"""
Anti-illusion metrics for verifying pole geometry learning.

This module implements metrics that prove the model truly learns
pole behavior and asymptotics, not just avoids singularities.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..autodiff import TRNode
from ..core import TRScalar, TRTag, real, tr_add, tr_div, tr_mul, tr_sub


@dataclass
class PoleLocation:
    """Represents a pole location in 1D or 2D."""

    x: float
    y: Optional[float] = None  # For 2D poles
    pole_type: str = "simple"  # "simple", "double", "complex"
    residue: Optional[complex] = None  # For complex analysis

    def distance_to(self, other: "PoleLocation") -> float:
        """Compute distance to another pole."""
        if self.y is None and other.y is None:
            # 1D case
            return abs(self.x - other.x)
        elif self.y is not None and other.y is not None:
            # 2D case
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        else:
            raise ValueError("Cannot compute distance between 1D and 2D poles")


class PoleLocalizationError:
    """
    Pole Localization Error (PLE) metric.

    Measures the distance between learned pole locations
    and ground-truth poles using Chamfer or Hausdorff distance.
    """

    def __init__(self, distance_type: str = "chamfer"):
        """
        Initialize PLE metric.

        Args:
            distance_type: "chamfer" or "hausdorff"
        """
        self.distance_type = distance_type
        self.history = []

    def find_poles_1d(
        self, model, x_range: Tuple[float, float], n_samples: int = 1000, q_threshold: float = 5e-2
    ) -> List[PoleLocation]:
        """
        Find poles by sampling Q(x) on a 1D grid.

        Args:
            model: Model with Q function access
            x_range: (x_min, x_max) range to search
            n_samples: Number of grid points
            q_threshold: |Q| threshold for pole detection

        Returns:
            List of detected pole locations
        """
        x_min, x_max = x_range
        x_vals = np.linspace(x_min, x_max, n_samples)

        poles = []
        q_values = []

        # Evaluate |Q| on grid
        for x_val in x_vals:
            try:
                if hasattr(model, "get_Q_value"):
                    # Use cached Q value if available
                    x = TRNode.constant(real(x_val))
                    _ = model.forward(x)
                    q_abs = model.get_Q_value()
                else:
                    # Compute Q directly if model supports it
                    q_abs = self._evaluate_Q_magnitude(model, x_val)

                q_values.append(q_abs if q_abs is not None else float("inf"))
            except:
                q_values.append(float("inf"))

        # Find local minima and apply adaptive threshold if needed
        # Use a looser threshold when grid is coarse
        grid_step = (x_max - x_min) / max(1, (n_samples - 1))
        adaptive_threshold = max(q_threshold, 0.4 * grid_step)

        for i in range(1, len(q_values) - 1):
            if (
                q_values[i] < adaptive_threshold
                and q_values[i] < q_values[i - 1]
                and q_values[i] < q_values[i + 1]
            ):
                # Refine pole location using parabolic interpolation
                refined_x = self._refine_pole_location(
                    x_vals[i - 1 : i + 2], q_values[i - 1 : i + 2]
                )
                poles.append(PoleLocation(x=refined_x))

        # Fallback: if none found, pick global minima if sufficiently small
        if not poles and q_values:
            min_idx = int(np.argmin(q_values))
            if q_values[min_idx] < adaptive_threshold * 5.0:
                refined_x = self._refine_pole_location(
                    x_vals[max(0, min_idx - 1) : min(len(x_vals), min_idx + 2)],
                    q_values[max(0, min_idx - 1) : min(len(q_values), min_idx + 2)],
                )
                poles.append(PoleLocation(x=refined_x))

        return poles

    def _evaluate_Q_magnitude(self, model, x_val: float) -> Optional[float]:
        """Evaluate |Q(x)| for a given model and x value."""
        try:
            # This is model-specific - would need to be implemented
            # based on the actual model architecture
            if hasattr(model, "phi") and hasattr(model, "basis"):
                # For TR-Rational models
                x = TRNode.constant(real(x_val))
                psi = model.basis(x, len(model.phi) + 1)

                # Q(x) = 1 + sum(phi_k * psi_k)
                Q = real(1.0)
                for k, phi_k in enumerate(model.phi):
                    if k + 1 < len(psi):
                        Q = tr_add(Q, tr_mul(phi_k.value, psi[k + 1].value))

                if Q.tag == TRTag.REAL:
                    return abs(Q.value)

            return None
        except:
            return None

    def _refine_pole_location(self, x_vals: List[float], q_vals: List[float]) -> float:
        """Refine pole location using parabolic interpolation."""
        if len(x_vals) != 3 or len(q_vals) != 3:
            return x_vals[len(x_vals) // 2]  # Return middle point if can't interpolate

        x0, x1, x2 = x_vals
        y0, y1, y2 = q_vals

        # Assume nearly uniform spacing; use standard quadratic vertex formula
        try:
            h1 = x1 - x0
            h2 = x2 - x1
            if abs(h1 - h2) < 1e-9 and abs(y0 - 2 * y1 + y2) > 1e-12:
                h = (h1 + h2) * 0.5
                x_min = x1 + 0.5 * h * ((y0 - y2) / (y0 - 2 * y1 + y2))
                if x0 <= x_min <= x2:
                    return x_min
            return x1
        except Exception:
            return x1

    def compute_chamfer_distance(
        self, predicted: List[PoleLocation], ground_truth: List[PoleLocation]
    ) -> float:
        """
        Compute Chamfer distance between pole sets.

        Args:
            predicted: Predicted pole locations
            ground_truth: Ground truth pole locations

        Returns:
            Chamfer distance
        """
        if not predicted or not ground_truth:
            return float("inf")

        # Sum of minimum distances (symmetric) then average over both sets once
        pred_to_gt_sum = 0.0
        for pred_pole in predicted:
            pred_to_gt_sum += min(pred_pole.distance_to(gt_pole) for gt_pole in ground_truth)
        gt_to_pred_sum = 0.0
        for gt_pole in ground_truth:
            gt_to_pred_sum += min(gt_pole.distance_to(pred_pole) for pred_pole in predicted)
        # Divide by 2 to match test convention
        return (pred_to_gt_sum + gt_to_pred_sum) / 2.0

    def compute_hausdorff_distance(
        self, predicted: List[PoleLocation], ground_truth: List[PoleLocation]
    ) -> float:
        """
        Compute Hausdorff distance between pole sets.

        Args:
            predicted: Predicted pole locations
            ground_truth: Ground truth pole locations

        Returns:
            Hausdorff distance
        """
        if not predicted or not ground_truth:
            return float("inf")

        # Max distance from predicted to ground truth
        pred_to_gt = max(
            min(pred_pole.distance_to(gt_pole) for gt_pole in ground_truth)
            for pred_pole in predicted
        )

        # Max distance from ground truth to predicted
        gt_to_pred = max(
            min(gt_pole.distance_to(pred_pole) for pred_pole in predicted)
            for gt_pole in ground_truth
        )

        return max(pred_to_gt, gt_to_pred)

    def compute_ple(
        self,
        model,
        ground_truth_poles: List[PoleLocation],
        x_range: Tuple[float, float] = (-2, 2),
        n_samples: int = 1000,
    ) -> float:
        """
        Compute Pole Localization Error.

        Args:
            model: Model to evaluate
            ground_truth_poles: Known pole locations
            x_range: Range to search for poles
            n_samples: Grid resolution

        Returns:
            PLE score (lower is better)
        """
        predicted_poles = self.find_poles_1d(model, x_range, n_samples)
        # If no poles found, return a large finite penalty instead of inf
        if not predicted_poles:
            self.history.append(1e9)
            return 1e9

        if self.distance_type == "chamfer":
            ple = self.compute_chamfer_distance(predicted_poles, ground_truth_poles)
        else:
            ple = self.compute_hausdorff_distance(predicted_poles, ground_truth_poles)

        self.history.append(ple)
        return ple


class SignConsistencyChecker:
    """
    Checks sign consistency when crossing poles.

    Verifies that the model correctly flips between +∞ and -∞
    when crossing a simple pole along a path.
    """

    def __init__(self):
        self.history = []

    def check_path_crossing(
        self,
        model,
        path_func: Callable[[float], float],
        t_range: Tuple[float, float],
        pole_t: float,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Check sign consistency along a parametric path crossing a pole.

        Args:
            model: Model to evaluate
            path_func: Function t -> x giving path parameter
            t_range: (t_min, t_max) parameter range
            pole_t: Parameter value where pole is crossed
            n_samples: Number of samples along path

        Returns:
            Dictionary with consistency metrics
        """
        t_min, t_max = t_range
        t_vals = np.linspace(t_min, t_max, n_samples)

        # Split into before and after pole
        before_pole = [t for t in t_vals if t < pole_t - 1e-6]
        after_pole = [t for t in t_vals if t > pole_t + 1e-6]

        before_signs = []
        after_signs = []

        # Evaluate signs before pole
        for t in before_pole[-10:]:  # Last 10 points before pole
            x_val = path_func(t)
            x = TRNode.constant(real(x_val))
            y, tag = model.forward(x)

            if tag == TRTag.PINF:
                before_signs.append(1)
            elif tag == TRTag.NINF:
                before_signs.append(-1)
            elif tag == TRTag.REAL and y.value.value != 0:
                before_signs.append(1 if y.value.value > 0 else -1)

        # Evaluate signs after pole
        for t in after_pole[:10]:  # First 10 points after pole
            x_val = path_func(t)
            x = TRNode.constant(real(x_val))
            y, tag = model.forward(x)

            if tag == TRTag.PINF:
                after_signs.append(1)
            elif tag == TRTag.NINF:
                after_signs.append(-1)
            elif tag == TRTag.REAL and y.value.value != 0:
                after_signs.append(1 if y.value.value > 0 else -1)

        # Compute consistency metrics
        metrics = {}

        if before_signs and after_signs:
            # Check if signs are consistently different
            before_sign = max(set(before_signs), key=before_signs.count)
            after_sign = max(set(after_signs), key=after_signs.count)

            metrics["sign_flip_correct"] = float(before_sign != after_sign)
            metrics["before_sign_consistency"] = before_signs.count(before_sign) / len(before_signs)
            metrics["after_sign_consistency"] = after_signs.count(after_sign) / len(after_signs)
            metrics["overall_consistency"] = (
                metrics["sign_flip_correct"]
                * metrics["before_sign_consistency"]
                * metrics["after_sign_consistency"]
            )
        else:
            metrics = {
                "sign_flip_correct": 0.0,
                "before_sign_consistency": 0.0,
                "after_sign_consistency": 0.0,
                "overall_consistency": 0.0,
            }

        self.history.append(metrics)
        return metrics


class AsymptoticSlopeAnalyzer:
    """
    Analyzes asymptotic slope near poles.

    For simple poles, expects log|y| ~ -log|Q| behavior,
    i.e., slope ≈ -1 in log-log plot.
    """

    def __init__(self):
        self.history = []

    def compute_asymptotic_slope(
        self, model, pole_location: float, window_size: float = 0.1, n_samples: int = 50
    ) -> Dict[str, float]:
        """
        Compute asymptotic slope near a pole.

        Args:
            model: Model to evaluate
            pole_location: x-coordinate of pole
            window_size: Size of window around pole
            n_samples: Number of samples in window

        Returns:
            Dictionary with slope metrics
        """
        # Sample points around pole (avoiding exact pole)
        x_vals = []
        log_y_vals = []
        log_q_vals = []

        # Left side of pole
        x_left = np.linspace(pole_location - window_size, pole_location - 1e-6, n_samples // 2)

        # Right side of pole
        x_right = np.linspace(pole_location + 1e-6, pole_location + window_size, n_samples // 2)

        for x_val in np.concatenate([x_left, x_right]):
            try:
                x = TRNode.constant(real(x_val))
                y, tag = model.forward(x)

                # Get |Q| value
                q_abs = None
                if hasattr(model, "get_Q_value"):
                    q_abs = model.get_Q_value()

                # Only include REAL outputs with valid Q
                if (
                    tag == TRTag.REAL
                    and abs(y.value.value) > 1e-12
                    and q_abs is not None
                    and q_abs > 1e-12
                ):
                    x_vals.append(x_val)
                    log_y_vals.append(math.log(abs(y.value.value)))
                    # Use log|Q| (no minus) so ideal slope is -1
                    log_q_vals.append(math.log(q_abs))

            except:
                continue

        # Fit linear regression: log|y| ~ slope * (-log|Q|)
        metrics = {"slope": float("nan"), "r_squared": 0.0, "n_points": len(x_vals)}

        if len(x_vals) >= 3:
            try:
                # Simple linear regression
                n = len(log_q_vals)
                sum_x = sum(log_q_vals)
                sum_y = sum(log_y_vals)
                sum_xx = sum(x * x for x in log_q_vals)
                sum_xy = sum(x * y for x, y in zip(log_q_vals, log_y_vals))

                # Slope and intercept
                denom = n * sum_xx - sum_x * sum_x
                if abs(denom) > 1e-12:
                    slope = (n * sum_xy - sum_x * sum_y) / denom
                    intercept = (sum_y - slope * sum_x) / n

                    # R-squared
                    y_mean = sum_y / n
                    ss_tot = sum((y - y_mean) ** 2 for y in log_y_vals)
                    ss_res = sum(
                        (y - (slope * x + intercept)) ** 2 for x, y in zip(log_q_vals, log_y_vals)
                    )

                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0

                    # For simple pole y ~ 1/Q, slope should be -1 against -log|Q|
                    metrics["slope"] = slope
                    metrics["r_squared"] = r_squared
                    metrics["intercept"] = intercept
                    metrics["slope_error"] = abs(slope - (-1.0))  # Ideal slope is -1

            except:
                pass

        self.history.append(metrics)
        return metrics


class ResidualConsistencyLoss:
    """
    Computes residual consistency loss: R(x) = Q(x)*y(x) - P(x).

    For rational functions, R should be ≈ 0 for all REAL samples,
    especially near poles where the constraint is most important.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.history = []

    def compute_residual(self, model, x: Union[float, TRNode]) -> Optional[TRNode]:
        """
        Compute residual R(x) = Q(x)*y(x) - P(x).

        Args:
            model: Rational model with P and Q access
            x: Input value

        Returns:
            Residual value or None if cannot compute
        """
        if isinstance(x, float):
            x = TRNode.constant(real(x))

        try:
            # Get model output y
            y, tag = model.forward(x)

            if tag != TRTag.REAL:
                return None

            # Compute P(x) and Q(x) if model supports it
            if hasattr(model, "phi") and hasattr(model, "theta") and hasattr(model, "basis"):
                # Evaluate basis
                max_degree = max(len(model.theta), len(model.phi) + 1)
                psi = model.basis(x, max_degree)

                # Compute P(x)
                P = TRNode.constant(real(0.0))
                for k, theta_k in enumerate(model.theta):
                    if k < len(psi):
                        P = P + theta_k * psi[k]

                # Compute Q(x) = 1 + sum(phi_k * psi_{k+1})
                Q = TRNode.constant(real(1.0))
                for k, phi_k in enumerate(model.phi):
                    if k + 1 < len(psi):
                        Q = Q + phi_k * psi[k + 1]

                # Residual: R = Q*y - P
                residual = Q * y - P
                return residual

        except:
            pass

        return None

    def compute_loss(
        self, model, inputs: List[Union[float, TRNode]], near_pole_threshold: float = 0.2
    ) -> TRNode:
        """
        Compute residual consistency loss over a batch.

        Args:
            model: Rational model
            inputs: Input values
            near_pole_threshold: |Q| threshold for "near pole"

        Returns:
            Residual loss
        """
        residuals = []

        for x in inputs:
            residual = self.compute_residual(model, x)

            if residual is not None:
                # Check if near pole
                q_abs = None
                if hasattr(model, "get_Q_value"):
                    if isinstance(x, float):
                        x_node = TRNode.constant(real(x))
                    else:
                        x_node = x
                    _ = model.forward(x_node)
                    q_abs = model.get_Q_value()

                # Weight more heavily if near pole
                weight = 1.0
                if q_abs is not None and q_abs < near_pole_threshold:
                    weight = 2.0  # Double weight for near-pole samples

                weighted_residual = TRNode.constant(real(weight)) * residual * residual
                residuals.append(weighted_residual)

        if not residuals:
            return TRNode.constant(real(0.0))

        # Mean squared residual
        total = residuals[0]
        for r in residuals[1:]:
            total = total + r

        mean_residual = total / TRNode.constant(real(float(len(residuals))))
        loss = TRNode.constant(real(self.weight)) * mean_residual

        self.history.append(loss.value.value if loss.tag == TRTag.REAL else float("inf"))
        return loss


class AntiIllusionMetrics:
    """
    Coordinator class for all anti-illusion metrics.

    Combines PLE, sign consistency, asymptotic slope, and residual
    consistency into a unified evaluation framework.
    """

    def __init__(self):
        self.ple_metric = PoleLocalizationError()
        self.sign_checker = SignConsistencyChecker()
        self.slope_analyzer = AsymptoticSlopeAnalyzer()
        self.residual_loss = ResidualConsistencyLoss()

        self.evaluation_history = []

    def evaluate_model(
        self,
        model,
        ground_truth_poles: List[PoleLocation],
        test_paths: Optional[List[Tuple]] = None,
        x_range: Tuple[float, float] = (-2, 2),
    ) -> Dict[str, float]:
        """
        Comprehensive anti-illusion evaluation.

        Args:
            model: Model to evaluate
            ground_truth_poles: Known pole locations
            test_paths: List of (path_func, t_range, pole_t) for sign consistency
            x_range: Range for pole detection

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # 1. Pole Localization Error
        ple = self.ple_metric.compute_ple(model, ground_truth_poles, x_range)
        metrics["ple"] = ple

        # 2. Sign consistency (if paths provided)
        if test_paths:
            sign_metrics = []
            for path_func, t_range, pole_t in test_paths:
                path_metrics = self.sign_checker.check_path_crossing(
                    model, path_func, t_range, pole_t
                )
                sign_metrics.append(path_metrics["overall_consistency"])

            metrics["sign_consistency"] = np.mean(sign_metrics) if sign_metrics else 0.0
        else:
            metrics["sign_consistency"] = 0.0

        # 3. Asymptotic slope analysis
        slope_errors = []
        for pole in ground_truth_poles:
            slope_metrics = self.slope_analyzer.compute_asymptotic_slope(model, pole.x)
            if not math.isnan(slope_metrics["slope"]):
                slope_errors.append(slope_metrics["slope_error"])

        metrics["asymptotic_slope_error"] = np.mean(slope_errors) if slope_errors else float("inf")

        # 4. Residual consistency
        test_points = np.linspace(x_range[0], x_range[1], 100)
        residual_loss = self.residual_loss.compute_loss(model, test_points.tolist())
        metrics["residual_consistency"] = (
            residual_loss.value.value if residual_loss.tag == TRTag.REAL else float("inf")
        )

        # 5. Composite score (lower is better)
        # Normalize and combine metrics
        ple_norm = min(ple, 1.0)  # Cap at 1.0
        sign_norm = 1.0 - metrics["sign_consistency"]  # Flip so lower is better
        slope_norm = min(metrics["asymptotic_slope_error"], 2.0) / 2.0  # Normalize
        residual_norm = min(metrics["residual_consistency"], 10.0) / 10.0

        metrics["anti_illusion_score"] = (ple_norm + sign_norm + slope_norm + residual_norm) / 4

        self.evaluation_history.append(metrics)
        return metrics

    def get_trends(self, window_size: int = 10) -> Dict[str, str]:
        """
        Analyze trends in metrics over recent evaluations.

        Args:
            window_size: Number of recent evaluations to consider

        Returns:
            Dictionary indicating trend direction for each metric
        """
        if len(self.evaluation_history) < 2:
            return {}

        recent = self.evaluation_history[-window_size:]
        trends = {}

        for metric in [
            "ple",
            "sign_consistency",
            "asymptotic_slope_error",
            "residual_consistency",
            "anti_illusion_score",
        ]:
            values = [eval_dict[metric] for eval_dict in recent if metric in eval_dict]

            if len(values) >= 2:
                # Simple linear trend
                if values[-1] < values[0]:
                    trends[metric] = "improving" if metric != "sign_consistency" else "declining"
                elif values[-1] > values[0]:
                    trends[metric] = "declining" if metric != "sign_consistency" else "improving"
                else:
                    trends[metric] = "stable"

        return trends
