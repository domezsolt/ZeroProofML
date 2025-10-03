# MIT License
# See LICENSE file in the project root for full license text.
"""
Comprehensive metrics for evaluating pole learning and singularity behavior.

This module provides metrics for assessing how well a model learns poles,
including localization accuracy, sign consistency, and asymptotic behavior.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..autodiff import TRNode
from ..autodiff.tr_ops_grad import tr_add, tr_div, tr_mul, tr_sub
from ..core import TRScalar, TRTag, ninf, phi, pinf, real
from .bridge import to_real_scalar


@dataclass
class PoleMetrics:
    """Container for pole-related metrics."""

    # Localization
    ple: float  # Pole Localization Error
    ple_breakdown: Dict[str, float]  # PLE per pole

    # Sign consistency
    sign_consistency: float  # Fraction of correct sign flips
    sign_flip_errors: List[Tuple[float, str]]  # Locations of errors

    # Asymptotic behavior
    asymptotic_slope_error: float  # Error in log|y| ~ -log|Q| relationship
    slope_correlation: float  # Correlation between log|y| and -log|Q|

    # Residual consistency
    residual_error: float  # Mean |R(x)| near poles
    residual_max: float  # Max |R(x)| near poles

    # Counts
    actual_pole_count: int
    predicted_pole_count: int
    true_positive_poles: int
    false_positive_poles: int
    false_negative_poles: int

    # Coverage breakdown
    coverage_near: float  # Coverage within 0.1 of poles
    coverage_mid: float  # Coverage 0.1-0.5 from poles
    coverage_far: float  # Coverage >0.5 from poles

    # Additional statistics
    mean_q_at_poles: float  # Mean |Q| at detected poles
    min_q_at_poles: float  # Min |Q| at detected poles


def compute_pole_localization_error(
    predicted_poles: List[float], true_poles: List[float], max_distance: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Pole Localization Error (PLE).

    Uses Hungarian algorithm for optimal matching between predicted and true poles.

    Args:
        predicted_poles: List of predicted pole locations
        true_poles: List of true pole locations
        max_distance: Maximum distance for matching (unmatched = max_distance)

    Returns:
        Tuple of (mean PLE, per-pole PLE breakdown)
    """
    if not true_poles:
        # No true poles, any prediction is false positive
        if predicted_poles:
            return max_distance, {}
        else:
            return 0.0, {}

    if not predicted_poles:
        # No predictions, all true poles are missed
        return max_distance, {f"pole_{i}": max_distance for i in range(len(true_poles))}

    # Compute distance matrix
    n_true = len(true_poles)
    n_pred = len(predicted_poles)
    distances = np.full((n_true, n_pred), max_distance)

    for i, true_pole in enumerate(true_poles):
        for j, pred_pole in enumerate(predicted_poles):
            distances[i, j] = abs(true_pole - pred_pole)

    # Simple greedy matching (for small pole counts)
    # For larger problems, use scipy.optimize.linear_sum_assignment
    matched_distances = []
    breakdown = {}
    used_preds = set()

    for i, true_pole in enumerate(true_poles):
        if len(used_preds) < n_pred:
            # Find best unused prediction
            best_j = None
            best_dist = max_distance
            for j in range(n_pred):
                if j not in used_preds and distances[i, j] < best_dist:
                    best_dist = distances[i, j]
                    best_j = j

            if best_j is not None:
                used_preds.add(best_j)
                matched_distances.append(best_dist)
                breakdown[f"pole_{i}_at_{true_pole:.3f}"] = best_dist
            else:
                matched_distances.append(max_distance)
                breakdown[f"pole_{i}_at_{true_pole:.3f}"] = max_distance
        else:
            # No more predictions available
            matched_distances.append(max_distance)
            breakdown[f"pole_{i}_at_{true_pole:.3f}"] = max_distance

    # Add penalty for extra predictions (false positives)
    n_extra = n_pred - len(used_preds)
    for _ in range(n_extra):
        matched_distances.append(max_distance)

    mean_ple = np.mean(matched_distances) if matched_distances else 0.0
    return mean_ple, breakdown


def check_sign_consistency(
    x_values: List[float],
    y_values: List[Union[TRNode, TRScalar]],
    true_poles: List[float],
    tolerance: float = 0.1,
) -> Tuple[float, List[Tuple[float, str]]]:
    """
    Check if model correctly flips between +∞ and -∞ across poles.

    Args:
        x_values: Input values (sorted)
        y_values: Model outputs
        true_poles: True pole locations
        tolerance: Distance from pole to check

    Returns:
        Tuple of (consistency score, list of errors)
    """
    if not true_poles or len(x_values) < 2:
        return 1.0, []

    errors = []
    checks = 0
    correct = 0

    # Check sign changes across each pole
    for pole in true_poles:
        # Find points just before and after pole
        before_idx = None
        after_idx = None

        for i, x in enumerate(x_values):
            if x < pole - tolerance / 2:
                before_idx = i
            elif x > pole + tolerance / 2 and after_idx is None:
                after_idx = i
                break

        if before_idx is not None and after_idx is not None:
            checks += 1

            # Get values
            y_before = y_values[before_idx]
            y_after = y_values[after_idx]

            # Check tags and signs
            tag_before = y_before.tag if hasattr(y_before, "tag") else y_before.value.tag
            tag_after = y_after.tag if hasattr(y_after, "tag") else y_after.value.tag

            # Should have opposite infinities
            if (tag_before == TRTag.PINF and tag_after == TRTag.NINF) or (
                tag_before == TRTag.NINF and tag_after == TRTag.PINF
            ):
                correct += 1
            else:
                errors.append((pole, f"No sign flip: {tag_before} -> {tag_after}"))

    consistency = correct / checks if checks > 0 else 1.0
    return consistency, errors


def compute_asymptotic_slope_error(
    x_values: List[float],
    y_values: List[Union[TRNode, TRScalar]],
    Q_values: List[float],
    near_pole_threshold: float = 0.2,
) -> Tuple[float, float]:
    """
    Check if log|y| ~ -log|Q| near poles (asymptotic behavior).

    Args:
        x_values: Input values
        y_values: Model outputs
        Q_values: Denominator values |Q(x)|
        near_pole_threshold: Threshold for "near pole"

    Returns:
        Tuple of (mean error, correlation)
    """
    log_y = []
    neg_log_q = []

    for i, (x, y, q) in enumerate(zip(x_values, y_values, Q_values)):
        if abs(q) < near_pole_threshold and abs(q) > 1e-10:
            # Coerce y to TRScalar semantics (robust to floats/np)
            try:
                y_tr = to_real_scalar(y)
            except Exception:
                # Best-effort: treat as REAL float
                try:
                    y_tr = real(float(y))
                except Exception:
                    continue

            if y_tr.tag == TRTag.REAL:
                y_val = y_tr.value
                if abs(y_val) > 1e-10:
                    log_y.append(math.log(abs(y_val)))
                    neg_log_q.append(-math.log(abs(q)))

    if len(log_y) < 2:
        return 0.0, 1.0  # Not enough data

    # Compute correlation and mean squared error
    log_y = np.array(log_y)
    neg_log_q = np.array(neg_log_q)

    # Normalize for comparison
    log_y_norm = (log_y - np.mean(log_y)) / (np.std(log_y) + 1e-10)
    neg_log_q_norm = (neg_log_q - np.mean(neg_log_q)) / (np.std(neg_log_q) + 1e-10)

    # Mean squared error
    mse = np.mean((log_y_norm - neg_log_q_norm) ** 2)

    # Correlation
    correlation = np.corrcoef(log_y, neg_log_q)[0, 1]

    return mse, correlation


def compute_residual_consistency(
    x_values: List[float],
    P_values: List[Union[TRNode, float]],
    Q_values: List[float],
    y_values: List[Union[TRNode, TRScalar]],
    near_pole_threshold: float = 0.2,
) -> Tuple[float, float]:
    """
    Compute residual R(x) = Q(x)·y(x) - P(x) near poles.

    Should be approximately 0 if the model is consistent.

    Args:
        x_values: Input values
        P_values: Numerator values P(x)
        Q_values: Denominator values Q(x)
        y_values: Model outputs y(x)
        near_pole_threshold: Threshold for "near pole"

    Returns:
        Tuple of (mean |R|, max |R|)
    """
    residuals = []

    for x, p, q, y in zip(x_values, P_values, Q_values, y_values):
        if abs(q) < near_pole_threshold:
            # Near pole - compute residual
            # Coerce y to TRScalar to handle numpy/float robustly
            try:
                y_tr = to_real_scalar(y)  # type: ignore[arg-type]
            except Exception:
                # Best-effort fallback: treat as REAL float
                try:
                    y_tr = real(float(y))
                except Exception:
                    continue

            if y_tr.tag == TRTag.REAL:
                # Extract values
                y_val = y_tr.value

                p_val = p
                if isinstance(p, TRNode):
                    p_val = p.value.value if p.value.tag == TRTag.REAL else 0
                elif isinstance(p, TRScalar):
                    p_val = p.value if p.tag == TRTag.REAL else 0

                # Compute residual
                residual = abs(q * y_val - p_val)
                residuals.append(residual)

    if not residuals:
        return 0.0, 0.0

    mean_residual = np.mean(residuals)
    max_residual = np.max(residuals)

    return mean_residual, max_residual


def count_singularities(
    y_values: List[Union[TRNode, TRScalar, float, int, "np.floating"]],
    Q_values: Optional[List[float]] = None,
    q_threshold: float = 0.1,
) -> Tuple[int, int]:
    """
    Count actual singularities (non-REAL outputs) vs predicted.

    Args:
        y_values: Model outputs
        Q_values: Optional Q values for prediction counting
        q_threshold: Threshold for predicted singularity

    Returns:
        Tuple of (actual count, predicted count)
    """
    # Count actual non-REAL outputs
    actual_count = 0
    for y in y_values:
        # Handle TRNode / TRScalar
        if hasattr(y, "tag"):
            tag = y.tag
        elif hasattr(y, "value") and hasattr(y.value, "tag"):
            tag = y.value.tag
        else:
            # Numeric path (float/int/np.floating)
            try:
                import numpy as np  # type: ignore

                if np.isnan(y):
                    tag = TRTag.PHI
                elif np.isposinf(y):
                    tag = TRTag.PINF
                elif np.isneginf(y):
                    tag = TRTag.NINF
                else:
                    tag = TRTag.REAL
            except Exception:
                # Fallback: treat as REAL
                tag = TRTag.REAL
        if tag != TRTag.REAL:
            actual_count += 1

    # Count predicted singularities based on Q values
    predicted_count = 0
    if Q_values is not None:
        try:
            import numpy as np  # type: ignore

            iterable = Q_values.tolist() if hasattr(Q_values, "tolist") else Q_values
        except Exception:
            iterable = Q_values
        for q in iterable:
            try:
                val = float(q)
            except Exception:
                # If cannot convert, skip
                continue
            if abs(val) < q_threshold:
                predicted_count += 1

    return actual_count, predicted_count


def compute_coverage_by_distance(
    x_values: List[float],
    y_values: List[Union[TRNode, TRScalar, float, int, "np.floating"]],
    true_poles: List[float],
    near_threshold: float = 0.1,
    mid_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute coverage breakdown by distance from poles.

    Args:
        x_values: Input values
        y_values: Model outputs
        true_poles: True pole locations
        near_threshold: Distance for "near" classification
        mid_threshold: Distance for "mid" classification

    Returns:
        Tuple of (near coverage, mid coverage, far coverage)
    """
    if not true_poles:
        # No poles, everything is "far"
        total = len(y_values)
        real_count = sum(
            1 for y in y_values if (y.tag if hasattr(y, "tag") else y.value.tag) == TRTag.REAL
        )
        return 0.0, 0.0, real_count / total if total > 0 else 1.0

    near_real = 0
    near_total = 0
    mid_real = 0
    mid_total = 0
    far_real = 0
    far_total = 0

    def _get_tag(y) -> TRTag:
        # TRNode or TRScalar path
        if hasattr(y, "tag"):
            return y.tag  # type: ignore[attr-defined]
        if hasattr(y, "value") and hasattr(y.value, "tag"):
            return y.value.tag  # type: ignore[attr-defined]
        # Numeric path
        try:
            import numpy as np  # type: ignore

            yf = float(y)
            if np.isnan(yf):
                return TRTag.PHI
            if np.isposinf(yf):
                return TRTag.PINF
            if np.isneginf(yf):
                return TRTag.NINF
            return TRTag.REAL
        except Exception:
            return TRTag.REAL

    for x, y in zip(x_values, y_values):
        # Find distance to nearest pole
        min_dist = min(abs(x - pole) for pole in true_poles)

        # Classify by distance
        if min_dist < near_threshold:
            near_total += 1
            tag = _get_tag(y)
            if tag == TRTag.REAL:
                near_real += 1
        elif min_dist < mid_threshold:
            mid_total += 1
            tag = _get_tag(y)
            if tag == TRTag.REAL:
                mid_real += 1
        else:
            far_total += 1
            tag = _get_tag(y)
            if tag == TRTag.REAL:
                far_real += 1

    # Compute coverage for each region
    near_coverage = near_real / near_total if near_total > 0 else 0.0
    mid_coverage = mid_real / mid_total if mid_total > 0 else 1.0
    far_coverage = far_real / far_total if far_total > 0 else 1.0

    return near_coverage, mid_coverage, far_coverage


def detect_poles_from_Q(
    x_values: List[float],
    Q_values: List[float],
    threshold: float = 0.1,
    min_separation: float = 0.1,
) -> List[float]:
    """
    Detect pole locations from Q values.

    Args:
        x_values: Input values
        Q_values: |Q(x)| values
        threshold: Threshold for pole detection
        min_separation: Minimum separation between poles

    Returns:
        List of detected pole locations
    """
    poles = []

    for i, (x, q) in enumerate(zip(x_values, Q_values)):
        if abs(q) < threshold:
            # Check if it's a local minimum
            is_minimum = True
            if i > 0 and abs(Q_values[i - 1]) < abs(q):
                is_minimum = False
            if i < len(Q_values) - 1 and abs(Q_values[i + 1]) < abs(q):
                is_minimum = False

            if is_minimum:
                # Check separation from existing poles
                too_close = False
                for pole in poles:
                    if abs(x - pole) < min_separation:
                        too_close = True
                        break

                if not too_close:
                    poles.append(x)

    return sorted(poles)


class PoleEvaluator:
    """
    Comprehensive evaluator for pole learning metrics.

    Combines all metrics into a unified evaluation interface.
    """

    def __init__(
        self,
        true_poles: Optional[List[float]] = None,
        near_threshold: float = 0.1,
        mid_threshold: float = 0.5,
    ):
        """
        Initialize pole evaluator.

        Args:
            true_poles: Ground truth pole locations
            near_threshold: Distance for "near pole"
            mid_threshold: Distance for "mid-range"
        """
        self.true_poles = true_poles or []
        self.near_threshold = near_threshold
        self.mid_threshold = mid_threshold

        # History tracking
        self.evaluation_history = []

    def evaluate(
        self,
        x_values: List[float],
        y_values: List[Union[TRNode, TRScalar, float, int, "np.floating"]],
        Q_values: List[float],
        P_values: Optional[List[Union[TRNode, float]]] = None,
        predicted_poles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Perform comprehensive pole evaluation.

        Args:
            x_values: Input values
            y_values: Model outputs
            Q_values: Denominator values |Q(x)|
            P_values: Optional numerator values
            predicted_poles: Optional predicted pole locations

        Returns:
            PoleMetrics object with all metrics
        """
        # Auto-detect poles if not provided
        if predicted_poles is None:
            predicted_poles = detect_poles_from_Q(x_values, Q_values, self.near_threshold)

        # Normalize y_values to TR-like where needed (accept numpy floats)
        processed_y: List[Union[TRNode, TRScalar]] = []
        import numpy as np

        for y in y_values:
            if hasattr(y, "tag") or (hasattr(y, "value") and hasattr(y.value, "tag")):
                processed_y.append(y)
            else:
                y_float = float(y)
                if np.isinf(y_float):
                    processed_y.append(pinf() if y_float > 0 else ninf())
                elif np.isnan(y_float):
                    processed_y.append(phi())
                else:
                    processed_y.append(real(y_float))

        # Compute PLE
        ple, ple_breakdown = compute_pole_localization_error(predicted_poles, self.true_poles)

        # Check sign consistency
        sign_consistency, sign_errors = check_sign_consistency(
            x_values, processed_y, self.true_poles
        )

        # Compute asymptotic slope
        slope_error, slope_corr = compute_asymptotic_slope_error(
            x_values, processed_y, Q_values, self.near_threshold
        )

        # Compute residual consistency
        if P_values:
            residual_mean, residual_max = compute_residual_consistency(
                x_values, P_values, Q_values, y_values, self.near_threshold
            )
        else:
            residual_mean, residual_max = 0.0, 0.0

        # Count singularities
        actual_count, predicted_count = count_singularities(y_values, Q_values, self.near_threshold)

        # Coverage breakdown
        near_cov, mid_cov, far_cov = compute_coverage_by_distance(
            x_values, y_values, self.true_poles, self.near_threshold, self.mid_threshold
        )

        # Pole detection accuracy
        true_positives = 0
        for pred_pole in predicted_poles:
            if any(abs(pred_pole - true) < self.near_threshold for true in self.true_poles):
                true_positives += 1

        false_positives = len(predicted_poles) - true_positives
        false_negatives = len(self.true_poles) - true_positives

        # Q values at detected poles
        q_at_poles = []
        for pole in predicted_poles:
            # Find closest x value
            closest_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - pole))
            q_at_poles.append(abs(Q_values[closest_idx]))

        mean_q = np.mean(q_at_poles) if q_at_poles else 0.0
        min_q = np.min(q_at_poles) if q_at_poles else 0.0

        # Create metrics object
        metrics_obj = PoleMetrics(
            ple=ple,
            ple_breakdown=ple_breakdown,
            sign_consistency=sign_consistency,
            sign_flip_errors=sign_errors,
            asymptotic_slope_error=slope_error,
            slope_correlation=slope_corr,
            residual_error=residual_mean,
            residual_max=residual_max,
            actual_pole_count=len(self.true_poles),
            predicted_pole_count=len(predicted_poles),
            true_positive_poles=true_positives,
            false_positive_poles=false_positives,
            false_negative_poles=false_negatives,
            coverage_near=near_cov,
            coverage_mid=mid_cov,
            coverage_far=far_cov,
            mean_q_at_poles=mean_q,
            min_q_at_poles=min_q,
        )

        # Store in history
        self.evaluation_history.append(metrics_obj)

        # Return as plain dict for downstream code expecting mapping
        from dataclasses import asdict

        result: Dict[str, float] = asdict(metrics_obj)  # type: ignore[assignment]
        return result

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of evaluation history.

        Returns:
            Dictionary of aggregated metrics
        """
        if not self.evaluation_history:
            return {}

        # Aggregate over history
        latest = self.evaluation_history[-1]

        summary = {
            "ple": latest.ple,
            "sign_consistency": latest.sign_consistency,
            "asymptotic_slope_error": latest.asymptotic_slope_error,
            "residual_error": latest.residual_error,
            "pole_precision": (
                latest.true_positive_poles
                / (latest.true_positive_poles + latest.false_positives + 1e-10)
            ),
            "pole_recall": (
                latest.true_positive_poles
                / (latest.true_positive_poles + latest.false_negatives + 1e-10)
            ),
            "coverage_near": latest.coverage_near,
            "coverage_mid": latest.coverage_mid,
            "coverage_far": latest.coverage_far,
        }

        # Add improvement if we have history
        if len(self.evaluation_history) > 1:
            first = self.evaluation_history[0]
            summary["ple_improvement"] = first.ple - latest.ple
            summary["coverage_near_improvement"] = latest.coverage_near - first.coverage_near

        return summary
