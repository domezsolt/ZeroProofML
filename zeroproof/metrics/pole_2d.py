"""
2D near-pole metrics for the RR arm robotics example.

These helpers compute PLE to analytic singularity lines (theta2 in {0, pi}),
sign consistency across a theta2-crossing path, slope error near poles, and
residual consistency using forward kinematics with predicted delta-theta.
"""

from typing import Dict, List, Tuple, Optional
import math

import numpy as np


def _forward_kinematics(theta1: float, theta2: float, L1: float = 1.0, L2: float = 1.0) -> Tuple[float, float]:
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y


def compute_residual_consistency(test_inputs: List[List[float]],
                                 predictions: List[List[float]],
                                 L1: float = 1.0, L2: float = 1.0) -> float:
    """Mean squared residual between desired and achieved displacement.

    Args:
        test_inputs: [[theta1, theta2, dx, dy], ...]
        predictions: [[dtheta1, dtheta2], ...]
    Returns:
        Mean squared residual over samples.
    """
    if not test_inputs or not predictions:
        return float('inf')
    errs = []
    for inp, pred in zip(test_inputs, predictions):
        if len(inp) < 4 or len(pred) < 2:
            continue
        th1, th2, dx_t, dy_t = inp
        dth1, dth2 = pred[:2]
        x0, y0 = _forward_kinematics(th1, th2, L1, L2)
        x1, y1 = _forward_kinematics(th1 + dth1, th2 + dth2, L1, L2)
        dx_hat, dy_hat = (x1 - x0), (y1 - y0)
        err = (dx_hat - dx_t) ** 2 + (dy_hat - dy_t) ** 2
        errs.append(err)
    return float(np.mean(errs)) if errs else float('inf')


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def compute_ple_to_lines(test_inputs: List[List[float]],
                         predictions: List[List[float]],
                         top_k_ratio: float = 0.05) -> float:
    """Approximate PLE by selecting top-|dtheta| samples and averaging distance to theta2 lines.

    For RR, analytic poles are theta2=0 and theta2=pi (modulo 2*pi). We approximate
    predicted pole candidates as samples with largest ||dtheta||.

    Args:
        test_inputs: [[theta1, theta2, dx, dy], ...]
        predictions: [[dtheta1, dtheta2], ...]
        top_k_ratio: fraction of test samples to consider as pole candidates
    Returns:
        Average distance in radians to the nearest pole line.
    """
    if not test_inputs or not predictions:
        return float('inf')
    n = min(len(test_inputs), len(predictions))
    norms = []
    for i in range(n):
        dth = predictions[i]
        if len(dth) < 2:
            continue
        norms.append((i, math.hypot(float(dth[0]), float(dth[1]))))
    if not norms:
        return float('inf')
    norms.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(norms) * top_k_ratio))
    top_idx = [idx for idx, _ in norms[:k]]
    # Compute distances to nearest line theta2 in {0, pi}
    dists = []
    for i in top_idx:
        th2 = float(test_inputs[i][1])
        # Distance to 0 and pi (wrap to [-pi, pi])
        d0 = abs(_wrap_pi(th2))
        d_pi = abs(_wrap_pi(th2 - math.pi))
        dists.append(min(d0, d_pi))
    return float(np.mean(dists)) if dists else float('inf')


def compute_sign_consistency_rate(test_inputs: List[List[float]],
                                  predictions: List[List[float]],
                                  n_paths: int = 5,
                                  th1_tol: float = 0.05,
                                  th2_window: float = 0.2) -> float:
    """Estimate sign flip consistency across theta2=0 crossing.

    For a few theta1 anchors, collect samples within |theta1 - anchor|<=tol and |theta2|<=window.
    Compute dominant sign of predicted dtheta2 before (theta2<0) and after (theta2>0).
    Return fraction of anchors exhibiting a sign flip.
    """
    if not test_inputs or not predictions:
        return 0.0
    th1_vals = np.array([float(inp[0]) for inp in test_inputs])
    th2_vals = np.array([float(inp[1]) for inp in test_inputs])
    dth2_vals = np.array([float(pred[1]) if len(pred) > 1 else 0.0 for pred in predictions])

    # Choose evenly spaced anchors for theta1
    if len(th1_vals) < 2:
        return 0.0
    anchors = np.linspace(np.min(th1_vals), np.max(th1_vals), num=n_paths)
    flips = 0
    valid = 0
    for a in anchors:
        mask = (np.abs(th1_vals - a) <= th1_tol) & (np.abs(th2_vals) <= th2_window)
        idx = np.where(mask)[0]
        if idx.size < 4:
            continue
        before = dth2_vals[idx[th2_vals[idx] < 0.0]]
        after = dth2_vals[idx[th2_vals[idx] > 0.0]]
        if before.size == 0 or after.size == 0:
            continue
        # Dominant sign
        sign_before = np.sign(np.mean(np.sign(before[before != 0]) if np.any(before != 0) else 0.0))
        sign_after = np.sign(np.mean(np.sign(after[after != 0]) if np.any(after != 0) else 0.0))
        if sign_before == 0 or sign_after == 0:
            continue
        valid += 1
        flips += 1 if sign_before != sign_after else 0
    return float(flips / valid) if valid > 0 else 0.0


def compute_slope_error_near_pole(test_inputs: List[List[float]],
                                  predictions: List[List[float]],
                                  detj_eps: float = 1e-6,
                                  max_detj: float = 1e-2) -> float:
    """Fit slope of log ||dtheta|| vs log |sin(theta2)| near poles; expect ~-1.

    Args:
        detj_eps: epsilon added inside log for stability
        max_detj: upper bound for |sin theta2| to consider near-pole samples
    Returns:
        |slope + 1| as slope error (lower is better)
    """
    if not test_inputs or not predictions:
        return float('inf')
    vals = []
    for inp, pred in zip(test_inputs, predictions):
        if len(inp) < 2 or len(pred) < 2:
            continue
        th2 = float(inp[1])
        q = abs(math.sin(th2))
        if q <= max_detj:
            y = math.log10(max(1e-12, math.hypot(float(pred[0]), float(pred[1]))))
            x = math.log10(max(detj_eps, q))
            vals.append((x, y))
    if len(vals) < 5:
        return float('inf')
    xs = np.array([v[0] for v in vals])
    ys = np.array([v[1] for v in vals])
    # Linear regression slope
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    denom = np.sum((xs - x_mean) ** 2)
    if denom == 0:
        return float('inf')
    slope = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
    return abs(slope + 1.0)


def compute_pole_metrics_2d(test_inputs: List[List[float]],
                             predictions: List[List[float]],
                             L1: float = 1.0, L2: float = 1.0) -> Dict[str, float]:
    """Compute a bundle of 2D near-pole metrics.

    Returns:
        Dict with keys: ple, sign_consistency, slope_error, residual_consistency
    """
    return {
        'ple': compute_ple_to_lines(test_inputs, predictions),
        'sign_consistency': compute_sign_consistency_rate(test_inputs, predictions),
        'slope_error': compute_slope_error_near_pole(test_inputs, predictions),
        'residual_consistency': compute_residual_consistency(test_inputs, predictions, L1=L1, L2=L2),
    }

