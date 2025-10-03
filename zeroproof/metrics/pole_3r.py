# MIT License
# See LICENSE file in the project root for full license text.
"""
3R near-pole metrics for a planar 3R arm.

We generalize the 2D helpers to the 3R setting where the Jacobian J is 2x3.
Singular sets (rank drop 2→1) occur when successive link orientations align,
in particular along the lines theta2 ∈ {0, π} and theta3 ∈ {0, π} (mod 2π).

Metrics provided:
- PLE to the nearest singular set (min distance to theta2 or theta3 lines)
- Sign consistency across theta2=0 and across theta3=0 (reported separately)
- Residual consistency via forward kinematics with predicted Δθ
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def _fk_3r(
    theta1: float, theta2: float, theta3: float, L1: float = 1.0, L2: float = 1.0, L3: float = 1.0
) -> Tuple[float, float]:
    """Planar 3R forward kinematics for end-effector position (x,y)."""
    t12 = theta1 + theta2
    t123 = t12 + theta3
    x = L1 * math.cos(theta1) + L2 * math.cos(t12) + L3 * math.cos(t123)
    y = L1 * math.sin(theta1) + L2 * math.sin(t12) + L3 * math.sin(t123)
    return x, y


def compute_residual_consistency_3r(
    test_inputs: List[List[float]],
    predictions: List[List[float]],
    L1: float = 1.0,
    L2: float = 1.0,
    L3: float = 1.0,
) -> float:
    """Mean squared residual between desired and achieved displacement (3R).

    Args:
        test_inputs: [[theta1, theta2, theta3, dx, dy], ...]
        predictions: [[dtheta1, dtheta2, dtheta3], ...]
    Returns:
        Mean squared residual over samples.
    """
    if not test_inputs or not predictions:
        return float("inf")
    errs: List[float] = []
    for inp, pred in zip(test_inputs, predictions):
        if len(inp) < 5 or len(pred) < 3:
            continue
        th1, th2, th3, dx_t, dy_t = [float(v) for v in inp[:5]]
        dth1, dth2, dth3 = [float(v) for v in pred[:3]]
        x0, y0 = _fk_3r(th1, th2, th3, L1, L2, L3)
        x1, y1 = _fk_3r(th1 + dth1, th2 + dth2, th3 + dth3, L1, L2, L3)
        dx_hat, dy_hat = (x1 - x0), (y1 - y0)
        err = (dx_hat - dx_t) ** 2 + (dy_hat - dy_t) ** 2
        errs.append(err)
    return float(np.mean(errs)) if errs else float("inf")


def compute_ple_to_3r_lines(
    test_inputs: List[List[float]], predictions: List[List[float]], top_k_ratio: float = 0.05
) -> float:
    """Approximate PLE to the nearest 3R singular line.

    We select the top-|Δθ| samples and average the distance (in radians)
    to the nearest of {theta2 ∈ {0,π}, theta3 ∈ {0,π}}.
    """
    if not test_inputs or not predictions:
        return float("inf")
    n = min(len(test_inputs), len(predictions))
    norms: List[Tuple[int, float]] = []
    for i in range(n):
        dth = predictions[i]
        if len(dth) < 3:
            continue
        # Use L2 norm of the joint update vector
        norms.append((i, float(math.sqrt(sum(float(v) * float(v) for v in dth[:3])))))
    if not norms:
        return float("inf")
    norms.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(norms) * top_k_ratio))
    top_idx = [idx for idx, _ in norms[:k]]
    dists: List[float] = []
    for i in top_idx:
        if len(test_inputs[i]) < 3:
            continue
        th2 = float(test_inputs[i][1])
        th3 = float(test_inputs[i][2])
        # Distances to each line (wrapped to [-pi,pi])
        d2_0 = abs(_wrap_pi(th2))
        d2_pi = abs(_wrap_pi(th2 - math.pi))
        d3_0 = abs(_wrap_pi(th3))
        d3_pi = abs(_wrap_pi(th3 - math.pi))
        dists.append(min(d2_0, d2_pi, d3_0, d3_pi))
    return float(np.mean(dists)) if dists else float("inf")


def compute_sign_consistency_rate_3r(
    test_inputs: List[List[float]],
    predictions: List[List[float]],
    n_paths: int = 8,
    th1_tol: float = 0.08,
    th2_window: float = 0.25,
    th3_window: float = 0.25,
    min_mag: float = 1e-3,
) -> Dict[str, float]:
    """Estimate sign flip consistency across theta2=0 and theta3=0.

    Returns a dict with keys 'theta2' and 'theta3' reporting the fraction
    of anchors exhibiting a sign flip for Δθ2 and Δθ3 respectively.
    """
    result = {"theta2": 0.0, "theta3": 0.0}
    if not test_inputs or not predictions:
        return result
    th1_vals = np.array([float(inp[0]) for inp in test_inputs])
    th2_vals = np.array([float(inp[1]) for inp in test_inputs])
    th3_vals = np.array([float(inp[2]) for inp in test_inputs])
    dth2_vals = np.array([float(pred[1]) if len(pred) > 1 else 0.0 for pred in predictions])
    dth3_vals = np.array([float(pred[2]) if len(pred) > 2 else 0.0 for pred in predictions])

    if len(th1_vals) < 2:
        return result

    anchors = np.linspace(np.min(th1_vals), np.max(th1_vals), num=n_paths)

    # Theta2 crossing
    flips = 0
    valid = 0
    for a in anchors:
        mask = (np.abs(th1_vals - a) <= th1_tol) & (np.abs(th2_vals) <= th2_window)
        idx = np.where(mask)[0]
        if idx.size < 4:
            continue
        before = dth2_vals[idx[th2_vals[idx] < 0.0]]
        after = dth2_vals[idx[th2_vals[idx] > 0.0]]
        # Filter tiny magnitudes
        before = before[np.abs(before) > min_mag]
        after = after[np.abs(after) > min_mag]
        if before.size == 0 or after.size == 0:
            continue
        sign_before = np.sign(np.mean(np.sign(before)))
        sign_after = np.sign(np.mean(np.sign(after)))
        if sign_before == 0 or sign_after == 0:
            continue
        valid += 1
        flips += 1 if sign_before != sign_after else 0
    result["theta2"] = float(flips / valid) if valid > 0 else 0.0

    # Theta3 crossing
    flips = 0
    valid = 0
    for a in anchors:
        mask = (np.abs(th1_vals - a) <= th1_tol) & (np.abs(th3_vals) <= th3_window)
        idx = np.where(mask)[0]
        if idx.size < 4:
            continue
        before = dth3_vals[idx[th3_vals[idx] < 0.0]]
        after = dth3_vals[idx[th3_vals[idx] > 0.0]]
        before = before[np.abs(before) > min_mag]
        after = after[np.abs(after) > min_mag]
        if before.size == 0 or after.size == 0:
            continue
        sign_before = np.sign(np.mean(np.sign(before)))
        sign_after = np.sign(np.mean(np.sign(after)))
        if sign_before == 0 or sign_after == 0:
            continue
        valid += 1
        flips += 1 if sign_before != sign_after else 0
    result["theta3"] = float(flips / valid) if valid > 0 else 0.0

    return result


def compute_pole_metrics_3r(
    test_inputs: List[List[float]],
    predictions: List[List[float]],
    L1: float = 1.0,
    L2: float = 1.0,
    L3: float = 1.0,
) -> Dict[str, float]:
    """Compute a bundle of 3R near-pole metrics.

    Returns a dict with keys:
      - ple: PLE to nearest 3R singular set
      - sign_consistency_theta2: sign flip consistency across theta2=0
      - sign_consistency_theta3: sign flip consistency across theta3=0
      - residual_consistency: FK residual MSE
    """
    signs = compute_sign_consistency_rate_3r(test_inputs, predictions)
    return {
        "ple": compute_ple_to_3r_lines(test_inputs, predictions),
        "sign_consistency_theta2": float(signs.get("theta2", 0.0)),
        "sign_consistency_theta3": float(signs.get("theta3", 0.0)),
        "residual_consistency": compute_residual_consistency_3r(
            test_inputs, predictions, L1=L1, L2=L2, L3=L3
        ),
    }


def compute_paired_sign_consistency_3r(
    test_inputs: List[List[float]],
    predictions: List[List[float]],
    joint: str = "theta2",
    phi_deg: float = 60.0,
    phi_tol_deg: float = 35.0,
    th_window: float = 0.35,
    k: int = 4,
    min_mag: float = 5e-4,
) -> Dict[str, float]:
    """Paired sign-flip consistency across theta2=0 or theta3=0 under a direction window.

    Pairs k closest |theta_j| samples on negative side with k on positive side
    (by |theta_j|), subject to a direction window on phi = atan2(dy, dx).

    Args:
        test_inputs: [[theta1, theta2, theta3, dx, dy], ...]
        predictions: [[dtheta1, dtheta2, dtheta3], ...]
        joint: 'theta2' or 'theta3'
        phi_deg: target displacement direction in degrees
        phi_tol_deg: tolerance around phi_deg
        th_window: window around 0 for |theta_j|
        k: number of pairs to evaluate (min with available pairs)
        min_mag: min |dtheta_j| magnitude to count as valid

    Returns:
        {'rate': float, 'pairs': int}
    """
    if not test_inputs or not predictions:
        return {"rate": 0.0, "pairs": 0}

    import math

    # Build filtered lists
    neg: List[Tuple[float, float]] = []  # (|theta_j|, dtheta_j)
    pos: List[Tuple[float, float]] = []
    j_idx = 1 if joint == "theta2" else 2
    for inp, pred in zip(test_inputs, predictions):
        if len(inp) < 5 or len(pred) < 3:
            continue
        thj = float(inp[j_idx])
        dx, dy = float(inp[3]), float(inp[4])
        phi = math.degrees(math.atan2(dy, dx))
        # Wrap to [-180,180]
        while phi > 180.0:
            phi -= 360.0
        while phi < -180.0:
            phi += 360.0

        # Distance to target angle (circular)
        def ang_diff(a, b):
            d = (a - b + 180.0) % 360.0 - 180.0
            return abs(d)

        if ang_diff(phi, phi_deg) > phi_tol_deg:
            continue
        if abs(thj) > th_window:
            continue
        dthj = float(pred[j_idx])
        if thj < 0:
            neg.append((abs(thj), dthj))
        elif thj > 0:
            pos.append((abs(thj), dthj))
        # thj==0 is ignored
    # Sort by |theta_j|
    neg.sort(key=lambda t: t[0])
    pos.sort(key=lambda t: t[0])
    m = min(int(k), len(neg), len(pos))
    if m == 0:
        return {"rate": 0.0, "pairs": 0}
    flips = 0
    valid = 0
    for i in range(m):
        _, d_before = neg[i]
        _, d_after = pos[i]
        if abs(d_before) <= min_mag or abs(d_after) <= min_mag:
            continue
        s_before = 1.0 if d_before > 0 else (-1.0 if d_before < 0 else 0.0)
        s_after = 1.0 if d_after > 0 else (-1.0 if d_after < 0 else 0.0)
        if s_before == 0.0 or s_after == 0.0:
            continue
        valid += 1
        if s_before != s_after:
            flips += 1
    rate = float(flips) / float(valid) if valid > 0 else 0.0
    return {"rate": rate, "pairs": valid}
