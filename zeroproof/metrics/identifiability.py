# MIT License
# See LICENSE file in the project root for full license text.
"""
Identifiability diagnostics for rational layers.

Provides a Sylvester matrix smallest singular value as a coprimeness
surrogate for univariate polynomials P and Q in TRRational layers.

If s_min ≈ 0, P and Q share (near-)common factors; larger s_min
indicates better-separated factors (coprime).
"""

from __future__ import annotations

from typing import List

import numpy as np


def _extract_coeffs(nodes: List) -> List[float]:
    coeffs: List[float] = []
    for n in nodes:
        try:
            # TRNode with TRScalar value -> Python float
            coeffs.append(float(n.value))
        except Exception:
            try:
                coeffs.append(float(n))
            except Exception:
                coeffs.append(0.0)
    return coeffs


def _sylvester_matrix(a: List[float], b: List[float]) -> np.ndarray:
    """
    Build the (m+n) x (m+n) Sylvester matrix for polynomials
    A(x)=a0 + a1 x + ... + am x^m, B(x)=b0 + ... + bn x^n.
    """
    m = len(a) - 1
    n = len(b) - 1
    size = m + n
    S = np.zeros((size, size), dtype=float)

    # First n rows: shifted copies of 'a'
    for row in range(n):
        S[row, row : row + m + 1] = a

    # Last m rows: shifted copies of 'b'
    for row in range(m):
        S[n + row, row : row + n + 1] = b

    return S


def compute_sylvester_smin(tr_rational) -> float:
    """
    Compute the smallest singular value of the Sylvester matrix S(P,Q).

    Args:
        tr_rational: TRRational-like object exposing .theta and .phi lists of parameters

    Returns:
        Smallest singular value s_min >= 0 (float). Lower => closer to common factor.
    """
    # Extract coefficients: P(x) = sum_{k=0..d_p} theta_k x^k
    a = _extract_coeffs(getattr(tr_rational, "theta", []))
    if not a:
        return float("nan")

    # Q(x) = 1 + sum_{k=1..d_q} phi_k x^k
    phi_nodes = getattr(tr_rational, "phi", [])
    b = [1.0] + _extract_coeffs(phi_nodes)

    # Remove trailing zeros to get true degrees, but keep at least constant term
    def _trim(c):
        while len(c) > 1 and abs(c[-1]) == 0.0:
            c.pop()
        return c

    a = _trim(a)
    b = _trim(b)
    try:
        S = _sylvester_matrix(a, b)
        # Compute smallest singular value
        s = np.linalg.svd(S, compute_uv=False)
        return float(s[-1])
    except Exception:
        return float("nan")
