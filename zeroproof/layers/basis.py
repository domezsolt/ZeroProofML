"""
Basis functions for TR-Rational layers.

This module provides various basis functions (monomials, Chebyshev, etc.)
for constructing the polynomials P and Q in rational layers.
"""

from __future__ import annotations

import math
from typing import List, Protocol, Union

from ..autodiff import TRNode
from ..core import TRScalar, real


class Basis(Protocol):
    """Protocol for basis functions."""

    def __call__(self, x: Union[TRScalar, TRNode], degree: int) -> List[Union[TRScalar, TRNode]]:
        """
        Evaluate basis functions up to given degree.

        Args:
            x: Input value (scalar)
            degree: Maximum degree (inclusive)

        Returns:
            List of basis function values [ψ_0(x), ψ_1(x), ..., ψ_degree(x)]
        """
        ...

    @property
    def name(self) -> str:
        """Name of the basis."""
        ...

    @property
    def bound(self) -> float:
        """Upper bound B such that |ψ_k(x)| ≤ B for all k on the domain."""
        ...

    # Optional API: derivative values up to degree
    # Implemented by bases that can provide analytic derivatives.
    def derivative(
        self, x: Union[TRScalar, TRNode], degree: int
    ) -> List[Union[TRScalar, TRNode]]:  # pragma: no cover - optional
        """
        Evaluate derivatives [ψ'_0(x), ψ'_1(x), ..., ψ'_degree(x)].
        Default: not implemented; callers should check hasattr(..., 'derivative').
        """
        raise NotImplementedError


class MonomialBasis:
    """
    Monomial basis: ψ_k(x) = x^k.

    Simple but can be numerically unstable for high degrees.
    Best for low-degree polynomials on bounded domains.
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize monomial basis.

        Args:
            domain: Input domain (a, b)
        """
        self.domain = domain
        self._bound = max(abs(domain[0]), abs(domain[1]))

    def __call__(self, x: Union[TRScalar, TRNode], degree: int) -> List[Union[TRScalar, TRNode]]:
        """Evaluate monomials [1, x, x², ..., x^degree]."""
        if isinstance(x, TRScalar):
            # Use core operations
            from ..core import tr_pow_int

            result = [real(1.0)]  # x^0 = 1
            for k in range(1, degree + 1):
                result.append(tr_pow_int(x, k))
        else:
            # Use autodiff operations
            from ..autodiff import tr_pow_int

            result = [TRNode.constant(real(1.0))]  # x^0 = 1
            for k in range(1, degree + 1):
                result.append(tr_pow_int(x, k))
        return result

    @property
    def name(self) -> str:
        return "monomial"

    @property
    def bound(self) -> float:
        return self._bound

    # Optional analytic derivative for monomials: d/dx x^k = k x^{k-1}
    def derivative(self, x: Union[TRScalar, TRNode], degree: int) -> List[Union[TRScalar, TRNode]]:
        if degree < 0:
            return []
        # ψ'_0 = 0
        if isinstance(x, TRScalar):
            from ..core import tr_pow_int

            derivs: List[Union[TRScalar, TRNode]] = [real(0.0)]
            for k in range(1, degree + 1):
                if k == 1:
                    derivs.append(real(1.0))
                else:
                    derivs.append(tr_pow_int(x, k - 1) * real(float(k)))
            return derivs
        else:
            from ..autodiff import tr_pow_int

            derivs = [TRNode.constant(real(0.0))]
            for k in range(1, degree + 1):
                if k == 1:
                    derivs.append(TRNode.constant(real(1.0)))
                else:
                    derivs.append(tr_pow_int(x, k - 1) * TRNode.constant(real(float(k))))
            return derivs


class ChebyshevBasis:
    """
    Chebyshev polynomials of the first kind.

    Defined by recurrence:
    - T_0(x) = 1
    - T_1(x) = x
    - T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

    Optimal for approximation on [-1, 1]. For other domains,
    use linear transformation.
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0), degree: int | None = None):
        """
        Initialize Chebyshev basis.

        Args:
            domain: Input domain (a, b), will be mapped to [-1, 1]
            degree: Optional degree hint for convenience; not required and not stored.
        """
        self.domain = domain
        self.a, self.b = domain
        # Chebyshev polynomials are bounded by 1 on [-1, 1]
        self._bound = 1.0

    def _transform_to_standard(self, x: Union[TRScalar, TRNode]) -> Union[TRScalar, TRNode]:
        """Transform x from [a, b] to [-1, 1]."""
        # t = 2(x - a)/(b - a) - 1
        if isinstance(x, TRScalar):
            from ..core import tr_div, tr_mul, tr_sub

            two = real(2.0)
            width = real(self.b - self.a)
            shifted = tr_sub(x, real(self.a))
            scaled = tr_div(tr_mul(two, shifted), width)
            return tr_sub(scaled, real(1.0))
        else:
            two = TRNode.constant(real(2.0))
            width = TRNode.constant(real(self.b - self.a))
            shifted = x - TRNode.constant(real(self.a))
            scaled = (two * shifted) / width
            return scaled - TRNode.constant(real(1.0))

    def __call__(self, x: Union[TRScalar, TRNode], degree: int) -> List[Union[TRScalar, TRNode]]:
        """Evaluate Chebyshev polynomials using recurrence."""
        # Transform to standard domain if needed
        if self.domain != (-1.0, 1.0):
            t = self._transform_to_standard(x)
        else:
            t = x

        if degree < 0:
            return []

        # Initialize recurrence
        if isinstance(t, TRScalar):
            from ..core import tr_mul, tr_sub

            T = [real(1.0)]  # T_0 = 1

            if degree >= 1:
                T.append(t)  # T_1 = x

            # Recurrence: T_{n+1} = 2x T_n - T_{n-1}
            for n in range(1, degree):
                two_t = tr_mul(real(2.0), t)
                next_T = tr_sub(tr_mul(two_t, T[n]), T[n - 1])
                T.append(next_T)
        else:
            T = [TRNode.constant(real(1.0))]  # T_0 = 1

            if degree >= 1:
                T.append(t)  # T_1 = x

            # Recurrence: T_{n+1} = 2x T_n - T_{n-1}
            two = TRNode.constant(real(2.0))
            for n in range(1, degree):
                two_t = two * t
                next_T = two_t * T[n] - T[n - 1]
                T.append(next_T)

        return T

    @property
    def name(self) -> str:
        return "chebyshev"

    @property
    def bound(self) -> float:
        return self._bound


class FourierBasis:
    """
    Fourier basis: combinations of sin and cos.

    Basis functions:
    - ψ_0(x) = 1
    - ψ_{2k-1}(x) = cos(kπx/L) for k ≥ 1
    - ψ_{2k}(x) = sin(kπx/L) for k ≥ 1

    where L is half the domain width.
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize Fourier basis.

        Args:
            domain: Input domain (a, b)
        """
        self.domain = domain
        self.a, self.b = domain
        self.L = (self.b - self.a) / 2.0
        # sin and cos are bounded by 1
        self._bound = 1.0

    def __call__(self, x: Union[TRScalar, TRNode], degree: int) -> List[Union[TRScalar, TRNode]]:
        """Evaluate Fourier basis functions."""
        raise NotImplementedError(
            "Fourier basis requires sin/cos operations not yet implemented in TR"
        )

    @property
    def name(self) -> str:
        return "fourier"

    @property
    def bound(self) -> float:
        return self._bound


def create_basis(name: str, domain: tuple[float, float] = (-1.0, 1.0)) -> Basis:
    """
    Create a basis function by name.

    Args:
        name: Name of the basis ("monomial", "chebyshev", "fourier")
        domain: Domain for the basis

    Returns:
        Basis instance

    Raises:
        ValueError: If basis name is not recognized
    """
    basis_map = {
        "monomial": MonomialBasis,
        "chebyshev": ChebyshevBasis,
        "fourier": FourierBasis,
    }

    if name not in basis_map:
        raise ValueError(f"Unknown basis: {name}. Choose from {list(basis_map.keys())}")

    return basis_map[name](domain)
