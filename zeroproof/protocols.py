# MIT License
# See LICENSE file in the project root for full license text.
"""
Typed Protocols for the public API.

These lightweight interfaces let users type-hint against the library
without importing heavy optional deps (e.g., torch/jax).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:  # Import only for type checkers to avoid runtime deps
    from .autodiff.tr_node import TRNode
    from .core.tr_scalar import TRTag


@runtime_checkable
class PolicyLike(Protocol):
    """Minimal interface for TR policy-like objects.

    Matches the attributes commonly used by layers/policies without
    requiring a concrete implementation.
    """

    tau_Q_on: float
    tau_Q_off: float
    tau_P_on: float
    tau_P_off: float
    keep_signed_zero: bool
    deterministic_reduction: bool
    softmax_one_hot_infinity: bool


@runtime_checkable
class ForwardModel(Protocol):
    """Model exposing a TR forward pass.

    Implementations should accept a `TRNode` and return `(value, tag)`
    as a pair of `TRNode` and `TRTag` respectively.
    """

    def forward(self, x: "TRNode") -> Tuple["TRNode", "TRTag"]:  # noqa: F821
        ...
