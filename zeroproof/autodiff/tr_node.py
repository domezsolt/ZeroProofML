"""
Transreal computational graph node for autodifferentiation.

This module implements nodes that track transreal values through computations
and enable gradient computation with the Mask-REAL rule.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from ..core import TRScalar, TRTag


class OpType(Enum):
    """Types of operations in the computational graph."""

    # Leaf nodes
    CONSTANT = auto()
    PARAMETER = auto()

    # Binary operations
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    # Unary operations
    NEG = auto()
    ABS = auto()
    SIGN = auto()
    LOG = auto()
    SQRT = auto()
    POW_INT = auto()

    # Special
    CLONE = auto()


@dataclass
class GradientInfo:
    """Information needed for gradient computation."""

    op_type: OpType
    inputs: List[weakref.ref]  # Weak references to input nodes
    forward_value: TRScalar
    extra_data: Dict[str, Any] = field(default_factory=dict)
    _input_cache: List[Any] = field(default_factory=list)  # Strong refs to prevent GC


class TRNode:
    """
    Node in the transreal computational graph.

    Each node represents a transreal value and tracks operations
    for automatic differentiation with the Mask-REAL rule.
    """

    # Class variable to track if we're in gradient tape context
    _gradient_tape_stack: List[Any] = []

    def __init__(
        self, value: TRScalar, requires_grad: bool = False, grad_info: Optional[GradientInfo] = None
    ):
        """
        Initialize a TRNode.

        Args:
            value: The transreal scalar value
            requires_grad: Whether to track gradients for this node
            grad_info: Information for gradient computation
        """
        self._value = value
        self._requires_grad = requires_grad
        self._grad_info = grad_info
        self._gradient: Optional[TRScalar] = None
        self._name: Optional[str] = None

        # Register with current gradient tape if any
        if self._gradient_tape_stack and grad_info is not None:
            self._gradient_tape_stack[-1]._record_operation(self)

    @property
    def value(self) -> TRScalar:
        """Get the transreal value."""
        return self._value

    @value.setter
    def value(self, new_value: TRScalar) -> None:
        """Allow updating the value for parameter/constant nodes in tests."""
        if not isinstance(new_value, TRScalar):
            from ..core import real as tr_real

            new_value = tr_real(float(new_value))
        self._value = new_value

    @property
    def tag(self) -> TRTag:
        """Get the transreal tag."""
        return self._value.tag

    @property
    def requires_grad(self) -> bool:
        """Check if gradients are tracked."""
        return self._requires_grad

    @property
    def gradient(self) -> Optional[TRScalar]:
        """Get the accumulated gradient."""
        return self._gradient

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no grad_info)."""
        return self._grad_info is None

    @property
    def name(self) -> Optional[str]:
        """Get the node name (for debugging)."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the node name."""
        self._name = value

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        if self._requires_grad:
            from ..core import real

            self._gradient = real(0.0)

    def accumulate_grad(self, grad: TRScalar) -> None:
        """
        Accumulate gradient value.

        Args:
            grad: Gradient to accumulate
        """
        if not self._requires_grad:
            return

        if self._gradient is None:
            self._gradient = grad
        else:
            # Use TR addition for gradient accumulation
            from ..core import tr_add

            self._gradient = tr_add(self._gradient, grad)

    def backward(self, grad_output: Optional[TRScalar] = None) -> None:
        """
        Compute gradients using backpropagation with Mask-REAL rule.

        Args:
            grad_output: Gradient from downstream. Defaults to real(1.0)
        """
        # Use the centralized backward pass implementation
        from .backward import backward_pass

        backward_pass(self, grad_output)

    def __repr__(self) -> str:
        """String representation for debugging."""
        name_str = f" '{self._name}'" if self._name else ""
        grad_str = " grad" if self._requires_grad else ""
        return f"TRNode({self._value}{name_str}{grad_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return str(self._value)

    def detach(self) -> TRNode:
        """Create a new node with the same value but no gradient tracking."""
        return TRNode(self._value, requires_grad=False)

    @classmethod
    def constant(cls, value: TRScalar) -> TRNode:
        """Create a constant node (no gradient tracking)."""
        return cls(value, requires_grad=False)

    @classmethod
    def parameter(cls, value: TRScalar, name: Optional[str] = None) -> TRNode:
        """Create a parameter node (with gradient tracking)."""
        node = cls(value, requires_grad=True)
        if name:
            node.name = name
        return node

    # Arithmetic operators
    def __add__(self, other):
        """Addition operator."""
        from .tr_ops_grad import tr_add

        return tr_add(self, other)

    def __radd__(self, other):
        """Right addition operator."""
        from .tr_ops_grad import tr_add

        return tr_add(other, self)

    def __sub__(self, other):
        """Subtraction operator."""
        from .tr_ops_grad import tr_sub

        return tr_sub(self, other)

    def __rsub__(self, other):
        """Right subtraction operator."""
        from .tr_ops_grad import tr_sub

        return tr_sub(other, self)

    def __mul__(self, other):
        """Multiplication operator."""
        from .tr_ops_grad import tr_mul

        return tr_mul(self, other)

    def __rmul__(self, other):
        """Right multiplication operator."""
        from .tr_ops_grad import tr_mul

        return tr_mul(other, self)

    def __truediv__(self, other):
        """Division operator."""
        from .tr_ops_grad import tr_div

        return tr_div(self, other)

    def __rtruediv__(self, other):
        """Right division operator."""
        from .tr_ops_grad import tr_div

        return tr_div(other, self)

    def __neg__(self):
        """Negation operator."""
        from .tr_ops_grad import tr_neg

        return tr_neg(self)

    def __abs__(self):
        """Absolute value operator."""
        from .tr_ops_grad import tr_abs

        return tr_abs(self)

    def __pow__(self, exponent):
        """Power operator (integer exponents only)."""
        from .tr_ops_grad import tr_pow_int

        if not isinstance(exponent, int):
            raise TypeError(f"TRNode power only supports integer exponents, got {type(exponent)}")
        return tr_pow_int(self, exponent)
