"""
Gradient-aware transreal operations.

This module provides lifted versions of transreal operations that work
with TRNodes and automatically track operations for differentiation.
"""

from __future__ import annotations

import weakref
from typing import Optional, Union

from ..core import TRScalar, real
from ..core import tr_abs as core_abs
from ..core import tr_add as core_add
from ..core import tr_div as core_div
from ..core import tr_log as core_log
from ..core import tr_mul as core_mul
from ..core import tr_neg as core_neg
from ..core import tr_pow_int as core_pow_int
from ..core import tr_sign as core_sign
from ..core import tr_sqrt as core_sqrt
from ..core import tr_sub as core_sub
from .tr_node import GradientInfo, OpType, TRNode

# Type alias for values that can be converted to nodes
NodeLike = Union[TRNode, TRScalar, float, int]


def _ensure_node(x: NodeLike) -> TRNode:
    """Convert a value to a TRNode if needed."""
    if isinstance(x, TRNode):
        return x
    elif isinstance(x, TRScalar):
        return TRNode.constant(x)
    elif isinstance(x, (int, float)):
        return TRNode.constant(real(float(x)))
    else:
        raise TypeError(f"Cannot convert {type(x)} to TRNode")


def _create_result_node(
    value: TRScalar, op_type: OpType, inputs: list[TRNode], extra_data: Optional[dict] = None
) -> TRNode:
    """Create a result node and record operation metadata.

    We always attach GradientInfo (for profiling/graph analysis),
    but preserve requires_grad based on inputs so backprop only
    accumulates gradients for trainable paths.
    """
    # Determine if gradients should be tracked for this result
    requires_grad = any(inp.requires_grad for inp in inputs)

    # Create gradient info for graph/profiling introspection
    grad_info = GradientInfo(
        op_type=op_type,
        inputs=[weakref.ref(inp) for inp in inputs],
        forward_value=value,
        extra_data=extra_data or {},
    )
    # Keep strong references to ALL inputs to prevent premature GC of the graph
    grad_info._input_cache = list(inputs)

    return TRNode(value, requires_grad=requires_grad, grad_info=grad_info)


# Binary operations


def tr_add(a: NodeLike, b: NodeLike) -> TRNode:
    """
    Transreal addition with gradient tracking.

    Args:
        a, b: Values to add (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    a_node = _ensure_node(a)
    b_node = _ensure_node(b)

    # Compute forward value
    result_value = core_add(a_node.value, b_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.ADD, [a_node, b_node])


def tr_sub(a: NodeLike, b: NodeLike) -> TRNode:
    """
    Transreal subtraction with gradient tracking.

    Args:
        a, b: Values for a - b (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    a_node = _ensure_node(a)
    b_node = _ensure_node(b)

    # Compute forward value
    result_value = core_sub(a_node.value, b_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.SUB, [a_node, b_node])


def tr_mul(a: NodeLike, b: NodeLike) -> TRNode:
    """
    Transreal multiplication with gradient tracking.

    Args:
        a, b: Values to multiply (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    a_node = _ensure_node(a)
    b_node = _ensure_node(b)

    # Compute forward value
    result_value = core_mul(a_node.value, b_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.MUL, [a_node, b_node])


def tr_div(a: NodeLike, b: NodeLike) -> TRNode:
    """
    Transreal division with gradient tracking.

    Args:
        a, b: Values for a / b (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    a_node = _ensure_node(a)
    b_node = _ensure_node(b)

    # Compute forward value
    result_value = core_div(a_node.value, b_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.DIV, [a_node, b_node])


# Unary operations


def tr_neg(x: NodeLike) -> TRNode:
    """
    Transreal negation with gradient tracking.

    Args:
        x: Value to negate (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_neg(x_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.NEG, [x_node])


def tr_abs(x: NodeLike) -> TRNode:
    """
    Transreal absolute value with gradient tracking.

    Args:
        x: Value (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_abs(x_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.ABS, [x_node])


def tr_sign(x: NodeLike) -> TRNode:
    """
    Transreal sign function with gradient tracking.

    Note: The derivative of sign is 0 almost everywhere,
    undefined at 0. We use 0 for the gradient.

    Args:
        x: Value (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_sign(x_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.SIGN, [x_node])


def tr_log(x: NodeLike) -> TRNode:
    """
    Transreal natural logarithm with gradient tracking.

    Args:
        x: Value (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_log(x_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.LOG, [x_node])


def tr_sqrt(x: NodeLike) -> TRNode:
    """
    Transreal square root with gradient tracking.

    Args:
        x: Value (TRNode, TRScalar, or numeric)

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_sqrt(x_node.value)

    # Create result node with gradient tracking
    return _create_result_node(result_value, OpType.SQRT, [x_node])


def tr_pow_int(x: NodeLike, n: int) -> TRNode:
    """
    Transreal integer power with gradient tracking.

    Args:
        x: Base value (TRNode, TRScalar, or numeric)
        n: Integer exponent

    Returns:
        TRNode containing the result
    """
    x_node = _ensure_node(x)

    # Compute forward value
    result_value = core_pow_int(x_node.value, n)

    # Create result node with gradient tracking
    extra_data = {"exponent": n}
    return _create_result_node(result_value, OpType.POW_INT, [x_node], extra_data)


# Operator overloading for TRNode


def _add_operators_to_node():
    """Add operator overloading to TRNode class."""

    def __add__(self, other):
        return tr_add(self, other)

    def __radd__(self, other):
        return tr_add(other, self)

    def __sub__(self, other):
        return tr_sub(self, other)

    def __rsub__(self, other):
        return tr_sub(other, self)

    def __mul__(self, other):
        return tr_mul(self, other)

    def __rmul__(self, other):
        return tr_mul(other, self)

    def __truediv__(self, other):
        return tr_div(self, other)

    def __rtruediv__(self, other):
        return tr_div(other, self)

    def __neg__(self):
        return tr_neg(self)

    def __abs__(self):
        return tr_abs(self)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("TRNode power requires integer exponent")
        return tr_pow_int(self, n)

    # Add methods to TRNode
    TRNode.__add__ = __add__
    TRNode.__radd__ = __radd__
    TRNode.__sub__ = __sub__
    TRNode.__rsub__ = __rsub__
    TRNode.__mul__ = __mul__
    TRNode.__rmul__ = __rmul__
    TRNode.__truediv__ = __truediv__
    TRNode.__rtruediv__ = __rtruediv__
    TRNode.__neg__ = __neg__
    TRNode.__abs__ = __abs__
    TRNode.__pow__ = __pow__


# Initialize operator overloading
_add_operators_to_node()
