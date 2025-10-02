"""
High-level gradient computation functions.

This module provides convenient functions for computing gradients
of transreal computations with the Mask-REAL rule.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

from .gradient_tape import gradient_tape
from .tr_node import TRNode


def tr_grad(func: Callable[..., TRNode], argnums: Union[int, List[int]] = 0) -> Callable:
    """
    Create a function that computes gradients of func.

    Args:
        func: Function that takes TRNode inputs and returns a TRNode output
        argnums: Which arguments to differentiate with respect to
                (0-indexed, can be int or list of ints)

    Returns:
        Function that returns gradients

    Example:
        def f(x):
            return x * x + 2 * x + 1

        grad_f = tr_grad(f)
        x = TRNode.parameter(real(3.0))
        df_dx = grad_f(x)  # Returns gradient at x=3
    """
    if isinstance(argnums, int):
        argnums = [argnums]

    def grad_func(*args, **kwargs):
        # Convert args to nodes and mark for gradient computation
        node_args = []
        sources = []

        for i, arg in enumerate(args):
            if isinstance(arg, TRNode):
                # Create a fresh copy to avoid gradient interference
                node = TRNode.parameter(arg.value)
            else:
                # Convert to node
                from ..core import real

                if isinstance(arg, (int, float)):
                    node = TRNode.parameter(real(float(arg)))
                else:
                    node = TRNode.parameter(arg)

            node_args.append(node)

            # Track nodes we want gradients for
            if i in argnums:
                sources.append(node)

        # Record computation
        with gradient_tape() as tape:
            # Watch the source nodes
            for source in sources:
                tape.watch(source)

            # Run the function
            result = func(*node_args, **kwargs)

        # Compute gradients
        grads = tape.gradient(result, sources)

        # Return single gradient or list
        if len(argnums) == 1:
            return grads[0]
        else:
            return grads

    return grad_func


def tr_value_and_grad(func: Callable[..., TRNode], argnums: Union[int, List[int]] = 0) -> Callable:
    """
    Create a function that computes both value and gradients.

    Args:
        func: Function that takes TRNode inputs and returns a TRNode output
        argnums: Which arguments to differentiate with respect to

    Returns:
        Function that returns (value, gradients) tuple

    Example:
        def f(x):
            return x * x * x

        value_and_grad_f = tr_value_and_grad(f)
        x = TRNode.parameter(real(2.0))
        val, grad = value_and_grad_f(x)  # val = 8, grad = 12
    """
    if isinstance(argnums, int):
        argnums = [argnums]
        single_grad = True
    else:
        single_grad = False

    def value_and_grad_func(*args, **kwargs):
        # Convert args to nodes and mark for gradient computation
        node_args = []
        sources = []

        for i, arg in enumerate(args):
            if isinstance(arg, TRNode):
                # Create a fresh copy to avoid gradient interference
                node = TRNode.parameter(arg.value)
            else:
                # Convert to node
                from ..core import real

                if isinstance(arg, (int, float)):
                    node = TRNode.parameter(real(float(arg)))
                else:
                    node = TRNode.parameter(arg)

            node_args.append(node)

            # Track nodes we want gradients for
            if i in argnums:
                sources.append(node)

        # Record computation
        with gradient_tape() as tape:
            # Watch the source nodes
            for source in sources:
                tape.watch(source)

            # Run the function
            result = func(*node_args, **kwargs)

        # Compute gradients
        grads = tape.gradient(result, sources)

        # Return value and gradients
        if single_grad:
            return result, grads[0]
        else:
            return result, grads

    return value_and_grad_func


def check_gradient(
    func: Callable[..., TRNode], x: Union[TRNode, float], eps: float = 1e-5, argnum: int = 0
) -> Tuple[Optional[TRNode], Optional[float], Optional[float]]:
    """
    Check gradient computation using finite differences.

    This is useful for testing gradient implementations.
    Only works for REAL-valued functions and inputs.

    Args:
        func: Function to check
        x: Point at which to check gradient
        eps: Finite difference step size
        argnum: Which argument to differentiate

    Returns:
        Tuple of (analytical_grad, numerical_grad, relative_error)
        Returns None values if gradient cannot be computed
    """
    from ..core import TRTag, real

    # Ensure x is a node
    if not isinstance(x, TRNode):
        x = TRNode.parameter(real(float(x)))

    # Check that x is REAL
    if x.tag != TRTag.REAL:
        return None, None, None

    # Compute analytical gradient
    grad_func = tr_grad(func, argnums=argnum)
    try:
        analytical = grad_func(x)
        if analytical is None or analytical.tag != TRTag.REAL:
            return analytical, None, None
    except Exception:
        return None, None, None

    # Compute numerical gradient using finite differences
    x_val = x.value.value

    # Create perturbed inputs
    x_plus = TRNode.parameter(real(x_val + eps))
    x_minus = TRNode.parameter(real(x_val - eps))

    # Evaluate function at perturbed points
    f_plus = func(x_plus)
    f_minus = func(x_minus)

    # Check that both are REAL
    if f_plus.tag != TRTag.REAL or f_minus.tag != TRTag.REAL:
        return analytical, None, None

    # Compute finite difference
    numerical = (f_plus.value.value - f_minus.value.value) / (2 * eps)

    # Compute relative error
    analytical_val = analytical.value.value
    if abs(analytical_val) > 1e-10:
        rel_error = abs(analytical_val - numerical) / abs(analytical_val)
    else:
        rel_error = abs(analytical_val - numerical)

    return analytical, numerical, rel_error
