"""
Gradient tape for recording transreal operations.

This module provides a context manager that records operations
for automatic differentiation with the Mask-REAL rule.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Iterable, List, Optional, Set

from .tr_node import TRNode


class TRGradientTape:
    """
    Context manager for recording transreal operations.

    The gradient tape records operations performed on TRNodes
    to enable automatic differentiation with backpropagation.
    """

    def __init__(self, persistent: bool = False):
        """
        Initialize a gradient tape.

        Args:
            persistent: If True, tape can be used multiple times.
                       If False, tape is cleared after first backward.
        """
        self._persistent = persistent
        self._recorded_ops: List[weakref.ref[TRNode]] = []
        self._watched_nodes: Set[int] = set()  # IDs of watched nodes
        self._is_recording = False
        self._used = False

    def __enter__(self) -> TRGradientTape:
        """Enter the gradient tape context."""
        # Push this tape onto the stack
        TRNode._gradient_tape_stack.append(self)
        self._is_recording = True
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Exit the gradient tape context."""
        # Pop this tape from the stack
        if TRNode._gradient_tape_stack and TRNode._gradient_tape_stack[-1] is self:
            TRNode._gradient_tape_stack.pop()
        self._is_recording = False

    def watch(self, node: TRNode) -> None:
        """
        Watch a node for gradient computation.

        By default, only nodes created within the tape context
        are tracked. Use this to track pre-existing nodes.

        Args:
            node: The node to watch
        """
        self._watched_nodes.add(id(node))

    def _record_operation(self, node: TRNode) -> None:
        """
        Record an operation node.

        Called automatically when operations create new nodes.

        Args:
            node: The node representing the operation result
        """
        if self._is_recording:
            self._recorded_ops.append(weakref.ref(node))

    def gradient(
        self,
        target: TRNode,
        sources: Iterable[TRNode] | TRNode,
        output_gradients: Optional[List[TRNode]] = None,
    ) -> List[Optional[TRNode]]:
        """
        Compute gradients of target with respect to sources.

        Args:
            target: The node to differentiate
            sources: List of nodes to compute gradients for
            output_gradients: Optional gradients for target (default: 1.0)

        Returns:
            List of gradient nodes corresponding to sources

        Raises:
            RuntimeError: If tape has been used and is not persistent
        """
        if self._used and not self._persistent:
            raise RuntimeError(
                "Gradient tape has already been used. " "Set persistent=True to use multiple times."
            )

        # Normalize sources to list
        if isinstance(sources, TRNode):
            sources_list: List[TRNode] = [sources]
        else:
            sources_list = list(sources)

        # Mark tape as used
        self._used = True

        # Clear any existing gradients
        from .backward import topological_sort

        nodes = topological_sort(target)
        for node in nodes:
            if node.requires_grad:
                node._gradient = None

        # Run backward pass
        from .backward import backward_pass

        if output_gradients is not None and len(output_gradients) > 0:
            backward_pass(target, output_gradients[0].value)
        else:
            backward_pass(target)

        # Collect gradients for requested sources
        gradients = []
        for source in sources_list:
            if source.gradient is not None:
                # Wrap gradient in a node
                grad_node = TRNode.constant(source.gradient)
                gradients.append(grad_node)
            else:
                # If no gradient was computed but node requires grad, it means gradient is zero
                if source.requires_grad:
                    from ..core import real

                    gradients.append(TRNode.constant(real(0.0)))
                else:
                    gradients.append(None)

        return gradients

    def reset(self) -> None:
        """Reset the tape for reuse (only if persistent)."""
        if not self._persistent:
            raise RuntimeError("Cannot reset non-persistent tape")
        self._recorded_ops.clear()
        self._watched_nodes.clear()
        self._used = False


@contextmanager
def gradient_tape(persistent: bool = False):
    """
    Context manager for gradient tape.

    Example:
        with gradient_tape() as tape:
            x = TRNode.parameter(real(2.0))
            y = x * x  # Will use overloaded operators

        grad = tape.gradient(y, [x])

    Args:
        persistent: If True, tape can be used multiple times

    Yields:
        TRGradientTape instance
    """
    tape = TRGradientTape(persistent=persistent)
    with tape:
        yield tape
