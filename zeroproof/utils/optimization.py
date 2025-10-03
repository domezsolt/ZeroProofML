# MIT License
# See LICENSE file in the project root for full license text.
"""
Optimization tools for transreal computations.

This module provides various optimization strategies for improving
performance of transreal arithmetic operations.
"""

import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..autodiff import OpType, TRNode
from ..core import TRScalar, TRTag, ninf, phi, pinf, real


def _as_node(x: Union[TRNode, TRScalar]) -> TRNode:
    """Ensure a TRNode instance (wrap TRScalar as constant node)."""
    if isinstance(x, TRNode):
        return x
    if isinstance(x, TRScalar):
        return TRNode.constant(x)
    raise TypeError(f"Expected TRNode or TRScalar, got {type(x)}")


OPTIMIZATION_AVAILABLE = True


@dataclass
class OptimizationConfig:
    """Configuration for TR optimizations."""

    fuse_operations: bool = True
    eliminate_dead_code: bool = True
    constant_folding: bool = True
    common_subexpression_elimination: bool = True
    memory_pooling: bool = True
    parallel_execution: bool = False
    cache_size: int = 10000
    profile_guided: bool = False


class TROptimizer:
    """Main optimizer for transreal computations."""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self._statistics = defaultdict(int)
        self._node_cache = weakref.WeakValueDictionary()

    def optimize(self, root: Union[TRNode, TRScalar]) -> TRNode:
        """
        Optimize a computational graph.

        Args:
            root: Root node of the graph

        Returns:
            Optimized root node
        """
        root = _as_node(root)

        if self.config.constant_folding:
            root = self._constant_fold(root)

        if self.config.common_subexpression_elimination:
            root = self._eliminate_common_subexpressions(root)

        if self.config.fuse_operations:
            root = self._fuse_operations(root)

        if self.config.eliminate_dead_code:
            root = self._eliminate_dead_code(root)

        return root

    def _constant_fold(self, node: TRNode) -> TRNode:
        """Fold constant expressions."""
        node = _as_node(node)
        if not node._grad_info or not node._grad_info.inputs:
            return node

        # Check if all inputs are constants
        inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]
        if not inputs:
            return node

        all_constant = all(not inp.requires_grad and not inp._grad_info for inp in inputs)

        if all_constant:
            # This is a constant expression, replace with constant node
            self._statistics["constant_folded"] += 1
            return TRNode.constant(node.value)

        return node

    def _eliminate_common_subexpressions(self, node: TRNode) -> TRNode:
        """Eliminate common subexpressions."""
        # Build expression signature
        signature = self._get_expression_signature(node)

        if signature in self._node_cache:
            cached = self._node_cache[signature]
            if cached is not None:
                self._statistics["cse_eliminated"] += 1
                return cached

        # Safe-Rewrite: only cache expressions whose subtree is confined to REAL slice
        if self._subtree_is_real_only(node):
            self._node_cache[signature] = node
        return node

    def _fuse_operations(self, node: TRNode) -> TRNode:
        """Fuse compatible operations."""
        # Example: fuse multiple additions into single multi-add
        if node._grad_info and node._grad_info.op_type == OpType.ADD:
            inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]
            if len(inputs) == 2:
                # Safe-Rewrite: only fuse if entire involved subtree is REAL-only
                if all(self._subtree_is_real_only(inp) for inp in inputs):
                    for inp in inputs:
                        if inp._grad_info and inp._grad_info.op_type == OpType.ADD:
                            # Found fusable pattern: (a + b) + c -> add3(a, b, c)
                            self._statistics["operations_fused"] += 1
                            # In a full impl, create fused op; here we just account

        return node

    def _eliminate_dead_code(self, node: TRNode) -> TRNode:
        """Remove dead code paths."""
        # In a real implementation, would track which nodes contribute
        # to the final output and remove those that don't
        return node

    def _get_expression_signature(self, node: Union[TRNode, TRScalar]) -> Tuple:
        """Get unique signature for an expression."""
        if isinstance(node, TRScalar):
            return ("constant", node.value, node.tag)
        if not node._grad_info:
            return ("constant", node.value.value, node.value.tag)

        inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]
        input_sigs = tuple(self._get_expression_signature(inp) for inp in inputs)

        return (node._grad_info.op_type, input_sigs)

    def _subtree_is_real_only(self, node: TRNode) -> bool:
        """Check if the entire subtree evaluates on REAL slice only.

        Conservatively requires that every encountered node has tag REAL and all
        leaves are constants or parameters currently tagged REAL.
        """
        visited = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if id(cur) in visited:
                continue
            visited.add(id(cur))
            if cur.value.tag != TRTag.REAL:
                return False
            if cur._grad_info and cur._grad_info.inputs:
                for ref in cur._grad_info.inputs:
                    inp = ref()
                    if inp is not None:
                        stack.append(inp)
        return True

    def get_statistics(self) -> Dict[str, int]:
        """Get optimization statistics."""
        return dict(self._statistics)


class GraphOptimizer:
    """Optimize computational graphs for transreal operations."""

    def __init__(self):
        self._rewrite_rules = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default graph rewrite rules."""
        # Rule: x + 0 -> x
        self.add_rule(
            lambda n: (
                n._grad_info
                and n._grad_info.op_type == OpType.ADD
                and any(self._is_zero(inp()) for inp in n._grad_info.inputs if inp())
            ),
            lambda n: self._apply_add_zero_rule(n),
        )

        # Rule: x * 1 -> x
        self.add_rule(
            lambda n: (
                n._grad_info
                and n._grad_info.op_type == OpType.MUL
                and any(self._is_one(inp()) for inp in n._grad_info.inputs if inp())
            ),
            lambda n: self._apply_mul_one_rule(n),
        )

        # Rule: x * 0 -> 0 (if x is REAL)
        self.add_rule(
            lambda n: (
                n._grad_info
                and n._grad_info.op_type == OpType.MUL
                and any(self._is_zero(inp()) for inp in n._grad_info.inputs if inp())
            ),
            lambda n: self._apply_mul_zero_rule(n),
        )

    def add_rule(self, pattern: Callable[[TRNode], bool], rewrite: Callable[[TRNode], TRNode]):
        """Add a graph rewrite rule."""
        self._rewrite_rules.append((pattern, rewrite))

    def optimize(self, root: TRNode) -> TRNode:
        """Optimize a computational graph."""
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            root, changed_now = self._apply_rules_recursive(root)
            changed = changed or changed_now
            iterations += 1

        return root

    def _apply_rules_recursive(self, node: TRNode) -> Tuple[TRNode, bool]:
        """Recursively apply rewrite rules."""
        changed = False

        # Apply rules to this node
        for pattern, rewrite in self._rewrite_rules:
            if pattern(node):
                node = rewrite(node)
                changed = True
                break

        return node, changed

    def _is_zero(self, node: Optional[TRNode]) -> bool:
        """Check if node is zero constant."""
        if node is None:
            return False
        return node.value.tag == TRTag.REAL and node.value.value == 0.0 and not node._grad_info

    def _is_one(self, node: Optional[TRNode]) -> bool:
        """Check if node is one constant."""
        if node is None:
            return False
        return node.value.tag == TRTag.REAL and node.value.value == 1.0 and not node._grad_info

    def _apply_add_zero_rule(self, node: TRNode) -> TRNode:
        """Apply x + 0 -> x rule."""
        inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]
        for inp in inputs:
            if not self._is_zero(inp):
                return inp
        return node

    def _apply_mul_one_rule(self, node: TRNode) -> TRNode:
        """Apply x * 1 -> x rule."""
        inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]
        for inp in inputs:
            if not self._is_one(inp):
                return inp
        return node

    def _apply_mul_zero_rule(self, node: TRNode) -> TRNode:
        """Apply x * 0 -> 0 rule (considering TR semantics)."""
        inputs = [inp() for inp in node._grad_info.inputs if inp() is not None]

        # Check TR semantics: 0 * inf = PHI
        for inp in inputs:
            if inp.value.tag in [TRTag.PINF, TRTag.NINF]:
                return TRNode.constant(phi())

        # Otherwise return 0
        return TRNode.constant(real(0.0))


class OperationFuser:
    """Fuse multiple operations into more efficient compound operations."""

    def __init__(self):
        self._fusion_patterns = []
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default fusion patterns."""
        # Pattern: Linear chain a*x + b
        self.add_pattern(
            "linear",
            lambda nodes: self._detect_linear_pattern(nodes),
            lambda nodes: self._fuse_linear_pattern(nodes),
        )

        # Pattern: Polynomial evaluation
        self.add_pattern(
            "polynomial",
            lambda nodes: self._detect_polynomial_pattern(nodes),
            lambda nodes: self._fuse_polynomial_pattern(nodes),
        )

    def add_pattern(
        self,
        name: str,
        detector: Callable[[List[TRNode]], bool],
        fuser: Callable[[List[TRNode]], TRNode],
    ):
        """Add a fusion pattern."""
        self._fusion_patterns.append((name, detector, fuser))

    def fuse(self, nodes: List[TRNode]) -> List[TRNode]:
        """Fuse operations in a list of nodes."""
        fused = []
        i = 0

        while i < len(nodes):
            fused_any = False

            for name, detector, fuser in self._fusion_patterns:
                # Try to match pattern starting at position i
                for length in range(min(5, len(nodes) - i), 1, -1):
                    candidate = nodes[i : i + length]
                    if detector(candidate):
                        fused_node = fuser(candidate)
                        fused.append(fused_node)
                        i += length
                        fused_any = True
                        break

                if fused_any:
                    break

            if not fused_any:
                fused.append(nodes[i])
                i += 1

        return fused

    def _detect_linear_pattern(self, nodes: List[TRNode]) -> bool:
        """Detect a*x + b pattern."""
        if len(nodes) < 2:
            return False

        # Simple detection - would be more sophisticated in practice
        return (
            nodes[0]._grad_info
            and nodes[0]._grad_info.op_type == OpType.MUL
            and nodes[1]._grad_info
            and nodes[1]._grad_info.op_type == OpType.ADD
        )

    def _fuse_linear_pattern(self, nodes: List[TRNode]) -> TRNode:
        """Fuse a*x + b pattern."""
        # In practice, would create a specialized LinearOp node
        return nodes[-1]  # Return last node for now

    def _detect_polynomial_pattern(self, nodes: List[TRNode]) -> bool:
        """Detect polynomial evaluation pattern."""
        # Would detect Horner's method or similar patterns
        return False

    def _fuse_polynomial_pattern(self, nodes: List[TRNode]) -> TRNode:
        """Fuse polynomial evaluation."""
        return nodes[-1]


class MemoryOptimizer:
    """Optimize memory usage in transreal computations."""

    def __init__(self, pool_size: int = 1000):
        """
        Initialize memory optimizer.

        Args:
            pool_size: Size of memory pools
        """
        self.pool_size = pool_size
        self._node_pool = []
        self._array_pools = {}
        self._statistics = defaultdict(int)

    def allocate_node(self) -> TRNode:
        """Allocate a node from pool."""
        if self._node_pool:
            self._statistics["pool_hits"] += 1
            return self._node_pool.pop()
        else:
            self._statistics["pool_misses"] += 1
            return None  # Would create new node

    def release_node(self, node: TRNode):
        """Release node back to pool."""
        if len(self._node_pool) < self.pool_size:
            # Clear node data
            node._gradient = None
            node._grad_info = None
            self._node_pool.append(node)

    def optimize_graph_memory(self, root: TRNode) -> Dict[str, Any]:
        """
        Analyze and optimize memory usage of a graph.

        Returns:
            Memory optimization report
        """
        # Analyze graph
        node_count = self._count_nodes(root)
        memory_estimate = self._estimate_memory(root)

        # Identify nodes that can share memory
        sharing_opportunities = self._find_memory_sharing(root)

        return {
            "node_count": node_count,
            "memory_estimate_mb": memory_estimate / (1024 * 1024),
            "sharing_opportunities": len(sharing_opportunities),
            "potential_savings_mb": self._estimate_savings(sharing_opportunities) / (1024 * 1024),
        }

    def _count_nodes(self, root: TRNode) -> int:
        """Count nodes in graph."""
        visited = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue

            visited.add(id(node))

            if node._grad_info and node._grad_info.inputs:
                for inp_ref in node._grad_info.inputs:
                    inp = inp_ref()
                    if inp is not None:
                        stack.append(inp)

        return len(visited)

    def _estimate_memory(self, root: TRNode) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimates
        node_size = 128  # bytes per node
        value_size = 16  # bytes per TRScalar

        node_count = self._count_nodes(root)
        return node_count * (node_size + value_size)

    def _find_memory_sharing(self, root: TRNode) -> List[Tuple[TRNode, TRNode]]:
        """Find nodes that can share memory."""
        # In practice, would identify nodes with non-overlapping lifetimes
        return []

    def _estimate_savings(self, sharing_opportunities: List) -> int:
        """Estimate memory savings from sharing."""
        return len(sharing_opportunities) * 144  # bytes per shared node


# Utility functions
def optimize_tr_graph(root: TRNode, config: Optional[OptimizationConfig] = None) -> TRNode:
    """
    Optimize a transreal computational graph.

    Args:
        root: Root node of the graph
        config: Optimization configuration

    Returns:
        Optimized root node
    """
    optimizer = TROptimizer(config)
    return optimizer.optimize(root)


def fuse_operations(nodes: List[TRNode]) -> List[TRNode]:
    """
    Fuse compatible operations in a list of nodes.

    Args:
        nodes: List of nodes to fuse

    Returns:
        List with fused operations
    """
    fuser = OperationFuser()
    return fuser.fuse(nodes)


def analyze_memory_usage(root: TRNode) -> Dict[str, Any]:
    """
    Analyze memory usage of a computational graph.

    Args:
        root: Root node of the graph

    Returns:
        Memory analysis report
    """
    optimizer = MemoryOptimizer()
    return optimizer.optimize_graph_memory(root)
