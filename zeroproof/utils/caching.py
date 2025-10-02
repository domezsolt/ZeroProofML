"""
Caching utilities for transreal computations.

This module provides caching mechanisms to improve performance
by avoiding redundant computations.
"""

import functools
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ..autodiff import TRNode
from ..core import TRScalar, TRTag

CACHING_AVAILABLE = True


@dataclass
class CacheEntry:
    """Entry in the cache."""

    value: Any
    size: int
    hits: int = 0
    last_access: float = 0.0
    compute_time: float = 0.0


class TRCache:
    """
    Cache for transreal computations with various eviction policies.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 100.0,
        eviction_policy: str = "lru",
        ttl_seconds: Optional[float] = None,
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            eviction_policy: One of 'lru', 'lfu', 'fifo'
            ttl_seconds: Time to live for entries (None = no expiry)
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_used = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if self.ttl_seconds is not None:
                if time.time() - entry.last_access > self.ttl_seconds:
                    self._evict_entry(key)
                    self._misses += 1
                    return None

            # Update access info
            entry.hits += 1
            entry.last_access = time.time()
            self._hits += 1

            # Move to end for LRU
            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)

            return entry.value

    def put(self, key: str, value: Any, size: Optional[int] = None, compute_time: float = 0.0):
        """Put value in cache."""
        with self._lock:
            # Estimate size if not provided
            if size is None:
                size = self._estimate_size(value)

            # Check if we need to evict
            while (
                len(self._cache) >= self.max_size
                or self._memory_used + size > self.max_memory_bytes
            ):
                if not self._evict_one():
                    break  # Can't evict anymore

            # Add entry
            entry = CacheEntry(
                value=value, size=size, last_access=time.time(), compute_time=compute_time
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._memory_used -= old_entry.size

            self._cache[key] = entry
            self._memory_used += size

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._memory_used = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            # Compute total compute time saved
            total_saved = sum(entry.compute_time * entry.hits for entry in self._cache.values())

            return {
                "size": len(self._cache),
                "memory_used_mb": self._memory_used / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "compute_time_saved": total_saved,
            }

    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False

        if self.eviction_policy == "lru":
            # Evict least recently used (first item)
            key = next(iter(self._cache))
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
        elif self.eviction_policy == "fifo":
            # Evict first in (first item)
            key = next(iter(self._cache))
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        self._evict_entry(key)
        return True

    def _evict_entry(self, key: str):
        """Evict a specific entry."""
        if key in self._cache:
            entry = self._cache[key]
            self._memory_used -= entry.size
            del self._cache[key]
            self._evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        if isinstance(value, TRScalar):
            return 24  # Approximate size
        elif isinstance(value, TRNode):
            return 128  # Approximate size including grad info
        else:
            try:
                return len(pickle.dumps(value))
            except:
                return 100  # Default estimate


# Global cache instance
_global_cache = TRCache()


def memoize_tr(
    cache: Optional[TRCache] = None,
    key_func: Optional[Callable] = None,
    ttl_seconds: Optional[float] = None,
):
    """
    Decorator to memoize transreal computations.

    Args:
        cache: Cache instance (uses global if None)
        key_func: Function to compute cache key from arguments
        ttl_seconds: Time to live for cached results
    """

    def decorator(func: Callable) -> Callable:
        nonlocal cache
        if cache is None:
            cache = _global_cache

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _default_key_func(func.__name__, args, kwargs)

            # Try to get from cache
            start_time = time.time()
            result = cache.get(key)

            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)
            compute_time = time.time() - start_time

            # Store in cache
            cache.put(key, result, compute_time=compute_time)

            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: cache.get_statistics()

        return wrapper

    return decorator


def cached_operation(op_type: str, inputs: Tuple[TRScalar, ...]) -> Optional[TRScalar]:
    """
    Check if an operation result is cached.

    Args:
        op_type: Operation type
        inputs: Input values

    Returns:
        Cached result or None
    """
    key = _operation_key(op_type, inputs)
    return _global_cache.get(key)


def cache_operation_result(op_type: str, inputs: Tuple[TRScalar, ...], result: TRScalar):
    """
    Cache an operation result.

    Args:
        op_type: Operation type
        inputs: Input values
        result: Result to cache
    """
    key = _operation_key(op_type, inputs)
    _global_cache.put(key, result)


def clear_cache():
    """Clear the global cache."""
    _global_cache.clear()


def cache_statistics() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_statistics()


def _default_key_func(func_name: str, args: tuple, kwargs: dict) -> str:
    """Default key function for memoization."""
    # Convert args to cacheable format
    key_parts = [func_name]

    for arg in args:
        if isinstance(arg, TRScalar):
            key_parts.append(f"{arg.value}:{arg.tag.name}")
        elif isinstance(arg, TRNode):
            key_parts.append(f"node:{id(arg)}")
        else:
            key_parts.append(str(arg))

    # Add kwargs
    for k, v in sorted(kwargs.items()):
        if isinstance(v, TRScalar):
            key_parts.append(f"{k}={v.value}:{v.tag.name}")
        else:
            key_parts.append(f"{k}={v}")

    return "|".join(key_parts)


def _operation_key(op_type: str, inputs: Tuple[TRScalar, ...]) -> str:
    """Generate cache key for an operation."""
    parts = [op_type]
    for inp in inputs:
        parts.append(f"{inp.value}:{inp.tag.name}")
    return "|".join(parts)


class ResultCache:
    """
    Specialized cache for computational results with dependency tracking.
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize result cache.

        Args:
            max_entries: Maximum number of cached results
        """
        self.max_entries = max_entries
        self._cache: Dict[int, Tuple[TRNode, Any]] = {}
        self._dependencies: Dict[int, set] = {}
        self._dependents: Dict[int, set] = {}
        self._lock = threading.RLock()

    def get_or_compute(self, node: TRNode, compute_func: Callable[[TRNode], Any]) -> Any:
        """
        Get cached result or compute it.

        Args:
            node: Node to compute
            compute_func: Function to compute result

        Returns:
            Computed or cached result
        """
        node_id = id(node)

        with self._lock:
            # Check cache
            if node_id in self._cache:
                cached_node, result = self._cache[node_id]
                # Verify node hasn't changed
                if self._node_unchanged(node, cached_node):
                    return result

            # Compute result
            result = compute_func(node)

            # Cache result
            self._cache_result(node, result)

            return result

    def invalidate(self, node: TRNode):
        """Invalidate cache for a node and its dependents."""
        node_id = id(node)

        with self._lock:
            to_invalidate = {node_id}
            processed = set()

            # Find all dependents
            while to_invalidate:
                current = to_invalidate.pop()
                if current in processed:
                    continue

                processed.add(current)

                # Remove from cache
                if current in self._cache:
                    del self._cache[current]

                # Add dependents
                if current in self._dependents:
                    to_invalidate.update(self._dependents[current])
                    del self._dependents[current]

                # Clean up dependencies
                if current in self._dependencies:
                    for dep in self._dependencies[current]:
                        if dep in self._dependents:
                            self._dependents[dep].discard(current)
                    del self._dependencies[current]

    def _cache_result(self, node: TRNode, result: Any):
        """Cache a computation result."""
        node_id = id(node)

        # Evict if needed
        if len(self._cache) >= self.max_entries:
            # Evict node with fewest dependents
            evict_id = min(self._cache.keys(), key=lambda k: len(self._dependents.get(k, set())))
            self.invalidate(TRNode.constant(TRScalar(0, TRTag.REAL)))  # Dummy node

        # Store result
        self._cache[node_id] = (node, result)

        # Track dependencies
        if node._grad_info and node._grad_info.inputs:
            deps = set()
            for inp_ref in node._grad_info.inputs:
                inp = inp_ref()
                if inp is not None:
                    inp_id = id(inp)
                    deps.add(inp_id)

                    # Update dependents
                    if inp_id not in self._dependents:
                        self._dependents[inp_id] = set()
                    self._dependents[inp_id].add(node_id)

            self._dependencies[node_id] = deps

    def _node_unchanged(self, node1: TRNode, node2: TRNode) -> bool:
        """Check if two nodes are effectively the same."""
        # Simple check - could be more sophisticated
        return node1.value == node2.value and node1.requires_grad == node2.requires_grad


# Specialized caches for common operations
class OperationCache:
    """Cache for specific transreal operations."""

    def __init__(self):
        self._add_cache = TRCache(max_size=1000)
        self._mul_cache = TRCache(max_size=1000)
        self._div_cache = TRCache(max_size=1000)
        self._pow_cache = TRCache(max_size=500)

    def get_add(self, a: TRScalar, b: TRScalar) -> Optional[TRScalar]:
        """Get cached addition result."""
        key = f"{a.value}:{a.tag.name}+{b.value}:{b.tag.name}"
        return self._add_cache.get(key)

    def cache_add(self, a: TRScalar, b: TRScalar, result: TRScalar):
        """Cache addition result."""
        key = f"{a.value}:{a.tag.name}+{b.value}:{b.tag.name}"
        self._add_cache.put(key, result)

    def get_mul(self, a: TRScalar, b: TRScalar) -> Optional[TRScalar]:
        """Get cached multiplication result."""
        key = f"{a.value}:{a.tag.name}*{b.value}:{b.tag.name}"
        return self._mul_cache.get(key)

    def cache_mul(self, a: TRScalar, b: TRScalar, result: TRScalar):
        """Cache multiplication result."""
        key = f"{a.value}:{a.tag.name}*{b.value}:{b.tag.name}"
        self._mul_cache.put(key, result)

    def clear_all(self):
        """Clear all operation caches."""
        self._add_cache.clear()
        self._mul_cache.clear()
        self._div_cache.clear()
        self._pow_cache.clear()

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "add": self._add_cache.get_statistics(),
            "mul": self._mul_cache.get_statistics(),
            "div": self._div_cache.get_statistics(),
            "pow": self._pow_cache.get_statistics(),
        }


# Global operation cache
_operation_cache = OperationCache()


def get_operation_cache() -> OperationCache:
    """Get the global operation cache."""
    return _operation_cache
