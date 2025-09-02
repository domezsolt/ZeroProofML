"""Property-based tests for caching utilities."""

import pytest
from hypothesis import given, strategies as st, assume, settings
import time
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

from zeroproof.core import real, pinf, ninf, phi, TRTag
from zeroproof.autodiff import TRNode, tr_add, tr_mul
from zeroproof.utils.caching import (
    TRCache, CacheEntry, memoize_tr, cached_operation, cache_operation_result,
    clear_cache, cache_statistics, ResultCache, OperationCache, get_operation_cache
)


class TestTRCache:
    """Test transreal cache."""
    
    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        cache = TRCache(max_size=10)
        
        # Put and get
        cache.put("key1", real(1.0))
        assert cache.get("key1") == real(1.0)
        
        # Miss
        assert cache.get("key2") is None
        
        # Update
        cache.put("key1", real(2.0))
        assert cache.get("key1") == real(2.0)
    
    @given(st.integers(min_value=1, max_value=20))
    def test_cache_eviction_lru(self, max_size):
        """Test LRU eviction policy."""
        cache = TRCache(max_size=max_size, eviction_policy='lru')
        
        # Fill cache beyond capacity
        for i in range(max_size + 5):
            cache.put(f"key{i}", real(float(i)))
        
        # First items should be evicted
        for i in range(5):
            assert cache.get(f"key{i}") is None
        
        # Last items should still be there
        for i in range(max_size + 5 - max_size, max_size + 5):
            assert cache.get(f"key{i}") is not None
    
    def test_cache_eviction_lfu(self):
        """Test LFU eviction policy."""
        cache = TRCache(max_size=3, eviction_policy='lfu')
        
        # Add items
        cache.put("key1", real(1.0))
        cache.put("key2", real(2.0))
        cache.put("key3", real(3.0))
        
        # Access key1 and key2 multiple times
        for _ in range(5):
            cache.get("key1")
            cache.get("key2")
        
        # Add new item - should evict key3 (least frequently used)
        cache.put("key4", real(4.0))
        
        assert cache.get("key3") is None
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key4") is not None
    
    def test_cache_memory_limit(self):
        """Test memory-based eviction."""
        cache = TRCache(max_size=1000, max_memory_mb=0.001)  # Very small limit
        
        # Add items until memory limit
        for i in range(100):
            cache.put(f"key{i}", TRNode.constant(real(i)), size=1000)
        
        # Should have evicted some items
        stats = cache.get_statistics()
        assert stats['evictions'] > 0
        assert stats['memory_used_mb'] <= 0.001 * 1.1  # Allow 10% overhead
    
    def test_cache_ttl(self):
        """Test time-to-live functionality."""
        cache = TRCache(ttl_seconds=0.1)
        
        cache.put("key1", real(1.0))
        assert cache.get("key1") == real(1.0)
        
        # Wait for TTL
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = TRCache()
        
        # Generate some activity
        cache.put("key1", real(1.0), compute_time=0.1)
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_statistics()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
        assert stats['compute_time_saved'] >= 0.2  # 2 hits * 0.1


class TestMemoization:
    """Test memoization decorator."""
    
    def test_basic_memoization(self):
        """Test basic memoization functionality."""
        call_count = 0
        
        @memoize_tr()
        def expensive_computation(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return tr_add(x, real(1.0))
        
        # First call
        result1 = expensive_computation(real(5.0))
        assert call_count == 1
        
        # Second call - should use cache
        result2 = expensive_computation(real(5.0))
        assert call_count == 1  # Not incremented
        assert result1 == result2
        
        # Different argument
        result3 = expensive_computation(real(6.0))
        assert call_count == 2
    
    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    def test_memoization_preserves_results(self, values):
        """Test that memoization preserves computation results."""
        @memoize_tr()
        def sum_values(*args):
            result = args[0]
            for val in args[1:]:
                result = tr_add(result, val)
            return result
        
        tr_values = [real(v) for v in values]
        
        # Compute twice
        result1 = sum_values(*tr_values)
        result2 = sum_values(*tr_values)
        
        assert result1 == result2
    
    def test_custom_key_function(self):
        """Test memoization with custom key function."""
        call_count = 0
        
        def key_func(x, precision=2):
            # Key based on rounded value
            # x is a TRScalar, not a TRNode
            rounded = round(x.value, precision)
            return f"{rounded}:{x.tag.name}:{precision}"
        
        @memoize_tr(key_func=key_func)
        def compute(x, precision=2):
            nonlocal call_count
            call_count += 1
            return tr_mul(TRNode.constant(x), TRNode.constant(real(2.0)))
        
        # These should use same cache entry due to rounding in key
        # Use values that definitely round to the same value
        result1 = compute(real(1.234), precision=1)  # rounds to 1.2
        result2 = compute(real(1.236), precision=1)  # rounds to 1.2
        
        # Should have only called compute once due to cache hit
        assert call_count == 1
        
        # Should be the exact same object from cache
        assert result1 is result2
    
    def test_cache_control_methods(self):
        """Test cache control methods on memoized functions."""
        # Use a fresh cache instance to avoid interference
        test_cache = TRCache()
        
        @memoize_tr(cache=test_cache)
        def compute(x):
            return tr_add(TRNode.constant(x), TRNode.constant(real(1.0)))
        
        # Generate some cache entries
        for i in range(5):
            compute(real(float(i)))
        
        # Check cache info
        info = compute.cache_info()
        assert info['size'] == 5
        
        # Clear cache
        compute.cache_clear()
        info = compute.cache_info()
        assert info['size'] == 0


class TestOperationCache:
    """Test specialized operation caches."""
    
    def test_operation_cache_basic(self):
        """Test basic operation caching."""
        op_cache = OperationCache()
        
        a = real(2.0)
        b = real(3.0)
        
        # Cache miss
        assert op_cache.get_add(a, b) is None
        
        # Compute and cache
        result = tr_add(a, b)
        op_cache.cache_add(a, b, result)
        
        # Cache hit
        cached = op_cache.get_add(a, b)
        assert cached == result
    
    @given(
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False)
    )
    def test_operation_cache_consistency(self, x, y):
        """Test that operation cache maintains consistency."""
        op_cache = get_operation_cache()
        
        a = real(x)
        b = real(y)
        
        # Check addition
        cached_add = op_cache.get_add(a, b)
        computed_add = tr_add(a, b)
        
        if cached_add is not None:
            assert cached_add == computed_add
        else:
            op_cache.cache_add(a, b, computed_add)
        
        # Check multiplication
        cached_mul = op_cache.get_mul(a, b)
        computed_mul = tr_mul(a, b)
        
        if cached_mul is not None:
            assert cached_mul == computed_mul
        else:
            op_cache.cache_mul(a, b, computed_mul)
    
    def test_operation_cache_statistics(self):
        """Test operation cache statistics."""
        op_cache = OperationCache()
        
        # Generate activity
        for i in range(10):
            a = real(float(i))
            b = real(float(i + 1))
            
            result = tr_add(a, b)
            op_cache.cache_add(a, b, result)
            
            # Some cache hits
            if i > 5:
                op_cache.get_add(real(0.0), real(1.0))
        
        stats = op_cache.get_statistics()
        
        assert 'add' in stats
        assert stats['add']['size'] > 0


class TestResultCache:
    """Test result cache with dependency tracking."""
    
    def test_dependency_tracking(self):
        """Test that dependencies are tracked correctly."""
        cache = ResultCache()
        
        # Create computation graph
        a = TRNode.constant(real(1.0))
        b = TRNode.constant(real(2.0))
        
        def compute_sum(node):
            if node == a:
                return node.value
            elif node == b:
                return node.value
            else:
                # c = a + b
                return real(3.0)
        
        # Create dependent node
        c = TRNode.constant(real(3.0))
        c._grad_info = type('GradInfo', (), {
            'inputs': [weakref.ref(a), weakref.ref(b)],
            'op_type': None
        })()
        
        # Cache result
        result = cache.get_or_compute(c, compute_sum)
        assert result == real(3.0)
        
        # Should be cached
        result2 = cache.get_or_compute(c, lambda x: real(999.0))
        assert result2 == real(3.0)  # Used cache, not recomputed
        
        # Invalidate dependency
        cache.invalidate(a)
        
        # Should recompute
        result3 = cache.get_or_compute(c, lambda x: real(4.0))
        assert result3 == real(4.0)  # Recomputed after invalidation


class TestCacheThreadSafety:
    """Test thread safety of caching."""
    
    def test_concurrent_cache_access(self):
        """Test cache under concurrent access."""
        cache = TRCache()
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"key_{worker_id}_{i % 10}"
                    
                    # Alternate between get and put
                    if i % 2 == 0:
                        cache.put(key, real(float(i)))
                    else:
                        value = cache.get(key)
                        # Verify if we get a value, it's correct
                        if value is not None:
                            assert value.value.tag == TRTag.REAL
            except Exception as e:
                errors.append(e)
        
        # Run workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for future in futures:
                future.result()
        
        # No errors should occur
        assert len(errors) == 0
        
        # Cache should have some entries
        stats = cache.get_statistics()
        assert stats['size'] > 0


@pytest.mark.benchmark
class TestCachingPerformance:
    """Benchmark caching performance."""
    
    def test_cache_speedup(self, benchmark):
        """Test that caching provides speedup."""
        @memoize_tr()
        def expensive_computation(x, n):
            # Simulate expensive computation
            result = x
            for i in range(n):
                result = tr_add(result, real(i))
                result = tr_mul(result, real(1.001))
            return result
        
        # Warm up cache
        expensive_computation(real(1.0), 100)
        
        # Benchmark cached call
        result = benchmark(expensive_computation, real(1.0), 100)
        
        # Verify result
        assert result.value.tag == TRTag.REAL
        
        # Check cache was used
        info = expensive_computation.cache_info()
        assert info['hits'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
