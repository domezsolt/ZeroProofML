"""Utilities for transreal arithmetic."""

from .optimization import *
from .profiling import *
from .caching import *
from .parallel import *
from .benchmarking import *

__all__ = []

# Export optimization utilities
try:
    from .optimization import (
        TROptimizer,
        GraphOptimizer,
        OperationFuser,
        MemoryOptimizer,
        optimize_tr_graph,
        fuse_operations,
        OPTIMIZATION_AVAILABLE,
    )
    __all__.extend([
        "TROptimizer",
        "GraphOptimizer", 
        "OperationFuser",
        "MemoryOptimizer",
        "optimize_tr_graph",
        "fuse_operations",
        "OPTIMIZATION_AVAILABLE",
    ])
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Export profiling utilities
try:
    from .profiling import (
        TRProfiler,
        profile_tr_operation,
        memory_profile,
        tag_statistics,
        performance_report,
        PROFILING_AVAILABLE,
    )
    __all__.extend([
        "TRProfiler",
        "profile_tr_operation",
        "memory_profile",
        "tag_statistics",
        "performance_report",
        "PROFILING_AVAILABLE",
    ])
except ImportError:
    PROFILING_AVAILABLE = False

# Export caching utilities
try:
    from .caching import (
        TRCache,
        memoize_tr,
        cached_operation,
        clear_cache,
        cache_statistics,
        CACHING_AVAILABLE,
    )
    __all__.extend([
        "TRCache",
        "memoize_tr",
        "cached_operation",
        "clear_cache",
        "cache_statistics",
        "CACHING_AVAILABLE",
    ])
except ImportError:
    CACHING_AVAILABLE = False

# Export parallel utilities
try:
    from .parallel import (
        parallel_map,
        parallel_reduce,
        TRThreadPool,
        TRProcessPool,
        vectorize_operation,
        PARALLEL_AVAILABLE,
    )
    __all__.extend([
        "parallel_map",
        "parallel_reduce",
        "TRThreadPool",
        "TRProcessPool",
        "vectorize_operation",
        "PARALLEL_AVAILABLE",
    ])
except ImportError:
    PARALLEL_AVAILABLE = False

# Export benchmarking utilities
try:
    from .benchmarking import (
        TRBenchmark,
        BenchmarkResult,
        OperationBenchmark,
        create_scaling_benchmark,
        profile_memory_usage,
        BENCHMARKING_AVAILABLE,
    )
    __all__.extend([
        "TRBenchmark",
        "BenchmarkResult",
        "OperationBenchmark",
        "create_scaling_benchmark",
        "profile_memory_usage",
        "BENCHMARKING_AVAILABLE",
    ])
except ImportError:
    BENCHMARKING_AVAILABLE = False