"""Utilities for transreal arithmetic.

This package exposes submodules in a best-effort manner to keep optional
dependencies optional. Do not import heavy/optional modules unconditionally.
"""

__all__: list[str] = []

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
        ParallelConfig,
        PARALLEL_AVAILABLE,
    )
    __all__.extend([
        "parallel_map",
        "parallel_reduce",
        "TRThreadPool",
        "TRProcessPool",
        "vectorize_operation",
        "ParallelConfig",
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

# Export metrics utilities
try:
    from .metrics import (
        PoleLocation,
        PoleLocalizationError,
        SignConsistencyChecker,
        AsymptoticSlopeAnalyzer,
        ResidualConsistencyLoss,
        AntiIllusionMetrics,
    )
    __all__.extend([
        "PoleLocation",
        "PoleLocalizationError",
        "SignConsistencyChecker", 
        "AsymptoticSlopeAnalyzer",
        "ResidualConsistencyLoss",
        "AntiIllusionMetrics",
    ])
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Export logging utilities
try:
    from .logging import (
        StructuredLogger,
        ExperimentTracker,
        MetricsAggregator,
        log_training_step,
        get_experiment_tracker,
    )
    __all__.extend([
        "StructuredLogger",
        "ExperimentTracker",
        "MetricsAggregator",
        "log_training_step",
        "get_experiment_tracker",
    ])
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

# Export plotting utilities (optional)
try:
    from .plotting import (
        TrainingCurvePlotter,
        PoleVisualizationPlotter,
        ResidualAnalysisPlotter,
        ComparisonPlotter,
        create_paper_ready_figures,
        save_all_plots,
        use_zeroproof_style,
        MATPLOTLIB_AVAILABLE,
        SEABORN_AVAILABLE,
    )
    __all__.extend([
        "TrainingCurvePlotter",
        "PoleVisualizationPlotter",
        "ResidualAnalysisPlotter",
        "ComparisonPlotter",
        "create_paper_ready_figures",
        "save_all_plots",
        "use_zeroproof_style",
        "MATPLOTLIB_AVAILABLE",
        "SEABORN_AVAILABLE",
    ])
    PLOTTING_AVAILABLE = True
except ImportError as e:
    PLOTTING_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    SEABORN_AVAILABLE = False

# Export dataset generation utilities
try:
    from .dataset_generation import (
        SingularityInfo,
        SingularDatasetGenerator,
        generate_robotics_singular_configurations,
    )
    __all__.extend([
        "SingularityInfo",
        "SingularDatasetGenerator",
        "generate_robotics_singular_configurations",
    ])
    DATASET_GENERATION_AVAILABLE = True
except ImportError:
    DATASET_GENERATION_AVAILABLE = False

# Export pole metrics utilities
try:
    from .pole_metrics import (
        PoleMetrics,
        PoleEvaluator,
        compute_pole_localization_error,
        check_sign_consistency,
        compute_asymptotic_slope_error,
        compute_residual_consistency,
        count_singularities,
        compute_coverage_by_distance,
        detect_poles_from_Q,
    )
    __all__.extend([
        "PoleMetrics",
        "PoleEvaluator",
        "compute_pole_localization_error",
        "check_sign_consistency",
        "compute_asymptotic_slope_error",
        "compute_residual_consistency",
        "count_singularities",
        "compute_coverage_by_distance",
        "detect_poles_from_Q",
    ])
    POLE_METRICS_AVAILABLE = True
except ImportError:
    POLE_METRICS_AVAILABLE = False

# Export pole visualization utilities
try:
    from .pole_visualization import (
        PoleVisualizer,
    )
    __all__.extend([
        "PoleVisualizer",
    ])
    POLE_VISUALIZATION_AVAILABLE = True
except ImportError:
    POLE_VISUALIZATION_AVAILABLE = False

# Export evaluation API
try:
    from .evaluation_api import (
        EvaluationConfig,
        IntegratedEvaluator,
        TrainingLogger,
        create_evaluator,
    )
    __all__.extend([
        "EvaluationConfig",
        "IntegratedEvaluator",
        "TrainingLogger",
        "create_evaluator",
    ])
    EVALUATION_API_AVAILABLE = True
except ImportError:
    EVALUATION_API_AVAILABLE = False
