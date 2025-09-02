"""Bridges for transreal arithmetic with various numerical libraries."""

# Core IEEE bridge
from .ieee_tr import from_ieee, to_ieee, from_numpy, to_numpy

# NumPy bridge
try:
    from .numpy_bridge import (
        TRArray,
        from_numpy as from_numpy_array,
        to_numpy as to_numpy_array,
        validate_array,
        check_finite,
        where_real,
        count_tags,
        real_values,
        clip_infinities,
        NUMPY_AVAILABLE,
    )
except ImportError:
    NUMPY_AVAILABLE = False
    TRArray = None

# PyTorch bridge (import only if torch available; else keep flag False and do not expose symbols)
try:
    import importlib.util
    if importlib.util.find_spec('torch') is not None:
        from .torch_bridge import (
            TRTensor,
            from_torch,
            to_torch,
            mask_real_backward,
            TRFunction,
            tr_tensor_from_list,
            batch_from_scalars as batch_from_scalars_torch,
            enable_tr_gradients,
            TORCH_AVAILABLE,
        )
    else:
        raise ImportError
except ImportError:
    TORCH_AVAILABLE = False

# JAX bridge (guard stronger: only import if jax is available)
try:
    import importlib.util
    if importlib.util.find_spec('jax') is not None:
        from .jax_bridge import (
            TRJaxArray,
            from_jax,
            to_jax,
            mask_real_grad,
            tr_add_jax,
            tr_mul_jax,
            tr_scalar_to_jax,
            batch_from_scalars_jax,
            JAX_AVAILABLE,
        )
    else:
        raise ImportError
except ImportError:
    JAX_AVAILABLE = False

# Precision utilities
from .precision import (
    Precision,
    get_precision_info,
    cast_to_precision,
    tr_scalar_with_precision,
    PrecisionContext,
    with_precision,
    check_precision_overflow,
    precision_safe_operation,
    MixedPrecisionStrategy,
    analyze_precision_requirements,
)

__all__ = [
    # Core IEEE
    "from_ieee",
    "to_ieee",
    "from_numpy",
    "to_numpy",
    
    # NumPy bridge
    "TRArray",
    "NUMPY_AVAILABLE",
    
    # PyTorch bridge
    "TORCH_AVAILABLE",
    
    # JAX bridge
    "JAX_AVAILABLE",
    
    # Precision
    "Precision",
    "PrecisionContext",
    "with_precision",
    "MixedPrecisionStrategy",
]

# Conditional exports based on availability
if NUMPY_AVAILABLE:
    __all__.extend([
        "from_numpy_array",
        "to_numpy_array",
        "validate_array",
        "check_finite",
        "where_real",
        "count_tags",
        "real_values",
        "clip_infinities",
    ])

if TORCH_AVAILABLE:
    __all__.extend([
        "TRTensor",
        "from_torch",
        "to_torch",
        "mask_real_backward",
        "TRFunction",
        "tr_tensor_from_list",
        "batch_from_scalars_torch",
        "enable_tr_gradients",
    ])

if JAX_AVAILABLE:
    __all__.extend([
        "TRJaxArray",
        "from_jax",
        "to_jax",
        "mask_real_grad",
        "tr_add_jax",
        "tr_mul_jax",
        "tr_scalar_to_jax",
        "batch_from_scalars_jax",
    ])

# Define __getattr__ to handle missing imports
def __getattr__(name):
    # Handle PyTorch imports when not available
    if not TORCH_AVAILABLE and name in {
        "TRTensor", "from_torch", "to_torch", "mask_real_backward",
        "TRFunction", "tr_tensor_from_list", "batch_from_scalars_torch",
        "enable_tr_gradients"
    }:
        raise ImportError("PyTorch not installed")
    
    # Handle JAX imports when not available
    if not JAX_AVAILABLE and name in {
        "TRJaxArray", "from_jax", "to_jax", "mask_real_grad",
        "tr_add_jax", "tr_mul_jax", "tr_scalar_to_jax", "batch_from_scalars_jax"
    }:
        raise ImportError("JAX not installed")
    
    raise AttributeError(f"module 'zeroproof.bridge' has no attribute '{name}'")

# Additional precision exports
__all__.extend([
    "get_precision_info",
    "cast_to_precision",
    "tr_scalar_with_precision",
    "check_precision_overflow",
    "precision_safe_operation",
    "analyze_precision_requirements",
])
