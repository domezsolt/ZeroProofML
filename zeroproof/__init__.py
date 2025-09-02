"""
ZeroProof: Transreal arithmetic for stable machine learning without epsilon hacks.

This library implements transreal (TR) arithmetic, extending real numbers with
special values for infinity and undefined forms, making all operations total
(never throwing exceptions).
"""

__version__ = "0.1.0"
__author__ = "ZeroProof Team"
__email__ = "zeroproof@example.com"

# Core imports
from .core import (
    TRScalar,
    TRTag,
    real,
    pinf,
    ninf,
    phi,
    bottom,
    is_real,
    is_pinf,
    is_ninf,
    is_phi,
    is_bottom,
    is_finite,
    is_infinite,
    PrecisionConfig,
    PrecisionMode,
    precision_context,
)

# Arithmetic operations (public API returns TRScalar; autodiff ops live under zeroproof.autodiff)
from .core import (
    tr_add,
    tr_sub,
    tr_mul,
    tr_div,
    tr_abs,
    tr_sign,
    tr_neg,
    tr_log,
    tr_sqrt,
    tr_pow_int,
    tr_sum,
    tr_mean,
    tr_prod,
    tr_min,
    tr_max,
)

# Bridge functions (framework-agnostic imports only)
from .bridge.ieee_tr import (
    from_ieee,
    to_ieee,
)

# NumPy bridge is optional but lightweight; import directly to avoid torch/jax side-effects
try:
    from .bridge.numpy_bridge import (
        from_numpy,
        to_numpy,
    )
except Exception:  # NumPy not available or optional
    from_numpy = None  # type: ignore
    to_numpy = None    # type: ignore

# Reduction modes
from .core import ReductionMode

# Import submodules to make them accessible
from . import autodiff
from . import layers
# Do not import heavy/optional bridges (torch/jax) at top level to keep core framework-agnostic
from . import utils
from . import training

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core types
    "TRScalar",
    "TRTag",
    "PrecisionConfig",
    "PrecisionMode",
    "precision_context",
    
        # Factory functions
    "real",
    "pinf",
    "ninf",
    "phi",
    "bottom",
    
    # Type checking
    "is_real",
    "is_pinf",
    "is_ninf",
    "is_phi",
    "is_bottom",
    "is_finite",
    "is_infinite",
    
    # Arithmetic
    "tr_add",
    "tr_sub",
    "tr_mul",
    "tr_div",
    "tr_abs",
    "tr_sign",
    "tr_neg",
    "tr_log",
    "tr_sqrt",
    "tr_pow_int",
    "tr_sum",
    "tr_mean",
    "tr_prod",
    "tr_min",
    "tr_max",
    
    # Bridge
    "from_ieee",
    "to_ieee",
    "from_numpy",
    "to_numpy",
    
    # Enums
    "ReductionMode",
    
    # Submodules
    "autodiff",
    "layers",
    "utils",
    "training",
    
    # Re-export gradient modes for convenience
    "GradientMode",
    "gradient_mode",
    "use_mask_real",
    "use_saturating",
    
    # Wheel mode
    "ArithmeticMode",
    "wheel_mode",
    "arithmetic_mode",
    "use_transreal",
    "use_wheel",
]

# Import gradient mode functionality for re-export
from .autodiff import GradientMode, gradient_mode, use_mask_real, use_saturating

# Import wheel mode functionality for re-export
from .core import ArithmeticMode, wheel_mode, arithmetic_mode, use_transreal, use_wheel
