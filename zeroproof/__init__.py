# MIT License
# See LICENSE file in the project root for full license text.
"""
ZeroProof: Transreal arithmetic for stable machine learning without epsilon hacks.

This library implements transreal (TR) arithmetic, extending real numbers with
special values for infinity and undefined forms, making all operations total
(never throwing exceptions).
"""

__version__ = "0.1.0"
__author__ = "ZeroProof Team"
__email__ = "zeroproof@example.com"

# Keep top-level import lightweight: only expose core types and operations which
# have no optional heavy dependencies. Subpackages (autodiff, layers, training,
# utils, etc.) can be imported explicitly (e.g., `from zeroproof.layers import TRRational`).

# Bridge functions (framework-agnostic imports only)
from .bridge.ieee_tr import from_ieee, to_ieee
from .core import (
    ArithmeticMode,
    PrecisionConfig,
    PrecisionMode,
    ReductionMode,
    TRScalar,
    TRTag,
    arithmetic_mode,
    bottom,
    is_bottom,
    is_finite,
    is_infinite,
    is_ninf,
    is_phi,
    is_pinf,
    is_real,
    ninf,
    phi,
    pinf,
    precision_context,
    real,
    tr_abs,
    tr_add,
    tr_div,
    tr_log,
    tr_max,
    tr_mean,
    tr_min,
    tr_mul,
    tr_neg,
    tr_pow_int,
    tr_prod,
    tr_sign,
    tr_sqrt,
    tr_sub,
    tr_sum,
    use_transreal,
    use_wheel,
    wheel_mode,
)

# Policy exports (lightweight, no heavy deps)
from .policy import TRPolicy, TRPolicyConfig

# NumPy bridge is optional; import guarded to keep minimal installs working
try:  # pragma: no cover - optional dependency
    from .bridge.numpy_bridge import from_numpy, to_numpy  # type: ignore
except Exception:  # NumPy not available or optional
    from_numpy = None  # type: ignore
    to_numpy = None  # type: ignore

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core types and checks
    "TRScalar",
    "TRTag",
    "real",
    "pinf",
    "ninf",
    "phi",
    "bottom",
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
    # Precision and modes
    "ReductionMode",
    "PrecisionConfig",
    "PrecisionMode",
    "precision_context",
    "ArithmeticMode",
    "wheel_mode",
    "arithmetic_mode",
    "use_transreal",
    "use_wheel",
    # Bridges
    "from_ieee",
    "to_ieee",
    "from_numpy",
    "to_numpy",
    # Policy
    "TRPolicy",
    "TRPolicyConfig",
    # Submodules (exposed lazily via __getattr__)
    "layers",
    "training",
    "autodiff",
    "utils",
]


def __getattr__(name):  # Lazy import heavy submodules on demand
    if name in {"layers", "training", "autodiff", "utils"}:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
