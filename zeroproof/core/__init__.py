"""Core transreal scalar types and arithmetic operations."""

from .tr_scalar import (
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
)

from .tr_ops import (
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
)

from .reduction import ReductionMode
from .reductions import tr_sum, tr_mean, tr_prod, tr_min, tr_max
from .precision_config import PrecisionConfig, PrecisionMode, precision_context
from .wheel_mode import ArithmeticMode, WheelModeConfig, wheel_mode, arithmetic_mode, use_transreal, use_wheel

__all__ = [
    # Types
    "TRScalar",
    "TRTag",
    "ReductionMode",
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
    
    # Arithmetic operations
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
    
    # Reduction operations
    "tr_sum",
    "tr_mean",
    "tr_prod",
    "tr_min",
    "tr_max",
    
    # Wheel mode
    "ArithmeticMode",
    "WheelModeConfig",
    "wheel_mode",
    "arithmetic_mode",
    "use_transreal",
    "use_wheel",
]
