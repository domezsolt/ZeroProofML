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

# Mode isolation imports
from .mode_isolation import (
    ModeIsolationConfig,
    ModeViolationError,
    ModeSwitchGuard,
    WheelAxioms,
    isolated_operation,
    compile_time_switch,
    ensure_mode_purity,
    tr_only,
    wheel_only,
    check_value_mode_compatibility,
    validate_mode_transition,
    IsolatedModule,
)

from .separated_ops import (
    safe_add,
    safe_mul,
)

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
    
    # Mode isolation
    "ModeIsolationConfig",
    "ModeViolationError",
    "ModeSwitchGuard",
    "WheelAxioms",
    "isolated_operation",
    "compile_time_switch",
    "ensure_mode_purity",
    "tr_only",
    "wheel_only",
    "check_value_mode_compatibility",
    "validate_mode_transition",
    "IsolatedModule",
    
    # Safe operations
    "safe_add",
    "safe_mul",
]
