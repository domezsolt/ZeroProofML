"""Core transreal scalar types and arithmetic operations."""

# Mode isolation imports
from .mode_isolation import (
    IsolatedModule,
    ModeIsolationConfig,
    ModeSwitchGuard,
    ModeViolationError,
    WheelAxioms,
    check_value_mode_compatibility,
    compile_time_switch,
    ensure_mode_purity,
    isolated_operation,
    tr_only,
    validate_mode_transition,
    wheel_only,
)
from .precision_config import PrecisionConfig, PrecisionMode, precision_context
from .reduction import ReductionMode
from .reductions import tr_max, tr_mean, tr_min, tr_prod, tr_sum
from .separated_ops import safe_add, safe_mul
from .tr_ops import (
    tr_abs,
    tr_add,
    tr_div,
    tr_log,
    tr_mul,
    tr_neg,
    tr_pow_int,
    tr_sign,
    tr_sqrt,
    tr_sub,
)
from .tr_scalar import (
    TRScalar,
    TRTag,
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
    real,
)
from .wheel_mode import (
    ArithmeticMode,
    WheelModeConfig,
    arithmetic_mode,
    use_transreal,
    use_wheel,
    wheel_mode,
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
