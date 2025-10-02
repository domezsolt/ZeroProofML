"""Transreal autodifferentiation with Mask-REAL rule."""

from .backward import backward_pass, topological_sort
from .grad_funcs import check_gradient, tr_grad, tr_value_and_grad
from .grad_mode import (
    GradientMode,
    GradientModeConfig,
    gradient_mode,
    use_mask_real,
    use_saturating,
)
from .gradient_tape import TRGradientTape, gradient_tape
from .hybrid_gradient import (
    HybridGradientContext,
    HybridGradientSchedule,
    ScheduleType,
    create_default_schedule,
)
from .tr_node import OpType, TRNode
from .tr_ops_grad import (
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

__all__ = [
    # Core classes
    "TRNode",
    "OpType",
    "TRGradientTape",
    "gradient_tape",
    # Gradient functions
    "tr_grad",
    "tr_value_and_grad",
    "check_gradient",
    # Gradient-aware operations
    "tr_add",
    "tr_sub",
    "tr_mul",
    "tr_div",
    "tr_neg",
    "tr_abs",
    "tr_sign",
    "tr_log",
    "tr_sqrt",
    "tr_pow_int",
    # Backward pass utilities
    "backward_pass",
    "topological_sort",
    # Gradient modes
    "GradientMode",
    "GradientModeConfig",
    "gradient_mode",
    "use_mask_real",
    "use_saturating",
    # Hybrid gradient schedule
    "HybridGradientSchedule",
    "HybridGradientContext",
    "ScheduleType",
    "create_default_schedule",
]
