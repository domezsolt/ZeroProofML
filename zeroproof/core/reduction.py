"""
Reduction modes for transreal tensors.

This module defines how aggregation operations (sum, mean, etc.) handle
non-REAL values in transreal arithmetic.
"""

from enum import Enum, auto


class ReductionMode(Enum):
    """
    Modes for handling non-REAL values in reductions.

    Attributes:
        STRICT: If any element is PHI, result is PHI. If any infinity present
                with conflicting signs, result is PHI. Otherwise, infinity
                dominates or all REAL.
        DROP_NULL: Ignore PHI elements and reduce over remaining. If none
                   remain, result is PHI. Used in metrics/monitoring.
    """

    STRICT = auto()
    DROP_NULL = auto()
