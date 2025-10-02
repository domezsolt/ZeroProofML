"""
Lightweight serialization helpers for JSON/NPZ round-trips.

Provides conversion of NumPy scalars and arrays to Python built-ins, and
safe encoding of non-finite floating values.
"""

from typing import Any, Dict, List


def to_builtin(obj: Any) -> Any:
    """Recursively convert NumPy/array scalars to Python built-ins.

    - numpy.floating -> float
    - numpy.integer -> int
    - numpy.bool_ -> bool
    - non-finite floats -> string tokens ("inf", "-inf", "nan")
    - dicts/lists/tuples are processed recursively
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore

    # Non-finite encoding for standard floats
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj

    if _np is not None:
        if isinstance(obj, (_np.floating,)):
            val = float(obj)
            if val != val:
                return "nan"
            if val == float("inf"):
                return "inf"
            if val == float("-inf"):
                return "-inf"
            return val
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, (_np.ndarray,)):
            return [to_builtin(v) for v in obj.tolist()]

    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]

    return obj
