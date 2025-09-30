"""
Global precision configuration for ZeroProof.

This module manages the default numeric precision used throughout the library.
By default, float64 is used for maximum precision in transreal arithmetic.
"""

import numpy as np
from typing import Type, Union
from enum import Enum


class PrecisionMode(Enum):
    """Supported precision modes."""
    FLOAT16 = np.float16
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    
    @property
    def numpy_dtype(self):
        """Get the numpy dtype for this precision."""
        return self.value
    
    @property
    def bits(self) -> int:
        """Get the number of bits for this precision."""
        return np.dtype(self.value).itemsize * 8


class PrecisionConfig:
    """
    Global precision configuration.
    
    By default, ZeroProof uses float64 for all computations to ensure
    maximum precision in transreal arithmetic operations.
    """
    
    _default_mode: PrecisionMode = PrecisionMode.FLOAT64
    _enforce_precision: bool = True
    
    @classmethod
    def set_precision(cls, mode: Union[PrecisionMode, str]) -> None:
        """
        Set the default precision mode.
        
        Args:
            mode: PrecisionMode enum or string ('float16', 'float32', 'float64')
            
        Raises:
            ValueError: If mode is not supported
        """
        if isinstance(mode, str):
            mode_map = {
                'float16': PrecisionMode.FLOAT16,
                'float32': PrecisionMode.FLOAT32,
                'float64': PrecisionMode.FLOAT64,
            }
            if mode not in mode_map:
                raise ValueError(f"Unsupported precision mode: {mode}")
            mode = mode_map[mode]
        
        if not isinstance(mode, PrecisionMode):
            raise ValueError(f"Invalid precision mode: {mode}")
        
        cls._default_mode = mode
    
    @classmethod
    def get_precision(cls) -> PrecisionMode:
        """Get the current default precision mode."""
        return cls._default_mode
    
    @classmethod
    def get_dtype(cls) -> Type[np.floating]:
        """Get the numpy dtype for the current precision."""
        return cls._default_mode.numpy_dtype
    
    @classmethod
    def enforce_precision(cls, value: Union[float, int, np.floating]) -> float:
        """
        Convert a value to the current default precision.
        
        Args:
            value: Numeric value to convert
            
        Returns:
            Value with enforced precision as Python float
        """
        if not cls._enforce_precision:
            return float(value)
        
        dtype = cls.get_dtype()
        # Convert to numpy dtype then back to Python float
        return float(dtype(value))
    
    @classmethod
    def set_enforcement(cls, enforce: bool) -> None:
        """
        Enable or disable precision enforcement.
        
        Args:
            enforce: Whether to enforce precision conversion
        """
        cls._enforce_precision = enforce
    
    @classmethod
    def is_enforcing(cls) -> bool:
        """Check if precision enforcement is enabled."""
        return cls._enforce_precision
    
    # Conservative overflow margin to avoid borderline magnitudes near dtype max
    _overflow_safety_margin: float = 1.0  # strict check; do not preemptively overflow

    @classmethod
    def check_overflow(cls, value: float) -> bool:
        """
        Check if a value would overflow in the current precision.
        
        Args:
            value: Value to check
            
        Returns:
            True if value would overflow
        """
        dtype = cls.get_dtype()
        finfo = np.finfo(dtype)
        # Be conservative near the limit to ensure deterministic overflow
        safety = cls._overflow_safety_margin
        try:
            threshold = float(finfo.max) * float(safety)
        except Exception:
            threshold = float(finfo.max)
        return abs(value) > threshold
    
    @classmethod
    def get_epsilon(cls) -> float:
        """Get machine epsilon for current precision."""
        return np.finfo(cls.get_dtype()).eps
    
    @classmethod
    def get_max(cls) -> float:
        """Get maximum representable value for current precision."""
        return np.finfo(cls.get_dtype()).max
    
    @classmethod
    def get_min(cls) -> float:
        """Get minimum positive value for current precision."""
        return np.finfo(cls.get_dtype()).tiny


# Context manager for temporary precision changes
class precision_context:
    """
    Context manager for temporary precision changes.
    
    Example:
        with precision_context('float32'):
            # Operations use float32
            x = real(1.0)
        # Back to previous precision
    """
    
    def __init__(self, mode: Union[PrecisionMode, str]):
        self.new_mode = mode
        self.old_mode = None
        self.old_enforcement = None
    
    def __enter__(self):
        self.old_mode = PrecisionConfig.get_precision()
        self.old_enforcement = PrecisionConfig.is_enforcing()
        PrecisionConfig.set_precision(self.new_mode)
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        PrecisionConfig.set_precision(self.old_mode)
        PrecisionConfig.set_enforcement(self.old_enforcement)
