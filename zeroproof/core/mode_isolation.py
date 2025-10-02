"""
Strict isolation mechanism for Wheel mode.

This module ensures that TR and Wheel semantics never mix within a single
operation, enforcing compile-time switching and proper axiom application.
"""

import threading
import warnings
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar

from .tr_scalar import TRScalar, TRTag, bottom, pinf, real
from .wheel_mode import ArithmeticMode, WheelModeConfig

# Type variable for function decorators
F = TypeVar("F", bound=Callable[..., Any])


class ModeViolationError(Exception):
    """Raised when mode isolation is violated."""

    pass


class ModeIsolationConfig:
    """
    Configuration for strict mode isolation.

    Ensures TR and Wheel modes cannot mix within operations.
    """

    # Thread-local storage for operation context
    _thread_local = threading.local()

    # Track if we're inside an operation
    @classmethod
    def _get_operation_depth(cls) -> int:
        """Get current operation nesting depth."""
        if not hasattr(cls._thread_local, "operation_depth"):
            cls._thread_local.operation_depth = 0
        return cls._thread_local.operation_depth

    @classmethod
    def _set_operation_depth(cls, depth: int) -> None:
        """Set operation nesting depth."""
        cls._thread_local.operation_depth = depth

    # Track the mode when entering an operation
    @classmethod
    def _get_operation_mode(cls) -> Optional[ArithmeticMode]:
        """Get the mode locked for current operation."""
        if not hasattr(cls._thread_local, "operation_mode"):
            cls._thread_local.operation_mode = None
        return cls._thread_local.operation_mode

    @classmethod
    def _set_operation_mode(cls, mode: Optional[ArithmeticMode]) -> None:
        """Set the mode for current operation."""
        cls._thread_local.operation_mode = mode

    @classmethod
    def enter_operation(cls) -> None:
        """Mark entry into an operation."""
        depth = cls._get_operation_depth()

        if depth == 0:
            # Entering outermost operation - lock the mode
            current_mode = WheelModeConfig.get_mode()
            cls._set_operation_mode(current_mode)
        else:
            # Nested operation - verify mode hasn't changed
            locked_mode = cls._get_operation_mode()
            current_mode = WheelModeConfig.get_mode()

            if locked_mode != current_mode:
                raise ModeViolationError(
                    f"Mode changed within operation! "
                    f"Started with {locked_mode}, now {current_mode}. "
                    f"TR and Wheel semantics must not mix."
                )

        cls._set_operation_depth(depth + 1)

    @classmethod
    def exit_operation(cls) -> None:
        """Mark exit from an operation."""
        depth = cls._get_operation_depth()

        if depth > 0:
            cls._set_operation_depth(depth - 1)

            if depth == 1:
                # Exiting outermost operation - unlock mode
                cls._set_operation_mode(None)

    @classmethod
    def check_mode_consistency(cls) -> None:
        """Check that mode hasn't changed during operation."""
        if cls._get_operation_depth() > 0:
            locked_mode = cls._get_operation_mode()
            current_mode = WheelModeConfig.get_mode()

            if locked_mode != current_mode:
                raise ModeViolationError(
                    f"Mode changed within operation! "
                    f"Started with {locked_mode}, now {current_mode}"
                )

    @classmethod
    def is_in_operation(cls) -> bool:
        """Check if currently inside an operation."""
        return cls._get_operation_depth() > 0


def isolated_operation(func: F) -> F:
    """
    Decorator to ensure mode isolation for an operation.

    This ensures that TR and Wheel modes cannot be mixed within
    a single operation, maintaining strict semantic separation.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ModeIsolationConfig.enter_operation()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            ModeIsolationConfig.exit_operation()

    return wrapper


def check_value_mode_compatibility(value: TRScalar) -> None:
    """
    Check if a value is compatible with current mode.

    Args:
        value: Value to check

    Raises:
        ModeViolationError: If value uses wrong mode's semantics
    """
    current_mode = WheelModeConfig.get_mode()

    if current_mode == ArithmeticMode.TRANSREAL:
        # In TR mode, BOTTOM should not appear
        if value.tag == TRTag.BOTTOM:
            raise ModeViolationError(
                "BOTTOM (⊥) value encountered in Transreal mode! "
                "BOTTOM is only valid in Wheel mode."
            )

    elif current_mode == ArithmeticMode.WHEEL:
        # In Wheel mode, certain PHI cases should be BOTTOM
        # This is handled by the operations themselves
        pass


class WheelAxioms:
    """
    Wheel axioms that must be enforced in Wheel mode.

    These replace TR semantics when Wheel mode is active.
    """

    @staticmethod
    def zero_times_infinity() -> TRScalar:
        """0 × ∞ = ⊥ in Wheel mode."""
        if not WheelModeConfig.is_wheel():
            raise ModeViolationError("Wheel axiom called in TR mode")
        return bottom()

    @staticmethod
    def infinity_plus_infinity() -> TRScalar:
        """∞ + ∞ = ⊥ in Wheel mode."""
        if not WheelModeConfig.is_wheel():
            raise ModeViolationError("Wheel axiom called in TR mode")
        return bottom()

    @staticmethod
    def infinity_minus_infinity() -> TRScalar:
        """∞ - ∞ = ⊥ in Wheel mode."""
        if not WheelModeConfig.is_wheel():
            raise ModeViolationError("Wheel axiom called in TR mode")
        return bottom()

    @staticmethod
    def infinity_over_infinity() -> TRScalar:
        """∞ / ∞ = ⊥ in Wheel mode."""
        if not WheelModeConfig.is_wheel():
            raise ModeViolationError("Wheel axiom called in TR mode")
        return bottom()

    @staticmethod
    def verify_axioms() -> bool:
        """
        Verify that Wheel axioms are properly enforced.

        Returns:
            True if all axioms hold
        """
        if not WheelModeConfig.is_wheel():
            return False

        from .tr_ops import tr_add, tr_div, tr_mul, tr_sub

        # Test 0 × ∞ = ⊥
        result = tr_mul(real(0.0), pinf())
        if result.tag != TRTag.BOTTOM:
            return False

        # Test ∞ + ∞ = ⊥
        result = tr_add(pinf(), pinf())
        if result.tag != TRTag.BOTTOM:
            return False

        # Test ∞ - ∞ = ⊥
        result = tr_sub(pinf(), pinf())
        if result.tag != TRTag.BOTTOM:
            return False

        # Test ∞ / ∞ = ⊥
        result = tr_div(pinf(), pinf())
        if result.tag != TRTag.BOTTOM:
            return False

        return True


class ModeSwitchGuard:
    """
    Guard to prevent mode switching during critical sections.

    This can be used to ensure a block of code runs entirely
    in one mode without possibility of switching.
    """

    def __init__(self, mode: Optional[ArithmeticMode] = None):
        """
        Initialize guard.

        Args:
            mode: Mode to enforce, or None to lock current mode
        """
        self.mode = mode or WheelModeConfig.get_mode()
        self.original_mode = None
        self.locked = False

    def __enter__(self):
        """Enter guarded section."""
        self.original_mode = WheelModeConfig.get_mode()

        # Set to required mode
        WheelModeConfig.set_mode(self.mode)

        # Lock mode changes
        self.locked = True

        # Store guard in thread-local
        if not hasattr(ModeIsolationConfig._thread_local, "mode_guards"):
            ModeIsolationConfig._thread_local.mode_guards = []
        ModeIsolationConfig._thread_local.mode_guards.append(self)

        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Exit guarded section."""
        # Remove guard from thread-local
        if hasattr(ModeIsolationConfig._thread_local, "mode_guards"):
            guards = ModeIsolationConfig._thread_local.mode_guards
            if self in guards:
                guards.remove(self)

        # Restore original mode
        if self.original_mode is not None:
            WheelModeConfig.set_mode(self.original_mode)

        self.locked = False

    @classmethod
    def is_mode_locked(cls) -> bool:
        """Check if mode is currently locked by a guard."""
        if hasattr(ModeIsolationConfig._thread_local, "mode_guards"):
            return len(ModeIsolationConfig._thread_local.mode_guards) > 0
        return False


def compile_time_switch(tr_impl: Callable, wheel_impl: Callable) -> Callable:
    """
    Create a compile-time switch between TR and Wheel implementations.

    This ensures completely separate code paths for each mode,
    preventing any mixing of semantics.

    Args:
        tr_impl: Implementation for TR mode
        wheel_impl: Implementation for Wheel mode

    Returns:
        Function that switches based on current mode
    """

    @wraps(tr_impl)
    def switcher(*args, **kwargs):
        # Check mode at call time (compile-time in spirit)
        if WheelModeConfig.is_wheel():
            return wheel_impl(*args, **kwargs)
        else:
            return tr_impl(*args, **kwargs)

    # Mark as mode-switching function
    switcher._is_mode_switch = True
    switcher._tr_impl = tr_impl
    switcher._wheel_impl = wheel_impl

    return switcher


class ModeValidator:
    """
    Validator to ensure mode consistency across a computation.
    """

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, raise errors; if False, issue warnings
        """
        self.strict = strict
        self.mode_history: List[ArithmeticMode] = []
        self.value_history: List[TRTag] = []

    def record_operation(
        self, op_name: str, mode: ArithmeticMode, inputs: List[TRScalar], output: TRScalar
    ) -> None:
        """Record an operation for validation."""
        self.mode_history.append(mode)

        # Check for mode-inappropriate values
        if mode == ArithmeticMode.TRANSREAL:
            # Check for BOTTOM in inputs/output
            for val in inputs + [output]:
                if val.tag == TRTag.BOTTOM:
                    msg = f"BOTTOM value in TR mode operation {op_name}"
                    if self.strict:
                        raise ModeViolationError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning)

        # Record tags
        for val in inputs + [output]:
            self.value_history.append(val.tag)

    def validate_consistency(self) -> bool:
        """
        Validate that modes were consistent.

        Returns:
            True if validation passes
        """
        if not self.mode_history:
            return True

        # Check for mode changes
        first_mode = self.mode_history[0]
        for i, mode in enumerate(self.mode_history[1:], 1):
            if mode != first_mode:
                msg = f"Mode changed from {first_mode} to {mode} at operation {i}"
                if self.strict:
                    raise ModeViolationError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning)
                    return False

        return True


def ensure_mode_purity(mode: ArithmeticMode) -> Callable[[F], F]:
    """
    Decorator to ensure a function runs in a specific mode.

    This guarantees the function executes entirely in the specified
    mode, preventing any mixing of semantics.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ModeSwitchGuard(mode):
                return func(*args, **kwargs)

        # Mark function with its required mode
        wrapper._required_mode = mode
        return wrapper

    return decorator


# Export convenient decorators
tr_only = ensure_mode_purity(ArithmeticMode.TRANSREAL)
wheel_only = ensure_mode_purity(ArithmeticMode.WHEEL)


def validate_mode_transition(
    from_mode: ArithmeticMode, to_mode: ArithmeticMode, values: List[TRScalar]
) -> None:
    """
    Validate that a mode transition is safe.

    Args:
        from_mode: Mode transitioning from
        to_mode: Mode transitioning to
        values: Values that need to remain valid

    Raises:
        ModeViolationError: If transition would invalidate values
    """
    if from_mode == to_mode:
        return  # No transition

    if from_mode == ArithmeticMode.WHEEL and to_mode == ArithmeticMode.TRANSREAL:
        # Check for BOTTOM values that would be invalid in TR
        for val in values:
            if val.tag == TRTag.BOTTOM:
                raise ModeViolationError(
                    "Cannot transition from Wheel to TR mode with BOTTOM values. "
                    "BOTTOM (⊥) has no meaning in Transreal arithmetic."
                )

    # TR to Wheel is always safe (PHI values remain valid)


class IsolatedModule:
    """
    Base class for modules that require mode isolation.

    Subclasses can specify their required mode and all methods
    will automatically enforce it.
    """

    _required_mode: Optional[ArithmeticMode] = None

    def __init_subclass__(cls, mode: Optional[ArithmeticMode] = None, **kwargs):
        """
        Initialize subclass with mode requirement.

        Args:
            mode: Required arithmetic mode for this module
        """
        super().__init_subclass__(**kwargs)
        cls._required_mode = mode

        # Wrap all methods to enforce mode
        if mode is not None:
            for name in dir(cls):
                if not name.startswith("_"):
                    attr = getattr(cls, name)
                    if callable(attr):
                        wrapped = ensure_mode_purity(mode)(attr)
                        setattr(cls, name, wrapped)


# Validation utilities for testing
def test_mode_isolation() -> bool:
    """
    Test that mode isolation is working correctly.

    Returns:
        True if all isolation tests pass
    """
    from .tr_ops import tr_add

    # Test 1: Operations lock mode
    with ModeSwitchGuard(ArithmeticMode.TRANSREAL):
        _result = tr_add(real(1.0), real(2.0))

        # Try to change mode within operation (should fail in strict implementations)
        try:
            with ModeSwitchGuard(ArithmeticMode.WHEEL):
                # This should be prevented
                pass
            return False  # Should not reach here
        except Exception:
            pass  # Expected

    # Test 2: Wheel axioms only in Wheel mode
    WheelModeConfig.set_mode(ArithmeticMode.TRANSREAL)
    try:
        WheelAxioms.zero_times_infinity()
        return False  # Should not reach here
    except ModeViolationError:
        pass  # Expected

    # Test 3: BOTTOM only in Wheel mode
    WheelModeConfig.set_mode(ArithmeticMode.TRANSREAL)
    try:
        check_value_mode_compatibility(bottom())
        return False  # Should not reach here
    except ModeViolationError:
        pass  # Expected

    return True
