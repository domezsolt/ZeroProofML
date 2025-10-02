"""
JAX bridge for transreal arithmetic.

This module provides conversions between JAX arrays and transreal
representations, supporting functional transformations and JIT compilation.
"""

import warnings
from functools import partial
from typing import Any, NamedTuple, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    custom_vjp = None

from ..core import TRScalar, TRTag, ninf, phi, pinf, real


class TRJaxArray(NamedTuple):
    """
    Transreal array for JAX.

    Uses NamedTuple for immutability and JAX pytree registration.
    """

    values: Any  # jax.Array for values
    tags: Any  # jax.Array for tags (uint8)

    @property
    def shape(self):
        """Get array shape."""
        return self.values.shape

    @property
    def dtype(self):
        """Get value dtype."""
        return self.values.dtype

    @property
    def ndim(self):
        """Get number of dimensions."""
        return self.values.ndim

    def is_real(self):
        """Return boolean mask of REAL elements."""
        return self.tags == TAG_CODES["REAL"]

    def is_pinf(self):
        """Return boolean mask of PINF elements."""
        return self.tags == TAG_CODES["PINF"]

    def is_ninf(self):
        """Return boolean mask of NINF elements."""
        return self.tags == TAG_CODES["NINF"]

    def is_phi(self):
        """Return boolean mask of PHI elements."""
        return self.tags == TAG_CODES["PHI"]

    def is_finite(self):
        """Return boolean mask of finite (REAL) elements."""
        return self.is_real()

    def is_infinite(self):
        """Return boolean mask of infinite (PINF/NINF) elements."""
        return (self.tags == TAG_CODES["PINF"]) | (self.tags == TAG_CODES["NINF"])


# Tag encoding constants
TAG_CODES = {
    "REAL": 0,
    "PINF": 1,
    "NINF": 2,
    "PHI": 3,
}

TAG_TO_CODE = {
    TRTag.REAL: 0,
    TRTag.PINF: 1,
    TRTag.NINF: 2,
    TRTag.PHI: 3,
}

CODE_TO_TAG = {v: k for k, v in TAG_TO_CODE.items()}


# Register TRJaxArray as a JAX pytree
if JAX_AVAILABLE:
    from jax.tree_util import register_pytree_node

    def _tr_flatten(tr_array):
        """Flatten TRJaxArray for pytree."""
        return (tr_array.values, tr_array.tags), None

    def _tr_unflatten(_aux_data, children):
        """Unflatten TRJaxArray from pytree."""
        values, tags = children
        return TRJaxArray(values, tags)

    register_pytree_node(TRJaxArray, _tr_flatten, _tr_unflatten)


def from_jax(array: "jax.Array") -> TRJaxArray:
    """
    Convert JAX array to transreal array.

    Args:
        array: JAX array

    Returns:
        TRJaxArray with appropriate tags
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for from_jax")

    # Create value and tag arrays
    values = array
    tags = jnp.zeros_like(array, dtype=jnp.uint8)

    # Classify elements
    finite_mask = jnp.isfinite(array)
    nan_mask = jnp.isnan(array)
    posinf_mask = jnp.isposinf(array)
    neginf_mask = jnp.isneginf(array)

    # Set tags using JAX operations
    tags = jnp.where(finite_mask, TAG_CODES["REAL"], tags)
    tags = jnp.where(nan_mask, TAG_CODES["PHI"], tags)
    tags = jnp.where(posinf_mask, TAG_CODES["PINF"], tags)
    tags = jnp.where(neginf_mask, TAG_CODES["NINF"], tags)

    return TRJaxArray(values, tags)


def to_jax(tr_array: TRJaxArray) -> "jax.Array":
    """
    Convert TRJaxArray to JAX array (IEEE representation).

    Args:
        tr_array: Transreal array

    Returns:
        JAX array with appropriate IEEE values
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for to_jax")

    # Map each tag type to IEEE values
    real_mask = tr_array.is_real()
    pinf_mask = tr_array.is_pinf()
    ninf_mask = tr_array.is_ninf()
    phi_mask = tr_array.is_phi()

    # Use JAX operations for conversion
    result = jnp.where(real_mask, tr_array.values, 0.0)
    result = jnp.where(pinf_mask, jnp.inf, result)
    result = jnp.where(ninf_mask, -jnp.inf, result)
    result = jnp.where(phi_mask, jnp.nan, result)

    return result


# Mask-REAL gradient rule for JAX
def mask_real_grad(grad_output: "jax.Array", tags: "jax.Array") -> "jax.Array":
    """
    Apply Mask-REAL rule to gradients.

    Args:
        grad_output: Gradient from downstream
        tags: Tag array

    Returns:
        Masked gradient (zero where tags are non-REAL)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    real_mask = tags == TAG_CODES["REAL"]
    return grad_output * real_mask.astype(grad_output.dtype)


if JAX_AVAILABLE:
    # Custom gradient rules for TR operations (only when JAX is available)
    @partial(custom_vjp, nondiff_argnums=(2,))
    def tr_op_with_grad(values: "jax.Array", tags: "jax.Array", op_fn):
        """
        Generic TR operation with automatic Mask-REAL gradient.
        """
        return op_fn(values, tags)

    def tr_op_fwd(values, tags, op_fn):
        """Forward pass for TR operation."""
        output_values, output_tags = op_fn(values, tags)
        return (output_values, output_tags), (output_tags,)

    def tr_op_bwd(op_fn, residuals, grads):
        """Backward pass with Mask-REAL rule."""
        (output_tags,) = residuals
        grad_values, grad_tags = grads
        masked_grad = mask_real_grad(grad_values, output_tags)
        return (masked_grad, None)

    # Register the custom gradient
    tr_op_with_grad.defvjp(tr_op_fwd, tr_op_bwd)
else:
    # Stubs when JAX is unavailable
    def tr_op_with_grad(*args, **kwargs):
        raise ImportError("JAX is required for tr_op_with_grad")


# JAX-compatible TR operations
@jax.jit
def tr_add_jax(a: TRJaxArray, b: TRJaxArray) -> TRJaxArray:
    """
    Transreal addition for JAX arrays.

    Args:
        a, b: TRJaxArray inputs

    Returns:
        TRJaxArray result
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    # Extract values and tags
    a_val, a_tag = a.values, a.tags
    b_val, b_tag = b.values, b.tags

    # PHI propagates
    phi_mask = (a_tag == TAG_CODES["PHI"]) | (b_tag == TAG_CODES["PHI"])

    # ∞ + (-∞) = PHI
    inf_conflict = ((a_tag == TAG_CODES["PINF"]) & (b_tag == TAG_CODES["NINF"])) | (
        (a_tag == TAG_CODES["NINF"]) & (b_tag == TAG_CODES["PINF"])
    )

    # Compute result
    result_val = a_val + b_val

    # Determine result tags
    result_tag = jnp.where(
        phi_mask | inf_conflict,
        TAG_CODES["PHI"],
        jnp.where(
            (a_tag == TAG_CODES["PINF"]) | (b_tag == TAG_CODES["PINF"]),
            TAG_CODES["PINF"],
            jnp.where(
                (a_tag == TAG_CODES["NINF"]) | (b_tag == TAG_CODES["NINF"]),
                TAG_CODES["NINF"],
                TAG_CODES["REAL"],
            ),
        ),
    )

    return TRJaxArray(result_val, result_tag)


@jax.jit
def tr_mul_jax(a: TRJaxArray, b: TRJaxArray) -> TRJaxArray:
    """
    Transreal multiplication for JAX arrays.

    Args:
        a, b: TRJaxArray inputs

    Returns:
        TRJaxArray result
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    a_val, a_tag = a.values, a.tags
    b_val, b_tag = b.values, b.tags

    # PHI propagates
    phi_mask = (a_tag == TAG_CODES["PHI"]) | (b_tag == TAG_CODES["PHI"])

    # 0 * ∞ = PHI
    a_zero = (a_tag == TAG_CODES["REAL"]) & (a_val == 0.0)
    b_zero = (b_tag == TAG_CODES["REAL"]) & (b_val == 0.0)
    a_inf = (a_tag == TAG_CODES["PINF"]) | (a_tag == TAG_CODES["NINF"])
    b_inf = (b_tag == TAG_CODES["PINF"]) | (b_tag == TAG_CODES["NINF"])
    zero_inf = (a_zero & b_inf) | (b_zero & a_inf)

    # Compute result value
    result_val = a_val * b_val

    # Determine result tags (simplified for demo)
    result_tag = jnp.where(
        phi_mask | zero_inf,
        TAG_CODES["PHI"],
        jnp.where(
            jnp.isfinite(result_val),
            TAG_CODES["REAL"],
            jnp.where(result_val > 0, TAG_CODES["PINF"], TAG_CODES["NINF"]),
        ),
    )

    return TRJaxArray(result_val, result_tag)


# Utility functions for JAX
def vmap_tr_scalar_fn(fn, _in_axes=0, _out_axes=0):
    """
    Vectorize a function that operates on TRScalars.

    Args:
        fn: Function taking TRScalar inputs
        in_axes: Input axes for vmap
        out_axes: Output axes for vmap

    Returns:
        Vectorized function operating on TRJaxArrays
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    def wrapped_fn(tr_array):
        # This is a placeholder - full implementation would properly
        # vectorize scalar TR operations for JAX
        raise NotImplementedError("Scalar TR function vectorization not yet implemented")

    return wrapped_fn


# Integration with JAX transformations
def make_jaxpr_with_tr(fn):
    """
    Create JAX intermediate representation for TR function.

    This helps with debugging and optimization.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    return jax.make_jaxpr(fn)


# Conversion utilities
def tr_scalar_to_jax(scalar: TRScalar, shape=()) -> TRJaxArray:
    """
    Convert TRScalar to TRJaxArray.

    Args:
        scalar: TRScalar value
        shape: Desired output shape (default: scalar)

    Returns:
        TRJaxArray
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    if scalar.tag == TRTag.REAL:
        value = scalar.value
    elif scalar.tag == TRTag.PINF:
        value = jnp.inf
    elif scalar.tag == TRTag.NINF:
        value = -jnp.inf
    else:  # PHI
        value = jnp.nan

    values = jnp.full(shape, value)
    tags = jnp.full(shape, TAG_TO_CODE[scalar.tag], dtype=jnp.uint8)

    return TRJaxArray(values, tags)


def batch_from_scalars_jax(scalars: list[TRScalar]) -> TRJaxArray:
    """
    Create TRJaxArray batch from list of TRScalars.

    Args:
        scalars: List of TRScalar values

    Returns:
        TRJaxArray with shape (len(scalars),)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")

    values = []
    tags = []

    for scalar in scalars:
        if scalar.tag == TRTag.REAL:
            values.append(scalar.value)
        elif scalar.tag == TRTag.PINF:
            values.append(jnp.inf)
        elif scalar.tag == TRTag.NINF:
            values.append(-jnp.inf)
        else:  # PHI
            values.append(jnp.nan)
        tags.append(TAG_TO_CODE[scalar.tag])

    return TRJaxArray(jnp.array(values), jnp.array(tags, dtype=jnp.uint8))
