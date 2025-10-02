"""
Demonstration of extended IEEE bridge functionality.

This example shows how to use transreal arithmetic with NumPy, PyTorch,
JAX, and different precision levels.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zeroproof as zp
from zeroproof.bridge import (
    JAX_AVAILABLE,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    Precision,
    PrecisionContext,
    analyze_precision_requirements,
    from_ieee,
    get_precision_info,
    to_ieee,
    with_precision,
)


def demonstrate_precision_handling():
    """Show precision-aware TR computations."""
    print("=== Precision Handling ===\n")

    # Show precision limits

    for prec in [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]:
        info = get_precision_info(prec)
        print(f"{prec.value}:")
        print(f"  Max value: {info['max_value']:.2e}")
        print(f"  Min normal: {info['min_normal']:.2e}")
        print(f"  Epsilon: {info['epsilon']:.2e}")
        print()

    # Demonstrate overflow handling
    print("Overflow handling:")
    large_value = 1e10
    tr_val = zp.real(large_value)

    for prec in [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]:
        result = with_precision(tr_val, prec)
        print(f"  {large_value} in {prec.value}: {result}")

    # Precision context
    print("\nPrecision context:")
    with PrecisionContext(Precision.FLOAT32):
        print(f"  Current precision: {PrecisionContext.get_current_precision().value}")

        # Operations use current precision
        a = zp.real(1e20)
        b = zp.real(1e20)
        result = zp.tr_mul(a, b)  # Would overflow in float32
        result_prec = with_precision(result)
        print(f"  1e20 * 1e20 = {result_prec}")


def demonstrate_numpy_bridge():
    """Show NumPy integration."""
    if not NUMPY_AVAILABLE:
        print("\n=== NumPy Bridge (Not Available) ===")
        print("Install NumPy to use this feature: pip install numpy")
        return

    print("\n=== NumPy Bridge ===\n")

    import numpy as np

    from zeroproof.bridge import TRArray, count_tags, from_numpy, to_numpy

    # Convert NumPy array to TR
    print("Array conversion:")
    arr = np.array([[1.0, 2.0, 3.0], [0.0, np.inf, -np.inf], [np.nan, 4.0, 5.0]])
    print(f"Input array:\n{arr}")

    tr_arr = from_numpy(arr)
    print(f"\nTR array: {tr_arr}")
    print(f"Shape: {tr_arr.shape}")
    print(f"Tag counts: {count_tags(tr_arr)}")

    # Extract REAL values
    from zeroproof.bridge import real_values

    reals = real_values(tr_arr)
    print(f"\nREAL values only: {reals}")

    # Convert back
    ieee_arr = to_numpy(tr_arr)
    print(f"\nRound-trip successful: {np.array_equal(arr, ieee_arr, equal_nan=True)}")

    # Demonstrate masking
    print("\nMasking operations:")
    print(f"REAL mask:\n{tr_arr.is_real()}")
    print(f"Infinite mask:\n{tr_arr.is_infinite()}")

    # Clip infinities for libraries that don't handle them
    from zeroproof.bridge import clip_infinities

    clipped = clip_infinities(tr_arr, max_value=1e6)
    print(f"\nClipped array (inf → 1e6):\n{to_numpy(clipped)}")


def demonstrate_torch_bridge():
    """Show PyTorch integration."""
    if not TORCH_AVAILABLE:
        print("\n=== PyTorch Bridge (Not Available) ===")
        print("Install PyTorch to use this feature: pip install torch")
        return

    print("\n=== PyTorch Bridge ===\n")

    import torch

    from zeroproof.bridge import TRTensor, from_torch, to_torch

    # Create tensor with special values
    print("Tensor conversion:")
    tensor = torch.tensor([1.0, 2.0, float("inf"), float("nan"), -3.0])
    print(f"Input tensor: {tensor}")

    tr_tensor = from_torch(tensor, requires_grad=True)
    print(f"TR tensor: {tr_tensor}")

    # Check tags
    print(f"\nTag analysis:")
    print(f"  REAL elements: {tr_tensor.is_real().sum().item()}")
    print(f"  PINF elements: {tr_tensor.is_pinf().sum().item()}")
    print(f"  PHI elements: {tr_tensor.is_phi().sum().item()}")

    # Gradient computation with Mask-REAL
    print("\nGradient computation:")
    ieee_tensor = to_torch(tr_tensor)

    # Simple function: sum of squares
    output = (ieee_tensor**2).sum()
    output.backward()

    print(f"Input gradient: {tensor.grad}")
    print("Note: Gradients are zero for non-REAL elements (Mask-REAL)")

    # Device handling
    if torch.cuda.is_available():
        print("\nGPU support:")
        cuda_tr = tr_tensor.cuda()
        print(f"  On GPU: {cuda_tr.device}")

        cpu_tr = cuda_tr.cpu()
        print(f"  Back to CPU: {cpu_tr.device}")


def demonstrate_jax_bridge():
    """Show JAX integration."""
    if not JAX_AVAILABLE:
        print("\n=== JAX Bridge (Not Available) ===")
        print("Install JAX to use this feature: pip install jax jaxlib")
        return

    print("\n=== JAX Bridge ===\n")

    import jax.numpy as jnp

    from zeroproof.bridge import TRJaxArray, from_jax, to_jax, tr_add_jax, tr_mul_jax

    # Create JAX array
    print("JAX array conversion:")
    arr = jnp.array([1.0, 2.0, jnp.inf, -jnp.inf, jnp.nan])
    print(f"Input array: {arr}")

    tr_arr = from_jax(arr)
    print(f"TR array: {tr_arr}")

    # TR operations in JAX
    print("\nTR operations:")
    a = from_jax(jnp.array([1.0, jnp.inf, 0.0]))
    b = from_jax(jnp.array([2.0, -jnp.inf, jnp.inf]))

    # Addition
    sum_result = tr_add_jax(a, b)
    print(f"a + b: {to_jax(sum_result)}")
    print(f"  Tags: REAL={sum_result.is_real().sum()}, PHI={sum_result.is_phi().sum()}")

    # Multiplication
    mul_result = tr_mul_jax(a, b)
    print(f"a * b: {to_jax(mul_result)}")
    print(f"  Tags: REAL={mul_result.is_real().sum()}, PHI={mul_result.is_phi().sum()}")

    # JAX transformations work with TR arrays
    print("\nJAX pytree registration allows transformations:")
    print(f"  Tree structure: {tr_arr}")


def demonstrate_mixed_precision():
    """Show mixed precision strategies."""
    print("\n=== Mixed Precision ===\n")

    from zeroproof.bridge import MixedPrecisionStrategy

    # Create strategy
    strategy = MixedPrecisionStrategy(
        compute_precision=Precision.FLOAT16,
        accumulate_precision=Precision.FLOAT32,
        output_precision=Precision.FLOAT16,
    )

    print("Strategy:")
    print(f"  Compute: {strategy.compute_precision.value}")
    print(f"  Accumulate: {strategy.accumulate_precision.value}")
    print(f"  Output: {strategy.output_precision.value}")

    # Demonstrate precision analysis
    print("\nPrecision analysis:")
    values = [1e-5, 1.0, 100.0, 1e4, 1e6]
    analysis = analyze_precision_requirements(values)

    print(f"Values: {values}")
    print(f"Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")


def demonstrate_cross_framework():
    """Show cross-framework compatibility."""
    print("\n=== Cross-Framework Compatibility ===\n")

    # Start with TR scalar
    tr_val = zp.real(3.14159)
    print(f"Original TR value: {tr_val}")

    # Convert to IEEE
    ieee_val = to_ieee(tr_val)
    print(f"IEEE value: {ieee_val}")

    # Through each framework (if available)
    if NUMPY_AVAILABLE:
        import numpy as np

        from zeroproof.bridge import from_numpy, to_numpy

        np_val = np.array(ieee_val)
        tr_from_np = from_numpy(np_val)
        back_to_ieee = to_numpy(tr_from_np)
        print(f"Through NumPy: {back_to_ieee}")

    if TORCH_AVAILABLE:
        import torch

        from zeroproof.bridge import from_torch, to_torch

        torch_val = torch.tensor(ieee_val)
        tr_from_torch = from_torch(torch_val)
        back_to_ieee = to_torch(tr_from_torch).item()
        print(f"Through PyTorch: {back_to_ieee}")

    if JAX_AVAILABLE:
        import jax.numpy as jnp

        from zeroproof.bridge import from_jax, to_jax

        jax_val = jnp.array(ieee_val)
        tr_from_jax = from_jax(jax_val)
        back_to_ieee = to_jax(tr_from_jax).item()
        print(f"Through JAX: {back_to_ieee}")

    print("\nAll conversions preserve the value!")


if __name__ == "__main__":
    print("ZeroProof: Extended Bridge Functionality Demo")
    print("============================================\n")

    demonstrate_precision_handling()
    demonstrate_numpy_bridge()
    demonstrate_torch_bridge()
    demonstrate_jax_bridge()
    demonstrate_mixed_precision()
    demonstrate_cross_framework()

    print("\n============================================")
    print("Transreal arithmetic works seamlessly with")
    print("your favorite numerical computing libraries!")
