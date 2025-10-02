"""
Demonstration of TR-Rational layers and TR-Norm.

This example shows how these layers handle singularities gracefully
and maintain stable gradients through the Mask-REAL rule.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zeroproof as zp
from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.layers import ChebyshevBasis, TRLayerNorm, TRNorm, TRRational


def demonstrate_tr_rational():
    """Show TR-Rational layer in action."""
    print("=== TR-Rational Layer Demo ===\n")

    # Create a rational layer: y = P(x) / Q(x)
    # with P degree 2, Q degree 1
    layer = TRRational(d_p=2, d_q=1)

    # Set specific coefficients for demonstration
    # P(x) = 1 + 2x + x²
    layer.theta[0]._value = zp.real(1.0)
    layer.theta[1]._value = zp.real(2.0)
    layer.theta[2]._value = zp.real(1.0)

    # Q(x) = 1 - x (has pole at x=1)
    layer.phi[0]._value = zp.real(-1.0)

    print("Rational function: P(x)/Q(x)")
    print("P(x) = 1 + 2x + x²")
    print("Q(x) = 1 - x")
    print("Pole at x = 1\n")

    # Test at various points
    test_points = [-1.0, 0.0, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0]

    print("x      | P(x)   | Q(x)   | y=P/Q  | tag")
    print("-------|--------|--------|--------|------")

    for x_val in test_points:
        x = TRNode.constant(zp.real(x_val))

        # Compute P and Q for display
        p_val = 1 + 2 * x_val + x_val**2
        q_val = 1 - x_val

        # Forward pass
        y, tag = layer.forward(x)

        # Format output
        if tag == zp.TRTag.REAL:
            y_str = f"{y.value.value:7.2f}"
        else:
            y_str = f"{tag.name:>7}"

        print(f"{x_val:6.2f} | {p_val:6.2f} | {q_val:6.2f} | {y_str} | {tag.name}")


def demonstrate_rational_gradients():
    """Show gradient computation through rational layer."""
    print("\n=== Rational Layer Gradients ===\n")

    layer = TRRational(d_p=1, d_q=1)

    # Simple function: y = x / (x + 2)
    layer.theta[0]._value = zp.real(0.0)
    layer.theta[1]._value = zp.real(1.0)
    layer.phi[0]._value = zp.real(2.0)

    print("Function: y = x / (x + 2)")
    print("Analytical derivative: dy/dx = 2 / (x + 2)²\n")

    # Test gradient at various points
    test_points = [-3.0, -2.5, -2.1, -2.0, -1.9, -1.0, 0.0, 1.0]

    print("x      | y      | dy/dx (analytical) | dy/dx (autodiff) | Match?")
    print("-------|--------|-------------------|------------------|-------")

    for x_val in test_points:
        with gradient_tape() as tape:
            x = TRNode.parameter(zp.real(x_val))
            tape.watch(x)
            y = layer(x)

        # Compute gradient
        grads = tape.gradient(y, [x])

        # Analytical gradient (if defined)
        if x_val != -2.0:
            analytical = 2.0 / (x_val + 2.0) ** 2
            analytical_str = f"{analytical:17.4f}"
        else:
            analytical_str = "      undefined   "

        # Format output
        if y.tag == zp.TRTag.REAL:
            y_str = f"{y.value.value:6.3f}"
        else:
            y_str = f"{y.tag.name:>6}"

        if grads[0] is not None:
            grad_str = f"{grads[0].value.value:16.4f}"
            if x_val != -2.0:
                match = "✓" if abs(grads[0].value.value - analytical) < 1e-10 else "✗"
            else:
                match = "N/A"
        else:
            grad_str = "      None      "
            match = "N/A"

        print(f"{x_val:6.2f} | {y_str} | {analytical_str} | {grad_str} | {match:^6}")

    print("\nNote: At x=-2 (pole), gradient is 0 due to Mask-REAL rule.")


def demonstrate_tr_norm():
    """Show TR-Norm in action."""
    print("\n=== TR-Norm Demo ===\n")

    # Create normalization layer
    norm = TRNorm(num_features=2)

    print("Case 1: Normal batch with variance > 0")
    print("--------------------------------------")

    # Batch with normal variance
    batch1 = [
        [zp.real(1.0), zp.real(10.0)],
        [zp.real(3.0), zp.real(14.0)],
        [zp.real(5.0), zp.real(18.0)],
        [zp.real(7.0), zp.real(22.0)],
    ]

    print("Input batch:")
    for i, sample in enumerate(batch1):
        print(f"  Sample {i}: [{sample[0].value:.1f}, {sample[1].value:.1f}]")

    # Forward pass
    output1 = norm(batch1)

    print("\nNormalized output:")
    for i, sample in enumerate(output1):
        vals = [s.value.value for s in sample]
        print(f"  Sample {i}: [{vals[0]:6.3f}, {vals[1]:6.3f}]")

    # Compute statistics to verify
    for feat in range(2):
        values = [output1[i][feat].value.value for i in range(4)]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        print(f"\nFeature {feat} stats: mean = {mean:.6f}, var = {var:.6f}")

    print("\n\nCase 2: Batch with zero variance (bypass)")
    print("-----------------------------------------")

    # Batch with zero variance
    batch2 = [
        [zp.real(5.0), zp.real(7.0)],
        [zp.real(5.0), zp.real(7.0)],
        [zp.real(5.0), zp.real(7.0)],
    ]

    # Set beta values to see bypass effect
    norm.beta[0]._value = zp.real(100.0)
    norm.beta[1]._value = zp.real(200.0)

    print("Input batch (all identical):")
    for i, sample in enumerate(batch2):
        print(f"  Sample {i}: [{sample[0].value:.1f}, {sample[1].value:.1f}]")

    print(f"\nBeta parameters: β₀ = {norm.beta[0].value.value}, β₁ = {norm.beta[1].value.value}")

    # Forward pass
    output2 = norm(batch2)

    print("\nOutput (bypassed to β):")
    for i, sample in enumerate(output2):
        vals = [s.value.value for s in sample]
        print(f"  Sample {i}: [{vals[0]:6.1f}, {vals[1]:6.1f}]")


def demonstrate_norm_with_non_real():
    """Show TR-Norm handling non-REAL values."""
    print("\n=== TR-Norm with Non-REAL Values ===\n")

    norm = TRNorm(num_features=1, affine=False)

    # Create batch with mixed REAL and non-REAL
    print("Input batch with infinities and PHI:")
    batch = [
        [zp.real(1.0)],
        [TRNode.constant(zp.pinf())],  # +∞
        [zp.real(3.0)],
        [TRNode.constant(zp.phi())],  # Φ
        [zp.real(5.0)],
        [TRNode.constant(zp.ninf())],  # -∞
        [zp.real(7.0)],
    ]

    for i, sample in enumerate(batch):
        tag = sample[0].tag.name
        if sample[0].tag == zp.TRTag.REAL:
            print(f"  Sample {i}: {sample[0].value.value:.1f} ({tag})")
        else:
            print(f"  Sample {i}: {tag}")

    # Forward pass
    output = norm(batch)

    print("\nNormalized output (stats from REAL values only):")
    print("REAL values: 1, 3, 5, 7 → mean=4, std≈2.58")

    for i, sample in enumerate(output):
        if sample[0].tag == zp.TRTag.REAL:
            print(f"  Sample {i}: {sample[0].value.value:6.3f} ({sample[0].tag.name})")
        else:
            print(f"  Sample {i}: {sample[0].tag.name}")


def demonstrate_chebyshev_basis():
    """Show different basis functions."""
    print("\n=== Basis Functions Demo ===\n")

    # Compare monomial vs Chebyshev
    layer_mono = TRRational(d_p=3, d_q=1)
    layer_cheb = TRRational(d_p=3, d_q=1, basis=ChebyshevBasis())

    print("Comparing Monomial vs Chebyshev basis (degree 3)")
    print("Domain: [-1, 1]\n")

    x_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]

    print("x     | Mono: 1    x      x²     x³   | Cheb: T₀   T₁    T₂     T₃")
    print("------|------------------------------|--------------------------------")

    for x_val in x_vals:
        # Monomial basis
        mono_vals = [1.0, x_val, x_val**2, x_val**3]

        # Chebyshev values
        T0 = 1.0
        T1 = x_val
        T2 = 2 * x_val**2 - 1
        T3 = 4 * x_val**3 - 3 * x_val
        cheb_vals = [T0, T1, T2, T3]

        # Format output
        mono_str = " ".join(f"{v:6.2f}" for v in mono_vals)
        cheb_str = " ".join(f"{v:6.2f}" for v in cheb_vals)

        print(f"{x_val:5.1f} | {mono_str} | {cheb_str}")

    print("\nNote: Chebyshev polynomials are bounded by [-1,1] on the domain,")
    print("making them numerically stable for high-degree approximations.")


def demonstrate_layer_norm():
    """Show layer normalization."""
    print("\n=== Layer Normalization Demo ===\n")

    ln = TRLayerNorm(4)

    print("Layer norm normalizes across features for each sample.\n")

    # Single sample
    sample = [zp.real(2.0), zp.real(4.0), zp.real(6.0), zp.real(8.0)]

    print("Input features:", [x.value for x in sample])
    print("Mean:", sum(x.value for x in sample) / len(sample))

    output = ln(sample)

    print("\nNormalized features:")
    for i, out in enumerate(output):
        print(f"  Feature {i}: {out.value.value:7.4f}")

    # Verify statistics
    values = [out.value.value for out in output]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)

    print(f"\nOutput statistics: mean = {mean:.6f}, var = {var:.6f}")


if __name__ == "__main__":
    print("ZeroProof: TR-Rational and TR-Norm Demo")
    print("=======================================\n")

    demonstrate_tr_rational()
    demonstrate_rational_gradients()
    demonstrate_tr_norm()
    demonstrate_norm_with_non_real()
    demonstrate_chebyshev_basis()
    demonstrate_layer_norm()

    print("\n=======================================")
    print("TR layers handle singularities gracefully!")
    print("No NaN propagation, stable gradients via Mask-REAL.")
