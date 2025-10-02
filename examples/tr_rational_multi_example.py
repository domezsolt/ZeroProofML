"""Quick example demonstrating TRRationalMulti usage.

Run:
	python examples/tr_rational_multi_example.py
"""

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import ChebyshevBasis, MonomialBasis, TRRationalMulti


def demo_shared_q():
    print("=== TRRationalMulti with shared Q ===")
    layer = TRRationalMulti(d_p=1, d_q=1, n_outputs=2, shared_Q=True, basis=ChebyshevBasis())

    # Set different numerators per output, shared denominator
    # Output 0: P0(x) = 1 + 0*x; Q(x) = 1 + 0.5*T1(x) = 1 + 0.5*x
    layer.layers[0].theta[0]._value = real(1.0)
    layer.layers[0].theta[1]._value = real(0.0)

    # Output 1: P1(x) = 0 + 1*x
    layer.layers[1].theta[0]._value = real(0.0)
    layer.layers[1].theta[1]._value = real(1.0)

    # Shared denominator coefficients φ
    layer.layers[0].phi[0]._value = real(0.5)

    # Forward on a few inputs
    for xv in [-0.5, 0.0, 0.5, 1.0]:
        x = TRNode.constant(real(xv))
        outputs = layer(x)
        vals = [y.value.value if y.tag == TRTag.REAL else y.tag.name for y in outputs]
        print(f"x={xv: .2f} -> outputs={vals}")

    # Show parameter summary
    params = layer.parameters()
    print(f"num parameters (shared Q counted once): {len(params)}")
    print(f"regularization loss: {layer.regularization_loss().value.value:.6f}")
    print()


def demo_independent_q():
    print("=== TRRationalMulti with independent Q ===")
    layer = TRRationalMulti(d_p=1, d_q=1, n_outputs=2, shared_Q=False, basis=MonomialBasis())

    # Output 0: y0 = (1 + 0*x)/(1 + 0.5*x)
    layer.layers[0].theta[0]._value = real(1.0)
    layer.layers[0].theta[1]._value = real(0.0)
    layer.layers[0].phi[0]._value = real(0.5)

    # Output 1: y1 = (0 + 1*x)/(1 - 0.5*x)
    layer.layers[1].theta[0]._value = real(0.0)
    layer.layers[1].theta[1]._value = real(1.0)
    layer.layers[1].phi[0]._value = real(-0.5)

    for xv in [-0.5, 0.0, 0.5, 1.0]:
        x = TRNode.constant(real(xv))
        outputs = layer(x)
        vals = [y.value.value if y.tag == TRTag.REAL else y.tag.name for y in outputs]
        print(f"x={xv: .2f} -> outputs={vals}")

    # Parameter summary (no sharing)
    params = layer.parameters()
    print(f"num parameters (independent): {len(params)}")
    print(f"regularization loss: {layer.regularization_loss().value.value:.6f}")
    print()


if __name__ == "__main__":
    demo_shared_q()
    demo_independent_q()
