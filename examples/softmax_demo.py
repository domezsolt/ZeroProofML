"""
Minimal demo for TR softmax surrogate.

Run:
  python examples/softmax_demo.py
"""

from zeroproof.autodiff import TRNode
from zeroproof.core import real
from zeroproof.layers import tr_softmax


def main():
    logits = [0.0, 1.5, -0.5, 0.7]
    nodes = [TRNode.constant(real(x)) for x in logits]
    probs = tr_softmax(nodes)
    vals = []
    for p in probs:
        if p.tag.name == "REAL":
            vals.append(float(p.value.value))
        else:
            vals.append(float("nan"))
    print("logits:", logits)
    print("probs:", vals)
    print("sum:", sum(v for v in vals if v == v))


if __name__ == "__main__":
    main()
