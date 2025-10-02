import pytest

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import ChebyshevBasis


def test_chebyshev_scalar_and_node_no_unboundlocal():
    base = ChebyshevBasis(domain=(-1, 1))

    # TRScalar path
    x_s = real(0.3)
    psi_s = base(x_s, 3)
    assert len(psi_s) == 4
    assert psi_s[0].value == pytest.approx(1.0)
    assert psi_s[1].value == pytest.approx(0.3)

    # TRNode path
    x_n = TRNode.constant(real(0.3))
    psi_n = base(x_n, 3)
    assert len(psi_n) == 4
    assert psi_n[0].tag == TRTag.REAL
    assert psi_n[0].value.value == pytest.approx(1.0)
    assert psi_n[1].value.value == pytest.approx(0.3)
