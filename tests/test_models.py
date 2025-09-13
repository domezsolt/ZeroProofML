from zeroproof.layers import MonomialBasis
from zeroproof.layers.multi_input_rational import TRMultiInputRational
from zeroproof.core import TRTag, real
from zeroproof.autodiff import TRNode


def test_tr_multi_forward_shapes():
    model = TRMultiInputRational(input_dim=4, n_outputs=2, d_p=2, d_q=1,
                                 basis=MonomialBasis(), hidden_dims=[4], shared_Q=True, enable_pole_head=True)
    vec = [0.2, -0.1, 0.05, -0.02]
    outs = model.forward(vec)
    assert isinstance(outs, list) and len(outs) == 2
    for y, tag in outs:
        assert hasattr(y, 'tag')
        assert tag in (TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI)

