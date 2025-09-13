from zeroproof.layers import MonomialBasis
from zeroproof.layers.multi_input_rational import TRMultiInputRational
from zeroproof.core import TRTag, real
from zeroproof.autodiff import TRNode


def test_tr_multi_forward_vector_returns_tags():
    model = TRMultiInputRational(input_dim=4, n_outputs=2, d_p=2, d_q=1, basis=MonomialBasis(), hidden_dims=[4], shared_Q=True, enable_pole_head=False)
    # Float input
    outputs = model.forward([0.1, -0.2, 0.05, -0.05])
    assert isinstance(outputs, list) and len(outputs) == 2
    for y, tag in outputs:
        assert hasattr(y, 'tag')
        assert tag in (TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI)

    # TRNode input
    tr_inp = [TRNode.constant(real(0.1)), TRNode.constant(real(-0.2)), TRNode.constant(real(0.05)), TRNode.constant(real(-0.05))]
    outputs2 = model.forward(tr_inp)
    assert isinstance(outputs2, list) and len(outputs2) == 2
    for y, tag in outputs2:
        assert hasattr(y, 'tag')
        assert tag in (TRTag.REAL, TRTag.PINF, TRTag.NINF, TRTag.PHI)


def test_tr_multi_forward_fully_integrated_shape():
    model = TRMultiInputRational(input_dim=4, n_outputs=2, d_p=2, d_q=1, basis=MonomialBasis(), hidden_dims=[4], shared_Q=True, enable_pole_head=True)
    vec = [0.1, -0.2, 0.05, -0.05]
    result = model.forward_fully_integrated(vec)
    assert isinstance(result, dict)
    assert isinstance(result.get('outputs'), list) and len(result['outputs']) == 2
    assert isinstance(result.get('tags'), list) and len(result['tags']) == 2
    # Optional fields
    if 'Q_abs_list' in result:
        assert len(result['Q_abs_list']) == 2
    # pole_score may be present when pole head is enabled
    assert 'pole_score' in result
