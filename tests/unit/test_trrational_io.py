import pytest

from zeroproof.core import real, TRTag
from zeroproof.autodiff import TRNode
from zeroproof.layers import TRRational, MonomialBasis


def make_simple_layer():
	# y = (1 + x) / (1 + 0.5 x)
	layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
	layer.theta[0]._value = real(1.0)
	layer.theta[1]._value = real(1.0)
	layer.phi[0]._value = real(0.5)
	return layer


def test_scalar_forward_and_call_contract():
	layer = make_simple_layer()
	# scalar TRScalar
	x = real(1.0)
	y, tag = layer.forward(x)
	assert tag == TRTag.REAL
	assert y.value.value == pytest.approx(4/3)
	# __call__ returns only y
	y2 = layer(x)
	assert y2.value.value == pytest.approx(4/3)
	# forward_with_tag mirrors forward
	y3, tag3 = layer.forward_with_tag(x)
	assert tag3 == tag
	assert y3.value.value == pytest.approx(4/3)


def test_forward_rejects_sequences_without_projection():
	layer = make_simple_layer()
	with pytest.raises(TypeError):
		layer.forward([real(0.0), real(1.0)])


def test_forward_batch_on_list_of_scalars():
	layer = make_simple_layer()
	xs = [real(-1.0), real(0.0), real(1.0)]
	ys = layer.forward_batch(xs)
	assert len(ys) == 3
	assert ys[1].value.value == pytest.approx(1.0)  # at x=0: (1+0)/(1+0)=1


def test_forward_batch_vector_inputs_with_projection():
	# Use projection_index=0 to select first feature
	layer = TRRational(d_p=1, d_q=1, basis=MonomialBasis(), projection_index=0)
	layer.theta[0]._value = real(0.0)
	layer.theta[1]._value = real(1.0)
	layer.phi[0]._value = real(1.0)
	# y = x / (1 + x)
	xs = [
		[real(0.0), real(10.0)],
		[real(1.0), real(20.0)],
	]
	ys = layer.forward_batch(xs)
	assert len(ys) == 2
	assert ys[0].value.value == pytest.approx(0.0)
	assert ys[1].value.value == pytest.approx(0.5)


