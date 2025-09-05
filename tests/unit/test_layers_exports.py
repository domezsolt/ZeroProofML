def test_layers_exports_create_basis():
	from zeroproof.layers import create_basis
	b = create_basis("chebyshev")
	assert b.name == "chebyshev"


