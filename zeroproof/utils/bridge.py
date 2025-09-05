"""Convenience coercion utilities for bridging Python/NumPy types to TR types.

These helpers centralize common conversions to avoid scattered type checks
and errors like "ndarray has no attribute tag" in training paths.
"""

from typing import Any

try:
	import numpy as np  # type: ignore
	NUMPY_AVAILABLE = True
except Exception:
	NUMPY_AVAILABLE = False
    
from ..core import TRScalar, real
from ..autodiff import TRNode


def to_real_scalar(x: Any) -> TRScalar:
	"""Coerce Python/NumPy numeric into TRScalar.

	- If x is TRScalar, return it
	- If x is TRNode, use its underlying scalar value (if available) else raise
	- If x is Python number or NumPy scalar, convert via float -> real
	- If x is NumPy ndarray with size==1, convert the single item
	- Otherwise raise TypeError
	"""
	if isinstance(x, TRScalar):
		return x
	if isinstance(x, TRNode):
		# Use its current value if tagged REAL; otherwise still wrap numeric best-effort
		if x.tag is not None and x.value is not None:
			try:
				return x.value
			except Exception:
				pass
		raise TypeError("Cannot coerce TRNode directly to TRScalar; wrap with TRNode.constant if needed.")
	# Python numeric
	try:
		return real(float(x))
	except Exception:
		pass
	# NumPy paths
	if NUMPY_AVAILABLE:
		try:
			if np.isscalar(x):  # numpy scalar
				return real(float(x))
			arr = np.asarray(x)
			if arr.shape == ():
				return real(float(arr))
			if arr.size == 1:
				return real(float(arr.reshape(())))
		except Exception:
			pass
	raise TypeError(f"Unsupported type for to_real_scalar: {type(x)}")


def to_trnode_constant(x: Any) -> TRNode:
	"""Coerce input into TRNode.constant(TRScalar(...)).

	- If x is TRNode, return as is
	- If x is TRScalar, wrap with TRNode.constant
	- Else coerce to TRScalar via to_real_scalar and wrap
	"""
	if isinstance(x, TRNode):
		return x
	if isinstance(x, TRScalar):
		return TRNode.constant(x)
	return TRNode.constant(to_real_scalar(x))


