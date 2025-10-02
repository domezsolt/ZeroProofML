from zeroproof.autodiff import TRNode, tr_grad, tr_mul
from zeroproof.core import real


# Test 1: Simple function
def f(x):
    return tr_mul(x, x)


# Create gradient function
grad_f = tr_grad(f)

# Test with a scalar
x_val = real(3.0)
grad = grad_f(x_val)

print(f"Input: {x_val}")
print(f"Gradient: {grad}")
print(f"Gradient value: {grad.value if hasattr(grad, 'value') else 'No value'}")
print(f"Expected: 6.0")
