import math

from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.multi_input_rational import TRMultiInputRational


def _compute_loss_with_constants(model, x_vals, target):
    """Compute scalar MSE loss using constant TRNodes (no grad)."""
    x_nodes = [TRNode.constant(real(float(v))) for v in x_vals]
    outs = model.forward(x_nodes)
    loss = TRNode.constant(real(0.0))
    valid = 0
    for (y, tag), t in zip(outs, target):
        if tag == TRTag.REAL:
            diff = y - TRNode.constant(real(float(t)))
            loss = loss + diff * diff
            valid += 1
    if valid == 0:
        return None
    # Average over valid outputs for stability
    loss = loss / TRNode.constant(real(float(valid)))
    return loss


def test_tr_multi_gradient_check_random_input():
    # Model: 4D->2D with small degrees for stability
    model = TRMultiInputRational(
        input_dim=4,
        n_outputs=2,
        d_p=2,
        d_q=1,
        basis=MonomialBasis(),
        hidden_dims=[4],
        shared_Q=True,
        enable_pole_head=False,
    )

    # Random-but-small input (deterministic values to avoid RNG deps)
    x0 = [0.12, -0.07, 0.03, -0.02]
    target = [0.01, -0.02]

    # Analytical gradients via gradient tape w.r.t. inputs
    with gradient_tape() as tape:
        x_nodes = [TRNode.parameter(real(v)) for v in x0]
        for xn in x_nodes:
            tape.watch(xn)
        outs = model.forward(x_nodes)
        loss = TRNode.constant(real(0.0))
        valid = 0
        for (y, tag), t in zip(outs, target):
            if tag == TRTag.REAL:
                diff = y - TRNode.constant(real(float(t)))
                loss = loss + diff * diff
                valid += 1
        assert valid > 0, "Loss had no REAL contributions; cannot perform gradcheck"
        loss = loss / TRNode.constant(real(float(valid)))

    assert loss.tag == TRTag.REAL, "Central loss must be REAL for gradcheck"
    grads = tape.gradient(loss, x_nodes)

    # Numerical gradients via central differences
    eps = 1e-5
    numeric = []
    checked_dims = 0
    for i in range(len(x0)):
        x_plus = list(x0)
        x_minus = list(x0)
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = _compute_loss_with_constants(model, x_plus, target)
        f_minus = _compute_loss_with_constants(model, x_minus, target)
        # Skip dims where perturbation yields non-REAL (rare with small eps)
        if f_plus is None or f_minus is None:
            numeric.append(None)
            continue
        if f_plus.tag != TRTag.REAL or f_minus.tag != TRTag.REAL:
            numeric.append(None)
            continue
        g = (f_plus.value.value - f_minus.value.value) / (2 * eps)
        numeric.append(g)
        checked_dims += 1

    # Require that at least half the dimensions yielded stable numeric gradients
    assert checked_dims >= 2, "Insufficient REAL perturbations for reliable gradcheck"

    # Compare analytical vs numerical for available dims
    for gi, gn in zip(grads, numeric):
        if gn is None:
            continue
        assert gi.tag == TRTag.REAL
        # Allow modest tolerance due to rational nonlinearity
        assert math.isfinite(gn)
        assert abs(gi.value.value - gn) <= (1e-2 * max(1.0, abs(gn)) + 1e-3)
