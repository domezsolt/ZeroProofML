# Topic 3: Autodiff Modes (Mask‑REAL, Saturating, Hybrid)

This topic explains ZeroProof’s gradient semantics and how to select between Mask‑REAL, Saturating, and Hybrid modes, with code pointers and usage examples.

## Goals
- Preserve classical gradients on REAL paths.
- Remain total and stable when forwards produce non‑REAL tags (±∞, Φ).
- Provide bounded alternatives near poles without ε hacks.

## Modes at a Glance
- Mask‑REAL (default): If a forward node tag is non‑REAL, send zero gradient to its inputs/params.
- Saturating: Replace singular growth terms with bounded, TR‑safe forms near poles.
- Hybrid: Scheduler picks Saturating near poles and Mask‑REAL elsewhere; tracks q‑statistics for decisions.

APIs
- Mode config/context: `zeroproof/autodiff/grad_mode.py:1`
- Hybrid schedule/context: `zeroproof/autodiff/hybrid_gradient.py:1`
- Backward pass rules: `zeroproof/autodiff/backward.py:1`
- Tape and lifted ops: `zeroproof/autodiff/gradient_tape.py:1`, `zeroproof/autodiff/tr_ops_grad.py:1`

## Mask‑REAL (Default)
Behavior
- Rule: If node.forward.tag ∈ {PINF, NINF, PHI}, gradients to inputs/params are exactly zero.
- Chain effect: Any path that encounters a non‑REAL node contributes zero to the Jacobian.

Code
- Check: `backward.py: MASK_REAL branch in backward_pass` and `compute_input_gradients`.
- Using default: no action required; `GradientModeConfig` defaults to MASK_REAL.

Usage
```python
from zeroproof.autodiff import grad_mode
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real
from zeroproof.autodiff.tr_ops_grad import tr_div

x = TRNode.parameter(real(0.0))
y = tr_div(real(1.0), x)   # forward tag = PINF
# Backprop uses Mask‑REAL → grad wrt x is 0
y.backward()
print(x.gradient)  # REAL 0.0
```

## Saturating Mode
Behavior
- Bounded gradients near poles; replaces singular terms like 1/Q² with a capped TR‑safe form.
- Keeps continuity of gradient magnitudes across pole neighborhoods.

Code
- Enable: `GradientModeConfig.set_mode(GradientMode.SATURATING)` in `grad_mode.py:1`.
- Div/log/sqrt/pow grad handlers call `saturating_ops` in `backward.py:200`.

Usage
```python
from zeroproof.autodiff.grad_mode import GradientMode, GradientModeConfig
from zeroproof.autodiff.tr_ops_grad import tr_div
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real

GradientModeConfig.set_mode(GradientMode.SATURATING)
GradientModeConfig.set_saturation_bound(1.0)

x = TRNode.parameter(real(1e-12))
y = tr_div(real(1.0), x)
y.backward()
print(x.gradient)  # finite, bounded by saturation rule
```

## Hybrid Mode
Intuition
- Early/far from poles: use Mask‑REAL’s stability guardrails.
- Near poles: switch to Saturating to allow informative gradients.
- The schedule controls a threshold δ(t) and tracks q‑statistics.

Code
- Schedule and context: `hybrid_gradient.py:1` (HybridGradientSchedule, HybridGradientContext).
- Decision point: `backward.py: DIV case → HybridGradientContext.should_use_saturating`.
- Mode flag: `GradientMode.HYBRID` in `grad_mode.py:1`.

Common Setup
```python
from zeroproof.autodiff.hybrid_gradient import create_default_schedule
from zeroproof.autodiff.grad_mode import GradientMode, GradientModeConfig
from zeroproof.autodiff.tr_ops_grad import tr_div
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real

schedule = create_default_schedule(aggressive=False, warmup_epochs=5, force_exploration=True)
for epoch in range(50):
    with schedule.apply(epoch):
        x = TRNode.parameter(real(1e-6))
        y = tr_div(real(1.0), x)
        y.backward()
        # Inspect context stats per batch/epoch
        from zeroproof.autodiff.hybrid_gradient import HybridGradientContext
        stats = HybridGradientContext.get_statistics()
        # stats includes q_min_batch, near_pole_ratio, saturating_ratio, etc.
```

Schedule Features
- Warmup and transition epochs with linear/exponential/cosine decay.
- Adaptive δ: Increase tolerance when batch q_min is very small.
- Forced exploration: Schedule regions around detected poles for saturating activations.
- Stats: q_min (batch/epoch), near_pole ratio, saturating vs mask‑real counts.

Notes
- No ε in arithmetic: δ is a scheduler parameter, not part of core TR ops.
- Deterministic decisions per backend given identical inputs and seeding.

## Tape and Lifted Ops
- Use `gradient_tape()` for functional differentiation and `tr_ops_grad` for graph building.
- `TRNode.parameter(...)` marks trainable nodes; `TRNode.constant(...)` for constants.

Example: Value and Gradient
```python
from zeroproof.autodiff.grad_funcs import tr_value_and_grad
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real
from zeroproof.autodiff.tr_ops_grad import tr_mul, tr_add

# f(x) = x*x + 2x + 1
def f(x):
    return tr_add(tr_add(tr_mul(x, x), tr_mul(real(2.0), x)), real(1.0))

val_and_grad = tr_value_and_grad(f)
x = TRNode.parameter(real(3.0))
val, grad = val_and_grad(x)
# val is REAL 16, grad is REAL 8
```

## Gradient Checking Utilities
- For finite‑difference checks on TR functions, use `zeroproof/autodiff/grad_funcs.py`:
  - `check_gradient(func, x, eps=1e-5)` computes analytical vs numerical grad and relative error on REAL paths.
  - `tr_grad` and `tr_value_and_grad` lift scalar functions into gradient evaluators for quick tests.
- See tests under `tests/unit/test_tr_autodiff.py` and the robotics multi‑input gradcheck at `tests/unit/test_robotics_gradcheck.py` for usage patterns.

## Practical Guidance
- Default to Mask‑REAL for safety; enable Hybrid for tasks requiring pole learning.
- Start with small warmup_epochs and moderate δ range (e.g., 1e‑2 → 1e‑6).
- Use stats to tune saturating_bound and schedule aggressiveness.
- For deterministic runs, fix seeds and backend deterministic flags (see spec repro checklist in `complete_v2.md:1`).

## Cross‑References
- Conceptual background: `concept_250908.md:200` (Hybrid Schedule section)
- Mask‑REAL guide: `docs/autodiff_mask_real.md:1`
- Saturating guide: `docs/saturating_grad_guide.md:1`
