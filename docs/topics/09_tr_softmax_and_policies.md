# Topic 9: TR‑Softmax and Policies

This topic describes ZeroProof’s TR‑safe softmax surrogate, design rationale, and policy options for extreme cases.

## Why a TR‑Softmax Surrogate?

Classical softmax uses `exp`, which can overflow/underflow and propagate NaNs/Infs through autodiff. In ZeroProof, we build a softmax‑like function from rational TR operations to:

- Preserve totality (never throw),
- Avoid NaN/Inf propagation in REAL regions,
- Remain compatible with Mask‑REAL/Hybrid autodiff.

## Implementation Sketch

1. Shift by `max(logits)` to reduce dynamic range (log‑sum‑exp trick).
2. Apply a monotone rational decay `r(z) = 1 / (1 + c t + d t²)`, where `t = max(0, −z)` (implemented via `t = (|z| − z)/2`).
3. Normalize: `p_i = r_i / Σ_j r_j` using TR reductions (deterministic reductions optional via policy).

Code: `zeroproof/layers/tr_softmax.py:1`.

## Policy: One‑Hot on +∞ (Optional)

By default, if a logit is `+∞`, the shift‑by‑max can yield non‑REAL tags in the surrogate. This is TR‑safe and deterministic, but sometimes you want a strict one‑hot in the presence of `+∞`.

Enable the policy toggle to force a one‑hot distribution at the first `+∞` index (deterministic tie‑break):

```python
from zeroproof.policy import TRPolicy, TRPolicyConfig

TRPolicyConfig.set_policy(TRPolicy(softmax_one_hot_infinity=True))
```

This preserves totality and avoids ambiguity when infinities appear in logits.

## Testing and Examples

- Unit tests: `tests/unit/test_tr_softmax.py` cover basic properties, shift invariance, extreme logits, and `+∞`/Φ cases (with/without policy).
- Tiny classifier example: `examples/classification_demo.py` demonstrates training a 2‑class linear model using `tr_softmax` and TR autodiff.

## Notes

- The surrogate uses only TR operations and remains differentiable in REAL regions. Mask‑REAL and Hybrid gradients apply as configured globally.
- For large products/sums in normalization, enable deterministic compensated reductions (policy flag) to reduce round‑off sensitivity.

