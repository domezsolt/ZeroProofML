# Getting Started

A fast path to using ZeroProof with robust singularity handling. Then branch into the topics you need.

## Install

- From source (dev):
  - `pip install -e .[dev]`
- From PyPI (if available in your environment):
  - `pip install zeroproof`

Optional extras:
- PyTorch: `pip install zeroproof[torch]`
- JAX: `pip install zeroproof[jax]`

## First Steps (5 minutes)

- Basic TR arithmetic
```python
import zeroproof as zp
x = zp.real(1.0) / zp.real(0.0)   # → +∞, no exception
z = zp.real(0.0) / zp.real(0.0)   # → Φ (nullity)
```

- A TR‑Rational layer
```python
from zeroproof.layers import TRRational, ChebyshevBasis
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real

layer = TRRational(d_p=2, d_q=1, basis=ChebyshevBasis())
y, tag = layer.forward(TRNode.constant(real(0.25)))
```

## Train with Poles (minimal sketch)

```python
from zeroproof.autodiff.hybrid_gradient import create_default_schedule
from zeroproof.training.adaptive_loss import AdaptiveLambda, AdaptiveLossConfig
from zeroproof.core import TRTag, real
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.autodiff.tr_ops_grad import tr_div
from zeroproof.layers import HybridTRRational, ChebyshevBasis

# Model
model = HybridTRRational(d_p=3, d_q=2, basis=ChebyshevBasis(),
                         hybrid_schedule=create_default_schedule(warmup_epochs=5))

# Policy
policy = AdaptiveLambda(AdaptiveLossConfig(target_coverage=0.95, learning_rate=0.01))

for epoch in range(10):
    with model.hybrid_schedule.apply(epoch):
        # simple 1D example: synthetic targets
        xs = [TRNode.constant(real(v)) for v in [-0.9,-0.3,0.1,0.3,0.9]]
        outputs = []
        tags = []
        for x in xs:
            y, tag = model.forward(x)
            outputs.append(y)
            tags.append(tag)
        # Update λ based on tags
        policy.update(tags)
```

## Run the Examples

Use the ready-made scripts to see the pieces working end-to-end.

Quick demos (no extra deps):
- Basic arithmetic and ops: `python examples/basic_usage.py`
- Layers (TR‑Rational, TR‑Norm): `python examples/layers_demo.py`

Hybrid gradients and coverage (matplotlib required):
- Hybrid gradient schedule: `python examples/hybrid_gradient_demo.py`
- Coverage control (adaptive λ): `python examples/coverage_control_demo.py`

Full showcase (heavier, integrates many features):
- Complete feature demo: `python examples/complete_demo.py`
- Full pipeline: `python examples/full_pipeline_demo.py`

Tip: If a script plots figures, install matplotlib: `pip install matplotlib`.

## Evaluate Poles

```python
from zeroproof.utils.evaluation_api import IntegratedEvaluator, EvaluationConfig

x_vals = [i/50 for i in range(-50,51)]
evalr = IntegratedEvaluator(EvaluationConfig(enable_visualization=False), true_poles=[0.0])
metrics = evalr.evaluate_model(model, x_vals)
print({k:v for k,v in metrics.__dict__.items() if isinstance(v,(int,float))})
```

## Where to Go Next

- Concepts: Topic 1–3 for principles, foundations, and autodiff modes.
- Layers: Topic 4 for TR‑Rational/TR‑Norm and enhanced variants.
- Training: Topic 5 for adaptive loss, coverage, and tag loss.
- Sampling: Topic 6 for near‑pole importance/active sampling.
- Evaluation: Topic 7 for pole metrics and integrated evaluator.

Shortcuts
- Quick reference: `docs/quick_reference.md`
- Layer guide: `docs/layers.md`
- Saturating/Hybrid details: `docs/saturating_grad_guide.md`, `docs/topics/03_autodiff_modes.md`
