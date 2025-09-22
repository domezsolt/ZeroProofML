# ZeroProofML – Library Update To-Do (Markdown)

## ✅ Mandatory (keep theory ⇄ code aligned)

- [ ] **Introduce a central TR policy config**
  - Define `TRPolicy` with:
    - `tau_Q_on`, `tau_Q_off`, `tau_P_on`, `tau_P_off` (ULP-scaled thresholds)
    - rounding mode, **keep signed zero**, deterministic reduction flag
    - sensitivity triggers: `g_on`, `g_off`
  - Add `resolve_thresholds(ulp, local_scales)` to derive guard bands from ULP + local sensitivities (e.g., ‖∇Q‖, ‖∇P‖).

- [ ] **Tagging with guard bands + hysteresis**
  - In all `P/Q` layers: classify **REAL / INF / NULL** using `tau_*` with **on/off hysteresis**.
  - Preserve **signed zeros** for `Q` to encode approach direction (±0).
  - Use **deterministic reduction** (fixed tree) for `Q`-related computations.

- [ ] **Mask-REAL backprop gating**
  - Wrap VJP/grad so each primitive multiplies by gate `χ_k ∈ {0,1}`.
  - When `χ_k = 0`, use **bounded surrogate derivative** `S_k` (smooth saturator), with `‖S_k‖ ≤ G_max`.

- [ ] **Hybrid controller (MR ↔ SAT) with hysteresis**
  - Implement a small state machine:
    - **Enter SAT** if `d(x) ≤ δ_on` **or** `max_k g_k ≥ g_on` **or** batch tag-flip event.
    - **Return to MR** if `d(x) ≥ δ_off` **and** `max_k g_k ≤ g_off` **and** no flips.
  - Batch aggregation via robust quantiles (e.g., 90th percentile of `d`, `g`).

- [ ] **Batch-safe learning-rate helper**
  - Provide: `η_safe = min( α / L̂_batch , c / (L_ell * ∏_k max{B_k, G_max}) )`.
  - Adjust for momentum/Adam effective step sizes:
    - Heavy-ball: ensure `η ≤ 2(1−β1)/L̂_batch`.
    - Adam: ensure `η ≤ (1−β1) / (√(1−β2) * L̂_batch)` (sufficient condition).

- [ ] **Core metrics & logging**
  - Log per batch: `min|Q|`, `flip_rate`, `%SAT_time`, quantiles of `d(x)` and `g_k`.
  - Expose hooks for the coverage controller and debugging.

---

## 👍 Strongly recommended (robustness & performance)

- [ ] **Deterministic reductions & memory layout**
  - Use **Kahan/Neumaier** or **pairwise/tree** summation for `P`, `Q`, and gradient reductions.
  - Adopt **struct-of-arrays** layout for `(value, tag)`; **bit-pack** tags (1 bit/node).

- [ ] **IEEE ↔ TR bridge hygiene**
  - Map **IEEE NaN → TR `NULL`**; export policy may map **`NULL → NaN`**.
  - Retain zero sign; document **partial homomorphism** (non-NaN regime).

- [ ] **Coprime/identifiability diagnostics (optional regularizer)**
  - Add **Sylvester-matrix `s_min`** surrogate or **resultant-based** penalty; make it toggleable and logged.

---

## ✨ Optional (advanced / nice to have)

- [ ] **Per-node hybrid modes** (instead of global), same hysteresis policy.
- [ ] **Second-order curvature safeguards** (GN/Fisher bounds) on SAT branches.
- [ ] **TR-policy variants or rational surrogates** for softmax/layernorm as needed.

---

## 📁 Suggested file structure / touch points

- `policy.py` — `TRPolicy`, threshold resolution, rounding & reduction settings.  
- `hybrid.py` — `TRHybridController` (state machine, batch quantiles, flip detector).  
- `layers.py` — `TRL_Rational`: deterministic reductions, tagging + hysteresis, (value, tag) **SoA**.  
- `autodiff_hooks.py` — **Mask-REAL** VJP wrappers, surrogate `S_k`.  
- `optim_utils.py` — batch-safe LR utilities; momentum/Adam effective step adjustments.  
- `metrics.py` — `flip_rate`, `%SAT_time`, `min|Q|`, diagnostics; logging integrations.

---

## ✅ Test & acceptance criteria

- [ ] **Determinism outside guard bands**  
  Same seeds/policy/device ⇒ identical tags/outputs for inputs with `|Q| ≥ τ_Q_off`.

- [ ] **Finite/low-density switching**  
  With default hysteresis + batch-safe LR, mode switches per K steps are finite and decline; no chatter.

- [ ] **Tag robustness**  
  Unit tests: outside guard bands, tag classification invariant to FP perturbations (within 1–2 ULP).

- [ ] **Hybrid safety**  
  With surrogate bounds `G_max`, parameter updates satisfy the bounded-update inequality across MR/SAT transitions.

- [ ] **Optimizer safety**  
  Heavy-ball and Adam pass descent-lemma smoke tests under provided bounds.

- [ ] **Overhead envelope**  
  Report `%SAT_time`, mask bandwidth, and total slowdown vs. baseline; show activation is rare and overhead localized near poles.

- [ ] **Identifiability diagnostics (if enabled)**  
  Coprime surrogate remains healthy during normal training; warns near shared-factor degeneracy.

---

## 🛠️ Rollout plan

1. Implement **policy + tagging + Mask-REAL** (core path green).  
2. Add **hybrid controller + logging + batch-safe LR**.  
3. Wire **deterministic reductions + SoA/bit-mask**.  
4. Integrate **optimizer bounds** (heavy-ball/Adam helpers).  
5. Add **diagnostics & unit tests** (determinism, finite switching, robustness).  
6. *(Optional)* Add **coprime regularizer** and **TR-policy softmax/layernorm**.
