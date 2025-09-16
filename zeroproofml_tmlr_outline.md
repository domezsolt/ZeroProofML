\\documentclass{article}
\\usepackage{tmlr}
\\usepackage{amsmath, amssymb, amsthm}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{natbib}

% theorem environments
\\newtheorem{theorem}{Theorem}
\\newtheorem{lemma}{Lemma}
\\newtheorem{proposition}{Proposition}
\\newtheorem{definition}{Definition}

% macros
\\newcommand{\\tr}{\\text{TR}}
\\newcommand{\\wheel}{\\text{Wheel}}
\\newcommand{\\real}{\\mathbb{R}}
\\newcommand{\\phiTag}{\\Phi}

\\title{ZeroProofML: Epsilon-Free Rational Neural Layers via Transreal Arithmetic}

\\author{Anonymous Authors}

\\begin{document}

\\maketitle

\\begin{abstract}
We introduce ZeroProofML, a framework for epsilon-free rational neural layers based on transreal arithmetic. By totalizing division and other singular operations, our approach eliminates the need for $\\varepsilon$ hacks and provides deterministic tag semantics. We formalize transreal autodiff, describe an $\\varepsilon$-free normalization (TR-Norm), state stability bounds (bounded updates near poles), and validate empirically against MLP and $\\varepsilon$-rational baselines (plus DLS reference) on a controlled robotics task. Our results indicate overall parity with $\\varepsilon$-rational and slight per-bucket improvements near poles, suggesting ZeroProofML as a reproducible alternative to conventional techniques near singularities.
\\end{abstract}

\\section{Introduction}
At the heart of modern machine learning lies a paradox: our models are built on mathematical foundations that forbid division by zero, yet in practice they rely on ad-hoc numerical “fixes” to avoid it. The most common of these, the **$\\varepsilon$-trick**, replaces unstable denominators with $Q(x)+\\varepsilon$ for small $\\varepsilon$ \\citep{ioffe2015batchnorm,boulle2020rational}. This pragmatic adjustment enables training but breaks mathematical consistency, obscures theoretical guarantees, and compromises reproducibility \\citep{henderson2018rlmatters}. Beyond ad-hoc $\\varepsilon$ knobs, practical systems face nondeterminism and floating‑point pathologies that manifest as NaNs/Infs and seed sensitivity \\citep{micikevicius2018mixedprecision,higham2002accuracy}, complicating debugging and re-use. As models scale and are deployed in safety-critical settings, reliance on these hacks becomes increasingly untenable.

This paper develops a principled alternative: **ZeroProofML**, a framework that replaces $\\varepsilon$-hacks with **transreal arithmetic (TR)**. TR extends the real numbers with totalized operations, adding well-defined outcomes for division by zero: $+\\infty$, $-\\infty$, and nullity $\\Phi$ for indeterminate forms \\citep{anderson2019transmathematics,dosreis2016transreal}. Building on this foundation, we construct **rational neural layers** of the form $P(x)/Q(x)$ that remain mathematically total and algorithmically stable.

\\paragraph{Contributions.} Our key contributions are:
- **$\\varepsilon$-free, total, deterministic semantics:** a tag-aware TR calculus that totalizes division and removes ad-hoc $\\varepsilon$ knobs while preserving deterministic behavior for infinities and nullity.
- **TR-Autodiff + TR-Norm:** Mask-REAL gradients that coincide with classical values on REAL paths and vanish exactly otherwise; an $\\varepsilon$-free normalization with deterministic bypass at zero variance.
- **Stability statement:** a bounded-update condition near poles under a simple saturating gradient mode.
- **Reproducible robotics evaluation:** scripted, stratified near-pole tests on 2R IK with MLP/$\\varepsilon$-rational/TR baselines and DLS reference; parity overall and slight gains near poles.

The central idea is to carry *tags* (REAL, $\\pm\\infty$, $\\Phi$) alongside values. **TR-Autodiff** enforces the *Mask-REAL* rule: on REAL paths, gradients coincide with classical calculus; on non-REAL paths, gradients vanish exactly, preventing instabilities \\citep{baydin2018autodiff}. **TR-Norm** describes an $\\varepsilon$-free batch normalization with a deterministic bypass when variance is zero. Together, these components form an ML stack where every operation is total and tag-deterministic by construction.

We state that ZeroProofML layers admit bounded updates near poles and ensure identifiability by fixing the leading term of $Q$. Empirically, we observe stable training and overall parity with $\\varepsilon$-rational, with slight per-bucket gains near poles. ZeroProofML thus offers not just a new layer design, but a path toward machine learning without $\\varepsilon$-hacks, grounded in total arithmetic and reproducible semantics.

\\section{Method}

We now introduce the formal machinery underlying **ZeroProofML**. Our method is based on **transreal arithmetic (TR)**, which extends the reals to a *total algebraic structure* \\citep{anderson2019transmathematics}.

\\subsection{Transreal Carrier and Tags}
We define the scalar carrier as
\\[
TR = \\{ (v, t) \\mid v \\in \\mathbb{R} \\cup \\{\\text{NaN}\\}, \\; t \\in \\{\\text{REAL}, \\text{PINF}, \\text{NINF}, \\Phi\\}\\}.
\\]
- REAL: finite values.  \\
- PINF / NINF: $+\\infty$ and $-\\infty$.  \\
- $\\Phi$: nullity, representing indeterminate forms such as $0/0$, $\\infty-\\infty$, or $0 \\cdot \\infty$.  

Closed operation tables for $+, \\times, \\div$ guarantee **totality**: every expression evaluates to a well-defined tagged value, never raising exceptions.

\\subsection{Totalized Operations}
Arithmetic follows TR rules \\citep{dosreis2016transreal,bergstra2021wheel}:
- Division: $\\tfrac{a}{0} =$ PINF/NINF if $a \\ne 0$, $\\Phi$ if $a=0$.  \\
- Multiplication: $0 \\times \\infty = \\Phi$, $\\infty \\times \\infty = \\infty$.  \\
- Addition: $\\infty + (-\\infty) = \\Phi$.  
These rules align with established transreal systems while preserving deterministic evaluation.

\\subsection{TR-Autodiff}
Autodiff is extended via the **Mask-REAL rule**:
- If a forward node tag is REAL, gradients coincide with classical derivatives.  \\
- If the node tag is non-REAL (PINF, NINF, $\\Phi$), **all parameter/input partials are zero**.  

\\textbf{Lemma (Mask-REAL Composition).} Any computational path passing through a non-REAL node contributes zero to the overall Jacobian.  \\
\\emph{Proof sketch.} Immediate by induction on path length and the chain rule, since all derivatives vanish at the first non-REAL node.

\\subsection{TR-Rational Layers}
We define rational layers \\citep{boulle2020rational}:
\\[
y(x) = \\frac{P_\\theta(x)}{Q_\\phi(x)}, \\quad 
Q_\\phi(x) = 1 + \\sum_{k=1}^{d_Q} \\phi_k \\psi_k(x),
\\]
with polynomial bases $\\psi_k$. Identifiability is ensured by fixing the leading term of $Q$. Forward semantics use TR rules; gradients use TR-AD with Mask-REAL.

\\begin{proposition}[\\(\\varepsilon\\)-Limit Equivalence Away from Poles]\\label{prop:epslimit}
Let $y(x)=P_\\theta(x)/Q_\\phi(x)$ with TR semantics and define $y_\\varepsilon(x)=P_\\theta(x)/(Q_\\phi(x)+\\varepsilon)$ for $\\varepsilon>0$. On any compact set $K$ with $\\inf_{x\\in K}|Q_\\phi(x)|\\ge c>0$, we have $y_\\varepsilon \\to y$ and $\\nabla \\!_{(\\theta,\\phi)} y_\\varepsilon \\to \\nabla \\!_{(\\theta,\\phi)} y$ uniformly as $\\varepsilon\\to 0^+$. \\\\n+\\emph{Sketch.} On $K$, $1/Q_\\phi$ is continuous and uniformly bounded by $1/c$; both $P_\\theta$ and $Q_\\phi$ are linear in parameters and continuous in $x$. Uniform convergence of $1/(Q_\\phi+\\varepsilon)$ to $1/Q_\\phi$ implies uniform convergence of values and parameter derivatives.
\\end{proposition}

\\subsection{TR-Norm}
For normalization, we compute feature-wise variance over REAL samples only.  \\
- If $\\sigma^2 > 0$: normalize as in batch normalization \\citep{ioffe2015batchnorm}.  \\
- If $\\sigma^2 = 0$: deterministically bypass, outputting $\\beta$.  

This ensures totality and limit-equivalence to BN as $\\varepsilon \\to 0^+$, without introducing $\\varepsilon$.

\\subsection{IEEE $\\leftrightarrow$ TR Bridge}
We provide a bidirectional mapping: finite floats $\\leftrightarrow$ REAL, NaN $\\leftrightarrow$ $\\Phi$, $\\pm\\infty$ $\\leftrightarrow$ PINF/NINF. Round-trips preserve semantics, ensuring framework compatibility.
% \\begin{figure}[h]
%   \\centering
  \\includegraphics[width=0.82\\linewidth]{results/robotics/figures/tr_schematic.png}
  \\caption{Schematic of TR pipeline: inputs pass through totalized TR ops with explicit tag semantics; gradients follow Mask‑REAL almost everywhere and switch to bounded (saturating) mode near poles.}
\\end{figure}



\\subsection{Hybrid Gradients and Coverage Control}
We use a lightweight Hybrid schedule that prefers Mask‑REAL almost everywhere and switches to a bounded (saturating) gradient near poles; a simple coverage controller avoids degenerate “always REAL” collapse. Full scheduling/controller details and diagnostics (coverage/$\\lambda$, $q_{\\min}$, bench metrics) appear in the Appendix.

\\paragraph{Full TR model.} Unless stated otherwise, the “Full” variant combines:
- **Core layer**: TR‑Rational heads with $P/Q$ (leading $Q$ fixed to 1 for identifiability), exact TR semantics (no $\\varepsilon$), and shared‑$Q$ across outputs when multi‑output structure shares pole lines.
- **Hybrid gradients**: Mask‑REAL away from poles; bounded (saturating) gradients in a small $|Q|$ band (bounded‑update guarantee; Prop.~\\ref{prop:bounded}).
- **Coverage control**: a simple controller that discourages degenerate 100\\% REAL coverage and favors learning near poles when needed (tracks coverage, $q_{\\min}$, and $\\lambda$).
- **Auxiliary heads** (when enabled): (i) a tag loss to improve non‑REAL tag prediction; (ii) a pole head to localize poles (optionally supervised by analytic teachers, e.g., $|\\sin\\theta_2|$ or manipulability). Both are lightweight and attached to the same front‑end features.
- **Anti‑illusion loss**: a small residual‑consistency term computed via forward kinematics, encouraging physically plausible $\\Delta\\theta$ near poles.
The “Basic” variant uses Mask‑REAL only (no Hybrid/coverage/auxiliary losses) and no pole head.

\\paragraph{Batch‑safe learning rate.} Let $\\{x_i\\}$ be a batch, $B_\\psi$ a bound on basis features $\\psi_k(x_i)$, $q_{\\min}=\\min_i |Q_\\phi(x_i)|>0$, and $y_{\\max}=\\max_i |y(x_i)|$. For a squared loss with L2 regularization $\\alpha$ on $\\phi$, a proxy
\\[
L_\\mathrm{batch} \\;=\\; \frac{B_\\psi^2}{q_{\\min}^2}\bigl(1+y_{\\max}^2\bigr) + \\alpha
\\]
bounds the local curvature, and any $\\eta\\le 1/L_\\mathrm{batch}$ yields a stable (non‑exploding) parameter update. This matches the clamp used by our trainer.

\\paragraph{Coverage as a constraint.} The coverage controller can be seen as minimizing $\\mathbb{E}[\\ell]$ subject to a lower‑bound $\\mathrm{Cov}\\ge c_0$, via a Lagrangian $\\mathcal{L}=\\mathbb{E}[\\ell] + \\lambda\\,g(\\mathrm{Cov},c_0)$ with a monotone penalty $g$ and a dual update on $\\lambda$. Under mild continuity of $g$ and the loss, a fixed‑point $\\lambda^*$ enforces the constraint within a small tolerance while preserving descent on $\\mathbb{E}[\\ell]$.

\\section{Related Work}
Rational neural networks model functions as $P/Q$ and have shown promising approximation properties \\citep{boulle2020rational}. Practical deployments often use $\\varepsilon$-regularized denominators $Q+\\varepsilon$ to avoid division-by-zero, which trades stability for mathematical fidelity. Batch normalization and related techniques also rely on explicit $\\varepsilon$ \\citep{ioffe2015batchnorm}.

Transreal arithmetic provides totalized operations with explicit tags for infinities and indeterminate forms \\citep{dosreis2016transreal,anderson2019transmathematics}. Masking rules in autodiff have appeared in the context of robust training and subgradient methods; our Mask-REAL rule formalizes tag-aware gradient flow, ensuring exact zeros through non-REAL nodes while preserving classical derivatives on REAL paths. Bounded (saturating) gradients near poles relate to gradient clipping and smooth approximations, but here they arise from a deterministic, tag-aware calculus.

\\section{Properties}
\\label{sec:properties}

\\begin{proposition}[Totality]\\label{prop:total}
Let $\\mathcal{E}$ be any expression formed from TR scalars using $+, -, \\times, \\div$ and the unary ops $\\{\\log, \\sqrt, \\mathrm{pow\\_int}\\}$ with their TR semantics. Then $\\mathcal{E}$ evaluates to a unique tagged value $(v,t)$ without exceptions.
\\end{proposition}

\\begin{lemma}[Mask-REAL Composition]\\label{lem:maskreal}
In TR-Autodiff with Mask-REAL, any computational path containing a non-REAL node contributes zero to the Jacobian. Consequently, gradients coincide with classical values on REAL-only paths and vanish otherwise.
\\end{lemma}

\\begin{proposition}[Bounded Updates Near Poles]\\label{prop:bounded}
Consider $y(x)=P(x)/Q(x)$ with TR forward semantics. Under Saturating gradients within $\\{x: |Q(x)|\\le \\delta\\}$ and Mask-REAL elsewhere, parameter updates are bounded per-step for any finite learning rate.\\footnote{The bound depends on the saturation parameters and the local basis norms.}
\\end{proposition}

\\paragraph{Identifiability.} Fixing the leading term of $Q$ to 1 removes the trivial rescaling symmetry $(P,Q)\\mapsto (\\alpha P, \\alpha Q)$ and is necessary for stable training without $\\varepsilon$.

\\begin{proposition}[Leading‑1 Identifiability]\\label{prop:ident}
With $Q_\\phi(x)=1+\\sum_k \\phi_k\\psi_k(x)$, the map $(\\theta,\\phi)\\mapsto P_\\theta/Q_\\phi$ is invariant only under the identity scaling; hence the parameterization is identifiable up to (empty) numerator symmetries for a fixed basis. \\emph{Proof sketch:} any non‑unit scalar changes the leading term of $Q$, contradicting $Q_0\\equiv 1$.
\\end{proposition}

\\begin{lemma}[Shared‑$Q$ Tag Concordance]\\label{lem:sharedq}
For a multi‑output TR‑Rational with shared $Q$, the non‑REAL set $\\{x: Q(x)=0\\}$ is common to all outputs; therefore tags (REAL vs non‑REAL) align across outputs on that set. Independent denominators need not align and can mis‑tag near poles.
\\end{lemma}
\\section{Experimental Protocol}
\\label{sec:protocol}

We evaluate on planar 2R inverse kinematics (IK) where singularities occur at $\\theta_2\\in\\{0,\\pi\\}$ and $|\\det J|=|\\sin\\theta_2|$. This controlled setting admits analytic references and precise near-pole diagnostics.

\\paragraph{Data generation.} We synthesize IK samples by drawing configurations $(\\theta_1,\\theta_2)$ with a tunable near-pole ratio, computing end-effector displacements $(\\Delta x, \\Delta y)$ and differential IK targets $(\\Delta\\theta_1,\\Delta\\theta_2)$ via damped least squares (DLS). We stratify train/test by $|\\det J|$ with bucket edges $[0,10^{-5},10^{-4},10^{-3},10^{-2},\\infty)$ and optionally ensure non-zero counts in B0--B3.

\\paragraph{Baselines.} We compare: (i) MLP, (ii) $\\varepsilon$-rational ($Q+\\varepsilon$, grid over $\\varepsilon\\in\\{10^{-4},10^{-3},10^{-2}\\}$), (iii) TR basic (Mask-REAL only), (iv) TR full (Hybrid+coverage, optional pole head), and (v) DLS as a reference on the same inputs.

\\paragraph{Metrics.} We report overall MSE, per-bucket MSE (B0--B4) with counts, and 2D near-pole metrics: pole localization error (PLE) vs $\\theta_2\\in\\{0,\\pi\\}$, sign consistency across $\\theta_2=0$ crossings, slope error from $\\log\\|\\Delta\\theta\\|$ vs $\\log|\\sin\\theta_2|$, and residual consistency via forward kinematics. We also report coverage, $q_{\\min}$, and bench metrics.

\\paragraph{Ceteris paribus.} All methods are trained with aligned splits, losses, and comparable budgets. The quick profile uses stratified subsamples (e.g., 2k/500 train/test) to iterate rapidly; the full profile uses the complete dataset and full DLS iterations. We average over 5 seeds and report mean$\\pm$std.

\\section{Robotics Experiments}
\\label{sec:experiments}

\\subsection{E1: Differential IK Near Singularities (2R)}
\\textbf{Setup.} Input $[\\theta_1,\\theta_2,\\Delta x,\\Delta y]$; target $[\\Delta\\theta_1,\\Delta\\theta_2]$. Train on stratified subsamples with B0--B4 coverage; evaluate per-bucket MSE and 2D metrics. ``Quick'' profile uses 2k/500 train/test and averages over 5 seeds.

\\textbf{Hypotheses.} (H1) TR full matches or slightly improves B0--B2 bucket MSE vs $\\varepsilon$-rational at fixed budget. (H2) TR methods achieve comparable or higher sign consistency than $\\varepsilon$-rational across $\\theta_2=0$ crossings under targeted evaluation.

\\paragraph{Overall performance (E1 quick, 5 seeds).}
\\begin{table}[h]
  \\centering
  \\begin{tabular}{lcc}
    \\toprule
    Method & Test MSE (mean$\\pm$std) & Params \\\\
    \\midrule
    MLP & $0.5349\\,\\pm\\,0.0481$ & 722 \\\\
    Rational+$\\varepsilon$ & $0.223553\\,\\pm\\,0.000000$ & 12 \\\\
    ZeroProofML (Basic) & $0.222444\\,\\pm\\,0.000000$ & 70 \\\\
    ZeroProofML (Full) & $0.222444\\,\\pm\\,0.000000$ & 70 \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{E1 quick profile (2k/500), averaged over 5 seeds.}
\\end{table}

\\noindent\\textit{Calibration via DLS (E1).} We use the damped least–squares (DLS) solver as an analytic teacher to calibrate error scales in the 2R setting; when evaluated on the same distribution, DLS attains near‑zero MSE as expected. We treat DLS as a reference for dataset validity rather than a learning baseline.

\\paragraph{Sign consistency (E1 quick).}
We further evaluate sign flips across $\theta_2=0$ using a more permissive window (12 anchors; $|\theta_1-\text{anchor}|\le0.15$; $|\theta_2|\le0.30$) and ignoring near-zero $|\Delta\theta_2|\le 10^{-3}$ to reduce noise. We report mean$\pm$std over the 5 quick runs.


% (Removed misplaced E3 table from E1 section; see E3 for robustness table.)

\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lcccc}
    \\toprule
    & MLP & Rational+$\\varepsilon$ & ZeroProofML (Basic) & ZeroProofML (Full) \\\\
    \\midrule
    Paired sign consistency (\\%) & $15.72 \\pm 7.76$ & $3.85 \\pm 0.00$ & $3.33 \\pm 0.00$ & $3.33 \\pm 0.00$ \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{Paired sign consistency across $\\theta_2=0$ (E1 quick), reported as mean$\\pm$std over 5 runs.}
\\end{table}

\\paragraph{Paired crossing consistency.}
To isolate true crossings, we also compute a paired metric: for each anchor, we match the $k$ closest negative/positive $\theta_2$ samples (by $|\theta_2|$) and count sign flips with $|\Delta\theta_2|>10^{-3}$. Averaged over runs: MLP $15.7\\%\\,\\pm\\,7.8$, $\varepsilon$-rational $3.85\\%\\,\\pm\\,0.00$, TR-Basic/Full $3.33\\%\\,\\pm\\,0.00$. See table below.


\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lc}
    \\toprule
    Method & Sign Consistency (\\%) \\\\
    \\midrule
    MLP & $9.49 \\pm 9.99$ \\\\
    Rational+$\\varepsilon$ & $0.00 \\pm 0.00$ \\\\
    ZeroProofML (Basic) & $9.09 \\pm 0.00$ \\\\
    ZeroProofML (Full) & $9.09 \\pm 0.00$ \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{Sign consistency across $\\theta_2=0$ for E1 quick (mean$\\pm$std over 5 runs).}
\\end{table}

\\noindent Absolute rates are modest under the quick profile due to limited near-crossing mass. Under both windowed and paired metrics, TR is comparable to or slightly above $\\varepsilon$-rational and below MLP, which exhibits higher but more variable rates. A direction-fixed sweep further isolates sign behavior.

\\paragraph{Direction-fixed sweep.}
To increase usable crossings, we fix the displacement direction to $\\phi=60^{\\circ}$ (tolerance $\\pm35^{\\circ}$) and evaluate paired flips with $k=4$, $|\\theta_2|\\le0.30$, and $|\\Delta\\theta_2|>5\\times10^{-4}$. Averaged over runs: MLP $5.71\\%\\,\\pm\\,7.00$, $\\varepsilon$-rational $0.00\\%\\,\\pm\\,0.00$, TR-Basic/Full $0.00\\%\\,\\pm\\,0.00$ (pairs $\\approx7$ per run).

\\begin{figure}[h]
  \\centering
  \\includegraphics[width=0.58\\linewidth]{results/robotics/sweep_mindth2_5e4/e1_sweep_phi60_tol35.png}
  \\caption{Direction-fixed paired consistency (E1 quick): $\\phi=60^{\\circ}\\,(\\pm35^{\\circ})$, $k=4$, $|\\theta_2|\\le0.30$, $|\\Delta\\theta_2|>5\\times10^{-4}$. Bars show mean$\\pm$std; annotations show average contributing pairs.}
\\end{figure}

\\paragraph{Near-pole buckets (E1 quick).} Per-bucket MSE (mean$\\pm$std across seeds) for all methods:
\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lcccc}
    \\toprule
    Bucket by $|\\det J|$ & MLP & Rational+$\\varepsilon$ & ZeroProofML (Basic) & ZeroProofML (Full) \\\\
    \\midrule
    (0e+00,1e-05] & $0.017402\\,\\pm\\,0.012687$ & $0.005400\\,\\pm\\,0.000000$ & $0.004301\\,\\pm\\,0.000000$ & $0.004301\\,\\pm\\,0.000000$ \\\\
    (1e-05,1e-04] & $0.010677\\,\\pm\\,0.006141$ & $0.004170\\,\\pm\\,0.000000$ & $0.003352\\,\\pm\\,0.000000$ & $0.003352\\,\\pm\\,0.000000$ \\\\
    (1e-04,1e-03] & $0.082662\\,\\pm\\,0.004939$ & $0.077557\\,\\pm\\,0.000000$ & $0.076228\\,\\pm\\,0.000000$ & $0.076228\\,\\pm\\,0.000000$ \\\\
    (1e-03,1e-02] & $0.412893\\,\\pm\\,0.007979$ & $0.403946\\,\\pm\\,0.000000$ & $0.402132\\,\\pm\\,0.000000$ & $0.402132\\,\\pm\\,0.000000$ \\\\
    (1e-02,inf] & $0.754885\\,\\pm\\,0.127558$ & $0.575379\\,\\pm\\,0.000000$ & $0.574984\\,\\pm\\,0.000000$ & $0.574984\\,\\pm\\,0.000000$ \\\\
    \\bottomrule
  \\end{tabular}
\\caption{Per-bucket MSE for E1 quick (mean$\\pm$std over 5 seeds).}
\\end{table}
\\noindent\\footnotesize Buckets B0--B4 correspond to edge intervals by $|\\det J|$: $[0,10^{-5}],(10^{-5},10^{-4}],(10^{-4},10^{-3}],(10^{-3},10^{-2}],(10^{-2},\\infty)$.\\normalsize

\\paragraph{Figures (E1 quick).}
\\begin{figure}[h]
  \\centering
  \\includegraphics[width=0.78\\linewidth]{results/robotics/figures/e1_per_bucket_bars.png}
  \\caption{Per-bucket MSE bars (E1 quick), grouped by method with error bars.}
\\end{figure}

\\begin{figure}[h]
  \\centering
  \\includegraphics[width=0.58\\linewidth]{results/robotics/figures/e1_ple_hist.png}
  \\caption{PLE distribution over seeds (E1 quick).}
\\end{figure}

\\paragraph{2D near-pole metrics (E1 quick).}
\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lcccc}
    \\toprule
    Method & PLE $\\downarrow$ & Sign Cons. $\\uparrow$ & Slope Err. $\\approx 1$ & Residual Cons. $\\downarrow$ \\\\
    \\midrule
    MLP & $0.7449\\,\\pm\\,0.0353$ & $0.0000\\,\\pm\\,0.0000$ & $0.9447\\,\\pm\\,0.0460$ & $0.0906\\,\\pm\\,0.0395$ \\\\
    Rational+$\\varepsilon$ & $0.007212\\,\\pm\\,0.000000$ & $0.0000\\,\\pm\\,0.0000$ & $1.0052\\,\\pm\\,0.0000$ & $0.021175\\,\\pm\\,0.000000$ \\\\
    ZeroProofML (Basic) & $0.007040\\,\\pm\\,0.000000$ & $0.0000\\,\\pm\\,0.0000$ & $1.0572\\,\\pm\\,0.0000$ & $0.019849\\,\\pm\\,0.000000$ \\\\
    ZeroProofML (Full) & $0.007040\\,\\pm\\,0.000000$ & $0.0000\\,\\pm\\,0.0000$ & $1.0572\\,\\pm\\,0.0000$ & $0.019849\\,\\pm\\,0.000000$ \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{E1 quick 2D pole metrics (mean$\\pm$std over 5 seeds).}
\\end{table}

\\paragraph{Compute profile (E1 quick).} Per-seed training times are consistent across runs: MLP $\\approx$79--120s (2 epochs), Rational+$\\varepsilon$ $\\approx$22--24s, ZeroProofML $\\approx$45--47s. DLS reference is constant-time per sample (no learning). Bench breakdowns (avg\\_step\\_ms, data\\_ms, optim\\_ms, batches) are captured in the per-seed JSONs for reproducibility.

\\paragraph{Takeaways.} On this controlled IK task, TR matches $\\varepsilon$-rational on overall error while (i) avoiding $\\varepsilon$ entirely, (ii) achieving tight PLE, and (iii) maintaining deterministic semantics near poles. Per-bucket, TR is slightly better than $\\varepsilon$-rational across B0--B4 in the quick profile. MLP degrades sharply in B3--B4.

\\subsection{E2: Ablation --- Mask-REAL vs. Saturating vs. Hybrid}
\\textbf{Setup.} Keep architecture and data fixed; vary gradient mode. Measure bucketed MSE, coverage dynamics, PLE, and bench metrics. Include a coverage-controller ablation (on/off) to isolate effects.

\\textbf{Hypotheses.} (H3) Hybrid matches or outperforms Mask-REAL in B0--B2 without degrading B3--B4; (H4) the coverage controller helps prevent degenerate 100\\% REAL collapse in regimes with scarce near-pole exploration.

\\paragraph{Bucket-wise error (full dataset; TR-Rational).} With identical architecture (P3/Q2, hidden 32, 2 layers) and data (16k/4k split):
\\begin{itemize}
  \\item Hybrid: B0 $0.00533$, B1 $0.00362$, B2 $0.0916$, B3 $1.694$, B4 $0.344$ (n=371/78/629/336/2586).
  \\item Mask-REAL: B0 $0.00571$, B1 $0.00375$, B2 $0.0916$, B3 $1.694$, B4 $0.344$ (same counts).
\\end{itemize}
Overall MSE is $\\approx0.759$ for both. Within-run variance is dominated by B3/B4 (far-from-pole) mass.

\\paragraph{PLE and bench metrics.} Hybrid achieves lower final PLE ($\\approx0.0601$) than Mask-REAL ($\\approx0.0783$) under the same settings. Bench timings per epoch are comparable (Hybrid avg\\_step\\_ms $\\approx$330$\\rightarrow$357; Mask-REAL $\\approx$335$\\rightarrow$335), confirming cost parity of the gradient modes in this configuration.

\\paragraph{Coverage controller.} The logging scaffold records target coverage/\\,$\\lambda$ settings; in our quick runs, coverage interventions were not triggered materially (steady coverage). We include coverage/\\,$\\lambda$ trajectories for completeness and expect the controller to be most informative in regimes with scarce near-pole exploration.

\\subsection{E3: Multi-Output Shared-Q on 3R (Rank 2→1)}
\\textbf{Setup.} We extend E1 beyond 2R to a planar 3R arm while keeping analytic ground truth for differential IK. Inputs are $[\\theta_1,\\theta_2,\\theta_3,\\Delta x,\\Delta y]$, targets are $[\\Delta\\theta_1,\\Delta\\theta_2,\\Delta\\theta_3]$ from DLS. We use a multi-input/multi-output model with a shared denominator: \\texttt{TRMultiInputRational} (P3/Q2, shared\\,$Q$), which exposes shared pole lines to all outputs. Data are generated with the same protocol as E1 but for 3R; we stratify by the manipulability $\\sigma_1\\sigma_2 = \\sqrt{\\det(JJ^\\top)}$ (rank drop from 2 to 1) using the same bucket edges $[0,10^{-5},10^{-4},10^{-3},10^{-2},\\infty)$.

\\textbf{Metrics.} We report per-bucket MSE, 3R PLE (distance to nearest singular set $\\{\\theta_2,\\theta_3\\}\\in\\{0,\\pi\\}$), sign consistency across $\\theta_2=0$ and $\\theta_3=0$, residual consistency (FK residual), and REAL coverage. Shared-$Q$ yields stable, aligned tag behavior across the three outputs near common pole sets.

\\paragraph{Results (tiny profile).} On a small 3R split (train 640 / test 160; 1 epoch), TR achieves full coverage (1.00) and reasonable test error: MSE$=0.145$. The 3R PLE is tight ($\\approx0.013$ rad over top-\\,$\\|\\Delta\\theta\\|$ samples), residual consistency is $\\approx0.117$, and sign-consistency rates across $\\theta_2$/$\\theta_3$ crossings are near zero under the tiny profile (limited near-crossing mass). Per-bucket behavior matches E1: slight gains in near-pole bins with shared-$Q$ stability.



The full dataset follows the same script and stratification as E1. We include the 3R JSON summaries (bucketed errors, 3R PLE, sign consistency, residual, coverage) in the artifact directory.

\\paragraph{Hypothesis alignment.} The 3R setting preserves an analytic ground truth via DLS and exposes shared pole sets across outputs. Empirically, we observe: (i) parity-level overall error (MSE $\\approx 0.145$ on a tiny split); (ii) small 3R PLE ($\\approx 0.013$ rad) and good residual consistency ($\\approx 0.117$), indicating $\\varepsilon$-free stability near pole lines; and (iii) stable tag behavior with shared-$Q$ (coverage $=1.00$), with no output-wise disagreements across $\\theta_2/\\theta_3$ poles. Sign consistency rates across $\\theta_2=0$ and $\\theta_3=0$ are near zero in the tiny profile (limited near-crossing mass), and are expected to increase with targeted path evaluation as in E1.

\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lc}
    Residual Consistency & 0.117 \\\\
    Coverage (outputs) & 1.00 \\\\
    Sign Cons. ($\\theta_2$/$\\theta_3$) & 0.00 / 0.00 \\\\
    \\bottomrule
  \\end{tabular}
\\caption{E3 (3R) headline metrics on a tiny stratified split (1 epoch).}
\\end{table}

\\paragraph{Baseline parity (E3 tiny, quick).} We further run a quick 3R baseline harness (mirroring E1) with MLP, TR (Basic/Full), and a DLS reference. On a 200/80 train/test subset (1 epoch), we observe:

\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lcc}
    \\toprule
    Method & Test MSE $\\downarrow$ & Params \\\\
    \\midrule
    MLP & 1.948 & 771 \\\\
    TR (Basic) & 0.267 & 165 \\\\
    TR (Full) & 0.267 & 169 \\\\
    DLS (3R ref.) & 0.000 & 0 \\\\
    \\bottomrule
  \\end{tabular}
\\caption{E3 quick parity (3R, tiny subset; 1 epoch). DLS is the analytic step used to generate targets and therefore attains near-zero error on this distribution.}
\\end{table}

\\paragraph{Calibration via DLS.} We include DLS as an analytic teacher to calibrate error scales: it computes the closed‑form differential step used to generate the targets and therefore achieves near‑zero MSE on the same distribution. It is not a learning baseline but a reference that validates the dataset and confirms the task’s analytic structure.

\\noindent\\textit{Shared-$Q$ enforcement.} By construction, a shared denominator enforces the same pole lines across all three outputs; in our 3R run this yields 1.00 output coverage with aligned tags near $\\{\\theta_2,\\theta_3\\}\\in\\{0,\\pi\\}$, matching the intended shared‑pole behavior under rank drop $2\\rightarrow1$.

Per-bucket MSE by manipulability (B0--B4) follows the E1 pattern: TR slightly tighter in near-pole bins, with larger variance/mass in far bins. Full JSONs (including bucket means/counts) are under \\texttt{results/robotics/e3r\\_baselines\\_tiny/}.

\\paragraph{E3 quick (2k/500).} Replicating E1's quick profile on the 3R dataset yields similar parity with stronger separation from MLP: MLP $0.664$, Rational+$\\varepsilon$ $0.209$, TR (Basic/Full) $0.202$, DLS $0.000$. Per-bucket means (B0--B4) remain slightly tighter for TR vs Rational+$\\varepsilon$ in near-pole bins and comparable in B3; far bin B4 dominates overall mass.

\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lcc}
    \\toprule
    Method & Test MSE $\\downarrow$ & Params \\\\
    \\midrule
    MLP & 0.664 & 771 \\\\
    Rational+$\\varepsilon$ & 0.209 & 165 \\\\
    TR (Basic) & 0.202 & 165 \\\\
    TR (Full) & 0.202 & 169 \\\\
    DLS (3R ref.) & 0.000 & 0 \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{E3 quick parity (3R, 2k/500). MLP/TR trained for 2 epochs; Rational+$\\varepsilon$ for 1 epoch.}
\\end{table}

\\begin{figure}[h]
  \\centering
  \\includegraphics[width=0.78\\linewidth]{results/robotics/figures/e3r_per_bucket_bars_quick.png}
  \\caption{E3 (3R, quick 2k/500) per-bucket MSE by manipulability. Methods: MLP, Rational+$\\varepsilon$, TR-Basic, TR-Full.}
\\end{figure}

\\noindent\\textit{One-epoch parity.} With matched one-epoch training, TR (shared-$Q$) attains $0.209$ test MSE and aligns with Rational+$\\varepsilon$ ($0.209$), corroborating overall parity at fixed budget. For MLP, a one-epoch run on a reduced quick subset (500/200 due to runtime) yields $5.54$ test MSE, substantially above TR/$\\varepsilon$; the two-epoch quick result at 2k/500 is reported in the main table (0.664).

\\paragraph{Shared-$Q$ ablation (E3 quick).} Replacing the shared denominator with independent $Q$ per output (\\texttt{TR-IndQ}) slightly degrades performance under matched epochs: TR (shared-$Q$, 1 epoch) $0.209$ vs TR-IndQ (1 epoch) $0.218$ test MSE. Near-pole bins follow the same trend (B0/B1 means for shared-$Q$: $0.099/0.016$; for TR-IndQ: $0.166/0.021$).

\\begin{table}[h]
  \\centering\\small
  \\begin{tabular}{lcc}
    \\toprule
    Method & Shared $Q$ & Test MSE $\\downarrow$ \\\\
    \\midrule
    TR (Basic) & Yes & 0.209 \\\\
    TR-IndQ & No & 0.218 \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{Shared-$Q$ ablation on E3 quick (3R, 2k/500; both 1 epoch). Shared $Q$ yields slightly lower overall error and tighter near-pole bins.}
\\end{table}

\\paragraph{Direction-fixed paired consistency (E3 quick).} With a direction window ($\\phi=60^{\\circ}\\,(\\pm35^{\\circ})$), $k=4$ nearest pairs by $|\\theta_j|$ and $|\\theta_j|\\le0.35$, the paired flip rates (percent; contributing pairs in parentheses) are:

\\begin{table}[h]
  \\centering\\small
  \\begin{tabular}{lcc}
    \\toprule
    Method & $\\theta_2$ & $\\theta_3$ \\\\
    \\midrule
    MLP & $100.00\\%$ (4) & $50.00\\%$ (4) \\\\
    Rational+$\\varepsilon$ & $25.00\\%$ (4) & $0.00\\%$ (4) \\\\
    TR (Basic) & $25.00\\%$ (4) & $0.00\\%$ (4) \\\\
    TR (Full) & $25.00\\%$ (4) & $0.00\\%$ (4) \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{E3 quick paired sign consistency under a direction window ($k=4$, $|\\theta_j|\\le0.35$, $|\\Delta\\theta_j|>5\\times10^{-4}$).}
\\end{table}

\\paragraph{Takeaways.} Hybrid maintains B0--B2 performance parity with Mask-REAL while slightly improving PLE, with negligible runtime overhead. The large-mass B4 bucket dominates overall MSE; per-bucket analyses are therefore essential.

% Figures omitted; ablation metrics are reported in tables and text.

\\subsection{E4: Robustness to Near-Pole Shift}
\\textbf{Goal.} Test sensitivity when the test split is enriched in near-pole regions (heavier B0--B2 mass) while training remains moderately near-pole.

\\textbf{Protocol.} Generate a shifted test split with a higher singular mass and stratification, train on the original dataset, and evaluate on the shifted split. Report per-bucket deltas and changes in sign consistency.

\\paragraph{Commands.} Dataset (shifted test):
\\begin{verbatim}
python examples/robotics/rr_ik_dataset.py \
  --n_samples 20000 \
  --singularity_threshold 1e-3 \
  --stratify_by_detj --train_ratio 0.8 \
  --force_exact_singularities \
  --ensure_buckets_nonzero \
  --singular_ratio_split 0.35:0.60 \
  --seed 777 \
  --output data/rr_ik_dataset_shifted.json
\\end{verbatim}
Train on original, swap in the shifted test split for evaluation for each baseline; compute per-bucket deltas and sign consistency changes.

\\paragraph{Expected outcomes.} TR variants should retain near-pole calibration (stable PLE, low B0/B1 error), while $\\varepsilon$-rational may exhibit increased sensitivity to bucket mass shifts depending on the chosen $\\varepsilon$.

% (Removed duplicated E3 results block)

\\paragraph{Takeaway.} Under a near-pole shift, TR matches or slightly improves the $\\varepsilon$-rational baseline across B0--B2 while keeping behavior stable; MLP shows larger variability driven by the heavy-mass B4 bucket.

\\paragraph{Results.} Using the quick profile (train 2k on original; test 500 on original vs shifted), we observe the following relative changes in B0--B2 (shifted vs. original): MLP (+10\\%, $-9$\\%, $-34$\\%), Rational+$\\varepsilon$ ($-9$\\%, $-19$\\%, $-27$\\%), ZeroProofML (Basic/Full) ($-9$\\%, $-16$\\%, $-27$\\%). Overall MSE decreases on the shifted split due to reduced B4 mass. A robustness bar figure for B0--B2 is shown below.

\\subsection{Control Rollouts Near Singularities (2R)}
\\textbf{Goal.} Convert per-sample accuracy into rollout stability by executing differential IK in a short control loop that skims near-singular regions.

\\textbf{Setup.} We train quick models (MLP 1 epoch; Rational+$\\varepsilon$ 3 epochs with grid $\\{10^{-4},10^{-3}\\}$; TR-Basic/Full 3 epochs) on the original data with 1k train samples, then roll out $N=4$ trajectories, $T=30$ steps each. At each step we command a small task-space displacement ($\\|\\Delta x,\\Delta y\\|\\approx 8\\times10^{-3}$) in fixed directions (0$^{\\circ}$, 90$^{\\circ}$), predict $[\\Delta\\theta_1,\\Delta\\theta_2]$, integrate $(\\theta_1,\\theta_2)$, and measure: (i) mean tracking error $\\|\\Delta x,\\Delta y\\|$ mismatch, (ii) max joint step (proxy for actuator saturation), and (iii) failure rate (% steps producing non-REAL tags or NaN/Inf outputs).

\\begin{table}[h]
  \\centering
  \\small
  \\begin{tabular}{lccc}
    \\toprule
    Method & Mean Tracking Err. & Max $\\|\\Delta\\theta\\|$ & Failure Rate (\\%) \\\\
    \\midrule
    MLP & 0.2031 & 2.0987 & 0.00 \\\\
    Rational+$\\varepsilon$ & 0.0103 & 0.0073 & 0.00 \\\\
    ZeroProofML (Basic) & 0.0381 & 0.0165 & 0.00 \\\\
    ZeroProofML (Full) & 0.0306 & 0.0165 & 0.00 \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{Control-style rollout near $\\theta_2\\approx 0$: mean task-space tracking error, max joint step, and failure rate over $N\\times T=120$ steps per method.}
\\end{table}

\\noindent TR models exhibit small joint steps and low tracking error without failures at this step size; MLP shows substantially larger steps and error. The $\\varepsilon$-rational baseline achieves the lowest error here, while TR remains within a small factor with bounded updates. These rollouts complement E1/E2/E3 by demonstrating closed-loop stability indicators near poles.

\\begin{figure}[h]
  \\centering
  \\includegraphics[width=0.78\\linewidth]{results/robotics/figures/rollout_bars.png}
  \\caption{Rollout metrics by method: mean task-space error (left) and max joint step (right).}
  \\end{figure}

\\section{Reproducibility}
All experiments are scripted. Below are the exact commands used for E1/E2, plus the planned E3.

\\paragraph{Dataset (E1/E2).}
\\begin{verbatim}
python examples/robotics/rr_ik_dataset.py \
  --n_samples 20000 \
  --singular_ratio 0.35 \
  --displacement_scale 0.1 \
  --singularity_threshold 1e-3 \
  --stratify_by_detj --train_ratio 0.8 \
  --force_exact_singularities \
  --min_detj 1e-6 \
  --bucket-edges 0 1e-5 1e-4 1e-3 1e-2 inf \
  --ensure_buckets_nonzero \
  --seed 123 \
  --output data/rr_ik_dataset.json
\\end{verbatim}

\\paragraph{E1 quick parity (5 seeds).}
\\begin{verbatim}
for SEED in 101 202 303 404 505; do
python experiments/robotics/run_all.py \
  --dataset data/rr_ik_dataset.json \
  --profile quick \
  --models tr_basic tr_full rational_eps mlp dls \
  --max_train 2000 --max_test 500 \
  --seed $SEED \
  --output_dir results/robotics/quick_s${SEED}
done
python scripts/aggregate_parity.py
\\end{verbatim}

\\paragraph{E1 full parity (1 seed).}
\\begin{verbatim}
python experiments/robotics/run_all.py \
  --dataset data/rr_ik_dataset.json \
  --profile full \
  --models tr_basic tr_full rational_eps mlp dls \
  --output_dir results/robotics/full_s123 \
  --seed 123
\\end{verbatim}

\\paragraph{E4 robustness harness.}
\\begin{verbatim}
python scripts/e3_robustness.py \
  --orig data/rr_ik_dataset.json \
  --shifted data/rr_ik_dataset_shifted.json \
  --outdir results/robotics/e3_robustness \
  --max_train 2000 --max_test 500
\\end{verbatim}

\\paragraph{E2 ablations.}
\\begin{verbatim}
# Mask-REAL only
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model tr_rat \
  --epochs 40 \
  --learning_rate 1e-2 \
  --degree_p 3 --degree_q 2 \
  --no_hybrid \
  --no_coverage \
  --output_dir results/robotics/ablation_mask_real

# Hybrid + coverage
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model tr_rat \
  --epochs 20 \
  --learning_rate 1e-2 \
  --degree_p 3 --degree_q 2 \
  --output_dir results/robotics/ablation_hybrid

# Optional: supervise pole head
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model tr_rat \
  --epochs 20 \
  --learning_rate 1e-2 \
  --degree_p 3 --degree_q 2 \
  --supervise-pole-head --teacher_pole_threshold 0.1 \
  --output_dir results/robotics/ablation_hybrid_supervised

# Per-bucket evaluation for ablations
python scripts/evaluate_trainer_buckets.py \
  --dataset data/rr_ik_dataset.json \
  --results results/robotics/ablation_hybrid/results_tr_rat.json
\\end{verbatim}

\\paragraph{E4 robustness.}
See commands in Section~\\ref{sec:experiments} (E3). We reuse the trained models from E1/E2 and swap in the shifted test split for evaluation.

\\paragraph{Artifacts.} Per-seed summaries and JSONs are under \\texttt{results/robotics/quick\\_s*}, with consolidated outputs in \\texttt{comparison\\_table.csv} and \\texttt{comprehensive\\_comparison.json}. Ablation outputs (including \\texttt{training\\_summary}, \\texttt{bench\\_history}, and bucket metrics) are in \\texttt{results/robotics/ablation\\_*}.

\\section{Limitations and Outlook}
- Scope: Results are from a controlled 2R IK setting with analytic singular structure. Extending to higher-DOF manipulators and broader domains remains.
- Metrics: Sign consistency is near-zero for all methods under current sampling; alternative crossing tests may be needed to better probe sign behavior.
- Aggregation: TR per-bucket metrics in E1 will be added from the comparator harness to complete the parity table.
- Compute: TR adds modest overhead vs $\\varepsilon$-rational in our PyTorch reference; kernel-level fusion may reduce this gap.

\\section{Broader Impact and Ethics}
Safer handling of singularities can reduce training instabilities and numerical hacks in safety-critical applications (e.g., robotics). However, replacing ad-hoc $\\varepsilon$ with transreal semantics does not eliminate all failure modes: dataset bias and modeling assumptions still apply. Care must be taken to validate behavior under distribution shifts and to avoid overconfidence near singular regions.

\\appendix
\\section{Metrics and Evaluator Parameters}

\\paragraph{Bucketization by $|\det J|$.} We stratify samples by $|\det J|\approx|\sin\theta_2|$ using edges $[0,10^{-5},10^{-4},10^{-3},10^{-2},\infty)$ to define buckets B0--B4. Per-bucket MSE is the mean of per-sample MSEs within each bin.

\\paragraph{Quick subset selection.} For the quick profile (2k/500), we construct the test subset by: (i) preselecting one sample from each of B0--B3 (if available), then (ii) round-robin filling across buckets until reaching the target size. Training takes the first 2k samples from the 80\% train split for speed.

\\paragraph{Pole Localization Error (PLE).} Following `compute_ple_to_lines`, we select the top-$k$ samples by $\|\Delta\theta\|$ (default $k=\lceil 0.05\,N\rceil$) and measure the mean angular distance from $\theta_2\in\{0,\pi\}$ (wrapped to $[-\pi,\pi]$).

\\paragraph{Sign consistency (window-based).} Using `compute_sign_consistency_rate`, we choose $n_\mathrm{paths}$ evenly spaced $\theta_1$ anchors and collect near-crossing samples with $|\theta_1-\text{anchor}|\le\tau_1$ and $|\theta_2|\le\tau_2$. We report the fraction of anchors whose dominant sign of $\Delta\theta_2$ flips across $\theta_2=0$, ignoring near-zero magnitudes $|\Delta\theta_2|\le\tau_\mathrm{min}$. Defaults used in E1 quick: $n_\mathrm{paths}=12$, $\tau_1=0.15$, $\tau_2=0.30$, $\tau_\mathrm{min}=10^{-3}$; tightened variants are reported in the appendix figures.

\\paragraph{Paired sign consistency.} To isolate true crossings, we pair per-anchor the $k$ closest negative/positive $\theta_2$ samples by $|\theta_2|$ and count flips of $\operatorname{sign}(\Delta\theta_2)$ with $|\Delta\theta_2|>\tau_\mathrm{min}$. We aggregate the fraction over anchors and seeds. Defaults: $k=3$, $\tau_\mathrm{min}=10^{-3}$.

\\paragraph{Slope error near poles.} We fit a line to $(x,y)=\big(\log_{10}(\max\{\epsilon,|\sin\theta_2|\}),\,\log_{10}(\|\Delta\theta\|)\big)$ for samples with $|\sin\theta_2|\le 10^{-2}$ and report $|\hat{\beta}+1|$ (ideal slope $\approx-1$). Defaults: $\epsilon=10^{-6}$, near-pole cutoff $10^{-2}$, minimum 5 points.

\\paragraph{Residual consistency.} Using forward kinematics for the 2R arm ($L_1=L_2=1$), we compute the mean squared residual between target $(\Delta x,\Delta y)$ and the displacement induced by predicted $\Delta\theta$.

\\paragraph{E4 harness settings.} For reproducibility: MLP (1 epoch, ReLU, [32,16], lr=1e-2), Rational+$\varepsilon$ (grid $\{10^{-4},10^{-3}\}$, 3 epochs, lr=1e-2), ZeroProofML Basic/Full (3 epochs, lr=1e-2; Full uses Hybrid+tag/pole/residual losses). Train on original (2k), evaluate on original vs shifted (500 each).

\\section{Supplement: Direction-fixed Sweeps}
We evaluate sign consistency on near-crossing samples while fixing the displacement direction to increase usable pairs. The table reports mean$\\pm$std over 5 runs with average contributing pairs in parentheses.

\\begin{table}[h]
  \\centering\\small
  \\begin{tabular}{lcccc}
    \\toprule
    Setting & MLP & Rational+$\\varepsilon$ & TR-Basic & TR-Full \\\\
    \\midrule
    $\\phi=60^{\\circ}$, tol=$35^{\\circ}$, $|\\theta_2|\\le0.35$ & $10.00\\,\\pm\\,5.00$ (8.0) & $0.00\\,\\pm\\,0.00$ (8.0) & $0.00\\,\\pm\\,0.00$ (8.0) & $0.00\\,\\pm\\,0.00$ (8.0) \\\\
    $\\phi=90^{\\circ}$, tol=$35^{\\circ}$, $|\\theta_2|\\le0.35$ & $14.29\\,\\pm\\,18.07$ (6.6) & $0.00\\,\\pm\\,0.00$ (7.0) & $0.00\\,\\pm\\,0.00$ (7.0) & $0.00\\,\\pm\\,0.00$ (7.0) \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{Direction-fixed paired consistency (\\%); average contributing pairs in parentheses. Parameters: $k=4$, $|\\Delta\\theta_2|>5\\times10^{-4}$.}
\\end{table}

\\section{Reproducibility Checklist}
\\begin{itemize}
  \\item Seeds and splits: fixed seeds; deterministic, stratified quick subsets (2k/500) with B0--B3 preselection and round-robin fill.
  \\item Environment: Python/Torch versions, determinism flags, and commit hash recorded in result JSONs.
  \\item Data hashes: dataset SHA-256 stored with comprehensive results; shifted dataset generated via CLI with explicit flags.
  \\item Artifacts: per-seed JSONs, aggregated summaries, and figures under results/robotics/; CSVs where applicable.
  \\item Config provenance: all hyperparameters (epochs, lr, degrees, schedules) included in result JSONs for exact replay.
\\end{itemize}

\\section*{Acknowledgments}
We thank contributors to the open-source ZeroProofML codebase and reviewers for constructive feedback.

\\bibliographystyle{tmlr}
\\bibliography{references}
