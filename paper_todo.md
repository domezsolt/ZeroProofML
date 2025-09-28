# ZeroProofML Paper — No-Rerun To-Do List
**Date:** 2025-09-28  
**Scope:** Tasks that only require text/LaTeX/documentation edits — **no experimental reruns**.

---

## Top Priorities (do these first)

### 1) Unify TR Tags & Macros Across the Paper
**Goal:** Eliminate ambiguity (`INF` vs `±INF`, `NULL` vs `Φ`) and enforce one canonical tag set everywhere (text, math, algorithms, tables, figure captions).
- [ ] Add macros to the preamble:
  ```tex
  % --- TR tag macros ---
  \newcommand{\TAGREAL}{\textsc{REAL}}
  \newcommand{\TAGPINF}{\textsc{+INF}}
  \newcommand{\TAGNINF}{\textsc{−INF}}
  \newcommand{\TAGPHI}{\textsc{$\Phi$}}
  % Optional helpers
  \newcommand{\TR}{\mathbb{T}}
  \DeclareMathOperator{\sign}{sign}
  ```
- [ ] Replace **all** occurrences of `REAL/INF/NULL/Φ` with `\TAGREAL/\TAGPINF/\TAGNINF/\TAGPHI` as appropriate (including algorithms and tables).
- [ ] Add a 1-paragraph **Notation** subsection listing `\TR`, `\TAGREAL`, `\TAGPINF/\TAGNINF`, `\TAGPHI` and what each means.

**Quick check commands**
```bash
# Inspect where legacy tags still appear (adjust path if needed)
grep -nE '\\b(INF|NULL|Phi|REAL)\\b' paper_v3.tex
```

---

### 2) Formalize Claims as Falsifiable Statements (Assumptions → Propositions)
**Goal:** Convert headline guarantees into testable propositions without new runs.
- [ ] Insert an **Assumptions** block (A1–A4) once (regularity/Lipschitz, boundedness, hysteresis margins \(\delta_\mathrm{{on/off}}\), randomness control).
- [ ] Add 3 short propositions with labels + proof sketches (3–6 lines each):
  ```tex
  \begin{assumption}[Regularity and Hysteresis]\label{ass:regularity}
  % A1–A4 concise statements
  \end{assumption}

  \begin{proposition}[Bounded Updates]\label{prop:bounded}
  Under Assumption~\ref{ass:regularity}, there exists \(C>0\) s.t. 
  \(\|\nabla_\theta \mathcal{L}_t\| \le C\) for all \(t\).
  \end{proposition}

  \begin{proposition}[Finite Switching]\label{prop:finite-switch}
  With hysteresis margins \(\delta_\mathrm{{on/off}}\), the number of
  mode switches on any compact interval is finite and bounded by 
  \(B(\delta_\mathrm{{on/off}}, L)\).
  \end{proposition}

  \begin{proposition}[Determinism]\label{prop:determinism}
  Fixing seeds, dataloader order, and deterministic kernels yields identical outputs for identical inputs.
  \end{proposition}
  ```
- [ ] Add a 1–2 sentence **Takeaway** at the end of each subsection that translates the math into operational language.

---

### 3) Align Abstract/Introduction With Existing Results (No New Runs)
**Goal:** Ensure every abstract/intro claim points to a concrete number in the current Results.
- [ ] In the **Abstract** and **end of the Introduction**, attach **numbers + pointers** pulled from existing tables/figures (no recomputation needed):
  ```tex
  We reduce near-singularity MSE by \textbf{37\%} in B0 and \textbf{22\%} in B1
  vs. the best \ensuremath{\varepsilon}-baseline (Table~\ref{tab:main}), 
  improve sign consistency from \textbf{88\%}→\textbf{99\%} (Fig.~\ref{fig:sign}), 
  and train \textbf{12×} faster than an ensemble (Table~\ref{tab:speed}).
  ```
- [ ] Mirror the phrasing in Results section headers to make claim→evidence mapping trivial.

---

### 4) LaTeX & Metadata Hygiene (Submission-Ready Polish)
**Goal:** Avoid style nits and placeholder leaks; make the PDF clean for reviewers.
- [ ] Move `hyperref` **last** and hide link boxes:
  ```tex
  \usepackage[hidelinks]{hyperref}  % load last
  ```
- [ ] Keep `microtype` enabled; remove **unused** packages (e.g., `subcaption`, `listings` if not used).
- [ ] Delete placeholders (e.g., `\editor{TBD}`); ensure consistent **Figure/Table** capitalization and caption style (sentence case unless venue requires Title Case).
- [ ] Ensure **every** figure/table/algorithm has a `\label{...}` **and** is cited in text. Move any uncited figure (e.g., `fig:conceptual`) to the appendix or cite it where relevant.
- [ ] Consistent hyphenation: near-singularity, sign-consistent, \(\varepsilon\)-regularization, run-time.

**Quick check commands**
```bash
grep -nE '\\editor\{|\\?\\?|TBD|todo|FIXME' paper_v3.tex
grep -nE '\\label\{|Fig\.|Figure|Tab\.|Table' paper_v3.tex
```

---

### 5) Pseudocode Harmonization (Documentation-Only)
**Goal:** Make the algorithm listings match the **intended semantics** (signed \(\pm\infty\), \(\Phi\) routing) without touching the experimental code.
- [ ] Update algorithm headers/outputs and branch logic explicitly:
  ```tex
  \textbf{Output:} \( (y, \text{tag}) \in \TR \times \{\TAGREAL,\TAGPINF,\TAGNINF,\TAGPHI\} \)

  \If{\( |Q| \le \tau \) \textbf{ and } \( |P| > \tau \)}{
     \( y \gets \sign(P)\cdot \infty \);\quad
     \( \text{tag} \gets \begin{cases}\TAGPINF,& P>0\\ \TAGNINF,& P<0\end{cases} \)
  }
  \ElseIf{\( |Q| \le \tau \) \textbf{ and } \( |P| \le \tau \)}{
     \( y \gets \Phi \);\quad \( \text{tag} \gets \TAGPHI \)
  }
  \Else{ \( y \gets P/Q \);\quad \( \text{tag} \gets \TAGREAL \) }
  ```
- [ ] Give every algorithm a `\label{alg:...}` and cite them (“see Alg.~\ref{alg:tr-forward}”).
- [ ] If the current implementation differs, add a **footnote**: “Listings describe intended semantics; implementation variant noted in Appendix X.”

---

## High-Value Additions (still no reruns)

### 6) Reproducibility Summary From Existing Logs Only
- [ ] Add a compact training-budget table using already-completed runs (fill from logs):
  ```tex
  \begin{table}[t]
  \centering\small
  \begin{tabular}{lcccccc}
  \toprule
  Model & Opt & LR & Epochs & Batch & GPU(s) & Seeds \\
  \midrule
  TR-Rational & AdamW & 3e-4 & 200 & 256 & 1×A100 & 3 \\
  Best \(\varepsilon\)-baseline & AdamW & 3e-4 & 200 & 256 & 1×A100 & 3 \\
  \bottomrule
  \end{tabular}
  \caption{Training budgets and configs (no new runs).}
  \end{table}
  ```
- [ ] Add a short **determinism note** (seeds used, CUDA/cuDNN deterministic flags), and record the **Git commit hash** of the code used to produce the current figures/tables.

### 7) Notation Mini-Table
- [ ] Include a half-column notation table early on:
  ```tex
  \begin{table}[h]
  \centering\small
  \begin{tabular}{ll}
  \toprule
  Symbol & Meaning \\ \midrule
  \(\TR\) & Transreal set (reals \(\cup\) \(\pm\infty\) \(\cup\) \(\Phi\)) \\
  \(\TAGREAL\) & Finite real value \\
  \(\TAGPINF,\TAGNINF\) & Signed infinities \\
  \(\TAGPHI\) & Indeterminate/nullity \\
  \(\tau,\delta_\mathrm{{on/off}}\) & Guard threshold, hysteresis margins \\
  \bottomrule
  \end{tabular}
  \caption{Notation summary.}
  \end{table}
  ```

### 8) Bibliography & Citation Pass
- [ ] Fix any `??` citations; ensure consistent use of `\citep` vs `\citet` per venue style.
- [ ] Add staple robotics-singularity refs (e.g., manipulability/damped least-squares) **only if already cited in text**; otherwise comment them out to avoid orphaned bib entries.

### 9) Figure/Caption Consistency
- [ ] Ensure vector fonts; unify “Figure” vs “Fig.” per venue; sentence-case captions.
- [ ] Cite `fig:conceptual` or move it to the appendix.

### 10) Submission Toggles (Optional but Quick)
- [ ] Add a simple preprint/submission switch:
  ```tex
  % In the preamble
  \newif\ifpreprint
  \preprinttrue   % or \preprintfalse for camera-ready

  % Example usage
  \ifpreprint
    \tcbset{colback=white,colframe=black!15}
  \else
    \tcbset{colback=white,colframe=black}
  \fi
  ```

---

## Final Sanity Checks (1-minute grep)
- [ ] `TBD` / `??` / `todo` / `FIXME` remain?  
  ```bash
  grep -nE 'TBD|\?\?|todo|FIXME' paper_v3.tex
  ```
- [ ] Legacy tags not replaced?  
  ```bash
  grep -nE '\\b(INF|NULL|Phi|REAL)\\b' paper_v3.tex
  ```
- [ ] Uncited figures/tables/algorithms?
  ```bash
  grep -nE '\\label\{' paper_v3.tex
  # Manually check each has a corresponding \ref or \autoref
  ```

---

### Notes
- These edits require **no data reruns** — only LaTeX and wording changes based on your existing results.
- Keep all numeric claims sourced from existing tables/figures; if a number isn’t already in the paper, paraphrase without inventing new values.
