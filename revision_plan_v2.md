# Major Revision Plan (v2)

## Paper: "Mechanistic Dissection of Learned Feedback-Contract Specificity in a 35-Neuron Network"

### (Original title: "Self-Correction Emerges from Recurrent Structure in a Minimal 35-Neuron Network: Ablation Evidence")

## Target Venue: Transactions on Machine Learning Research (TMLR)

---

## Executive Summary

The original manuscript demonstrates the existence and ablation-sensitivity of self-correction in a 35-neuron recurrent network. Critical review identifies four gaps that must be addressed for TMLR acceptance:

1. The mechanistic transparency promised by the minimal architecture is underexploited — the analyses remain behavioral (black-box ablation) despite having a glass-box model.
2. The fake mirror effect — the paper's most novel contribution — lacks graded evidence for its boundary conditions.
3. All findings are confined to a single architecture scale.
4. The static-input design makes key ablation results (Group A gain = 0) structurally tautological, undermining the causal claims.

This revised plan addresses all four gaps. Critically, it adds a **variable-noise task** (Workstream 0) that breaks the static-input tautology while preserving the clean separation of memory and self-correction — the paper's core experimental strength. It also incorporates key framing corrections: complete removal of "emergence" language, title change, and reframing of the feedforward baseline's superior absolute performance from a limitation into a substantive discussion point.

**Estimated total effort: 4 weeks**

---

## Critical Framing Changes

### Title Change

The word "emerges" must be completely eliminated. Setting w1 = 0.0 is an explicit training signal; the ML community does not consider induced behavior to be emergence.

**New title**: *Mechanistic Dissection of Learned Feedback-Contract Specificity in a 35-Neuron Network*

Alternative: *Mechanisms of Induced Self-Correction in a Minimal Recurrent Network: A Glass-Box Ablation Study*

### "Emergence" Purge

Every instance of "emergence," "emerges," or "emergent" in the manuscript must be replaced. Suggested substitutions:

| Original | Replacement |
|----------|-------------|
| "self-correction emerges" | "self-correction develops" or "self-correction is induced" |
| "emergence of self-correction" | "development of learned self-correction" |
| "emergent capability" | "induced capability" or "learned capability" |
| "emergence depends on conditions" | "induction requires specific training conditions" |

### D' Reframing: From Limitation to Discussion

The current manuscript treats the parameter-matched feedforward model's superior absolute accuracy (0.822 vs. 0.740) as Limitation 5. This should be promoted to a substantive discussion point:

*"Given an identical parameter budget, the feedforward architecture allocates all capacity toward single-pass accuracy — a 'one-shot optimization' strategy. The recurrent architecture accepts lower initial accuracy in exchange for iterative refinement through self-reference — a 'progressive refinement' strategy. These represent fundamentally different computational trade-offs, not a simple superiority relationship. Under capacity constraints where single-pass accuracy is insufficient, the recurrent strategy may prove essential."*

### Tone Adjustment

Reduce speculative connections to LLMs, Chain-of-Thought, and biological metacognition. The introduction and discussion should maintain a disciplined empirical tone: "We mechanistically dissect iterative refinement in a fully transparent minimal system." Broader implications may be noted briefly but should not frame the narrative.

### Limitation Consolidation

Compress from 11 items to 5:

1. Variable-noise task preserves memory-correction separation but does not fully replicate the entangled memory+correction demands of autoregressive systems.
2. Self-correction was induced by time-weighted loss, not spontaneously developed.
3. Effect size is modest; the mechanism's practical value under larger-scale capacity constraints remains untested.
4. Mechanistic analysis is specific to this architecture; transferability to other recurrent architectures is untested.
5. Statistical power at N=10 limits detection of subtle effect-size differences between conditions (e.g., C1 vs. C2).

Remaining items from the original 11 may be moved to a supplementary appendix.

---

## Workstream 0: Variable-Noise Task (Tautology Defense)

### Motivation

This is the single most critical addition to the revision. Under static input, removing recurrence guarantees gain = 0 — this is mathematical tautology, not an empirical finding. Every subsequent analysis (mechanistic, interpolation, scale) inherits this flaw if the static-input design is the only experimental setting.

The solution must break the tautology while preserving the paper's key design strength: the clean separation of memory and self-correction. Temporal blanking (input at t=1 only, zero at t=2,3) does not satisfy this requirement — it forces recurrence to simultaneously serve as memory and correction, making Group A ablation results ambiguous (performance drops could reflect memory loss, correction loss, or both).

### Design: Independent Noise Realizations Per Timestep

At each timestep t, the network receives the same class prototype corrupted by an independently sampled noise vector:

```
x_t = prototype_k + ε_t,  where ε_t ~ N(0, σ²I),  ε_1 ⊥ ε_2 ⊥ ε_3
```

This design has three critical properties:

1. **Breaks the tautology.** Because x_1 ≠ x_2 ≠ x_3, a feedforward-only network (Group A) can produce different outputs at each timestep. Gain ≠ 0 is no longer structurally guaranteed to be zero — if it is zero (or negative), that is an empirical finding, not a logical necessity.
2. **Preserves memory-correction separation.** The class identity is the same at every timestep; only the noise realization differs. The recurrent pathway carries information about the model's *own prior output*, not about past inputs (since each input is an independent draw from the same distribution). Memory of past inputs is neither necessary nor helpful — only self-correction through output feedback can systematically improve performance.
3. **Adds ecological validity.** Real-world systems often receive noisy, varying observations of the same underlying state. This setting is a natural extension of the static case.

### Implementation

- Modify the existing data generation pipeline to sample independent noise per timestep instead of reusing the same noisy input.
- Retrain all 10 models (seeds 0–9) with the same hyperparameters (w1=0.0, w2=0.2, τ=2.0).
- Retrain 10 clone donor models (seeds 100–109) for the C2 condition.
- Run the full ablation battery: Baseline, Group A, C1, C2, D', D''.
- If the primary hyperparameters do not produce self-correction under variable noise, conduct a targeted sweep of τ and w2.

### Expected Outcomes and Analysis

- **Baseline gain > 0**: Self-correction persists under variable noise (not dependent on static-input artifact).
- **Group A gain ≈ 0 (empirically, not tautologically)**: Removing recurrence eliminates correction even when the feedforward path could, in principle, produce varying outputs. This is now a genuine causal finding.
- **C1/C2 gain < 0**: The fake mirror effect replicates under non-static conditions.
- Report all results with the same statistical framework (exact Wilcoxon signed-rank, Holm-Bonferroni correction, bootstrap CIs).

### Presentation Strategy

Present the variable-noise results as the **primary experiment** in the revised manuscript, with the original static-input results repositioned as a controlled special case that enables cleaner mechanistic analysis. This inversion of emphasis directly addresses the tautology criticism.

**Estimated time: 4–5 days** (data pipeline modification, retraining, full ablation battery)

---

## Workstream 1: Mechanistic Analysis

### Motivation

The paper's central design choice — restricting the network to 35 neurons — is justified by the goal of mechanistic transparency. Yet the current analyses are entirely behavioral. This workstream transforms the paper from a behavioral ablation study into a mechanistic dissection, making the 35-neuron constraint the paper's primary methodological asset.

All mechanistic analyses in this workstream are performed on the **static-input** models, where the clean separation of memory and correction enables unambiguous interpretation. The variable-noise task (WS0) establishes that the phenomenon is not a static-input artifact; the mechanistic analysis explains *how* it works.

### 1.1 Activation Logging Infrastructure

**Goal**: Record the complete internal state of the network at every timestep for every trial.

**Specification**:
- Instrument the existing NumPy forward pass to record, per trial per timestep:
  - Raw feedback signal: y_{t-1} (5-dim)
  - Scaled feedback: tanh(y_{t-1} / τ) (5-dim)
  - Feedback contribution to hidden layer 1: tanh(y_{t-1} / τ) · W_rec (10-dim)
  - Feedforward contribution to hidden layer 1: x · W_ih1 + b_h1 (10-dim)
  - Pre-activation and post-activation at hidden layer 1 (10-dim each)
  - Pre-activation and post-activation at hidden layer 2 (10-dim each)
  - Output logits (5-dim)
  - Softmax probabilities (5-dim)
- Store as structured arrays indexed by (model_seed, trial_index, timestep).
- Run across all 10 trained models (seeds 0–9) on the full test set (200 samples per model).
- Implement with full test coverage (TDD): verify activation shapes, value ranges, and consistency with model predictions.

**Deliverable**: An `ActivationLogger` class producing a NumPy archive of activation traces.

**Estimated time**: 1–2 days

### 1.2 Trial-Level Trajectory Analysis

**Goal**: Visualize how the network's internal state evolves across timesteps, and how this evolution differs across trial outcomes and ablation conditions.

**Specification**:
- Classify all test trials into four categories:
  - **Corrected**: wrong at t=1, correct at t=3
  - **Stable correct**: correct at t=1 and t=3
  - **Stable incorrect**: wrong at t=1 and t=3
  - **Over-corrected**: correct at t=1, wrong at t=3
- For hidden layer 1 activations (10-dim), apply PCA fitted on the Baseline t=1 activations to extract the top 2–3 principal components. Plot the t=1 → t=2 → t=3 trajectory for each trial category as colored arrows in this shared PC space.
- Overlay trajectories from ablation conditions (Group A, C1, C2) in the same PC space:
  - Group A: stationary points (no movement)
  - C1: incoherent divergence
  - C2: coherent movement toward wrong attractors
- For each of the 5 classes, plot the class-conditional trajectory centroid to show whether correction moves the hidden state toward the correct class prototype.

**Key figure — the paper's Hero Image**: A 2D PC-space plot showing Baseline "corrected" trials converging toward class centroids, contrasted with C2 trajectories veering toward plausible but incorrect attractors. This single figure visually demonstrates that clone feedback is not generic OOD noise but a structured misdirection — the geometric proof of feedback-contract specificity.

**Estimated time**: 3–4 days

### 1.3 W_rec Dissection

**Goal**: Directly interpret the 50 learned recurrent weights and show how they implement correction.

**Specification**:
- Visualize W_rec as a 5×10 heatmap (output neuron → hidden layer 1 neuron), annotated with magnitude and sign. Repeat for all 10 models to assess structural consistency.
- Decompose the feedback contribution at each timestep:
  - Given output y_{t-1}, compute feedback vector f = tanh(y_{t-1} / τ)
  - Compute per-output-neuron contribution: for each output dimension k, compute f_k · W_rec[k, :] (10-dim vector)
  - On corrected trials, identify which output dimensions drive the hidden state in the corrective direction
- Targeted analysis: for trials where the network predicts class A but the true label is class B, trace how W_rec translates "high class-A logit" feedback into hidden activations that suppress class A and promote class B.
- Assess whether W_rec exhibits interpretable structure:
  - Anti-diagonal patterns (error-dampening: high logit for class k suppresses class k)
  - Cross-class promotion (uncertainty redistribution)
  - Distributed encoding (itself an informative finding — even 35 neurons can produce distributed representations)

**Deliverable**: A mechanistic narrative explaining *how* W_rec implements correction, with weight heatmaps and per-trial decomposition figures.

**Estimated time**: 3–4 days

### 1.4 Clone Feedback Geometric Analysis

**Goal**: Mechanistically explain why clone feedback (C2) is harmful, not just observe that it is.

**Specification**:
- For each trial, compute:
  - Self-feedback hidden contribution: tanh(y_self / τ) · W_rec (10-dim)
  - Clone-feedback hidden contribution: tanh(y_clone / τ) · W_rec (10-dim)
- Measure angular divergence (cosine distance) and magnitude ratio between these two vectors.
- Correlate divergence with correction outcome: high-divergence trials should show the most C2 degradation.
- Visualize high-divergence trials in the PC space from §1.2.

**Key finding to demonstrate**: Even when self and clone outputs agree on the predicted class (same argmax), their continuous logit vectors produce meaningfully different W_rec activations. This proves feedback-contract specificity operates at the continuous geometry level, not at the discrete class level — distinguishing it from standard distribution shift.

**Estimated time**: 2–3 days

---

## Workstream 2: Feedback Interpolation Experiment

### Motivation

The current paper presents a binary contrast: self-feedback works, everything else fails. The interpolation experiment maps the continuous boundary, providing graded evidence for the specificity of the learned feedback contract and bypassing the statistical power limitations of N=10 discrete comparisons.

### 2.1 Design

**Goal**: Map correction gain as a continuous function of feedback authenticity.

**Specification**:
- Interpolated feedback: feedback_t = α · y_self_{t-1} + (1 - α) · y_other_{t-1}
- Sweep α ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
- Three interpolation variants:
  - **Self–Clone**: y_other = clone model output (C2 donor, seeds 100–109)
  - **Self–Shuffled**: y_other = element-wise permuted self output
  - **Self–Zero**: y_other = zero vector (interpolation between full feedback and no feedback)
- Evaluate across all 10 model pairs on the full test set.
- Primary measure: correction gain. Secondary: per-timestep accuracy, hidden state divergence from pure-self trajectory.

### 2.2 Analysis

- Plot correction gain vs. α for all three interpolation types on shared axes.
- Key diagnostic: **threshold vs. linear degradation**.
  - If a sharp threshold exists (e.g., gain collapses below α ≈ 0.7), this strongly supports feedback-contract specificity — the network requires high-fidelity self-signal, not merely "some" feedback.
  - If degradation is linear in α, the contract is "soft" — still informative but a weaker claim.
- Compare degradation profiles: clone contamination vs. shuffled contamination may have different slopes, revealing whether structured misdirection (clone) is worse than unstructured noise (shuffled) at the continuous level.

### 2.3 Connection to Mechanistic Analysis

- At the threshold α (if one exists), examine hidden-state trajectories in the PC space from WS1.2 to identify where geometric divergence becomes critical.
- Compute cosine similarity between interpolated feedback contribution (interpolated_f · W_rec) and pure self-feedback contribution as a function of α.

**Key figure — Phase Diagram**: Correction gain (y-axis) vs. α (x-axis), three curves (clone, shuffled, zero), annotated with the threshold region if present.

**Estimated time**: 3–4 days

---

## Workstream 3: Scale Verification

### Motivation

A single-scale finding invites the criticism that it is an artifact of the specific architecture. Scale verification is lower priority than WS0–WS2 but strengthens the paper if time permits.

### Design: Fixed Task, Varying Width (Recommended)

Hold the task fixed (5-class, 10-dim input) and vary only the hidden layer width. This isolates model capacity from task complexity, avoiding confounds introduced by scaling the number of classes.

| Label | Architecture | Neurons | Parameters |
|-------|-------------|---------|------------|
| S-35 (original) | Input(10) → H1(10) → H2(10) → Output(5) | 35 | 325 |
| S-55 | Input(10) → H1(20) → H2(20) → Output(5) | 55 | 825 |
| S-105 | Input(10) → H1(45) → H2(45) → Output(5) | 105 | 2,700 |
| S-505 | Input(10) → H1(245) → H2(245) → Output(5) | 505 | 63,750 |

For each scale:
- Train 10 models (seeds 0–9) with the same hyperparameters. If emergence fails at larger scales, conduct a targeted sweep of τ and lr.
- Train 10 clone donor models (seeds 100–109).
- Run core ablation conditions: Baseline, Group A, C1, C2.
- Use both static-input and variable-noise settings.

### Analysis

- Report correction gain for Baseline, A, C1, C2 at each scale.
- Key question: Does the fake mirror effect (C1/C2 gain < 0) persist across scales?
- Secondary: Does correction gain magnitude change with capacity?

**Key figure**: Multi-panel plot showing qualitative pattern (Baseline > 0, A ≈ 0, C1 < 0, C2 < 0) across scales.

**Estimated time**: 4–5 days

---

## Revised Paper Structure

1. **Introduction**: Reframe around "minimal sufficient conditions and mechanisms for learned iterative refinement." Remove speculative LLM/metacognition framing. Disciplined empirical tone.

2. **Methods**: Add subsections for variable-noise task, activation logging, trajectory analysis, interpolation protocol, and scale verification.

3. **Results — Variable-Noise Task** (new, primary): Present the variable-noise ablation results as the main experiment. This establishes that self-correction and the fake mirror effect are not static-input artifacts.

4. **Results — Behavioral Ablation** (current §3, condensed): Retain the original static-input ablation results as a controlled special case. Present more concisely — this is now setup for the mechanistic analysis, not the main event.

5. **Results — Mechanistic Dissection** (new, centerpiece): Trajectory analysis, W_rec dissection, clone feedback geometric analysis. This is the paper's primary contribution.

6. **Results — Interpolation**: Feedback interpolation curves and phase diagram. Connect to mechanistic analysis.

7. **Results — Scale Verification**: Cross-scale replication of qualitative pattern.

8. **Discussion**: Reorganize around mechanistic findings. Feedback-contract specificity grounded in geometric evidence. D' reframing as computational trade-off. Consolidated limitations (5 items). Brief, cautious notes on broader implications.

---

## Figures

| # | Content | Source | Role |
|---|---------|--------|------|
| 1 | Variable-noise ablation results (Baseline, A, C1, C2 gain) | WS0 | Tautology defense |
| 2 | Hidden-state trajectory PC plot: corrected vs. over-corrected trials, with A/C1/C2 overlays | WS1.2 | **Hero Image** — geometric proof of feedback-contract specificity |
| 3 | W_rec heatmap (5×10), consistency across 10 seeds | WS1.3 | Mechanistic transparency |
| 4 | Per-trial correction decomposition through W_rec | WS1.3 | Mechanistic narrative |
| 5 | Angular divergence: self vs. clone feedback contributions, correlated with degradation | WS1.4 | Clone feedback mechanism |
| 6 | Interpolation phase diagram: gain vs. α for clone, shuffled, zero | WS2.2 | Graded evidence |
| 7 | Scale replication: gain across network sizes for Baseline, A, C1, C2 | WS3 | Generalizability |

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | **WS0** + WS1.1 | Variable-noise task implementation and training; activation logging infrastructure with tests |
| 2 | **WS1.2 + WS1.3** | Trial classification and trajectory visualization; W_rec dissection and mechanistic narrative |
| 3 | **WS1.4 + WS2** + WS3 | Clone geometric analysis; interpolation experiment and phase diagram; scale verification (parallel) |
| 4 | **Paper rewrite** | Title change; "emergence" purge; restructured results; new figures; D' reframing; limitation consolidation |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Variable-noise task does not produce self-correction with current hyperparameters | Medium | High | Targeted sweep of τ (1.0–5.0) and w2 (0.1–0.5); if no configuration works, report as informative negative result and retain static-input as primary |
| W_rec structure is distributed and resists simple interpretation | Medium | Medium | Report as finding ("even 35 neurons produce distributed representations"); trajectory analysis provides interpretable results regardless |
| Interpolation shows no sharp threshold | Low | Low | Linear degradation is still informative and publishable; soft vs. hard contract is a meaningful distinction |
| Fake mirror effect does not replicate at larger scales | Low–Medium | Medium | Report scale boundaries honestly; the 35-neuron mechanistic analysis retains value |
| Hyperparameters require re-tuning at larger scales | Medium | Low | Use fixed-task varying-width design to minimize confounds; targeted sweep if needed |

---

## Changes from Plan v1

| Item | v1 | v2 | Rationale |
|------|----|----|-----------|
| Title | Original title retained | Changed to "Mechanistic Dissection..." | "Emerges" in title is a red flag for TMLR reviewers |
| "Emergence" language | "Reframe" | Complete purge with substitution table | Half-measures insufficient; every instance must be replaced |
| Non-static task | Listed as optional (noted 60–70% without, 85%+ with) | Promoted to WS0, first priority | Tautology critique is structurally fatal; must be addressed before all other workstreams |
| Task design | Temporal blanking suggested by external review | Variable-noise (independent noise per timestep) | Temporal blanking conflates memory and correction; variable noise breaks tautology while preserving the paper's core separation |
| D' performance | Limitation 5 | Promoted to Discussion as computational trade-off | "One-shot optimization vs. progressive refinement" is a substantive insight, not a weakness |
| Limitations | 11 items | 5 items (rest to appendix) | Reduce defensive impression; retain intellectual honesty |
| Scale verification | Both 3.1 and 3.2 presented equally | 3.2 (fixed-task, varying-width) recommended as primary | Scaling task complexity introduces confounds |
| Speculative framing | "Reduce" | Explicitly minimize CoT, biological metacognition connections | Evidence level does not support these links as framing devices |
| WS1.2 trajectory plot | One of several figures | Designated as Hero Image | This single figure can visually prove feedback-contract specificity is not standard distribution shift |

---

## Expected Outcome

With all four workstreams complete, the paper transforms from a behavioral ablation study into a **mechanistic dissection of self-correction at minimal scale, with non-tautological causal evidence, graded interpolation data, and cross-scale validation**. The variable-noise task neutralizes the most dangerous reviewer objection; the mechanistic analysis justifies the 35-neuron design choice; the interpolation maps the continuous boundary of feedback specificity; and the scale check demonstrates robustness.

**Estimated acceptance probability: 75–85%**

This estimate reflects the strengthened evidence base while acknowledging residual risk: the variable-noise task may require hyperparameter re-tuning, the mechanistic narrative depends on W_rec exhibiting at least partially interpretable structure, and reviewer assignment introduces irreducible variance.
