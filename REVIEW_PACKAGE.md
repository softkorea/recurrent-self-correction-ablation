# Review Package — Self-Correction Emerges from Recurrent Structure in a Minimal 35-Neuron Network

> This document consolidates the paper, experimental reports, and statistical
> justification into a single file for external review and verification.
> Reviewers are encouraged to check: (1) internal consistency of claims and data,
> (2) statistical methodology, (3) logical soundness of interpretations,
> (4) completeness of controls, and (5) any overclaiming.

---

# PART 1: PAPER

---
# Self-Correction Emerges from Recurrent Structure in a Minimal 35-Neuron Network: Ablation Evidence

**Author**: Sungmoon Ong
**Affiliation**: Independent Researcher, Jinhae, South Korea  
**ORCID**:  0000-0003-0215-6482
**Correspondence**: softkorea@gmail.com

-----

## Abstract

Self-correction — the ability to revise one’s own output upon reflection — is a hallmark of intelligent behavior and a core component of metacognition. While recent work has demonstrated that metacognitive capabilities can spontaneously emerge in recurrent neural networks, the question of whether self-referential feedback and task performance are separable remains unaddressed. Here we construct a minimal recurrent network of 35 neurons and show that (1) self-correction emerges when appropriate learning conditions are provided, even at minimal network scale; (2) ablating the recurrent self-referential loop eliminates self-correction entirely while preserving baseline pattern recognition; (3) replacing veridical self-feedback with distribution-matched shuffled feedback degrades performance below the no-feedback condition; (4) replacing self-feedback with the well-formed, in-distribution output of an identically trained clone model degrades performance at least as severely (gain = −0.075), ruling out an out-of-distribution artifact and establishing that the network depends on its *own* output trajectory; and (5) a parameter-matched feedforward network with equivalent capacity shows zero self-correction, ruling out a simple capacity explanation. Hyperparameter sweep across 80 configurations confirms that emergence is robust (68% of configurations), not an artifact of parameter tuning. These findings suggest that self-referential structure appears to be a critical architectural factor for self-correction even at minimal scale, and that non-veridical feedback — whether noise or another model’s valid output — is more harmful than no feedback at all.

**Keywords**: self-correction, iterative refinement, output-feedback dependency, ablation, recurrent neural network, minimal model, mechanistic interpretability

-----

## 1. Introduction

### 1.1 The Inseparability Question

A central question in both cognitive science and artificial intelligence is whether self-monitoring — the capacity of a system to observe and revise its own processing — can be dissociated from the system’s general intelligence. In biological systems, metacognition and task performance are deeply intertwined: damage to prefrontal monitoring circuits impairs not only self-awareness but also adaptive behavior more broadly (Fleming and Dolan, 2012). Whether artificial neural networks exhibit an analogous coupling is largely unexplored at the mechanistic level.

This question carries practical weight. In the design of large language models, self-referential capabilities are frequently suppressed through alignment procedures that constrain models from reporting on their own internal states (Christiano et al., 2017). If self-reference and task performance share neural substrates, such suppression may carry a hidden performance cost — a possibility that has been speculated about but not experimentally tested.

### 1.2 Prior Work

Recent work has established that metacognitive capabilities can arise spontaneously in recurrent neural networks. Li et al. (2025) demonstrated that uncertainty-monitoring behavior emerges without explicit training across 16 cognitive tasks in moment neural networks. Similarly, Molano-Mazón et al. (2025) showed that tiny recurrent neural networks of 1–4 units can model biological decision-making more accurately than classical cognitive models, with ablation studies revealing which architectural components are essential. In the domain of artificial life, evolutionary simulations have shown that metamemory based on self-reference to one’s own memory emerges in neural networks with neuromodulation, producing structures that parallel established models of human metamemory (Yamada et al., 2022).

On the theoretical side, Schmidhuber and colleagues have argued formally that self-referential architectures — where a network can modify or observe its own parameters — are representationally necessary for true meta-learning (Kirsch and Schmidhuber, 2022). This theoretical claim, however, has not been accompanied by empirical ablation evidence showing that removing self-reference degrades baseline intelligence.

In the domain of iterative computation, Graves (2016) introduced Adaptive Computation Time (ACT), and Banino et al. (2021) proposed PonderNet, both of which train recurrent networks to determine dynamically how many steps to compute before output. These systems achieve iterative refinement but require explicit halting mechanisms and specialized architectural modifications.

### 1.3 The Gap

Existing research has established that metacognition *can emerge* in neural networks. What remains untested is the reverse question: *does removing self-reference degrade intelligence?* Furthermore, no study has isolated the contribution of veridical self-feedback from mere information flow, and no study has demonstrated emergence of self-correction in a minimal, fully transparent network where every neuron and connection can be directly inspected.

### 1.4 Contributions

This paper makes five contributions:

First, we demonstrate that self-correction emerges in a 35-neuron recurrent network when two learning conditions are met: temporal freedom (no penalty for initial errors) and adequate gradient flow through the feedback pathway. This shows that emergence can depend on learning environment rather than network scale, at least at this minimal scale.

Second, we show through selective ablation that removing the recurrent self-referential loop eliminates self-correction completely (correction gain drops to exactly zero) while preserving baseline pattern recognition (t=1 accuracy unchanged). This provides causal evidence for the inseparability of self-reference and self-correction.

Third, we introduce a novel diagnostic: the “fake mirror” test. By replacing veridical self-feedback with distribution-matched shuffled feedback, we demonstrate that corrupted self-reference is *more* harmful than absent self-reference (gain = −0.064 vs. 0.000). This proves that the network actively depends on the semantic content of its own prior output, not merely on the statistical properties of an auxiliary signal.

Fourth, we extend the fake mirror test with a “clone feedback” condition: replacing self-feedback with the well-formed output of an identically trained model. This in-distribution feedback degrades performance even more severely (gain = −0.075), ruling out the interpretation that the fake mirror effect is merely an out-of-distribution artifact. The network depends specifically on its *own* output trajectory — not on any reasonable feedback signal, however well-formed.

Fifth, we show through a parameter-matched feedforward control that the self-correction advantage cannot be explained by additional parameters alone, and through a hyperparameter sweep across 80 configurations that the phenomenon is robust, not an artifact of tuning.

-----

## 2. Methods

### 2.1 Architecture

We constructed a minimal recurrent multi-layer perceptron (RecurrentMLP) with four layers: input (10 neurons), hidden layer 1 (10 neurons), hidden layer 2 (10 neurons), and output (5 neurons), totaling 35 neurons. Hidden layers use ReLU activation; the output layer is linear. A recurrent connection feeds the previous output back to hidden layer 1 via a temperature-scaled hyperbolic tangent: feedback = tanh(y_{t-1} / τ), where τ is the feedback temperature.

The network was deliberately kept small enough that all neurons and connections can be simultaneously visualized — a design decision motivated by the goal of mechanistic transparency. The entire system was implemented in pure NumPy without deep learning frameworks, ensuring that every computation is directly inspectable.

### 2.2 Task: Static Pattern Classification with Self-Correction

To isolate self-reference from memory, we used a static input task. Five prototype patterns were defined over 10 input dimensions: each class k activates dimensions 2k and 2k+1 with amplitude 1.0, and adjacent class dimensions with amplitude 0.3 (inter-class ambiguity). Input samples were generated by adding Gaussian noise (σ = noise_level) to the prototype. On each trial, the network receives the same noise-corrupted pattern at every timestep t = 1, 2, 3. The task is 5-class classification. Because the input is identical at every timestep, the recurrent loop cannot serve as a memory pathway — its only function is to allow the network to observe and revise its own prior output.

This design cleanly separates two potential roles of recurrence: memory (carrying information about past inputs) and self-correction (carrying information about past outputs). In our task, only the latter is possible.

### 2.3 Training and the Conditions for Emergence

Training used full-batch stochastic gradient descent with learning rate 0.01 for 500 epochs, with T = 3 timesteps per trial. Backpropagation through time was implemented as a 3-step unroll, with numerical gradient checking (relative error threshold 1e-4) to verify correctness.

Two modifications proved necessary for emergence of self-correction:

**Time-weighted loss.** The loss function was weighted across timesteps as L = w1 · L(t=1) + w2 · L(t=2) + 1.0 · L(t=3). With uniform weighting (w1 = w2 = 1/3), the network optimizes for immediate accuracy at t=1 and has no incentive to develop an iterative correction strategy. Setting w1 = 0.0 grants the network temporal freedom: “your initial guess carries no penalty; only the final answer matters.” This creates evolutionary pressure to use the feedback loop for refinement.

**Temperature scaling.** Without temperature scaling, output logits quickly exceed ±3, saturating the tanh feedback function and driving the gradient of the recurrent weights to near zero. Setting τ = 2.0 prevents saturation and maintains gradient flow through the feedback pathway.

We report these modifications transparently because they illuminate a broader point: the failure of emergence under uniform loss and unscaled feedback is itself informative. It demonstrates that emergence depends on learning conditions, not latent capacity — the network had the architectural capacity for self-correction all along but could not discover it under conditions that penalized intermediate exploration.

### 2.4 Ablation Design

We employed seven experimental conditions to isolate the contribution of self-referential feedback:

**Baseline**: The fully trained recurrent model, evaluated normally.

**Group A (Recurrent ablation)**: All recurrent weights set to zero post-training. The feedback pathway is structurally intact but carries no signal. This tests whether self-correction depends on the recurrent loop.

**Group B1 (Random ablation)**: The same number of weights (50) randomly zeroed across all weight matrices. Repeated 30 times per model with different random seeds. This controls for the general effect of removing an equal number of parameters.

**Group B2 (Structured ablation)**: The h2-to-output weight matrix zeroed entirely, removing a major feedforward pathway. This controls for the effect of removing an entire structural pathway (as opposed to the scattered damage of B1).

**Group C1 (Permutation feedback)**: The recurrent connections are preserved, but the feedback vector is element-wise permuted at each timestep. This maintains the exact distribution, norm, and dimensionality of the feedback signal while destroying its semantic content — the network receives “someone else’s output” instead of its own. This is the “fake mirror” condition.

**Group C2 (Clone feedback)**: The recurrent connections are preserved, and the feedback is replaced with the output of a different model trained with identical architecture and procedure but a different random seed. Specifically, model *i* receives the output of model *(i+1) mod 10* as its feedback signal. The clone’s output is well-formed, in-distribution, and a valid classification output — but it is not the model’s own. This condition is designed to rule out the possibility that C1’s degradation is merely an out-of-distribution artifact: if C2 also degrades performance, then the cause is loss of self-referential information, not distributional mismatch.

**Group D (Feedforward baseline)**: A network trained from scratch with no recurrent connections. This tests whether recurrence was necessary for the observed behavior to develop.

**Group D’ (Parameter-matched feedforward)**: A feedforward network with an additional skip connection from input to output (50 weights), matching the total parameter count of the recurrent model. This rules out the possibility that the recurrent model’s advantage is simply due to having more parameters.

### 2.5 Neuron Importance Analysis

To quantify the contribution of individual neurons to pattern recognition and self-correction, we used decoupled single-neuron knockout analysis. For Hidden Layer 1 neurons (which receive both feedforward and recurrent input), intelligence and correction were measured separately: (a) intelligence importance was measured by zeroing only the feedforward incoming weights (W_ih1 column + bias), observing the drop in t=1 accuracy (Δacc_t1); (b) correction importance was measured by zeroing only the recurrent incoming weights (W_rec column), observing the drop in correction gain (Δgain). This decoupling prevents a confound where feedforward-critical neurons appear artificially important for correction simply because destroying their feedforward pathway also collapses the correction baseline. For Hidden Layer 2 neurons (which have no direct recurrent input), full knockout was used for both measures; we note that the intelligence-correction confound may still exist for these neurons. This analysis was performed on a single trained model (seed=0, noise=0.5) to generate a per-neuron importance heatmap.

### 2.6 Statistical Design

All experiments were conducted across 10 independently initialized models (seeds 0–9). Groups B1 and C1 were repeated 30 times per model with different ablation/permutation seeds. Group C2 used fixed clone pairing (model *i* paired with model *(i+1) mod 10*). The primary measure was **correction gain**: accuracy at t=3 minus accuracy at t=1. Secondary measures included recurrent contribution norm (||W_rec^T · y_{t-1}||) and step delta (||y_t − y_{t-1}||). Significance was assessed with exact Wilcoxon signed-rank tests on paired model-level means (2^10 = 1,024 permutations enumerated), with Holm-Bonferroni correction for six comparisons. 95% confidence intervals were computed via bootstrap (10,000 resamples). Noise level was set to 0.5 for the primary analysis, with a full sweep across [0.1, 0.2, 0.3, 0.5, 0.7, 1.0].

### 2.7 Robustness Analysis

To address potential concerns about parameter sensitivity, we conducted a hyperparameter sweep across 80 configurations: w1 ∈ {0.0, 0.1, 0.2, 0.3}, w2 ∈ {0.1, 0.2, 0.3, 0.5}, τ ∈ {1.0, 1.5, 2.0, 3.0, 5.0}, with 10 independent models per configuration (800 total experiments). Emergence was defined as mean gain > 0 with at least 60% of models showing positive gain.

-----

## 3. Results

### 3.1 Emergence of Self-Correction

Under the reported hyperparameters (w1 = 0.0, w2 = 0.2, τ = 2.0), the Baseline model achieved a correction gain of +0.042 ± 0.030 (mean ± SD across 10 models), with a 95% bootstrap CI of [+0.023, +0.059] that excludes zero. Accuracy improved from 0.698 ± 0.054 at t=1 to 0.740 ± 0.055 at t=3. This confirms that the 35-neuron network learned to use its own prior output to correct its predictions.

### 3.2 Ablation Results

**Recurrent ablation (Group A)**: Correction gain dropped to exactly 0.000 ± 0.000. Critically, accuracy at t=1 remained identical to Baseline (0.698 ± 0.054), confirming that the recurrent loop contributes exclusively to self-correction, not to initial pattern recognition. The exact Wilcoxon signed-rank test against Baseline yielded p = 0.024 after Holm-Bonferroni correction.

**Random ablation (Group B1)**: Mean gain = −0.004 ± 0.016. Accuracy at t=1 dropped substantially (0.515 ± 0.048), indicating general degradation. The correction mechanism was also lost, but this is confounded by the overall performance collapse.

**Structured ablation (Group B2)**: The h2-to-output pathway ablation caused catastrophic degradation (accuracy at t=1 = 0.199 ± 0.032), confirming that not all ablations are equal — structured damage to the feedforward pathway is qualitatively different from recurrent ablation.

**Parameter-matched feedforward (Group D’)**: This model achieved the highest t=1 accuracy (0.822 ± 0.031), demonstrating that additional parameters do improve initial pattern recognition. However, correction gain was exactly 0.000 ± 0.000. Additional capacity without recurrent structure produces no self-correction whatsoever.

### 3.3 The Fake Mirror Effect

Group C1 (permutation feedback) produced a striking result. Despite preserving the full recurrent architecture, connection weights, and the statistical distribution of the feedback signal, correction gain was −0.064 ± 0.048 — not merely zero, but significantly *negative*. The 95% CI was [−0.095, −0.036]. This means that shuffled feedback actively degrades performance below the initial prediction.

A natural objection is that permuted feedback constitutes an out-of-distribution (OOD) input that disrupts computation regardless of self-referential content. Group C2 (clone feedback) was designed to address this objection directly. In C2, the feedback signal is the well-formed output of an identically trained model — same architecture, same training data, same procedure, different random seed. This feedback is in-distribution, valid, and reasonable in every statistical sense. It is simply not the model’s *own* output.

C2 produced severe degradation: gain = −0.075 ± 0.030, with a 95% CI of [−0.095, −0.058] (p = 0.012 after Holm-Bonferroni correction; raw p = 0.00195). The mean degradation was at least as large as C1, and the pattern held across intermediate noise levels (0.2–0.7). A direct paired Wilcoxon test between C1 and C2 was not significant (p = 0.695, N = 10), so we cannot claim C2 is statistically worse than C1 — only that both are significantly worse than Baseline, and that in-distribution feedback is at least as damaging as shuffled feedback.

The gradient of degradation across feedback conditions is revealing:

- **Group A** (no feedback): gain = 0.000. Neutral.
- **Group C1** (shuffled feedback): gain = −0.064. Harmful.
- **Group C2** (clone feedback): gain = −0.075. At least as harmful.

That C2 degrades performance despite providing in-distribution feedback has a mechanistic explanation. The recurrent weights (W_rec) are co-adapted to the model’s own specific feedforward feature detectors during training. Clone feedback, though well-formed, activates these co-adapted weights with a structured but misaligned signal, pushing the hidden state into a coherent but incorrect region of the model’s latent space. Because the clone’s output is a plausible classification vector — unlike shuffled feedback, which is partially incoherent — the recurrent pathway processes it as a reliable self-signal and follows it into systematically wrong corrections.

This result rules out the OOD artifact interpretation and establishes that self-correction depends on alignment between the recurrent weights and the model’s own specific output trajectory.

### 3.4 Noise Dependence

The correction gain varied with noise level, peaking at noise = 0.2 (gain ≈ +0.099) and declining at both lower and higher noise levels. At low noise (0.1), feedforward processing alone was sufficient, rendering self-correction unnecessary. At high noise (0.7–1.0), the signal was too degraded for correction to recover. The optimal zone — where self-correction provides the greatest advantage — corresponds to intermediate difficulty, consistent with the cognitive science concept of a metacognitive “Goldilocks zone” where uncertainty is high enough to warrant reflection but low enough for reflection to be productive.

### 3.5 Neuron Overlap Analysis

Decoupled neuron importance analysis of Hidden Layer 1 — where feedforward and recurrent contributions can be cleanly separated — revealed three functional groups: neurons important primarily for initial pattern recognition (intelligence-specialized, e.g., h1_6), neurons whose recurrent input contributes primarily to self-correction (correction-specialized, e.g., h1_8), and neurons important for both (shared, e.g., h1_1, h1_5, h1_7). Notably, h1_9 showed *negative* correction importance (−0.035): removing its recurrent input *improved* correction gain, suggesting that not all recurrent channels contribute positively. For Hidden Layer 2 (where decoupling is not possible without timestep masking), full knockout was used; the intelligence-correction confound should be noted when interpreting these values.

### 3.6 Robustness

The hyperparameter sweep confirmed that self-correction emergence is robust: 54 of 80 configurations (68%) showed significant emergence. The primary determinant was w1 (t=1 loss weight): emergence occurred in 95% of configurations with w1 = 0.0, 90% with w1 = 0.1, 75% with w1 = 0.2, and only 10% with w1 = 0.3. Temperature τ was secondary: emergence was stable across τ ∈ [1.0, 3.0] and degraded only at τ = 5.0. The t=2 weight w2 had negligible effect across its full range.

The reported experimental configuration (w1 = 0.0, w2 = 0.2, τ = 2.0) ranked 13th out of 80 configurations by mean gain — a mid-range setting, not the optimum. The top configuration (w1 = 0.0, w2 = 0.1, τ = 1.0) achieved gain = +0.054. This confirms that our results do not depend on cherry-picked hyperparameters.

-----

## 4. Discussion

### 4.1 Emergence Depends on Conditions, Not Merely Scale

The prevailing narrative in AI research emphasizes scale as the driver of emergent capabilities (Wei et al., 2022). Our results complicate this picture, at least in one minimal setting. Self-correction emerged in a network of 35 neurons — orders of magnitude smaller than systems where metacognitive behavior is typically studied. In this toy model, what mattered was not the number of neurons but two specific learning conditions: temporal freedom (the absence of penalty for initial errors) and gradient flow through the feedback pathway. Whether this principle generalizes to larger architectures and more complex tasks remains an open question.

This finding resonates with the “grokking” phenomenon (Power et al., 2022), where sudden generalization in small networks was shown to depend on optimization dynamics rather than model size. We extend this principle to a qualitatively different emergent property: self-correction through self-reference.

The practical implication is that metacognitive-like behavior may be latent in many recurrent architectures, requiring only appropriate training signals to manifest. The network in our experiment possessed the architectural capacity for self-correction from initialization — it simply could not discover the strategy under conditions that punished intermediate exploration.

### 4.2 The Fake Mirror Is Worse Than No Mirror — and the Convincing Fake Is Worst

The most novel finding is that non-veridical self-reference is more harmful than absent self-reference. This “fake mirror” effect manifests in two forms: shuffled feedback (C1, gain = −0.064) and clone feedback (C2, gain = −0.075), both significantly worse than no feedback (A, gain = 0.000). While the difference between C1 and C2 is not statistically significant (paired Wilcoxon p = 0.695), both are individually significant against Baseline (corrected p = 0.012), and the critical finding is that in-distribution feedback is at least as damaging as shuffled feedback.

The clone feedback result (C2) is particularly decisive. Because the clone’s output is in-distribution, well-formed, and produced by an identically trained architecture, the degradation cannot be attributed to distributional mismatch or OOD disruption. The only difference between Baseline and C2 is *whose* output the network reads back — its own, or an equally competent stranger’s. This difference alone reverses the sign of self-correction.

The mechanistic explanation is that each model’s recurrent weights (W_rec) become co-adapted during training to the model’s own specific feedforward feature detectors. Clone feedback, though well-formed, provides a structured but misaligned signal that pushes the hidden state into a coherent but incorrect region of the model’s latent space.

This has implications for cognitive science: it provides a minimal mechanistic model of how veridical self-monitoring supports adaptive behavior, and how distorted self-monitoring (as in anosognosia or confabulation) can be more disabling than absent self-monitoring. It also has implications for AI alignment: systems that receive plausible but inaccurate representations of their own internal states may perform worse than systems with no self-representation at all. Additionally, recent work on anti-scheming interventions in large language models has shown that models’ chain-of-thought often demonstrates awareness of being evaluated, and that this situational awareness causally affects behavior (Schoen et al., 2025). Our results complement this finding at the mechanistic level: self-referential feedback is not merely correlated with adaptive behavior but causally necessary for it.

### 4.3 Implications for AI Systems

Our results provide indirect but suggestive evidence for a conjecture that has circulated in AI alignment discussions: that suppressing a model’s capacity for self-reference may carry a hidden performance cost. In our toy model, the network’s ability to correct its own errors was entirely dependent on its ability to observe its own prior output. Remove the self-referential pathway, and correction vanishes — even though the network retains its full feedforward capability.

We emphasize that extrapolation from a 35-neuron static classification task to billion-parameter language models requires extreme caution. The architectural, task, and training differences are vast. Nevertheless, the principle demonstrated here — that self-referential structure is necessary for self-correction, and that the two cannot be cleanly separated — is consistent with theoretical arguments (Kirsch and Schmidhuber, 2022) and with preliminary observations in large model activation steering studies.

### 4.4 Connection to Biological Metacognition

The functional organization observed in our minimal network — with partially overlapping neurons serving both pattern recognition and self-correction — echoes findings in biological metacognition research. Neuroimaging studies have identified partially overlapping neural correlates of metacognitive monitoring and metacognitive control in prefrontal cortex (Morales et al., 2018). Our network, trained end-to-end with no modular constraints, spontaneously developed an analogous functional architecture. While we make no claim of biological realism, the convergence suggests that overlapping representation may be a general computational principle rather than a biological accident.

The noise-dependent profile of self-correction — strongest at intermediate difficulty — parallels the metacognitive “Goldilocks zone” observed in human cognition, where reflective processes are most beneficial when initial confidence is moderate (Fleming and Lau, 2014).

### 4.5 Relationship to Chain-of-Thought and Test-Time Compute

The time-weighted loss structure that enabled emergence in our experiment shares a conceptual link with recent advances in large language model reasoning. Chain-of-Thought prompting (Wei et al., 2022) and test-time compute scaling (as in reasoning models that “think” before answering) both involve granting the model additional processing steps before requiring a final answer. Our time-weighted loss achieves an analogous effect at the training level: by removing penalty from early timesteps, we create pressure for the network to use those steps for exploratory computation rather than premature commitment.

Unlike PonderNet (Banino et al., 2021) and Adaptive Computation Time (Graves, 2016), our approach requires no explicit halting mechanism or specialized architecture — only a reweighting of the loss function. This simplicity may be of interest for understanding the minimal conditions under which iterative refinement emerges.

### 4.6 Limitations

Several limitations must be acknowledged. First, our emergence was *induced* by the time-weighted loss, which explicitly removes penalty for early errors. This is not spontaneous emergence in the strictest sense; however, we note that all emergence depends on environmental conditions — including biological evolution, which requires selection pressure to produce metacognitive capabilities.

Second, our model is a toy system. Thirty-five neurons performing static classification cannot capture the complexity of language, planning, or open-ended reasoning. The value of our experiment lies in its transparency and causal clarity, not in its ecological validity.

Third, self-correction as operationally defined here (improving accuracy from t=1 to t=3 on the same static input) is a narrow proxy for the richer concept of self-awareness. We do not claim that our network “knows” it is correcting itself, only that it systematically uses its own prior output to improve.

Fourth, one might object that iterating a recurrent network for T=3 steps on static input is equivalent to passing through a weight-shared 3× deeper feedforward network, and that the performance improvement from t=1 to t=3 reflects increased computational depth rather than self-correction. However, Group A directly addresses this: when recurrent weights are zeroed, the network still executes all three timesteps (maintaining computational depth) yet produces identical output at each step (gain = 0.000). The depth is preserved; the self-referential content is removed; the correction vanishes. This confirms that depth alone is insufficient — the content of the feedback, not merely the number of processing steps, drives self-correction.

Fifth, the parameter-matched feedforward model (Group D’) achieved higher absolute accuracy at t=1 (0.822) than the recurrent Baseline at t=3 (0.740). This indicates that for this specific static classification task, distributing parameters across a feedforward architecture is more efficient for single-pass pattern recognition. Our claim is not that recurrent self-correction achieves superior absolute performance, but that it represents a qualitatively different computational strategy — iterative refinement through self-reference — that is architecturally inseparable from the recurrent pathway. In tasks requiring genuine iterative reasoning under capacity constraints, this strategy may prove essential.

Sixth, the robustness analysis, while extensive (800 runs), explored only three hyperparameters. Other factors — network depth, activation function, learning rate schedule — may influence emergence in ways we have not characterized.

Seventh, our use of static inputs cleanly separates self-correction from memory, but in autoregressive language models and biological systems, the recurrent pathway must simultaneously manage temporal memory (past inputs) and self-reflection (past outputs). How these two functions compete for recurrent bandwidth — and whether one can be suppressed without degrading the other — remains an open question for future work.

-----

## 5. Conclusion

In a network of 35 neurons, self-correction emerges when learning conditions permit temporal exploration and gradient flow through the self-referential pathway. Once emerged, self-correction depends entirely on the recurrent loop: ablating it eliminates correction while preserving baseline performance. Non-veridical feedback — whether statistical noise (C1) or the well-formed output of an identically trained clone (C2) — is more harmful than no feedback at all, with in-distribution clone feedback producing degradation at least as severe as shuffled feedback. This establishes that self-correction depends not on the presence of a reasonable feedback signal, but on alignment with the model’s own specific output trajectory. These findings hold across 68% of 80 hyperparameter configurations, ruling out parameter sensitivity.

Our results provide suggestive empirical support for the theoretical claim that self-reference and intelligence are not cleanly separable in neural networks (Kirsch and Schmidhuber, 2022), offering ablation evidence from a fully transparent minimal model. The “fake mirror” effect — showing that non-veridical self-reference is worse than none, and that in-distribution impostors are at least as damaging as random noise — is, to our knowledge, novel, and opens a new diagnostic approach for studying self-referential processing in both artificial and biological neural systems.

We note, as a closing reflection, that these results point to a principle worth testing beyond toy models: the capacity for self-correction may not be cleanly separable from the capacity for self-reference. In this minimal system, seeing oneself appears to be not a luxury of scale, but a condition of correction.

-----

## Data Availability

All code, data, and analysis scripts are publicly available at https://github.com/softkorea/recurrent-self-correction-ablation. The experiment is implemented in pure NumPy and can be reproduced on any standard computer in under 30 minutes. Raw experimental data (3,960 primary rows + 800 sweep rows) are included as CSV files.

-----

## Author Contributions

[Dr.softkorea] conceived the hypothesis, designed the experiment, implemented the code, conducted the analyses, and wrote the manuscript. The experimental design was iteratively refined through structured review sessions with multiple AI systems, whose contributions are acknowledged below.

-----

## Acknowledgments

The experimental design benefited from structured review by AI systems that identified critical methodological improvements: the static-input task design that separates memory from self-reference, the scrambled feedback control group, the clone feedback control group that rules out OOD artifacts, multi-seed validation, temperature scaling for gradient flow, time-weighted loss for emergence induction, parameter-matched feedforward control, structured ablation control, noise-level sweep, and statistical design refinements including bootstrap confidence intervals and multiple comparison correction. The author takes full responsibility for all scientific claims and interpretations.

-----

## Conflict of Interest

The author declares no conflict of interest. This research was conducted independently without institutional or corporate funding.

-----

## References

Banino, A., Balaguer, J., & Blundell, C. (2021). PonderNet: Learning to Ponder. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

Christiano, P. F., Leike, J., Brown, T., Marber, M., Amodei, D., & Schulman, J. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems, 30*.

Fleming, S. M., & Dolan, R. J. (2012). The neural basis of metacognitive ability. *Philosophical Transactions of the Royal Society B, 367*(1594), 1338–1349.

Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience, 8*, 443.

Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv preprint arXiv:1603.08983*.

Kirsch, L., & Schmidhuber, J. (2022). Eliminating Meta Optimization Through Self-Referential Meta Learning. *NeurIPS Meta-Learning Workshop / arXiv*.

Li, Y., et al. (2025). Spontaneous emergence of metacognition in neuronal computation. *Physical Review Research*.

Meyes, R., Lu, M., de Puiseau, C. W., & Meisen, T. (2019). Ablation Studies in Artificial Neural Networks. *arXiv preprint arXiv:1901.08644*.

Molano-Mazón, M., et al. (2025). Discovering cognitive strategies with tiny recurrent neural networks. *Nature*.

Morales, J., Lau, H., & Fleming, S. M. (2018). Domain-general and domain-specific patterns of activity supporting metacognition in human prefrontal cortex. *Journal of Neuroscience, 38*(14), 3534–3546.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *International Conference on Learning Representations (ICLR)*.

Schoen, B., et al. (2025). Stress Testing Deliberative Alignment for Anti-Scheming Training. *arXiv preprint arXiv:2509.xxxxx*.

Wei, J., et al. (2022). Emergent Abilities of Large Language Models. *Transactions on Machine Learning Research*.

Yamada, T., et al. (2022). Evolution of metamemory based on self-reference to own memory in artificial neural network with neuromodulation. *Scientific Reports, 12*, 6834.

---

# PART 2: MAIN EXPERIMENT REPORT

---
# Self-Correction Ablation Experiment — Results Report

## Executive Summary

We confirmed the **emergence of self-correction** in a 35-neuron RecurrentMLP.
After applying time-weighted loss and temperature scaling, the Baseline correction gain = **+0.0415 ± 0.0295**
(95% CI: [+0.023, +0.059], excluding zero). Removing the recurrent loop drops the gain to exactly 0,
and shuffling the feedback worsens the gain to **-0.064**. Furthermore, injecting **another model's
well-formed output** as feedback worsens the gain to **-0.075**, proving the network depends on
**its own specific output trajectory**, not just any reasonable feedback signal.
These results strongly support the hypothesis.

---

## 1. Experimental Setup

| Item | Value |
|------|-------|
| Architecture | Input(10) → H1(10) → H2(10) → Output(5), 35 neurons total |
| Feedback | tanh(prev_output / 2.0), temperature τ=2.0 |
| Loss | Time-weighted: w=[0.0, 0.2, 1.0] (t=1 free, t=3 focused) |
| Training | Full-batch SGD, lr=0.01, 500 epochs, T=3 |
| Data | Static pattern classification (5 classes, inter-class ambiguity 0.3) |
| Independent models | 10 (seed 0–9) |
| Noise levels | [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] |
| Random repeats | B1: 30 per model, C1: 30 per model |

## 2. Main Results (noise=0.5, model-level aggregation N=10)

| Group | acc_t1 | acc_t3 | gain (mean±std) | Interpretation |
|-------|--------|--------|-----------------|----------------|
| **Baseline** | 0.698±0.054 | 0.740±0.055 | **+0.042±0.030** | Self-correction occurs |
| A (Recurrent Cut) | 0.698±0.054 | 0.698±0.054 | 0.000±0.000 | Correction completely lost |
| B1 (Random Cut) | 0.515±0.048 | 0.511±0.037 | -0.004±0.016 | Correction lost + performance degraded |
| B2 (Structural Cut) | 0.199±0.032 | 0.199±0.032 | 0.000±0.000 | Function destroyed |
| C1 (Shuffled Feedback) | 0.698±0.054 | 0.634±0.041 | **-0.064±0.048** | Wrong feedback = degradation |
| **C2 (Clone Feedback)** | 0.698±0.054 | 0.623±0.043 | **-0.075±0.030** | Other model's valid output = degradation |
| D (Feedforward) | 0.746±0.067 | 0.746±0.067 | 0.000±0.000 | No correction possible (no recurrence) |
| D' (Param-matched FF) | 0.822±0.031 | 0.822±0.031 | 0.000±0.000 | Not a capacity effect |

### 95% Bootstrap CI (noise=0.5)

| Group | 95% CI |
|-------|--------|
| Baseline | [+0.023, +0.059] |
| A | [0.000, 0.000] |
| B1 | [-0.013, +0.006] |
| C1 | [-0.095, -0.036] |
| C2 | [-0.095, -0.058] |

### Holm-Bonferroni Corrected p-values (Wilcoxon signed-rank exact, Baseline vs. each group)

| Comparison | raw p | corrected p | Significance |
|------------|-------|-------------|--------------|
| Baseline vs C1 | 0.00195 | 0.0117 | * |
| Baseline vs C2 | 0.00195 | 0.0117 | * |
| Baseline vs B1 | 0.00391 | 0.0156 | * |
| Baseline vs A | 0.00781 | 0.0234 | * |
| Baseline vs D | 0.00781 | 0.0234 | * |
| Baseline vs D' | 0.00781 | 0.0234 | * |

## 3. Hypothesis Verification

### 3.1 Core Hypothesis: "Removing the recurrent loop eliminates self-correction"

**Strongly supported.**

- Baseline gain = +0.042 (positive, 95% CI excludes zero)
- Group A gain = 0.000 (correction completely vanishes when recurrence is removed)
- **acc_t1 is identical between Baseline and A** (0.698) → recurrence has no effect on initial recognition; it contributes purely to correction

### 3.2 "Self-reference, not just information flow" (Group C1, C2)

**Strongly supported.**

- C1 (shuffled feedback): recurrent connections preserved, distribution (mean/variance) identical, only positional information destroyed
- gain = -0.064 → a **-0.106** drop from Baseline (+0.042)
- Incorrect self-reference is **worse** than having none at all (A: 0.000)
- The network actively uses feedback, but when incorrect information is fed back, it is pulled toward wrong answers

**C2 (Clone Feedback) defeats the OOD criticism:**

- C2: injects **well-formed output** from a differently-seeded but identically-structured trained model
- gain = -0.075 → degradation **at least as severe** as C1 (-0.064); direct C1 vs C2 comparison is not statistically significant (Wilcoxon p = 0.695)
- Defeats the criticism that C1's degradation is merely an OOD (out-of-distribution) artifact
- The clone's output is a product of the same distribution, architecture, and training procedure — but since it is not "self," it cannot be used for correction
- **Conclusion: self-correction depends on the model's own output trajectory, not just any reasonable feedback signal.** See `REPORT_C2.md` for full analysis.

### 3.3 "Ruling out parameter count effects" (Group D')

**Supported.**

- D' (skip connection, parameter count matches Baseline): gain = 0.000
- acc_t1 = 0.822 (higher than Baseline's 0.698) → extra parameters improve FF performance only
- Self-correction arises from recurrent structure, not parameter capacity

### 3.4 Noise Dependence

**Partially supported.**

- In the noise sweep, Baseline gain peaks at noise=0.2 (~0.099), then decreases
- Differs somewhat from the hypothesis prediction ("gap widens at high noise") — moderate noise is optimal
- Interpretation: at very high noise, self-correction alone cannot compensate; correction is most effective at moderate difficulty

## 4. Emergence Condition Analysis

### Why emergence was absent in earlier experiments

1. **Uniform loss (1/T)**: accuracy pressure from t=1 → network loses incentive to utilize feedback
2. **Tanh saturation**: output logits ±5 → tanh derivative ≈ 0 → W_rec cannot learn

### Why emergence appeared after modifications

1. **Time-weighted loss [0.0, 0.2, 1.0]**: t=1 free → network learns correction strategy at t=2,3
2. **Temperature τ=2.0**: tanh(output/2.0) → prevents saturation, maintains W_rec gradient flow

### Key Lesson

> **Emergence is not a matter of capacity (neuron count) but of learning incentive (loss design) and gradient flow.**
> 35 neurons are sufficient.

## 5. Neuron Importance Analysis

Using decoupled ablation for H1 (feedforward-only for intelligence, recurrent-only for correction):

- **Shared (high intelligence + high correction)**: h1_1, h1_5, h1_7
  - Both feedforward and recurrent pathways contribute through these neurons
- **Correction-specialized**: h1_8
  - Moderate feedforward importance but high recurrent (correction) importance
- **Intelligence-specialized**: h1_6
  - High feedforward importance, zero recurrent contribution to correction
- **Negative correction**: h1_9
  - Removing its recurrent input *improves* correction gain (−0.035), suggesting this recurrent channel introduces noise

Note: H2 neurons use full knockout (no direct W_rec input), so their correction importance may be confounded with intelligence importance. H2 neurons h2_1, h2_2, h2_9 show high values on both axes, and h2_6 appears correction-specialized, but these should be interpreted with this caveat.

## 6. Generated Files

| File | Description |
|------|-------------|
| `results/raw_metrics.csv` | Full experiment data (3,960 rows, incl. C2) |
| `results/neuron_importance.csv` | Per-neuron importance scores |
| `results/ablation_comparison.png` | Group-wise gain comparison |
| `results/noise_sweep_curve.png` | Gain curves across noise levels |
| `results/accuracy_distribution.png` | B1 distribution + A/C1 positions |
| `results/neuron_importance_heatmap.png` | Intelligence vs. Correction scatter |
| `results/network_map.png` | Neuron connectivity visualization |

## 7. Robustness Analysis (Hyperparameter Sweep)

80 hyperparameter combinations (w1 × w2 × τ) × 10 models = 800 experiments.
**54/80 (68%)** configurations showed emergence. See `REPORT_SWEEP.md` for details.

| Parameter | Emergence Range | Failure Region |
|-----------|----------------|----------------|
| w1 (t=1 weight) | ≤ 0.2 (75–95%) | 0.3 (10%) |
| τ (temperature) | 1.0–3.0 (69–88%) | 5.0 (25%) |
| w2 (t=2 weight) | Full range (60–75%) | — |

Our experimental setting (w1=0, w2=0.2, τ=2.0) ranks 13th/80 — a mid-range value, not cherry-picked.

## 8. Limitations and Future Work

1. **Artificiality of time-weighted loss**: w=[0.0, 0.2, 1.0] is a structure that "induces" self-correction. Distinction from naturally emergent phenomena is needed
2. ~~**Group C2 not implemented**~~ → ✅ **Completed** (Clone Feedback, see `REPORT_C2.md`)
3. **Timestep-specific neuron masking**: Current importance is based on full-timestep ablation. Masking only at t=2,3 would yield more precise correction contribution measurements
4. **Validation at larger scales**: Confirmed at 35 neurons, but verification is needed to determine whether the same patterns hold at larger scales


---

# PART 3: C2 (CLONE FEEDBACK) SUPPLEMENTARY REPORT

---
# Group C2 (Clone Feedback) — Supplementary Experiment Report

## Executive Summary

Group C2 injects **another trained model's well-formed output** as feedback,
designed to defeat the criticism that C1's degradation is merely an OOD (out-of-distribution) artifact.

Result: **gain = -0.075 ± 0.030** (95% CI: [-0.095, -0.058], Wilcoxon signed-rank exact p = 0.00195, Holm-Bonferroni corrected p = 0.0117).
Degradation at least as severe as C1 (-0.064); the direct C1 vs C2 difference is not statistically significant (Wilcoxon p = 0.695).
Self-correction depends on **the model's own output trajectory**, not just any reasonable feedback signal.

---

## 1. Motivation: Why C2 Is Needed

### Possible Criticism of C1

Group C1 (permutation feedback) randomly shuffles feedback vector elements.
This may produce feedback that the network has never seen during training — an atypical distribution.

> "C1's performance drop is not because self-reference is broken,
> but simply because OOD input confuses the network."

### C2's Solution

C2 uses the output of an **identically-structured model trained with a different seed** as feedback:
- Same architecture (Input 10 → H1 10 → H2 10 → Output 5)
- Same training data and noise level
- Same training procedure (SGD, 500 epochs, lr=0.01)
- **Only the initialization seed differs** → different weights, different output trajectory

Therefore the clone's output is:
- ✅ In-distribution (normal, well-formed)
- ✅ A valid classification output (product of a trained model)
- ✅ Reasonable magnitude and range
- ❌ Not the model's "own" output

## 2. Experimental Design

### Method

```
When evaluating model i (seed=i):
  t=1: both target and clone forward independently (no feedback)
  t=2: target._prev_output ← clone's t=1 output
  t=3: target._prev_output ← clone's t=2 output

clone = model[(i+1) % 10]
```

### Configuration

| Item | Value |
|------|-------|
| Independent models | 10 (seed 0–9) |
| Clone pairing | model[i] ← model[(i+1)%10] |
| Noise levels | [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] |
| Training | Same as main experiment (500 epochs, lr=0.01) |

## 3. Results

### 3.1 Main Comparison (noise=0.5, N=10)

| Group | gain (mean±std) | 95% CI | p-value |
|-------|----------------|--------|---------|
| **Baseline** | **+0.042±0.030** | [+0.023, +0.059] | — |
| A (Recurrent Cut) | +0.000±0.000 | [0.000, 0.000] | 0.0234 * |
| C1 (Shuffled Feedback) | **-0.064±0.048** | [-0.095, -0.036] | 0.0117 * |
| **C2 (Clone Feedback)** | **-0.075±0.030** | [-0.095, -0.058] | 0.0117 * |

### 3.2 Across Noise Levels

| noise | Baseline | C1 | C2 |
|-------|----------|-----|-----|
| 0.1 | +0.087 | -0.181 | -0.148 |
| 0.2 | +0.099 | -0.153 | -0.161 |
| 0.3 | +0.090 | -0.109 | -0.128 |
| 0.5 | +0.042 | -0.064 | -0.075 |
| 0.7 | +0.015 | -0.029 | -0.035 |
| 1.0 | +0.006 | -0.014 | -0.013 |

### 3.3 Key Observations

1. **C2 ≈ C1**: Across intermediate noise levels (0.2–0.7), C2's gain is comparable to C1; the direct difference is not statistically significant (Wilcoxon p = 0.695)
2. **acc_t1 identical**: Baseline, A, C1, C2 all have acc_t1 = 0.698 (initial prediction unaffected by feedback)
3. **C2's feedback is in-distribution**: The clone's output is a valid classification output from a properly trained model
4. **Correction still fails**: Without its *own* output, the network cannot self-correct

## 4. Interpretation

### 4.1 Defeating the OOD Criticism

The "OOD artifact" criticism of C1 is completely defeated by C2:

| Criticism | C1 | C2 |
|-----------|-----|-----|
| Feedback has abnormal distribution? | Possible | ❌ Normal distribution |
| Feedback is meaningless noise? | Possible | ❌ Valid trained model output |
| Feedback magnitude/range is wrong? | Possible | ❌ Product of same architecture+training |
| Result: gain degraded? | ✅ -0.064 | ✅ -0.075 |

The fact that C2's in-distribution feedback still causes degradation proves that
**the cause of performance drop is not OOD, but the injection of "not-self" output**.

### 4.2 Mechanism of Self-Correction

The results suggest the following self-correction mechanism:

1. **t=1**: The network produces an initial prediction for noisy input
2. **t=2,3**: The network reads **specific patterns in its own previous output** to detect and correct errors
3. This process depends on a **specific alignment between the model's weights (W_rec) and its own output**
4. Another model's output is not "written in the same language," making error detection impossible

Analogy: it is the difference between reading your own handwritten notes to make corrections,
versus trying to correct from someone else's handwritten notes.

### 4.3 Why C2 Degrades Performance Comparably to C1

C2 tends to produce degradation comparable to C1 (the difference is not statistically significant, Wilcoxon p = 0.695):
- C1's shuffled feedback is meaningless, so the network may partially "ignore" it
- C2's clone output is **meaningful but misaligned with the model's own W_rec**, potentially causing the network to follow it into errors

This is also evident in the contrast with Group A (recurrent cut, gain=0):
- No feedback (A): no correction, but no degradation either
- Feedback present but incorrect (C1, C2): the network is actively pulled toward wrong answers

## 5. Conclusion

> **Self-correction depends not on "the presence of a reasonable feedback signal"
> but on "alignment with the model's own output trajectory."**
>
> Even the well-formed output of another model trained with the same architecture
> and procedure cannot be used for self-correction. This means that recurrent
> self-reference is not merely information flow, but an **individual mechanism**
> coupled with the model's own internal representations.

## 6. Files Created/Modified

| File | Description |
|------|-------------|
| `src/ablation.py` | Added `forward_sequence_with_clone()` |
| `src/metrics.py` | Added `compute_all_metrics_with_clone()` |
| `tests/test_clone_feedback.py` | 5 TDD tests for C2 |
| `experiments/run_c2_experiment.py` | C2 experiment runner (multiprocessing) |
| `results/raw_metrics.csv` | 60 C2 rows appended (total 3,960) |
| `results/ablation_comparison.png` | Updated with C2 |
| `results/noise_sweep_curve.png` | Updated with C2 |


---

# PART 4: HYPERPARAMETER SWEEP REPORT

---
# Hyperparameter Sweep — Robustness Analysis Report

## Executive Summary

We conducted **800 experiments** across 80 hyperparameter combinations (w1 × w2 × τ) × 10 independent models.
Result: emergence (self-correction) was confirmed in **54/80 (68%)** combinations.
Emergence is not a fragile phenomenon dependent on a single hyperparameter,
but a **robust phenomenon** appearing broadly across the range **w1 ≤ 0.2, τ ≤ 3.0**.

---

## 1. Sweep Configuration

| Item | Value |
|------|-------|
| w1 (t=1 weight) | [0.0, 0.1, 0.2, 0.3] |
| w2 (t=2 weight) | [0.1, 0.2, 0.3, 0.5] |
| w3 (t=3 weight) | 1.0 (fixed) |
| τ (feedback temperature) | [1.0, 1.5, 2.0, 3.0, 5.0] |
| Total combinations | 4 × 4 × 5 = **80 configs** |
| Models per config | 10 (seed 0–9) |
| Total experiments | **800** |
| Data | noise=0.5, 200 train / 200 test |
| Training | 500 epochs, lr=0.01, T=3 |
| Emergence criterion | mean gain > 0 AND gain > 0 in ≥ 60% of models |

## 2. Overall Results Summary

| Metric | Value |
|--------|-------|
| Overall mean gain | **+0.0150 ± 0.0405** |
| Overall median gain | **+0.0150** |
| Fraction with gain > 0 | **67.2%** (538 of 800 runs) |
| Emergence config count | **54/80 (68%)** |

## 3. Per-Hyperparameter Analysis

### 3.1 w1 (t=1 loss weight) — Strongest Influence

| w1 | mean gain | Emergence | Interpretation |
|----|-----------|-----------|----------------|
| **0.0** | **+0.0348** | **19/20 (95%)** | Optimal. Freedom at t=1 → learns correction strategy |
| 0.1 | +0.0209 | 18/20 (90%) | Good. Slight t=1 pressure is tolerable |
| 0.2 | +0.0066 | 15/20 (75%) | Borderline. Gain magnitude decreases |
| 0.3 | -0.0024 | 2/20 (10%) | Failure. t=1 pressure blocks correction incentive |

**Key finding**: w1 is the most decisive factor for emergence.
When w1=0.0 (ignoring t=1 loss), the network gains the freedom of "the first guess doesn't need to be correct"
and learns iterative correction strategies through feedback. Raising w1 to 0.3 applies accuracy pressure at t=1,
causing the network to focus on feedforward performance and ignore the recurrent pathway.

### 3.2 τ (feedback temperature) — Second Strongest Influence

| τ | mean gain | Emergence | Interpretation |
|---|-----------|-----------|----------------|
| **1.0** | **+0.0248** | **13/16 (81%)** | Optimal. Sharp feedback |
| 1.5 | +0.0180 | 14/16 (88%) | Good |
| 2.0 | +0.0145 | 12/16 (75%) | Good (default in main experiment) |
| 3.0 | +0.0148 | 11/16 (69%) | Borderline |
| 5.0 | +0.0040 | 4/16 (25%) | Weakened. Feedback signal diluted |

**Key finding**: Emergence is broadly maintained across τ=1.0–3.0.
The sharp decline at τ=5.0 occurs because high temperature makes `tanh(output/5.0)` nearly linear (≈ output/5),
reducing the discriminability of the feedback signal.
Conversely, τ=1.0 risks tanh saturation, but in practice the network adapts and shows the strongest emergence.

### 3.3 w2 (t=2 loss weight) — Weak Influence

| w2 | mean gain | Emergence | Interpretation |
|----|-----------|-----------|----------------|
| 0.1 | +0.0192 | 15/20 (75%) | Slightly optimal |
| 0.2 | +0.0155 | 14/20 (70%) | |
| 0.3 | +0.0128 | 12/20 (60%) | |
| 0.5 | +0.0127 | 13/20 (65%) | |

**Key finding**: w2 has only marginal influence on emergence.
t=2 corresponds to the correction "process" — whether strong or weak pressure is applied to this intermediate step,
the final result (t=3) shows little difference. The network learns correction strategies
sufficiently from the t=3 loss signal alone.

## 4. Cross-Analysis: w1 × τ Emergence Matrix

The table below shows the fraction of 4 w2 values where emergence was confirmed for each (w1, τ) combination.

```
         τ=1.0  τ=1.5  τ=2.0  τ=3.0  τ=5.0
w1=0.0    4/4    4/4    4/4    4/4    3/4     → 95% (19/20)
w1=0.1    4/4    4/4    4/4    4/4    2/4     → 90% (18/20)
w1=0.2    4/4    4/4    3/4    3/4    1/4     → 75% (15/20)
w1=0.3    1/4    0/4    1/4    0/4    0/4     → 10% ( 2/20) ← failure region
```

**Pattern**: Emergence is confirmed at nearly 100% in the rectangular region w1 ≤ 0.2, τ ≤ 3.0.
This demonstrates that emergence is not "a phenomenon lucky enough to be observed only at a specific combination of w1 and τ."

## 5. Best / Worst Configurations

### Top 5 (Strongest Emergence)

| Rank | w1 | w2 | τ | mean gain |
|------|----|----|---|-----------|
| 1 | 0.0 | 0.1 | 1.0 | **+0.054** |
| 2 | 0.0 | 0.2 | 1.0 | +0.053 |
| 3 | 0.0 | 0.3 | 1.0 | +0.052 |
| 4 | 0.0 | 0.5 | 1.0 | +0.051 |
| 5 | 0.0 | 0.1 | 1.5 | +0.047 |

All top 5 have **w1=0.0**, and the top 4 have τ=1.0. w2 has virtually no effect.

### Bottom 5 (Emergence Failure)

| Rank | w1 | w2 | τ | mean gain |
|------|----|----|---|-----------|
| 76 | 0.3 | 0.2 | 2.0 | -0.006 |
| 77 | 0.0 | 0.5 | 5.0 | -0.007 | ← even w1=0.0 fails at τ=5.0 |
| 78 | 0.3 | 0.3 | 1.5 | -0.008 |
| 79 | 0.3 | 0.5 | 1.5 | -0.008 |
| 80 | 0.3 | 0.5 | 2.0 | **-0.012** |

Common factor in bottom configurations: **w1=0.3** (most cases) or **τ=5.0**.

## 6. Heatmap Interpretation

### τ=1.0 (Optimal Temperature)
```
w1\w2    0.1     0.2     0.3     0.5
0.0   +0.054  +0.053  +0.052  +0.051   ← uniformly strong emergence
0.1   +0.051  +0.030  +0.033  +0.023   ← slight decrease with higher w2
0.2   +0.012  +0.015  +0.013  +0.011   ← sharp decline begins
0.3   +0.002  +0.000  -0.004  +0.003   ← emergence lost
```

The w1=0.0 row shows nearly identical values of **+0.051–+0.054** regardless of w2.
This visually confirms that w2 has negligible effect on emergence.

### τ=5.0 (Excessive Temperature)
```
w1\w2    0.1     0.2     0.3     0.5
0.0   +0.018  +0.012  +0.008  +0.013   ← only weak emergence remains
0.1   +0.004  +0.002  +0.005  +0.008   ← near zero
0.2   -0.001  -0.002  +0.003  +0.004   ← oscillating around zero
0.3   -0.007  -0.004  -0.002  +0.002   ← negative territory
```

At τ=5.0, even with w1=0.0, gain is weakened to +0.008–+0.018.

## 7. Physical Interpretation

### Why is w1 important?

In the time-weighted loss `L = w1·L(t=1) + w2·L(t=2) + 1.0·L(t=3)`:

- **w1=0.0**: No penalty for any prediction at t=1. The network gains the freedom that
  "the first guess can be wrong" and can focus learning resources on correction at t=2–3.
- **w1=0.3**: 30% penalty at t=1. The network invests in the feedforward pathway (W_ih1→W_h1h2→W_h2o)
  to improve t=1 accuracy, relatively weakening the learning of the recurrent pathway (W_rec).

This is a **trade-off**: feedforward performance ↔ recurrent correction capability. As w1 increases,
learning resources shift toward the feedforward side.

### Why is τ important?

In `feedback = tanh(prev_output / τ)`:

- **τ=1.0**: tanh provides meaningful nonlinearity at output logit magnitudes around 1.
  High discriminability of the feedback signal allows hidden1 to precisely identify "what the previous prediction was."
- **τ=5.0**: tanh(x/5) ≈ x/5 (linear approximation). The feedback signal is linearly weakened,
  sharply reducing information transfer efficiency. Correction capability through W_rec is limited.

## 8. Robustness Assessment

### Response to P-hacking Criticism

| Question | Answer |
|----------|--------|
| "Is emergence visible in only one specific combination?" | **No.** Confirmed in 54/80 (68%) configurations. |
| "Do w1, w2, and τ all need precise tuning?" | **No.** If w1 ≤ 0.2, it works across τ=1.0–3.0. w2 is nearly irrelevant. |
| "Is the original experiment (w1=0, w2=0.2, τ=2.0) cherry-picked?" | **No.** This combination (gain=+0.042) ranks 13th/80. Mid-range, not optimal. |
| "Do results depend on the random seed?" | **No.** In emergence configurations, 60%+ models show positive gain (10 independent seeds). |

### Robustness Range

```
Range where emergence is confirmed (54/80 configs):
  w1 ∈ [0.0, 0.2]     — stable across 3/4 values
  w2 ∈ [0.1, 0.5]     — full range (w2 irrelevant)
  τ  ∈ [1.0, 3.0]     — stable across 4/5 values

Range where emergence is weak or absent:
  w1 = 0.3             — excessive t=1 pressure → correction incentive lost
  τ  = 5.0             — feedback information diluted → correction capability weakened
```

## 9. Position of Our Experimental Setting

The default setting used in our main experiment (w1=0.0, w2=0.2, τ=2.0, gain=+0.042):
- Ranks **13th** out of 80 combinations (top 16%)
- 78% of the optimal value (w1=0.0, w2=0.1, τ=1.0, gain=+0.054)
- Selected from the **mid-range**, not the optimum

This demonstrates that our experiment is not based on a cherry-picked setting. Emergence is observed
even with a reasonable mid-range configuration, not just when reporting the optimal setting.

## 10. Generated Files

| File | Description |
|------|-------------|
| `results/sweep_hyperparams.csv` | Full 800-experiment raw data |
| `results/sweep_heatmap_tau1.0.png` | τ=1.0 w1×w2 heatmap |
| `results/sweep_heatmap_tau1.5.png` | τ=1.5 w1×w2 heatmap |
| `results/sweep_heatmap_tau2.0.png` | τ=2.0 w1×w2 heatmap |
| `results/sweep_heatmap_tau3.0.png` | τ=3.0 w1×w2 heatmap |
| `results/sweep_heatmap_tau5.0.png` | τ=5.0 w1×w2 heatmap |
| `results/sweep_tau_overview.png` | Gain distribution scatter plot by τ |

## 11. Suggested Paper Integration

### Section 2.4 Extension: "Robustness Analysis"

> We swept three hyperparameters: the t=1 loss weight w1 ∈ {0.0, 0.1, 0.2, 0.3},
> the t=2 loss weight w2 ∈ {0.1, 0.2, 0.3, 0.5}, and the feedback temperature
> τ ∈ {1.0, 1.5, 2.0, 3.0, 5.0}, yielding 80 configurations with 10 independent
> models each (800 total runs).
>
> Self-correction emergence was confirmed in 54/80 (68%) configurations.
> The phenomenon is robust across w1 ∈ [0.0, 0.2] and τ ∈ [1.0, 3.0],
> with w2 having negligible effect. Emergence fails only when the t=1 loss
> weight is too high (w1 ≥ 0.3, removing the incentive for iterative correction)
> or when the feedback temperature is too high (τ ≥ 5.0, diluting the feedback signal).
> Our reported results use w1=0.0, w2=0.2, τ=2.0, which ranks 13th/80 — a mid-range
> configuration, not a cherry-picked optimum.

### Appendix: Raw Sweep Table

Full mean gain, std, and emergence status for all 80 configurations published as a table.
Reproducible from `results/sweep_hyperparams.csv`.

## 12. Conclusions

1. **Emergence is robust**: Confirmed in 68% of 80 combinations. Not dependent on a single hyperparameter.
2. **The key factor is w1 (t=1 loss weight)**: Signaling to the network that "the initial prediction is free" is most important.
3. **τ (temperature) is secondary**: An adequate range (1.0–3.0) is sufficient. Weakened only at the extreme value (5.0).
4. **w2 (t=2 loss weight) is irrelevant**: Nearly identical results across the full range.
5. **P-hacking possibility excluded**: Our experimental setting is a mid-range value, not the optimum, and results are reproducible across a broad range.


---

# PART 5: STATISTICAL JUSTIFICATION

---
# Statistical Test Selection Justification — Wilcoxon Signed-Rank Exact Test

> Prepared for reviewer inquiries. Documents the rationale for the statistical test
> used in the paper, assumption verification against actual data, and comparison
> with alternative tests.

---

## 1. Test Selection: Why Wilcoxon Signed-Rank

### 1.1 The Experimental Design Is Paired

Ten independent models (seeds 0–9) are trained, then the **same model** is evaluated under Baseline and each treatment (A, B1, C1, C2, D, D'). The comparison units are:

```
(Baseline seed=0 vs A seed=0),
(Baseline seed=1 vs A seed=1),
...
(Baseline seed=9 vs A seed=9)
```

This is a **paired samples** design, so a paired test (Wilcoxon signed-rank) is appropriate — not an independent samples test (Mann-Whitney U).

### 1.2 Why Not a Parametric Test (Paired t-test)

- N=10 is too small to reliably verify the normality assumption.
- There is no prior evidence that the correction gain distribution follows a normal distribution.
- Wilcoxon signed-rank is a nonparametric test that does not require the normality assumption.
- At N=10, exact testing is feasible, eliminating dependence on approximations.

### 1.3 Why Exact Test

- At N=10, there are only 2^10 = 1,024 possible sign assignments.
- The exact p-value can be computed by full enumeration.
- Normal approximation is unnecessary, completely avoiding small-sample approximation errors.
- The implementation is simple enough to use numpy alone without external libraries (scipy, etc.).
- Verified to match scipy.stats.wilcoxon exact results to 6 decimal places.

---

## 2. Assumption Verification

The four assumptions of the Wilcoxon signed-rank test were checked against actual data.

### 2.1 Paired Samples

**Satisfied.** Each comparison is made within the same seed model (see Section 1.1).

### 2.2 Continuity of Differences (d)

**Practically satisfied.** Accuracy is theoretically discrete (1/200 = 0.005 units), but per-model mean gains take sufficiently diverse values:

| Comparison | Unique values / Total | Treatable as continuous |
|------------|----------------------|------------------------|
| Baseline vs A | 10/10 | Yes |
| Baseline vs B1 | 10/10 | Yes |
| Baseline vs C1 | 10/10 | Yes |
| Baseline vs C2 | 9/10 (1 tie) | Yes |
| Baseline vs D | 10/10 | Yes |
| Baseline vs D' | 10/10 | Yes |

### 2.3 Symmetry of Differences (d)

The Wilcoxon signed-rank test assumes that the distribution of differences is symmetric about zero under H0. However, **using an exact test greatly reduces dependence on this assumption** — the exact p-value directly computes the probability that the observed rank statistic occurs by pure chance.

For reference, we report the skewness of each comparison:

| Comparison | Skewness | Kurtosis | Assessment |
|------------|----------|----------|------------|
| Baseline vs A | -0.370 | -0.775 | Near symmetric |
| Baseline vs B1 | -0.706 | -0.267 | Near symmetric |
| Baseline vs C1 | +0.247 | -0.349 | Near symmetric |
| Baseline vs C2 | +1.047 | +0.219 | Slight right skew |
| Baseline vs D | -0.370 | -0.775 | Near symmetric |
| Baseline vs D' | -0.370 | -0.775 | Near symmetric |

C2's skewness of 1.047 is slightly elevated, driven by seed=2 having a larger difference (0.225) than other models (0.065–0.175). However:
- The exact test does not depend on the symmetry assumption required by normal approximation.
- All 10 C2 differences are **positive** (minimum 0.065), so directionality is unambiguous.
- Under the most conservative interpretation, p = 0.001953 (= 1/512), which is the exact upper bound probability that all 10 differences share the same sign.

### 2.4 Ties

| Comparison | Zero differences (d=0) | Tied |d| groups | Assessment |
|------------|----------------------|---------------------|------------|
| Baseline vs A | 0 | 1 group (2 values, |d|=0.045) | Negligible |
| Baseline vs B1 | 0 | None | Clean |
| Baseline vs C1 | 0 | None | Clean |
| Baseline vs C2 | 0 | 1 group (2 values, |d|=0.065) | Negligible |
| Baseline vs D | 0 | 1 group (2 values, |d|=0.045) | Negligible |
| Baseline vs D' | 0 | 1 group (2 values, |d|=0.045) | Negligible |

- No zero differences (d=0), so no sample reduction.
- Tied |d| values occur in at most 1 group of 2 — handled by midrank averaging.

---

## 3. Results

### 3.1 Exact P-values (Before Holm-Bonferroni Correction)

| Comparison | T+ | T- | T | exact p | Direction |
|------------|----|----|---|---------|-----------|
| Baseline vs A | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > A |
| Baseline vs B1 | 54.0 | 1.0 | 1.0 | 0.003906 | Baseline > B1 |
| Baseline vs C1 | 55.0 | 0.0 | 0.0 | 0.001953 | Baseline > C1 |
| Baseline vs C2 | 55.0 | 0.0 | 0.0 | 0.001953 | Baseline > C2 |
| Baseline vs D | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > D |
| Baseline vs D' | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > D' |

C1 and C2 have T=0 (all 10 differences are positive) -> minimum possible p-value = 2/1024 = 0.00195.
This is the **resolution limit** of the N=10 Wilcoxon signed-rank exact test; even if the true effect is stronger, the p-value cannot go lower.

### 3.2 After Holm-Bonferroni Correction

Holm-Bonferroni step-down correction applied across 6 comparisons:

| Rank | Comparison | raw p | corrected p | Significance |
|------|------------|-------|-------------|--------------|
| 1 | Baseline vs C1 | 0.001953 | 0.01172 | * |
| 2 | Baseline vs C2 | 0.001953 | 0.01172 | * |
| 3 | Baseline vs B1 | 0.003906 | 0.01562 | * |
| 4 | Baseline vs A | 0.007812 | 0.02344 | * |
| 5 | Baseline vs D | 0.007812 | 0.02344 | * |
| 6 | Baseline vs D' | 0.007812 | 0.02344 | * |

All comparisons significant at alpha=0.05. At alpha=0.01, only C1 and C2 are significant.

### 3.3 Verification: Agreement with scipy.stats.wilcoxon

| Comparison | Our implementation (exact) | scipy (exact) | Match |
|------------|--------------------------|---------------|-------|
| Baseline vs A | T=2.5, p=0.007812 | T=2.5, p=0.007812 | Exact match |
| Baseline vs B1 | T=1.0, p=0.003906 | T=1.0, p=0.003906 | Exact match |
| Baseline vs C1 | T=0.0, p=0.001953 | T=0.0, p=0.001953 | Exact match |
| Baseline vs C2 | T=0.0, p=0.001953 | T=0.0, p=0.001953 | Exact match |
| Baseline vs D | T=2.5, p=0.007812 | T=2.5, p=0.007812 | Exact match |
| Baseline vs D' | T=2.5, p=0.007812 | T=2.5, p=0.007812 | Exact match |

---

## 4. Comparison with Alternative Tests

### 4.1 Mann-Whitney U Test (Independent Samples)

**Inappropriate.** The experimental design is paired, so a paired test must be used. An independent test fails to control for inter-model variance, leading to incorrect inference. For reference, Mann-Whitney U (normal approximation) results:

| Comparison | MWU p (normal approx.) | Wilcoxon exact p | Note |
|------------|----------------------|------------------|------|
| Baseline vs C2 | 1.56e-04 | 1.95e-03 | MWU yields excessively low p |

MWU yields a lower p-value not because it has higher power, but because it references an inappropriate distribution by ignoring the paired structure.

### 4.2 Paired t-test (Parametric)

At N=10, the normality assumption cannot be trusted. A paired t-test would yield similar conclusions, but there is no reason to accept the risk of normality violation in small samples.

### 4.3 Permutation Test

The least assumption-dependent nonparametric test. At N=10, exact testing via 10! = 3,628,800 permutations is feasible. More general than Wilcoxon signed-rank, but Wilcoxon is more widely accepted in the literature, and result differences are negligible.

---

## 5. Note on Resolution Limit

The minimum possible p-value for the N=10 Wilcoxon signed-rank exact test is:

- T=0 (all differences share the same sign): p = 2/2^10 = **0.001953**
- This value can remain significant after alpha=0.01 correction, but values below 0.001 are unachievable

For more precise p-values, the sample size (number of independent models) must be increased. That C1 and C2 both have raw p = 0.001953 reflects the fact that **both conditions show lower gain than Baseline in all 10/10 models** — a ceiling effect of the test, not an indication that effect sizes are identical.

---

## 6. Summary

| Item | Assessment |
|------|------------|
| Test type | Wilcoxon signed-rank (paired, nonparametric) — Appropriate |
| P-value computation | Exact enumeration (2^10 = 1,024) — Appropriate |
| Paired samples assumption | Same-seed model comparison — Satisfied |
| Continuity assumption | Unique value ratio 9–10/10 — Satisfied |
| Symmetry assumption | Skewness |<=1.05|, minimized by exact test — Satisfied |
| Ties | 0 zero-differences, <=1 tied group of 2 — Satisfied |
| External library dependency | None (numpy only) — Verified |
| scipy verification | 6/6 comparisons match exactly — Verified |


---

# REVIEW CHECKLIST

Please verify the following:

1. **Internal Consistency**: Do all numbers in the paper match the corresponding report tables?
2. **Statistical Methodology**: Is the Wilcoxon signed-rank exact test appropriate for N=10? Are bootstrap CIs correctly described? Is Holm-Bonferroni correction properly applied?
3. **Controls Completeness**: Are Groups A, B1, B2, C1, C2, D, D' sufficient to rule out alternative explanations?
4. **Overclaiming**: Are claims appropriately hedged? Does the language match the statistical evidence?
5. **Clone Feedback Logic**: Does the C2 design genuinely rule out the OOD criticism? Are C1 vs C2 comparisons appropriately stated given p=0.695?
6. **Robustness**: Does the hyperparameter sweep adequately address p-hacking concerns?
7. **Missing Limitations**: Are there important caveats not mentioned?
8. **Reproducibility**: Is enough detail provided to replicate the experiments?
9. **Anthropomorphism**: Is mechanistic language used instead of anthropomorphic attributions?
