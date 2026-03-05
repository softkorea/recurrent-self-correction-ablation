# Self-Awareness Ablation Experiment — Results Report

> **[한국어 원문 / Korean original: `REPORT.md`]**

## Executive Summary

We confirmed the **emergence of self-correction** in a 35-neuron RecurrentMLP.
After applying time-weighted loss and temperature scaling, the Baseline correction gain = **+0.0415 ± 0.0295**
(95% CI: [+0.023, +0.059], excluding zero). Removing the recurrent loop drops the gain to exactly 0,
and shuffling the feedback worsens the gain to **-0.064**. These results strongly support the hypothesis.

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
| B2 (Structural Cut) | 0.207±0.034 | 0.207±0.034 | 0.000±0.000 | Function destroyed |
| C1 (Shuffled Feedback) | 0.698±0.054 | 0.634±0.041 | **-0.064±0.048** | Wrong feedback = degradation |
| D (Feedforward) | 0.746±0.067 | 0.746±0.067 | 0.000±0.000 | No correction possible (no recurrence) |
| D' (Param-matched FF) | 0.822±0.031 | 0.822±0.031 | 0.000±0.000 | Not a capacity effect |

### 95% Bootstrap CI (noise=0.5)

| Group | 95% CI |
|-------|--------|
| Baseline | [+0.023, +0.059] |
| A | [0.000, 0.000] |
| B1 | [-0.013, +0.006] |
| C1 | [-0.095, -0.036] |

### Holm-Bonferroni Corrected p-values (Baseline vs. each group)

| Comparison | p-value | Significance |
|------------|---------|--------------|
| Baseline vs C1 | 1.90e-03 | ** |
| Baseline vs A | 5.68e-03 | ** |
| Baseline vs D | 5.68e-03 | ** |
| Baseline vs D' | 5.68e-03 | ** |
| Baseline vs B1 | 5.68e-03 | ** |

## 3. Hypothesis Verification

### 3.1 Core Hypothesis: "Removing the recurrent loop eliminates self-correction"

**Strongly supported.**

- Baseline gain = +0.042 (positive, 95% CI excludes zero)
- Group A gain = 0.000 (correction completely vanishes when recurrence is removed)
- **acc_t1 is identical between Baseline and A** (0.698) → recurrence has no effect on initial recognition; it contributes purely to correction

### 3.2 "Self-reference, not just information flow" (Group C1)

**Strongly supported.**

- C1 (shuffled feedback): recurrent connections preserved, distribution (mean/variance) identical, only positional information destroyed
- gain = -0.064 → a **-0.106** drop from Baseline (+0.042)
- Incorrect self-reference is **worse** than having none at all (A: 0.000)
- The network actively uses feedback, but when incorrect information is fed back, it is pulled toward wrong answers

### 3.3 "Ruling out parameter count effects" (Group D')

**Supported.**

- D' (skip connection, parameter count matches Baseline): gain = 0.000
- acc_t1 = 0.822 (higher than Baseline's 0.698) → extra parameters improve FF performance only
- Self-correction arises from recurrent structure, not parameter capacity

### 3.4 Noise Dependence

**Partially supported.**

- In the noise sweep, Baseline gain peaks at noise=0.3 (~0.095), then decreases
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

From the Neuron Importance Heatmap:

- **Upper-right (important for both intelligence + correction)**: h1_7, h1_8, h2_2, h2_9, h1_1
  - These neurons are critical for both pattern recognition and self-correction
- **Upper-left (important for correction only)**: h2_6, h2_7
  - Low contribution to intelligence but specialized for self-correction
- **Lower-right (important for intelligence only)**: h1_6, h2_0, h2_4
  - Contribute to pattern recognition only; irrelevant or detrimental to correction

## 6. Generated Files

| File | Description |
|------|-------------|
| `results/raw_metrics.csv` | Full experiment data (3,900 rows) |
| `results/neuron_importance.csv` | Per-neuron importance scores |
| `results/ablation_comparison.png` | Group-wise gain comparison |
| `results/noise_sweep_curve.png` | Gain curves across noise levels |
| `results/accuracy_distribution.png` | B1 distribution + A/C1 positions |
| `results/neuron_importance_heatmap.png` | Intelligence vs. Correction scatter |
| `results/network_map.png` | Neuron connectivity visualization |

## 7. Robustness Analysis (Hyperparameter Sweep)

80 hyperparameter combinations (w1 × w2 × τ) × 10 models = 800 experiments.
**54/80 (68%)** configurations showed emergence. See `results/REPORT_SWEEP.en.md` for details.

| Parameter | Emergence Range | Failure Region |
|-----------|----------------|----------------|
| w1 (t=1 weight) | ≤ 0.2 (75–95%) | 0.3 (10%) |
| τ (temperature) | 1.0–3.0 (69–88%) | 5.0 (25%) |
| w2 (t=2 weight) | Full range (60–75%) | — |

Our experimental setting (w1=0, w2=0.2, τ=2.0) ranks 13th/80 — a mid-range value, not cherry-picked.

## 8. Limitations and Future Work

1. **Artificiality of time-weighted loss**: w=[0.0, 0.2, 1.0] is a structure that "induces" self-correction. Distinction from naturally emergent phenomena is needed
2. **Group C2 (batch-shuffle) not implemented**: Present in the plan but not yet applied
3. **Timestep-specific neuron masking**: Current importance is based on full-timestep ablation. Masking only at t=2,3 would yield more precise correction contribution measurements
4. **Validation at larger scales**: Confirmed at 35 neurons, but verification is needed to determine whether the same patterns hold at larger scales
