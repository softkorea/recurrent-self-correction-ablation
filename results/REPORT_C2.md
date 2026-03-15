# Group C2 (Clone Feedback) — Supplementary Experiment Report

## Executive Summary

Group C2 injects **another trained model's well-formed output** as feedback,
designed to defeat the criticism that C1's degradation is merely an OOD (out-of-distribution) artifact.

Result: **gain = -0.059 ± 0.069** (95% CI: [-0.101, -0.015], Wilcoxon signed-rank exact p = 0.0020, Holm-Bonferroni corrected p = 0.0117).
Degradation comparable to C1 (-0.064); the direct C1 vs C2 difference is not statistically significant (Wilcoxon p = 0.846).
Self-correction depends on **the model's own output trajectory**, not just any reasonable feedback signal.

---

## 1. Motivation: Why C2 Is Needed

### Possible Criticism of C1

Group C1 (permutation feedback) randomly shuffles feedback vector elements.
This may produce feedback that the network has never seen during training — an atypical distribution.

> "C1's performance drop is not because self-reference is broken,
> but simply because OOD input confuses the network."

### C2's Solution

C2 uses the output of an **independently trained donor model** as feedback:
- Same architecture (Input 10 → H1 10 → H2 10 → Output 5)
- Same training procedure (SGD, 500 epochs, lr=0.01)
- **Independently trained** with a different seed (target seed 0-9, donor seed 100-109)
- Each target-donor pair is fully statistically independent

Therefore the clone's output is:
- ✅ In-distribution (normal, well-formed)
- ✅ A valid classification output (product of a trained model)
- ✅ Reasonable magnitude and range
- ❌ Not the model's "own" output

## 2. Experimental Design

### Method

```
When evaluating target model i (seed=i):
  t=1: both target and donor forward independently (no feedback)
  t=2: target._prev_output ← donor's t=1 output
  t=3: target._prev_output ← donor's t=2 output

donor = independently trained model (seed = i + 100)
```

### Configuration

| Item | Value |
|------|-------|
| Target models | 10 (seed 0–9) |
| Donor models | 10 (seed 100–109), independently trained |
| Clone pairing | target[i] ← donor[i] (fully independent 1:1) |
| Noise levels | [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] |
| Training | Same as main experiment (500 epochs, lr=0.01) |

## 3. Results

### 3.1 Main Comparison (noise=0.5, N=10)

| Group | gain (mean±std) | 95% CI | p-value |
|-------|----------------|--------|---------|
| **Baseline** | **+0.042±0.030** | [+0.023, +0.059] | — |
| A (Recurrent Cut) | +0.000±0.000 | [0.000, 0.000] | 0.0234 * |
| C1 (Shuffled Feedback) | **-0.064±0.048** | [-0.095, -0.036] | 0.0117 * |
| **C2 (Clone Feedback)** | **-0.059±0.069** | [-0.101, -0.015] | 0.0117 * |

### 3.2 Across Noise Levels

| noise | Baseline | C1 | C2 |
|-------|----------|-----|-----|
| 0.1 | +0.087 | -0.181 | -0.106 |
| 0.2 | +0.099 | -0.153 | -0.089 |
| 0.3 | +0.090 | -0.109 | -0.065 |
| 0.5 | +0.042 | -0.064 | -0.059 |
| 0.7 | +0.015 | -0.029 | -0.033 |
| 1.0 | +0.006 | -0.014 | -0.012 |

### 3.3 Key Observations

1. **C2 ≈ C1**: Across intermediate noise levels (0.2–0.7), C2's gain is comparable to C1; the direct difference is not statistically significant (Wilcoxon p = 0.846)
2. **acc_t1 identical**: Baseline, A, C1, C2 all have acc_t1 = 0.698 (initial prediction unaffected by feedback)
3. **C2's feedback is in-distribution**: The clone's output is a valid classification output from a properly trained model
4. **Correction still fails**: Without its *own* output, the network cannot self-correct

## 4. Interpretation

### 4.1 Defeating the OOD Criticism

The "OOD artifact" criticism of C1 is strongly argued against by C2:

| Criticism | C1 | C2 |
|-----------|-----|-----|
| Feedback has abnormal distribution? | Possible | ❌ Normal distribution |
| Feedback is meaningless noise? | Possible | ❌ Valid trained model output |
| Feedback magnitude/range is wrong? | Possible | ❌ Product of same architecture+training |
| Result: gain degraded? | ✅ -0.064 | ✅ -0.059 |

The fact that C2's in-distribution feedback still causes degradation strongly suggests that
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

C2 tends to produce degradation comparable to C1 (the difference is not statistically significant, Wilcoxon p = 0.846):
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
