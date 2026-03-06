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

The Wilcoxon signed-rank test assumes that the distribution of differences is symmetric about zero under H0. The exact test avoids the additional normality approximation used for large samples, but **the symmetry assumption itself is not eliminated by using exact computation** — it remains a structural assumption of the test. However, for our data the assumption is approximately satisfied (see skewness values below), and when all differences share the same sign (as in C1 and C2), the result is robust to moderate asymmetry.

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
- All 10 C2 differences are **positive** (minimum 0.065), so directionality is unambiguous.
- When all differences share the same sign, the Wilcoxon test is robust to moderate asymmetry because the rank ordering is consistent.
- Under the most conservative interpretation (sign test), p = 2 × (1/2)^10 = 0.001953, which is the exact probability that all 10 differences share the same sign by chance.

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

### 3.2.1 Additional Contrasts: C1 vs A and C2 vs A

To directly test the claim that non-veridical feedback is worse than no feedback, we performed pairwise tests comparing C1 and C2 against A (no feedback):

| Comparison | T+ | T- | T | exact p | 95% Bootstrap CI | Direction |
|------------|----|----|---|---------|-------------------|-----------|
| C1 vs A | 1.0 | 54.0 | 1.0 | 0.003906 | [−0.095, −0.036] | C1 < A |
| C2 vs A | 0.0 | 55.0 | 0.0 | 0.001953 | [−0.095, −0.058] | C2 < A |

Both comparisons confirm that non-veridical feedback actively degrades performance below the no-feedback baseline. These are reported as supplementary contrasts outside the primary Holm-Bonferroni family.

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

The least assumption-dependent nonparametric test for paired data. For N=10 paired observations, the exact sign-flip (paired permutation) test enumerates all 2^10 = 1,024 sign assignments — the same enumeration space as the Wilcoxon signed-rank test. The difference is that the permutation test uses the mean difference as its statistic (ignoring rank information), while Wilcoxon uses signed ranks. For our data, both yield comparable conclusions. Wilcoxon is more widely accepted in the literature and has slightly higher power when the symmetry assumption holds.

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
| Symmetry assumption | Skewness |<=1.05|, approximately satisfied; robust when all diffs share sign — Satisfied |
| Ties | 0 zero-differences, <=1 tied group of 2 — Satisfied |
| External library dependency | None (numpy only) — Verified |
| scipy verification | 6/6 comparisons match exactly — Verified |
