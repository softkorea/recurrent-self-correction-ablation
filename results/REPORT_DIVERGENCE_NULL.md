# Cosine Divergence Null Baseline

## Summary Statistics

| Condition | Mean | SD | Min | Max |
|-----------|------|----|-----|-----|
| (a) Isotropic random | 1.001 | 0.053 | 0.802 | 1.200 |
| (b) Same-class resampled | 0.361 | 0.428 | 0.000 | 1.984 |
| (c) Clone | 0.734 | 0.485 | 0.006 | 1.972 |

## Interpretation

### Per-seed means

| Seed | Random | Resampled | Clone |
|------|--------|-----------|-------|
| 0 | 0.999 | 0.437 | 0.521 |
| 1 | 1.005 | 0.366 | 0.532 |
| 2 | 1.002 | 0.267 | 0.601 |
| 3 | 1.002 | 0.268 | 0.937 |
| 4 | 0.999 | 0.405 | 1.049 |
| 5 | 1.004 | 0.469 | 0.736 |
| 6 | 1.003 | 0.476 | 0.602 |
| 7 | 1.001 | 0.361 | 0.634 |
| 8 | 0.997 | 0.217 | 0.864 |
| 9 | 1.003 | 0.340 | 0.866 |

### Critical ordering

Ordering: Resampled (0.361) < Clone (0.734) < Random (1.001)

**Clone divergence < Random divergence**: Clone is MORE aligned than random, not less. The 0.73 divergence must be interpreted in this context: it represents substantial but partial alignment, not evidence of maximal misalignment.

**Same-class resampled < Clone**: Within-model same-class trials are more aligned than cross-model trials, supporting model-specific feedback contract beyond class-level effects.
