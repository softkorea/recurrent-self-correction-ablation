# Wrong-Trajectory Substitution Experiment

## Static Input

| Seed | Self-Current Gain | Self-Wrong-Trial Gain | Clone Gain | Group A Gain | Matched |
|------|-------------------|-----------------------|------------|--------------|----------|
| 0 | -0.025 | +0.010 | -0.065 | +0.000 | 200/200 |
| 1 | +0.140 | +0.170 | +0.075 | +0.000 | 200/200 |
| 2 | +0.100 | +0.125 | +0.050 | +0.000 | 200/200 |
| 3 | +0.115 | +0.095 | -0.130 | +0.000 | 200/200 |
| 4 | +0.050 | +0.060 | -0.100 | +0.000 | 199/200 |
| 5 | +0.040 | +0.025 | +0.005 | +0.000 | 200/200 |
| 6 | -0.035 | -0.050 | -0.015 | +0.000 | 200/200 |
| 7 | +0.055 | +0.070 | -0.050 | +0.000 | 199/200 |
| 8 | +0.075 | +0.090 | -0.010 | +0.000 | 199/200 |
| 9 | +0.015 | +0.010 | -0.145 | +0.000 | 200/200 |

**Mean gains** (N=10):
- self_current: +0.0530 +/- 0.0544
- self_wrong_trial: +0.0606 +/- 0.0608
- clone_current: -0.0385 +/- 0.0695
- group_a: +0.0000 +/- 0.0000

**Wilcoxon signed-rank tests** (exact, two-sided):

- **Self-current vs Self-wrong-trial**: mean diff = -0.0076, T = 20.0, p = 0.4922
- **Self-current vs Clone-current**: mean diff = +0.0915, T = 8.0, p = 0.0488
- **Self-wrong-trial vs Clone-current (KEY)**: mean diff = +0.0991, T = 8.0, p = 0.0488
- **Self-wrong-trial vs Group A**: mean diff = +0.0606, T = 9.0, p = 0.0645

### Interpretation

Self-wrong-trial gain > Clone gain: foreign-model identity matters beyond state-conditional mismatch. Supports feedback-contract specificity.

## VN Input

| Seed | Self-Current Gain | Self-Wrong-Trial Gain | Clone Gain | Group A Gain | Matched |
|------|-------------------|-----------------------|------------|--------------|----------|
| 0 | +0.070 | +0.055 | -0.115 | -0.035 | 200/200 |
| 1 | +0.180 | +0.130 | -0.015 | -0.065 | 200/200 |
| 2 | +0.235 | +0.220 | +0.165 | -0.030 | 200/200 |
| 3 | +0.090 | +0.055 | -0.110 | -0.010 | 200/200 |
| 4 | +0.170 | +0.170 | -0.020 | +0.045 | 200/200 |
| 5 | +0.185 | +0.205 | +0.170 | +0.095 | 200/200 |
| 6 | +0.140 | +0.115 | -0.095 | +0.050 | 200/200 |
| 7 | +0.125 | +0.150 | -0.090 | +0.005 | 200/200 |
| 8 | +0.135 | +0.120 | -0.055 | -0.005 | 200/200 |
| 9 | +0.175 | +0.160 | -0.010 | +0.030 | 200/200 |

**Mean gains** (N=10):
- self_current: +0.1505 +/- 0.0462
- self_wrong_trial: +0.1380 +/- 0.0525
- clone_current: -0.0175 +/- 0.0996
- group_a: +0.0080 +/- 0.0451

**Wilcoxon signed-rank tests** (exact, two-sided):

- **Self-current vs Self-wrong-trial**: mean diff = +0.0125, T = 12.0, p = 0.2500
- **Self-current vs Clone-current**: mean diff = +0.1680, T = 0.0, p = 0.0020
- **Self-wrong-trial vs Clone-current (KEY)**: mean diff = +0.1555, T = 0.0, p = 0.0020
- **Self-wrong-trial vs Group A**: mean diff = +0.1300, T = 0.0, p = 0.0020

### Interpretation

Self-wrong-trial gain > Clone gain: foreign-model identity matters beyond state-conditional mismatch. Supports feedback-contract specificity.

