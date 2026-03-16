# Supplementary Checks

## 7a. Train-Test Accuracy Gap (t=3)

| Task | Seed | Train Acc | Test Acc | Gap |
|------|------|-----------|----------|-----|
| static | 0 | 0.840 | 0.815 | +0.025 |
| static | 1 | 0.880 | 0.805 | +0.075 |
| static | 2 | 0.825 | 0.755 | +0.070 |
| static | 3 | 0.745 | 0.690 | +0.055 |
| static | 4 | 0.865 | 0.775 | +0.090 |
| static | 5 | 0.825 | 0.735 | +0.090 |
| static | 6 | 0.635 | 0.630 | +0.005 |
| static | 7 | 0.785 | 0.790 | -0.005 |
| static | 8 | 0.695 | 0.700 | -0.005 |
| static | 9 | 0.780 | 0.775 | +0.005 |
| vn | 0 | 0.915 | 0.865 | +0.050 |
| vn | 1 | 0.980 | 0.940 | +0.040 |
| vn | 2 | 0.930 | 0.895 | +0.035 |
| vn | 3 | 0.820 | 0.755 | +0.065 |
| vn | 4 | 0.890 | 0.885 | +0.005 |
| vn | 5 | 0.905 | 0.850 | +0.055 |
| vn | 6 | 0.730 | 0.700 | +0.030 |
| vn | 7 | 0.935 | 0.850 | +0.085 |
| vn | 8 | 0.845 | 0.740 | +0.105 |
| vn | 9 | 0.905 | 0.880 | +0.025 |

**Static mean gap**: +0.040 ± 0.038
**VN mean gap**: +0.050 ± 0.028

**WARNING**: Max gap = 10.5% — potential overfitting.

## 7b. Gradient Check Max Relative Error

- Static (seed=0): **6.57e-08**
- Variable-noise (seed=0): **1.48e-07**
- Acceptance threshold: 1e-4
- Both well below threshold (max = 1.48e-07).

## 7c. ReLU Dead Neuron Check (Hidden Layer 1)

| Seed | Baseline Dead | Group A Dead | Newly Dead |
|------|---------------|--------------|------------|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |
| 9 | 0 | 0 | 0 |

No neurons become permanently dead under Group A that were active under Baseline.

### Detailed H1 activation fractions (seed=0)

| Neuron | Baseline Active % | Group A Active % |
|--------|-------------------|------------------|
| h1_0 | 35.3% | 34.5% |
| h1_1 | 73.3% | 76.5% |
| h1_2 | 51.7% | 64.0% |
| h1_3 | 42.0% | 43.0% |
| h1_4 | 77.7% | 83.0% |
| h1_5 | 53.2% | 54.0% |
| h1_6 | 67.5% | 72.0% |
| h1_7 | 71.3% | 72.0% |
| h1_8 | 56.0% | 44.5% |
| h1_9 | 60.7% | 73.0% |
