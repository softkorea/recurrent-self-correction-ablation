# Scale Verification Report

## Configuration
- Hidden widths: [10, 20, 45, 245]
- Seeds: 0-9 (targets), 100-109 (donors)
- Epochs: 500, LR: 0.01, tau: 2.0
- Noise: 0.5

## Static Results

| Hidden Width | Baseline | Group A | Group C1 | Group C2 |
|---|---|---|---|---|
| 10 | +0.0530 +/- 0.0442 | +0.0000 +/- 0.0000 | -0.0437 +/- 0.0390 | -0.0295 +/- 0.0642 |
| 20 | +0.0540 +/- 0.0419 | +0.0000 +/- 0.0000 | -0.0959 +/- 0.0348 | -0.0910 +/- 0.0676 |
| 45 | +0.0415 +/- 0.0435 | +0.0000 +/- 0.0000 | -0.1134 +/- 0.0267 | -0.0225 +/- 0.0478 |
| 245 | +0.0325 +/- 0.0447 | +0.0000 +/- 0.0000 | -0.2590 +/- 0.0484 | -0.0545 +/- 0.0708 |

## Variable-Noise Results

| Hidden Width | Baseline | Group A | Group C1 | Group C2 |
|---|---|---|---|---|
| 10 | +0.1460 +/- 0.0530 | -0.0040 +/- 0.0196 | -0.0732 +/- 0.0681 | -0.0270 +/- 0.0910 |
| 20 | +0.1830 +/- 0.0726 | -0.0160 +/- 0.0379 | -0.1189 +/- 0.0702 | -0.0310 +/- 0.1107 |
| 45 | +0.1540 +/- 0.0351 | -0.0050 +/- 0.0214 | -0.1621 +/- 0.0554 | +0.0320 +/- 0.0627 |
| 245 | +0.1485 +/- 0.0240 | +0.0000 +/- 0.0325 | -0.2850 +/- 0.0553 | +0.0595 +/- 0.0544 |

## Fake Mirror Effect Persistence

### Static

- group_c1 gain < 0 at all scales? **YES**
- group_c2 gain < 0 at all scales? **YES**

### VN

- group_c1 gain < 0 at all scales? **YES**
- group_c2 gain < 0 at all scales? **NO (exceptions: h=45 (gain=+0.0320), h=245 (gain=+0.0595))**

