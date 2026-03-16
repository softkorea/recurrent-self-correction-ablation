# Training Dynamics: When Does the Feedback Contract Form?

## STATIC

| Epoch | Baseline Gain | Group A | C1 | C2 |
|-------|---------------|---------|------|------|
| 0 | -0.0065 | +0.0000 | +0.0175 | +0.0025 |
| 10 | -0.0085 | +0.0000 | +0.0105 | -0.0010 |
| 25 | -0.0045 | +0.0000 | +0.0050 | +0.0005 |
| 50 | +0.0090 | +0.0000 | -0.0015 | -0.0110 |
| 75 | +0.0385 | +0.0000 | +0.0000 | -0.0125 |
| 100 | +0.0365 | +0.0000 | -0.0055 | -0.0235 |
| 150 | +0.0260 | +0.0000 | -0.0175 | -0.0485 |
| 200 | +0.0205 | +0.0000 | -0.0330 | -0.0485 |
| 300 | +0.0165 | +0.0000 | -0.0285 | -0.0500 |
| 400 | +0.0395 | +0.0000 | -0.0410 | -0.0505 |
| 500 | +0.0530 | +0.0000 | -0.0420 | -0.0385 |

## VN

| Epoch | Baseline Gain | Group A | C1 | C2 |
|-------|---------------|---------|------|------|
| 0 | -0.0205 | -0.0030 | -0.0070 | +0.0195 |
| 10 | -0.0245 | -0.0010 | -0.0095 | +0.0090 |
| 25 | -0.0140 | -0.0115 | -0.0195 | +0.0075 |
| 50 | +0.0030 | -0.0060 | -0.0195 | +0.0095 |
| 75 | +0.0310 | +0.0110 | -0.0125 | +0.0105 |
| 100 | +0.0375 | +0.0115 | -0.0215 | +0.0030 |
| 150 | +0.0350 | -0.0045 | -0.0425 | -0.0185 |
| 200 | +0.0495 | -0.0060 | -0.0420 | -0.0175 |
| 300 | +0.0710 | -0.0030 | -0.0655 | -0.0215 |
| 400 | +0.1180 | +0.0195 | -0.0615 | -0.0195 |
| 500 | +0.1505 | +0.0080 | -0.0845 | -0.0175 |

## Onset Detection

  static seed=0: onset at epoch 50
  static seed=1: onset at epoch 10
  static seed=2: onset at epoch 0
  static seed=3: onset at epoch 25
  static seed=4: onset at epoch 0
  static seed=5: onset at epoch 150
  static seed=6: onset at epoch 0
  static seed=7: onset at epoch 10
  static seed=8: onset at epoch 75
  static seed=9: onset at epoch 0

  vn seed=0: onset at epoch 10
  vn seed=1: onset at epoch 75
  vn seed=2: onset at epoch 200
  vn seed=3: onset at epoch 100
  vn seed=4: onset at epoch 0
  vn seed=5: onset at epoch 150
  vn seed=6: onset at epoch 75
  vn seed=7: onset at epoch 200
  vn seed=8: onset at epoch 75
  vn seed=9: onset at epoch 0

