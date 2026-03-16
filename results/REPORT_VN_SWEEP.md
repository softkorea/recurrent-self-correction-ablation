# VN Hyperparameter Sweep Report

## Configuration
- w1: [0.0, 0.1, 0.2, 0.3]
- tau: [1.0, 1.5, 2.0, 3.0]
- w2: 0.2 (fixed)
- Seeds: 0-9 (10 models per config)
- Total runs: 160

## Emergence Analysis

Emergence criterion: mean gain > 0 AND >= 60% seeds positive.

### w1 x tau Emergence Matrix

| w1 \ tau | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|
| **0.0** | +0.159 (100%) | +0.154 (100%) | +0.146 (100%) | +0.109 (90%) |
| **0.1** | +0.150 (100%) | +0.143 (100%) | +0.136 (100%) | +0.112 (100%) |
| **0.2** | +0.146 (100%) | +0.146 (100%) | +0.131 (100%) | +0.115 (100%) |
| **0.3** | +0.139 (100%) | +0.136 (100%) | +0.130 (100%) | +0.111 (100%) |

**Overall emergence rate: 16/16 (100%)**

### Per-w1 Emergence Rate

| w1 | Emergence |
|---|---|
| 0.0 | 4/4 |
| 0.1 | 4/4 |
| 0.2 | 4/4 |
| 0.3 | 4/4 |

### Per-tau Emergence Rate

| tau | Emergence |
|---|---|
| 1.0 | 4/4 |
| 1.5 | 4/4 |
| 2.0 | 4/4 |
| 3.0 | 4/4 |

**Best config:** w1=0.0, tau=1.0, mean gain=+0.1590

## Comparison: Static vs VN

Static sweep (from existing results): 54/80 configs = 68% emergence
VN sweep (this experiment): 16/16 configs = 100% emergence
