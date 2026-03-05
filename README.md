# Recurrent Self-Correction Ablation Experiment

**Can a neural network correct its own mistakes using output feedback — and does removing that feedback destroy self-correction?**

This experiment demonstrates that a minimal recurrent neural network (35 neurons) learns to iteratively correct its predictions through an output-to-hidden feedback loop. Systematically ablating or corrupting this loop eliminates self-correction, providing evidence that recurrent self-reference is a necessary mechanism — not a byproduct of extra parameters.

## Key Results

| Group | acc (t=1) | acc (t=3) | Correction Gain | Interpretation |
|-------|-----------|-----------|-----------------|----------------|
| **Baseline** | 0.698 | 0.740 | **+0.042** | Self-correction occurs |
| A (Recurrent Cut) | 0.698 | 0.698 | 0.000 | Correction completely lost |
| B1 (Random Cut) | 0.515 | 0.511 | -0.004 | General damage |
| C1 (Shuffled Feedback) | 0.698 | 0.634 | **-0.064** | Wrong self-reference is *worse* than none |
| D' (Param-matched FF) | 0.822 | 0.822 | 0.000 | Extra params help FF, not correction |

- Baseline 95% CI: **[+0.023, +0.059]** (does not contain zero)
- All ablation groups significantly different from Baseline (Holm-Bonferroni corrected p < 0.01)
- Robust across **54/80 (68%)** hyperparameter configurations

## Architecture

```
Input(10) → Hidden1(10) → Hidden2(10) → Output(5)
                ↑                            │
                └──── tanh(output/τ) ────────┘
                     recurrent feedback
```

- **35 neurons total** — small enough for complete human inspection
- **Pure NumPy** — no PyTorch/TensorFlow, every weight is directly accessible
- **3-step unroll**: same static input presented 3 times; network corrects via feedback

## Experimental Groups (7 conditions)

| Group | Description | Purpose |
|-------|-------------|---------|
| **Baseline** | Full recurrent network | Reference |
| **A** | Recurrent weights zeroed | Does feedback matter? |
| **B1** | Random weights zeroed (×30) | Is it about *which* weights? |
| **B2** | Structured cut (h1→h2) | General path ablation control |
| **C1** | Feedback permuted (×30) | Information vs self-reference |
| **D** | Feedforward only (no recurrence in training) | Recurrence necessity |
| **D'** | Param-matched FF (skip connection) | Parameter count control |

## Robustness (Hyperparameter Sweep)

Swept 80 configurations (w1 × w2 × τ) × 10 models = 800 experiments:

| Parameter | Emergence Range | Failure Region |
|-----------|----------------|----------------|
| w1 (t=1 loss weight) | ≤ 0.2 (75-95%) | 0.3 (10%) |
| τ (temperature) | 1.0-3.0 (69-88%) | 5.0 (25%) |
| w2 (t=2 loss weight) | All values (60-75%) | — |

Our reported configuration (w1=0.0, w2=0.2, τ=2.0) ranks **13th/80** — a mid-range setting, not a cherry-picked optimum.

## Project Structure

```
├── CLAUDE.md                          # Experiment constraints & rules
├── plan.md                            # Detailed experiment plan (Korean)
├── src/
│   ├── network.py                     # RecurrentMLP (35 neurons)
│   ├── training.py                    # 3-step unroll backprop + SGD
│   ├── ablation.py                    # 7 ablation methods
│   ├── metrics.py                     # Accuracy, gain, ECE, norms
│   └── visualize.py                   # All plot generation
├── tests/
│   ├── test_network.py                # Network structure tests
│   ├── test_training.py               # Training + gradient check
│   ├── test_ablation.py               # Ablation correctness
│   └── test_metrics.py                # Metric validity
├── experiments/
│   ├── run_experiment.py              # Full experiment (10 models × 7 groups × 6 noise levels)
│   └── sweep_hyperparams.py           # Hyperparameter robustness sweep
└── results/
    ├── REPORT.md                      # Main experiment report (Korean)
    ├── REPORT.en.md                   # Main experiment report (English)
    ├── REPORT_SWEEP.md                # Hyperparameter sweep analysis (Korean)
    ├── REPORT_SWEEP.en.md             # Hyperparameter sweep analysis (English)
    ├── raw_metrics.csv                # Full raw data (3,900 rows)
    ├── sweep_hyperparams.csv          # Sweep raw data (800 rows)
    ├── ablation_comparison.png        # Group comparison bar chart
    ├── noise_sweep_curve.png          # Noise level × group curves
    ├── accuracy_distribution.png      # Random cut distribution
    ├── neuron_importance_heatmap.png   # Intelligence vs correction scatter
    ├── network_map.png                # Full network visualization
    └── sweep_heatmap_tau*.png         # Sweep heatmaps (5 plots)
```

## Quick Start

### Requirements

```bash
pip install numpy matplotlib pytest
# Optional: pip install tqdm (for progress bars)
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Full Experiment

```bash
python experiments/run_experiment.py
```

### Run Hyperparameter Sweep

```bash
python experiments/sweep_hyperparams.py
```

## Design Principles

- **TDD workflow**: tests written before implementation
- **Full transparency**: 35 neurons, every weight inspectable
- **Reproducibility**: all randomness seeded
- **Statistical rigor**: 10 independent models, bootstrap 95% CI, Holm-Bonferroni correction
- **No deep learning frameworks**: pure NumPy for complete control

## Key Findings

1. **Self-correction is real**: Baseline gain = +0.042, significantly above zero
2. **Feedback is necessary**: Removing recurrence (Group A) eliminates all correction
3. **Self-reference, not just information**: Shuffled feedback (Group C1) is *worse* than no feedback — the network actively uses its own output, and wrong information misleads it
4. **Not a capacity effect**: Extra parameters (Group D') improve feedforward accuracy but cannot produce self-correction
5. **Emergence requires incentive**: Time-weighted loss (freeing t=1 from accuracy pressure) is the key enabler, not network size

## License

MIT
