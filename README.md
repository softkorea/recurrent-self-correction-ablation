# Recurrent Self-Correction Ablation Experiment

**Can a neural network correct its own mistakes using output feedback — and does removing that feedback destroy self-correction?**

This experiment demonstrates that a minimal recurrent neural network (35 neurons) learns to iteratively correct its predictions through an output-to-hidden feedback loop. Systematically ablating or corrupting this loop eliminates self-correction, providing evidence that recurrent self-reference is a necessary mechanism — not a byproduct of extra parameters.

Paper: [`paper.txt`](paper.txt)
Repository: https://github.com/softkorea/recurrent-self-correction-ablation

## Key Results

| Group | acc (t=1) | acc (t=3) | Correction Gain | Interpretation |
|-------|-----------|-----------|-----------------|----------------|
| **Baseline** | 0.698 | 0.740 | **+0.042** | Self-correction occurs |
| A (Recurrent Cut) | 0.698 | 0.698 | 0.000 | Correction completely lost |
| B1 (Random Cut) | 0.515 | 0.511 | -0.004 | General damage |
| C1 (Shuffled Feedback) | 0.698 | 0.634 | **-0.064** | Wrong self-reference is *worse* than none |
| **C2 (Clone Feedback)** | 0.698 | 0.623 | **-0.075** | Another model's valid output = degradation |
| D' (Param-matched FF) | 0.822 | 0.822 | 0.000 | Extra params help FF, not correction |

- Baseline 95% CI: **[+0.023, +0.059]** (does not contain zero)
- All ablation groups significantly different from Baseline (Holm-Bonferroni corrected p < 0.025)
- C2 (Clone Feedback) strongly argues for dependency on *own* output trajectory, not just any valid signal (C1 vs C2 not statistically distinguishable, Wilcoxon p = 0.695)
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

## Experimental Groups (8 conditions)

| Group | Description | Purpose |
|-------|-------------|---------|
| **Baseline** | Full recurrent network | Reference |
| **A** | Recurrent weights zeroed | Does feedback matter? |
| **B1** | Random weights zeroed (x30) | Is it about *which* weights? |
| **B2** | Structured cut (h2→output, 50 params) | General path ablation control |
| **C1** | Feedback permuted (x30) | Information vs self-reference |
| **C2** | Clone model's output as feedback | OOD criticism control |
| **D** | Feedforward only (no recurrence in training) | Recurrence necessity |
| **D'** | Param-matched FF (skip connection) | Parameter count control |

## Robustness (Hyperparameter Sweep)

Swept 80 configurations (w1 x w2 x tau) x 10 models = 800 experiments:

| Parameter | Emergence Range | Failure Region |
|-----------|----------------|----------------|
| w1 (t=1 loss weight) | <= 0.2 (75-95%) | 0.3 (10%) |
| tau (temperature) | 1.0-3.0 (69-88%) | 5.0 (25%) |
| w2 (t=2 loss weight) | All values (60-75%) | — |

Our reported configuration (w1=0.0, w2=0.2, tau=2.0) ranks **13th/80** — a mid-range setting, not a cherry-picked optimum.

## Project Structure

```
├── analyze_results.py                    # Verify all reported numbers from raw data
├── paper.txt                             # Full manuscript
├── CLAUDE.md                             # Experiment constraints & rules
├── plan.md                               # Detailed experiment plan
├── REVIEW_PACKAGE.md                     # Reviewer checklist & self-assessment
├── REFERENCES_URLS.md                    # Reference verification URLs
├── STATISTICAL_JUSTIFICATION.md          # Statistical methodology notes
├── src/
│   ├── network.py                        # RecurrentMLP (35 neurons)
│   ├── training.py                       # 3-step unroll backprop + SGD
│   ├── ablation.py                       # 8 ablation methods (incl. clone feedback)
│   ├── metrics.py                        # Accuracy, gain, ECE, norms, neuron importance
│   └── visualize.py                      # All plot generation
├── tests/
│   ├── test_network.py                   # Network structure tests
│   ├── test_training.py                  # Training + gradient check
│   ├── test_ablation.py                  # Ablation correctness
│   ├── test_metrics.py                   # Metric validity
│   └── test_clone_feedback.py            # Clone feedback (C2) tests
├── experiments/
│   ├── run_experiment.py                 # Full experiment (10 models x 8 groups x 6 noise levels)
│   ├── run_c2_experiment.py              # Group C2 clone feedback experiment
│   └── sweep_hyperparams.py              # Hyperparameter robustness sweep
├── results/
│   ├── REPORT.md                         # Main experiment report
│   ├── REPORT_C2.md                      # Clone Feedback analysis
│   ├── REPORT_SWEEP.md                   # Hyperparameter sweep analysis
│   ├── raw_metrics.csv                   # Full raw data (3,960 rows)
│   ├── neuron_importance.csv             # Per-neuron importance scores
│   ├── sweep_hyperparams.csv             # Sweep raw data (800 rows)
│   ├── ablation_comparison.png           # Group comparison bar chart
│   ├── noise_sweep_curve.png             # Noise level x group curves
│   ├── accuracy_distribution.png         # Random cut distribution
│   ├── neuron_importance_heatmap.png     # Intelligence vs correction scatter
│   ├── network_map.png                   # Full network visualization
│   └── sweep_heatmap_tau*.png            # Sweep heatmaps (5 plots)
└── results_archive/                      # Earlier bilingual report versions
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

### Verify Reported Numbers

```bash
python analyze_results.py
```

Reads `results/raw_metrics.csv` and reproduces every statistic cited in the paper — group means, bootstrap CIs, Wilcoxon p-values, Holm-Bonferroni corrections, noise sweep, and recurrent contribution norms. No experiments are re-run; this script only performs analysis on existing data.

### Reproduce All Results

> **Warning**: Each experiment script overwrites `results/raw_metrics.csv` (and other result files). If you want to preserve existing results, back up the `results/` directory before re-running.

```bash
# Main experiment (~1 hour on modern CPU)
python experiments/run_experiment.py

# Clone feedback experiment
python experiments/run_c2_experiment.py

# Hyperparameter sweep
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
4. **Own output trajectory matters**: Clone feedback (Group C2) — another trained model's well-formed output — degrades performance at least as severely as shuffled feedback, strongly arguing for dependency on *self*
5. **Not a capacity effect**: Extra parameters (Group D') improve feedforward accuracy but cannot produce self-correction
6. **Emergence requires incentive**: Time-weighted loss (freeing t=1 from accuracy pressure) and temperature scaling are the key enablers, not network size

## License

MIT
