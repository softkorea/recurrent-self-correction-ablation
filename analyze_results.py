"""Reproduce all reported numbers from raw_metrics.csv.

Reads results/raw_metrics.csv and prints every statistic cited in the paper,
enabling independent verification without re-running experiments.

Usage:
    python analyze_results.py
"""

import csv
import itertools
import math
import os
import random

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "raw_metrics.csv")
NOISE_PRIMARY = 0.5
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 999  # match main experiment scripts


def load_csv():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def model_means(rows, group, noise, metric="gain"):
    """Compute per-model mean of `metric` for a given group and noise level."""
    by_model = {}
    for r in rows:
        if r["group"] != group or abs(float(r["noise_level"]) - noise) > 1e-9:
            continue
        seed = int(r["seed_model"])
        by_model.setdefault(seed, []).append(float(r[metric]))
    return {s: sum(v) / len(v) for s, v in sorted(by_model.items())}


def mean_sd(values):
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))


def bootstrap_ci(values, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def wilcoxon_exact_twosided(diffs):
    """Exact two-sided Wilcoxon signed-rank test for small N."""
    abs_diffs = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    abs_diffs.sort(key=lambda x: x[0])
    n = len(abs_diffs)
    if n == 0:
        return 1.0, 0

    # Assign ranks with midrank averaging for ties
    abs_vals = [v for v, _ in abs_diffs]
    ranks = list(range(1, n + 1))
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_vals[j] == abs_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = sum(ranks[i:j]) / (j - i)
            for k in range(i, j):
                ranks[k] = avg_rank
        i = j
    signs = [s for _, s in abs_diffs]

    t_plus = sum(r for r, s in zip(ranks, signs) if s > 0)
    t_minus = sum(r for r, s in zip(ranks, signs) if s < 0)
    t_stat = min(t_plus, t_minus)

    # Enumerate all 2^n sign assignments
    total = 2 ** n
    count = 0
    for bits in range(total):
        wp = 0
        wm = 0
        for i in range(n):
            if bits & (1 << i):
                wp += ranks[i]
            else:
                wm += ranks[i]
        if min(wp, wm) <= t_stat:
            count += 1

    return count / total, t_stat


def holm_bonferroni(pvals_labels, alpha=0.05):
    """Apply Holm-Bonferroni correction. Returns list of (label, raw_p, adjusted_p, reject)."""
    m = len(pvals_labels)
    sorted_items = sorted(pvals_labels, key=lambda x: x[1])
    results = []
    cummax = 0.0
    for i, (label, p) in enumerate(sorted_items):
        adj = p * (m - i)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        results.append((label, p, cummax, cummax < alpha))
    return results


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    rows = load_csv()
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # ── Section 3.1: Emergence ──
    section("3.1 Emergence of Self-Correction (noise=0.5)")

    groups = ["Baseline", "A", "B1", "B2", "C1", "C2", "D", "D'", "D''"]
    model_data = {}
    for g in groups:
        mm = model_means(rows, g, NOISE_PRIMARY, "gain")
        model_data[g] = list(mm.values())

    for g in groups:
        vals = model_data[g]
        m, sd = mean_sd(vals)
        print(f"  {g:10s}  gain = {m:+.3f} ± {sd:.3f}  (N={len(vals)})")

    # Baseline CI
    bl_vals = model_data["Baseline"]
    lo, hi = bootstrap_ci(bl_vals)
    print(f"\n  Baseline 95% CI: [{lo:+.3f}, {hi:+.3f}]")

    # Accuracy at t=1 and t=3
    section("Accuracy at t=1 and t=3 (noise=0.5)")
    for g in groups:
        t1 = model_means(rows, g, NOISE_PRIMARY, "acc_t1")
        t3 = model_means(rows, g, NOISE_PRIMARY, "acc_t3")
        t1_vals = list(t1.values())
        t3_vals = list(t3.values())
        m1, s1 = mean_sd(t1_vals)
        m3, s3 = mean_sd(t3_vals)
        print(f"  {g:10s}  t1 = {m1:.3f} ± {s1:.3f}   t3 = {m3:.3f} ± {s3:.3f}")

    # ── Section 3.2-3.3: Wilcoxon tests ──
    section("Wilcoxon Signed-Rank Tests vs Baseline (noise=0.5)")

    comparisons = ["A", "B1", "C1", "C2", "D", "D'", "D''"]
    pvals_primary = []
    for g in comparisons:
        diffs = [model_data["Baseline"][i] - model_data[g][i]
                 for i in range(len(model_data["Baseline"]))]
        p, t = wilcoxon_exact_twosided(diffs)
        pvals_primary.append((g, p))
        print(f"  Baseline vs {g:5s}:  T = {t:5.1f},  raw p = {p:.4f}")

    print(f"\n  Holm-Bonferroni correction (m={len(pvals_primary)}):")
    hb = holm_bonferroni(pvals_primary)
    for label, raw_p, adj_p, reject in hb:
        print(f"    {label:5s}:  raw p = {raw_p:.4f}  →  adjusted p = {adj_p:.4f}  {'*' if reject else ''}")

    # ── C1 vs A, C2 vs A direct tests ──
    section("Direct Tests: C1 vs A and C2 vs A (noise=0.5)")

    for g in ["C1", "C2"]:
        diffs = [model_data[g][i] - model_data["A"][i]
                 for i in range(len(model_data["A"]))]
        p, t = wilcoxon_exact_twosided(diffs)
        lo, hi = bootstrap_ci(diffs)
        print(f"  {g} vs A:  T = {t:.1f},  p = {p:.4f},  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # C1 vs C2
    diffs_c1c2 = [model_data["C1"][i] - model_data["C2"][i]
                  for i in range(len(model_data["C1"]))]
    p_c1c2, t_c1c2 = wilcoxon_exact_twosided(diffs_c1c2)
    print(f"  C1 vs C2: T = {t_c1c2:.1f},  p = {p_c1c2:.3f}")

    # ── C1 and C2 standalone CIs ──
    section("Standalone CIs for C1 and C2 (noise=0.5)")
    for g in ["C1", "C2"]:
        lo, hi = bootstrap_ci(model_data[g])
        m, sd = mean_sd(model_data[g])
        print(f"  {g}:  mean = {m:+.3f},  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ── Noise sweep ──
    section("Noise Sweep: Baseline Gain by Noise Level")
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for nl in noise_levels:
        mm = model_means(rows, "Baseline", nl, "gain")
        vals = list(mm.values())
        if vals:
            m, sd = mean_sd(vals)
            print(f"  noise={nl:.1f}:  gain = {m:+.3f} ± {sd:.3f}")

    # ── Recurrent contribution norm ──
    section("Recurrent Contribution Norm (noise=0.5)")
    for g in ["Baseline", "A", "D'", "D''"]:
        mm = model_means(rows, g, NOISE_PRIMARY, "r_norm")
        vals = list(mm.values())
        m, sd = mean_sd(vals)
        print(f"  {g:10s}  r_norm = {m:.3f} ± {sd:.3f}")

    print("\n" + "=" * 70)
    print("  All numbers above should match those reported in paper.txt")
    print("  Note: values at exact rounding boundaries (e.g. 0.7395, 0.0295)")
    print("  may show ±1 in the last decimal digit vs. paper due to")
    print("  floating-point representation differences in CSV parsing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
