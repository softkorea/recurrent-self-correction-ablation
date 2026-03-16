"""WS0: Variable-Noise Task 실험.

Static-input tautology를 해소하는 핵심 실험.
x_t = prototype_k + ε_t (독립 노이즈), T=3.

Groups: Baseline, A (recurrent cut), C1 (shuffled feedback), C2 (clone feedback)
10 models × 6 noise levels = 60 rows per group.

Usage:
    python experiments/run_variable_noise.py
"""

import sys
import os
import csv
import time
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import generate_data_variable_noise, train_vn
from src.metrics import compute_all_metrics_vn, compute_all_metrics_with_clone_vn
from src.ablation import deep_copy_weights, restore_weights


def train_model_vn(seed, noise_level=0.5, n_samples=200, epochs=500,
                   lr=0.01, tau=2.0, time_weights=None):
    """VN 학습된 RecurrentMLP 반환."""
    if time_weights is None:
        time_weights = [0.0, 0.2, 1.0]
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=tau)
    X_seq, y = generate_data_variable_noise(
        n_samples, noise_level=noise_level, T=3, seed=seed)
    train_vn(net, X_seq, y, epochs=epochs, lr=lr,
             time_weights=time_weights)
    return net


def evaluate_group(net, X_seq_test, y_test, group_name, seed_model,
                   noise_level, seed_ablation=0, clone_net=None):
    """그룹별 메트릭 계산 → CSV row dict."""
    saved = deep_copy_weights(net)

    if group_name == 'Baseline':
        metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
    elif group_name == 'A':
        net.W_rec[:] = 0.0
        metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
    elif group_name == 'C1':
        net.enable_scrambled_feedback(seed=seed_ablation)
        metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
        net.disable_scrambled_feedback()
    elif group_name == 'C2':
        assert clone_net is not None
        metrics = compute_all_metrics_with_clone_vn(
            net, clone_net, X_seq_test, y_test)
    else:
        raise ValueError(f"Unknown group: {group_name}")

    restore_weights(net, saved)

    return {
        'seed_model': seed_model,
        'group': group_name,
        'seed_ablation': seed_ablation,
        'noise_level': noise_level,
        'acc_t1': f"{metrics['acc_t1']:.6f}",
        'acc_t2': f"{metrics['acc_t2']:.6f}",
        'acc_t3': f"{metrics['acc_t3']:.6f}",
        'gain': f"{metrics['gain']:.6f}",
        'ece': f"{metrics['ece']:.6f}",
        'r_norm': f"{metrics['r_norm']:.6f}",
        'delta_norm': f"{metrics['delta_norm']:.6f}",
    }


def main():
    seeds = list(range(10))
    donor_seeds = list(range(100, 110))
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_test = 200
    n_c1_repeats = 30

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    csv_path = os.path.join(results_dir, 'variable_noise_metrics.csv')

    fieldnames = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece',
                  'r_norm', 'delta_norm']

    all_rows = []
    total_start = time.time()

    # Train all models
    print("=" * 60)
    print("WS0: Variable-Noise Task Experiment")
    print("=" * 60)

    # Phase 1: Train target models
    print("\n[Phase 1] Training 10 target models (VN)...")
    target_models = {}
    for s in seeds:
        t0 = time.time()
        net = train_model_vn(seed=s, noise_level=0.5)
        target_models[s] = net
        print(f"  seed={s}: {time.time()-t0:.1f}s")

    # Phase 2: Train donor models (for C2)
    print("\n[Phase 2] Training 10 donor models (VN)...")
    donor_models = {}
    for ds in donor_seeds:
        t0 = time.time()
        net = train_model_vn(seed=ds, noise_level=0.5)
        donor_models[ds] = net
        print(f"  seed={ds}: {time.time()-t0:.1f}s")

    # Phase 3: Evaluate across noise levels
    print("\n[Phase 3] Evaluation...")

    for s_idx, s in enumerate(seeds):
        net = target_models[s]
        clone_net = donor_models[donor_seeds[s_idx]]

        for noise in noise_levels:
            # Generate test data for this noise level
            X_test, y_test = generate_data_variable_noise(
                n_test, noise_level=noise, T=3, seed=1000 + s)

            # Baseline
            row = evaluate_group(net, X_test, y_test, 'Baseline', s, noise)
            all_rows.append(row)

            # Group A (recurrent cut)
            row = evaluate_group(net, X_test, y_test, 'A', s, noise)
            all_rows.append(row)

            # Group C1 (shuffled feedback) x 30 repeats → average
            c1_gains = []
            c1_acc_t1s = []
            c1_acc_t3s = []
            for rep in range(n_c1_repeats):
                row_c1 = evaluate_group(
                    net, X_test, y_test, 'C1', s, noise,
                    seed_ablation=rep)
                c1_gains.append(float(row_c1['gain']))
                c1_acc_t1s.append(float(row_c1['acc_t1']))
                c1_acc_t3s.append(float(row_c1['acc_t3']))
            # Store average of C1 repeats
            avg_row = {
                'seed_model': s,
                'group': 'C1',
                'seed_ablation': 'avg30',
                'noise_level': noise,
                'acc_t1': f"{np.mean(c1_acc_t1s):.6f}",
                'acc_t2': f"{0.0:.6f}",  # Not tracked for averaged
                'acc_t3': f"{np.mean(c1_acc_t3s):.6f}",
                'gain': f"{np.mean(c1_gains):.6f}",
                'ece': f"{0.0:.6f}",
                'r_norm': f"{0.0:.6f}",
                'delta_norm': f"{0.0:.6f}",
            }
            all_rows.append(avg_row)

            # Group C2 (clone feedback)
            row = evaluate_group(
                net, X_test, y_test, 'C2', s, noise,
                clone_net=clone_net)
            all_rows.append(row)

        print(f"  Model {s}: done ({len(noise_levels)} noise levels × 4 groups)")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Done! {len(all_rows)} rows written to {csv_path}")
    print(f"Total time: {total_time:.0f}s")

    # Summary at noise=0.5
    print(f"\n{'='*60}")
    print("Summary at noise=0.5 (model-level means):")
    print(f"{'='*60}")

    for group in ['Baseline', 'A', 'C1', 'C2']:
        gains = [float(r['gain']) for r in all_rows
                 if r['group'] == group and float(r['noise_level']) == 0.5]
        if gains:
            mean_g = np.mean(gains)
            std_g = np.std(gains)
            print(f"  {group:10s}: gain = {mean_g:+.4f} ± {std_g:.4f}"
                  f"  (N={len(gains)})")


if __name__ == '__main__':
    main()
