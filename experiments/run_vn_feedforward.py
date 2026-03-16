"""VN Feedforward Controls: D/D'/D'' under variable noise.

Tests whether feedforward networks can accumulate evidence across
timesteps when receiving different noisy inputs — the key empirical
(not structural) test that recurrent self-reference is necessary.

Uses multiprocessing.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool
import csv

from src.network import RecurrentMLP, DeepFeedforwardMLP
from src.training import (
    generate_data_variable_noise, train_vn, train_deep_ff,
)


def _process_seed(seed):
    """Train and evaluate D, D', D'' under VN for one seed."""
    time_weights = [0.0, 0.2, 1.0]
    X_train_seq, y_train = generate_data_variable_noise(200, noise_level=0.5, seed=seed)
    X_test_seq, y_test = generate_data_variable_noise(200, noise_level=0.5, seed=seed + 1000)
    n_test = len(X_test_seq)

    results = {'seed': seed}

    # Baseline recurrent (for comparison)
    net = RecurrentMLP(seed=seed)
    train_vn(net, X_train_seq, y_train, epochs=500, lr=0.01, time_weights=time_weights)
    correct_t1, correct_t3 = 0, 0
    for i in range(n_test):
        outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
        if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1 += 1
        if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3 += 1
    results['baseline_acc_t1'] = correct_t1 / n_test
    results['baseline_acc_t3'] = correct_t3 / n_test
    results['baseline_gain'] = results['baseline_acc_t3'] - results['baseline_acc_t1']

    # Group D: feedforward (no recurrence), trained under VN
    net_d = RecurrentMLP(seed=seed)
    net_d.disable_recurrent_loop()
    train_vn(net_d, X_train_seq, y_train, epochs=500, lr=0.01, time_weights=time_weights)
    correct_t1, correct_t3 = 0, 0
    for i in range(n_test):
        outputs, _ = net_d.forward_sequence_vn(X_test_seq[i], T=3)
        if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1 += 1
        if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3 += 1
    results['d_acc_t1'] = correct_t1 / n_test
    results['d_acc_t3'] = correct_t3 / n_test
    results['d_gain'] = results['d_acc_t3'] - results['d_acc_t1']

    # Group D': parameter-matched feedforward with skip connection
    net_dp = RecurrentMLP(seed=seed, skip_connection=True)
    net_dp.disable_recurrent_loop()
    train_vn(net_dp, X_train_seq, y_train, epochs=500, lr=0.01, time_weights=time_weights)
    correct_t1, correct_t3 = 0, 0
    for i in range(n_test):
        outputs, _ = net_dp.forward_sequence_vn(X_test_seq[i], T=3)
        if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1 += 1
        if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3 += 1
    results['dp_acc_t1'] = correct_t1 / n_test
    results['dp_acc_t3'] = correct_t3 / n_test
    results['dp_gain'] = results['dp_acc_t3'] - results['dp_acc_t1']

    # Group D'': deep feedforward (6 layers)
    from src.training import generate_data
    # D'' needs single-input training, but under VN we evaluate per-timestep
    net_dpp = DeepFeedforwardMLP(seed=seed)
    # Train D'' on VN data — use t=3 targets only (single-pass network)
    X_train_flat = X_train_seq[:, 2, :]  # use t=3 input for training
    train_deep_ff(net_dpp, X_train_flat, y_train, epochs=500, lr=0.01)
    # Evaluate: feed each timestep's input, compare predictions
    correct_t1, correct_t3 = 0, 0
    for i in range(n_test):
        # t=1 input
        pred_t1 = np.argmax(net_dpp.forward(X_test_seq[i, 0]))
        # t=3 input
        pred_t3 = np.argmax(net_dpp.forward(X_test_seq[i, 2]))
        true = np.argmax(y_test[i])
        if pred_t1 == true: correct_t1 += 1
        if pred_t3 == true: correct_t3 += 1
    results['dpp_acc_t1'] = correct_t1 / n_test
    results['dpp_acc_t3'] = correct_t3 / n_test
    results['dpp_gain'] = results['dpp_acc_t3'] - results['dpp_acc_t1']

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_workers = min(os.cpu_count(), 10)
    print(f"Workers: {n_workers}\n")

    with Pool(n_workers) as pool:
        results = pool.map(_process_seed, range(10))

    print("Results:")
    print(f"{'Seed':>4} {'Baseline':>10} {'D':>10} {'D_prime':>10} {'D_dblprime':>12}")
    for r in results:
        print(f"{r['seed']:4d} {r['baseline_gain']:+10.3f} {r['d_gain']:+10.3f} "
              f"{r['dp_gain']:+10.3f} {r['dpp_gain']:+12.3f}")

    print(f"\nMeans:")
    for g in ['baseline', 'd', 'dp', 'dpp']:
        gains = [r[f'{g}_gain'] for r in results]
        print(f"  {g:>10}: {np.mean(gains):+.4f} +/- {np.std(gains):.4f}")

    # Save
    fieldnames = ['seed', 'baseline_acc_t1', 'baseline_acc_t3', 'baseline_gain',
                  'd_acc_t1', 'd_acc_t3', 'd_gain',
                  'dp_acc_t1', 'dp_acc_t3', 'dp_gain',
                  'dpp_acc_t1', 'dpp_acc_t3', 'dpp_gain']
    with open('results/vn_feedforward_controls.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved: results/vn_feedforward_controls.csv")
