"""Experiment A: Same-Model Wrong-Trajectory Substitution.

Tests whether feedback-contract specificity is model-specific (clone hurts more
than wrong-trial self) or just state-conditional (both hurt equally).

Uses multiprocessing for parallel seed evaluation.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool
from collections import defaultdict
import csv

from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn,
)
from src.ablation import (
    forward_sequence_with_clone, forward_sequence_with_clone_vn,
    ablate_recurrent, deep_copy_weights, restore_weights,
)


def _forward_with_wrong_trial_feedback(net, x_current, y_wrong_t1, T=3):
    """Forward pass using another trial's t=1 output as feedback at t>=2.

    t=1: normal (no feedback)
    t=2: inject y_wrong_t1 as _prev_output
    t=3: use t=2's own output as feedback (normal from there)
    """
    net.reset_state()
    outputs = []

    # t=1: no feedback
    y1 = net.forward(x_current)
    outputs.append(y1.copy())

    # t=2: inject wrong-trial t=1 output
    net._prev_output = y_wrong_t1.copy()
    net._has_feedback = True
    y2 = net.forward(x_current)
    outputs.append(y2.copy())

    # t=3: normal self-feedback from t=2
    y3 = net.forward(x_current)
    outputs.append(y3.copy())

    return outputs


def _forward_with_wrong_trial_feedback_vn(net, x_seq, y_wrong_t1, T=3):
    """VN version of wrong-trial feedback injection."""
    x_seq = np.asarray(x_seq, dtype=np.float64)
    net.reset_state()
    outputs = []

    y1 = net.forward(x_seq[0])
    outputs.append(y1.copy())

    net._prev_output = y_wrong_t1.copy()
    net._has_feedback = True
    y2 = net.forward(x_seq[1])
    outputs.append(y2.copy())

    y3 = net.forward(x_seq[2])
    outputs.append(y3.copy())

    return outputs


def _process_seed_static(seed):
    """Process one seed for static input."""
    net = RecurrentMLP(seed=seed)
    X_train, y_train = generate_data(200, noise_level=0.5, seed=seed)
    X_test, y_test = generate_data(200, noise_level=0.5, seed=seed + 1000)
    train(net, X_train, y_train, epochs=500, lr=0.01, time_weights=[0.0, 0.2, 1.0])

    clone = RecurrentMLP(seed=seed + 100)
    X_clone_train, y_clone_train = generate_data(200, noise_level=0.5, seed=seed + 100)
    train(clone, X_clone_train, y_clone_train, epochs=500, lr=0.01,
          time_weights=[0.0, 0.2, 1.0])

    n_test = len(X_test)

    # Step 1: Get self-current outputs for all trials
    self_outputs_all = []
    for i in range(n_test):
        outputs, _ = net.forward_sequence(X_test[i], T=3)
        self_outputs_all.append(outputs)

    # Step 2: Build match index (predicted_class, correctness) → trial indices
    preds_t1 = np.array([np.argmax(self_outputs_all[i][0]) for i in range(n_test)])
    true_labels = np.array([np.argmax(y_test[i]) for i in range(n_test)])
    correct_t1 = (preds_t1 == true_labels)

    groups = defaultdict(list)
    for i in range(n_test):
        key = (int(preds_t1[i]), bool(correct_t1[i]))
        groups[key].append(i)

    # Step 3: Evaluate four conditions
    rng = np.random.RandomState(seed + 5000)
    conditions = {'self_current': [], 'self_wrong_trial': [],
                  'clone_current': [], 'group_a': []}

    # Group A
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    for i in range(n_test):
        outputs, _ = net.forward_sequence(X_test[i], T=3)
        pred_t1 = np.argmax(outputs[0])
        pred_t3 = np.argmax(outputs[2])
        conditions['group_a'].append((pred_t1 == true_labels[i],
                                       pred_t3 == true_labels[i]))
    restore_weights(net, saved)

    # Self-current, Self-wrong-trial, Clone-current
    matched_count = 0
    for i in range(n_test):
        true_i = true_labels[i]

        # Self-current
        pred_t1_self = np.argmax(self_outputs_all[i][0])
        pred_t3_self = np.argmax(self_outputs_all[i][2])
        conditions['self_current'].append((pred_t1_self == true_i,
                                            pred_t3_self == true_i))

        # Clone-current
        clone_outputs, _ = forward_sequence_with_clone(net, clone, X_test[i], T=3)
        pred_t1_clone = np.argmax(clone_outputs[0])
        pred_t3_clone = np.argmax(clone_outputs[2])
        conditions['clone_current'].append((pred_t1_clone == true_i,
                                              pred_t3_clone == true_i))

        # Self-wrong-trial: find a matching trial
        key = (int(preds_t1[i]), bool(correct_t1[i]))
        candidates = [j for j in groups[key] if j != i]
        if candidates:
            j = candidates[rng.randint(len(candidates))]
            wrong_outputs = _forward_with_wrong_trial_feedback(
                net, X_test[i], self_outputs_all[j][0])
            pred_t1_wt = np.argmax(wrong_outputs[0])
            pred_t3_wt = np.argmax(wrong_outputs[2])
            conditions['self_wrong_trial'].append((pred_t1_wt == true_i,
                                                     pred_t3_wt == true_i))
            matched_count += 1
        else:
            # Exclude unmatched trial
            conditions['self_wrong_trial'].append(None)

    # Compute gains
    results = {'seed': seed, 'n_test': n_test, 'n_matched': matched_count}
    for cond in ['self_current', 'self_wrong_trial', 'clone_current', 'group_a']:
        valid = [x for x in conditions[cond] if x is not None]
        if valid:
            acc_t1 = np.mean([x[0] for x in valid])
            acc_t3 = np.mean([x[1] for x in valid])
            results[f'{cond}_acc_t1'] = acc_t1
            results[f'{cond}_acc_t3'] = acc_t3
            results[f'{cond}_gain'] = acc_t3 - acc_t1
        else:
            results[f'{cond}_acc_t1'] = np.nan
            results[f'{cond}_acc_t3'] = np.nan
            results[f'{cond}_gain'] = np.nan

    return results


def _process_seed_vn(seed):
    """Process one seed for variable noise."""
    net = RecurrentMLP(seed=seed)
    X_train_seq, y_train = generate_data_variable_noise(200, noise_level=0.5, seed=seed)
    X_test_seq, y_test = generate_data_variable_noise(200, noise_level=0.5, seed=seed + 1000)
    train_vn(net, X_train_seq, y_train, epochs=500, lr=0.01,
             time_weights=[0.0, 0.2, 1.0])

    clone = RecurrentMLP(seed=seed + 100)
    X_clone_seq, y_clone = generate_data_variable_noise(200, noise_level=0.5, seed=seed + 100)
    train_vn(clone, X_clone_seq, y_clone, epochs=500, lr=0.01,
             time_weights=[0.0, 0.2, 1.0])

    n_test = len(X_test_seq)

    # Get self-current outputs
    self_outputs_all = []
    for i in range(n_test):
        outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
        self_outputs_all.append(outputs)

    preds_t1 = np.array([np.argmax(self_outputs_all[i][0]) for i in range(n_test)])
    true_labels = np.array([np.argmax(y_test[i]) for i in range(n_test)])
    correct_t1 = (preds_t1 == true_labels)

    groups = defaultdict(list)
    for i in range(n_test):
        key = (int(preds_t1[i]), bool(correct_t1[i]))
        groups[key].append(i)

    rng = np.random.RandomState(seed + 5000)
    conditions = {'self_current': [], 'self_wrong_trial': [],
                  'clone_current': [], 'group_a': []}

    # Group A
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    for i in range(n_test):
        outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
        pred_t1 = np.argmax(outputs[0])
        pred_t3 = np.argmax(outputs[2])
        conditions['group_a'].append((pred_t1 == true_labels[i],
                                       pred_t3 == true_labels[i]))
    restore_weights(net, saved)

    matched_count = 0
    for i in range(n_test):
        true_i = true_labels[i]

        pred_t1_self = np.argmax(self_outputs_all[i][0])
        pred_t3_self = np.argmax(self_outputs_all[i][2])
        conditions['self_current'].append((pred_t1_self == true_i,
                                            pred_t3_self == true_i))

        clone_outputs, _ = forward_sequence_with_clone_vn(net, clone, X_test_seq[i], T=3)
        pred_t1_clone = np.argmax(clone_outputs[0])
        pred_t3_clone = np.argmax(clone_outputs[2])
        conditions['clone_current'].append((pred_t1_clone == true_i,
                                              pred_t3_clone == true_i))

        key = (int(preds_t1[i]), bool(correct_t1[i]))
        candidates = [j for j in groups[key] if j != i]
        if candidates:
            j = candidates[rng.randint(len(candidates))]
            wrong_outputs = _forward_with_wrong_trial_feedback_vn(
                net, X_test_seq[i], self_outputs_all[j][0])
            pred_t1_wt = np.argmax(wrong_outputs[0])
            pred_t3_wt = np.argmax(wrong_outputs[2])
            conditions['self_wrong_trial'].append((pred_t1_wt == true_i,
                                                     pred_t3_wt == true_i))
            matched_count += 1
        else:
            conditions['self_wrong_trial'].append(None)

    results = {'seed': seed, 'n_test': n_test, 'n_matched': matched_count}
    for cond in ['self_current', 'self_wrong_trial', 'clone_current', 'group_a']:
        valid = [x for x in conditions[cond] if x is not None]
        if valid:
            acc_t1 = np.mean([x[0] for x in valid])
            acc_t3 = np.mean([x[1] for x in valid])
            results[f'{cond}_acc_t1'] = acc_t1
            results[f'{cond}_acc_t3'] = acc_t3
            results[f'{cond}_gain'] = acc_t3 - acc_t1
        else:
            results[f'{cond}_acc_t1'] = np.nan
            results[f'{cond}_acc_t3'] = np.nan
            results[f'{cond}_gain'] = np.nan

    return results


def wilcoxon_exact(diffs):
    """Exact Wilcoxon signed-rank test (2^N enumeration)."""
    diffs = np.array(diffs)
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0, 1.0

    ranks = np.argsort(np.abs(nonzero)) + 1
    T_obs = np.sum(ranks[nonzero > 0])

    count = 0
    for mask in range(2**n):
        T_perm = 0
        for i in range(n):
            if mask & (1 << i):
                T_perm += ranks[i]
        if T_perm <= T_obs or T_perm >= n*(n+1)/2 - T_obs:
            count += 1

    # Two-sided: count extreme values on both ends
    # Actually use the standard approach
    total = 2**n
    # Count how many T values are <= min(T_obs, n(n+1)/2 - T_obs)
    T_min = min(T_obs, n*(n+1)/2 - T_obs)
    extreme_count = 0
    for mask in range(total):
        T_perm = sum(ranks[i] for i in range(n) if mask & (1 << i))
        if T_perm <= T_min:
            extreme_count += 1
    p = 2 * extreme_count / total
    p = min(p, 1.0)

    return T_min, p


def write_report(static_results, vn_results):
    """Write summary report with Wilcoxon tests."""
    with open('results/REPORT_WRONG_TRAJECTORY.md', 'w', encoding='utf-8') as f:
        f.write("# Wrong-Trajectory Substitution Experiment\n\n")

        for task_name, results in [('Static', static_results), ('VN', vn_results)]:
            f.write(f"## {task_name} Input\n\n")

            # Table
            f.write("| Seed | Self-Current Gain | Self-Wrong-Trial Gain | Clone Gain | Group A Gain | Matched |\n")
            f.write("|------|-------------------|-----------------------|------------|--------------|----------|\n")
            for r in results:
                f.write(f"| {r['seed']} | {r['self_current_gain']:+.3f} | "
                        f"{r['self_wrong_trial_gain']:+.3f} | "
                        f"{r['clone_current_gain']:+.3f} | "
                        f"{r['group_a_gain']:+.3f} | "
                        f"{r['n_matched']}/{r['n_test']} |\n")

            # Means
            conds = ['self_current', 'self_wrong_trial', 'clone_current', 'group_a']
            f.write("\n**Mean gains** (N=10):\n")
            for cond in conds:
                gains = [r[f'{cond}_gain'] for r in results if not np.isnan(r[f'{cond}_gain'])]
                f.write(f"- {cond}: {np.mean(gains):+.4f} +/- {np.std(gains):.4f}\n")

            # Wilcoxon tests
            f.write("\n**Wilcoxon signed-rank tests** (exact, two-sided):\n\n")
            comparisons = [
                ('Self-current vs Self-wrong-trial', 'self_current', 'self_wrong_trial'),
                ('Self-current vs Clone-current', 'self_current', 'clone_current'),
                ('Self-wrong-trial vs Clone-current (KEY)', 'self_wrong_trial', 'clone_current'),
                ('Self-wrong-trial vs Group A', 'self_wrong_trial', 'group_a'),
            ]
            for name, a, b in comparisons:
                diffs = []
                for r in results:
                    ga = r[f'{a}_gain']
                    gb = r[f'{b}_gain']
                    if not np.isnan(ga) and not np.isnan(gb):
                        diffs.append(ga - gb)
                if len(diffs) >= 5:
                    T, p = wilcoxon_exact(diffs)
                    mean_diff = np.mean(diffs)
                    f.write(f"- **{name}**: mean diff = {mean_diff:+.4f}, "
                            f"T = {T:.1f}, p = {p:.4f}\n")
                else:
                    f.write(f"- **{name}**: insufficient data\n")

            # Interpretation
            f.write("\n### Interpretation\n\n")
            wt_gains = [r['self_wrong_trial_gain'] for r in results]
            cl_gains = [r['clone_current_gain'] for r in results]
            diffs_key = [w - c for w, c in zip(wt_gains, cl_gains)
                        if not np.isnan(w) and not np.isnan(c)]
            if diffs_key:
                mean_key = np.mean(diffs_key)
                if mean_key > 0.01:
                    f.write("Self-wrong-trial gain > Clone gain: foreign-model identity "
                            "matters beyond state-conditional mismatch. Supports "
                            "feedback-contract specificity.\n")
                elif mean_key < -0.01:
                    f.write("Self-wrong-trial gain < Clone gain: unexpected. "
                            "Needs further investigation.\n")
                else:
                    f.write("Self-wrong-trial gain ~ Clone gain: effect may be "
                            "ordinary state-conditional co-adaptation rather than "
                            "model-specific contract. Consider reframing.\n")
            f.write("\n")

    # Write CSV files
    for task_name, results, fname in [
        ('static', static_results, 'results/wrong_trajectory_static.csv'),
        ('vn', vn_results, 'results/wrong_trajectory_vn.csv'),
    ]:
        fieldnames = ['seed', 'condition', 'acc_t1', 'acc_t3', 'gain']
        with open(fname, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                for cond in ['self_current', 'self_wrong_trial', 'clone_current', 'group_a']:
                    writer.writerow({
                        'seed': r['seed'],
                        'condition': cond,
                        'acc_t1': r[f'{cond}_acc_t1'],
                        'acc_t3': r[f'{cond}_acc_t3'],
                        'gain': r[f'{cond}_gain'],
                    })


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_workers = min(os.cpu_count(), 10)
    print(f"Using {n_workers} workers\n")

    print("=" * 60)
    print("Static Input -Wrong-Trajectory Experiment")
    print("=" * 60)
    with Pool(processes=n_workers) as pool:
        static_results = pool.map(_process_seed_static, range(10))
    for r in static_results:
        print(f"  seed={r['seed']}: self={r['self_current_gain']:+.3f} "
              f"wrong_trial={r['self_wrong_trial_gain']:+.3f} "
              f"clone={r['clone_current_gain']:+.3f} "
              f"groupA={r['group_a_gain']:+.3f} "
              f"matched={r['n_matched']}/{r['n_test']}")

    print("\n" + "=" * 60)
    print("Variable Noise -Wrong-Trajectory Experiment")
    print("=" * 60)
    with Pool(processes=n_workers) as pool:
        vn_results = pool.map(_process_seed_vn, range(10))
    for r in vn_results:
        print(f"  seed={r['seed']}: self={r['self_current_gain']:+.3f} "
              f"wrong_trial={r['self_wrong_trial_gain']:+.3f} "
              f"clone={r['clone_current_gain']:+.3f} "
              f"groupA={r['group_a_gain']:+.3f} "
              f"matched={r['n_matched']}/{r['n_test']}")

    write_report(static_results, vn_results)
    print("\nResults saved to results/wrong_trajectory_*.csv and REPORT_WRONG_TRAJECTORY.md")
