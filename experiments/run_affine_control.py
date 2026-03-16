"""Affine Alignment Control: C2 clone feedback의 성능 저하가
단순 representation misalignment인지 검증.

핵심 질문: donor logit에 learned affine transform을 적용하면
C2 degradation이 사라지는가?

방법:
  1. target/donor 각각 self-feedback로 calibration data 실행
  2. donor_logit → target_logit 선형 회귀 (W, b) 학습
  3. 평가 시 donor output에 W @ d + b 적용 후 target에 주입
  4. Baseline, C2-raw, C2-learned-affine gain 비교

결과 해석:
  - C2-learned-affine ≈ Baseline → 단순 representation misalignment
  - C2-learned-affine ≈ C2-raw → feedback-contract specificity 확인
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import csv

from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise, train, train_vn,
    softmax,
)
from src.ablation import (
    forward_sequence_with_clone,
    forward_sequence_with_clone_vn,
    fit_learned_affine,
    fit_learned_affine_vn,
    forward_sequence_with_learned_affine_clone,
    forward_sequence_with_learned_affine_clone_vn,
)
from src.metrics import wilcoxon_exact

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

N_MODELS = 10
NOISE_LEVEL = 0.5
TRAIN_EPOCHS = 500
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3
DONOR_SEED_OFFSET = 100


def compute_gain_from_forward_fn(forward_fn, X, y, **kwargs):
    """Generic gain computation from a forward function.

    forward_fn(x, **kwargs) -> (outputs, caches)
    """
    n = len(X)
    correct_t1 = 0
    correct_t3 = 0

    for i in range(n):
        outputs, _ = forward_fn(X[i], **kwargs)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

    return correct_t3 / n - correct_t1 / n


def compute_gain_from_forward_fn_vn(forward_fn, X_seq, y, **kwargs):
    """Generic VN gain computation."""
    n = len(X_seq)
    correct_t1 = 0
    correct_t3 = 0

    for i in range(n):
        outputs, _ = forward_fn(X_seq[i], **kwargs)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

    return correct_t3 / n - correct_t1 / n


def eval_baseline_gain(net, X, y):
    """Baseline gain (self-feedback)."""
    n = len(X)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = net.forward_sequence(X[i], T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


def eval_baseline_gain_vn(net, X_seq, y):
    """Baseline VN gain."""
    n = len(X_seq)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = net.forward_sequence_vn(X_seq[i], T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


def eval_c2_gain(target, donor, X, y):
    """C2 raw gain (static)."""
    n = len(X)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = forward_sequence_with_clone(target, donor, X[i], T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


def eval_c2_gain_vn(target, donor, X_seq, y):
    """C2 raw gain (VN)."""
    n = len(X_seq)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = forward_sequence_with_clone_vn(target, donor, X_seq[i], T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


def eval_c2_learned_affine_gain(target, donor, X, y, W, b):
    """C2 learned-affine gain (static)."""
    n = len(X)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = forward_sequence_with_learned_affine_clone(
            target, donor, X[i], W, b, T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


def eval_c2_learned_affine_gain_vn(target, donor, X_seq, y, W, b):
    """C2 learned-affine gain (VN)."""
    n = len(X_seq)
    correct_t1 = 0
    correct_t3 = 0
    for i in range(n):
        outputs, _ = forward_sequence_with_learned_affine_clone_vn(
            target, donor, X_seq[i], W, b, T=T)
        true_cls = np.argmax(y[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1
    return correct_t3 / n - correct_t1 / n


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def run_affine_control():
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("AFFINE ALIGNMENT CONTROL EXPERIMENT")
    print("=" * 70)
    print(f"N_models={N_MODELS}, noise={NOISE_LEVEL}, "
          f"epochs={TRAIN_EPOCHS}, T={T}")
    print(f"Calibration: training data ({N_TRAIN} samples)")
    print(f"Evaluation: test data ({N_TEST} samples)")
    print()

    # Storage
    static_results = {
        'Baseline': [], 'C2-raw': [], 'C2-learned-affine': [],
    }
    vn_results = {
        'Baseline': [], 'C2-raw': [], 'C2-learned-affine': [],
    }
    alignment_stats = []  # (seed, setting, R2, residual_norm)

    for seed in range(N_MODELS):
        donor_seed = seed + DONOR_SEED_OFFSET
        print(f"\n--- Model pair {seed} / {donor_seed} ---")

        # ── Train target and donor (static) ────────────
        target_st = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                                  output_size=5, seed=seed)
        X_train_st, y_train_st = generate_data(N_TRAIN, NOISE_LEVEL, seed=seed)
        X_test_st, y_test_st = generate_data(N_TEST, NOISE_LEVEL, seed=seed + 500)
        train(target_st, X_train_st, y_train_st,
              epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

        donor_st = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                                 output_size=5, seed=donor_seed)
        X_train_d_st, y_train_d_st = generate_data(
            N_TRAIN, NOISE_LEVEL, seed=donor_seed)
        train(donor_st, X_train_d_st, y_train_d_st,
              epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

        # ── Train target and donor (VN) ────────────
        target_vn = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                                  output_size=5, seed=seed)
        X_seq_train_vn, y_train_vn = generate_data_variable_noise(
            N_TRAIN, NOISE_LEVEL, T=T, seed=seed)
        X_seq_test_vn, y_test_vn = generate_data_variable_noise(
            N_TEST, NOISE_LEVEL, T=T, seed=seed + 500)
        train_vn(target_vn, X_seq_train_vn, y_train_vn,
                 epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

        donor_vn = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                                 output_size=5, seed=donor_seed)
        X_seq_train_d_vn, y_train_d_vn = generate_data_variable_noise(
            N_TRAIN, NOISE_LEVEL, T=T, seed=donor_seed)
        train_vn(donor_vn, X_seq_train_d_vn, y_train_d_vn,
                 epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

        # ── Fit learned affine (calibration on training data) ────────
        # Static: both models run on TARGET's training data
        W_st, b_st = fit_learned_affine(target_st, donor_st, X_train_st, T=T)

        # VN: need shared input sequences for calibration
        # Generate shared VN calibration data (same noise for both)
        X_seq_calib_vn, _ = generate_data_variable_noise(
            N_TRAIN, NOISE_LEVEL, T=T, seed=seed + 2000)
        W_vn, b_vn = fit_learned_affine_vn(
            target_vn, donor_vn, X_seq_calib_vn, T=T)

        # ── Compute alignment quality (R²) ────────
        for setting, tgt, dnr, W, b_vec, X_cal in [
            ('static', target_st, donor_st, W_st, b_st, None),
            ('VN', target_vn, donor_vn, W_vn, b_vn, None),
        ]:
            # Recompute on calibration data for R²
            donor_all = []
            target_all = []
            if setting == 'static':
                for i in range(len(X_train_st)):
                    tgt.reset_state()
                    dnr.reset_state()
                    for t in range(T):
                        ty = tgt.forward(X_train_st[i])
                        dy = dnr.forward(X_train_st[i])
                        donor_all.append(dy.copy())
                        target_all.append(ty.copy())
            else:
                for i in range(len(X_seq_calib_vn)):
                    tgt.reset_state()
                    dnr.reset_state()
                    for t in range(T):
                        ty = tgt.forward(X_seq_calib_vn[i, t])
                        dy = dnr.forward(X_seq_calib_vn[i, t])
                        donor_all.append(dy.copy())
                        target_all.append(ty.copy())

            D = np.array(donor_all)
            T_ref = np.array(target_all)
            predicted = D @ W + b_vec
            ss_res = np.sum((T_ref - predicted) ** 2)
            ss_tot = np.sum((T_ref - T_ref.mean(axis=0)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            residual = np.sqrt(np.mean((T_ref - predicted) ** 2))
            alignment_stats.append((seed, setting, r2, residual))
            print(f"  {setting} alignment: R²={r2:.4f}, RMSE={residual:.4f}")

        # ── Evaluate: Static ────────────
        g_baseline_st = eval_baseline_gain(target_st, X_test_st, y_test_st)
        g_c2_st = eval_c2_gain(target_st, donor_st, X_test_st, y_test_st)
        g_affine_st = eval_c2_learned_affine_gain(
            target_st, donor_st, X_test_st, y_test_st, W_st, b_st)

        static_results['Baseline'].append(g_baseline_st)
        static_results['C2-raw'].append(g_c2_st)
        static_results['C2-learned-affine'].append(g_affine_st)

        print(f"  Static:  Baseline={g_baseline_st:+.3f}  "
              f"C2-raw={g_c2_st:+.3f}  C2-affine={g_affine_st:+.3f}")

        # ── Evaluate: VN ────────────
        # VN test data: shared noise for target and donor
        X_seq_test_shared, y_test_shared = generate_data_variable_noise(
            N_TEST, NOISE_LEVEL, T=T, seed=seed + 500)

        g_baseline_vn = eval_baseline_gain_vn(
            target_vn, X_seq_test_shared, y_test_shared)
        g_c2_vn = eval_c2_gain_vn(
            target_vn, donor_vn, X_seq_test_shared, y_test_shared)
        g_affine_vn = eval_c2_learned_affine_gain_vn(
            target_vn, donor_vn, X_seq_test_shared, y_test_shared, W_vn, b_vn)

        vn_results['Baseline'].append(g_baseline_vn)
        vn_results['C2-raw'].append(g_c2_vn)
        vn_results['C2-learned-affine'].append(g_affine_vn)

        print(f"  VN:      Baseline={g_baseline_vn:+.3f}  "
              f"C2-raw={g_c2_vn:+.3f}  C2-affine={g_affine_vn:+.3f}")

    # ──────────────────────────────────────────────
    # Results summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (N=10 models, noise=0.5)")
    print("=" * 70)

    for label, results in [('STATIC', static_results), ('VN', vn_results)]:
        print(f"\n  [{label}]")
        print(f"  {'Group':22s}  {'gain (mean±SD)':>18s}  {'95% CI':>24s}")
        print(f"  {'-'*22}  {'-'*18}  {'-'*24}")

        rng = np.random.RandomState(42)
        for g in ['Baseline', 'C2-raw', 'C2-learned-affine']:
            vals = np.array(results[g])
            boot = [np.mean(rng.choice(vals, len(vals), replace=True))
                    for _ in range(10000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {g:22s}  {np.mean(vals):+.4f}±{np.std(vals):.4f}"
                  f"  [{lo:+.4f}, {hi:+.4f}]")

    # ──────────────────────────────────────────────
    # Statistical tests
    # ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS (Wilcoxon signed-rank, exact)")
    print("=" * 70)

    for label, results in [('STATIC', static_results), ('VN', vn_results)]:
        print(f"\n  [{label}]")
        baseline = np.array(results['Baseline'])
        c2_raw = np.array(results['C2-raw'])
        c2_affine = np.array(results['C2-learned-affine'])

        tests = [
            ('Baseline vs C2-raw', baseline, c2_raw),
            ('Baseline vs C2-learned-affine', baseline, c2_affine),
            ('C2-raw vs C2-learned-affine', c2_raw, c2_affine),
        ]

        p_values = []
        for name, a, b_arr in tests:
            T_stat, p = wilcoxon_exact(a, b_arr)
            p_values.append((name, p, T_stat))

        # Holm-Bonferroni correction
        sorted_ps = sorted(p_values, key=lambda x: x[1])
        m = len(sorted_ps)
        prev_adj = 0.0
        for rank, (name, p, T_stat) in enumerate(sorted_ps):
            adj_p = min(p * (m - rank), 1.0)
            adj_p = max(prev_adj, adj_p)
            prev_adj = adj_p
            sig = ("***" if adj_p < 0.001 else "**" if adj_p < 0.01
                   else "*" if adj_p < 0.05 else "ns")
            print(f"    {name:38s}: T={T_stat:.0f}, "
                  f"p_adj={adj_p:.4f} {sig}")

    # ──────────────────────────────────────────────
    # Alignment quality summary
    # ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ALIGNMENT QUALITY (calibration set)")
    print("=" * 70)
    for setting in ['static', 'VN']:
        stats = [(s, r2, rmse) for s, st, r2, rmse in alignment_stats
                 if st == setting]
        r2s = [r2 for _, r2, _ in stats]
        rmses = [rmse for _, _, rmse in stats]
        print(f"  {setting:6s}: R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
              f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}")

    # ──────────────────────────────────────────────
    # Key question answer
    # ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)

    for label, results in [('STATIC', static_results), ('VN', vn_results)]:
        baseline_mean = np.mean(results['Baseline'])
        c2_raw_mean = np.mean(results['C2-raw'])
        c2_affine_mean = np.mean(results['C2-learned-affine'])

        rescue_pct = 0.0
        if baseline_mean - c2_raw_mean != 0:
            rescue_pct = ((c2_affine_mean - c2_raw_mean)
                          / (baseline_mean - c2_raw_mean) * 100)

        print(f"\n  [{label}]")
        print(f"    C2-raw degradation:    {c2_raw_mean - baseline_mean:+.4f}")
        print(f"    Affine recovery:       {c2_affine_mean - c2_raw_mean:+.4f}")
        print(f"    Rescue percentage:     {rescue_pct:.1f}% of lost gain recovered")

        if rescue_pct > 80:
            print("    → C2 harm was primarily representation misalignment.")
            print("      Feedback-contract specificity claim needs revision.")
        elif rescue_pct > 40:
            print("    → Partial rescue: both misalignment and model-specific "
                  "geometry contribute.")
        else:
            print("    → Affine alignment fails to rescue C2.")
            print("      Feedback-contract specificity confirmed beyond "
                  "linear representation mismatch.")

    # ──────────────────────────────────────────────
    # Save CSV
    # ──────────────────────────────────────────────
    csv_path = 'results/affine_control.csv'
    fields = ['seed_model', 'setting', 'group', 'gain']
    rows = []
    for seed in range(N_MODELS):
        for setting, results in [('static', static_results),
                                 ('vn', vn_results)]:
            for g in ['Baseline', 'C2-raw', 'C2-learned-affine']:
                rows.append({
                    'seed_model': seed,
                    'setting': setting,
                    'group': g,
                    'gain': results[g][seed],
                })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Saved] {csv_path} ({len(rows)} rows)")
    print("[Done] Affine alignment control experiment complete.")


if __name__ == '__main__':
    run_affine_control()
