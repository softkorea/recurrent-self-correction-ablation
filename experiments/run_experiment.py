"""전체 실험 실행 스크립트 (병렬화).

병렬화 전략:
- multiprocessing.Pool로 (seed_model, noise_level) 조합을 병렬 처리
- 각 워커가 모델 학습 + 전 그룹 실험을 독립 수행
- CPU 코어 수에 맞춰 자동 조절
- 재현성: 각 워커 내 seed 고정 유지
"""

# NumPy 내부 멀티스레딩 비활성화 — mp.Pool과의 CPU 스래싱 방지 (Gemini review #2)
# 반드시 numpy import 전에 설정해야 함
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import csv
import multiprocessing as mp
from functools import partial

from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import (
    ablate_recurrent, ablate_random, ablate_structural,
    deep_copy_weights, restore_weights,
)
from src.metrics import compute_all_metrics, compute_neuron_importance
from src.visualize import (
    plot_network_map, plot_ablation_comparison,
    plot_accuracy_distribution, plot_noise_sweep,
    plot_neuron_importance_heatmap, ensure_results_dir,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

N_MODELS = 10
N_RANDOM_ABLATIONS = 30
N_SCRAMBLE_SEEDS = 30
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EPOCHS = 500
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3
N_REC_WEIGHTS = 50


# ──────────────────────────────────────────────
# 워커 함수 (각 (seed, noise) 조합 독립 처리)
# ──────────────────────────────────────────────

def run_single_model(args):
    """단일 모델 학습 + 전 그룹 실험.

    Args:
        args: (seed_model, noise_level)

    Returns:
        list of row dicts
    """
    seed_model, noise_level = args
    rows = []

    # 학습
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed_model)
    X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed_model)
    X_test, y_test = generate_data(N_TEST, noise_level, seed=seed_model + 500)
    train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    def make_row(group, metrics, seed_abl=0):
        return {
            'seed_model': seed_model,
            'group': group,
            'seed_ablation': seed_abl,
            'noise_level': noise_level,
            **{k: metrics[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                         'gain', 'ece', 'r_norm', 'delta_norm']}
        }

    # Baseline
    rows.append(make_row('Baseline', compute_all_metrics(net, X_test, y_test)))

    # Group A: 재귀 절단
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    rows.append(make_row('A', compute_all_metrics(net, X_test, y_test)))
    restore_weights(net, saved)

    # Group B1: 랜덤 절단 (30회)
    for seed_abl in range(N_RANDOM_ABLATIONS):
        saved = deep_copy_weights(net)
        ablate_random(net, n_connections=N_REC_WEIGHTS, seed=seed_abl + 1000)
        rows.append(make_row('B1', compute_all_metrics(net, X_test, y_test), seed_abl + 1000))
        restore_weights(net, saved)

    # Group B2: 구조적 절단 (h2_to_output, 50 params = W_rec과 동일)
    saved = deep_copy_weights(net)
    ablate_structural(net, layer='h2_to_output')
    rows.append(make_row('B2', compute_all_metrics(net, X_test, y_test)))
    restore_weights(net, saved)

    # Group C1: Permutation feedback (30회)
    for seed_scr in range(N_SCRAMBLE_SEEDS):
        net.enable_scrambled_feedback(seed=seed_scr + 2000)
        rows.append(make_row('C1', compute_all_metrics(net, X_test, y_test), seed_scr + 2000))
        net.disable_scrambled_feedback()

    # Group D: Feedforward (재귀 없이 훈련)
    net_d = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                         output_size=5, seed=seed_model)
    net_d.disable_recurrent_loop()
    X_train_d, y_train_d = generate_data(N_TRAIN, noise_level, seed=seed_model)
    train(net_d, X_train_d, y_train_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(make_row('D', compute_all_metrics(net_d, X_test, y_test)))

    # Group D': Param-matched FF (skip connection)
    net_dp = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                          output_size=5, seed=seed_model, skip_connection=True)
    net_dp.disable_recurrent_loop()
    train(net_dp, X_train_d, y_train_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(make_row("D'", compute_all_metrics(net_dp, X_test, y_test)))

    return rows


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def run_full_experiment():
    ensure_results_dir('results')

    # 모든 (seed, noise) 조합 생성
    tasks = [(seed, nl) for nl in NOISE_LEVELS for seed in range(N_MODELS)]
    n_workers = min(max(1, mp.cpu_count() - 4), len(tasks))

    print(f"[EXP] Starting: {len(tasks)} tasks on {n_workers} workers "
          f"({N_MODELS} models x {len(NOISE_LEVELS)} noise levels)", flush=True)

    # 병렬 실행 (tqdm 진행률 표시, Gemini review #2)
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_single_model, tasks),
                              total=len(tasks), desc="Experiments"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_single_model, tasks)
            for batch in results:
                all_rows.extend(batch)

    print(f"[EXP] Collected {len(all_rows)} rows", flush=True)

    # CSV 저장
    csv_path = 'results/raw_metrics.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # ──────────────────────────────────────────
    # 시각화
    # ──────────────────────────────────────────
    print("[EXP] Generating plots...", flush=True)

    # Ablation comparison (noise=0.5)
    comparison = {}
    for row in all_rows:
        if row['noise_level'] == 0.5:
            comparison.setdefault(row['group'], []).append(row['gain'])
    plot_ablation_comparison(comparison, 'results/ablation_comparison.png')

    # Accuracy distribution (noise=0.5)
    b1 = [r['gain'] for r in all_rows if r['group'] == 'B1' and r['noise_level'] == 0.5]
    a_g = np.mean([r['gain'] for r in all_rows if r['group'] == 'A' and r['noise_level'] == 0.5])
    c1_g = np.mean([r['gain'] for r in all_rows if r['group'] == 'C1' and r['noise_level'] == 0.5])
    if b1:
        plot_accuracy_distribution(b1, a_g, c1_g, 'results/accuracy_distribution.png')

    # Noise sweep
    sweep = {}
    for row in all_rows:
        sweep.setdefault(row['group'], {}).setdefault(row['noise_level'], []).append(row['gain'])
    plot_noise_sweep(sweep, 'results/noise_sweep_curve.png')

    # Network map + Neuron importance heatmap (noise=0.5, seed=0)
    net_hm = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    X_train_hm, y_train_hm = generate_data(N_TRAIN, 0.5, seed=0)
    X_test_hm, y_test_hm = generate_data(N_TEST, 0.5, seed=500)
    train(net_hm, X_train_hm, y_train_hm, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    plot_network_map(net_hm, 'results/network_map.png')

    print("[EXP] Computing neuron importance heatmap...", flush=True)
    intelligence, correction = compute_neuron_importance(net_hm, X_test_hm, y_test_hm)
    plot_neuron_importance_heatmap(intelligence, correction, 'results/neuron_importance_heatmap.png')

    # Neuron importance CSV 저장 (ChatGPT review P1-3: 재현성)
    ni_path = 'results/neuron_importance.csv'
    with open(ni_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['neuron_id', 'intelligence_importance',
                                                'correction_importance'])
        writer.writeheader()
        for nid in sorted(intelligence.keys()):
            writer.writerow({'neuron_id': nid,
                             'intelligence_importance': intelligence[nid],
                             'correction_importance': correction[nid]})
    print(f"[EXP] Saved neuron importance to {ni_path}", flush=True)

    # ──────────────────────────────────────────
    # 모델 단위 집계 (ChatGPT review P0-4)
    # B1/C1은 모델당 30회 반복 → seed_model별 평균으로 축약하여 N=10
    # ──────────────────────────────────────────
    def aggregate_by_model(rows, noise=0.5):
        """그룹별로 seed_model 단위로 집계. {group: [model_mean_gain x 10]}"""
        from collections import defaultdict
        raw = defaultdict(lambda: defaultdict(list))
        raw_acc_t1 = defaultdict(lambda: defaultdict(list))
        raw_acc_t3 = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r['noise_level'] == noise:
                raw[r['group']][r['seed_model']].append(r['gain'])
                raw_acc_t1[r['group']][r['seed_model']].append(r['acc_t1'])
                raw_acc_t3[r['group']][r['seed_model']].append(r['acc_t3'])
        result = {}
        result_t1 = {}
        result_t3 = {}
        for g in raw:
            result[g] = [np.mean(raw[g][s]) for s in sorted(raw[g].keys())]
            result_t1[g] = [np.mean(raw_acc_t1[g][s]) for s in sorted(raw_acc_t1[g].keys())]
            result_t3[g] = [np.mean(raw_acc_t3[g][s]) for s in sorted(raw_acc_t3[g].keys())]
        return result, result_t1, result_t3

    model_gains, model_t1, model_t3 = aggregate_by_model(all_rows, noise=0.5)

    # ──────────────────────────────────────────
    # 통계 요약 (모델 단위, N=10)
    # ──────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY (noise_level=0.5, model-level aggregation, N=10)", flush=True)
    print('='*70, flush=True)
    print(f"  {'Group':12s}  {'acc_t1':>10s}  {'acc_t3':>10s}  {'gain':>14s}  {'n':>3s}", flush=True)
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*3}", flush=True)

    for g in ['Baseline', 'A', 'B1', 'B2', 'C1', 'D', "D'"]:
        if g in model_gains:
            gains = model_gains[g]
            t1s = model_t1[g]
            t3s = model_t3[g]
            print(f"  {g:12s}  {np.mean(t1s):.4f}±{np.std(t1s):.3f}"
                  f"  {np.mean(t3s):.4f}±{np.std(t3s):.3f}"
                  f"  {np.mean(gains):+.4f}±{np.std(gains):.4f}"
                  f"  {len(gains):3d}", flush=True)

    # Bootstrap 95% CI (모델 단위)
    print("\n95% Bootstrap CI (noise=0.5, model-level):", flush=True)
    rng = np.random.RandomState(999)
    for g in ['Baseline', 'A', 'B1', 'C1', 'D', "D'"]:
        if g in model_gains and len(model_gains[g]) > 1:
            gains = np.array(model_gains[g])
            boot = [np.mean(rng.choice(gains, len(gains), replace=True)) for _ in range(10000)]
            print(f"  {g:12s}: [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Holm-Bonferroni (모델 단위 paired comparison)
    from src.metrics import wilcoxon_exact
    print("\nHolm-Bonferroni corrected p-values (Baseline vs each, noise=0.5, N=10):", flush=True)
    baseline_gains = np.array(model_gains.get('Baseline', []))
    p_values = {}
    for g in ['A', 'B1', 'C1', 'C2', 'D', "D'"]:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(baseline_gains) > 1:
                _, p = wilcoxon_exact(baseline_gains, g_gains)
                p_values[g] = p

    if p_values:
        sorted_ps = sorted(p_values.items(), key=lambda x: x[1])
        m_comp = len(sorted_ps)
        prev_adj_p = 0.0
        for rank, (g, p) in enumerate(sorted_ps):
            adj_p = min(p * (m_comp - rank), 1.0)
            adj_p = max(prev_adj_p, adj_p)  # 단조 증가
            prev_adj_p = adj_p
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"  Baseline vs {g:4s}: p={adj_p:.4e} {sig}", flush=True)

    print(f"\nExperiment complete. Results in results/", flush=True)


if __name__ == '__main__':
    run_full_experiment()
