"""Group C2 (Clone Feedback) 실험 실행 (병렬화).

기존 raw_metrics.csv에 C2 행을 추가.
각 target 모델(seed 0-9)에 대해 독립적으로 훈련된 donor 모델(seed 100-109)의
출력을 피드백으로 대체하여 평가. 완전 독립 1:1 매칭.

병렬화 전략:
- noise_level 단위로 병렬 (같은 noise에서 target + donor 모두 학습 후 clone 평가)
- n_workers = cpu_count() - 4 (다른 작업에 지장 없도록)
"""

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
from collections import defaultdict

from src.network import RecurrentMLP
from src.training import generate_data, train
from src.metrics import compute_all_metrics_with_clone
from src.visualize import plot_ablation_comparison, plot_noise_sweep, ensure_results_dir

# ──────────────────────────────────────────────
# 설정 (기존 실험과 동일)
# ──────────────────────────────────────────────

N_MODELS = 10
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EPOCHS = 500
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3


# ──────────────────────────────────────────────
# 워커 함수 (noise_level 단위 독립 처리)
# ──────────────────────────────────────────────

DONOR_SEED_OFFSET = 100  # donor seeds: 100-109 (독립)


def run_c2_for_noise(noise_level):
    """한 noise_level에서 target 10모델 + donor 10모델 학습 후 C2 평가.

    Target 모델(seed 0-9)과 독립 donor 모델(seed 100-109)을 1:1 매칭.
    통계적 독립성 완전 확보.

    Returns:
        list of row dicts
    """
    rows = []

    # 10개 target 모델 학습
    targets = []
    test_data = []
    for seed in range(N_MODELS):
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed)
        X_test, y_test = generate_data(N_TEST, noise_level, seed=seed + 500)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        targets.append(net)
        test_data.append((X_test, y_test))

    # 10개 독립 donor 모델 학습 (seed 100-109)
    # 같은 분포(동일 noise_level)에서 독립 생성된 데이터로 훈련
    donors = []
    for seed in range(N_MODELS):
        donor_seed = seed + DONOR_SEED_OFFSET
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=donor_seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=donor_seed)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        donors.append(net)

    # C2: Clone feedback — target[i]의 피드백을 donor[i]로 (독립 1:1 매칭)
    for seed in range(N_MODELS):
        target = targets[seed]
        donor = donors[seed]
        X_test, y_test = test_data[seed]

        metrics = compute_all_metrics_with_clone(target, donor, X_test, y_test)

        row = {
            'seed_model': seed,
            'group': 'C2',
            'seed_ablation': seed + DONOR_SEED_OFFSET,
            'noise_level': noise_level,
            **{k: metrics[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                         'gain', 'ece', 'r_norm', 'delta_norm']}
        }
        rows.append(row)

    return rows


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def run_c2_experiment():
    ensure_results_dir('results')

    n_workers = max(1, mp.cpu_count() - 4)

    print(f"[C2] Starting: {len(NOISE_LEVELS)} noise levels on {n_workers} workers "
          f"(cpu_count={mp.cpu_count()}, reserved 4)", flush=True)

    # 병렬 실행
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_c2_for_noise, NOISE_LEVELS),
                              total=len(NOISE_LEVELS), desc="C2 Experiment"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_c2_for_noise, NOISE_LEVELS)
            for batch in results:
                all_rows.extend(batch)

    print(f"[C2] Collected {len(all_rows)} C2 rows", flush=True)

    # 기존 CSV에 추가 (append, 기존 C2 제거 후)
    csv_path = 'results/raw_metrics.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
    else:
        print(f"[C2] Warning: {csv_path} not found. Creating new file.", flush=True)

    # 기존 C2 행 제거 (재실행 시 중복 방지)
    existing_rows = [r for r in existing_rows if r['group'] != 'C2']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(all_rows)

    print(f"[C2] Appended {len(all_rows)} C2 rows to {csv_path}", flush=True)

    # 시각화 재생성
    all_data = existing_rows + all_rows

    # Ablation comparison (noise=0.5)
    comparison = {}
    for row in all_data:
        nl = float(row['noise_level']) if isinstance(row['noise_level'], str) else row['noise_level']
        if nl == 0.5:
            g = row['group']
            gain = float(row['gain']) if isinstance(row['gain'], str) else row['gain']
            comparison.setdefault(g, []).append(gain)
    plot_ablation_comparison(comparison, 'results/ablation_comparison.png')
    print("[C2] Updated ablation_comparison.png", flush=True)

    # Noise sweep (all groups including C2)
    sweep = {}
    for row in all_data:
        g = row['group']
        nl = float(row['noise_level']) if isinstance(row['noise_level'], str) else row['noise_level']
        gain = float(row['gain']) if isinstance(row['gain'], str) else row['gain']
        sweep.setdefault(g, {}).setdefault(nl, []).append(gain)
    plot_noise_sweep(sweep, 'results/noise_sweep_curve.png')
    print("[C2] Updated noise_sweep_curve.png", flush=True)

    # 통계 요약
    print(f"\n{'='*70}", flush=True)
    print("C2 RESULTS SUMMARY (model-level aggregation, N=10)", flush=True)
    print('='*70, flush=True)

    def aggregate(rows, noise=0.5):
        raw = defaultdict(lambda: defaultdict(list))
        for r in rows:
            nl = float(r['noise_level']) if isinstance(r['noise_level'], str) else r['noise_level']
            if nl == noise:
                g = r['group']
                gain = float(r['gain']) if isinstance(r['gain'], str) else r['gain']
                sm = int(r['seed_model']) if isinstance(r['seed_model'], str) else r['seed_model']
                raw[g][sm].append(gain)
        return {g: [np.mean(raw[g][s]) for s in sorted(raw[g].keys())] for g in raw}

    model_gains = aggregate(all_data, noise=0.5)

    print(f"\n  {'Group':12s}  {'gain (mean±std)':>20s}  {'n':>3s}", flush=True)
    print(f"  {'-'*12}  {'-'*20}  {'-'*3}", flush=True)

    for g in ['Baseline', 'A', 'C1', 'C2']:
        if g in model_gains:
            gains = model_gains[g]
            print(f"  {g:12s}  {np.mean(gains):+.4f}±{np.std(gains):.4f}"
                  f"          {len(gains):3d}", flush=True)

    # Bootstrap 95% CI
    rng = np.random.RandomState(999)
    print("\n95% Bootstrap CI (noise=0.5, model-level):", flush=True)
    for g in ['Baseline', 'A', 'C1', 'C2']:
        if g in model_gains and len(model_gains[g]) > 1:
            gains = np.array(model_gains[g])
            boot = [np.mean(rng.choice(gains, len(gains), replace=True)) for _ in range(10000)]
            print(f"  {g:12s}: [{np.percentile(boot, 2.5):+.4f}, "
                  f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Holm-Bonferroni corrected p-values
    from src.metrics import wilcoxon_exact
    print("\nHolm-Bonferroni corrected p-values (Baseline vs each, noise=0.5):", flush=True)
    baseline_gains = np.array(model_gains.get('Baseline', []))
    p_values = {}
    for g in ['A', 'C1', 'C2']:
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
            adj_p = max(prev_adj_p, adj_p)
            prev_adj_p = adj_p
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"  Baseline vs {g:4s}: p={adj_p:.4e} {sig}", flush=True)

    print(f"\n[C2] Experiment complete.", flush=True)


if __name__ == '__main__':
    run_c2_experiment()
