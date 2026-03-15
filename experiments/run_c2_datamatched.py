"""Group C2 Data-Matched variant (robustness check).

논문 Section 3.3의 robustness check:
donor 모델이 target과 완전히 동일한 훈련 데이터를 사용하되,
가중치 초기화만 다르게 하여 초기화 궤적의 순수 효과를 검증.

결과는 별도 CSV(results/c2_datamatched.csv)에 저장.
메인 raw_metrics.csv는 수정하지 않음.
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
from src.metrics import compute_all_metrics_with_clone, wilcoxon_exact

N_MODELS = 10
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EPOCHS = 500
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3
DONOR_SEED_OFFSET = 100


def run_c2_datamatched_for_noise(noise_level):
    """Data-matched C2: donor는 target과 동일 데이터, 다른 초기화."""
    rows = []

    # Target 모델 학습
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

    # Donor 모델: 동일 데이터(seed=seed), 다른 초기화(seed=donor_seed)
    donors = []
    for seed in range(N_MODELS):
        donor_seed = seed + DONOR_SEED_OFFSET
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=donor_seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed)  # target과 동일
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        donors.append(net)

    for seed in range(N_MODELS):
        target = targets[seed]
        donor = donors[seed]
        X_test, y_test = test_data[seed]

        metrics = compute_all_metrics_with_clone(target, donor, X_test, y_test)

        row = {
            'seed_model': seed,
            'group': 'C2_datamatched',
            'seed_ablation': seed + DONOR_SEED_OFFSET,
            'noise_level': noise_level,
            **{k: metrics[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                         'gain', 'ece', 'r_norm', 'delta_norm']}
        }
        rows.append(row)

    return rows


def main():
    n_workers = max(1, mp.cpu_count() - 4)
    print(f"[C2-DM] Starting data-matched variant: {len(NOISE_LEVELS)} noise levels "
          f"on {n_workers} workers", flush=True)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_c2_datamatched_for_noise,
                              NOISE_LEVELS), total=len(NOISE_LEVELS),
                              desc="C2 Data-Matched"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_c2_datamatched_for_noise, NOISE_LEVELS)
            for batch in results:
                all_rows.extend(batch)

    # 별도 CSV에 저장
    csv_path = 'results/c2_datamatched.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    os.makedirs('results', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[C2-DM] Saved {len(all_rows)} rows to {csv_path}", flush=True)

    # 요약
    raw = defaultdict(list)
    for r in all_rows:
        if float(r['noise_level']) == 0.5:
            raw[int(r['seed_model'])].append(float(r['gain']))
    model_gains = [np.mean(raw[s]) for s in sorted(raw.keys())]

    print(f"\n[C2-DM] noise=0.5: gain = {np.mean(model_gains):+.4f} "
          f"+/- {np.std(model_gains):.4f} (N={len(model_gains)})", flush=True)

    rng = np.random.RandomState(999)
    boot = [np.mean(rng.choice(model_gains, len(model_gains), replace=True))
            for _ in range(10000)]
    print(f"[C2-DM] 95% CI: [{np.percentile(boot, 2.5):+.4f}, "
          f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Wilcoxon exact tests vs Baseline and Group A from main experiment
    main_csv = 'results/raw_metrics.csv'
    if os.path.exists(main_csv):
        baseline_gains = defaultdict(list)
        a_gains = defaultdict(list)
        with open(main_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if float(r['noise_level']) == 0.5:
                    if r['group'] == 'Baseline':
                        baseline_gains[int(r['seed_model'])].append(float(r['gain']))
                    elif r['group'] == 'A':
                        a_gains[int(r['seed_model'])].append(float(r['gain']))
        baseline_means = [np.mean(baseline_gains[s]) for s in sorted(baseline_gains.keys())]
        a_means = [np.mean(a_gains[s]) for s in sorted(a_gains.keys())]

        if len(baseline_means) == len(model_gains):
            T_bl, p_bl = wilcoxon_exact(model_gains, baseline_means)
            print(f"[C2-DM] Wilcoxon exact vs Baseline: T={T_bl:.1f}, p={p_bl:.6f}", flush=True)
        if len(a_means) == len(model_gains):
            T_a, p_a = wilcoxon_exact(model_gains, a_means)
            print(f"[C2-DM] Wilcoxon exact vs Group A:  T={T_a:.1f}, p={p_a:.6f}", flush=True)
    else:
        print(f"[C2-DM] Warning: {main_csv} not found, skipping Wilcoxon tests", flush=True)


if __name__ == '__main__':
    main()
