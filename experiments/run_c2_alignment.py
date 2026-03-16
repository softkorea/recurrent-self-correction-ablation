"""C2 Alignment Controls 실험 실행 (병렬화).

C2(clone feedback)의 성능 저하가 "not-self" 때문인지
"not calibrated the way W_rec expects" 때문인지 분리하는 세 가지 정렬 조건:
  - C2-norm: donor 출력을 target의 L2 norm에 맞춤
  - C2-affine: donor 출력을 target의 mean/std에 맞춤
  - C2-multi: 5개 donor 출력의 앙상블 평균을 피드백으로 사용

결과는 별도 CSV(results/c2_alignment.csv)에 저장.
메인 raw_metrics.csv는 수정하지 않음.

병렬화 전략:
- noise_level 단위로 병렬 (같은 noise에서 모든 모델 학습 후 평가)
- n_workers = cpu_count() - 4
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
from src.ablation import align_norm, align_affine
from src.metrics import (
    compute_all_metrics_with_aligned_clone,
    compute_all_metrics_multi_donor,
    wilcoxon_exact,
)

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

DONOR_SEED_OFFSET = 100         # donor seeds: 100-109 (독립)
MULTI_DONOR_SEEDS = [100, 101, 102, 103, 104]  # 5 donors for ensemble


# ──────────────────────────────────────────────
# 워커 함수 (noise_level 단위 독립 처리)
# ──────────────────────────────────────────────

def run_alignment_for_noise(noise_level):
    """한 noise_level에서 target + donor 학습 후 C2 alignment 세 조건 평가.

    Target 모델(seed 0-9)과 독립 donor 모델(seed 100-109)을 1:1 매칭.
    Multi-donor는 seed 100-104 5개 모델 앙상블.

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
    donors = []
    for seed in range(N_MODELS):
        donor_seed = seed + DONOR_SEED_OFFSET
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=donor_seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=donor_seed)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        donors.append(net)

    # Multi-donor 모델 학습 (seed 100-104, 5개)
    multi_donors = []
    for donor_seed in MULTI_DONOR_SEEDS:
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=donor_seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=donor_seed)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        multi_donors.append(net)

    # 각 target 모델에 대해 세 가지 alignment 조건 평가
    for seed in range(N_MODELS):
        target = targets[seed]
        donor = donors[seed]
        X_test, y_test = test_data[seed]

        # C2-norm: norm-matched alignment
        metrics_norm = compute_all_metrics_with_aligned_clone(
            target, donor, X_test, y_test, align_norm
        )
        rows.append({
            'seed_model': seed,
            'group': 'C2-norm',
            'seed_ablation': seed + DONOR_SEED_OFFSET,
            'noise_level': noise_level,
            **{k: metrics_norm[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                              'gain', 'ece', 'r_norm', 'delta_norm']}
        })

        # C2-affine: affine-matched alignment
        metrics_affine = compute_all_metrics_with_aligned_clone(
            target, donor, X_test, y_test, align_affine
        )
        rows.append({
            'seed_model': seed,
            'group': 'C2-affine',
            'seed_ablation': seed + DONOR_SEED_OFFSET,
            'noise_level': noise_level,
            **{k: metrics_affine[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                                'gain', 'ece', 'r_norm', 'delta_norm']}
        })

        # C2-multi: multi-donor ensemble
        metrics_multi = compute_all_metrics_multi_donor(
            target, multi_donors, X_test, y_test
        )
        rows.append({
            'seed_model': seed,
            'group': 'C2-multi',
            'seed_ablation': -1,  # ensemble, no single donor seed
            'noise_level': noise_level,
            **{k: metrics_multi[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                               'gain', 'ece', 'r_norm', 'delta_norm']}
        })

    return rows


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def run_c2_alignment_experiment():
    os.makedirs('results', exist_ok=True)

    n_workers = max(1, mp.cpu_count() - 4)

    print(f"[C2-Align] Starting: {len(NOISE_LEVELS)} noise levels on {n_workers} workers "
          f"(cpu_count={mp.cpu_count()}, reserved 4)", flush=True)
    print(f"[C2-Align] Conditions: C2-norm, C2-affine, C2-multi "
          f"({len(MULTI_DONOR_SEEDS)} donors)", flush=True)

    # 병렬 실행
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_alignment_for_noise, NOISE_LEVELS),
                              total=len(NOISE_LEVELS), desc="C2 Alignment"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_alignment_for_noise, NOISE_LEVELS)
            for batch in results:
                all_rows.extend(batch)

    print(f"[C2-Align] Collected {len(all_rows)} rows", flush=True)

    # 별도 CSV에 저장
    csv_path = 'results/c2_alignment.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[C2-Align] Saved {len(all_rows)} rows to {csv_path}", flush=True)

    # ──────────────────────────────────────────────
    # 통계 요약
    # ──────────────────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print("C2 ALIGNMENT RESULTS SUMMARY (model-level aggregation, N=10)", flush=True)
    print('='*70, flush=True)

    def aggregate(rows, noise=0.5):
        """noise 레벨에서 그룹별 모델별 평균 gain."""
        raw = defaultdict(lambda: defaultdict(list))
        for r in rows:
            nl = float(r['noise_level']) if isinstance(r['noise_level'], str) else r['noise_level']
            if nl == noise:
                g = r['group']
                gain = float(r['gain']) if isinstance(r['gain'], str) else r['gain']
                sm = int(r['seed_model']) if isinstance(r['seed_model'], str) else r['seed_model']
                raw[g][sm].append(gain)
        return {g: [np.mean(raw[g][s]) for s in sorted(raw[g].keys())] for g in raw}

    model_gains = aggregate(all_rows, noise=0.5)

    # Load baseline and C2 from main experiment for comparison
    main_csv = 'results/raw_metrics.csv'
    if os.path.exists(main_csv):
        main_rows = []
        with open(main_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                main_rows.append(r)
        main_gains = aggregate(main_rows, noise=0.5)
        model_gains.update(main_gains)

    print(f"\n  {'Group':12s}  {'gain (mean+/-std)':>20s}  {'n':>3s}", flush=True)
    print(f"  {'-'*12}  {'-'*20}  {'-'*3}", flush=True)

    display_order = ['Baseline', 'A', 'C1', 'C2', 'C2-norm', 'C2-affine', 'C2-multi']
    for g in display_order:
        if g in model_gains:
            gains = model_gains[g]
            print(f"  {g:12s}  {np.mean(gains):+.4f}+/-{np.std(gains):.4f}"
                  f"          {len(gains):3d}", flush=True)

    # Bootstrap 95% CI
    rng = np.random.RandomState(999)
    print("\n95% Bootstrap CI (noise=0.5, model-level):", flush=True)
    for g in display_order:
        if g in model_gains and len(model_gains[g]) > 1:
            gains = np.array(model_gains[g])
            boot = [np.mean(rng.choice(gains, len(gains), replace=True)) for _ in range(10000)]
            print(f"  {g:12s}: [{np.percentile(boot, 2.5):+.4f}, "
                  f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Holm-Bonferroni corrected p-values
    print("\nHolm-Bonferroni corrected p-values (noise=0.5):", flush=True)
    baseline_gains = np.array(model_gains.get('Baseline', []))

    # Test 1: Each alignment condition vs Baseline
    p_values = {}
    for g in ['C2', 'C2-norm', 'C2-affine', 'C2-multi']:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(baseline_gains) > 1:
                _, p = wilcoxon_exact(baseline_gains, g_gains)
                p_values[f'Baseline vs {g}'] = p

    # Test 2: Each alignment condition vs C2
    c2_gains = np.array(model_gains.get('C2', []))
    for g in ['C2-norm', 'C2-affine', 'C2-multi']:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(c2_gains) > 1:
                _, p = wilcoxon_exact(c2_gains, g_gains)
                p_values[f'C2 vs {g}'] = p

    if p_values:
        sorted_ps = sorted(p_values.items(), key=lambda x: x[1])
        m_comp = len(sorted_ps)
        prev_adj_p = 0.0
        for rank, (label, p) in enumerate(sorted_ps):
            adj_p = min(p * (m_comp - rank), 1.0)
            adj_p = max(prev_adj_p, adj_p)
            prev_adj_p = adj_p
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"  {label:25s}: p={adj_p:.4e} {sig}", flush=True)

    # ──────────────────────────────────────────────
    # Noise sweep comparison
    # ──────────────────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print("NOISE SWEEP: C2 alignment conditions", flush=True)
    print('='*70, flush=True)

    sweep = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        g = r['group']
        nl = float(r['noise_level']) if isinstance(r['noise_level'], str) else r['noise_level']
        gain = float(r['gain']) if isinstance(r['gain'], str) else r['gain']
        sweep[g][nl].append(gain)

    print(f"\n  {'Noise':>6s}", end="", flush=True)
    groups = ['C2-norm', 'C2-affine', 'C2-multi']
    for g in groups:
        print(f"  {g:>16s}", end="", flush=True)
    print(flush=True)
    print(f"  {'-'*6}" + f"  {'-'*16}" * len(groups), flush=True)

    for nl in sorted(NOISE_LEVELS):
        print(f"  {nl:6.1f}", end="", flush=True)
        for g in groups:
            vals = sweep[g].get(nl, [])
            if vals:
                print(f"  {np.mean(vals):+.4f}+/-{np.std(vals):.4f}", end="", flush=True)
            else:
                print(f"  {'N/A':>16s}", end="", flush=True)
        print(flush=True)

    # ──────────────────────────────────────────────
    # Interpretation guide
    # ──────────────────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print("INTERPRETATION", flush=True)
    print('='*70, flush=True)
    print("""
  If C2-norm/affine recover gain close to Baseline:
    -> C2 harm was due to statistical mismatch, NOT feedback-contract specificity.
    -> The recurrent pathway mainly cares about signal statistics.

  If C2-norm/affine still show degraded gain (similar to C2):
    -> Feedback-contract specificity confirmed.
    -> The recurrent pathway encodes model-specific information beyond statistics.

  If C2-multi reduces harm vs C2 (but not fully):
    -> Ensemble averaging partially recovers generic self-correction signal.
    -> Individual models learn idiosyncratic feedback patterns.
""", flush=True)

    print(f"[C2-Align] Experiment complete.", flush=True)


if __name__ == '__main__':
    run_c2_alignment_experiment()
