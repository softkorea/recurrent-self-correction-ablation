"""시각화 모듈 — 실험 결과 그래프 생성.

그래프 5종:
1. network_map.png — 뉴런 + 연결 시각화
2. ablation_comparison.png — 그룹별 correction_gain 비교
3. accuracy_distribution.png — 랜덤 절단 분포 + 재귀/스크램블 위치
4. neuron_importance_heatmap.png — intelligence vs self-correction importance
5. noise_sweep_curve.png — 노이즈 레벨별 correction_gain 곡선
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def ensure_results_dir(path='results'):
    os.makedirs(path, exist_ok=True)
    return path


# ──────────────────────────────────────────────
# 1. Network Map
# ──────────────────────────────────────────────

def plot_network_map(net, save_path='results/network_map.png'):
    """전체 뉴런 + 연결 시각화. 재귀 연결 빨간색."""
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RecurrentMLP Network Map (35 neurons)', fontsize=14)

    layers = {
        'Input': (0, 10),
        'Hidden1': (1.5, 10),
        'Hidden2': (3, 10),
        'Output': (4.5, 5),
    }

    neuron_pos = {}

    for layer_name, (x, n) in layers.items():
        for i in range(n):
            y = i * (10 / max(n - 1, 1))
            neuron_pos[(layer_name, i)] = (x, y)
            color = '#4ECDC4' if layer_name != 'Output' else '#FF6B6B'
            ax.add_patch(plt.Circle((x, y), 0.15, color=color, ec='black', lw=0.5, zorder=5))

        ax.text(x, -0.7, f'{layer_name}\n({n})', ha='center', fontsize=9)

    weights = net.get_all_weights()

    # Feedforward connections (gray, thin)
    connections = [
        ('Input', 'Hidden1', weights['input_to_h1']),
        ('Hidden1', 'Hidden2', weights['h1_to_h2']),
        ('Hidden2', 'Output', weights['h2_to_output']),
    ]

    for src_layer, dst_layer, W in connections:
        n_src = W.shape[0]
        n_dst = W.shape[1]
        max_w = max(np.abs(W).max(), 1e-6)
        for i in range(n_src):
            for j in range(n_dst):
                if abs(W[i, j]) > 0.01 * max_w:
                    x1, y1 = neuron_pos[(src_layer, i)]
                    x2, y2 = neuron_pos[(dst_layer, j)]
                    alpha = min(abs(W[i, j]) / max_w, 1.0) * 0.3
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=alpha, lw=0.3, zorder=1)

    # Recurrent connections (red)
    W_rec = weights['recurrent']
    max_wr = max(np.abs(W_rec).max(), 1e-6)
    for i in range(W_rec.shape[0]):
        for j in range(W_rec.shape[1]):
            if abs(W_rec[i, j]) > 0.01 * max_wr:
                x1, y1 = neuron_pos[('Output', i)]
                x2, y2 = neuron_pos[('Hidden1', j)]
                alpha = min(abs(W_rec[i, j]) / max_wr, 1.0) * 0.5
                ax.annotate('', xy=(x2 + 0.15, y2), xytext=(x1 - 0.15, y1),
                           arrowprops=dict(arrowstyle='->', color='red',
                                          alpha=alpha, lw=0.8),
                           zorder=3)

    ax.text(3.0, 10.7, '→ Red: Recurrent (output→h1)', color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 2. Ablation Comparison
# ──────────────────────────────────────────────

def plot_ablation_comparison(results_df, save_path='results/ablation_comparison.png'):
    """그룹별 correction_gain 막대 그래프.

    Args:
        results_df: dict with keys = group names,
                    values = list of gain values (across models/seeds)
    """
    ensure_results_dir(os.path.dirname(save_path))

    groups = list(results_df.keys())
    means = [np.mean(results_df[g]) for g in groups]
    stds = [np.std(results_df[g]) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    x = np.arange(len(groups))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors[:len(groups)], edgecolor='black', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Correction Gain (acc_t3 - acc_t1)', fontsize=11)
    ax.set_title('Ablation Comparison: Correction Gain by Group\n(10 models, mean ± std)',
                 fontsize=13)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.grid(axis='y', alpha=0.3)

    # 값 표시
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 3. Accuracy Distribution
# ──────────────────────────────────────────────

def plot_accuracy_distribution(random_gains, recurrent_gain, scrambled_gain,
                                save_path='results/accuracy_distribution.png'):
    """랜덤 절단 분포 히스토그램 + 재귀/스크램블 위치 표시.

    Args:
        random_gains: list of gain values from random ablation (B1)
        recurrent_gain: float, gain after recurrent ablation (A)
        scrambled_gain: float, gain after scrambled feedback (C)
    """
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(random_gains, bins=20, alpha=0.7, color='#3498db',
            edgecolor='black', label='Random ablation (B1)')
    ax.axvline(recurrent_gain, color='red', linestyle='--', lw=2,
               label=f'Recurrent ablation (A): {recurrent_gain:.3f}')
    ax.axvline(scrambled_gain, color='purple', linestyle='--', lw=2,
               label=f'Scrambled feedback (C): {scrambled_gain:.3f}')

    ax.set_xlabel('Correction Gain', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Correction Gain: Random vs Targeted Ablation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 4. Neuron Importance Heatmap
# ──────────────────────────────────────────────

def plot_neuron_importance_heatmap(intelligence_importance, correction_importance,
                                    save_path='results/neuron_importance_heatmap.png'):
    """Intelligence vs Self-correction importance 산점도.

    Args:
        intelligence_importance: dict {neuron_id: importance_value}
        correction_importance: dict {neuron_id: importance_value}
    """
    ensure_results_dir(os.path.dirname(save_path))

    neuron_ids = sorted(intelligence_importance.keys())
    x = [intelligence_importance[n] for n in neuron_ids]
    y = [correction_importance[n] for n in neuron_ids]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x, y, c='#e74c3c', s=60, alpha=0.7, edgecolors='black')

    for i, nid in enumerate(neuron_ids):
        ax.annotate(str(nid), (x[i], y[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    ax.set_xlabel('Intelligence Importance\n(acc_t1 drop when neuron ablated)', fontsize=11)
    ax.set_ylabel('Self-Correction Importance\n(correction_gain drop when neuron ablated at t≥2)',
                  fontsize=11)
    ax.set_title('Neuron Importance: Intelligence vs Self-Correction', fontsize=13)

    # 4분면 가이드
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 5. Noise Sweep Curve
# ──────────────────────────────────────────────

def plot_noise_sweep(sweep_data, save_path='results/noise_sweep_curve.png'):
    """노이즈 레벨별 correction_gain 곡선.

    Args:
        sweep_data: dict {group_name: {noise_level: [gain_values]}}
    """
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    # 그룹 코드 → 스타일 매핑 (ChatGPT review P0-2: 코드/라벨 불일치 수정)
    GROUP_STYLE = {
        'Baseline': ('#2ecc71', 'o', 'Baseline'),
        'A':        ('#e74c3c', 's', 'A (Recurrent Cut)'),
        'B1':       ('#3498db', '^', 'B1 (Random Cut)'),
        'B2':       ('#8e44ad', 'v', 'B2 (Structural Cut)'),
        'C1':       ('#9b59b6', 'D', 'C1 (Permutation)'),
        'C2':       ('#f39c12', 'P', 'C2 (Clone Feedback)'),
        'D':        ('#95a5a6', 'x', 'D (Feedforward)'),
        "D'":       ('#1abc9c', '+', "D' (Param-matched FF)"),
    }

    for group_code, noise_gains in sweep_data.items():
        noise_levels = sorted(noise_gains.keys())
        means = [np.mean(noise_gains[nl]) for nl in noise_levels]
        stds = [np.std(noise_gains[nl]) for nl in noise_levels]

        c, m, label = GROUP_STYLE.get(group_code, ('gray', 'o', group_code))
        ax.errorbar(noise_levels, means, yerr=stds, label=label,
                    color=c, marker=m, capsize=3, lw=1.5, markersize=6)

    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Correction Gain', fontsize=11)
    ax.set_title('Noise Sweep: Correction Gain by Group and Noise Level\n'
                 '(10 models, mean ± std)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', lw=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
