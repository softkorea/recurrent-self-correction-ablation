"""Ablation 모듈 — 선택적 연결 절단.

Group A: ablate_recurrent — 재귀 가중치 전체 0
Group B1: ablate_random — 랜덤 n개 연결 0
Group B2: ablate_structural — 특정 레이어 전체 0
Group C: network.enable_scrambled_feedback() 사용
Group D: RecurrentMLP(recurrent disabled)로 훈련
Group D': param-matched feedforward (skip connection)
"""

import numpy as np
from src.network import RecurrentMLP
from src.training import generate_data, train


# ──────────────────────────────────────────────
# 헬퍼: 학습된 네트워크 생성
# ──────────────────────────────────────────────

def create_trained_network(seed=42, epochs=200, noise_level=0.3,
                           n_samples=200, lr=0.01, T=3):
    """학습된 RecurrentMLP 반환."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed)
    X, y = generate_data(n_samples=n_samples, noise_level=noise_level, seed=seed)
    train(net, X, y, epochs=epochs, lr=lr, T=T)
    return net


# ──────────────────────────────────────────────
# Group A: 재귀 절단
# ──────────────────────────────────────────────

def ablate_recurrent(net):
    """재귀 가중치(W_rec) 전체를 0으로 설정."""
    net.W_rec[:] = 0.0


# ──────────────────────────────────────────────
# Group B1: 랜덤 절단
# ──────────────────────────────────────────────

def ablate_random(net, n_connections, seed=42):
    """전체 가중치에서 무작위 n_connections개를 0으로 설정.

    이미 0인 가중치는 건너뛰고, 정확히 n_connections개의
    non-zero 가중치를 0으로 만듦.
    """
    rng = np.random.RandomState(seed)

    # 모든 가중치를 flat 인덱스로 수집 (non-zero만)
    weight_keys = ['input_to_h1', 'h1_to_h2', 'h2_to_output', 'recurrent']
    weights = net.get_all_weights()

    candidates = []  # (key, flat_index)
    for k in weight_keys:
        w = weights[k]
        nonzero_idx = np.flatnonzero(w)
        for idx in nonzero_idx:
            candidates.append((k, idx))

    assert len(candidates) >= n_connections, \
        f"Non-zero 가중치({len(candidates)})가 요청({n_connections})보다 적음"

    chosen = rng.choice(len(candidates), size=n_connections, replace=False)
    for c in chosen:
        k, idx = candidates[c]
        weights[k].flat[idx] = 0.0


# ──────────────────────────────────────────────
# Group B2: 구조적 절단
# ──────────────────────────────────────────────

def ablate_structural(net, layer):
    """특정 레이어 가중치 전체를 0으로 설정.

    Args:
        layer: 'input_to_h1', 'h1_to_h2', 'h2_to_output', 'recurrent'
    """
    weights = net.get_all_weights()
    assert layer in weights, f"Unknown layer: {layer}"
    weights[layer][:] = 0.0


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def count_zeroed_weights(net):
    """전체 가중치 중 0인 것의 수."""
    total = 0
    for w in net.get_all_weights().values():
        total += np.sum(w == 0)
    return int(total)


def deep_copy_weights(net):
    """모든 가중치의 딥 카피 반환."""
    return {k: v.copy() for k, v in net.get_all_weights().items()}


def restore_weights(net, saved):
    """저장된 가중치를 복원."""
    weights = net.get_all_weights()
    for k in saved:
        weights[k][:] = saved[k]
