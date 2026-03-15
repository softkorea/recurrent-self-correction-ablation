"""Ablation 모듈 — 선택적 연결 절단.

Group A: ablate_recurrent — 재귀 가중치 전체 0
Group B1: ablate_random — 랜덤 n개 연결 0
Group B2: ablate_structural — 특정 레이어 전체 0
Group C1: network.enable_scrambled_feedback() 사용
Group C2: forward_sequence_with_clone() — 다른 모델의 출력을 피드백으로 주입
Group C2-norm/affine/multi: aligned clone feedback controls
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

# ──────────────────────────────────────────────
# Group C2: Clone Feedback
# ──────────────────────────────────────────────

def forward_sequence_with_clone(target_net, clone_net, x, T=3):
    """target_net의 피드백을 clone_net의 출력으로 대체하여 forward.

    t=1: 두 모델 모두 독립 forward (피드백 없음)
    t=2,3: target의 _prev_output을 clone의 이전 출력으로 교체

    Args:
        target_net: 평가 대상 모델
        clone_net: 피드백 제공 모델 (다른 seed로 학습된 동일 구조)
        x: 입력 벡터
        T: 타임스텝 수

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    target_net.reset_state()
    clone_net.reset_state()

    target_outputs = []
    target_caches = []
    clone_outputs = []

    for t in range(T):
        clone_y = clone_net.forward(x)
        clone_outputs.append(clone_y.copy())

        if t > 0:
            target_net._prev_output = clone_outputs[t - 1].copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


# ──────────────────────────────────────────────
# C2 Alignment Functions
# ──────────────────────────────────────────────

def align_norm(donor_output, target_output):
    """Norm-matched alignment: scale donor to match target's L2 norm.

    aligned = donor * (||target|| / ||donor||)
    Preserves direction of donor, matches magnitude of target.

    Args:
        donor_output: donor model's output vector
        target_output: target model's own output vector (alignment reference)

    Returns:
        aligned output vector with same direction as donor, same norm as target
    """
    donor_output = np.asarray(donor_output, dtype=np.float64)
    target_output = np.asarray(target_output, dtype=np.float64)

    donor_norm = np.linalg.norm(donor_output)
    target_norm = np.linalg.norm(target_output)

    if donor_norm < 1e-12 or target_norm < 1e-12:
        return np.zeros_like(donor_output)

    return donor_output * (target_norm / donor_norm)


def align_affine(donor_output, target_output):
    """Affine alignment: match mean and std element-wise.

    aligned = (donor - mean(donor)) / std(donor) * std(target) + mean(target)
    Transforms donor to have same mean and std as target.

    Args:
        donor_output: donor model's output vector
        target_output: target model's own output vector (alignment reference)

    Returns:
        aligned output vector matching target's mean and std
    """
    donor_output = np.asarray(donor_output, dtype=np.float64)
    target_output = np.asarray(target_output, dtype=np.float64)

    donor_mean = np.mean(donor_output)
    donor_std = np.std(donor_output)
    target_mean = np.mean(target_output)
    target_std = np.std(target_output)

    if donor_std < 1e-12:
        # Constant donor: return target mean broadcast
        return np.full_like(donor_output, target_mean)

    # Standardize donor, then rescale to target statistics
    aligned = (donor_output - donor_mean) / donor_std * target_std + target_mean
    return aligned


# ──────────────────────────────────────────────
# C2 Aligned Clone Forward
# ──────────────────────────────────────────────

def forward_sequence_with_aligned_clone(target_net, donor_net, x, align_fn, T=3):
    """target_net의 피드백을 align_fn으로 정렬된 donor 출력으로 대체.

    target의 자체 forward를 병렬로 실행하여 alignment reference를 제공.
    t=0: 두 모델 모두 독립 forward (피드백 없음)
    t>0:
      - target_ref = target의 정상 forward에서의 t-1 출력 (alignment reference)
      - donor_prev = donor의 t-1 출력
      - aligned = align_fn(donor_prev, target_ref)
      - target._prev_output = aligned
      - target forward

    Args:
        target_net: 평가 대상 모델
        donor_net: 피드백 제공 모델
        x: 입력 벡터
        align_fn: alignment function (align_norm or align_affine)
        T: 타임스텝 수

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    x = np.asarray(x, dtype=np.float64)

    # Step 1: Run target normally to get reference outputs for alignment
    target_net.reset_state()
    target_ref_outputs = []
    for t in range(T):
        ref_y = target_net.forward(x)
        target_ref_outputs.append(ref_y.copy())

    # Step 2: Run donor normally to get donor outputs
    donor_net.reset_state()
    donor_outputs = []
    for t in range(T):
        donor_y = donor_net.forward(x)
        donor_outputs.append(donor_y.copy())

    # Step 3: Run target with aligned donor feedback
    target_net.reset_state()
    target_outputs = []
    target_caches = []

    for t in range(T):
        if t > 0:
            # Align donor's t-1 output to match target's t-1 reference output
            aligned = align_fn(donor_outputs[t - 1], target_ref_outputs[t - 1])
            target_net._prev_output = aligned.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


def forward_sequence_multi_donor(target_net, donor_nets, x, T=3):
    """Multi-donor ensemble: average output of multiple donors before injection.

    t=0: All forward independently (no feedback)
    t>0: target._prev_output = mean of all donors' t-1 outputs

    Args:
        target_net: 평가 대상 모델
        donor_nets: list of donor models
        x: 입력 벡터
        T: 타임스텝 수

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    x = np.asarray(x, dtype=np.float64)

    # Run all donors normally to collect their outputs
    all_donor_outputs = []
    for donor in donor_nets:
        donor.reset_state()
        donor_outputs = []
        for t in range(T):
            donor_y = donor.forward(x)
            donor_outputs.append(donor_y.copy())
        all_donor_outputs.append(donor_outputs)

    # Run target with averaged donor feedback
    target_net.reset_state()
    target_outputs = []
    target_caches = []

    for t in range(T):
        if t > 0:
            # Average of all donors' t-1 outputs
            avg_output = np.mean(
                [all_donor_outputs[d][t - 1] for d in range(len(donor_nets))],
                axis=0
            )
            target_net._prev_output = avg_output.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


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
