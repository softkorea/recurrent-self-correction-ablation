"""Phase 3: Ablation (절단) 테스트.

그룹: A(재귀절단), B1(랜덤절단), B2(구조적절단),
      C1(permutation feedback), C2(batch-shuffle feedback),
      D(feedforward), D'(param-matched FF)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import (
    ablate_recurrent,
    ablate_random,
    ablate_structural,
    count_zeroed_weights,
    create_trained_network,
)


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
@pytest.fixture
def trained_net():
    """학습된 네트워크 (seed=42)."""
    return create_trained_network(seed=42, epochs=200, noise_level=0.3)


# ──────────────────────────────────────────────
# Group A: 재귀 절단
# ──────────────────────────────────────────────
def test_ablate_recurrent_zeroes_weights(trained_net):
    """재귀 가중치가 실제로 0이 되는지."""
    assert np.any(trained_net.get_all_weights()['recurrent'] != 0)
    ablate_recurrent(trained_net)
    assert np.all(trained_net.get_all_weights()['recurrent'] == 0)


def test_ablate_recurrent_preserves_other_weights(trained_net):
    """재귀 절단이 다른 가중치에 영향 안 줌."""
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    ablate_recurrent(trained_net)
    w_after = trained_net.get_all_weights()
    for k in ['input_to_h1', 'h1_to_h2', 'h2_to_output']:
        assert np.allclose(w_before[k], w_after[k])


# ──────────────────────────────────────────────
# Group B1: 랜덤 절단
# ──────────────────────────────────────────────
def test_ablate_random_correct_count(trained_net):
    """랜덤 절단이 정확한 수의 연결을 끊는지."""
    n_target = 50  # recurrent 가중치 수와 동일
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    ablate_random(trained_net, n_connections=n_target, seed=42)

    zeroed = 0
    w_after = trained_net.get_all_weights()
    for k in w_before:
        # 이전에 0이 아니었는데 지금 0인 것 카운트
        zeroed += np.sum((w_before[k] != 0) & (w_after[k] == 0))
    assert zeroed == n_target, f"Expected {n_target} zeroed, got {zeroed}"


def test_ablate_random_deterministic(trained_net):
    """같은 seed → 같은 절단 결과."""
    net2 = create_trained_network(seed=42, epochs=200, noise_level=0.3)

    ablate_random(trained_net, n_connections=20, seed=7)
    ablate_random(net2, n_connections=20, seed=7)

    for k in trained_net.get_all_weights():
        assert np.allclose(
            trained_net.get_all_weights()[k],
            net2.get_all_weights()[k]
        ), f"seed=7인데 {k} 절단 결과가 다름"


# ──────────────────────────────────────────────
# Group B2: 구조적 절단
# ──────────────────────────────────────────────
def test_ablate_structural():
    """특정 레이어 전체 절단."""
    net = create_trained_network(seed=42, epochs=200, noise_level=0.3)
    ablate_structural(net, layer='h1_to_h2')
    assert np.all(net.get_all_weights()['h1_to_h2'] == 0)
    # 다른 레이어는 유지
    assert np.any(net.get_all_weights()['input_to_h1'] != 0)


# ──────────────────────────────────────────────
# Group C: Scrambled Feedback (네트워크 자체 기능)
# ──────────────────────────────────────────────
def test_scrambled_feedback_preserves_weights(trained_net):
    """Scrambled feedback: 가중치 불변."""
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    trained_net.enable_scrambled_feedback(seed=42)
    w_after = trained_net.get_all_weights()
    for k in w_before:
        assert np.allclose(w_before[k], w_after[k]), \
            f"Scrambled feedback가 {k} 가중치를 변경함"


def test_scrambled_feedback_changes_output(trained_net):
    """Scrambled feedback 활성화 시 t≥2 출력 변화."""
    x = np.random.RandomState(0).randn(10)

    trained_net.reset_state()
    trained_net.forward(x)  # t=1
    y_normal = trained_net.forward(x)  # t=2 (정상 피드백)

    trained_net.reset_state()
    trained_net.enable_scrambled_feedback(seed=42)
    trained_net.forward(x)  # t=1
    y_scrambled = trained_net.forward(x)  # t=2 (스크램블 피드백)

    assert not np.allclose(y_normal, y_scrambled), \
        "Scrambled feedback가 출력을 변경하지 않음"


def test_scrambled_feedback_t1_unchanged(trained_net):
    """Scrambled feedback는 t=1 출력에 영향 없음 (피드백 없으니까)."""
    x = np.random.RandomState(0).randn(10)

    trained_net.reset_state()
    y_normal_t1 = trained_net.forward(x)

    trained_net.reset_state()
    trained_net.enable_scrambled_feedback(seed=42)
    y_scrambled_t1 = trained_net.forward(x)

    trained_net.disable_scrambled_feedback()
    assert np.allclose(y_normal_t1, y_scrambled_t1), \
        "Scrambled feedback가 t=1 출력을 변경함 (버그)"


# ──────────────────────────────────────────────
# 다중 모델 검증
# ──────────────────────────────────────────────
def test_multi_seed_models_differ():
    """다른 seed의 훈련된 모델이 실제로 다른지."""
    net1 = create_trained_network(seed=1, epochs=100, noise_level=0.3)
    net2 = create_trained_network(seed=2, epochs=100, noise_level=0.3)
    w1 = net1.get_all_weights()['input_to_h1']
    w2 = net2.get_all_weights()['input_to_h1']
    assert not np.allclose(w1, w2), "다른 seed인데 학습 후 가중치가 동일"


# ──────────────────────────────────────────────
# 절단 수 통계
# ──────────────────────────────────────────────
def test_count_zeroed_weights(trained_net):
    """가중치 0 카운트 함수."""
    n_before = count_zeroed_weights(trained_net)
    ablate_recurrent(trained_net)
    n_after = count_zeroed_weights(trained_net)
    assert n_after > n_before
    assert n_after - n_before == 50  # W_rec is 5×10
