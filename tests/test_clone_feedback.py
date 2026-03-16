"""Phase 3+: Group C2 (Clone Feedback) 테스트.

Clone Feedback: 각 target 모델의 피드백을 독립 훈련된 donor 모델의 출력으로 대체.
같은 분포, 같은 구조, 다른 "self" — OOD 비판 차단.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import create_trained_network, forward_sequence_with_clone


@pytest.fixture
def two_trained_nets():
    """두 개의 서로 다른 seed로 학습된 네트워크."""
    net_a = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    net_b = create_trained_network(seed=1, epochs=200, noise_level=0.3)
    return net_a, net_b


# ──────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────

def test_clone_feedback_uses_different_model_output(two_trained_nets):
    """Clone feedback가 실제로 다른 모델의 출력을 사용하는지 확인.

    target의 자체 forward_sequence 결과와
    clone feedback을 사용한 결과가 달라야 함.
    """
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    # Normal forward
    normal_outputs, _ = target.forward_sequence(x, T=3)

    # Clone feedback forward
    clone_outputs, _ = forward_sequence_with_clone(target, clone, x, T=3)

    # t=1은 동일해야 함 (피드백 없음)
    assert np.allclose(normal_outputs[0], clone_outputs[0]), \
        "t=1 출력이 달라짐 — 피드백 없는 시점인데 변경됨"

    # t=2 이후는 달라야 함 (다른 모델의 피드백 사용)
    assert not np.allclose(normal_outputs[1], clone_outputs[1]), \
        "Clone feedback가 t=2 출력을 변경하지 않음"


def test_clone_feedback_preserves_original_weights(two_trained_nets):
    """Clone feedback 실행 후 target 모델의 가중치가 변하지 않아야 함."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    w_before = {k: v.copy() for k, v in target.get_all_weights().items()}

    forward_sequence_with_clone(target, clone, x, T=3)

    w_after = target.get_all_weights()
    for k in w_before:
        assert np.allclose(w_before[k], w_after[k]), \
            f"Clone feedback가 {k} 가중치를 변경함"


def test_clone_output_is_valid(two_trained_nets):
    """Clone 모델의 출력이 유효한지 — 0이 아니고, 합리적 범위."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    clone.reset_state()
    y = clone.forward(x)

    assert not np.allclose(y, 0), "Clone 출력이 전부 0"
    assert np.all(np.isfinite(y)), "Clone 출력에 inf/nan 포함"
    assert np.linalg.norm(y) > 0.01, "Clone 출력 노름이 너무 작음"


def test_clone_feedback_deterministic(two_trained_nets):
    """같은 입력, 같은 모델 쌍 → 같은 결과."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    out1, _ = forward_sequence_with_clone(target, clone, x, T=3)
    out2, _ = forward_sequence_with_clone(target, clone, x, T=3)

    for t in range(3):
        assert np.allclose(out1[t], out2[t]), \
            f"t={t+1}에서 결과가 비결정적"


def test_clone_vs_self_feedback():
    """자기 자신을 clone으로 쓰면 normal forward와 동일해야 함."""
    net = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    x = np.random.RandomState(42).randn(10)

    # Normal
    normal_outputs, _ = net.forward_sequence(x, T=3)

    # Self-clone (same model as both target and clone)
    # Need a second copy since forward modifies state
    net2 = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    clone_outputs, _ = forward_sequence_with_clone(net, net2, x, T=3)

    for t in range(3):
        assert np.allclose(normal_outputs[t], clone_outputs[t], atol=1e-10), \
            f"Self-clone이 normal forward와 다름 at t={t+1}"
