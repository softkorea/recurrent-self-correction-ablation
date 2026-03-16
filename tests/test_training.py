"""Phase 2: 학습 테스트.

TDD — 테스트 먼저, 구현 나중.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import (
    generate_data,
    softmax,
    cross_entropy_loss,
    compute_loss_and_gradients,
    compute_batch_loss_and_gradients,
    gradient_check,
    train,
    evaluate_accuracy_at_timestep,
)


# ──────────────────────────────────────────────
# 1. 데이터 생성
# ──────────────────────────────────────────────
def test_generate_data_shape():
    """데이터 shape 및 one-hot 타겟 확인."""
    X, y = generate_data(n_samples=50, noise_level=0.3, seed=0)
    assert X.shape == (50, 10)
    assert y.shape == (50, 5)
    # 각 타겟은 one-hot
    assert np.allclose(y.sum(axis=1), 1.0)
    assert np.all((y == 0) | (y == 1))


def test_generate_data_reproducibility():
    """같은 seed → 같은 데이터."""
    X1, y1 = generate_data(50, 0.3, seed=42)
    X2, y2 = generate_data(50, 0.3, seed=42)
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_generate_data_class_balance():
    """클래스 분포가 대략 균등."""
    X, y = generate_data(500, 0.3, seed=0)
    counts = y.sum(axis=0)
    # 각 클래스 50~150 범위 (기대값 100)
    assert np.all(counts >= 50) and np.all(counts <= 150)


# ──────────────────────────────────────────────
# 2. 소프트맥스 / 손실함수
# ──────────────────────────────────────────────
def test_softmax_valid_distribution():
    """softmax → 유효한 확률 분포."""
    p = softmax(np.array([2.0, 1.0, 0.1, -1.0, 3.5]))
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_softmax_numerical_stability():
    """큰 값에서도 softmax가 안정적."""
    p = softmax(np.array([1000.0, 1001.0, 999.0, 1000.5, 998.0]))
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_cross_entropy_loss_positive():
    """CE loss는 양수."""
    output = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = cross_entropy_loss(output, target)
    assert loss > 0


def test_cross_entropy_perfect_prediction():
    """정확한 예측에서 CE loss 최소."""
    # 정답 클래스에 큰 logit
    output_good = np.array([10.0, -10.0, -10.0, -10.0, -10.0])
    output_bad = np.array([-10.0, 10.0, -10.0, -10.0, -10.0])
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert cross_entropy_loss(output_good, target) < cross_entropy_loss(output_bad, target)


# ──────────────────────────────────────────────
# 3. Gradient Check
# ──────────────────────────────────────────────
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("skip_conn", [False, True])
def test_gradient_check(recurrent, skip_conn):
    """모든 아키텍처 변형에 대한 수치 미분 검증."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5,
                       seed=7, skip_connection=skip_conn)
    if not recurrent:
        net.disable_recurrent_loop()

    rng = np.random.RandomState(7)
    x = rng.randn(10)
    target = np.zeros(5)
    target[2] = 1.0

    max_rel_error = gradient_check(net, x, target, T=3)
    print(f"Gradient check (rec={recurrent}, skip={skip_conn}): {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, \
        f"Gradient check failed (rec={recurrent}, skip={skip_conn}): {max_rel_error:.2e}"


# ──────────────────────────────────────────────
# 4. 학습
# ──────────────────────────────────────────────
def test_training_reduces_loss():
    """학습이 loss를 50% 이상 줄이는지."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X, y = generate_data(n_samples=100, noise_level=0.3, seed=42)

    loss_before, _ = compute_batch_loss_and_gradients(net, X, y, T=3)
    train(net, X, y, epochs=300, lr=0.01, T=3)
    loss_after, _ = compute_batch_loss_and_gradients(net, X, y, T=3)

    print(f"Loss: {loss_before:.4f} → {loss_after:.4f} ({loss_after/loss_before:.1%})")
    assert loss_after < loss_before * 0.5


# ──────────────────────────────────────────────
# 5. Self-Correction (핵심 가설 전제)
# ──────────────────────────────────────────────
def test_self_correction_occurs():
    """훈련 후 acc_t3 > acc_t1 — 자기 교정 발생."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X_train, y_train = generate_data(n_samples=200, noise_level=0.5, seed=42)
    X_test, y_test = generate_data(n_samples=100, noise_level=0.5, seed=99)

    train(net, X_train, y_train, epochs=500, lr=0.01, T=3)

    acc_t1 = evaluate_accuracy_at_timestep(net, X_test, y_test, t=1)
    acc_t3 = evaluate_accuracy_at_timestep(net, X_test, y_test, t=3)
    gain = acc_t3 - acc_t1

    print(f"t=1 acc: {acc_t1:.3f}, t=3 acc: {acc_t3:.3f}, gain: {gain:.3f}")
    assert acc_t3 > acc_t1, f"자기 교정 없음: t=1={acc_t1:.3f}, t=3={acc_t3:.3f}"


def test_state_isolation_between_samples():
    """샘플 간 상태 누출이 없는지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X, y = generate_data(n_samples=10, noise_level=0.3, seed=42)

    # 순서대로 처리
    outputs_forward = []
    for i in range(len(X)):
        outs, _ = net.forward_sequence(X[i], T=3)
        outputs_forward.append([o.copy() for o in outs])

    # 역순으로 처리 — forward_sequence가 reset하므로 결과 동일해야 함
    outputs_reverse = []
    for i in reversed(range(len(X))):
        outs, _ = net.forward_sequence(X[i], T=3)
        outputs_reverse.append([o.copy() for o in outs])
    outputs_reverse.reverse()

    for i in range(len(X)):
        for t in range(3):
            assert np.allclose(outputs_forward[i][t], outputs_reverse[i][t]), \
                f"샘플 {i}, t={t+1}: 처리 순서에 따라 출력이 다름 — 상태 누출"
