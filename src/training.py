"""학습 모듈 — 데이터 생성, 3-step unroll backprop, 학습 루프.

CLAUDE.md 제약:
- #6: forward_sequence 시 샘플마다 reset_state()
- #8: 3-step unroll backprop. 수치미분은 gradient check 전용. rel error < 1e-4.
- #4: BPTT 에러 2회 이상 시 수치미분으로 전면 대체.

Gemini review 반영:
- tanh bounded feedback → backward에서 tanh 미분 반영
- 데이터 inter-class ambiguity 추가
- W_skip (Group D') gradient 처리
"""

import numpy as np
from src.network import RecurrentMLP


# ──────────────────────────────────────────────
# 유틸: softmax, cross-entropy
# ──────────────────────────────────────────────

def softmax(x):
    """수치 안정 softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy_loss(output, target):
    """Cross-entropy loss (with softmax)."""
    p = softmax(output)
    return -np.sum(target * np.log(p + 1e-12))


def d_cross_entropy(output, target):
    """∂CE/∂output = softmax(output) - target."""
    return softmax(output) - target


# ──────────────────────────────────────────────
# 데이터 생성
# ──────────────────────────────────────────────

def generate_data(n_samples, noise_level, n_classes=5, input_size=10, seed=0):
    """정적 패턴 분류 데이터 생성.

    5개 기본 패턴: 각 패턴은 2개 인접 차원에 주 신호(1.0),
    인접 클래스와 겹치는 차원에 약한 신호(0.3) — inter-class ambiguity.

    Args:
        n_samples: 샘플 수
        noise_level: 가우시안 노이즈 표준편차
        n_classes: 클래스 수 (기본 5, output_size와 일치)
        input_size: 입력 차원 (기본 10)
        seed: 랜덤 시드

    Returns:
        X: (n_samples, input_size)
        y: (n_samples, n_classes) one-hot
    """
    rng = np.random.RandomState(seed)

    base_patterns = np.zeros((n_classes, input_size))
    for k in range(n_classes):
        base_patterns[k, 2 * k: 2 * k + 2] = 1.0
        # inter-class ambiguity: 인접 클래스 차원에 약한 신호
        base_patterns[k, (2 * k + 2) % input_size] = 0.3
        base_patterns[k, (2 * k - 1) % input_size] = 0.3

    X = np.zeros((n_samples, input_size))
    y = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        cls = rng.randint(n_classes)
        X[i] = base_patterns[cls] + noise_level * rng.randn(input_size)
        y[i, cls] = 1.0

    return X, y


# ──────────────────────────────────────────────
# 3-step unroll backprop
# ──────────────────────────────────────────────

def compute_loss_and_gradients(net, x, target, T=3, time_weights=None):
    """단일 샘플에 대한 loss + analytical gradients (3-step unroll).

    Time-Weighted Loss: t=1은 페널티 없음, t=3에 집중.
    네트워크가 피드백을 통한 자기 교정을 학습하도록 유도.
    feedback = tanh(prev_output / τ).
    """
    if time_weights is None:
        time_weights = [0.0, 0.2, 1.0]

    outputs, caches = net.forward_sequence(x, T=T)

    # 총 loss (가중 합)
    total_loss = 0.0
    for t in range(T):
        total_loss += time_weights[t] * cross_entropy_loss(outputs[t], target)

    # gradient 초기화
    grads = {
        'W_ih1': np.zeros_like(net.W_ih1),
        'b_h1': np.zeros_like(net.b_h1),
        'W_h1h2': np.zeros_like(net.W_h1h2),
        'b_h2': np.zeros_like(net.b_h2),
        'W_h2o': np.zeros_like(net.W_h2o),
        'b_out': np.zeros_like(net.b_out),
        'W_rec': np.zeros_like(net.W_rec),
    }
    if net.W_skip is not None:
        grads['W_skip'] = np.zeros_like(net.W_skip)

    # backward: 마지막 타임스텝부터 역순으로
    d_output_future = np.zeros(net.output_size)

    for t in reversed(range(T)):
        cache = caches[t]

        # loss gradient at this timestep (시간 가중)
        d_out = time_weights[t] * d_cross_entropy(outputs[t], target)

        # + gradient from future timestep via recurrent
        d_out = d_out + d_output_future

        # ── skip connection (Group D') ──
        if net.W_skip is not None:
            grads['W_skip'] += np.outer(cache['x'], d_out)

        # ── output layer (linear) ──
        grads['W_h2o'] += np.outer(cache['a_h2'], d_out)
        grads['b_out'] += d_out

        d_a_h2 = net.W_h2o @ d_out
        d_z_h2 = d_a_h2 * (cache['z_h2'] > 0).astype(np.float64)

        # ── hidden 2 ──
        grads['W_h1h2'] += np.outer(cache['a_h1'], d_z_h2)
        grads['b_h2'] += d_z_h2

        d_a_h1 = net.W_h1h2 @ d_z_h2
        d_z_h1 = d_a_h1 * (cache['z_h1'] > 0).astype(np.float64)

        # ── hidden 1 ──
        grads['W_ih1'] += np.outer(cache['x'], d_z_h1)
        grads['b_h1'] += d_z_h1
        grads['W_rec'] += np.outer(cache['feedback'], d_z_h1)

        # propagate gradient to previous timestep's output
        # feedback = tanh(prev_output / τ) → d/d(prev) = (1 - feedback²) / τ
        # 재귀 비활성화 시 미래→과거 그래디언트 차단 (Gemini review #2)
        if net._recurrent_enabled:
            d_feedback = net.W_rec @ d_z_h1
            d_output_future = d_feedback * (1.0 - cache['feedback'] ** 2) / net.feedback_tau
        else:
            d_output_future = np.zeros(net.output_size)

    return total_loss, grads


def compute_batch_loss_and_gradients(net, X, Y, T=3, time_weights=None):
    """배치 평균 loss + gradients."""
    n = len(X)
    total_loss = 0.0
    batch_grads = None

    for i in range(n):
        loss, grads = compute_loss_and_gradients(net, X[i], Y[i], T, time_weights)
        total_loss += loss
        if batch_grads is None:
            batch_grads = {k: v.copy() for k, v in grads.items()}
        else:
            for k in grads:
                batch_grads[k] += grads[k]

    total_loss /= n
    for k in batch_grads:
        batch_grads[k] /= n

    return total_loss, batch_grads


# ──────────────────────────────────────────────
# Gradient check (수치 미분)
# ──────────────────────────────────────────────

def numerical_gradient(net, x, target, T=3, epsilon=1e-5):
    """중심 차분법으로 수치 gradient 계산."""
    num_grads = {}
    time_weights = [0.0, 0.2, 1.0]

    params = [
        ('W_ih1', net.W_ih1),
        ('b_h1', net.b_h1),
        ('W_h1h2', net.W_h1h2),
        ('b_h2', net.b_h2),
        ('W_h2o', net.W_h2o),
        ('b_out', net.b_out),
        ('W_rec', net.W_rec),
    ]
    if net.W_skip is not None:
        params.append(('W_skip', net.W_skip))

    for name, param in params:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + epsilon
            outputs_plus, _ = net.forward_sequence(x, T)
            loss_plus = sum(
                time_weights[t] * cross_entropy_loss(outputs_plus[t], target)
                for t in range(T)
            )

            param[idx] = old_val - epsilon
            outputs_minus, _ = net.forward_sequence(x, T)
            loss_minus = sum(
                time_weights[t] * cross_entropy_loss(outputs_minus[t], target)
                for t in range(T)
            )

            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            param[idx] = old_val

            it.iternext()

        num_grads[name] = grad

    return num_grads


def gradient_check(net, x, target, T=3, epsilon=1e-5):
    """Analytical vs numerical gradient 비교. max relative error 반환."""
    _, ana_grads = compute_loss_and_gradients(net, x, target, T)
    num_grads = numerical_gradient(net, x, target, T, epsilon)

    max_rel_error = 0.0
    for k in ana_grads:
        a = ana_grads[k].ravel()
        n = num_grads[k].ravel()
        for i in range(len(a)):
            denom = max(abs(a[i]), abs(n[i]), 1e-8)
            rel_error = abs(a[i] - n[i]) / denom
            max_rel_error = max(max_rel_error, rel_error)

    return max_rel_error


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def train(net, X, y, epochs=100, lr=0.01, T=3, verbose=False, time_weights=None):
    """Full-batch SGD로 학습.

    Returns:
        loss_history: 에폭별 loss 리스트
    """
    loss_history = []

    for epoch in range(epochs):
        loss, grads = compute_batch_loss_and_gradients(net, X, y, T, time_weights)
        loss_history.append(loss)

        # SGD update
        net.W_ih1  -= lr * grads['W_ih1']
        net.b_h1   -= lr * grads['b_h1']
        net.W_h1h2 -= lr * grads['W_h1h2']
        net.b_h2   -= lr * grads['b_h2']
        net.W_h2o  -= lr * grads['W_h2o']
        net.b_out  -= lr * grads['b_out']
        net.W_rec  -= lr * grads['W_rec']
        if 'W_skip' in grads:
            net.W_skip -= lr * grads['W_skip']

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.4f}")

    return loss_history


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate_accuracy_at_timestep(net, X, y, t):
    """특정 타임스텝에서의 분류 정확도.

    Args:
        t: 타임스텝 (1-indexed). t=1 초기 예측, t=3 최종.
    """
    correct = 0
    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        pred = np.argmax(outputs[t - 1])
        true = np.argmax(y[i])
        if pred == true:
            correct += 1
    return correct / len(X)
