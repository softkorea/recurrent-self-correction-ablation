"""측정 모듈 — 실험 메트릭 계산.

Gemini review 반영: compute_all_metrics 단일 패스 (O(N) → O(N) from O(6N)).
뉴런 중요도 분석 함수 추가.
"""

import numpy as np
from src.training import softmax, evaluate_accuracy_at_timestep


def compute_correction_gain(net, X, y):
    """correction_gain = acc_t3 - acc_t1."""
    acc_t1 = evaluate_accuracy_at_timestep(net, X, y, t=1)
    acc_t3 = evaluate_accuracy_at_timestep(net, X, y, t=3)
    return acc_t3 - acc_t1


def compute_recurrent_contribution_norm(net, X):
    """||W_rec @ feedback|| 평균 (t=2, t=3에서의 피드백 기여 크기)."""
    norms = []
    for i in range(len(X)):
        outputs, caches = net.forward_sequence(X[i], T=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ net.W_rec
            norms.append(np.linalg.norm(contrib))
    return float(np.mean(norms))


def compute_step_delta(net, X):
    """||y_t - y_{t-1}|| 평균 (t=2,3에서의 출력 변화량)."""
    deltas = []
    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        for t in range(1, 3):
            delta = np.linalg.norm(outputs[t] - outputs[t - 1])
            deltas.append(delta)
    return float(np.mean(deltas))


def compute_ece(net, X, y, n_bins=10):
    """Expected Calibration Error.

    softmax 최대값을 confidence로 사용.
    """
    confidences = []
    accuracies = []

    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        probs = softmax(outputs[2])
        conf = np.max(probs)
        pred = np.argmax(probs)
        true = np.argmax(y[i])
        confidences.append(conf)
        accuracies.append(float(pred == true))

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += mask.sum() / len(X) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_all_metrics(net, X, y):
    """단일 패스로 모든 메트릭을 계산.

    Gemini review 반영: forward_sequence를 샘플당 1회만 호출 (기존 6회 → 1회).

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = net.forward_sequence(X[i], T=3)
        true_cls = np.argmax(y[i])

        # accuracy at each timestep
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta (t=2, t=3)
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    # ECE
    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)),
        'delta_norm': float(np.mean(deltas)),
    }


def compute_all_metrics_with_clone(target_net, clone_net, X, y):
    """Clone feedback를 사용한 메트릭 계산.

    target_net의 피드백을 clone_net의 출력으로 대체.
    compute_all_metrics와 동일한 dict 반환.
    """
    from src.ablation import forward_sequence_with_clone

    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = forward_sequence_with_clone(target_net, clone_net, X[i], T=3)
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3) — uses per-timestep caches
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ target_net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)) if r_norms else 0.0,
        'delta_norm': float(np.mean(deltas)) if deltas else 0.0,
    }


# ──────────────────────────────────────────────
# Wilcoxon Signed-Rank Exact Test (scipy 불필요)
# ──────────────────────────────────────────────

def wilcoxon_exact(x, y):
    """Wilcoxon signed-rank exact test (paired, two-sided).

    N≤25이면 2^N 전수 열거로 exact p-value 계산.
    scipy.stats.wilcoxon과 동일한 결과 (검증 완료).

    Args:
        x, y: 1-D array-like paired samples (같은 길이)

    Returns:
        (T, p_value) where T = min(T+, T-)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x - y

    # Remove zero differences
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Handle ties: assign average rank
    sorted_abs = abs_d[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[order[i:j]])
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    # T+ and T-
    T_plus = float(np.sum(ranks[d > 0]))
    T_minus = float(np.sum(ranks[d < 0]))
    T = min(T_plus, T_minus)

    # Exact enumeration: 2^n sign assignments
    rank_sum_total = float(ranks.sum())
    n_perms = 1 << n  # 2^n
    count = 0
    for mask in range(n_perms):
        t_val = 0.0
        for bit in range(n):
            if mask & (1 << bit):
                t_val += ranks[bit]
        t_min = min(t_val, rank_sum_total - t_val)
        if t_min <= T:
            count += 1

    p = count / n_perms
    return float(T), float(p)


# ──────────────────────────────────────────────
# Neuron Importance (Heatmap 데이터 생성)
# ──────────────────────────────────────────────

def compute_neuron_importance(net, X, y):
    """각 히든 뉴런의 intelligence / self-correction 중요도 측정.

    H1 neurons: decoupled ablation to avoid intelligence→correction confound.
      - Intelligence: ablate feedforward input (W_ih1 + bias) only, measure Δacc_t1
      - Correction: ablate recurrent input (W_rec) only, measure Δgain
    H2 neurons: full knockout (W_h1h2 + bias), since H2 has no direct W_rec input.
      - Note: H2 correction importance may be confounded with intelligence importance.

    Returns:
        intelligence: dict {neuron_id: importance}
        correction: dict {neuron_id: importance}
    """
    baseline = compute_all_metrics(net, X, y)
    baseline_acc_t1 = baseline['acc_t1']
    baseline_gain = baseline['gain']

    intelligence = {}
    correction = {}

    # Hidden1 뉴런 (0~9) — decoupled ablation
    for idx in range(net.hidden1):
        # 1. Intelligence: ablate feedforward input only (W_ih1 + bias)
        col_ih1 = net.W_ih1[:, idx].copy()
        b_h1_val = net.b_h1[idx]
        net.W_ih1[:, idx] = 0.0
        net.b_h1[idx] = 0.0

        m_intel = compute_all_metrics(net, X, y)
        intelligence[f'h1_{idx}'] = baseline_acc_t1 - m_intel['acc_t1']

        # Restore feedforward
        net.W_ih1[:, idx] = col_ih1
        net.b_h1[idx] = b_h1_val

        # 2. Correction: ablate recurrent input only (W_rec)
        col_rec = net.W_rec[:, idx].copy()
        net.W_rec[:, idx] = 0.0

        m_corr = compute_all_metrics(net, X, y)
        correction[f'h1_{idx}'] = baseline_gain - m_corr['gain']

        # Restore recurrent
        net.W_rec[:, idx] = col_rec

    # Hidden2 뉴런 (0~9) — full knockout (no direct W_rec input)
    for idx in range(net.hidden2):
        col_h1h2 = net.W_h1h2[:, idx].copy()
        b_h2_val = net.b_h2[idx]

        net.W_h1h2[:, idx] = 0.0
        net.b_h2[idx] = 0.0

        m = compute_all_metrics(net, X, y)
        intelligence[f'h2_{idx}'] = baseline_acc_t1 - m['acc_t1']
        correction[f'h2_{idx}'] = baseline_gain - m['gain']

        net.W_h1h2[:, idx] = col_h1h2
        net.b_h2[idx] = b_h2_val

    return intelligence, correction
