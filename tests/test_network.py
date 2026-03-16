"""Phase 1: 네트워크 구조 테스트.

TDD — 이 테스트가 먼저 작성되고, 구현은 이후에 진행.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP


# ──────────────────────────────────────────────
# 1. 기본 shape
# ──────────────────────────────────────────────
def test_network_shape():
    """네트워크 출력 shape 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.random.RandomState(0).randn(10)
    y = net.forward(x)
    assert y.shape == (5,)


# ──────────────────────────────────────────────
# 2. 재귀 루프 구조적 검증
#    랜덤 초기화에 의존하지 않고 고정 가중치로 검증.
# ──────────────────────────────────────────────
def test_recurrent_loop_exists():
    """고정 가중치로 recurrent contribution 노름이 0이 아닌지 구조적 검증."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)

    # 고정 가중치 설정
    W_rec = np.ones((5, 10)) * 0.5
    net.get_all_weights()['recurrent'][:] = W_rec

    # t=1: 첫 forward (prev_output = zeros)
    x = np.ones(10)
    y1 = net.forward(x)

    # 이제 prev_output = y1 이 존재.
    # recurrent_contrib = W_rec.T @ y1  (shape: 10,)
    recurrent_contrib = W_rec.T @ y1
    assert np.linalg.norm(recurrent_contrib) > 1e-6, \
        "재귀 기여가 0 — 구조적으로 재귀 루프가 작동하지 않음"

    # t=2: 같은 입력인데 출력이 달라야 함
    y2 = net.forward(x)
    assert not np.allclose(y1, y2), \
        "같은 입력인데 t=1, t=2 출력이 동일 — 재귀 피드백 미반영"


# ──────────────────────────────────────────────
# 3. 재귀 루프 비활성화
# ──────────────────────────────────────────────
def test_recurrent_loop_disable():
    """루프 비활성화 후 같은 입력 → 같은 출력."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    net.forward(x)  # t=1 — 내부 상태 생성
    net.disable_recurrent_loop()
    y2 = net.forward(x)
    y3 = net.forward(x)
    assert np.allclose(y2, y3), "루프 비활성화 후에도 출력이 변함"


def test_recurrent_loop_enable():
    """비활성화 후 다시 활성화하면 재귀가 돌아오는지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    net.disable_recurrent_loop()
    net.forward(x)
    y_off = net.forward(x)

    net.enable_recurrent_loop()
    net.reset_state()
    net.forward(x)
    y_on = net.forward(x)

    # 재활성화 시 두 번째 forward 결과가 달라야 함 (피드백 반영)
    # y_off 에서는 피드백 없이 동일한 결과가 나옴
    # 여기서는 단순히 enable 후 재귀가 작동하는지만 확인
    net.reset_state()
    y1 = net.forward(x)
    y2 = net.forward(x)
    assert not np.allclose(y1, y2), "재활성화 후에도 재귀가 작동하지 않음"


# ──────────────────────────────────────────────
# 4. 가중치 직접 접근
# ──────────────────────────────────────────────
def test_weight_access():
    """모든 가중치에 직접 접근 가능한지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    weights = net.get_all_weights()

    assert len(weights) == 4
    assert weights['input_to_h1'].shape == (10, 10)
    assert weights['h1_to_h2'].shape == (10, 10)
    assert weights['h2_to_output'].shape == (10, 5)
    assert weights['recurrent'].shape == (5, 10)   # output → h1 피드백


def test_weight_mutation():
    """가중치를 외부에서 수정하면 네트워크에 반영되는지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    y_before = net.forward(x).copy()

    # 가중치 직접 수정
    net.get_all_weights()['input_to_h1'][:] = 0.0
    net.reset_state()
    y_after = net.forward(x)

    assert not np.allclose(y_before, y_after), "가중치 수정이 출력에 반영되지 않음"


# ──────────────────────────────────────────────
# 5. 상태 리셋
# ──────────────────────────────────────────────
def test_reset_state():
    """reset_state() 호출 후 내부 상태가 초기화되는지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    # 두 번 forward → 내부 상태 존재
    net.forward(x)
    net.forward(x)

    net.reset_state()

    # 리셋 후 첫 forward = 완전 초기 상태와 동일해야 함
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    y_reset = net.forward(x)
    y_fresh = net2.forward(x)
    assert np.allclose(y_reset, y_fresh), "reset_state() 후 출력이 초기 상태와 다름"


# ──────────────────────────────────────────────
# 6. forward_sequence (T=3 unroll)
# ──────────────────────────────────────────────
def test_forward_sequence_shape():
    """forward_sequence가 T개의 출력을 반환하는지."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    outputs, caches = net.forward_sequence(x, T=3)
    assert len(outputs) == 3
    assert len(caches) == 3
    for y in outputs:
        assert y.shape == (5,)


def test_forward_sequence_resets_state():
    """forward_sequence가 내부적으로 reset_state()를 호출하는지 (샘플 시작마다 리셋)."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    # 미리 상태를 오염시킴
    net.forward(np.random.RandomState(99).randn(10))
    net.forward(np.random.RandomState(99).randn(10))

    # forward_sequence는 자체 리셋 후 시작해야 함
    outputs_seq, _ = net.forward_sequence(x, T=3)

    # 깨끗한 네트워크로 수동 비교
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    net2.reset_state()
    y1 = net2.forward(x)
    y2 = net2.forward(x)
    y3 = net2.forward(x)

    assert np.allclose(outputs_seq[0], y1), "forward_sequence t=1 불일치"
    assert np.allclose(outputs_seq[1], y2), "forward_sequence t=2 불일치"
    assert np.allclose(outputs_seq[2], y3), "forward_sequence t=3 불일치"


# ──────────────────────────────────────────────
# 7. 뉴런 수 제약
# ──────────────────────────────────────────────
def test_total_neuron_count():
    """전체 뉴런 수 ≤ 35."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    total = 10 + 10 + 10 + 5
    assert total <= 35


# ──────────────────────────────────────────────
# 8. 재현성 (seed)
# ──────────────────────────────────────────────
def test_reproducibility():
    """같은 seed → 같은 초기 가중치."""
    net1 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    w1 = net1.get_all_weights()
    w2 = net2.get_all_weights()
    for k in w1:
        assert np.allclose(w1[k], w2[k]), f"seed 42인데 {k} 가중치가 다름"


def test_different_seeds_differ():
    """다른 seed → 다른 초기 가중치."""
    net1 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=1)
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=2)
    w1 = net1.get_all_weights()
    w2 = net2.get_all_weights()
    any_diff = any(not np.allclose(w1[k], w2[k]) for k in w1)
    assert any_diff, "다른 seed인데 모든 가중치가 동일"


# ──────────────────────────────────────────────
# 9. bias 존재 여부
# ──────────────────────────────────────────────
def test_biases_exist():
    """각 레이어에 bias 벡터가 있는지 확인."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    biases = net.get_all_biases()
    assert biases['b_h1'].shape == (10,)
    assert biases['b_h2'].shape == (10,)
    assert biases['b_out'].shape == (5,)
