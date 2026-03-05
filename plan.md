# Self-Awareness Ablation Experiment

# Claude Code 실행 계획

## 가설

신경망에서 self-awareness(자기 출력 관찰 재귀 루프)를 제거하면
iterative self-correction(자기 교정) 능력이 함께 떨어진다.
= output feedback이 intelligence의 한 핵심 축인 self-correction과 분리 불가능하다.

⚠️ 이 실험에서 “intelligence”는 “self-correction을 통한 추론 성능 향상”으로
운영적(operational)으로 정의한다.
t=1 정확도(초기 패턴 인식)는 재귀와 무관하므로
주 측정 대상은 correction_gain(t=3 - t=1)과 accuracy_t3이다.

## 핵심 원칙

- TDD: 테스트 먼저. 코드 나중.
- Python 3.12
- 의존성 최소화: numpy + matplotlib만.
- torch 안 씀. 전체가 투명해야 함.
- 인간이 모든 뉴런과 연결을 눈으로 볼 수 있는 크기.

-----

## Phase 0: 프로젝트 구조

```
self_awareness_experiment/
├── CLAUDE.md              # Claude Code 지시사항
├── tests/
│   ├── test_network.py    # 네트워크 구조 테스트
│   ├── test_training.py   # 학습 정상 동작 테스트
│   ├── test_ablation.py   # 절단 실험 테스트
│   └── test_metrics.py    # 측정 도구 테스트
├── src/
│   ├── network.py         # 신경망 구현
│   ├── training.py        # 학습 루프
│   ├── ablation.py        # 선택적 뉴런/연결 절단
│   ├── metrics.py         # 성능 측정
│   └── visualize.py       # 시각화
├── experiments/
│   └── run_experiment.py  # 전체 실험 실행
└── results/               # 그래프, 데이터 저장
```

-----

## Phase 1: 네트워크 구현

### 1.1 구조

- 순수 numpy. class로 구현.
- Input layer: 10 뉴런 (정적 패턴 입력 — 매 타임스텝 동일)
- Hidden layer 1: 10 뉴런
- Hidden layer 2: 10 뉴런
- Output layer: 5 뉴런 (예측 + confidence)
- **Recurrent loop**: output → hidden layer 1로 피드백
- 총 뉴런: 35개. 인간이 전부 볼 수 있는 크기.
- 활성함수: ReLU (hidden), Linear (output)

### 1.1.1 Task 재정의: 정적 입력 + 자기 교정 (DT 피드백 반영)

- ⚠️ 순차 시퀀스 예측 사용하지 않음.
- 매 타임스텝 동일한 정적 입력을 줌 (예: 노이즈 섞인 패턴).
- t=1: 첫 예측 (재귀 피드백 없음).
- t=2, t=3: 동일한 원본 입력 + 자기 이전 출력을 참조하여 수정.
- 재귀 루프의 유일한 역할 = 자기 교정 (Self-correction).
- 기억(Memory)과 자기 인식(Self-awareness) 완전 분리.
- 루프를 끊으면 순수하게 자기참조 능력만 사라짐.

### 1.2 테스트 먼저

```python
# test_network.py

def test_network_shape():
    """네트워크 출력 shape 확인"""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5)
    x = np.random.randn(10)
    y = net.forward(x)
    assert y.shape == (5,)

def test_recurrent_loop_exists():
    """재귀 루프가 존재하는지 확인"""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5)
    # 첫 번째 forward: 피드백 없음 (초기 상태)
    y1 = net.forward(np.ones(10))
    # 두 번째 forward: 이전 출력이 피드백으로 들어감
    y2 = net.forward(np.ones(10))
    # 재귀 루프가 있으면 같은 입력이라도 출력이 다름
    assert not np.allclose(y1, y2), "재귀 루프가 작동하지 않음"

def test_recurrent_loop_disable():
    """재귀 루프 비활성화 가능한지 확인"""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5)
    y1 = net.forward(np.ones(10))
    net.disable_recurrent_loop()
    y2 = net.forward(np.ones(10))
    y3 = net.forward(np.ones(10))
    # 루프 없으면 같은 입력 → 같은 출력
    assert np.allclose(y2, y3), "루프 비활성화 후에도 출력이 변함"

def test_weight_access():
    """모든 가중치에 직접 접근 가능한지 확인"""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5)
    weights = net.get_all_weights()
    # 가중치 행렬 4개: input→h1, h1→h2, h2→output, output→h1(재귀)
    assert len(weights) == 4
    assert weights['input_to_h1'].shape == (10, 10)
    assert weights['h1_to_h2'].shape == (10, 10)
    assert weights['h2_to_output'].shape == (10, 5)
    assert weights['recurrent'].shape == (5, 10)  # output→h1 피드백
```

### 1.3 구현

- forward pass: 입력 → h1 → h2 → output
- recurrent: 이전 output을 h1 입력에 concatenate 또는 add
- 내부 상태(previous_output) 저장
- disable_recurrent_loop(): recurrent 가중치를 0으로

-----

## Phase 2: 학습 데이터 및 훈련

### 2.1 Task: 정적 입력 패턴 분류 + 자기 교정

```python
# 학습 데이터: 정적 패턴 (매 타임스텝 동일한 입력)
# 노이즈가 섞인 패턴을 한 번에 입력
# 모델은 t=1에서 초기 예측, t=2,t=3에서 자기 출력 보고 교정

patterns = {
    'cluster_A': base_pattern_A + noise,  # 정답: [1,0,0,0,0]
    'cluster_B': base_pattern_B + noise,  # 정답: [0,1,0,0,0]
    'cluster_C': base_pattern_C + noise,  # 정답: [0,0,1,0,0]
    # 노이즈 레벨을 높여서 t=1에서 틀리게 유도
    # t=2,t=3에서 자기 교정으로 맞추는 것이 목표
}
# 핵심: 입력이 매번 같으므로 "기억"이 필요 없음
# 재귀 루프 = 순수하게 "자기 이전 출력을 관찰" 용도
```

### 2.2 테스트 먼저

```python
# test_training.py

def test_training_reduces_loss():
    """훈련이 loss를 줄이는지 확인"""
    net = RecurrentMLP(...)
    data = generate_static_pattern_data(n_samples=100, noise_level=0.3)
    loss_before = evaluate_loss(net, data)
    train(net, data, epochs=100, lr=0.01)
    loss_after = evaluate_loss(net, data)
    assert loss_after < loss_before * 0.5, "훈련이 loss를 충분히 줄이지 못함"

def test_self_correction_occurs():
    """t=2,t=3에서 정확도가 t=1보다 높은지 확인"""
    net = create_trained_network()
    data = generate_static_pattern_data(n_samples=100, noise_level=0.5)
    acc_t1 = evaluate_accuracy_at_timestep(net, data, t=1)
    acc_t3 = evaluate_accuracy_at_timestep(net, data, t=3)
    assert acc_t3 > acc_t1, "자기 교정이 발생하지 않음 — 실험 전제 불성립"
    print(f"t=1: {acc_t1:.3f}, t=3: {acc_t3:.3f}, 교정폭: {acc_t3-acc_t1:.3f}")

def test_recurrent_vs_feedforward():
    """재귀 루프가 자기교정에 필수적인지 사전 확인"""
    net_recurrent = RecurrentMLP(..., recurrent=True)
    net_feedforward = RecurrentMLP(..., recurrent=False)
    data = generate_static_pattern_data(n_samples=200, noise_level=0.5)
    train(net_recurrent, data, epochs=500, lr=0.01)
    train(net_feedforward, data, epochs=500, lr=0.01)
    # feedforward는 t=1,t=2,t=3 모두 같은 답 (입력 동일, 피드백 없음)
    acc_r_t3 = evaluate_accuracy_at_timestep(net_recurrent, data['test'], t=3)
    acc_f_t3 = evaluate_accuracy_at_timestep(net_feedforward, data['test'], t=3)
    print(f"Recurrent t=3: {acc_r_t3:.3f}, Feedforward t=3: {acc_f_t3:.3f}")
```

### 2.3 구현

- BPTT 먼저 시도. 실패 시 수치미분(numerical gradient)으로 즉시 전환.
- gradient check: 해석적 미분과 수치미분 비교 (차이 < 1e-5).
- 파라미터 ~400개이므로 수치미분도 충분히 빠름.
- 학습률, 에폭 등 하이퍼파라미터 외부 설정.
- ⚠️ BPTT 디버깅이 2회 이상 실패하면 수치미분으로 전면 전환.

-----

## Phase 3: Ablation 실험

### 3.1 다섯 그룹 (+서브그룹)

- **Group A**: recurrent 연결 절단 — 가중치를 0으로 (self-awareness 제거)
- **Group B1**: 같은 수(50개)의 랜덤 연결 절단 (산발적 대조군) — 모델당 N=30회 반복
- **Group B2**: 다른 구조적 경로 통째 절단 (예: h2→output 전체) — 구조적 대조군
- **Group C1**: Permutation feedback — y_{t-1} 원소를 랜덤 셔플. 분포/노름 유지, 의미만 파괴.
- **Group C2**: Batch-shuffle feedback — 같은 배치 내 다른 샘플의 y_{t-1}을 사용. 분포 유지 + 자기성만 파괴.
- **Group D**: Feedforward baseline — 처음부터 재귀 없이 훈련.
- **Group D’**: Param-matched feedforward — 재귀 대신 input→output skip connection (50 weights) 추가. 파라미터 수 동일. “그냥 파라미터가 많아서”를 배제.

### 3.1.1 다중 모델 검증 (DT 피드백 반영)

- 단일 모델이 아닌 Random Seed 다른 독립 모델 10개 훈련.
- 각 모델마다 Group A/B/C/D 실험 수행.
- 결과를 10개 모델에 걸쳐 평균 ± 표준편차로 보고.
- 단일 모델의 우연을 배제.

### 3.2 테스트 먼저

```python
# test_ablation.py

def test_ablate_recurrent_zeroes_weights():
    """재귀 가중치가 실제로 0이 되는지"""
    net = create_trained_network()
    assert np.any(net.get_all_weights()['recurrent'] != 0)
    ablate_recurrent(net)
    assert np.all(net.get_all_weights()['recurrent'] == 0)

def test_scrambled_feedback_preserves_weights():
    """Scrambled feedback: 가중치는 그대로, 피드백 값만 교체"""
    net = create_trained_network()
    weights_before = {k: v.copy() for k, v in net.get_all_weights().items()}
    net.enable_scrambled_feedback(seed=42)
    weights_after = net.get_all_weights()
    for k in weights_before:
        assert np.allclose(weights_before[k], weights_after[k]), \
            f"Scrambled feedback이 가중치를 변경함: {k}"

def test_scrambled_feedback_changes_output():
    """Scrambled feedback 활성화 시 출력이 변하는지"""
    net = create_trained_network()
    x = np.random.randn(10)
    net.forward(x)  # t=1
    y_normal = net.forward(x)  # t=2 (정상 피드백)
    
    net.reset_state()
    net.enable_scrambled_feedback(seed=42)
    net.forward(x)  # t=1
    y_scrambled = net.forward(x)  # t=2 (스크램블 피드백)
    
    assert not np.allclose(y_normal, y_scrambled), \
        "Scrambled feedback이 출력을 변경하지 않음"

def test_ablate_random_same_count():
    """랜덤 절단이 재귀 절단과 같은 수의 연결을 끊는지"""
    net = create_trained_network()
    n_recurrent = np.count_nonzero(net.get_all_weights()['recurrent'])
    ablate_random(net, n_connections=n_recurrent, seed=42)
    total_zeroed = count_newly_zeroed(net)
    assert total_zeroed == n_recurrent

def test_ablation_is_deterministic_with_seed():
    """같은 seed → 같은 결과"""
    net1 = create_trained_network()
    net2 = create_trained_network()
    ablate_random(net1, n_connections=10, seed=42)
    ablate_random(net2, n_connections=10, seed=42)
    for k in net1.get_all_weights():
        assert np.allclose(net1.get_all_weights()[k], net2.get_all_weights()[k])

def test_multi_seed_models_differ():
    """서로 다른 seed의 모델이 실제로 다른지"""
    net1 = create_and_train_network(seed=1)
    net2 = create_and_train_network(seed=2)
    w1 = net1.get_all_weights()['input_to_h1']
    w2 = net2.get_all_weights()['input_to_h1']
    assert not np.allclose(w1, w2), "다른 seed인데 가중치가 같음"
```

### 3.3 구현

- ablate_recurrent(net): recurrent 가중치 행렬을 0으로
- ablate_random(net, n, seed): 전체 가중치에서 무작위 n개를 0으로
- enable_scrambled_feedback(net, seed): 재귀 연결 유지, 피드백 값을 랜덤 노이즈로
- 원본 가중치는 deep copy로 보존
- 모든 실험을 10개 독립 모델에서 반복

-----

## Phase 4: 측정 및 분석

### 4.1 측정 항목

1. **accuracy_t1**: t=1 정확도 (재귀 전, 초기 예측. 재귀와 무관한 baseline)
1. **accuracy_t3**: t=3 정확도 (자기 교정 후, 최종 예측. 주 측정 대상)
1. **correction_gain**: accuracy_t3 - accuracy_t1 (자기 교정 폭. 핵심 지표)
1. **recurrent_contribution_norm**: ||W_rec.T @ y_{t-1}|| 평균. 피드백이 실제로 쓰였는지 직접 측정.
1. **step_delta**: ||y_t - y_{t-1}|| 평균. 교정이 실제 발생하는지의 직접 신호.
1. **ece**: Expected Calibration Error (softmax 최대값을 confidence로 사용)

### 4.1.1 노이즈 레벨 스윕 (ChatGPT 피드백 반영)

- noise_level을 [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]으로 스윕.
- 각 노이즈 레벨에서 모든 그룹의 correction_gain 측정.
- “낮은 노이즈 = FF도 맞춤. 높은 노이즈 = recurrent만 교정” 곡선 확인.
- 이 곡선에서 그룹 간 격차가 벌어지는 구간이 가설의 핵심 증거.

### 4.2 테스트 먼저

```python
# test_metrics.py

def test_accuracy_range():
    """정확도가 0~1 사이"""
    acc = evaluate_accuracy(net, data)
    assert 0.0 <= acc <= 1.0

def test_confidence_calibration():
    """calibration 함수가 작동하는지"""
    cal = evaluate_calibration(net, data)
    assert isinstance(cal, float)
    assert 0.0 <= cal <= 1.0
```

### 4.3 통계

- 독립 모델: N=10 (서로 다른 seed)
- 각 모델 내 Group B (랜덤 절단): N=30 반복
- Group A (재귀 절단): 모델당 1회 (결정적)
- Group C (스크램블): 모델당 N=30 반복 (노이즈 seed 다르게)
- Group D (feedforward): 모델당 1회 (별도 훈련)
- 비교: 10개 모델 평균의 paired t-test 또는 Wilcoxon signed-rank test
- 보고: 평균 ± 표준편차, p-value, effect size

-----

## Phase 5: 시각화

### 5.1 그래프 목록

1. **network_map.png**: 전체 뉴런 + 연결 시각화
- 재귀 연결: 빨간색
- 일반 연결: 회색
- 공유 뉴런(self-awareness + intelligence): 노란색 하이라이트
1. **ablation_comparison.png**:
- 막대 그래프: Baseline vs Recurrent Cut(A) vs Random Cut(B, mean±std) vs Scrambled(C) vs Feedforward(D)
- 10개 모델 평균 ± 표준편차
1. **accuracy_distribution.png**:
- 히스토그램: Random Cut N=30의 분포
- Recurrent Cut 위치 + Scrambled Cut 위치 표시
- 10개 모델 오버레이
1. **neuron_importance_heatmap.png**:
- x축: Intelligence importance(i) = 뉴런 i를 t=1에서 0으로 만들었을 때 accuracy_t1 감소량
- y축: Self-correction importance(i) = 뉴런 i를 t=2~3에서만 0으로 만들었을 때 correction_gain 감소량
- 겹치는 영역(둘 다 큰 뉴런)이 가설의 핵심 증거
1. **noise_sweep_curve.png**:
- x축: noise_level, y축: correction_gain
- 각 그룹별 곡선. 고노이즈에서 격차 벌어지는지 확인.

-----

## CLAUDE.md 내용

```markdown
# Self-Awareness Ablation Experiment

## 목적
신경망에서 self-awareness(재귀 루프)를 제거하면 
intelligence(패턴 인식)가 함께 떨어지는지 검증.

## 기술 스택
- Python 3.12
- numpy (신경망 + 수치 계산)
- matplotlib (시각화)
- pytest (테스트)

## 워크플로우
TDD. 반드시 테스트 먼저 작성, 테스트 실패 확인, 구현, 테스트 통과 순서.

## 실행 순서
1. `pytest tests/test_network.py` — 네트워크 구조
2. `pytest tests/test_training.py` — 학습
3. `pytest tests/test_ablation.py` — 절단
4. `pytest tests/test_metrics.py` — 측정
5. `python experiments/run_experiment.py` — 전체 실행
6. 결과는 `results/` 에 저장

## 핵심 제약
- numpy만 사용. torch/tensorflow 금지.
- 뉴런 35개 이하. 인간이 전체를 시각화할 수 있어야 함.
- 모든 가중치에 직접 접근 가능해야 함.
- 재현성: 모든 random에 seed 지정.
- 통계적 유의성: 랜덤 절단 최소 30회 반복. 독립 모델 최소 10개.

## ⚠️ 추가 실험 제약 및 지시사항 (Crucial)

1. **Task 재정의 (Self-Correction):** 단순 시퀀스 암기 예측이 아니라, 매 타임스텝 동일한 '정적 입력'을 주고 t=1, t=2, t=3를 거치며 자신의 이전 출력을 기반으로 오답을 스스로 교정(Self-correction)하는 태스크로 훈련 및 평가하라.
2. **대조군 Group C (Scrambled Feedback) 필수:** 가중치를 0으로 끊는 것(Group A) 외에, 재귀 연결선은 유지하되 피드백되는 이전 출력값을 '랜덤 노이즈'로 덮어씌우는 Group C를 반드시 추가하라.
3. **다중 모델 검증:** Random Seed가 다른 최소 10개의 독립적인 모델을 학습시킨 후 결과를 평균 내어 통계적 유의성을 확보하라.
4. **BPTT 구현 타협:** 수치 미분과 BPTT 계산값 일치 먼저 검증. 구현 에러 2회 이상 반복 시 수치 미분으로 전면 대체.
5. **Phase 단계별 승인:** 각 Phase 테스트 통과 시 요약 출력 후 사용자 승인(y/n) 받아라.
6. **상태 리셋 규칙:** forward_sequence(x, T=3)에서 샘플 시작마다 반드시 reset_state(). 이전 샘플의 피드백이 다음 샘플에 누출되면 안 됨.
7. **결정론적 테스트:** test_recurrent_loop_exists()는 랜덤 초기화에 의존하지 말고, 고정 가중치를 설정한 뒤 recurrent_contrib = W_rec.T @ prev_output의 노름이 0이 아닌지로 구조적 검증하라.
8. **Gradient 구현:** T=3 고정이므로 완전 BPTT 대신 "3-step unroll backprop"으로 구현. 수치미분은 gradient check 전용. relative error 기준 1e-4 tolerance.
9. **결과 파일 포맷:** results/raw_metrics.csv에 반드시 다음 컬럼 저장: seed_model, group, seed_ablation, noise_level, acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm. 그래프는 이 raw에서 재생성 가능하도록.

## 결과 해석 가이드
- Group A(재귀 절단) correction_gain 하락 >> Group B1(랜덤 절단) 하락 → 가설 지지
- Group C1/C2(피드백 셔플) correction_gain 하락 ≈ Group A → 강력한 가설 지지 (정보량 아닌 자기참조가 핵심)
- Group D' (param-matched FF) vs Baseline 비교 → "파라미터 수" 효과 배제
- Group B2(구조적 절단) vs Group A → "경로 절단" 일반 효과 통제
- accuracy_t1은 그룹 간 동일해야 함 (재귀와 무관 확인)
- noise_level 스윕에서 고노이즈 구간에서만 격차 벌어짐 → 난이도 의존적 self-correction 증거
- recurrent_contribution_norm이 baseline에서 유의미 → 피드백 실제 사용 확인
- 위 결과를 10개 독립 모델 평균 ± 표준편차 + 부트스트랩 95% CI로 보고
- 다중 비교 시 Holm-Bonferroni 보정 적용
```

-----

## 실행 계획 요약

|단계|내용                                          |예상 시간|
|--|--------------------------------------------|-----|
|0 |프로젝트 구조 생성                                  |5분   |
|1 |네트워크 테스트 + 구현 (정적 입력 + 재귀 + state reset)    |40분  |
|2 |학습 테스트 + 구현 (3-step unroll + gradient check)|45분  |
|3 |Ablation 테스트 + 구현 (A/B1/B2/C1/C2/D/D’ 7그룹)  |40분  |
|4 |측정 + 통계 + 노이즈 스윕 (10모델 × 7그룹 × 6노이즈)        |40분  |
|5 |시각화 (5종 그래프)                                |25분  |
|6 |전체 실험 실행 + 결과 + raw CSV                     |20분  |

총 약 3.5시간. Claude Code가 TDD로 진행.

-----

## 주의사항

1. Grok이 "가설 지지"라 했지만 신뢰 불가. 직접 검증.
1. ~~35개 뉴런에서 emergence가 안 나올 수 있음.~~ → ✅ emergence 확인 (gain=+0.042, CI [+0.023, +0.059]).
1. ~기억 vs 자기인식 혼동~ → ✅ 해결: 정적 입력 Task.
1. ~정보량 vs 자기참조 혼동~ → ✅ 해결: Group C1/C2 (permutation/batch-shuffle).
1. ~파라미터 수 차이~ → ✅ 해결: Group D' (param-matched FF).
1. ~산발적 vs 구조적 절단~ → ✅ 해결: Group B2 (structured cut).
1. ~모델이 t=1부터 맞출 수 있음~ → ✅ 해결: 노이즈 스윕.
1. 결과가 어느 쪽이든 정직하게 기록.
1. 다중 모델(10개 seed)로 단일 모델의 우연 배제.
1. BPTT → 3-step unroll로 단순화. gradient check은 수치미분.

-----

## Emergence 조건 발견 (2026-03-05)

### 이전 실험에서 emergence 실패 원인

1. **균등 Loss (1/T)**: t=1부터 정답 압력 → 피드백 활용 동기 상실
2. **Tanh 포화**: linear output의 logit ±5 → tanh 미분 ≈ 0 → W_rec 학습 불가
3. **Feedforward 역전파 누수**: Group D/D'에서 d_output_future가 차단되지 않음

### 적용된 수정 (Gemini review #2 + emergence-suggestion 기반)

1. **Time-Weighted Loss**: `w=[0.0, 0.2, 1.0]` — t=1 자유, t=3 집중
2. **Temperature Scaling**: `feedback = tanh(prev_output / 2.0)`, τ=2.0
3. **Feedforward gradient 차단**: `if net._recurrent_enabled:` 분기
4. **NumPy 스레드 스래싱 방지**: `OMP_NUM_THREADS=1` 등

### 결과 요약 (noise=0.5, N=10 models)

| Group | acc_t1 | acc_t3 | gain |
|-------|--------|--------|------|
| Baseline | 0.698 | 0.740 | **+0.042** |
| A (재귀 절단) | 0.698 | 0.698 | 0.000 |
| C1 (셔플) | 0.698 | 0.634 | **-0.064** |
| D' (param-matched FF) | 0.822 | 0.822 | 0.000 |

**핵심**: acc_t1이 Baseline과 A에서 동일(0.698) → 재귀는 초기 인식에 영향 없음.
gain이 Baseline에서만 양수 → 자기 교정은 재귀 루프에 의존.
C1의 gain이 음수 → 잘못된 자기참조는 없는 것보다 해로움.

### 결론

> Emergence는 용량(뉴런 수)이 아니라 학습 동기(Loss)와 기울기 흐름(Gradient)의 문제.
> 35개 뉴런으로 충분. 뉴런 수 탐색은 불필요.

-----

## Hyperparameter Sweep — Robustness 검증 (2026-03-05)

### 설정

80개 조합 (w1×w2×τ) × 10 모델 = 800 실험.

| 파라미터 | 범위 |
|----------|------|
| w1 (t=1 weight) | [0.0, 0.1, 0.2, 0.3] |
| w2 (t=2 weight) | [0.1, 0.2, 0.3, 0.5] |
| w3 (t=3 weight) | 1.0 (고정) |
| τ (temperature) | [1.0, 1.5, 2.0, 3.0, 5.0] |

### 결과

- **54/80 (68%)** 조합에서 emergence 확인
- w1=0.0: 19/20 (95%), w1=0.1: 18/20 (90%), w1=0.2: 15/20 (75%), w1=0.3: 2/20 (10%)
- τ=1.0: 13/16 (81%), τ=1.5: 14/16 (88%), τ=2.0: 12/16 (75%), τ=3.0: 11/16 (69%), τ=5.0: 4/16 (25%)
- w2는 거의 무관 (전 범위에서 유사한 결과)

### 핵심 발견

1. **w1이 가장 중요**: t=1 loss를 줄여야 교정 동기가 생김
2. **τ는 보조적**: 1.0~3.0 범위면 충분
3. **w2는 무관**: 전체 범위에서 emergence 유지
4. **본 실험 설정(w1=0, w2=0.2, τ=2.0)은 80개 중 13위** — cherry-pick이 아님
5. **P-hacking 비판 차단**: 넓은 범위에서 robust하게 재현

### 생성 파일

- `results/sweep_hyperparams.csv` — raw 데이터 800행
- `results/sweep_heatmap_tau*.png` — τ별 w1×w2 heatmap (5장)
- `results/sweep_tau_overview.png` — τ vs gain scatter plot
- `results/REPORT_SWEEP.md` — 상세 보고서