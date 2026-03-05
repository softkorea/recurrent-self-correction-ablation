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
