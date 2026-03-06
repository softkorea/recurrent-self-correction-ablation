# paper.txt 피드백

> 수정 전 검토 메모. 심각도: 🔴 수정 필수 / 🟡 강하게 권고 / 🟢 개선 제안

---

## 🔴 수정 필수

### 1. 통계 검정 명칭 불일치 (Section 2.5)

> "Significance was assessed with Wilcoxon signed-rank tests on paired model-level means"

실제 구현은 **Mann-Whitney U test** (비대응 검정)이다. Wilcoxon signed-rank는 대응 표본 검정이고, Mann-Whitney U는 독립 표본 검정. 코드(`mannwhitneyu`)와 논문의 기술이 불일치한다.

**두 가지 옵션:**
- (a) 논문을 "Mann-Whitney U test"로 수정 (코드에 맞춤)
- (b) 코드를 Wilcoxon signed-rank로 변경 (paired design이 더 적절할 수 있음 — 같은 seed 모델 간 비교이므로 paired가 통계적으로 더 강력)

**(b)가 더 나은 선택일 수 있음**: Baseline seed=0 vs A seed=0, Baseline seed=1 vs A seed=1, ... 이런 쌍이 자연스럽게 존재하므로 paired test가 검정력이 높다.

### 2. Section 3.2 p-value 불일치

> "The Wilcoxon test against Baseline yielded p = 5.68 × 10⁻³ after Holm-Bonferroni correction."

현재 코드(mannwhitneyu + 정규근사)로 산출된 보정 후 p-value는:
- A: 1.42e-03 (구 scipy 기준) 또는 1.23e-03 (현 구현 기준)
- C2: 5.45e-04

논문에 적힌 5.68e-03은 이전 scipy 실행 결과인데, C2 추가로 Holm-Bonferroni 보정 대상이 6개→7개가 되면 보정값이 달라진다. **C2 추가 후 전체 그룹에 대해 p-value를 다시 계산하고 논문에 반영 필요.**

### 3. Schoen et al. (2025) 인용 불완전 (References)

> "Schoen, B., et al. (2025). Stress Testing Deliberative Alignment for Anti-Scheming Training. *arXiv preprint arXiv:2509.xxxxx*."

arXiv ID가 `2509.xxxxx` placeholder로 남아 있다. 실제 ID를 찾아 기입하거나, 해당 논문이 아직 공개 전이면 인용을 제거하거나 "forthcoming"으로 표기해야 한다.

### 4. Author Contributions에 핸들명 사용

> "[Dr.softkorea] conceived the hypothesis..."

학술 논문에는 실명(Sungmoon Ong)을 사용해야 한다. 핸들명은 부적절.

---

## 🟡 강하게 권고

### 5. Section 2.5: "correction gain" 정의가 Methods 후반에 등장

correction gain은 논문의 핵심 지표인데 Section 2.5에 가서야 정의된다. Section 2.2 (Task 설명) 직후 또는 별도 소절에서 먼저 정의하는 것이 독자 친화적.

### 6. C2의 "반복 30회" 표현 부재

Section 2.5에 B1과 C1은 "30 times per model"이라고 기술. C2는 "fixed clone pairing"이라고만 적혀 있는데, 실제로 모델당 1회 고정 페어링(model i ← model (i+1)%10)이므로 N=10이다. 이 점을 명확히 하고, B1/C1과 반복 횟수가 다른 이유를 설명해야 한다. 리뷰어가 "왜 C2만 반복 안 했나"라고 물을 수 있다.

### 7. "Fake mirror" 용어의 범위 혼란

Section 1.4에서 "fake mirror test"를 C1(셔플)에 대해 도입하고, 이후 C2(클론)는 "extend the fake mirror test"로 기술한다. 그런데 Section 3.3 제목이 "The Fake Mirror Effect"로 C1과 C2를 함께 다룬다. C1과 C2를 통칭하는 상위 개념으로 "fake mirror"를 사용할 것인지, C1 고유의 명칭으로 사용할 것인지 일관성 필요.

**제안**: "fake mirror" = C1+C2 통칭, C1 = "shuffled mirror", C2 = "clone mirror"로 명명하면 깔끔.

### 8. Abstract이 너무 길다

현재 Abstract은 약 200단어. 학회/저널에 따라 150단어 제한이 흔하다. (5)번 기여(hyperparameter sweep)는 Abstract에서 빼고 본문에서 다뤄도 충분.

### 9. Discussion 4.2의 인지과학/AI 안전 연결이 과도 확장

> "It also has implications for AI alignment: systems that receive plausible but inaccurate representations of their own internal states may perform worse..."

35개 뉴런 정적 분류 실험에서 AI alignment 시사점을 도출하는 것은 과도한 비약으로 읽힐 수 있다. Section 4.6에서 이미 한계를 인정하고 있지만, 4.2에서 먼저 강한 주장을 펼치면 인상이 약해진다. **"speculative" 또는 "suggestive" 수식어를 더 강하게 달거나, 이 단락을 4.3(Implications for AI Systems)으로 옮기는 것이 나을 수 있다.**

### 10. Noise sweep 결과의 가설 불일치 인정이 약하다

Section 3.4에서 noise=0.3이 최적이라고 보고하는데, 원래 가설은 "고노이즈에서 격차 확대"였다. 이 불일치를 "Goldilocks zone"으로 재해석하는데, 이는 사후 해석(post-hoc)이다. 리뷰어가 "hypothesis was wrong, you reframed it"이라고 지적할 수 있다. **원래 가설과 실제 결과의 차이를 더 솔직하게 기술**하는 것이 오히려 신뢰도를 높인다.

### 11. Table/Figure 부재

논문에 Table이나 Figure 참조가 하나도 없다. 학술 논문에서 핵심 결과는 반드시 Table과 Figure로 제시해야 한다:
- Table 1: 그룹별 주요 결과 (acc_t1, acc_t3, gain, CI, p-value)
- Figure 1: 아키텍처 다이어그램
- Figure 2: ablation_comparison 막대 그래프
- Figure 3: noise sweep 곡선
- Figure 4: neuron importance heatmap

현재 텍스트로만 숫자를 나열하고 있어 가독성이 매우 낮다.

---

## 🟢 개선 제안

### 12. Section 2.3: "evolutionary pressure" 비유

> "This creates evolutionary pressure to use the feedback loop for refinement."

신경망 학습에서 "evolutionary pressure"는 다소 부적절한 비유. "optimization pressure" 또는 "learning incentive"가 더 정확.

### 13. Section 2.1: "deliberately kept small enough" 표현

네트워크가 작은 이유를 "deliberately"로 기술하지만, 실제로는 처음부터 35 뉴런이 제약이었고 이것이 충분하다는 것을 나중에 발견한 것이다. 독자에게는 "35 뉴런이면 부족하지 않나?"라는 의문이 먼저 들 수 있으므로, "충분한 이유"를 여기서 한 줄 요약하면 좋다.

### 14. "Intelligence"라는 단어 사용

논문 전반에서 "intelligence"를 조심스럽게 사용하고 있지만, Section 3.5에서 "intelligence-specialized neurons"이라는 표현이 나온다. plan.md에서 "intelligence"를 "t=1 정확도(초기 패턴 인식)"로 운영적 정의했는데, 논문에서는 이 정의가 명시되지 않았다. "pattern-recognition-specialized"가 더 정확하고 덜 논쟁적.

### 15. Meyes et al. (2019) 인용이 본문에 없음

References에 Meyes et al. (2019) "Ablation Studies in Artificial Neural Networks"이 있는데, 본문에서 한 번도 인용되지 않았다. 사용하지 않는 참고문헌은 제거하거나, Section 2.4(Ablation Design)에서 방법론적 근거로 인용.

### 16. Data Availability의 URL placeholder

> "[GitHub/Zenodo repository URL]"

최종 제출 전 실제 URL 필요. Zenodo DOI를 확보하면 영구 참조 가능.

### 17. Section 4.5: Chain-of-Thought 연결이 표면적

CoT와의 연결을 주장하지만, 실질적 차이가 크다: CoT는 순차적으로 다른 토큰을 생성하고, 우리 모델은 동일 입력에 대해 동일 구조를 반복한다. 이 차이를 인정하면서 연결하면 더 설득력이 있다.

### 18. Group D'의 결과 해석에 방어적 논변 추가 권고

D'의 acc_t1=0.822 > Baseline t=3의 0.740이라는 점은 리뷰어가 "recurrence 없이도 더 잘한다"라고 공격할 수 있다. Section 4.6에서 이를 다루고 있지만, 핵심 논점("우리 주장은 절대 성능이 아니라 자기 교정이라는 질적으로 다른 전략의 존재")을 **더 일찍** (Section 3.2에서) 명시하면 리뷰어의 초기 인상을 관리할 수 있다.

### 19. 재현성 정보 보강

Section 2 어딘가에 "total training time: ~N minutes on [hardware spec]"을 넣으면 재현성에 도움. Data Availability에서 "under 30 minutes"라고 했으니 본문에도 기술.

---

## 구조적 제안

현재 구조는 전반적으로 잘 짜여있다. 다만:

1. **Section 3의 순서 재배치 고려**: 3.1(Emergence) → 3.2(Ablation) → 3.3(Fake Mirror)인데, 3.3이 가장 강력한 결과이므로 논문의 클라이맥스가 중간에 위치한다. 3.4~3.6은 보조적. 이 구조 자체는 나쁘지 않지만, Discussion에서 4.2(Fake Mirror)가 가장 길고 강한 것과 잘 호응한다.

2. **Conclusion에 "future direction" 한 문장 추가**: 현재 Conclusion은 결과 요약만 한다. "이 진단 방법(fake mirror test)을 더 큰 모델에 적용하는 것이 향후 과제"라는 방향성을 한 줄 넣으면 마무리가 더 강해진다.
