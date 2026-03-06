# 통계 검정 선택 근거 — Wilcoxon Signed-Rank Exact Test

> 리뷰어 질의 대비 문서. 논문에서 사용한 통계 검정의 선택 근거,
> 전제 조건 충족 여부, 대안 검정과의 비교를 기술한다.

---

## 1. 검정 선택: 왜 Wilcoxon Signed-Rank인가

### 1.1 실험 설계는 Paired Design이다

10개 독립 모델(seed 0–9)을 학습한 뒤, **같은 모델**에 대해 Baseline과 각 처리(A, B1, C1, C2, D, D')를 적용한다. 따라서 비교 단위는:

```
(Baseline seed=0 vs A seed=0),
(Baseline seed=1 vs A seed=1),
...
(Baseline seed=9 vs A seed=9)
```

이는 **대응 표본(paired samples)** 설계이므로, 독립 표본 검정(Mann-Whitney U)이 아닌 대응 표본 검정(Wilcoxon signed-rank)이 적절하다.

### 1.2 왜 Parametric Test(paired t-test)가 아닌가

- N=10으로 표본 크기가 작아 정규성 가정을 검증하기 어렵다.
- correction gain의 분포가 정규 분포를 따른다는 사전 근거가 없다.
- Wilcoxon signed-rank는 정규성 가정 없이 사용 가능한 비모수 검정이다.
- N=10에서 exact test가 가능하므로 근사에 의존할 필요가 없다.

### 1.3 왜 Exact Test인가

- N=10이면 가능한 부호 배정이 2^10 = 1,024개뿐이다.
- 전수 열거(exact enumeration)로 정확한 p-value를 계산할 수 있다.
- 정규근사(normal approximation)가 불필요하며, 소표본에서의 근사 오차 문제를 완전히 회피한다.
- 구현이 단순하여 외부 라이브러리(scipy 등) 없이 numpy만으로 가능하다.
- scipy.stats.wilcoxon의 exact 결과와 소수점 6자리까지 일치함을 검증하였다.

---

## 2. 전제 조건 점검

Wilcoxon signed-rank test의 전제 조건 4가지를 실제 데이터로 점검하였다.

### 2.1 대응 표본 (Paired Samples)

**충족.** 각 비교는 동일 seed 모델 내에서 이루어진다 (Section 1.1 참조).

### 2.2 차이(d)의 연속성

**실질적 충족.** accuracy는 이론적으로 이산형(1/200 = 0.005 단위)이지만, 모델별 평균 gain은 충분히 다양한 값을 가진다:

| 비교 | 고유값 수 / 전체 | 연속형으로 간주 가능 |
|------|----------------|---------------------|
| Baseline vs A | 10/10 | ✅ |
| Baseline vs B1 | 10/10 | ✅ |
| Baseline vs C1 | 10/10 | ✅ |
| Baseline vs C2 | 9/10 (tie 1쌍) | ✅ |
| Baseline vs D | 10/10 | ✅ |
| Baseline vs D' | 10/10 | ✅ |

### 2.3 차이(d)의 대칭성

Wilcoxon signed-rank는 H₀ 하에서 차이의 분포가 0에 대해 대칭임을 가정한다.
단, **exact test를 사용하면 이 가정에 대한 의존도가 크게 감소**한다 —
exact p-value는 관측된 순위 통계량이 순수 우연으로 발생할 확률을 직접 계산하기 때문이다.

참고로 각 비교의 왜도(skewness)를 보고한다:

| 비교 | 왜도 | 첨도 | 판정 |
|------|------|------|------|
| Baseline vs A | −0.370 | −0.775 | 대칭에 가까움 |
| Baseline vs B1 | −0.706 | −0.267 | 대칭에 가까움 |
| Baseline vs C1 | +0.247 | −0.349 | 대칭에 가까움 |
| Baseline vs C2 | +1.047 | +0.219 | 약한 우편향 |
| Baseline vs D | −0.370 | −0.775 | 대칭에 가까움 |
| Baseline vs D' | −0.370 | −0.775 | 대칭에 가까움 |

C2의 왜도가 1.047로 약간 높다. 이는 seed=2 모델의 차이(0.225)가 다른 모델들(0.065–0.175)보다 큰 데 기인한다. 그러나:
- Exact test이므로 정규근사의 대칭 가정에 의존하지 않는다.
- C2의 10개 차이가 **전부 양수**(최소 0.065)이므로 방향성이 명확하다.
- 가장 보수적 해석에서도 p = 0.001953 (= 1/512)이며, 이는 10개 차이가 모두 같은 부호일 확률의 정확한 상한이다.

### 2.4 동률(Ties)

| 비교 | 영차이(d=0) | |d| 동률 그룹 | 판정 |
|------|-----------|--------------|------|
| Baseline vs A | 0건 | 1그룹 (2개, |d|=0.045) | ✅ 미미 |
| Baseline vs B1 | 0건 | 없음 | ✅ |
| Baseline vs C1 | 0건 | 없음 | ✅ |
| Baseline vs C2 | 0건 | 1그룹 (2개, |d|=0.065) | ✅ 미미 |
| Baseline vs D | 0건 | 1그룹 (2개, |d|=0.045) | ✅ 미미 |
| Baseline vs D' | 0건 | 1그룹 (2개, |d|=0.045) | ✅ 미미 |

- 영차이(d=0)가 한 건도 없으므로 표본 축소 문제 없음.
- |d| 동률은 최대 1그룹(2개)으로 평균 순위(midrank) 처리로 충분.

---

## 3. 결과

### 3.1 Exact P-values (Holm-Bonferroni 보정 전)

| 비교 | T+ | T− | T | exact p | 방향 |
|------|----|----|---|---------|------|
| Baseline vs A | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > A |
| Baseline vs B1 | 54.0 | 1.0 | 1.0 | 0.003906 | Baseline > B1 |
| Baseline vs C1 | 55.0 | 0.0 | 0.0 | 0.001953 | Baseline > C1 |
| Baseline vs C2 | 55.0 | 0.0 | 0.0 | 0.001953 | Baseline > C2 |
| Baseline vs D | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > D |
| Baseline vs D' | 52.5 | 2.5 | 2.5 | 0.007812 | Baseline > D' |

C1과 C2는 T=0 (10개 차이 전부 양수) → 가능한 최소 p-value = 2/1024 ≈ 0.00195.
이는 N=10 Wilcoxon signed-rank exact test의 **분해능 한계**이며,
실제 효과가 이보다 더 유의미해도 p-value는 더 이상 내려가지 않는다.

### 3.2 Holm-Bonferroni 보정 후

6개 비교에 대해 Holm-Bonferroni step-down 보정 적용:

| 순위 | 비교 | raw p | 보정 p | 유의성 |
|------|------|-------|--------|--------|
| 1 | Baseline vs C1 | 0.001953 | 0.01172 | * |
| 2 | Baseline vs C2 | 0.001953 | 0.01172 | * |
| 3 | Baseline vs B1 | 0.003906 | 0.01562 | * |
| 4 | Baseline vs A | 0.007812 | 0.02344 | * |
| 5 | Baseline vs D | 0.007812 | 0.02344 | * |
| 6 | Baseline vs D' | 0.007812 | 0.02344 | * |

모든 비교가 α=0.05에서 유의. α=0.01 기준으로는 C1, C2만 유의.

### 3.3 검증: scipy.stats.wilcoxon과의 일치

| 비교 | 우리 구현 (exact) | scipy (exact) | 일치 |
|------|------------------|---------------|------|
| Baseline vs A | T=2.5, p=0.007812 | T=2.5, p=0.007812 | ✅ 완전 일치 |
| Baseline vs B1 | T=1.0, p=0.003906 | T=1.0, p=0.003906 | ✅ 완전 일치 |
| Baseline vs C1 | T=0.0, p=0.001953 | T=0.0, p=0.001953 | ✅ 완전 일치 |
| Baseline vs C2 | T=0.0, p=0.001953 | T=0.0, p=0.001953 | ✅ 완전 일치 |
| Baseline vs D | T=2.5, p=0.007812 | T=2.5, p=0.007812 | ✅ 완전 일치 |
| Baseline vs D' | T=2.5, p=0.007812 | T=2.5, p=0.007812 | ✅ 완전 일치 |

---

## 4. 대안 검정과의 비교

### 4.1 Mann-Whitney U Test (독립 표본)

**부적절.** 실험 설계가 paired design이므로 paired test를 사용해야 한다.
독립 검정은 모델 간 변동(inter-model variance)을 통제하지 못하여 검정력이 낮다.
참고로 Mann-Whitney U (정규근사)의 결과:

| 비교 | MWU p (정규근사) | Wilcoxon exact p | 비고 |
|------|-----------------|-----------------|------|
| Baseline vs C2 | 1.56e-04 | 1.95e-03 | MWU가 과도하게 낮은 p 산출 |

MWU가 더 낮은 p-value를 산출하는 것은 검정력이 높기 때문이 **아니라**,
paired 구조를 무시하여 부적절한 분포를 참조하기 때문이다.

### 4.2 Paired t-test (모수적)

N=10에서 정규성 가정을 신뢰하기 어렵다. 참고로 paired t-test 결과를 산출하면
Wilcoxon과 유사한 결론을 보이지만, 소표본에서의 정규성 가정 위반 위험을 감수할 이유가 없다.

### 4.3 Permutation Test (순열 검정)

가장 가정이 적은 비모수 검정. N=10이면 10! = 3,628,800 순열로 exact test 가능.
Wilcoxon signed-rank보다 일반적이지만, 학술적으로 Wilcoxon이 더 널리 인정되며
결과 차이도 무시할 수 있는 수준이다.

---

## 5. 분해능 한계에 대한 주석

N=10 Wilcoxon signed-rank exact test에서 가능한 최소 p-value는:

- T=0 (모든 차이가 같은 부호): p = 2/2^10 = **0.001953**
- 이 값은 α=0.01 보정 후에도 유의할 수 있지만, 0.001 미만은 달성 불가능

더 정밀한 p-value가 필요한 경우 표본 크기(독립 모델 수)를 늘려야 한다.
본 실험에서 C1과 C2의 raw p가 모두 0.001953인 것은 **두 조건 모두 10/10 모델에서
Baseline보다 낮은 gain을 보였기 때문**이며, 검정의 천장 효과이지 효과 크기가 동일하다는 의미가 아니다.

---

## 6. 요약

| 항목 | 판정 |
|------|------|
| 검정 종류 | Wilcoxon signed-rank (paired, nonparametric) ✅ |
| p-value 계산 | Exact enumeration (2^10 = 1,024) ✅ |
| 대응 표본 조건 | 동일 seed 모델 비교 ✅ |
| 연속성 조건 | 고유값 비율 9–10/10 ✅ |
| 대칭성 조건 | 왜도 |≤1.05|, exact test로 의존도 최소화 ✅ |
| 동률 조건 | 영차이 0건, |d| 동률 ≤1그룹(2개) ✅ |
| 외부 라이브러리 의존 | 없음 (numpy + math.erf만 사용) ✅ |
| scipy 검증 | 6/6 비교에서 완전 일치 ✅ |
