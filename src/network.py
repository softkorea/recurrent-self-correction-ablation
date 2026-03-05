"""RecurrentMLP — 자기 교정(Self-Correction) 실험용 재귀 신경망.

구조:
    Input(10) → Hidden1(10) → Hidden2(10) → Output(5)
                  ↑                              │
                  └──── recurrent feedback ───────┘

총 뉴런: 35개. 인간이 모든 연결을 시각화할 수 있는 크기.
활성함수: ReLU (hidden), Linear (output).
피드백: tanh(prev_output) — bounded [-1, 1].
"""

import numpy as np


class RecurrentMLP:
    def __init__(self, input_size=10, hidden1=10, hidden2=10, output_size=5,
                 seed=0, skip_connection=False, feedback_tau=2.0):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.feedback_tau = feedback_tau

        rng = np.random.RandomState(seed)

        # He initialization (적합: ReLU)
        self.W_ih1 = rng.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b_h1 = np.zeros(hidden1)

        self.W_h1h2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b_h2 = np.zeros(hidden2)

        self.W_h2o = rng.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
        self.b_out = np.zeros(output_size)

        # Recurrent: output(5) → hidden1(10) 피드백
        self.W_rec = rng.randn(output_size, hidden1) * np.sqrt(2.0 / output_size)

        # Skip connection: input(10) → output(5) — Group D' param-matched FF
        if skip_connection:
            self.W_skip = rng.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.W_skip = None

        # 내부 상태
        self._prev_output = np.zeros(output_size)
        self._has_feedback = False  # t=1에서는 False
        self._recurrent_enabled = True
        self._scrambled_feedback = False
        self._scramble_rng = None

        # forward pass 중간값 저장 (backprop 용)
        self._cache = {}

    # ── forward ──────────────────────────────────

    def forward(self, x):
        """단일 타임스텝 forward pass.

        Args:
            x: 입력 벡터 (input_size,)

        Returns:
            output 벡터 (output_size,)
        """
        x = np.asarray(x, dtype=np.float64)

        # recurrent feedback (tanh bounded, temperature scaling)
        # tanh 포화 방지: logit ±5 → tanh(5)≈1.0 (미분≈0) vs tanh(2.5)≈0.987 (미분≈0.026)
        if self._recurrent_enabled:
            feedback = np.tanh(self._prev_output / self.feedback_tau)
            if self._scrambled_feedback and self._has_feedback:
                # Permutation: 분포(평균/분산) 유지, 위치(정보)만 파괴
                feedback = feedback.copy()
                self._scramble_rng.shuffle(feedback)
            rec_contrib = feedback @ self.W_rec
        else:
            feedback = np.zeros(self.output_size)
            rec_contrib = np.zeros(self.hidden1)

        # Hidden 1: input + recurrent contribution
        z_h1 = x @ self.W_ih1 + rec_contrib + self.b_h1
        a_h1 = np.maximum(0, z_h1)  # ReLU

        # Hidden 2
        z_h2 = a_h1 @ self.W_h1h2 + self.b_h2
        a_h2 = np.maximum(0, z_h2)  # ReLU

        # Output (linear)
        z_out = a_h2 @ self.W_h2o + self.b_out
        if self.W_skip is not None:
            z_out = z_out + x @ self.W_skip
        output = z_out

        # 중간값 캐시 (backprop 용)
        self._cache = {
            'x': x,
            'feedback': feedback,
            'rec_contrib': rec_contrib,
            'z_h1': z_h1,
            'a_h1': a_h1,
            'z_h2': z_h2,
            'a_h2': a_h2,
            'z_out': z_out,
            'output': output,
        }

        # 내부 상태 갱신
        self._prev_output = output.copy()
        self._has_feedback = True

        return output

    def forward_sequence(self, x, T=3):
        """T 타임스텝 unroll. 샘플마다 상태 리셋 (CLAUDE.md 제약 #6).

        Args:
            x: 정적 입력 벡터 (input_size,) — 매 타임스텝 동일
            T: 타임스텝 수 (기본 3)

        Returns:
            list of T output vectors, list of T cache dicts
        """
        self.reset_state()
        outputs = []
        caches = []
        for _ in range(T):
            y = self.forward(x)
            outputs.append(y.copy())
            caches.append(self._cache.copy())
        return outputs, caches

    # ── 상태 관리 ────────────────────────────────

    def reset_state(self):
        """내부 상태(이전 출력) 초기화."""
        self._prev_output = np.zeros(self.output_size)
        self._has_feedback = False
        self._cache = {}

    def disable_recurrent_loop(self):
        """재귀 루프 비활성화."""
        self._recurrent_enabled = False

    def enable_recurrent_loop(self):
        """재귀 루프 활성화."""
        self._recurrent_enabled = True

    def enable_scrambled_feedback(self, seed=42):
        """Scrambled feedback 모드: 재귀 연결 유지, 피드백을 permutation으로 교란."""
        self._scrambled_feedback = True
        self._scramble_rng = np.random.RandomState(seed)

    def disable_scrambled_feedback(self):
        """Scrambled feedback 모드 해제."""
        self._scrambled_feedback = False
        self._scramble_rng = None

    # ── 가중치 접근 ──────────────────────────────

    def get_all_weights(self):
        """모든 가중치 행렬을 dict로 반환 (뷰, 수정 가능)."""
        w = {
            'input_to_h1': self.W_ih1,
            'h1_to_h2': self.W_h1h2,
            'h2_to_output': self.W_h2o,
            'recurrent': self.W_rec,
        }
        if self.W_skip is not None:
            w['skip'] = self.W_skip
        return w

    def get_all_biases(self):
        """모든 bias 벡터를 dict로 반환 (뷰, 수정 가능)."""
        return {
            'b_h1': self.b_h1,
            'b_h2': self.b_h2,
            'b_out': self.b_out,
        }

    def get_all_params(self):
        """모든 파라미터(가중치 + bias) 목록 반환."""
        params = [
            self.W_ih1, self.b_h1,
            self.W_h1h2, self.b_h2,
            self.W_h2o, self.b_out,
            self.W_rec,
        ]
        if self.W_skip is not None:
            params.append(self.W_skip)
        return params

    def count_params(self):
        """총 파라미터 수."""
        return sum(p.size for p in self.get_all_params())
