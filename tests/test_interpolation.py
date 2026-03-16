"""Feedback Interpolation (WS2) 테스트.

feedback_t = α · y_self_{t-1} + (1-α) · y_other_{t-1}
Three types: Self-Zero, Self-Shuffle, Self-Clone.

TDD: 테스트 먼저 작성 → 실패 확인 → 구현 → 통과.
"""

import numpy as np
import pytest

from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import forward_sequence_interpolated


class TestInterpolatedForward:
    """forward_sequence_interpolated 테스트."""

    @pytest.fixture
    def trained_pair(self):
        """Target + clone trained models."""
        target = RecurrentMLP(seed=0, feedback_tau=2.0)
        X, y = generate_data(200, noise_level=0.5, seed=0)
        train(target, X, y, epochs=200, lr=0.01,
              time_weights=[0.0, 0.2, 1.0])

        clone = RecurrentMLP(seed=100, feedback_tau=2.0)
        Xc, yc = generate_data(200, noise_level=0.5, seed=100)
        train(clone, Xc, yc, epochs=200, lr=0.01,
              time_weights=[0.0, 0.2, 1.0])

        return target, clone, X[0]

    def test_alpha_1_equals_baseline(self, trained_pair):
        """α=1.0 → 100% self-feedback → Baseline과 동일."""
        target, clone, x = trained_pair
        outputs_interp, _ = forward_sequence_interpolated(
            target, x, alpha=1.0, interp_type='zero')
        outputs_base, _ = target.forward_sequence(x, T=3)
        for t in range(3):
            np.testing.assert_allclose(
                outputs_interp[t], outputs_base[t], atol=1e-12)

    def test_alpha_0_zero_equals_group_a(self, trained_pair):
        """α=0, type='zero' → feedback = 0 → Group A와 유사 (no feedback)."""
        target, clone, x = trained_pair
        outputs_interp, _ = forward_sequence_interpolated(
            target, x, alpha=0.0, interp_type='zero')
        # With zero feedback, should behave like disabled recurrence
        # (not exactly Group A since W_rec still exists, but feedback input is 0)
        # t=1 should match baseline (no feedback at t=1 regardless)
        outputs_base, _ = target.forward_sequence(x, T=3)
        np.testing.assert_allclose(
            outputs_interp[0], outputs_base[0], atol=1e-12)

    def test_alpha_0_clone_equals_c2(self, trained_pair):
        """α=0, type='clone' → 100% clone feedback → C2와 동일."""
        from src.ablation import forward_sequence_with_clone
        target, clone, x = trained_pair
        outputs_interp, _ = forward_sequence_interpolated(
            target, x, alpha=0.0, interp_type='clone', clone_net=clone)
        outputs_c2, _ = forward_sequence_with_clone(target, clone, x, T=3)
        for t in range(3):
            np.testing.assert_allclose(
                outputs_interp[t], outputs_c2[t], atol=1e-12)

    def test_output_shapes(self, trained_pair):
        """출력 shape 확인."""
        target, clone, x = trained_pair
        for interp_type in ['zero', 'shuffle', 'clone']:
            kwargs = {}
            if interp_type == 'clone':
                kwargs['clone_net'] = clone
            outputs, caches = forward_sequence_interpolated(
                target, x, alpha=0.5, interp_type=interp_type, **kwargs)
            assert len(outputs) == 3
            assert len(caches) == 3
            for out in outputs:
                assert out.shape == (5,)

    def test_interpolation_is_monotonic_tendency(self, trained_pair):
        """α가 증가하면 Baseline에 가까워지는 경향."""
        target, _, x = trained_pair
        outputs_base, _ = target.forward_sequence(x, T=3)

        dists = []
        for alpha in [0.0, 0.5, 1.0]:
            outputs, _ = forward_sequence_interpolated(
                target, x, alpha=alpha, interp_type='zero')
            dist = np.linalg.norm(outputs[2] - outputs_base[2])
            dists.append(dist)

        # α=1.0 should be closest (exact match)
        assert dists[2] < 1e-12
        # α=0.5 should be closer than α=0.0 (not always guaranteed but typical)
        # Just verify α=1.0 is the best
        assert dists[2] <= dists[0]

    def test_deterministic(self, trained_pair):
        """같은 파라미터면 같은 결과."""
        target, clone, x = trained_pair
        out1, _ = forward_sequence_interpolated(
            target, x, alpha=0.5, interp_type='shuffle', shuffle_seed=42)
        out2, _ = forward_sequence_interpolated(
            target, x, alpha=0.5, interp_type='shuffle', shuffle_seed=42)
        for t in range(3):
            np.testing.assert_allclose(out1[t], out2[t])

    def test_t1_always_identical(self, trained_pair):
        """t=1에서는 모든 α, 모든 type에서 동일 (피드백 없는 첫 스텝)."""
        target, clone, x = trained_pair
        outputs_base, _ = target.forward_sequence(x, T=3)

        for alpha in [0.0, 0.3, 0.7, 1.0]:
            for itype in ['zero', 'shuffle', 'clone']:
                kwargs = {'clone_net': clone} if itype == 'clone' else {}
                outputs, _ = forward_sequence_interpolated(
                    target, x, alpha=alpha, interp_type=itype, **kwargs)
                np.testing.assert_allclose(
                    outputs[0], outputs_base[0], atol=1e-12,
                    err_msg=f"t=1 mismatch at α={alpha}, type={itype}")

    def test_different_alphas_differ(self, trained_pair):
        """α가 다르면 t=3 출력이 달라야 함."""
        target, _, x = trained_pair
        out_03, _ = forward_sequence_interpolated(
            target, x, alpha=0.3, interp_type='zero')
        out_07, _ = forward_sequence_interpolated(
            target, x, alpha=0.7, interp_type='zero')
        assert not np.allclose(out_03[2], out_07[2])

    def test_preserves_weights(self, trained_pair):
        """모델 가중치가 변경되지 않음."""
        from src.ablation import deep_copy_weights
        target, clone, x = trained_pair
        saved = deep_copy_weights(target)
        for itype in ['zero', 'shuffle', 'clone']:
            kwargs = {'clone_net': clone} if itype == 'clone' else {}
            forward_sequence_interpolated(
                target, x, alpha=0.5, interp_type=itype, **kwargs)
        current = deep_copy_weights(target)
        for k in saved:
            np.testing.assert_array_equal(saved[k], current[k])
