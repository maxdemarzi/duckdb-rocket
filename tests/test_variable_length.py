"""Variable-length series — SPEC.md 8.

The trap this guards against is quiet. Kernel weights and lengths do not depend on series
length, but dilation and padding do, so a bank generated per row from that row's own length
gives every row a different instrument while producing a perfectly well-formed feature matrix.
Nothing errors; the columns simply stop meaning the same thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from duckdb_rocket.rocket import generate_kernels, transform, transform_variable

N_REF = 64


class TestKernelsDependOnLength:
    """The reason SPEC.md 8 exists at all — demonstrated rather than asserted."""

    def test_weights_and_lengths_do_not_depend_on_n(self):
        a, b = generate_kernels(0, 64, 16), generate_kernels(0, 512, 16)
        np.testing.assert_array_equal(a.lengths, b.lengths)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_dilations_do_depend_on_n(self):
        a, b = generate_kernels(0, 64, 16), generate_kernels(0, 512, 16)
        assert not np.array_equal(a.dilations, b.dilations)

    def test_one_extra_timepoint_is_enough_to_change_them(self):
        """Not a large-difference effect: 64 vs 65 already diverges."""
        a, b = generate_kernels(0, 64, 8), generate_kernels(0, 65, 8)
        assert not np.array_equal(a.dilations, b.dilations)


class TestTransformVariable:
    def test_equal_length_matches_the_fixed_path_exactly(self):
        """The compatibility guarantee: nothing changes for equal-length data."""
        bank = generate_kernels(0, N_REF, 16)
        x = np.random.RandomState(0).randn(5, N_REF)
        np.testing.assert_array_equal(
            transform_variable(list(x), bank), transform(x, bank)
        )

    def test_accepts_longer_series(self):
        bank = generate_kernels(0, N_REF, 16)
        rng = np.random.RandomState(1)
        features = transform_variable([rng.randn(64), rng.randn(97), rng.randn(200)], bank)
        assert features.shape == (3, 32)
        assert np.isfinite(features).all()

    def test_rejects_a_series_shorter_than_the_reference(self):
        """Rejected rather than padded: padding fabricates data (SPEC.md 8.2)."""
        bank = generate_kernels(0, N_REF, 8)
        with pytest.raises(ValueError, match="rejected rather than padded"):
            transform_variable([np.zeros(N_REF - 1)], bank)

    def test_ppv_stays_a_proportion_at_every_length(self):
        bank = generate_kernels(0, N_REF, 32)
        rng = np.random.RandomState(2)
        features = transform_variable([rng.randn(n) for n in (64, 128, 256)], bank)
        ppv = features[:, 1::2]
        assert ((ppv >= 0.0) & (ppv <= 1.0)).all()

    def test_is_deterministic(self):
        bank = generate_kernels(3, N_REF, 8)
        rng = np.random.RandomState(4)
        series = [rng.randn(64), rng.randn(90)]
        np.testing.assert_array_equal(
            transform_variable(series, bank), transform_variable(series, bank)
        )

    def test_multivariate_ragged(self):
        bank = generate_kernels(0, N_REF, 12, n_channels=3)
        rng = np.random.RandomState(5)
        features = transform_variable([rng.randn(3, 64), rng.randn(3, 130)], bank)
        assert features.shape == (2, 24)
        assert np.isfinite(features).all()

    def test_multivariate_rejects_wrong_rank(self):
        bank = generate_kernels(0, N_REF, 8, n_channels=3)
        with pytest.raises(ValueError, match="2-D"):
            transform_variable([np.zeros(N_REF)], bank)


class TestLengthBias:
    """SPEC.md 8.3 makes a strong claim about `max`. It is checked, not assumed."""

    def test_max_is_biased_upward_by_length_and_ppv_is_not(self):
        bank = generate_kernels(0, N_REF, 120)
        rng = np.random.RandomState(11)

        def means(n: int) -> tuple[float, float]:
            # Identically distributed at every length, so any difference is length alone.
            features = transform_variable([rng.randn(n) for _ in range(40)], bank)
            return float(features[:, 0::2].mean()), float(features[:, 1::2].mean())

        short_max, short_ppv = means(64)
        long_max, long_ppv = means(512)

        # The measured effect is around +40%; assert something well clear of noise but far
        # below it, so this stays a statement about direction rather than a brittle number.
        assert long_max > short_max * 1.15
        assert short_ppv == pytest.approx(long_ppv, abs=0.02)
