"""Multivariate kernels — SPEC.md 7.

The first class is the important one. Everything else here checks that multivariate works; that
one checks that adding it did not silently change what univariate does, which is the property
every committed golden vector and the whole C++ conformance suite depend on.
"""

from __future__ import annotations

import numpy as np
import pytest

from duckdb_rocket.prng import SplitMix64
from duckdb_rocket.rocket import (
    KERNEL_LENGTHS,
    generate_kernel,
    generate_kernels,
    select_channels,
    transform,
)

N = 128


class TestUnivariateIsUnchanged:
    """SPEC.md 7.1: at C == 1 no channel draw is made, so the stream is exactly as before."""

    @pytest.mark.parametrize("index", [0, 1, 7, 99, 9_000])
    def test_explicit_one_channel_matches_the_default(self, index):
        assert generate_kernel(42, index, N, 1) == generate_kernel(42, index, N)

    def test_a_one_channel_bank_is_not_marked_multivariate(self):
        assert not generate_kernels(0, N, 8, n_channels=1).is_multivariate

    def test_three_dimensional_input_with_one_channel_is_accepted(self):
        """A (n, 1, t) array is the shape `aeon` hands back; squeezing it must be a no-op."""
        bank = generate_kernels(3, N, 16)
        x = np.random.RandomState(0).randn(5, N)
        np.testing.assert_array_equal(
            transform(x, bank), transform(x[:, None, :], bank)
        )

    def test_two_channels_differ_from_one(self):
        """Sanity in the other direction: the channel draws must actually perturb the stream."""
        uni = generate_kernel(42, 0, N, 1)
        multi = generate_kernel(42, 0, N, 4)
        assert uni[:5] != multi[:5]


class TestNormalisation:
    """`normalize_series` has to work on both shapes, and per channel on the 3-D one."""

    def test_univariate_is_unchanged(self):
        from duckdb_rocket.rocket import normalize_series

        x = np.random.RandomState(0).randn(6, N) * 3.0 + 7.0
        got = normalize_series(x)
        np.testing.assert_allclose(got.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(got.std(axis=1), 1.0, atol=1e-12)

    def test_multivariate_normalises_each_channel_separately(self):
        """Channels are different physical quantities in different units. Pooling them would
        let a large-amplitude channel set the scale for a small one."""
        from duckdb_rocket.rocket import normalize_series

        rng = np.random.RandomState(1)
        x = np.stack(
            [rng.randn(4, N) * 1000.0 + 500.0, rng.randn(4, N) * 0.001], axis=1
        )  # (4, 2, N), wildly different scales
        got = normalize_series(x)
        np.testing.assert_allclose(got.mean(axis=2), 0.0, atol=1e-10)
        np.testing.assert_allclose(got.std(axis=2), 1.0, atol=1e-10)

    def test_a_single_channel_matches_the_two_dimensional_result(self):
        from duckdb_rocket.rocket import normalize_series

        x = np.random.RandomState(2).randn(5, N)
        np.testing.assert_allclose(
            normalize_series(x), normalize_series(x[:, None, :])[:, 0, :], rtol=0, atol=0
        )

    def test_a_constant_channel_is_centred_not_amplified(self):
        """The noise-floor guard has to survive the extra dimension. Without it a constant
        channel becomes a series of exactly +/-1.0 fabricated from rounding error."""
        from duckdb_rocket.rocket import normalize_series

        x = np.stack([np.full((3, N), 4.2), np.random.RandomState(3).randn(3, N)], axis=1)
        got = normalize_series(x)
        np.testing.assert_allclose(got[:, 0, :], 0.0, atol=1e-9)


class TestChannelSelection:
    def test_returns_the_requested_count_without_duplicates(self):
        rng = SplitMix64(1)
        for n_channels in (2, 3, 6, 20):
            for k in range(1, n_channels + 1):
                got = select_channels(rng, n_channels, k)
                assert len(got) == k
                assert len(set(got)) == k
                assert all(0 <= c < n_channels for c in got)

    def test_is_sorted(self):
        rng = SplitMix64(9)
        for _ in range(50):
            got = select_channels(rng, 12, 5)
            assert got == sorted(got)

    def test_consumes_exactly_n_selected_draws(self):
        """A data-dependent draw count would desynchronise every later value in the stream."""
        for k in (1, 3, 7):
            rng = SplitMix64(5)
            select_channels(rng, 10, k)
            after_selection = rng.state

            reference = SplitMix64(5)
            for _ in range(k):
                reference.next_u64()
            assert after_selection == reference.state

    def test_covers_every_channel_across_many_kernels(self):
        """No channel should be unreachable — a bug in the swap would strand the last index."""
        seen: set[int] = set()
        for i in range(400):
            *_, channels = generate_kernel(0, i, N, 6)
            seen.update(channels)
        assert seen == set(range(6))


class TestMultivariateKernels:
    def test_channel_count_stays_within_bounds(self):
        for i in range(300):
            *_, channels = generate_kernel(11, i, N, 5)
            assert 1 <= len(channels) <= 5

    def test_weights_are_channel_major_and_sized_accordingly(self):
        for i in range(50):
            length, weights, _, _, _, channels = generate_kernel(4, i, N, 4)
            assert length in KERNEL_LENGTHS
            assert len(weights) == length * len(channels)

    def test_each_channel_block_is_centred_independently(self):
        """SPEC.md 7.4 — per channel, not one global mean over the whole weight vector."""
        for i in range(60):
            length, weights, _, _, _, channels = generate_kernel(8, i, N, 5)
            for c in range(len(channels)):
                block = weights[c * length : (c + 1) * length]
                assert sum(block) == pytest.approx(0.0, abs=1e-12)

    def test_bank_layout_is_self_consistent(self):
        bank = generate_kernels(2, N, 32, n_channels=4)
        assert bank.is_multivariate
        assert bank.channels is not None
        for i in range(bank.num_kernels):
            channels = bank.channels_for(i)
            width = bank.offsets[i + 1] - bank.offsets[i]
            assert width == bank.lengths[i] * len(channels)

    def test_groups_partition_one_bank(self):
        """The property the whole G-group design rests on, in the multivariate case."""
        full = generate_kernels(6, N, 16, n_channels=3)
        second = generate_kernels(6, N, 8, first_kernel=8, n_channels=3)
        x = np.random.RandomState(1).randn(4, 3, N)
        np.testing.assert_allclose(
            transform(x, second), transform(x, full)[:, 16:], rtol=0, atol=0
        )


class TestMultivariateTransform:
    def test_shape_and_ppv_range(self):
        bank = generate_kernels(0, N, 24, n_channels=4)
        x = np.random.RandomState(2).randn(7, 4, N)
        features = transform(x, bank)
        assert features.shape == (7, 48)
        ppv = features[:, 1::2]
        assert ((ppv >= 0.0) & (ppv <= 1.0)).all()

    def test_matches_a_naive_reference(self):
        """Independent re-implementation of SPEC.md 7.5, straight from the text."""
        rng = np.random.RandomState(3)
        n_channels, n_series = 4, 3
        x = rng.randn(n_series, n_channels, N)
        bank = generate_kernels(21, N, 12, n_channels=n_channels)
        got = transform(x, bank)

        for i in range(bank.num_kernels):
            length = int(bank.lengths[i])
            dilation = int(bank.dilations[i])
            padding = int(bank.paddings[i])
            bias = float(bank.biases[i])
            channels = bank.channels_for(i)
            weights = bank.weights[bank.offsets[i] : bank.offsets[i + 1]]
            out_len = N + 2 * padding - (length - 1) * dilation

            for s in range(n_series):
                conv = np.full(out_len, bias)
                for k in range(out_len):
                    total = bias
                    for ci, channel in enumerate(channels):
                        for j in range(length):
                            idx = k + j * dilation - padding
                            if 0 <= idx < N:
                                total += weights[ci * length + j] * x[s, channel, idx]
                    conv[k] = total
                assert got[s, 2 * i] == pytest.approx(conv.max(), abs=1e-12)
                assert got[s, 2 * i + 1] == pytest.approx(
                    (conv > 0).sum() / out_len, abs=1e-12
                )

    def test_rejects_a_channel_count_mismatch(self):
        bank = generate_kernels(0, N, 4, n_channels=3)
        with pytest.raises(ValueError, match="channels"):
            transform(np.zeros((2, 5, N)), bank)

    def test_rejects_two_dimensional_input(self):
        bank = generate_kernels(0, N, 4, n_channels=3)
        with pytest.raises(ValueError, match="3-D"):
            transform(np.zeros((2, N)), bank)

    def test_is_deterministic(self):
        bank = generate_kernels(5, N, 8, n_channels=3)
        x = np.random.RandomState(4).randn(3, 3, N)
        np.testing.assert_array_equal(transform(x, bank), transform(x, bank))
