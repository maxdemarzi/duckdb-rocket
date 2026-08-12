"""Tests for kernel generation and the max/PPV transform."""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckdb_rocket.rocket import (
    KERNEL_LENGTHS,
    apply_kernel,
    generate_kernel,
    generate_kernels,
    normalize_series,
    transform,
)

N = 128


def naive_apply_kernel(x, weights, length, bias, dilation, padding):
    """Deliberately slow, deliberately obvious reference for `apply_kernel`.

    This is the skip-out-of-range-taps formulation from the ROCKET reference implementation,
    written as a plain loop. `apply_kernel` instead zero-pads and slices, which is much faster
    and much less obviously correct -- so the two are checked against each other. This function
    is also the clearest statement of what the C++ extension has to compute.
    """
    n = len(x)
    output_length = n + 2 * padding - (length - 1) * dilation
    maximum = -math.inf
    positives = 0
    for i in range(-padding, -padding + output_length):
        total = bias
        idx = i
        for j in range(length):
            if 0 <= idx < n:
                total += weights[j] * x[idx]
            idx += dilation
        maximum = max(maximum, total)
        positives += total > 0
    return maximum, positives / output_length


class TestGenerateKernel:
    def test_is_a_pure_function_of_seed_and_index(self):
        assert generate_kernel(7, 42, N) == generate_kernel(7, 42, N)

    def test_length_is_drawn_from_the_paper_s_set(self):
        seen = {generate_kernel(0, i, N)[0] for i in range(500)}
        assert seen == set(KERNEL_LENGTHS)

    def test_weights_are_mean_centred(self):
        for i in range(50):
            _, weights, _, _, _ = generate_kernel(3, i, N)
            assert abs(sum(weights)) < 1e-12

    def test_weight_count_matches_length(self):
        for i in range(50):
            length, weights, _, _, _ = generate_kernel(3, i, N)
            assert len(weights) == length

    def test_bias_is_within_its_range(self):
        for i in range(200):
            _, _, bias, _, _ = generate_kernel(5, i, N)
            assert -1.0 <= bias < 1.0

    def test_dilation_keeps_the_kernel_inside_the_series(self):
        """The property that makes `output_length >= 1` structural rather than checked."""
        for i in range(500):
            length, _, _, dilation, _ = generate_kernel(9, i, N)
            assert dilation >= 1
            assert (length - 1) * dilation <= N - 1

    def test_padding_is_absent_or_exactly_centring(self):
        for i in range(300):
            length, _, _, dilation, padding = generate_kernel(11, i, N)
            assert padding in (0, ((length - 1) * dilation) // 2)

    def test_both_padding_choices_occur(self):
        paddings = [generate_kernel(13, i, N)[4] for i in range(200)]
        assert any(p == 0 for p in paddings)
        assert any(p > 0 for p in paddings)


class TestGenerateKernels:
    def test_groups_partition_a_single_bank(self):
        """The property the whole group design rests on.

        Ten groups of 1,000 must be the same 10,000 kernels as one bank of 10,000, in the same
        positions. If this ever fails, group g's features stop being comparable with the
        features the classifier saw for any other group.
        """
        whole = generate_kernels(2024, N, 20)
        first = generate_kernels(2024, N, 10, first_kernel=0)
        second = generate_kernels(2024, N, 10, first_kernel=10)

        assert np.array_equal(whole.lengths, np.concatenate([first.lengths, second.lengths]))
        assert np.array_equal(whole.weights, np.concatenate([first.weights, second.weights]))
        assert np.allclose(whole.biases, np.concatenate([first.biases, second.biases]), atol=0)
        assert np.array_equal(
            whole.dilations, np.concatenate([first.dilations, second.dilations])
        )
        assert np.array_equal(whole.paddings, np.concatenate([first.paddings, second.paddings]))

    def test_offsets_index_the_flat_weight_array(self):
        bank = generate_kernels(1, N, 32)
        assert bank.offsets[0] == 0
        assert bank.offsets[-1] == bank.weights.shape[0]
        for i in range(bank.num_kernels):
            block = bank.weights[bank.offsets[i] : bank.offsets[i + 1]]
            assert block.shape[0] == bank.lengths[i]

    def test_feature_count_is_two_per_kernel(self):
        assert generate_kernels(1, N, 1000).num_features == 2000

    def test_rejects_series_shorter_than_the_longest_kernel(self):
        with pytest.raises(ValueError, match="shorter than the longest kernel"):
            generate_kernels(1, 6, 4)

    def test_rejects_non_positive_kernel_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            generate_kernels(1, N, 0)


class TestApplyKernel:
    @pytest.mark.parametrize("index", range(25))
    def test_matches_the_naive_reference(self, index):
        """Vectorised zero-pad-and-slice vs. the obvious skip-taps loop."""
        rng = np.random.default_rng(index)
        x = rng.standard_normal((3, N))
        length, weights, bias, dilation, padding = generate_kernel(77, index, N)

        maxima, ppv = apply_kernel(x, weights, length, bias, dilation, padding)
        for s in range(x.shape[0]):
            want_max, want_ppv = naive_apply_kernel(
                x[s], weights, length, bias, dilation, padding
            )
            assert maxima[s] == pytest.approx(want_max, rel=1e-12, abs=1e-12)
            assert ppv[s] == pytest.approx(want_ppv, rel=1e-12, abs=1e-12)

    def test_ppv_is_a_proportion(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, N))
        for i in range(30):
            length, weights, bias, dilation, padding = generate_kernel(1, i, N)
            _, ppv = apply_kernel(x, weights, length, bias, dilation, padding)
            assert np.all((ppv >= 0.0) & (ppv <= 1.0))

    def test_constant_series_gives_ppv_of_zero_or_one(self):
        """A mean-centred kernel annihilates a constant, so every output equals the bias."""
        x = np.full((1, N), 3.7)
        for i in range(30):
            length, weights, bias, dilation, padding = generate_kernel(2, i, N)
            if padding:
                continue  # padded edges see a step, not a constant
            maxima, ppv = apply_kernel(x, weights, length, bias, dilation, padding)
            assert maxima[0] == pytest.approx(bias, abs=1e-12)
            assert ppv[0] == (1.0 if bias > 0 else 0.0)


class TestTransform:
    def test_output_shape_and_layout(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, N))
        bank = generate_kernels(4, N, 16)
        features = transform(x, bank)

        assert features.shape == (5, 32)
        # Odd columns are PPVs, so they must all be proportions; even columns are maxima and
        # are unbounded. This is what pins the interleaving down as a tested contract.
        assert np.all((features[:, 1::2] >= 0.0) & (features[:, 1::2] <= 1.0))

    def test_group_features_match_the_matching_slice_of_the_whole_bank(self):
        """End-to-end version of the partition property, at the feature level."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, N))

        whole = transform(x, generate_kernels(2024, N, 20))
        group1 = transform(x, generate_kernels(2024, N, 10, first_kernel=10))
        assert np.array_equal(whole[:, 20:], group1)

    def test_rejects_a_length_mismatch(self):
        rng = np.random.default_rng(0)
        bank = generate_kernels(1, N, 4)
        with pytest.raises(ValueError, match="does not match"):
            transform(rng.standard_normal((2, N + 1)), bank)

    def test_rejects_non_2d_input(self):
        bank = generate_kernels(1, N, 4)
        with pytest.raises(ValueError, match="expected 2-D"):
            transform(np.zeros(N), bank)

    def test_is_deterministic(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, N))
        bank = generate_kernels(8, N, 24)
        assert np.array_equal(transform(x, bank), transform(x, bank))

    def test_separates_a_trivially_separable_signal(self):
        """Smoke test that the features carry signal at all.

        Two classes -- pure noise vs. noise plus a square pulse. If ROCKET features could not
        separate these, something is wrong far upstream of any classifier.
        """
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((20, N))
        pulsed = rng.standard_normal((20, N))
        pulsed[:, 40:60] += 4.0

        bank = generate_kernels(0, N, 200)
        a = transform(noise, bank)
        b = transform(pulsed, bank)

        # At least one feature should differ far more between classes than within them.
        within = np.concatenate([a, b]).std(axis=0)
        between = np.abs(a.mean(axis=0) - b.mean(axis=0))
        assert np.max(between / (within + 1e-12)) > 1.0


class TestNormalizeSeries:
    def test_produces_zero_mean_unit_variance(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, N)) * 5.0 + 3.0
        out = normalize_series(x)
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
        assert np.allclose(out.std(axis=1), 1.0, atol=1e-12)

    @pytest.mark.parametrize("value", [0.0, 4.2, -1e6, 1e-9])
    def test_constant_series_is_flattened_not_amplified(self, value):
        """Regression test for a real bug: `std > 0` is not a sufficient guard.

        `np.std(np.full(128, 4.2))` is about 8.9e-16, not 0.0 -- the mean is not exactly
        representable, so the residuals are not identically zero. Dividing those residuals by
        that standard deviation turned every value into exactly -1.0: finite, plausible, and
        entirely manufactured from rounding error.
        """
        out = normalize_series(np.full((2, N), value))
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0), "rounding error was amplified into signal"

    def test_genuine_signal_is_still_normalised_at_small_scales(self):
        """The noise-floor guard must not swallow real variation in small-valued series."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, N)) * 1e-9
        out = normalize_series(x)
        assert np.allclose(out.std(axis=1), 1.0, atol=1e-9)
