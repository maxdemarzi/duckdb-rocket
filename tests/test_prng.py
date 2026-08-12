"""Conformance tests for the SplitMix64 stream.

The reference vectors below are the point of this file. They are not our numbers -- they are
the published SplitMix64 outputs, so a C++ port can be checked against the same constants
without trusting our Python. If these fail, every golden vector in the project is wrong.
"""

from __future__ import annotations

import math

import pytest

from duckdb_rocket.prng import SplitMix64, kernel_seed

# The canonical first five outputs of SplitMix64 seeded with 0, as published with the
# reference implementation.
REFERENCE_SEED_0 = [
    0xE220A8397B1DCDAF,
    0x6E789E6AA1B965F4,
    0x06C45D188009454F,
    0xF88BB8A8724C81EC,
    0x1B39896A51A8749B,
]


def test_matches_published_reference_vectors():
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(5)] == REFERENCE_SEED_0


def test_outputs_stay_in_64_bit_range():
    rng = SplitMix64(0xDEADBEEF)
    for _ in range(1000):
        assert 0 <= rng.next_u64() <= 0xFFFFFFFFFFFFFFFF


def test_seed_is_reduced_not_rejected():
    """Callers should never have to mask a seed themselves."""
    assert SplitMix64(-1).next_u64() == SplitMix64(0xFFFFFFFFFFFFFFFF).next_u64()
    assert SplitMix64(2**64).next_u64() == SplitMix64(0).next_u64()


def test_next_double_is_in_unit_interval():
    rng = SplitMix64(42)
    values = [rng.next_double() for _ in range(10_000)]
    assert all(0.0 <= v < 1.0 for v in values)
    # Sanity on the mean; 10k samples puts the standard error near 0.003.
    assert abs(sum(values) / len(values) - 0.5) < 0.02


def test_next_below_covers_its_range_and_stays_inside_it():
    rng = SplitMix64(7)
    counts = [0, 0, 0]
    for _ in range(3000):
        counts[rng.next_below(3)] += 1
    assert all(c > 800 for c in counts), f"suspiciously uneven: {counts}"


def test_next_below_rejects_non_positive():
    with pytest.raises(ValueError):
        SplitMix64(1).next_below(0)


def test_next_normal_has_the_right_shape():
    rng = SplitMix64(2024)
    values = [rng.next_normal() for _ in range(20_000)]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    assert abs(mean) < 0.05
    assert abs(var - 1.0) < 0.05
    # A normal should put ~99.7% inside 3 sigma and essentially nothing past 6.
    assert sum(abs(v) > 3.0 for v in values) < len(values) * 0.01
    assert all(abs(v) < 8.0 for v in values)


def test_next_normal_discards_its_spare():
    """Guards the documented decision in `next_normal`.

    If the spare were cached, the second draw would come from the first draw's accepted pair
    and consume no new randomness -- so a fresh generator advanced by one draw would agree
    with an unadvanced one on its next value. It must not.
    """
    a = SplitMix64(11)
    a.next_normal()
    consumed_by_first = a.state

    b = SplitMix64(11)
    b.next_normal()
    b.next_normal()
    assert b.state != consumed_by_first, "second normal consumed no randomness"


def test_streams_are_reproducible():
    assert [SplitMix64(99).next_u64() for _ in range(3)] == [
        SplitMix64(99).next_u64() for _ in range(3)
    ]


class TestKernelSeed:
    def test_is_a_pure_function_of_seed_and_index(self):
        assert kernel_seed(5, 100) == kernel_seed(5, 100)

    def test_distinct_indices_give_distinct_seeds(self):
        seeds = {kernel_seed(0, i) for i in range(10_000)}
        assert len(seeds) == 10_000, "collision in the per-kernel seed space"

    def test_distinct_master_seeds_give_distinct_streams(self):
        assert kernel_seed(0, 0) != kernel_seed(1, 0)

    def test_kernel_zero_is_not_the_raw_master_seed(self):
        assert kernel_seed(12345, 0) != 12345

    def test_adjacent_kernels_do_not_share_a_stream(self):
        """The failure mode this construction exists to avoid.

        Seeding kernel i with `master + i * GOLDEN_GAMMA` would look reasonable and be badly
        broken: SplitMix64 advances its state by exactly GOLDEN_GAMMA per call, so kernel i's
        stream would be kernel i+1's stream shifted by one, and neighbouring kernels would
        share almost all their randomness.
        """
        a = SplitMix64(kernel_seed(0, 0))
        b = SplitMix64(kernel_seed(0, 1))
        first_of_a = [a.next_u64() for _ in range(8)]
        first_of_b = [b.next_u64() for _ in range(8)]
        assert not set(first_of_a) & set(first_of_b)

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError):
            kernel_seed(0, -1)

    def test_derived_streams_look_independent(self):
        """Cheap smoke test: means of per-kernel normal draws should scatter around zero."""
        means = []
        for i in range(200):
            rng = SplitMix64(kernel_seed(31337, i))
            draws = [rng.next_normal() for _ in range(30)]
            means.append(sum(draws) / len(draws))
        grand = sum(means) / len(means)
        assert abs(grand) < 0.1
        # Standard error of a 30-sample mean is 1/sqrt(30) ~ 0.18; the spread of those means
        # should sit near that, not collapse toward zero (which would signal shared streams).
        spread = math.sqrt(sum((m - grand) ** 2 for m in means) / len(means))
        assert 0.10 < spread < 0.30, f"per-kernel means have spread {spread}"
