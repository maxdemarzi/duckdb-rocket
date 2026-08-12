"""Tests for the golden-vector fixtures.

The important test here is `test_committed_fixtures_still_match`. The rest of the suite checks
that the code is self-consistent; that one checks the code still agrees with what is *on
disk and committed*, which is the only thing the C++ conformance test will actually see. An
accidental change to draw order would keep every other test in this repo green.
"""

from __future__ import annotations

import numpy as np
import pytest

from duckdb_rocket.golden import (
    GOLDEN_N_SERIES,
    GOLDEN_N_TIMEPOINTS,
    GOLDEN_NUM_KERNELS,
    GOLDEN_OFFSET_FIRST_KERNEL,
    GOLDEN_OFFSET_NUM_KERNELS,
    GOLDEN_SEED,
    build_golden,
    golden_input,
    write_golden,
)
from duckdb_rocket.rocket import generate_kernels, transform

GOLDEN_DIR = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent / "reference" / "golden"


def _read(name):
    pq = pytest.importorskip("pyarrow.parquet")
    return pq.read_table(GOLDEN_DIR / name).to_pydict()


class TestGoldenInput:
    def test_shape_and_determinism(self):
        a = golden_input()
        assert a.shape == (GOLDEN_N_SERIES, GOLDEN_N_TIMEPOINTS)
        assert np.array_equal(a, golden_input())

    def test_is_row_major_over_series(self):
        """Pins the documented draw order: series 0 fully, then series 1."""
        flat = golden_input(n_series=2, n_timepoints=4).ravel()
        long_single = golden_input(n_series=1, n_timepoints=8).ravel()
        assert np.array_equal(flat, long_single)

    def test_looks_standard_normal(self):
        values = golden_input(n_series=40, n_timepoints=512).ravel()
        assert abs(values.mean()) < 0.05
        assert abs(values.std() - 1.0) < 0.05


class TestBuildGolden:
    def test_is_deterministic(self):
        _, k1, f1 = build_golden()
        _, k2, f2 = build_golden()
        assert np.array_equal(f1, f2)
        assert k1["weights"] == k2["weights"]

    def test_feature_matrix_shape(self):
        _, _, features = build_golden()
        assert features.shape == (GOLDEN_N_SERIES, GOLDEN_NUM_KERNELS * 2)

    def test_offset_fixture_matches_the_matching_slice_of_a_full_bank(self):
        """The property that makes `first_kernel` meaningful.

        A port that ignores `first_kernel` and always generates from index 0 passes the base
        fixture and fails here -- which is exactly why the offset fixture exists.
        """
        x = golden_input()
        whole = generate_kernels(
            GOLDEN_SEED,
            GOLDEN_N_TIMEPOINTS,
            GOLDEN_OFFSET_FIRST_KERNEL + GOLDEN_OFFSET_NUM_KERNELS,
        )
        expected = transform(x, whole)[:, 2 * GOLDEN_OFFSET_FIRST_KERNEL :]

        _, _, actual = build_golden(
            GOLDEN_SEED, GOLDEN_OFFSET_NUM_KERNELS, GOLDEN_OFFSET_FIRST_KERNEL
        )
        assert np.array_equal(actual, expected)


class TestCommittedFixtures:
    def test_committed_fixtures_still_match(self):
        """Regression guard on the specification itself.

        If this fails and the change was intentional, re-run `scripts/emit_golden.py` and
        review the diff. If it was not intentional, something reordered the PRNG draws.
        """
        stored_input = np.asarray(_read("input_series.parquet")["values"], dtype=np.float64)
        assert np.array_equal(stored_input, golden_input())

        for label, n_kernels, first in (
            ("base", GOLDEN_NUM_KERNELS, 0),
            ("offset", GOLDEN_OFFSET_NUM_KERNELS, GOLDEN_OFFSET_FIRST_KERNEL),
        ):
            _, kernel_table, features = build_golden(GOLDEN_SEED, n_kernels, first)

            stored_kernels = _read(f"kernels_{label}.parquet")
            assert stored_kernels["length"] == kernel_table["length"].tolist()
            assert stored_kernels["dilation"] == kernel_table["dilation"].tolist()
            assert stored_kernels["padding"] == kernel_table["padding"].tolist()
            assert np.allclose(stored_kernels["bias"], kernel_table["bias"], rtol=0, atol=0)
            for stored_w, want_w in zip(stored_kernels["weights"], kernel_table["weights"]):
                assert np.allclose(stored_w, want_w, rtol=0, atol=0)

            stored_features = np.asarray(
                _read(f"features_{label}.parquet")["features"], dtype=np.float64
            )
            assert np.array_equal(stored_features, features), f"{label} features drifted"

    def test_writer_is_idempotent(self, tmp_path):
        first = {p.name: p.read_bytes() for p in write_golden(tmp_path)}
        second = {p.name: p.read_bytes() for p in write_golden(tmp_path)}
        assert first.keys() == second.keys()
        for name in first:
            assert first[name] == second[name], f"{name} is not byte-stable"

    def test_kernel_indices_are_global_not_local(self):
        stored = _read("kernels_offset.parquet")
        assert stored["kernel_index"][0] == GOLDEN_OFFSET_FIRST_KERNEL
