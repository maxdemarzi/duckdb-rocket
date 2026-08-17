"""The student and the teacher must score the same rows, and nothing else can detect it if they don't.

Routing compares a student's predictions against a teacher's on the same test split. Both sides now
take a `--resample`, and the failure mode this file exists for is that a resample **preserves
n_train and n_test exactly** — so every size check in the codebase passes when one side is looking
at split 3 and the other at split 7. Right lengths, wrong rows, no exception, and a routing gain
computed against another split's teacher.

Two defences, both tested here:

* `distill_gate` imports `resample_split` from `phase5_pipeline` rather than reimplementing it, so
  the two sides cannot drift. A second stratified re-split would eventually disagree about class
  ordering or tie-breaking, and the disagreement would be invisible.
* the teacher's sidecar records which resample it describes, and `assert_same_split` refuses a
  mismatch. Sidecars predating the field are the archive's split, i.e. resample 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_gate import assert_same_split, load_split  # noqa: E402
import distill_gate  # noqa: E402
import phase5_pipeline  # noqa: E402


def test_the_two_sides_share_one_split_function():
    """Not "produce equal output" -- literally the same object, so they cannot fork."""
    assert distill_gate.resample_split is phase5_pipeline.resample_split


def test_a_mismatched_sidecar_is_refused():
    soft = {"n_train": 10, "n_test": 20, "resample": 3}
    assert_same_split(soft, "X", 3)                      # matching is fine
    with pytest.raises(ValueError, match="resample 3.*resample 7"):
        assert_same_split(soft, "X", 7)


def test_a_sidecar_without_the_field_is_treated_as_the_archive_split():
    """Every archived sidecar predates the field; they must keep working at resample 0."""
    assert_same_split({"n_train": 10, "n_test": 20}, "X", 0)
    with pytest.raises(ValueError):
        assert_same_split({"n_train": 10, "n_test": 20}, "X", 1)


def test_the_guard_survives_identical_row_counts():
    """The point of the whole file: sizes agree across resamples, so sizes cannot be the check."""
    a = {"n_train": 400, "n_test": 139, "resample": 1}
    b = {"n_train": 400, "n_test": 139, "resample": 2}
    assert a["n_train"] == b["n_train"] and a["n_test"] == b["n_test"]
    with pytest.raises(ValueError):
        assert_same_split(a, "DistalPhalanxTW", 2)


def test_load_split_is_deterministic_and_moves_rows():
    try:
        one = load_split("Coffee", 0)
        two = load_split("Coffee", 0)
        r3a = load_split("Coffee", 3)
        r3b = load_split("Coffee", 3)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Coffee not available: {exc}")

    assert np.array_equal(one[0], two[0]) and np.array_equal(one[3], two[3])
    assert np.array_equal(r3a[0], r3b[0]), "resample 3 is not reproducible"
    assert not np.array_equal(one[0], r3a[0]), "resample 3 did not change the split"
    # Sizes preserved -- which is exactly why the sidecar needs its own marker.
    assert one[0].shape == r3a[0].shape and len(one[3]) == len(r3a[3])
