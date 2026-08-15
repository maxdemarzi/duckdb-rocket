"""The claims that make the group sweep one run instead of four.

Everything here defends one sentence: averaging the first G groups of a 40-group run is EXACTLY what
a G-group run computes. If that is false the whole group experiment is measuring nothing, and it is
false in a quiet way -- the numbers stay plausible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import phase5_pipeline as p5  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from perf_levers import load_pergroup, prefix_predictions, report_kernels  # noqa: E402


def test_prefix_predictions_equal_explicit_means():
    """The cumulative-sum shortcut must agree with averaging the first G groups outright."""
    rng = np.random.default_rng(0)
    cube = rng.dirichlet(np.ones(4), size=(7, 25))  # 7 groups, 25 rows, 4 classes
    classes = ["a", "b", "c", "d"]
    got = prefix_predictions(cube, classes)
    cls = np.asarray(classes, dtype=object)
    for g in range(cube.shape[0]):
        want = cls[np.argmax(cube[: g + 1].mean(axis=0), axis=1)]
        assert np.array_equal(got[g], want), f"G={g + 1}"


def test_prefix_predictions_depend_on_group_order():
    """A sanity check on what the exactness claim rests on: the prefix, not the set.

    If this passed for a shuffled cube the claim would be about the group SET, and the run being
    reproduced -- groups 0..G-1 of a fixed kernel bank -- would not be pinned down.
    """
    rng = np.random.default_rng(1)
    cube = rng.dirichlet(np.ones(3), size=(6, 40))
    classes = ["x", "y", "z"]
    a = prefix_predictions(cube, classes)
    b = prefix_predictions(cube[::-1], classes)
    assert np.array_equal(a[-1], b[-1]), "all groups averaged, order cannot matter"
    assert not all(np.array_equal(a[g], b[g]) for g in range(len(a) - 1)), \
        "a prefix of a reordered cube is a different run; the test data failed to show it"


def test_kernels_per_group_is_what_makes_a_prefix_a_run():
    """G=10 reproduces the first ten groups only when --num-kernels is scaled with --n-groups.

    `kernels_per_group = num_kernels // n_groups`, so `--n-groups 10` at the default 10,000 kernels
    makes each group four times wider -- 2000 features against tabicl's 512 cap -- and is a
    different experiment, not a cheaper one.
    """
    full = RocketPFNConfig(num_kernels=10_000, n_groups=40, seed=0, n_estimators=1)
    prefix = RocketPFNConfig(num_kernels=2_500, n_groups=10, seed=0, n_estimators=1)
    assert full.kernels_per_group == prefix.kernels_per_group == 250
    assert full.features_per_group == prefix.features_per_group == 500
    # ... and the version that does NOT reproduce it.
    wrong = RocketPFNConfig(num_kernels=10_000, n_groups=10, seed=0, n_estimators=1)
    assert wrong.kernels_per_group != full.kernels_per_group
    # Group g starts at kernel g * kernels_per_group, so equal kernels_per_group is exactly the
    # condition for the two runs to read the same slices.
    for g in range(10):
        assert g * prefix.kernels_per_group == g * full.kernels_per_group


def _sql(per_group: bool) -> str:
    cfg = RocketPFNConfig(num_kernels=2_500, n_groups=10, seed=0, n_estimators=1)
    meta = {"dataset": "X", "n_train": 10, "n_test": 5, "n_channels": 1, "n_timepoints": 16,
            "multivariate": False, "raw_parquet": "x.parquet"}
    return p5.build_sql(cfg, meta, Path("."), 4, "8GB", Path("."), 128, 4, per_group=per_group)


def test_per_group_export_is_off_by_default():
    assert "per_group.json" not in _sql(False)


def test_per_group_export_reads_the_unaveraged_table():
    """It must come from `all_groups`, not `per_class`: per_class has already averaged over groups."""
    sql = _sql(True)
    block = sql[sql.index(".once 'per_group.json'"):]
    stmt = block[: block.index(";")]
    assert "all_groups" in stmt
    assert "per_class" not in stmt
    assert "grp" in stmt, "without the group column the dump cannot be sliced by G"


def test_load_pergroup_rejects_a_cube_that_disagrees_with_its_own_group_count(tmp_path):
    p = tmp_path / "phase5_X_tabicl-v2_pergroup.json"
    p.write_text(json.dumps({"dataset": "X", "model": "m", "n_train": 1, "n_test": 2,
                             "n_groups": 40, "kernels_per_group": 250, "ids": [1, 2],
                             "classes": ["a"], "proba": [[[1.0], [1.0]]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="groups of probabilities"):
        load_pergroup(p)


def test_report_kernels_pairs_datasets_by_name_not_position(capsys):
    """A fit that fails at one size must not shift every later dataset's comparison by one.

    The failing size is where the two lists stop agreeing, so slicing by length silently compares
    dataset i of one against dataset i of the other -- and prints a difference either way.
    """
    def row(ds, k, student, routed):
        return {"dataset": ds, "n_kernels": k, "n_test": 100, "student": student,
                "routed": {"0.1": routed, "0.2": routed, "0.3": routed},
                "fit_seconds": 1.0, "transform_seconds": 1.0}

    # Three datasets at the full bank, but the middle one is missing at k=250.
    rows = [row("A", 10_000, 0.50, 0.50), row("B", 10_000, 0.90, 0.90), row("C", 10_000, 0.70, 0.70),
            row("A", 250, 0.50, 0.50), row("C", 250, 0.70, 0.70)]
    report_kernels(rows, [250, 10_000])
    out = capsys.readouterr().out
    # A and C are unchanged at k=250, so every routing difference is exactly zero. Pairing by
    # position would take the full-bank list [A=.50, B=.90, C=.70], slice it to two, and compare
    # C's 0.70 against B's 0.90 -- a mean of (0.00 + -0.20) / 2 = -0.1000, printed as a finding.
    assert "250: +0.0000" in out
    assert "-0.1000" not in out
