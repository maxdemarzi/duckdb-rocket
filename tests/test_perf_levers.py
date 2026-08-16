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


def _timings(d: Path, classify_total: float, transform_total: float, n_groups: int = 40) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "timings.json").write_text(json.dumps(
        [{"grp": g, "transform_seconds": transform_total / n_groups,
          "classify_seconds": classify_total / n_groups} for g in range(n_groups)]),
        encoding="utf-8")
    return d


def test_cost_model_recovers_a_known_fixed_and_marginal_cost(tmp_path, capsys):
    """Two batch sizes must separate the pass routing avoids from the one it cannot.

    Built from a cost that is known by construction -- 1.4 s per group plus 9 ms per row -- so the
    fit is checked against the truth rather than against itself.
    """
    from route_serve import cost_model
    a, b, g = 1.4, 0.009, 40
    small = _timings(tmp_path / "s", (a + b * 14) * g, 1.7, g)
    big = _timings(tmp_path / "b", (a + b * 64) * g, 2.7, g)
    cost_model(small, big, 14, 64, g, (a + b * 14) * g + 2.2, (a + b * 64) * g + 3.2)
    out = capsys.readouterr().out
    assert "1.400 s fixed per group per call + 9.0 ms per query row" in out
    assert f"{a * g:.1f} s that escalating cannot avoid" in out
    # One call per arm at the default chunk, so the chunk-aware fit must reduce to the two-point
    # one every archived number here was produced by.
    assert "call(s) for the escalated arm" not in out
    # The claim the whole experiment turns on: a 22% escalation does NOT cost 22%.
    assert "not 22%" in out


def test_cost_model_charges_the_context_pass_once_per_chunk(tmp_path, capsys):
    """A chunked big arm pays the fixed pass more than once; charging that to rows inflates b.

    --test-chunk below the batch is how a long-series dataset finishes at all, and each chunk is a
    separate tabfm_classify call with its own pass over the labelled context. Built at the same
    1.4 s + 9 ms truth as the un-chunked test, but with the 64-row arm split into two 32-row calls,
    so the fit is checked against a cost that is known by construction. A fit that ignored the
    count would read a + 1.4/50 s = 37 ms per row -- four times the real marginal cost, on the one
    number that decides whether routing is worth anything.
    """
    from route_serve import cost_model
    a, b, g, chunk = 1.4, 0.009, 40, 32
    small = _timings(tmp_path / "s", (a + b * 14) * g, 1.7, g)          # 14 rows -> 1 call
    big = _timings(tmp_path / "b", (2 * a + b * 64) * g, 2.7, g)        # 64 rows -> 2 calls
    cost_model(small, big, 14, 64, g, (a + b * 14) * g + 2.2, (2 * a + b * 64) * g + 3.2, chunk)
    out = capsys.readouterr().out
    assert "1.400 s fixed per group per call + 9.0 ms per query row" in out
    assert "1 call(s) for the escalated arm and 2 for the full batch" in out
    # The full batch now pays 2 x 1.4 x 40 = 112 s of context against 0.009 x 64 x 40 = 23 s of
    # rows, so escalating 14 of 64 costs even more of teacher-everywhere than it did un-chunked.
    assert "not 22%" in out


def test_cost_model_declines_to_fit_when_the_points_do_not_separate(tmp_path, capsys):
    """A bigger batch that ran FASTER gives a negative marginal cost; say so, do not print it."""
    from route_serve import cost_model
    small = _timings(tmp_path / "s", 80.0, 1.0)
    big = _timings(tmp_path / "b", 60.0, 1.0)
    cost_model(small, big, 14, 64, 40, 81.0, 61.0)
    assert "do not separate" in capsys.readouterr().out


def test_cost_model_warns_when_one_group_stalled(tmp_path, capsys):
    """A 13x-median group has been seen in the archive; the totals are then not a steady rate."""
    from route_serve import cost_model
    small, big = tmp_path / "s", tmp_path / "b"
    _timings(small, 56.0, 1.7)
    _timings(big, 78.0, 2.7)
    t = json.loads((small / "timings.json").read_text(encoding="utf-8"))
    t[0]["classify_seconds"] *= 13
    (small / "timings.json").write_text(json.dumps(t), encoding="utf-8")
    cost_model(small, big, 14, 64, 40, 60.0, 82.0)
    assert "CAUTION" in capsys.readouterr().out


def test_steadiness_reads_an_idle_run_as_idle(tmp_path):
    """The real ScreenType/128 shape: a warm first group, then flat to a percent."""
    from route_serve import steadiness
    cl = [7.661] + [6.2, 6.25, 6.2, 6.21, 6.23, 6.29, 6.24, 6.26, 6.22]
    (tmp_path / "timings.json").write_text(json.dumps(
        [{"grp": g, "transform_seconds": 0.3, "classify_seconds": c} for g, c in enumerate(cl)]),
        encoding="utf-8")
    cv, warm = steadiness(tmp_path)
    assert cv < 0.05, f"an idle run read as contended ({cv:.3f})"
    assert 1.1 < warm < 1.5, "group 0's warm-up should be visible but modest"


def test_steadiness_flags_scatter_and_ignores_the_warm_first_group(tmp_path):
    """Background load arrives unevenly, so it shows up as scatter in the repeated groups."""
    from route_serve import steadiness
    cl = [7.6] + [6.2, 11.4, 6.3, 9.8, 6.2, 14.1, 6.3, 6.2, 10.9]
    (tmp_path / "timings.json").write_text(json.dumps(
        [{"grp": g, "transform_seconds": 0.3, "classify_seconds": c} for g, c in enumerate(cl)]),
        encoding="utf-8")
    cv, _ = steadiness(tmp_path)
    assert cv > 0.05, f"a visibly contended run read as clean ({cv:.3f})"


def test_report_kernels_warns_when_rows_cover_different_datasets(capsys):
    """A failed fit makes one row's mean cover a different subset; that must not be invisible.

    The first pod run lost 44 of 168 fits to a dataset-cache race and printed a table whose rows
    silently averaged over different subsets. The per-row `n` and this warning are the fix.
    """
    def row(ds, k, v):
        return {"dataset": ds, "n_kernels": k, "n_test": 100, "student": v,
                "routed": {"0.1": v, "0.2": v, "0.3": v},
                "fit_seconds": 1.0, "transform_seconds": 1.0}

    rows = [row("A", 10_000, 0.5), row("B", 10_000, 0.9), row("A", 250, 0.5)]
    report_kernels(rows, [250, 10_000])
    out = capsys.readouterr().out
    assert "WARNING: the rows do not cover the same datasets" in out
    assert "  250   1 " in out and "10000   2 " in out, "per-row n should show 1 then 2"


def test_report_kernels_is_quiet_when_coverage_is_even(capsys):
    def row(ds, k, v):
        return {"dataset": ds, "n_kernels": k, "n_test": 100, "student": v,
                "routed": {"0.1": v, "0.2": v, "0.3": v},
                "fit_seconds": 1.0, "transform_seconds": 1.0}

    rows = [row(d, k, 0.5) for d in ("A", "B") for k in (250, 10_000)]
    report_kernels(rows, [250, 10_000])
    assert "WARNING" not in capsys.readouterr().out


# ------------------------------------------------------------------ shrinking the teacher's context

def test_subsample_context_keeps_every_class_at_any_budget():
    """A missing class would look like context size mattering when a label simply went absent.

    This is the whole reason the draw is stratified with a floor of one row per class: 25% of
    Beef's 30 training rows is 7 against 5 classes, and a uniform draw loses one often.
    """
    y = np.array(["a"] * 30 + ["b"] * 10 + ["c"] * 2)
    x = np.arange(len(y) * 4, dtype=float).reshape(len(y), 4)
    for budget in (3, 4, 7, 21, 41):
        xs, ys, keep = p5.subsample_context(x, y, budget, seed=0)
        assert set(np.unique(ys)) == {"a", "b", "c"}, f"a class vanished at budget {budget}"
        assert len(ys) <= budget
        assert np.array_equal(xs, x[keep]), "rows and labels came apart"
        assert np.array_equal(ys, y[keep])


def test_subsample_context_is_a_no_op_when_the_budget_covers_everything():
    y = np.array(["a"] * 5 + ["b"] * 5)
    x = np.arange(20, dtype=float).reshape(10, 2)
    for budget in (10, 50):
        xs, ys, keep = p5.subsample_context(x, y, budget, seed=0)
        assert np.array_equal(xs, x) and np.array_equal(ys, y)
        assert np.array_equal(keep, np.arange(10))


def test_subsample_context_is_deterministic_and_seed_sensitive():
    y = np.array(["a"] * 20 + ["b"] * 20)
    x = np.arange(len(y) * 3, dtype=float).reshape(len(y), 3)
    a = p5.subsample_context(x, y, 10, seed=0)[2]
    b = p5.subsample_context(x, y, 10, seed=0)[2]
    c = p5.subsample_context(x, y, 10, seed=1)[2]
    assert np.array_equal(a, b), "same seed must give the same context"
    assert not np.array_equal(a, c), "different seeds should draw differently"


def test_subsample_context_respects_a_class_smaller_than_its_share():
    """A rare class cannot contribute more rows than it has, and the budget goes elsewhere."""
    y = np.array(["a"] * 97 + ["b"] * 2 + ["c"])
    x = np.arange(len(y) * 2, dtype=float).reshape(len(y), 2)
    _, ys, _ = p5.subsample_context(x, y, 50, seed=0)
    counts = dict(zip(*np.unique(ys, return_counts=True)))
    assert counts["c"] == 1 and counts["b"] <= 2
    assert len(ys) == 50, "the budget should still be spent, on the classes that have rows"


# ------------------------------------------------------- n_kernels and n_groups are one decision

def test_deploy_rejects_a_bank_that_is_not_250_per_group(tmp_path):
    """The teacher's groups are slices of the student's bank, so the two cannot be set apart.

    Without this, deploying 2,500 kernels and serving 40 groups gives 62-kernel groups -- 124
    features against the 500 every accuracy here was measured at -- and still answers plausibly.
    """
    from route_serve import deploy
    with pytest.raises(ValueError, match="kernels per group"):
        deploy("GunPoint", 0.2, n_groups=40, seed=0, folds=5, out=tmp_path, n_kernels=2_500)


def test_serve_rejects_a_group_count_the_deployment_cannot_support(tmp_path):
    from route_serve import serve
    (tmp_path / "meta.json").write_text(json.dumps(
        {"dataset": "GunPoint", "seed": 0, "n_kernels": 2_500, "n_groups": 10,
         "n_timepoints": 150, "target": 0.2, "threshold": 0.1, "folds": 5, "n_train": 50,
         "classes": ["1", "2"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="kernels per group"):
        serve("GunPoint", tmp_path, batch=8, n_groups=40, seed=0, shell=Path("duckdb"),
              workdir=tmp_path / "w")


def test_a_crashed_teacher_does_not_answer_from_the_last_runs_predictions(tmp_path):
    """The workdir is reused across arms, and the failure check is "no predictions.json".

    A G=10 arm, then a G=40 arm, then a --test-chunk retry ladder all write to the same directory.
    If the second run dies, the first run's file is still sitting there and every guard downstream
    -- the id recovery, the bank fingerprint -- passes, because the predictions are real. They are
    just answers to a different question. `sys.executable` stands in for the shell here: it is
    handed `-c ".read <path>"`, raises SyntaxError and exits non-zero without writing anything,
    which is precisely the shape of the crash being guarded against.
    """
    from route_serve import teacher_predict
    stale = [{"id": 50 + k, "yhat": "1"} for k in range(4)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "predictions.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no predictions"):
        teacher_predict("GunPoint", np.arange(4), tmp_path, n_groups=2, num_kernels=500, seed=0,
                        shell=Path(sys.executable))
    assert not (tmp_path / "predictions.json").exists(), "the stale file survived a failed run"
    assert (tmp_path / "crash.log").exists(), "a crash should leave the shell's output on disk"


def test_the_serving_default_is_ten_groups_not_forty():
    """The archived pipeline stays at 40; the serving path does not, and that is deliberate."""
    import route_serve
    assert route_serve.DEFAULT_GROUPS == 10
    assert route_serve.KERNELS_PER_GROUP == 250
    # 10 x 250 = 2,500 kernels = 5,000 student features, and 500 features per teacher call.
    assert route_serve.DEFAULT_GROUPS * route_serve.KERNELS_PER_GROUP == 2_500


def test_oof_margins_cache_key_separates_bank_sizes(tmp_path):
    """A hit that ignored the kernel count would return another model's margins as a fast path."""
    from distill_gate import oof_margins
    a = oof_margins("GunPoint", "rocket+ridge", 0, 3, str(tmp_path), n_kernels=500)
    b = oof_margins("GunPoint", "rocket+ridge", 0, 3, str(tmp_path), n_kernels=2_000)
    files = sorted(f.name for f in tmp_path.glob("*.json"))
    assert len(files) == 2, f"both sizes shared one cache entry: {files}"
    assert any("__n500" in f for f in files) and any("__n2000" in f for f in files)
    # Different banks, different decision scale -- that is the whole reason for the key.
    assert not np.allclose(a, b), "two bank sizes produced identical margins"


def test_oof_margins_still_reads_the_legacy_cache_at_10000(tmp_path):
    """The pre-existing un-suffixed entries were all computed at 10,000; do not orphan them."""
    from distill_gate import oof_margins
    (tmp_path / "GunPoint__rocket_ridge__seed0__k3.json").write_text(
        json.dumps({"margins": [0.5] * 50}), encoding="utf-8")
    got = oof_margins("GunPoint", "rocket+ridge", 0, 3, str(tmp_path), n_kernels=10_000)
    assert np.allclose(got, 0.5), "legacy cache ignored"
