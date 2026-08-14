"""Structural tests for the SQL `phase5_pipeline.build_sql` emits.

No DuckDB and no model here, for the same reason `test_pipeline.py` avoids TabPFN: what is worth
checking cheaply and constantly is the arithmetic around the inference, because that is where an
error produces a plausible-looking wrong number rather than a crash.

Every invariant below was verified by hand while the generator was being written, and each one
protects against a failure that actually happened:

* Chunk windows must tile the test range exactly. A gap silently drops rows from the ensemble;
  an overlap scores a row twice and skews its averaged probabilities.
* Nothing may be PREPAREd before its source table has rows. DuckDB fixes a filter's selectivity
  from the source's statistics at prepare time, so a statement prepared against an empty table
  has its predicate pruned to always-false -- permanently, silently, inserting nothing.
* The kernel-bank fingerprint must be recorded once per group and read after all of them. It is
  the only thing that catches a stale refill, since every other assertion still passes when one
  group's features are scored under another group's label.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402

import phase5_pipeline as p5  # noqa: E402

WORKDIR = Path("/tmp/wd")


def build(n_train: int, n_test: int, chunk: int | None, n_groups: int = 40,
          onnx_threads: int = 16) -> str:
    cfg = RocketPFNConfig(num_kernels=10_000, n_groups=n_groups, seed=0, n_estimators=1)
    cfg.validate()
    meta = {"raw_parquet": "/tmp/raw.parquet", "n_train": n_train, "n_test": n_test}
    return p5.build_sql(cfg, meta, WORKDIR, 4, "20GB", WORKDIR, chunk, onnx_threads)


def windows(sql: str) -> list[tuple[int, int]]:
    """Distinct (lo, hi) chunk windows, in order.

    Read off `EXECUTE fill_test(lo, hi)` rather than a WHERE clause: under the prepared-plan
    generator the predicate is `id >= CAST($1 AS BIGINT)`, and the literals only ever appear as
    EXECUTE arguments. Duplicates are collapsed -- the first window is also emitted once during
    priming, and every group repeats the same set.
    """
    seen, out = set(), []
    for lo, hi in re.findall(r"EXECUTE fill_test\((\d+), (\d+)\)", sql):
        w = (int(lo), int(hi))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


class TestChunkWindows:
    @pytest.mark.parametrize(
        "n_train,n_test,chunk",
        [(67, 1029, 128), (500, 4500, 32), (50, 150, 128), (28, 28, 128), (10, 1, 128)],
    )
    def test_windows_tile_the_test_range_exactly(self, n_train, n_test, chunk):
        # Every group repeats the same windows, so the distinct set IS one group's.
        first_group = windows(build(n_train, n_test, chunk))
        assert len(first_group) == -(-n_test // chunk), "one window per chunk, no more"
        covered = sum(hi - lo for lo, hi in first_group)
        assert covered == n_test, "a gap drops rows; an overlap scores them twice"
        assert first_group[0][0] == n_train, "test ids start at n_train"
        assert first_group[-1][1] == n_train + n_test
        for (_, hi), (lo, _) in zip(first_group, first_group[1:]):
            assert hi == lo, "windows must be contiguous"

    def test_unchunked_is_a_single_window(self):
        assert windows(build(67, 1029, None)) == [(67, 67 + 1029)]

    def test_chunk_larger_than_the_split_does_not_split(self):
        assert windows(build(28, 28, 9999))[0] == (28, 56)


class TestPrepareOrder:
    """DuckDB prunes a filtered PREPARE against an empty source to a permanent no-op."""

    def test_feat_cur_is_filled_before_anything_reading_it_is_prepared(self):
        sql = build(50, 150, 128)
        fill_feat_run = sql.index("EXECUTE fill_feat(0)")
        for stmt in ("PREPARE fill_train", "PREPARE fill_test", "PREPARE fill_src"):
            assert sql.index(stmt) > fill_feat_run, (
                f"{stmt} is prepared while feat_cur is still empty; its WHERE clause will be "
                f"pruned to always-false for the whole run"
            )

    def test_context_tables_are_primed_before_the_classify_is_prepared(self):
        sql = build(50, 150, 128)
        # tabfm_classify validates its context at BIND time, not execute time.
        assert sql.index("EXECUTE fill_train") < sql.index("PREPARE score")
        assert sql.index("EXECUTE fill_test") < sql.index("PREPARE score")

    def test_priming_is_checked(self):
        assert "prime_check" in build(50, 150, 128)

    def test_the_loop_runs_after_everything_is_prepared(self):
        sql = build(50, 150, 128)
        assert sql.index("PREPARE score") < sql.index("-- Group 0:")


class TestAssertions:
    def test_one_fingerprint_row_per_group_and_none_from_priming(self):
        # n_groups >= 40 only: 10,000 kernels over fewer groups exceeds the feature cap and
        # RocketPFNConfig.validate() rejects it.
        for n_groups in (40, 50):
            sql = build(50, 150, 128, n_groups=n_groups)
            assert sql.count("INSERT INTO f0_checks") == n_groups, (
                "priming must not write a fingerprint row, or distinct_group_banks is off by one"
            )

    def test_fingerprints_are_read_after_every_group_has_written_one(self):
        sql = build(50, 150, 128)
        assert sql.rindex("INSERT INTO f0_checks") < sql.index("distinct_group_banks")

    def test_counts_are_cast_to_bigint(self):
        # An aggregate over BIGINT can widen to HUGEINT, which `.mode json` renders as a *string*
        # -- and every non-empty string is truthy, so "0" once fired a guard on a count of zero.
        sql = build(50, 150, 128)
        assert "CAST(coalesce(max(duplicate_series), 0) AS BIGINT)" in sql
        assert "CAST(count(DISTINCT fingerprint) AS BIGINT)" in sql

    def test_duplicate_series_are_maxed_not_summed(self):
        # Every group writes the same count of duplicate test series, so sum() reports it 40 times
        # over: InlineSkate's 29 duplicates would be published as 1160.
        sql = build(50, 150, 128)
        assert "sum(duplicate_series)" not in sql


class TestSoftLabels:
    """The distillation arms need the teacher's distribution, not just its argmax."""

    def test_soft_labels_are_dumped(self):
        sql = build(50, 150, 128)
        assert "soft_labels.json" in sql
        assert "SELECT id, cls, mean_p FROM per_class ORDER BY id, cls;" in sql

    def test_soft_labels_come_from_the_averaged_probabilities(self):
        # per_class holds avg(proba) across groups. Dumping anything derived from `yhat_score`
        # instead would look similar and be wrong: that is confidence in whichever class won, not
        # P(class), and it once produced accuracy 1.0 alongside sub-chance AUROC.
        sql = build(50, 150, 128)
        # Comment lines are excluded on purpose: the generator carries a comment naming
        # `yhat_score` as the thing not to use, and a bare substring check flags that warning as
        # the offence it warns about.
        code = [l for l in sql.splitlines() if not l.lstrip().startswith("--")]
        assert not [l for l in code if "yhat_score" in l]
        assert sql.index("CREATE OR REPLACE TABLE per_class") < sql.index("soft_labels.json")


class TestTimingSplit:
    """Answers PLAN.md's "does inference dominate so the C++ transform is pointless?" risk."""

    def test_one_mark_per_group_per_phase(self):
        for n_groups in (40, 50):
            sql = build(50, 150, 128, n_groups=n_groups)
            for phase in ("group_start", "transform_done", "classify_done"):
                # n_groups INSERTs, plus one reference in the FILTER clause that reads them.
                assert sql.count(f"'{phase}'") == n_groups + 1

    def test_transform_mark_sits_between_the_fill_and_the_classifies(self):
        sql = build(50, 150, 128)
        start = sql.index("VALUES (0, 'group_start'")
        fill = sql.index("EXECUTE fill_feat(0);", start)
        done = sql.index("VALUES (0, 'transform_done'")
        first_score = sql.index("EXECUTE score(0);")
        assert start < fill < done < first_score, (
            "transform_done must fall after the feature fill and before any classify, or the "
            "split attributes the transform's cost to inference"
        )

    def test_marks_are_all_written_before_they_are_read(self):
        sql = build(50, 150, 128)
        assert sql.rindex("'classify_done'") < sql.index("FROM timings GROUP BY grp") or \
               sql.index("FROM timings GROUP BY grp") > sql.rindex("INSERT INTO timings")


class TestThreadBudget:
    """The extension's ONNX default reads the host's cores and ignores the cpuset."""

    def test_onnx_thread_count_is_set_explicitly(self):
        assert "SET anofox_tabfm_threads = 16;" in build(50, 150, 128, onnx_threads=16)

    def test_pools_are_sized_to_divide_the_cores_available(self):
        from duckdb_rocket.budget import default_onnx_threads

        # 4 DuckDB tasks each build their own session, so the product is what hits the CPU.
        assert default_onnx_threads(4) >= 1
        assert default_onnx_threads(10_000) == 1, "never below one thread"


class TestSqlSize:
    def test_the_feature_list_is_emitted_once_however_many_chunks(self):
        # It is ~4 KB. Emitted per call it was 74% of a 7.6 MB script; an earlier form reached
        # 18.7 MB and was OOM-killed in the planner.
        for n_test, chunk in ((150, 128), (4500, 32)):
            assert build(500, n_test, chunk).count("features := [") == 1

    def test_marginal_cost_per_chunk_stays_small(self):
        # SQL does grow linearly in chunk count -- that was never the claim. What matters is the
        # constant. The old generator repeated the 500-column projection in every chunk, ~5 KB
        # each, so ECG5000 at chunk 32 would have been 27.9 MB. Here a chunk is a few EXECUTEs.
        groups = 40
        few, many = len(build(500, 4500, 128)), len(build(500, 4500, 32))
        chunks_few, chunks_many = -(-4500 // 128) * groups, -(-4500 // 32) * groups
        per_chunk = (many - few) / (chunks_many - chunks_few)
        assert per_chunk < 300, f"{per_chunk:.0f} bytes per chunk is projection-sized, not EXECUTE-sized"


# --- device / graph selection (GPU runs) -------------------------------------------------
#
# A GPU run differs from a CPU run in three places at once: which extension is loaded, which
# device is set, and which graph the model id resolves to. Getting one of the three wrong is
# the failure that produces a plausible number from the wrong thing -- most obviously a "GPU"
# run that silently executed on the CPU.


def build_device(device: str = "cpu", extension: Path | None = None,
                 register_dir: Path | None = None) -> str:
    cfg = RocketPFNConfig(num_kernels=10_000, n_groups=40, seed=0, n_estimators=1)
    cfg.validate()
    meta = {"raw_parquet": "/tmp/raw.parquet", "n_train": 20, "n_test": 10}
    return p5.build_sql(cfg, meta, WORKDIR, 4, "20GB", WORKDIR, None, 8,
                        device=device, anofox_extension=extension, register_dir=register_dir)


def test_default_run_is_cpu_and_uses_the_installed_extension():
    sql = build_device()
    assert "LOAD anofox_tabfm;" in sql
    assert "SET anofox_tabfm_device = 'cpu';" in sql
    # No registration means the shipped, bundled graph -- the pre-GPU behaviour, unchanged.
    assert "tabfm_register_model" not in sql


def test_device_is_always_stated_explicitly():
    # Not left to anofox's 'auto', so the SQL records what it ran on.
    assert "SET anofox_tabfm_device = 'cuda';" in build_device("cuda", Path("/x/ext.duckdb_extension"))


def test_explicit_extension_replaces_the_installed_load():
    sql = build_device("cuda", Path("/opt/anofox_tabfm.duckdb_extension"))
    assert "LOAD '/opt/anofox_tabfm.duckdb_extension';" in sql
    # The installed one must NOT also be loaded: it is the CPU-only community build, and
    # loading it instead would run the whole thing on the CPU while reporting success.
    assert "LOAD anofox_tabfm;" not in sql


def test_registered_graph_is_bound_to_the_model_id_used_by_classify():
    sql = build_device("cuda", Path("/x/ext"), Path("/models/tabicl"))
    assert f"tabfm_register_model(id := '{p5.MODEL}'" in sql
    assert "base_dir := '/models/tabicl'" in sql
    assert "classification_graph := 'graph_tabicl_classification.onnx'" in sql
    # The id registered must be the id classify asks for, or the run silently uses the bundled
    # graph and the registration is dead code.
    assert f"model := '{p5.MODEL}'" in sql


def test_registration_precedes_every_classify_call():
    sql = build_device("cuda", Path("/x/ext"), Path("/models/tabicl"))
    assert sql.index("tabfm_register_model") < sql.index(f"model := '{p5.MODEL}'")


# --- id recovery key ---------------------------------------------------------------------
#
# anofox_tabfm echoes back only the columns named in `features`, so a plain id column is dropped
# and scored rows are rejoined on feature values. A single-column key held for all ten datasets of
# the original subset and then fanned out on ScreenType and InlineSkate, scoring rows 75 and 80
# times instead of 40 -- silent double-counting in the averaged ensemble, caught only because the
# alignment assertion happened to also fail. Widening the prefix to 4 fixed neither dataset; 16
# fixed InlineSkate and left 5 distinct ScreenType series still sharing a key.
#
# So the tests below no longer measure how WIDE the key is. Width was the wrong axis: every answer
# on it was a guess, and two of the three guesses were wrong in production. They check instead
# that the key is the entire feature vector, which is the only width that is not a guess.


def _join_key_columns(sql: str) -> list[str]:
    """The columns the id-recovery join actually reads, in order. Read out of the SQL rather than
    assumed, so this still measures the real join if its shape changes again."""
    m = re.search(r"JOIN test_src_cur s ON s\.k = \[([^\]]*)\]", sql)
    assert m is not None, "no id-recovery join of the expected form in the generated SQL"
    return [c.strip().removeprefix("c.") for c in m.group(1).split(",")]


def _feature_names(sql: str) -> list[str]:
    """The columns handed to the model as `features`, which is also every column it echoes back."""
    m = re.search(r"features := \[([^\]]*)\]", sql)
    assert m is not None, "no features := [...] argument in the generated SQL"
    return [c.strip().strip("'") for c in m.group(1).split(",")]


def test_id_recovery_joins_on_the_whole_feature_vector():
    sql = build(50, 150, None)
    # Not "wide enough" -- complete. Two rows can share this key only if they are the same feature
    # vector, and identical vectors have identical predictions, so the dedup below is exact.
    assert _join_key_columns(sql) == _feature_names(sql)


def test_id_source_table_carries_the_vector_whole():
    sql = build(50, 150, None)
    # A schema narrower than the join leaves the key partly NULL and it joins to nothing, which
    # surfaces as "scored 0 of 40 groups" and reads like a model problem rather than a plumbing
    # one. Storing the list itself makes the two impossible to disagree.
    assert "test_src_cur (id BIGINT, k DOUBLE[])" in sql
    assert "INSERT INTO test_src_cur SELECT id, f FROM feat_cur" in sql


def test_no_prefix_of_the_vector_is_used_as_a_key_anywhere():
    sql = build(50, 150, None)
    # The specific shape of all three historical bugs: a per-column equality between the source
    # table and the classifier output. If one reappears, the key has silently become a prefix
    # again, and prefixes fail on exactly the datasets nobody tests on.
    assert not re.search(r"s\.f(\d+) = c\.f\1", sql)


def test_duplicate_series_are_reported_but_not_asserted_on():
    sql = build(50, 150, None)
    # count(*) - count(DISTINCT f) is byte-identical series (InlineSkate: 29). Under a prefix key
    # this column counted distinct series a too-narrow key merged, which was a real failure. It
    # cannot be one now, so it is recorded for the reader and nothing branches on it.
    assert "SELECT 0, count(*) - count(DISTINCT f), sum(f[1])" in sql
    assert "count(DISTINCT (" not in sql


def test_duplicate_series_are_collapsed_not_counted_twice():
    sql = build(50, 150, None)
    # InlineSkate has 29 byte-identical test series. No feature key can separate them, however
    # wide; their predictions are identical, so the scoring insert must collapse them to one row
    # per (group, id) instead of counting every pairing.
    assert "any_value(proba)" in sql and "GROUP BY grp, id" in sql
