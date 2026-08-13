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


def build(n_train: int, n_test: int, chunk: int | None, n_groups: int = 40) -> str:
    cfg = RocketPFNConfig(num_kernels=10_000, n_groups=n_groups, seed=0, n_estimators=1)
    cfg.validate()
    meta = {"raw_parquet": "/tmp/raw.parquet", "n_train": n_train, "n_test": n_test}
    return p5.build_sql(cfg, meta, WORKDIR, 4, "20GB", WORKDIR, chunk)


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

    def test_collision_and_bank_counts_are_cast_to_bigint(self):
        # sum() over BIGINT returns HUGEINT, which `.mode json` renders as a *string* -- and
        # every non-empty string is truthy, so "0" once fired the collision guard on zero
        # collisions.
        sql = build(50, 150, 128)
        assert "CAST(coalesce(sum(collisions), 0) AS BIGINT)" in sql
        assert "CAST(count(DISTINCT fingerprint) AS BIGINT)" in sql


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
