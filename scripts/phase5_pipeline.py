"""Phase 5 — the whole pipeline inside DuckDB: raw series to predictions.

Phase 3 proved the composition with ROCKET features computed in Python. This replaces that half
with the `rocket_transform` extension function, so every arithmetic step from raw series to
predicted label happens inside the database:

    raw series (DOUBLE[])
      -> rocket_transform(values, 250, seed, g*250)          per group, in the extension
      -> 500 scalar feature columns
      -> tabfm_classify(...)                                 per group, in anofox_tabfm
      -> average `proba` across groups, argmax               in plain SQL

    uv run python scripts/phase5_pipeline.py --dataset GunPoint
    uv run python scripts/phase5_pipeline.py --dataset Coffee --compare reference/phase3_Coffee.json

**What Python still does, stated honestly:** it downloads the UCR dataset, writes it to Parquet,
and generates the SQL text. It computes none of the result. The generation is unavoidable
mechanical templating -- `tabfm_classify` needs 500 named scalar columns rather than one LIST
column (Phase 2 found the LIST form crashes the engine), and the resulting script is around half
a megabyte, which is also why it goes through a file rather than `duckdb -c` on Windows.

The check that matters here is `--compare`: the same dataset, seed and grouping run through
Phase 3's Python-computed features should predict the same labels. That is the end-to-end
statement that the C++ port is not merely conformant on the golden fixtures but interchangeable
with the oracle in the real pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.budget import (  # noqa: E402
    binding_cpu_count,
    binding_memory_bytes,
    default_memory_limit,
    default_onnx_threads,
)
from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.shells import built_shell  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import normalize_series  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SHELL = built_shell()
MODEL = "tabicl-v2"


def write_raw_parquet(dataset: str, outdir: Path, normalize: bool) -> tuple[dict, np.ndarray]:
    """Write the dataset as one table of (id, split, label, values DOUBLE[]).

    Series normalisation stays a caller-side step (SPEC.md 7) and is therefore done here rather
    than inside `rocket_transform`; doing it in the extension would silently change what the
    golden vectors mean.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    x_train, y_train = load(dataset, "train")
    x_test, y_test = load(dataset, "test")
    if normalize:
        x_train, x_test = normalize_series(x_train), normalize_series(x_test)

    n_train, n_test = x_train.shape[0], x_test.shape[0]
    ids = np.arange(n_train + n_test)
    splits = ["train"] * n_train + ["test"] * n_test
    labels = [str(v) for v in y_train] + [str(v) for v in y_test]

    # Univariate stays DOUBLE[]; multivariate becomes DOUBLE[][] (channels of timepoints),
    # which is the shape rocket_transform's second overload takes.
    multivariate = x_train.ndim == 3
    if multivariate:
        value_type = pa.list_(pa.list_(pa.float64()))
        values = [[list(channel) for channel in series] for series in x_train]
        values += [[list(channel) for channel in series] for series in x_test]
    else:
        value_type = pa.list_(pa.float64())
        values = list(x_train) + list(x_test)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "split": pa.array(splits),
                "label": pa.array(labels),
                "values": pa.array(values, type=value_type),
            }
        ),
        path,
    )
    return (
        {
            "dataset": dataset,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_channels": int(x_train.shape[1]) if multivariate else 1,
            "n_timepoints": int(x_train.shape[-1]),
            "multivariate": multivariate,
            "raw_parquet": path.as_posix(),
        },
        np.asarray([str(v) for v in y_test]),
    )


def build_sql(config: RocketPFNConfig, meta: dict, outdir: Path, threads: int,
              memory_limit: str, temp_dir: Path, test_chunk: int | None,
              onnx_threads: int) -> str:
    n_features = config.features_per_group
    names = [f"f{j}" for j in range(n_features)]
    feature_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    # DuckDB lists are 1-based, so feature j lives at f[j + 1].
    projection = ", ".join(f"f[{j + 1}] AS f{j}" for j in range(n_features))
    select_features = ", ".join(names)

    parts = [
        "LOAD anofox_tabfm;",
        "SET anofox_tabfm_accept_hf_license = true;",
        # Costs nothing to raise. Read the extension's source rather than guessing: it is a
        # bind-time guard only -- `if (fields.size() > max_features) throw BinderException` in
        # tabfm_generate.cpp -- and sizes no allocation. The comparison is strictly greater, so
        # {n_features} features would in fact pass at exactly {n_features}; the doubling is
        # margin, not necessity. Left as it is because it is free, and noted here so nobody
        # investigates it a second time looking for a memory or speed win. There isn't one.
        f"SET anofox_tabfm_max_features = {max(n_features * 2, 1000)};",
        # Thread count is set explicitly rather than inherited from the visible core count.
        # On a 112-core pod, four concurrent runs each sized their own pool from that number,
        # on top of ONNX's per-session threads, and every run died near completion with no
        # error message at all. A container's visible core count is not its budget, especially
        # when several of these run side by side.
        f"SET threads = {threads};",
        # Third instance of the same trap, and the one that had never been touched.
        # anofox_tabfm's ONNX intra-op default is hardware_concurrency()/2, which reads the
        # HOST's cores: on a 64-core pod inside a 256-core host it defaulted to 128 threads per
        # session, and DuckDB runs `threads` of them at once. Observed there: 132 threads in one
        # process, load average 143. Sized here so the pools sum to the cores we actually have.
        f"SET anofox_tabfm_threads = {onnx_threads};",
        # Same reasoning as the thread count, for memory. Without an explicit limit DuckDB sizes
        # itself against RAM it cannot actually have, and a temp directory is what turns the
        # overflow into a slow query instead of a dead process.
        f"SET memory_limit = '{memory_limit}';",
        f"SET temp_directory = '{temp_dir.as_posix()}';",
        f"CREATE OR REPLACE TABLE raw AS "
        f"SELECT * FROM read_parquet('{meta['raw_parquet']}');",
        # One row per group, filled as each group's features are built. Per group because f0 is
        # a different kernel's output in each one, so a collision in group 17 is invisible to a
        # check that only looks at group 0.
        #
        # `fingerprint` guards the refill architecture below. Every group writes into the same
        # feat_cur/train_cur tables, so a missing or misordered refill would score one group's
        # features while labelling them as another's -- and every other assertion here still
        # passes in that case: 40 groups per row, no collisions, an accuracy that looks fine.
        # Distinct fingerprints are what says the 40 groups really were 40 different kernel
        # banks. Cheap enough to be unconditional.
        "CREATE OR REPLACE TABLE f0_checks (grp BIGINT, collisions BIGINT, fingerprint DOUBLE);",
        # Where the wall clock actually goes. PLAN.md's risk table asks whether inference
        # dominates so completely that the C++ transform is pointless; that is answerable only by
        # splitting the two, and it is also what stops the paper's ~30s/fold median being
        # compared against a pipeline that spends its time somewhere else entirely.
        #
        # Marked per group rather than per chunk: 40 rows of overhead instead of 1440, and the
        # group boundary is exactly where the transform ends and the classifies begin.
        # current_timestamp is transaction-scoped, which is fine here -- the CLI autocommits, so
        # each INSERT is its own transaction and gets a fresh reading.
        "CREATE OR REPLACE TABLE timings (grp BIGINT, phase VARCHAR, ts TIMESTAMP);",
    ]

    # Test ids are contiguous and start at n_train: write_raw_parquet lays the table out as
    # arange(n_train + n_test) with the train rows first.
    first_test_id = meta["n_train"]
    n_test = meta["n_test"]
    size = test_chunk or n_test
    bounds = [
        (first_test_id + lo, first_test_id + min(lo + size, n_test))
        for lo in range(0, n_test, size)
    ]

    # Every relation the classify statement names is created ONCE and refilled in place, so that
    # statement's text never varies and DuckDB plans it once instead of once per chunk.
    #
    # The alternative -- a distinctly-named view per (group, chunk) -- makes every call a
    # separate statement to parse, bind and optimise, and drags the {n_features}-name
    # `features := [...]` argument along with it. On ECG5000 that was 1440 statements and 7.6 MB
    # of SQL, of which 74% was that one argument repeated; an earlier form reached 18.7 MB and
    # was OOM-killed at 18.3s. Here the argument appears once, in the PREPARE.
    #
    # PREPARE over a statement containing tabfm_classify, and EXECUTE re-reading the refilled
    # contents, were both verified against the real extension before this was written.
    schema_cols = ", ".join(f"{n} DOUBLE" for n in names)
    parts.append(f"""
CREATE OR REPLACE TABLE feat_cur (id BIGINT, split VARCHAR, label VARCHAR, f DOUBLE[]);
CREATE OR REPLACE TABLE train_cur (y VARCHAR, {schema_cols});
CREATE OR REPLACE TABLE test_cur ({schema_cols});
-- Only the two columns the join reads (swan PERFORMANCE_TUNING.md 1).
CREATE OR REPLACE TABLE test_src_cur (id BIGINT, f0 DOUBLE);
-- MAP(VARCHAR, DOUBLE) is what anofox_tabfm bc6d8af returns, checked rather than assumed. A
-- version that changes it fails loudly on the first INSERT rather than silently coercing.
CREATE OR REPLACE TABLE all_groups (grp BIGINT, id BIGINT, proba MAP(VARCHAR, DOUBLE));

PREPARE fill_feat AS
  INSERT INTO feat_cur
  SELECT id, split, label,
         rocket_transform(values, {config.kernels_per_group}, {config.seed},
                          CAST($1 AS BIGINT))
  FROM raw;

-- ORDER BELOW IS LOAD-BEARING. Nothing may be PREPAREd against an empty source table.
--
-- DuckDB fixes a filter's selectivity from the source's statistics at PREPARE time. Prepared
-- against an empty table, `WHERE split = 'train'` is pruned to always-false, and that plan is
-- then replayed for the rest of the run -- inserting nothing, raising nothing. Measured on
-- v1.5.5: the same statement prepared on an empty source inserts 0 of 5 rows, and prepared on a
-- populated one inserts 5 of 5. The first symptom here was tabfm_classify reporting an empty
-- context, three steps downstream of the actual cause.
--
-- So: fill feat_cur first, then prepare everything that reads it.
EXECUTE fill_feat({0});

PREPARE fill_train AS
  INSERT INTO train_cur SELECT label, {projection} FROM feat_cur WHERE split = 'train';

PREPARE fill_test AS
  INSERT INTO test_cur SELECT {projection} FROM feat_cur
   WHERE split = 'test' AND id >= CAST($1 AS BIGINT) AND id < CAST($2 AS BIGINT);

PREPARE fill_src AS
  INSERT INTO test_src_cur SELECT id, f[1] FROM feat_cur
   WHERE split = 'test' AND id >= CAST($1 AS BIGINT) AND id < CAST($2 AS BIGINT);

-- Prime train_cur and test_cur too: tabfm_classify validates its context at BIND time, so
-- preparing `score` against an empty train_cur fails with "target 'y' has no non-NULL rows to
-- use as context".
EXECUTE fill_train;
EXECUTE fill_test({bounds[0][0]}, {bounds[0][1]});
EXECUTE fill_src({bounds[0][0]}, {bounds[0][1]});

-- Fail loudly if the priming produced nothing. Every failure in this area has been silent, and
-- the loop below would otherwise run to completion producing an empty result set.
SELECT CASE
         WHEN (SELECT count(*) FROM train_cur) = 0 OR (SELECT count(*) FROM test_cur) = 0
         THEN CAST('priming inserted no rows -- a prepared fill was pruned to a no-op' AS BIGINT)
         ELSE 0
       END AS prime_check;

-- test_cur omits the target: tabfm_classify unions train and test BY NAME, and a target present
-- in both is a duplicate-name binder error naming neither cause (Phase 2).
PREPARE score AS
  INSERT INTO all_groups
  SELECT CAST($1 AS BIGINT), s.id, c.proba
  FROM tabfm_classify('train_cur', 'y', test := 'test_cur',
                      model := '{MODEL}', features := {feature_list}) c
  JOIN test_src_cur s ON s.f0 = c.f0;

-- f0_checks is filled by the loop below, which starts from group 0 again; nothing above wrote
-- to it, so there is no priming row to remove.
""")

    for g in range(config.n_groups):
        first_kernel = g * config.kernels_per_group
        # DELETE rather than CREATE OR REPLACE: replacing the table swaps the catalog entry the
        # prepared statements are bound to. Refilling keeps the entry, which is the whole point.
        parts.append(f"""
-- Group {g}: global kernel indices [{first_kernel}, {first_kernel + config.kernels_per_group}).
INSERT INTO timings VALUES ({g}, 'group_start', current_timestamp);
DELETE FROM feat_cur;
EXECUTE fill_feat({first_kernel});
INSERT INTO timings VALUES ({g}, 'transform_done', current_timestamp);
INSERT INTO f0_checks
SELECT {g}, count(*) - count(DISTINCT f[1]), sum(f[1]) FROM feat_cur WHERE split = 'test';
DELETE FROM train_cur;
EXECUTE fill_train;
""")
        # One classify call per chunk of test rows. Not an approximation: an in-context learner
        # treats each test row as an independent query against the train context, so a row's
        # prediction cannot depend on which other rows shared its call. What it changes is peak
        # memory, which is set by the widest call rather than by the dataset. Verified identical
        # on GunPoint -- 150/150 rows, 0 disagreements -- before being relied on.
        for lo, hi in bounds:
            parts.append(
                f"DELETE FROM test_cur; DELETE FROM test_src_cur;\n"
                f"EXECUTE fill_test({lo}, {hi}); EXECUTE fill_src({lo}, {hi});\n"
                f"EXECUTE score({g});"
            )
        parts.append(f"INSERT INTO timings VALUES ({g}, 'classify_done', current_timestamp);")
        # Something to watch. Between "[3/3] running the pipeline" and the accuracy line there
        # was previously no output at all -- fine for a 60s dataset, useless for one that ran
        # four hours with no way to tell progress from a hang. `.print` is a CLI dot command, so
        # it emits a bare line that survives the ONNX schema-noise filter.
        parts.append(f".print   [rocket] group {g + 1}/{config.n_groups} scored")

    parts.append(f"""

-- Average `proba`, keyed on the class label from the map. Never `yhat_score`: it is confidence
-- in whichever class was predicted, not P(class), and swan measured it yielding accuracy 1.0
-- alongside sub-chance AUROC.
CREATE OR REPLACE TABLE per_class AS
SELECT id, e.key AS cls, avg(e.value) AS mean_p, count(*) AS n_groups_seen
FROM all_groups, UNNEST(map_entries(proba)) AS t(e)
GROUP BY id, e.key;

CREATE OR REPLACE TABLE predictions AS
SELECT p.id, arg_max(p.cls, p.mean_p) AS yhat, r.label AS y
FROM per_class p JOIN raw r USING (id)
GROUP BY p.id, r.label;

.mode json
.once '{(outdir / "predictions.json").as_posix()}'
SELECT id, yhat, y FROM predictions ORDER BY id;

.once '{(outdir / "timings.json").as_posix()}'
-- Seconds spent computing ROCKET features, versus seconds spent in tabfm_classify. The gap
-- between (transform_done -> classify_done) and the classify calls is the chunk fills and the
-- id-recovery joins, which are counted as classify here because they exist only to feed it.
WITH t AS (
  SELECT grp,
         max(ts) FILTER (WHERE phase = 'group_start')    AS t0,
         max(ts) FILTER (WHERE phase = 'transform_done') AS t1,
         max(ts) FILTER (WHERE phase = 'classify_done')  AS t2
  FROM timings GROUP BY grp
)
SELECT round(sum(epoch(t1 - t0)), 2) AS transform_seconds,
       round(sum(epoch(t2 - t1)), 2) AS classify_seconds,
       count(*)                      AS groups_timed
FROM t;

.once '{(outdir / "assertions.json").as_posix()}'
-- `f0_collisions` guards the one assumption the id recovery rests on. anofox_tabfm echoes back
-- only the target and the columns named in `features`, so a plain id column is dropped and the
-- scored rows are rejoined to their ids on the feature value f0. Two test rows sharing an f0
-- would fan that join out and score both against each other's id. The row-alignment counts
-- below already fail in that case, but they report it as "a row was scored by only N of 40
-- groups", which names the symptom and not the cause. Measured 0 across all ten datasets in the
-- subset, ECG5000's 4500 rows included -- this is here so a future dataset says so directly.
SELECT (SELECT count(*) FROM predictions)          AS predicted_rows,
       (SELECT count(DISTINCT id) FROM all_groups) AS distinct_ids,
       (SELECT count(*) FROM all_groups)           AS group_rows,
       (SELECT min(n_groups_seen) FROM per_class)  AS min_groups_per_row,
       (SELECT max(n_groups_seen) FROM per_class)  AS max_groups_per_row,
       -- CAST because sum() over BIGINT returns HUGEINT, and `.mode json` renders HUGEINT as a
       -- *string* to avoid precision loss. "0" is truthy in Python, so the guard below fired on
       -- every run while reporting zero collisions.
       (SELECT CAST(coalesce(sum(collisions), 0) AS BIGINT)
          FROM f0_checks)                         AS f0_collisions,
       (SELECT CAST(count(DISTINCT fingerprint) AS BIGINT)
          FROM f0_checks)                         AS distinct_group_banks;
""")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--n-groups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shell", type=Path, default=SHELL)
    parser.add_argument("--compare", type=Path,
                        help="a Phase 3 report to check predictions against")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="DuckDB threads for this run. Deliberately not the core count: on a many-core "
             "box, several concurrent runs each sizing a pool from the visible cores is what "
             "killed the pod sweep.",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="DuckDB memory_limit, e.g. '20GB'. Defaults to 70%% of the cgroup limit when there "
             "is one, and of visible RAM otherwise -- inside a container those differ, and it is "
             "the cgroup that kills you.",
    )
    parser.add_argument(
        "--test-chunk",
        type=int,
        default=None,
        help="classify this many test rows per tabfm_classify call instead of all of them. "
             "Peak memory is set by the widest call, so this bounds it by a number you choose "
             "rather than by the dataset. Verify identity against an unchunked run before "
             "trusting it on a dataset that has never fitted.",
    )
    parser.add_argument(
        "--onnx-threads",
        type=int,
        default=None,
        help="ONNX intra-op threads per session. Defaults to cores/duckdb-threads so the "
             "concurrent sessions sum to the cores this process actually has. The "
             "extension's own default reads the host's core count and ignores the cpuset.",
    )
    parser.add_argument("--keep-sql", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.shell.exists():
        print(f"no such shell: {args.shell}\nBuild with scripts/build_extension.bat",
              file=sys.stderr)
        return 1

    config = RocketPFNConfig(
        num_kernels=args.num_kernels, n_groups=args.n_groups, seed=args.seed, n_estimators=1
    )
    config.validate()
    out = args.out or ROOT / "reference" / f"phase5_{args.dataset}.json"
    workdir = ROOT / "data" / "phase5" / args.dataset

    print(f"config: {config.n_groups} groups x {config.kernels_per_group} kernels "
          f"= {config.features_per_group} features/group")

    print(f"\n[1/3] writing raw series -> {workdir}", flush=True)
    meta, y_test = write_raw_parquet(args.dataset, workdir, config.normalize)
    print(f"      {meta['n_train']} train / {meta['n_test']} test, "
          f"{meta['n_timepoints']} timepoints")

    memory_limit = args.memory_limit or default_memory_limit()
    _, budget_source = binding_memory_bytes()
    print(f"      memory_limit {memory_limit} (from {budget_source}), "
          f"spilling to {workdir}", flush=True)

    onnx_threads = args.onnx_threads or default_onnx_threads(args.threads)
    cores, core_source = binding_cpu_count()
    print(f"      anofox_tabfm_threads {onnx_threads} x {args.threads} duckdb threads "
          f"= {onnx_threads * args.threads} of {cores} cores (from {core_source})", flush=True)

    sql = build_sql(config, meta, workdir, args.threads, memory_limit, workdir,
                    args.test_chunk, onnx_threads)
    script = workdir / "pipeline.sql"
    script.write_text(sql, encoding="utf-8")
    print(f"[2/3] generated {len(sql):,} characters of SQL")

    print(f"[3/3] running the whole pipeline in DuckDB", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(
        [str(args.shell), "-f", str(script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    seconds = time.perf_counter() - started
    raw_stderr = proc.stderr or ""
    stderr = "\n".join(
        ln for ln in raw_stderr.splitlines()
        if not ln.startswith("Schema error: Trying to register schema") and ln.strip()
    )
    if proc.returncode != 0:
        print(f"FAILED after {seconds:.1f}s (exit {proc.returncode})", file=sys.stderr)
        if stderr:
            print(stderr[:3000], file=sys.stderr)
        else:
            # The filter above removes ONNX's schema-registration spam, which is necessary --
            # it is thousands of lines. But when a run dies without producing any *other*
            # stderr, filtering leaves nothing and the failure reports itself as a bare
            # "FAILED after Ns". That is exactly what happened on the pod, and it turned a
            # diagnosable crash into an hour of guessing. Fall back to the raw tail.
            print(
                "no error message survived the ONNX-noise filter; raw stderr tail follows "
                f"({len(raw_stderr)} chars total):",
                file=sys.stderr,
            )
            print(raw_stderr[-1500:] or "(stderr was completely empty — the process was "
                                        "probably killed by a signal)", file=sys.stderr)
        return 1
    print(f"      {seconds:.1f}s")

    predictions = json.loads((workdir / "predictions.json").read_text(encoding="utf-8"))
    facts = json.loads((workdir / "assertions.json").read_text(encoding="utf-8"))[0]

    # Optional: a report from before this was emitted still loads.
    timing_path = workdir / "timings.json"
    split = None
    if timing_path.exists():
        row = json.loads(timing_path.read_text(encoding="utf-8"))[0]
        tr, cl = float(row["transform_seconds"]), float(row["classify_seconds"])
        total = tr + cl
        split = {
            "transform_seconds": round(tr, 2),
            "classify_seconds": round(cl, 2),
            "classify_share": round(cl / total, 4) if total else None,
            "groups_timed": int(row["groups_timed"]),
        }
        print(f"\n  transform {tr:.1f}s | classify {cl:.1f}s "
              f"({100 * cl / total:.1f}% of measured work)" if total else "")

    n_test = len(y_test)
    failures = []
    if facts["predicted_rows"] != n_test:
        failures.append(f"{facts['predicted_rows']} predictions for {n_test} test rows")
    if facts["group_rows"] != n_test * config.n_groups:
        failures.append(
            f"{facts['group_rows']} group rows, expected {n_test * config.n_groups}: the "
            f"feature-value join dropped or duplicated rows"
        )
    # int() rather than a bare truth test: DuckDB hands some aggregates back as strings, and
    # every non-empty string is truthy. Belt and braces with the CAST in the SQL above -- this
    # guard exists to catch a wrong answer, so it must not itself become one.
    banks = int(float(facts.get("distinct_group_banks") or 0))
    if banks != config.n_groups:
        failures.append(
            f"{banks} distinct kernel banks across {config.n_groups} groups: the per-group "
            f"feature tables were not all refilled, so some groups were scored against another "
            f"group's features under their own label"
        )
    if int(float(facts.get("f0_collisions") or 0)):
        failures.append(
            f"{facts['f0_collisions']} test rows share an f0 with another row across the "
            f"{config.n_groups} groups; ids are recovered by joining on f0, so those rows were "
            f"scored against each other's id"
        )
    if facts["min_groups_per_row"] != config.n_groups:
        failures.append(
            f"a row was scored by only {facts['min_groups_per_row']} of {config.n_groups} "
            f"groups; averaging a partial ensemble is silently wrong"
        )

    by_id = {int(r["id"]): r["yhat"] for r in predictions}
    ordered_ids = sorted(by_id)
    y_pred = np.asarray([by_id[i] for i in ordered_ids])
    accuracy = float((y_pred == y_test).mean()) if len(y_pred) == n_test else float("nan")

    print(f"\n  accuracy ({MODEL}, e=1, G={config.n_groups}): {accuracy:.4f}")
    print(f"  row alignment: {facts['distinct_ids']}/{n_test} ids, "
          f"{facts['group_rows']} group-rows, "
          f"{facts['min_groups_per_row']}-{facts['max_groups_per_row']} groups per row")
    # The end-to-end statement: C++ features must be interchangeable with the oracle's.
    #
    # Compared per row, not by accuracy. Two runs can reach identical accuracy while disagreeing
    # about which rows they got right, so equal accuracy is consistent with the C++ transform
    # being subtly wrong -- it is the weaker claim, and the weaker claim is not the one worth
    # making here.
    comparison = None
    if args.compare and args.compare.exists():
        phase3 = json.loads(args.compare.read_text(encoding="utf-8"))
        delta = accuracy - phase3["accuracy"]
        comparison = {
            "phase3_report": str(args.compare),
            "phase3_accuracy": phase3["accuracy"],
            "phase5_accuracy": accuracy,
            "delta": delta,
            "identical_accuracy": abs(delta) < 1e-12,
        }
        print(f"\n  Phase 3 (python features): {phase3['accuracy']:.4f}")
        print(f"  Phase 5 (C++ features):    {accuracy:.4f}")
        print(f"  delta:                     {delta:+.4f}")

        # Phase 3's test table is test-rows-only, so its ids start at 0; Phase 5 keeps train and
        # test in one table, so its test ids start at n_train. Align by that offset rather than
        # by position.
        phase3_predictions = ROOT / "data" / "phase3" / args.dataset / "predictions.json"
        if phase3_predictions.exists():
            other = {
                int(r["id"]): r["yhat"]
                for r in json.loads(phase3_predictions.read_text(encoding="utf-8"))
            }
            if len(other) == len(by_id):
                offset = min(by_id) - min(other)
                agree = sum(1 for k, v in other.items() if by_id.get(k + offset) == v)
                comparison["rows_compared"] = len(other)
                comparison["rows_agreeing"] = agree
                comparison["identical_predictions"] = agree == len(other)
                print(f"  per-row agreement:         {agree}/{len(other)}")
                if agree != len(other):
                    failures.append(
                        f"C++ and Python features disagree on {len(other) - agree} rows"
                    )
            else:
                comparison["rows_compared"] = None
                print("  per-row agreement:         skipped (row counts differ)")

    # Printed last so that failures found during the comparison are reported too.
    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)

    # Where this number came from, recorded rather than assumed. PLAN.md's rule is that every
    # number in a table comes from a pod; a report that cannot say which it is cannot be checked
    # against that rule. A cgroup limit is the observable that distinguishes a container from
    # the bare box -- it is what the budget lookup already found.
    containerised = budget_source.startswith("cgroup")
    environment = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "containerised": containerised,
    }
    if platform.system() == "Windows":
        caveat = ("local Windows timing on a contended box. WDDM spills to host RAM instead of "
                  "raising OOM, so local timings are not trustworthy even directionally "
                  "(PLAN.md); PLAN.md requires reported numbers to come from a pod")
    elif not containerised:
        caveat = ("no cgroup limit found, so this is a bare box rather than a pod; PLAN.md "
                  "requires reported numbers to come from a pod")
    else:
        caveat = None

    report = {
        "dataset": args.dataset,
        "model": MODEL,
        "config": {
            "num_kernels": config.num_kernels,
            "n_groups": config.n_groups,
            "kernels_per_group": config.kernels_per_group,
            "features_per_group": config.features_per_group,
            "n_estimators": config.n_estimators,
            "seed": config.seed,
            "threads": args.threads,
            # Part of the environment tuple, not a tuning knob: a timing is not comparable
            # against another run that was given a different budget, or none.
            "memory_limit": memory_limit,
            "memory_budget_source": budget_source,
            "test_chunk": args.test_chunk,
            "onnx_threads": onnx_threads,
            "cpu_count_source": core_source,
        },
        "shape": meta,
        "accuracy": accuracy,
        "row_alignment": facts,
        # Answers PLAN.md's standing risk "TabPFN inference dominates runtime, making C++ ROCKET
        # pointless" with a number instead of an intuition, per dataset.
        "time_split": split,
        "failures": failures,
        "seconds": round(seconds, 1),
        "comparison": comparison,
        "environment": environment,
        # Observed, not hardcoded. This string used to say "local Windows timing on a contended
        # box" unconditionally, so every pod run -- the whole point of which is to produce
        # reportable numbers -- archived itself as unreportable.
        "caveat": caveat,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if not args.keep_sql:
        script.unlink(missing_ok=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
