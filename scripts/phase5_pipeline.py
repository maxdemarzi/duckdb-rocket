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
from duckdb_rocket.shells import built_shell, rocket_shell  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import normalize_series  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SHELL = built_shell()
MODEL = "tabicl-v2"

#: Every in-context model the extension can run for classification, measured on the 2026.08.14
#: community build after scripts/convert_model_weights.sh. The non-commercial two are absent by
#: choice, not by capability -- see that script.
LABELLERS = ("tabicl-v2", "mitra", "tabpfn-v2", "orion-bix")


def subsample_context(x, y, max_rows: int, seed: int):
    """Keep at most `max_rows` labelled rows, stratified, deterministically.

    **The training rows are the teacher's context, and the context is what the call costs.**
    `tabfm_classify` has no trained weights for the task, so every call re-encodes these rows --
    measured at ~14 ms per training row per group, which is 71-80% of a full-batch teacher call
    (RESULTS.md, "What routing actually costs"). Halving the context therefore halves the dominant
    term, and the only question is what it does to accuracy.

    Stratified rather than uniform because the small end of this sweep is small: 25% of Beef's 30
    rows is 7, and a uniform draw can drop a class entirely, which would look like the context size
    mattering when what happened is that a label went missing.
    """
    y = np.asarray(y)
    if max_rows >= len(y):
        return x, y, np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # One row per class first, then the remainder shared out in proportion. Guarantees every class
    # survives however small the budget, which is the property a uniform draw lacks.
    take = np.ones(len(classes), dtype=int)
    spare = max_rows - len(classes)
    if spare > 0:
        share = np.floor(spare * counts / counts.sum()).astype(int)
        take = np.minimum(take + share, counts)
        while take.sum() < max_rows and (take < counts).any():
            take[np.argmax(np.where(take < counts, counts - take, -1))] += 1
    keep = []
    for c, k in zip(classes, take):
        idx = np.nonzero(y == c)[0]
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return x[keep], y[keep], keep


def resample_split(x_train, y_train, x_test, y_test, resample: int):
    """Re-split the pooled data the way the paper's protocol does, holding every size fixed.

    **This is the missing axis, and it is the one that decides whether any recent result here is
    real.** Every number in RESULTS.md comes from a single train/test split -- the one the archive
    ships -- and the effects being chased are smaller than what one split can resolve. Beef has 30
    test rows, so a single row is 0.0333 of accuracy; G=10 costs -0.0033 and the best ensemble rule
    found gains +0.0024. Both are an order of magnitude under the quantisation, never mind the
    variance. The paper averages 30 resamples for exactly this reason.

    `--seed` does NOT do this and cannot be substituted for it: it varies the kernel bank and the
    context subsample while the split stays put, so it measures a different noise term -- the one
    that is already small. Split luck is the term that dominates, and only this touches it.

    resample=0 returns the archive's own split untouched, byte for byte, so every archived result
    reproduces. resample>=1 pools train and test, then draws a new split preserving the ORIGINAL
    PER-CLASS train count -- not merely the total. Preserving only the total would let the class
    balance of the context drift between resamples, and the context composition is itself a
    treatment: an in-context model reads those rows as its entire training signal. Two resamples
    that disagree because one of them happened to draw a thinner minority class would be measuring
    the sampler, not the pipeline.
    """
    if resample == 0:
        return x_train, y_train, x_test, y_test

    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    # Seeded by the resample index alone, so resample k is the same split whatever --seed says.
    # That independence is the point: it lets one run vary the split with the kernel bank held
    # fixed, which is the only way to attribute a difference to one of them.
    rng = np.random.default_rng(resample)
    train_idx, test_idx = [], []
    for c in np.unique(y_all):
        idx = np.nonzero(y_all == c)[0]
        rng.shuffle(idx)
        n_c = int(np.sum(y_train == c))          # this class's original train count
        train_idx.append(idx[:n_c])
        test_idx.append(idx[n_c:])
    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    # A class present only in test would give n_c = 0 and contribute nothing to the context, which
    # is a real property of the archive split rather than a bug -- but it must not silently change
    # the shapes, because the row-alignment assertions downstream are stated in those terms.
    if len(train_idx) != len(y_train) or len(test_idx) != len(y_test):
        raise ValueError(
            f"resample {resample} produced {len(train_idx)}/{len(test_idx)} train/test rows, "
            f"expected {len(y_train)}/{len(y_test)}; the pooled class counts do not admit the "
            f"archive's per-class train sizes")
    return x_all[train_idx], y_all[train_idx], x_all[test_idx], y_all[test_idx]


def write_raw_parquet(dataset: str, outdir: Path, normalize: bool,
                      max_train_rows: int = 0, seed: int = 0,
                      resample: int = 0) -> tuple[dict, np.ndarray]:
    """Write the dataset as one table of (id, split, label, values DOUBLE[]).

    Series normalisation stays a caller-side step (SPEC.md 7) and is therefore done here rather
    than inside `rocket_transform`; doing it in the extension would silently change what the
    golden vectors mean.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    x_train, y_train = load(dataset, "train")
    x_test, y_test = load(dataset, "test")
    # Before the context subsample, not after: --max-train-rows thins the context the model sees,
    # and thinning a split is a different operation from thinning a pool. Reversing these would
    # make --resample silently undo --max-train-rows by drawing from the full pool again.
    x_train, y_train, x_test, y_test = resample_split(x_train, y_train, x_test, y_test, resample)
    if max_train_rows:
        x_train, y_train, _ = subsample_context(x_train, y_train, max_train_rows, seed)
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


def ts_feature_names(shell: Path, raw_parquet: str) -> list[str]:
    """The `ts_features_by` output columns, in table order, read from the database.

    Ordered names are needed at SQL-generation time -- for the `features := [...]` argument, the
    projection and the id-recovery key, which must agree -- and DESCRIBE is the only authority on
    the order. `ts_features_list()` is not a substitute: it has 117 rows against the table's 116
    columns -- those rows are feature *definitions*, some parameterised -- and its order is not
    the column order, so reading it would both miscount and silently transpose.

    Probed against two real rows of the actual dataset rather than a synthetic series, so a dataset
    whose shape the extension rejects fails here, before a pod run rather than during one.
    """
    sql = f"""
INSTALL anofox_forecast FROM community;
LOAD anofox_forecast;
CREATE TABLE probe_long AS
  SELECT id, u.i AS ts, u.v AS v
  FROM (SELECT id, values FROM read_parquet('{raw_parquet}') LIMIT 2),
       unnest(values) WITH ORDINALITY AS u(v, i);
SELECT column_name FROM (DESCRIBE SELECT * FROM ts_features_by('probe_long', id, ts, v));
"""
    r = subprocess.run([str(shell), "-noheader", "-list", "-c", sql],
                       capture_output=True, text=True)
    cols = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    names = [c for c in cols if c != "id"]
    if not names:
        raise RuntimeError(f"could not read ts feature names.\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
    return names


def _per_group_export(outdir: Path) -> str:
    """Dump each group's own probabilities, not just the average over them.

    **This turns a sweep over the group count into one run.** Group g covers global kernel indices
    [g*kpg, (g+1)*kpg), and the prediction is the argmax of the MEAN of the groups' probabilities.
    So as long as kernels_per_group is held fixed -- which means scaling --num-kernels with
    --n-groups, since kernels_per_group is num_kernels // n_groups -- a G-group run reads exactly
    groups 0..G-1 of the bank a 40-group run reads, and averaging the first G groups here
    reproduces its predictions exactly rather than approximately.

    Concretely: `--n-groups 10 --num-kernels 2500` is the 40-group run's first ten groups. It is
    NOT `--n-groups 10 --num-kernels 10000`, which would make each group 1000 kernels wide -- 2000
    features against tabicl's 512 cap -- and is a different experiment.

    One row per (group, row, class): 40 x n_test x n_classes, a few MB at UCR sizes. Behind a flag
    because it is dead weight for a normal run.
    """
    return f"""
.once '{(outdir / "per_group.json").as_posix()}'
SELECT grp, id, e.key AS cls, e.value AS p
FROM all_groups, UNNEST(map_entries(proba)) AS t(e)
ORDER BY grp, id, cls;
"""


def build_sql(config: RocketPFNConfig, meta: dict, outdir: Path, threads: int,
              memory_limit: str, temp_dir: Path, test_chunk: int | None,
              onnx_threads: int, load_rocket: str = "", device: str = "cpu",
              model: str = MODEL, anofox_extension: Path | None = None,
              register_dir: Path | None = None, features: str = "rocket",
              ts_names: list[str] | None = None, per_group: bool = False,
              tabfm_max_memory: str | None = None, context_cache: bool = False) -> str:
    # Which feature families the classifier sees. `rocket` is the 500 random-convolution features
    # per group that every result so far uses. `ts` is anofox_forecast's 116 statistics, which beat
    # 10,000 ROCKET features on three of six hard datasets under a ridge. `both` is the open
    # question: concatenation gained nothing under a ridge (+0.0017 mean), but a ridge on 500
    # standardised random features drowns 116 statistics, and an in-context model need not.
    #
    # 500 + 116 = 616 stays inside the max_features raised above.
    if features not in ("rocket", "ts", "both"):
        raise ValueError(f"features must be rocket, ts or both, not {features!r}")
    ts_names = list(ts_names or [])
    if features in ("ts", "both") and not ts_names:
        raise ValueError(f"features={features} needs ts_names; probe them with ts_feature_names()")

    n_features = config.features_per_group
    use_rocket = features in ("rocket", "both")
    use_ts = features in ("ts", "both")

    rocket_names = [f"f{j}" for j in range(n_features)] if use_rocket else []
    ts_only_names = list(ts_names) if use_ts else []

    # Two forms of every name, because they are read in two ways and conflating them is a silent
    # failure. `names` is what goes inside the string literals of `features := ['...']`; `quoted` is
    # the identifier. They differ for the ts columns: those are the extension's own names, not ones
    # we chose, and `quantile_0.1` contains a dot that DuckDB otherwise reads as a qualifier.
    names = rocket_names + ts_only_names
    quoted = rocket_names + [f'"{n}"' for n in ts_only_names]

    # The ts features are computed once per SERIES, not per group -- they have no kernel bank and so
    # no ensemble axis. With features=ts every one of the 40 groups would therefore see identical
    # columns and score identically, which is 40x the cost of one group for none of the benefit.
    #
    # The rocket bank still runs in ts mode, cheaply, because it is what feeds the kernel-bank
    # fingerprint and the id/split/label columns; only the classifier stops seeing it. So ts mode
    # wants a SMALL --num-kernels as well as one group -- 10,000 over one group is 20,000 features
    # per group, which RocketPFNConfig rejects against the feature cap, and the error would arrive
    # sounding like a cap problem rather than a mode problem.
    if features == "ts" and config.n_groups != 1:
        raise ValueError("features=ts has no per-group variation; use --n-groups 1 "
                         "(and a small --num-kernels, e.g. 500: the bank only feeds the "
                         "integrity fingerprint in this mode)")
    # The id-recovery key is the WHOLE feature vector, not a prefix of it.
    #
    # A prefix is a bet on how many leading features it takes to separate two series, and the bet
    # was lost twice. One column measured zero collisions across all ten datasets of the original
    # subset -- which is exactly why the weakness survived -- and then fanned rows out to 75 and 80
    # groups of 40 on ScreenType and InlineSkate. Widening to four fixed neither. Sixteen fixed
    # InlineSkate and left five distinct ScreenType series still sharing a key. Each widening was
    # the same guess with a bigger number.
    #
    # The full vector ends the question by construction: two rows share this key only if they ARE
    # the same feature vector, and identical vectors get identical predictions, so collapsing them
    # (the GROUP BY in `score`) is exact rather than merely tolerable. Measured on v1.5.5 before
    # being relied on -- a DOUBLE[] equality plans as a HASH_JOIN, not a nested loop, and DuckDB
    # holds NaN = NaN, so a non-finite feature cannot silently drop a row out of the join.
    key_decl = "k DOUBLE[]"
    key_join = "s.k = [" + ", ".join(f"c.{n}" for n in quoted) + "]"
    feature_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"

    # The projection that turns stored features into the named scalar columns the classifier takes.
    # DuckDB lists are 1-based, so rocket feature j lives at r.f[j + 1].
    #
    # ts columns are guarded for finiteness. They are unbounded statistics on real data -- a
    # near-constant series makes a variance-normalised one non-finite -- and the screen measured
    # between 101 and 540 non-finite values per dataset. A NaN reaching the classifier is not a
    # loud failure, it is a quietly worse number.
    proj_parts = [f"r.f[{j + 1}] AS f{j}" for j in range(n_features)] if use_rocket else []
    proj_parts += [f'CASE WHEN isfinite(t."{n}") THEN t."{n}" ELSE 0.0 END AS "{n}"'
                   for n in ts_only_names]
    projection = ", ".join(proj_parts)

    # The id-recovery key must be the whole vector the classifier echoes back, in the same order as
    # `features`. With ts columns that is the rocket list concatenated with the guarded ts values --
    # `||` on LISTs -- and the order here and in key_join above are the one invariant that cannot
    # drift, which is why a test compares them element by element rather than trusting this.
    ts_key_list = ("[" + ", ".join(f'CASE WHEN isfinite(t."{n}") THEN t."{n}" ELSE 0.0 END'
                                   for n in ts_only_names) + "]") if use_ts else ""
    if use_rocket and use_ts:
        key_from_list = f"r.f || {ts_key_list}"
    elif use_ts:
        key_from_list = ts_key_list
    else:
        key_from_list = "r.f"

    # feat_cur holds the rocket features; tsfeat holds the per-series ts features. Every statement
    # that reads features reads this same FROM clause, so the two can never fall out of step.
    feat_from = "feat_cur r" + (" JOIN tsfeat t USING (id)" if use_ts else "")

    # `LOAD anofox_tabfm` takes the installed (community, CPU-only) extension. A GPU run needs a
    # self-built cuda flavor loaded from a path instead -- no GPU build is published for any
    # platform (the ext.anofox.com host in anofox's own error message does not resolve).
    load_anofox = (f"LOAD '{anofox_extension.as_posix()}';" if anofox_extension
                   else "LOAD anofox_tabfm;")

    parts = [
        load_rocket,
        load_anofox,
        "SET anofox_tabfm_accept_hf_license = true;",
        # Costs nothing to raise. Read the extension's source rather than guessing: it is a
        # bind-time guard only -- `if (fields.size() > max_features) throw BinderException` in
        # tabfm_generate.cpp -- and sizes no allocation. The comparison is strictly greater, so
        # {n_features} features would in fact pass at exactly {n_features}; the doubling is
        # margin, not necessity. Left as it is because it is free, and noted here so nobody
        # investigates it a second time looking for a memory or speed win. There isn't one.
        f"SET anofox_tabfm_max_features = {max(n_features * 2, 1000)};",
        # Opt-in, and it must stay opt-in: `anofox_tabfm_max_memory` landed upstream in #36 and is
        # in no released build, so emitting it unconditionally would make every run against the
        # community extension fail on an unrecognised parameter. What it buys is the difference
        # between a diagnosis and a guess -- the setting reads VmRSS from /proc/self/status and
        # refuses a predict call above the ceiling, where today the kernel takes the process with
        # exit -9 and an empty stderr. That failure mode cost this project several sessions and one
        # withdrawn explanation on SemgHandMovementCh2.
        *( [f"SET anofox_tabfm_max_memory = '{tabfm_max_memory}';"] if tabfm_max_memory else [] ),
        # Opt-in for the same reason, and with a second condition on top of the build: the model
        # must ship the support/query graph pair from #38, which no published one does. So this
        # needs BOTH a #40 build (--anofox-extension) and a model directory carrying the pair
        # (--register-model-dir), and it is silently inert without the second -- the engine falls
        # back to the combined graph when it cannot find all four artifacts.
        #
        # What it changes is the shape of the work, not the answer: the labelled context is encoded
        # once per support set instead of once per call. Measured through the extension at 7.2x on
        # a repeated call and 0.36x on the first, so the case that gains is --test-chunk, where a
        # group's chunks all share one context. A run without --test-chunk makes one call per group
        # against a context that changes every time, and pays the cold penalty 40 times for nothing.
        *( ["SET anofox_tabfm_context_cache = true;"] if context_cache else [] ),
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
        # 'cpu' is emitted explicitly rather than left to anofox's 'auto' default, so a run's
        # device is recorded in the SQL that produced it rather than inferred from the box.
        f"SET anofox_tabfm_device = '{device}';",
        # A registered model points at OUR graph file. Needed on CUDA: the shipped tabicl-v2
        # graph fails at a ScatterND node there (DataZooDE/anofox-tabfm#21), and the workaround
        # is a graph edit (#23). Inert on CPU -- the patched graph is bit-identical there.
        (f"CALL tabfm_register_model(id := '{model}', base_dir := '{register_dir.as_posix()}', "
         f"classification_graph := 'graph_tabicl_classification.onnx', "
         f"classification_weights := 'model.ckpt', "
         f"classification_tensor_map := 'tensor_map_tabicl_classification.json', "
         f"license := 'bsd-3-clause', preprocessing_profile := 'tabicl_v2_raw');"
         if register_dir else ""),
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
        "CREATE OR REPLACE TABLE f0_checks (grp BIGINT, duplicate_series BIGINT, "
        "fingerprint DOUBLE);",
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
    schema_cols = ", ".join(f"{n} DOUBLE" for n in quoted)

    if use_ts:
        # Computed ONCE for the whole dataset, before the group loop: these statistics depend on the
        # raw series and not on the kernel bank, so recomputing them per group would be 40x the work
        # for identical numbers.
        #
        # anofox_forecast is BSL 1.1 -- production use is permitted, offering it to third parties
        # hosted or embedded is not -- so this is an opt-in experiment behind --features and never a
        # dependency of the `rocket` extension. See reference/RESULTS.md on which of the 116 are
        # worth reimplementing from the tsfresh catalogue instead.
        #
        # ts_features_by wants long format, one row per (series, timepoint), while `raw` holds one
        # row per series with a LIST. WITH ORDINALITY supplies the time index; the values are
        # already normalised, since write_raw_parquet normalises before writing.
        parts.append(f"""
INSTALL anofox_forecast FROM community;
LOAD anofox_forecast;
CREATE OR REPLACE TABLE ts_long AS
  SELECT id, u.i AS ts, u.v AS v FROM raw, unnest(values) WITH ORDINALITY AS u(v, i);
CREATE OR REPLACE TABLE tsfeat AS SELECT * FROM ts_features_by('ts_long', id, ts, v);

-- One row per series or the joins below silently multiply the feature tables.
SELECT CASE WHEN (SELECT count(*) FROM tsfeat) <> (SELECT count(*) FROM raw)
            THEN CAST('tsfeat has a different row count than raw' AS BIGINT) ELSE 0 END AS ts_check;
""")

    parts.append(f"""
CREATE OR REPLACE TABLE feat_cur (id BIGINT, split VARCHAR, label VARCHAR, f DOUBLE[]);
CREATE OR REPLACE TABLE train_cur (y VARCHAR, {schema_cols});
CREATE OR REPLACE TABLE test_cur ({schema_cols});
-- Only the two columns the join reads (swan PERFORMANCE_TUNING.md 1).
-- Id recovery: the full feature vector as the key, AND a dedup.
--
-- anofox_tabfm echoes back only the target and the columns named in `features`, so a plain id is
-- dropped and scored rows must be rejoined on feature values. The two datasets that broke the
-- prefix keys broke them for different reasons, and only one of the two is a key problem at all:
--
--   ScreenType    375 test rows, 375 DISTINCT series -- genuine feature collisions. Quantised
--                 electricity data makes ROCKET's max/PPV coincide across different series, and
--                 it kept doing so through a 16-wide key. Only the whole vector separates them.
--   InlineSkate   550 test rows, 521 distinct -- 29 series are byte-identical. No feature key
--                 can ever separate them, however wide. But identical series get identical
--                 predictions, so the GROUP BY below collapses the duplicates to one row per
--                 (group, id), which is exactly right rather than merely tolerable.
CREATE OR REPLACE TABLE test_src_cur (id BIGINT, {key_decl});
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
  INSERT INTO train_cur SELECT r.label, {projection} FROM {feat_from} WHERE r.split = 'train';

PREPARE fill_test AS
  INSERT INTO test_cur SELECT {projection} FROM {feat_from}
   WHERE r.split = 'test' AND r.id >= CAST($1 AS BIGINT) AND r.id < CAST($2 AS BIGINT);

PREPARE fill_src AS
  INSERT INTO test_src_cur SELECT r.id, {key_from_list} FROM {feat_from}
   WHERE r.split = 'test' AND r.id >= CAST($1 AS BIGINT) AND r.id < CAST($2 AS BIGINT);

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

-- Every name in the requested feature list must exist in train_cur. A name that matches no column
-- is SILENTLY DROPPED -- the macro filters with COLUMNS(lambda), which keeps what matches and says
-- nothing about what does not, so by bind time a typo is indistinguishable from a deliberate
-- omission. The call then succeeds having trained on fewer features than asked for, and no accuracy
-- this project has recorded would show it. DataZooDE/anofox-tabfm#34, fixed 2026-08-15; the
-- community build serves 2026.08.14, so it is still silent for us today.
--
-- Our names and our schema come from one list, so this can only fire on a regression. That is the
-- point: it costs one statement per run to assert, and the alternative is to keep assuming it.
SELECT CASE
         WHEN (SELECT count(*) FROM (SELECT unnest({feature_list}) AS n)
                WHERE lower(n) NOT IN (SELECT lower(column_name) FROM (DESCRIBE train_cur))) > 0
         THEN CAST('the requested feature list names a column that train_cur does not have; the '
                   'extension drops it silently and trains on the rest' AS BIGINT)
         ELSE 0
       END AS features_check;

-- test_cur omits the target: tabfm_classify unions train and test BY NAME, and a target present
-- in both is a duplicate-name binder error naming neither cause (Phase 2).
-- GROUP BY, not a bare join: byte-identical test series join many-to-many and would otherwise be
-- counted once per pairing, inflating a row's group count without changing its average. Their
-- predictions are identical by construction, so collapsing them is exact.
PREPARE score AS
  INSERT INTO all_groups
  SELECT grp, id, any_value(proba) FROM (
    SELECT CAST($1 AS BIGINT) AS grp, s.id AS id, c.proba AS proba
    FROM tabfm_classify('train_cur', 'y', test := 'test_cur',
                        model := '{model}', features := {feature_list}) c
    JOIN test_src_cur s ON {key_join}
  ) GROUP BY grp, id;

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
-- Duplicate test series: reported, not asserted. Under a prefix key this column counted distinct
-- series the key failed to separate, and that number could be non-zero -- it was 5 on ScreenType
-- at a width of 16. It cannot be non-zero now: the key is the full vector, so distinct vectors
-- have distinct keys by construction and the old expression would be a tautology dressed up as a
-- check. What is worth recording instead is how many test series are byte-identical to another
-- (InlineSkate: 29), because that is the reason the GROUP BY in `score` exists.
--
-- The failure the old column stood in for is covered directly by min/max groups per row.
-- Counted on the actual join key, not on the rocket vector: under --features ts the key is the
-- statistics and a duplicate count over `f` would describe columns the classifier never saw.
-- The fingerprint stays r.f[1], which is the kernel bank and is what that column is for.
SELECT {g}, count(*) - count(DISTINCT ({key_from_list})), sum(r.f[1])
  FROM {feat_from} WHERE r.split = 'test';
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

.once '{(outdir / "soft_labels.json").as_posix()}'
-- The averaged class probabilities, not only the argmax. Distillation wants these: a student
-- trained on a hard argmax inherits the teacher's decisions, while one trained on the distribution
-- inherits its uncertainty too, which is the part worth having on datasets where the teacher is
-- only slightly better than the student (docs/DISTILLATION_PLAN.md).
--
-- Written for every run, not behind a flag. It is one row per (id, class) -- 375 x 3 on ScreenType
-- -- so it costs nothing, and the alternative is discovering it was needed after the pod is gone.
-- That has already happened once: six hard datasets had to be re-run because only accuracy was kept.
SELECT id, cls, mean_p FROM per_class ORDER BY id, cls;
{_per_group_export(outdir) if per_group else ""}

.once '{(outdir / "timings.json").as_posix()}'
-- Seconds spent computing ROCKET features, versus seconds spent in tabfm_classify. The gap
-- between (transform_done -> classify_done) and the classify calls is the chunk fills and the
-- id-recovery joins, which are counted as classify here because they exist only to feed it.
-- One row PER GROUP, not a sum. Summing here threw away the only evidence that could explain an
-- anomaly: SyntheticControl once took 2975s of classify where the same dataset took 653s an hour
-- earlier, and with only the total there was no way to tell a steady slowdown from one stalled
-- group. The caller aggregates; the archive keeps the detail.
WITH t AS (
  SELECT grp,
         max(ts) FILTER (WHERE phase = 'group_start')    AS t0,
         max(ts) FILTER (WHERE phase = 'transform_done') AS t1,
         max(ts) FILTER (WHERE phase = 'classify_done')  AS t2
  FROM timings GROUP BY grp
)
SELECT grp,
       round(epoch(t1 - t0), 3) AS transform_seconds,
       round(epoch(t2 - t1), 3) AS classify_seconds
FROM t ORDER BY grp;

.once '{(outdir / "assertions.json").as_posix()}'
-- The row-alignment counts are the id-recovery test. anofox_tabfm echoes back only the target and
-- the columns named in `features`, so a plain id column is dropped and scored rows are rejoined to
-- their ids on feature values; rows sharing the join key fan that join out and score against each
-- other's ids. `min_groups_per_row` and `max_groups_per_row` catch that directly -- both must be
-- exactly {config.n_groups}. The fan-out that started this only ever moved `max` (to 75 and 80),
-- so a run checking `min` alone passed while averaging a duplicated ensemble.
--
-- `duplicate_test_series` is descriptive. Now that the key is the whole feature vector, distinct
-- series cannot share a key, and what remains is byte-identical series -- which no key of any
-- width could separate, and which `score` collapses exactly because their predictions are equal.
SELECT (SELECT count(*) FROM predictions)          AS predicted_rows,
       (SELECT count(DISTINCT id) FROM all_groups) AS distinct_ids,
       (SELECT count(*) FROM all_groups)           AS group_rows,
       (SELECT min(n_groups_seen) FROM per_class)  AS min_groups_per_row,
       (SELECT max(n_groups_seen) FROM per_class)  AS max_groups_per_row,
       -- CAST because max() over BIGINT can widen to HUGEINT, and `.mode json` renders HUGEINT as
       -- a *string* to avoid precision loss. "0" is truthy in Python, so a guard reading this
       -- fired on every run while reporting zero.
       -- max(), not sum(): every group reports the same count, so summing multiplied it by 40.
       (SELECT CAST(coalesce(max(duplicate_series), 0) AS BIGINT)
          FROM f0_checks)                         AS duplicate_test_series,
       (SELECT CAST(count(DISTINCT fingerprint) AS BIGINT)
          FROM f0_checks)                         AS distinct_group_banks;
""")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--model", default=MODEL, choices=LABELLERS,
                        help="which in-context model labels the test rows; the three besides "
                             "tabicl-v2 need scripts/convert_model_weights.sh first")
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
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Execution device for anofox_tabfm. 'cuda' needs a self-built cuda-flavor "
             "extension (--anofox-extension) because no GPU build is published, and for "
             "tabicl-v2 it also needs the patched graph (--register-model-dir): the shipped "
             "graph fails at a ScatterND node on CUDA (DataZooDE/anofox-tabfm#21).",
    )
    parser.add_argument(
        "--features",
        default="rocket",
        choices=("rocket", "ts", "both"),
        help="Which feature families the classifier sees. 'rocket' is the 500 "
             "random-convolution features per group behind every result so far. 'ts' is "
             "anofox_forecast's 116 in-database statistics, which beat 10,000 ROCKET features on "
             "three of six hard datasets under a ridge -- univariate only, and it forces "
             "--n-groups 1 because those statistics have no kernel bank and so no ensemble axis. "
             "'both' is the open question: concatenation gained nothing under a ridge, but a ridge "
             "drowns 116 statistics in 500 random features and an in-context model need not. "
             "anofox_forecast is BSL 1.1, so ts and both are experiments, never a dependency.",
    )
    parser.add_argument(
        "--anofox-extension",
        type=Path,
        default=None,
        help="Path to an anofox_tabfm.duckdb_extension to LOAD instead of the installed one.",
    )
    parser.add_argument(
        "--register-model-dir",
        type=Path,
        default=None,
        help="Directory holding graph_tabicl_classification.onnx, model.ckpt and "
             "tensor_map_tabicl_classification.json. Registered under the model id so the run "
             "uses that graph. On CPU the patched graph is bit-identical to the shipped one.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="subsample the labelled context to at most this many rows, stratified (0 = all). "
             "The context is re-encoded on every classify call and is 71-80%% of a full-batch "
             "teacher call, so this is the one knob that cuts the dominant term without an "
             "upstream change.",
    )
    parser.add_argument(
        "--tabfm-max-memory",
        help="set anofox_tabfm_max_memory, which refuses a predict call once resident memory is "
             "above the ceiling (e.g. '24GB'). Unreleased upstream as of 2026-08-16, so it is "
             "off by default: a build without it fails on an unrecognised parameter. This is the "
             "only setting that turns an OOM kill into an error you can read -- memory_limit "
             "governs DuckDB alone and the model allocates outside it.")
    parser.add_argument(
        "--workdir", type=Path, default=None,
        help="where raw.parquet, predictions.json and the DuckDB temp directory go. Defaults to "
             "data/phase5/<dataset>, which is shared by every run of that dataset -- fine one at "
             "a time, and a data race the moment two run concurrently. Any driver running jobs in "
             "parallel must give each one its own.")
    parser.add_argument(
        "--resample", type=int, default=0,
        help="which train/test split to use. 0 (default) is the archive's own, so archived "
             "results reproduce unchanged; 1..N are stratified re-splits of the pooled data at "
             "the same per-class sizes, seeded by this number alone. This is the axis every "
             "result in RESULTS.md is missing -- one split cannot separate a real half-point from "
             "a lucky one, and the paper averages 30. Not interchangeable with --seed, which "
             "varies the kernel bank while the split stays put.")
    parser.add_argument(
        "--context-cache",
        action="store_true",
        help="set anofox_tabfm_context_cache, which encodes the labelled context once per support "
             "set instead of once per classify call. Unreleased upstream as of 2026-08-16 "
             "(DataZooDE/anofox-tabfm#40), and it additionally needs a --register-model-dir whose "
             "graphs include the split pair from #38 -- without that the engine quietly uses the "
             "combined graph and this flag does nothing. Pair it with --test-chunk: the cache pays "
             "off across chunks of one group, and costs about 2.5x on each group's first call.")
    parser.add_argument(
        "--per-group-soft",
        action="store_true",
        help="also archive each group's own probabilities, not just their average. One run then "
             "answers the whole group-count sweep exactly: averaging the first G groups is what a "
             "G-group run computes, provided --num-kernels is scaled with --n-groups to hold "
             "kernels-per-group fixed.",
    )
    parser.add_argument("--keep-sql", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not args.anofox_extension:
        parser.error("--device cuda needs --anofox-extension: the installed community build is "
                     "CPU-only and would silently run on the CPU.")

    if not args.shell.exists():
        print(f"no such shell: {args.shell}\nBuild with scripts/build_extension.bat",
              file=sys.stderr)
        return 1

    # The cache engages only if the engine finds ALL FOUR split artifacts beside the combined
    # graph, and falls back to the combined graph in silence if it does not. Silence is the whole
    # problem: the run then completes, reports a normal accuracy, and its timing gets written down
    # as "with the cache" when nothing was cached. Checked here, against the same filenames the
    # engine derives, because a precondition that is cheap to verify should never be inferred from
    # a wall clock afterwards.
    if args.context_cache:
        if not args.register_model_dir:
            parser.error("--context-cache needs --register-model-dir: no published model ships "
                         "the split pair, so there is nothing for the engine to find.")
        missing = [n for n in ("graph_prepare_tabicl_classification.onnx",
                               "graph_query_tabicl_classification.onnx",
                               "tensor_map_prepare_tabicl_classification.json",
                               "tensor_map_query_tabicl_classification.json")
                   if not (args.register_model_dir / n).exists()]
        if missing:
            parser.error(f"--context-cache: {args.register_model_dir} is missing "
                         f"{', '.join(missing)}. Export them with "
                         f"tools/export_tabicl --split-context; without all four the engine uses "
                         f"the combined graph and the run would be mistimed rather than fail.")

    config = RocketPFNConfig(
        num_kernels=args.num_kernels, n_groups=args.n_groups, seed=args.seed, n_estimators=1
    )
    config.validate()
    out = args.out or ROOT / "reference" / f"phase5_{args.dataset}.json"
    # Every run of a dataset shared one directory until concurrent resamples collided in it.
    # `raw.parquet`, `predictions.json` and the DuckDB temp directory all live here, so two runs
    # of the same dataset raced on writing and reading the same files. The crash it produced --
    # "No magic bytes found at end of file" from a parquet read mid-write -- was the LUCKY
    # outcome: the same race can just as easily have one resample read the split another wrote,
    # and then every number is quietly attributed to the wrong split.
    workdir = args.workdir or (ROOT / "data" / "phase5" / args.dataset)

    print(f"config: {config.n_groups} groups x {config.kernels_per_group} kernels "
          f"= {config.features_per_group} features/group")

    print(f"\n[1/3] writing raw series -> {workdir}", flush=True)
    meta, y_test = write_raw_parquet(args.dataset, workdir, config.normalize,
                                     args.max_train_rows, args.seed, args.resample)
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

    shell, shell_args, load_rocket = rocket_shell(args.shell if args.shell != SHELL else None)
    if load_rocket:
        print(f"      {shell.name} + prebuilt rocket extension (no local build)", flush=True)
    # `LOAD '<path>'` of a locally-built extension is refused without -unsigned. The built shell
    # needs no flag for `rocket` itself (statically linked), so this is only reached when an
    # anofox extension is being loaded by path. It fails in 0.3s rather than mid-run, but it
    # fails after the dataset has been written and the SQL generated, which is a slow way to
    # learn it.
    if args.anofox_extension and "-unsigned" not in shell_args:
        shell_args = [*shell_args, "-unsigned"]

    ts_names: list[str] = []
    if args.features in ("ts", "both"):
        if meta["multivariate"]:
            print(f"      --features {args.features} is univariate only: ts_features_by takes one "
                  f"(group, time, value) triple and {args.dataset} has {meta['n_channels']} channels")
            return 1
        ts_names = ts_feature_names(shell, meta["raw_parquet"])
        print(f"      + {len(ts_names)} anofox_forecast statistics per series "
              f"({args.features}); BSL 1.1, so this is an experiment and not a dependency",
              flush=True)

    sql = build_sql(config, meta, workdir, args.threads, memory_limit, workdir,
                    args.test_chunk, onnx_threads, load_rocket,
                    device=args.device, model=args.model,
                    anofox_extension=args.anofox_extension,
                    register_dir=args.register_model_dir,
                    features=args.features, ts_names=ts_names,
                    per_group=args.per_group_soft,
                    tabfm_max_memory=args.tabfm_max_memory,
                    context_cache=args.context_cache)
    script = workdir / "pipeline.sql"
    script.write_text(sql, encoding="utf-8")
    print(f"[2/3] generated {len(sql):,} characters of SQL")

    print(f"[3/3] running the whole pipeline in DuckDB", flush=True)
    started = time.perf_counter()
    # stdout is INHERITED, not captured, so the per-group `.print` heartbeat reaches whoever is
    # watching. capture_output=True swallowed it -- the dot commands ran and their output went
    # into a string nobody reads unless the run fails, which is exactly backwards for a progress
    # signal on a four-hour job.
    #
    # stderr stays piped: it carries thousands of lines of ONNX schema-registration noise that
    # has to be filtered before anything useful is visible.
    #
    # Caveat: with stdout redirected to a file the CLI block-buffers, so the heartbeat arrives in
    # bursts rather than smoothly. Still the difference between "progressing" and "possibly hung".
    proc = subprocess.run(
        [str(shell), *shell_args, "-f", str(script)],
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
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
        rows = json.loads(timing_path.read_text(encoding="utf-8"))
        # Tolerate the older single-row aggregate form so archived runs still load.
        if rows and "grp" in rows[0]:
            per_group = [(float(r["transform_seconds"]), float(r["classify_seconds"]))
                         for r in rows]
            tr = sum(t for t, _ in per_group)
            cl = sum(c for _, c in per_group)
            classify_each = sorted(c for _, c in per_group)
            mid = len(classify_each) // 2
            median = (classify_each[mid] if len(classify_each) % 2
                      else (classify_each[mid - 1] + classify_each[mid]) / 2)
            # Slowest against median: a shared box that stalls one group looks nothing like one
            # that is uniformly slow, and the total cannot tell them apart.
            spread = {
                "classify_min": round(classify_each[0], 3),
                "classify_median": round(median, 3),
                "classify_max": round(classify_each[-1], 3),
                "max_over_median": round(classify_each[-1] / median, 2) if median else None,
            }
        else:
            row = rows[0]
            tr, cl = float(row["transform_seconds"]), float(row["classify_seconds"])
            per_group, spread = [], None

        total = tr + cl
        split = {
            "transform_seconds": round(tr, 2),
            "classify_seconds": round(cl, 2),
            "classify_share": round(cl / total, 4) if total else None,
            "groups_timed": len(per_group) or int(rows[0].get("groups_timed", 0)),
            "per_group_classify_spread": spread,
        }
        if total:
            msg = f"\n  transform {tr:.1f}s | classify {cl:.1f}s ({100 * cl / total:.1f}%)"
            if spread:
                msg += (f"\n  per-group classify: min {spread['classify_min']:.1f}s "
                        f"median {spread['classify_median']:.1f}s "
                        f"max {spread['classify_max']:.1f}s "
                        f"({spread['max_over_median']:.1f}x median)")
            print(msg)

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
    # No guard on duplicate_test_series. It is not a failure: byte-identical series get identical
    # predictions and `score` collapses them, so the count is context for the reader rather than
    # something to assert on. The assertion that matters is the pair below.
    if facts["min_groups_per_row"] != config.n_groups:
        failures.append(
            f"a row was scored by only {facts['min_groups_per_row']} of {config.n_groups} "
            f"groups; averaging a partial ensemble is silently wrong"
        )
    # The symmetric case, which was missing and is the one that actually happened: the
    # id-recovery join fanned out and rows were scored 75 and 80 times against 40 groups. `min`
    # stayed at 40 throughout, so only the collision count noticed, and that is a proxy for this
    # rather than a test of it.
    if facts["max_groups_per_row"] != config.n_groups:
        failures.append(
            f"a row was scored by {facts['max_groups_per_row']} of {config.n_groups} groups; "
            f"the id recovery fanned out and that row's average counts some groups twice"
        )

    by_id = {int(r["id"]): r["yhat"] for r in predictions}
    ordered_ids = sorted(by_id)
    y_pred = np.asarray([by_id[i] for i in ordered_ids])
    accuracy = float((y_pred == y_test).mean()) if len(y_pred) == n_test else float("nan")

    print(f"\n  accuracy ({args.model}, e=1, G={config.n_groups}): {accuracy:.4f}")
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
        "model": args.model,
        # Which device produced this number, and whether it came from the shipped graph or a
        # patched one. Without both, a GPU result is indistinguishable from a CPU result that
        # silently fell back -- and on CUDA the graph is not the shipped graph.
        "device": args.device,
        # Which feature families the classifier actually saw, and how many columns that was.
        # `config.features_per_group` is the ROCKET half alone, so on a --features both run it
        # reads 500 while the model saw 616 -- and the first archived batch of those reports
        # recorded the mode nowhere but in the filename. A number whose provenance lives in a
        # filename is a number that will eventually be misread.
        "features": args.features,
        "n_feature_columns": (len(ts_names) if args.features == "ts"
                              else config.features_per_group + len(ts_names)),
        "ts_feature_source": ("anofox_forecast (BSL 1.1, experiment only)" if ts_names else None),
        "anofox_extension": (str(args.anofox_extension) if args.anofox_extension else None),
        "registered_graph": (str(args.register_model_dir) if args.register_model_dir else None),
        "config": {
            "num_kernels": config.num_kernels,
            "n_groups": config.n_groups,
            "kernels_per_group": config.kernels_per_group,
            "features_per_group": config.features_per_group,
            "n_estimators": config.n_estimators,
            "seed": config.seed,
            # Which split this is. Recorded next to the seed because the two are separate axes and
            # a report that names only one of them cannot be placed: resample 0 seed 0 and
            # resample 7 seed 0 are different experiments that would otherwise look identical.
            "resample": args.resample,
            "threads": args.threads,
            # Part of the environment tuple, not a tuning knob: a timing is not comparable
            # against another run that was given a different budget, or none.
            "memory_limit": memory_limit,
            "memory_budget_source": budget_source,
            "test_chunk": args.test_chunk,
            # Environment tuple, not a tuning knob, and the one flag here that changes which graph
            # runs. A timing is not comparable against a run that encoded the context differently.
            "context_cache": args.context_cache,
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

    # Soft labels go to a sidecar rather than into the report: the report is read by people and a
    # few thousand probabilities would bury it, but they are the one artifact that cannot be
    # reconstructed after the pod is gone. `n_train` is recorded with them because the id space is
    # arange(n_train + n_test) with the train rows first, so test row k is id n_train + k -- a
    # consumer that has to rediscover that offset will eventually get it wrong.
    soft_path = out.with_name(out.stem + "_soft.json")
    soft_src = workdir / "soft_labels.json"
    if soft_src.exists():
        soft = json.loads(soft_src.read_text(encoding="utf-8"))
        by_row: dict[int, dict[str, float]] = {}
        for r in soft:
            by_row.setdefault(int(r["id"]), {})[str(r["cls"])] = float(r["mean_p"])
        soft_path.write_text(json.dumps({
            "dataset": args.dataset,
            # args.model, not `model`: that name is a parameter of build_sql and does not exist
            # scope. It cost two datasets on a rented pod, because this block only runs at the very
            # end of a full run and nothing local reaches it.
            "model": args.model,
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            # WHICH SPLIT these labels describe. A resample preserves n_train and n_test exactly,
            # so every existing consumer check -- and they do check both -- passes when a sidecar
            # from resample 3 is read against resample 7's split. Right sizes, wrong rows, no
            # error, and a routing or distillation result computed against another split's
            # teacher. The only defence is to record it and have the consumer assert it.
            "resample": args.resample,
            "note": "test row k of the dataset's test split is id n_train + k",
            "classes": sorted({c for v in by_row.values() for c in v}),
            "mean_proba": {str(k): by_row[k] for k in sorted(by_row)},
        }, indent=2), encoding="utf-8")
        print(f"wrote {soft_path}  ({len(by_row)} rows of soft labels)")
    else:
        # Loud, because a distillation run that silently has no teacher labels looks like a
        # student that learned nothing.
        print(f"WARNING: no soft labels at {soft_src}; distillation arms cannot use this run")

    # Per-group probabilities, when asked for. Written as nested arrays keyed by explicit `ids` and
    # `classes` lists rather than as one object per (group, row, class): the object form of a
    # 40 x 760 x 10 dump is ~8 MB of repeated key strings, and the array form is a fifth of that.
    pg_src = workdir / "per_group.json"
    if args.per_group_soft and pg_src.exists():
        recs = json.loads(pg_src.read_text(encoding="utf-8"))
        ids = sorted({int(r["id"]) for r in recs})
        classes = sorted({str(r["cls"]) for r in recs})
        ri = {v: i for i, v in enumerate(ids)}
        ci = {v: i for i, v in enumerate(classes)}
        cube = [[[0.0] * len(classes) for _ in ids] for _ in range(config.n_groups)]
        for r in recs:
            cube[int(r["grp"])][ri[int(r["id"])]][ci[str(r["cls"])]] = round(float(r["p"]), 6)
        pg_path = out.with_name(out.stem + "_pergroup.json")
        pg_path.write_text(json.dumps({
            "dataset": args.dataset, "model": args.model,
            "n_train": meta["n_train"], "n_test": meta["n_test"],
            "n_groups": config.n_groups,
            # The number that makes a prefix of these groups equal to a shorter run. Recorded so a
            # consumer can check it matches the run it wants to compare against instead of assuming.
            "kernels_per_group": config.kernels_per_group,
            "note": "proba[g][i][c] is group g's probability of classes[c] for row ids[i]; "
                    "averaging g < G reproduces a G-group run at the same kernels_per_group",
            "ids": ids, "classes": classes, "proba": cube,
        }), encoding="utf-8")
        print(f"wrote {pg_path}  ({config.n_groups} groups x {len(ids)} rows x "
              f"{len(classes)} classes)")
    elif args.per_group_soft:
        print(f"WARNING: --per-group-soft was asked for but {pg_src} does not exist")

    if not args.keep_sql:
        script.unlink(missing_ok=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
