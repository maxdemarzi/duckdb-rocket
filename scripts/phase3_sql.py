"""Phase 3 — the SQL composition prototype.

ROCKET features are computed in Python (the Phase 1 oracle), written to Parquet, and everything
after that happens in DuckDB: G classify calls, probability averaging across groups, argmax.
This is the milestone that proves the idea; Phase 4's C++ work is performance engineering on
top of a composition that already works.

    uv run python scripts/phase3_sql.py --dataset GunPoint
    uv run python scripts/phase3_sql.py --dataset Coffee --keep-sql   # inspect the generated SQL

**The plan's original exit criterion has to change, and it is worth being explicit about why.**
It asked Phase 3 to "reproduce Phase 1 accuracy exactly -- same features in, same predictions
out", which assumed both sides run the same classifier. They cannot: `tabpfn-v2-5` does not load
in `anofox_tabfm bc6d8af` (Phase 2, finding 1), so the oracle runs TabPFN v2.5 under PyTorch and
the SQL path runs TabICL v2 under ONNX at an unknown, unsettable precision. Identical predictions
are not a meaningful target across two different models.

What *is* meaningful, and what this script tests instead:

1. **Plumbing conformance, exactly.** The same per-group probabilities are averaged and argmaxed
   both in SQL and in Python (`numpy`), and the two must agree to within floating-point
   tolerance. This isolates the composition logic -- which is what Phase 3 exists to prove --
   from the choice of backbone entirely.
2. **Row alignment, asserted rather than assumed.** Phase 2 found output rows come back in the
   test view's order, but only checked it at 40 rows on one thread. Relying on that at UCR scale
   without checking would corrupt every number while leaving the output well-formed, so rows are
   joined back on an echoed feature value and the join is required to be total.
3. **Accuracy, reported beside the oracle's rather than equated to it.**
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.budget import default_memory_limit  # noqa: E402
from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.shells import pinned_shell  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DUCKDB = pinned_shell()

# Phase 2: tabpfn-v2-5 does not load in bc6d8af; tabicl-v2 is the working backbone.
MODEL = "tabicl-v2"


def preamble(max_features: int, threads: int = 4) -> str:
    # anofox's feature ceiling is a configurable guard rather than a model limit, but it
    # defaults to 500 and a 500-feature group sits exactly on the boundary. Set it explicitly.
    #
    # threads and memory_limit are pinned for the reason phase5_pipeline.py pins them: inside a
    # container neither `nproc` nor `free` reports this process's budget. A cgroup OOM kill
    # leaves no DuckDB error and no Python traceback, so it reads as a hang.
    #
    # This does NOT make Phase 3 safe on a large test split. `tabfm_classify` below is called
    # once for the whole split, and its allocation is ONNX's rather than the buffer manager's,
    # so no memory_limit contains it -- on Phase 5 a 6 GB limit died faster than a 20 GB one.
    # Phase 5 solved that with --test-chunk; Phase 3 has no equivalent and is bounded to the
    # small datasets it was built for. GunPoint, its only archived comparison, is 150 test rows.
    return (
        "LOAD anofox_tabfm;\n"
        "SET anofox_tabfm_accept_hf_license = true;\n"
        f"SET anofox_tabfm_max_features = {max(max_features * 2, 1000)};\n"
        f"SET threads = {threads};\n"
        f"SET memory_limit = '{default_memory_limit()}';\n"
    )


def write_group_parquet(
    config: RocketPFNConfig, dataset: str, outdir: Path
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute ROCKET features per group and write train/test Parquet for each.

    Returns (metadata, y_train, y_test). Feature computation is the Phase 1 oracle's, unchanged
    -- Phase 3 is about what happens *after* the features exist.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    x_train, y_train = load(dataset, "train")
    x_test, y_test = load(dataset, "test")
    if config.normalize:
        x_train, x_test = normalize_series(x_train), normalize_series(x_test)
    n_timepoints = x_train.shape[1]

    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.parquet"):
        stale.unlink()

    for g in range(config.n_groups):
        kernels = generate_kernels(
            config.seed,
            n_timepoints,
            config.kernels_per_group,
            first_kernel=g * config.kernels_per_group,
        )
        for split, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
            features = transform(x, kernels)
            names = [f"f{j}" for j in range(features.shape[1])]
            columns = {"id": pa.array(np.arange(features.shape[0]), type=pa.int64())}
            # The target is written for train only. tabfm_classify unions train and test BY
            # NAME internally, so a target column present in both is a binder error whose
            # message names neither the union nor the target (Phase 2).
            if split == "train":
                columns["y"] = pa.array([str(v) for v in y])
            for j, name in enumerate(names):
                columns[name] = pa.array(features[:, j], type=pa.float64())
            pq.write_table(pa.table(columns), outdir / f"{split}_g{g}.parquet")

    return (
        {
            "dataset": dataset,
            "n_groups": config.n_groups,
            "kernels_per_group": config.kernels_per_group,
            "features_per_group": config.features_per_group,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "n_timepoints": int(n_timepoints),
        },
        y_train,
        y_test,
    )


def build_sql(config: RocketPFNConfig, outdir: Path) -> str:
    """Generate the whole composition as one SQL script.

    Written to a file rather than passed with `duckdb -c`: a 500-column feature list already
    exceeds the Windows 32,767-character command-line limit, and that failure reports itself as
    "[WinError 206] The filename or extension is too long" (Phase 2).
    """
    names = [f"f{j}" for j in range(config.features_per_group)]
    feature_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    select_features = ", ".join(names)
    parts = [preamble(config.features_per_group)]

    for g in range(config.n_groups):
        train = (outdir / f"train_g{g}.parquet").as_posix()
        test = (outdir / f"test_g{g}.parquet").as_posix()
        parts.append(f"""
CREATE OR REPLACE VIEW train_g{g} AS
    SELECT y, {select_features} FROM read_parquet('{train}');
CREATE OR REPLACE VIEW test_src_g{g} AS
    SELECT id, {select_features} FROM read_parquet('{test}');
CREATE OR REPLACE VIEW test_g{g} AS
    SELECT {select_features} FROM read_parquet('{test}');

-- Rows are recovered by joining on an echoed feature value, not by output position.
-- `f0` is a ROCKET global-max over continuous data, so it is unique in practice; the
-- assertion below requires the join to be total, which is what makes that safe to rely on.
CREATE OR REPLACE TABLE scored_g{g} AS
SELECT s.id, {g} AS grp, c.proba, row_number() OVER () AS out_pos
FROM tabfm_classify('train_g{g}', 'y', test := 'test_g{g}',
                    model := '{MODEL}', features := {feature_list}) c
JOIN test_src_g{g} s ON s.f0 = c.f0;
""")

    union = "\n  UNION ALL\n".join(
        f"  SELECT id, grp, proba FROM scored_g{g}" for g in range(config.n_groups)
    )
    parts.append(f"""
CREATE OR REPLACE TABLE all_groups AS
{union};

-- The paper's ensembling: a plain mean of class probabilities across the G groups. Keyed on
-- the class label out of `proba`'s map, never on position -- and never on `yhat_score`, which
-- is confidence in whichever class was predicted rather than P(class), and which swan measured
-- producing accuracy 1.0 alongside sub-chance AUROC.
CREATE OR REPLACE TABLE per_class AS
SELECT id, e.key AS cls, avg(e.value) AS mean_p, count(*) AS n_groups_seen
FROM all_groups, UNNEST(map_entries(proba)) AS t(e)
GROUP BY id, e.key;

CREATE OR REPLACE TABLE predictions AS
SELECT id, arg_max(cls, mean_p) AS yhat, max(mean_p) AS confidence
FROM per_class GROUP BY id;

.mode json
.once '{(outdir / "predictions.json").as_posix()}'
SELECT id, yhat, confidence FROM predictions ORDER BY id;

.once '{(outdir / "per_class.json").as_posix()}'
SELECT id, cls, mean_p, n_groups_seen FROM per_class ORDER BY id, cls;

-- Alignment facts go to a file and are checked in Python. Printing them to stdout would make
-- them something to read rather than something that fails.
.once '{(outdir / "assertions.json").as_posix()}'
SELECT
    (SELECT count(*) FROM predictions)          AS predicted_rows,
    (SELECT count(DISTINCT id) FROM all_groups) AS distinct_ids,
    (SELECT count(*) FROM all_groups)           AS group_rows,
    (SELECT min(n_groups_seen) FROM per_class)  AS min_groups_per_row,
    (SELECT max(n_groups_seen) FROM per_class)  AS max_groups_per_row,
    -- Phase 2 saw output order match input order at 40 rows on one thread. Recorded at UCR
    -- scale here as an observation, NOT depended on: the join above is on feature value.
    (SELECT count(*) FILTER (WHERE id != out_pos - 1) FROM scored_g0) AS positional_mismatches;
""")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--n-groups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-sql", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "phase3.json")
    args = parser.parse_args()

    config = RocketPFNConfig(
        num_kernels=args.num_kernels, n_groups=args.n_groups, seed=args.seed, n_estimators=1
    )
    config.validate()
    print(
        f"config: {config.n_groups} groups x {config.kernels_per_group} kernels "
        f"= {config.features_per_group} features/group; "
        f"anofox_reachable={config.anofox_reachable}"
    )

    workdir = ROOT / "data" / "phase3" / args.dataset
    print(f"\n[1/4] computing ROCKET features -> {workdir}", flush=True)
    started = time.perf_counter()
    meta, _, y_test = write_group_parquet(config, args.dataset, workdir)
    feature_seconds = time.perf_counter() - started
    print(f"      {meta['n_train']} train / {meta['n_test']} test, {feature_seconds:.1f}s")

    print("[2/4] generating SQL", flush=True)
    sql = build_sql(config, workdir)
    script = workdir / "compose.sql"
    script.write_text(sql, encoding="utf-8")
    print(f"      {len(sql):,} characters -> {script}")

    print(f"[3/4] running {config.n_groups} classify calls in DuckDB", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(
        [str(DUCKDB), "-f", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sql_seconds = time.perf_counter() - started
    stderr = "\n".join(
        ln
        for ln in (proc.stderr or "").splitlines()
        if not ln.startswith("Schema error: Trying to register schema") and ln.strip()
    )
    print(proc.stdout)
    if stderr:
        print("STDERR:", stderr[:2000], file=sys.stderr)
    if proc.returncode != 0:
        print(f"DuckDB failed after {sql_seconds:.1f}s", file=sys.stderr)
        return 1
    print(f"      {sql_seconds:.1f}s")

    print("[4/4] verifying", flush=True)
    predictions = json.loads((workdir / "predictions.json").read_text(encoding="utf-8"))
    per_class = json.loads((workdir / "per_class.json").read_text(encoding="utf-8"))

    facts = json.loads((workdir / "assertions.json").read_text(encoding="utf-8"))[0]
    n_test = len(y_test)
    failures = []
    if facts["predicted_rows"] != n_test:
        failures.append(f"{facts['predicted_rows']} predictions for {n_test} test rows")
    if facts["distinct_ids"] != n_test:
        failures.append(f"{facts['distinct_ids']} distinct ids joined, expected {n_test}")
    if facts["group_rows"] != n_test * config.n_groups:
        failures.append(
            f"{facts['group_rows']} group rows, expected {n_test * config.n_groups} "
            f"-- the feature-value join was not total, so rows were dropped or duplicated"
        )
    if facts["min_groups_per_row"] != config.n_groups:
        failures.append(
            f"some row was scored by only {facts['min_groups_per_row']} of "
            f"{config.n_groups} groups; averaging over a partial ensemble is silently wrong"
        )
    if facts["max_groups_per_row"] != config.n_groups:
        failures.append(f"some row was scored {facts['max_groups_per_row']} times")

    print(
        f"  row alignment: {facts['distinct_ids']}/{n_test} ids, "
        f"{facts['group_rows']} group-rows, "
        f"{facts['min_groups_per_row']}-{facts['max_groups_per_row']} groups per row"
    )
    print(
        f"  output order matched input order: "
        f"{facts['positional_mismatches'] == 0} "
        f"({facts['positional_mismatches']} mismatches, observed not relied upon)"
    )
    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)

    by_id = {int(r["id"]): r for r in predictions}

    y_pred = np.array([by_id[i]["yhat"] for i in range(len(y_test))])
    accuracy = float((y_pred == np.asarray([str(v) for v in y_test])).mean())

    # Plumbing conformance: redo SQL's averaging and argmax in numpy over the same per-class
    # means and require identical labels. This is the part of Phase 3 that must be exact -- it
    # tests the composition rather than the backbone.
    classes = sorted({r["cls"] for r in per_class})
    matrix = np.zeros((len(y_test), len(classes)))
    for row in per_class:
        matrix[int(row["id"]), classes.index(row["cls"])] = float(row["mean_p"])
    numpy_labels = np.array([classes[i] for i in matrix.argmax(axis=1)])
    plumbing_matches = int((numpy_labels == y_pred).sum())
    rows_sum_to_one = bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6))

    print(f"\n  accuracy ({MODEL}, e=1, G={config.n_groups}): {accuracy:.4f}")
    print(f"  SQL argmax == numpy argmax: {plumbing_matches}/{len(y_test)}")
    print(f"  averaged probabilities sum to 1: {rows_sum_to_one}")

    report = {
        "dataset": args.dataset,
        "model": MODEL,
        "config": {
            "num_kernels": config.num_kernels,
            "n_groups": config.n_groups,
            "features_per_group": config.features_per_group,
            "n_estimators": config.n_estimators,
            "seed": config.seed,
            "anofox_reachable": config.anofox_reachable,
        },
        "shape": meta,
        "accuracy": accuracy,
        "plumbing_conformance": {
            "sql_matches_numpy": plumbing_matches,
            "of": len(y_test),
            "exact": plumbing_matches == len(y_test),
            "probabilities_sum_to_one": rows_sum_to_one,
        },
        "row_alignment": facts,
        "failures": failures,
        "timing": {
            "feature_seconds": round(feature_seconds, 1),
            "sql_seconds": round(sql_seconds, 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if not args.keep_sql:
        script.unlink(missing_ok=True)

    return 0 if plumbing_matches == len(y_test) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
