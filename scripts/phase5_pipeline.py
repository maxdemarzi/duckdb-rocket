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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import normalize_series  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SHELL = ROOT / "build" / "release" / "duckdb.exe"
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
    values = list(x_train) + list(x_test)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "split": pa.array(splits),
                "label": pa.array(labels),
                "values": pa.array(values, type=pa.list_(pa.float64())),
            }
        ),
        path,
    )
    return (
        {
            "dataset": dataset,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_timepoints": int(x_train.shape[1]),
            "raw_parquet": path.as_posix(),
        },
        np.asarray([str(v) for v in y_test]),
    )


def build_sql(config: RocketPFNConfig, meta: dict, outdir: Path) -> str:
    n_features = config.features_per_group
    names = [f"f{j}" for j in range(n_features)]
    feature_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    # DuckDB lists are 1-based, so feature j lives at f[j + 1].
    projection = ", ".join(f"f[{j + 1}] AS f{j}" for j in range(n_features))
    select_features = ", ".join(names)

    parts = [
        "LOAD anofox_tabfm;",
        "SET anofox_tabfm_accept_hf_license = true;",
        f"SET anofox_tabfm_max_features = {max(n_features * 2, 1000)};",
        f"CREATE OR REPLACE TABLE raw AS "
        f"SELECT * FROM read_parquet('{meta['raw_parquet']}');",
    ]

    for g in range(config.n_groups):
        first_kernel = g * config.kernels_per_group
        parts.append(f"""
-- Group {g}: global kernel indices [{first_kernel}, {first_kernel + config.kernels_per_group}).
-- Materialised as a TABLE, not a VIEW: the train and test projections below each reference it,
-- and a view would recompute the whole transform per reference.
CREATE OR REPLACE TABLE feat_g{g} AS
SELECT id, split, label,
       rocket_transform(values, {config.kernels_per_group}, {config.seed}, {first_kernel}) AS f
FROM raw;

CREATE OR REPLACE VIEW train_g{g} AS
    SELECT label AS y, {projection} FROM feat_g{g} WHERE split = 'train';
-- The test view omits the target: tabfm_classify unions train and test BY NAME, and a target
-- present in both is a duplicate-name binder error naming neither cause (Phase 2).
CREATE OR REPLACE VIEW test_g{g} AS
    SELECT {projection} FROM feat_g{g} WHERE split = 'test';
CREATE OR REPLACE VIEW test_src_g{g} AS
    SELECT id, {projection} FROM feat_g{g} WHERE split = 'test';

CREATE OR REPLACE TABLE scored_g{g} AS
SELECT s.id, {g} AS grp, c.proba
FROM tabfm_classify('train_g{g}', 'y', test := 'test_g{g}',
                    model := '{MODEL}', features := {feature_list}) c
JOIN test_src_g{g} s ON s.f0 = c.f0;

DROP TABLE feat_g{g};
""")

    union = "\n  UNION ALL\n".join(
        f"  SELECT id, grp, proba FROM scored_g{g}" for g in range(config.n_groups)
    )
    parts.append(f"""
CREATE OR REPLACE TABLE all_groups AS
{union};

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

.once '{(outdir / "assertions.json").as_posix()}'
SELECT (SELECT count(*) FROM predictions)          AS predicted_rows,
       (SELECT count(DISTINCT id) FROM all_groups) AS distinct_ids,
       (SELECT count(*) FROM all_groups)           AS group_rows,
       (SELECT min(n_groups_seen) FROM per_class)  AS min_groups_per_row,
       (SELECT max(n_groups_seen) FROM per_class)  AS max_groups_per_row;
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

    sql = build_sql(config, meta, workdir)
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
    stderr = "\n".join(
        ln for ln in (proc.stderr or "").splitlines()
        if not ln.startswith("Schema error: Trying to register schema") and ln.strip()
    )
    if proc.returncode != 0:
        print(f"FAILED after {seconds:.1f}s", file=sys.stderr)
        print(stderr[:3000], file=sys.stderr)
        return 1
    print(f"      {seconds:.1f}s")

    predictions = json.loads((workdir / "predictions.json").read_text(encoding="utf-8"))
    facts = json.loads((workdir / "assertions.json").read_text(encoding="utf-8"))[0]

    n_test = len(y_test)
    failures = []
    if facts["predicted_rows"] != n_test:
        failures.append(f"{facts['predicted_rows']} predictions for {n_test} test rows")
    if facts["group_rows"] != n_test * config.n_groups:
        failures.append(
            f"{facts['group_rows']} group rows, expected {n_test * config.n_groups}: the "
            f"feature-value join dropped or duplicated rows"
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
    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)

    # The end-to-end statement: C++ features must be interchangeable with the oracle's.
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
        },
        "shape": meta,
        "accuracy": accuracy,
        "row_alignment": facts,
        "failures": failures,
        "seconds": round(seconds, 1),
        "comparison": comparison,
        "caveat": "local Windows timing on a contended box; PLAN.md requires reported numbers "
                  "to come from a pod",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if not args.keep_sql:
        script.unlink(missing_ok=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
