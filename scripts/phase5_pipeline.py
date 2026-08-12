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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def binding_memory_bytes() -> tuple[int, str]:
    """The memory this process may actually use, and where that number came from.

    The same trap as the thread count below, one level down. Inside a container `free` and
    `/proc/meminfo` report the *host's* RAM, so DuckDB's default limit -- 80% of what it can see
    -- can land far above the cgroup ceiling. It then allocates happily until the kernel kills it,
    which is indistinguishable from a hang: no DuckDB error, no Python traceback, just a dead
    child. Read the cgroup first and fall back to visible RAM only when there is no cgroup.
    """
    for path, kind in (
        (Path("/sys/fs/cgroup/memory.max"), "cgroup v2"),                      # unified
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), "cgroup v1"),    # legacy
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":  # v2's "no limit"
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 reports a sentinel near 2^63 when unlimited; anything that large is not a limit.
        if 0 < value < (1 << 62):
            return value, kind

    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "visible RAM"

    import ctypes  # Windows: no cgroups, no sysconf

    class _Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _Status()
    status.dwLength = ctypes.sizeof(_Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullTotalPhys), "visible RAM"


def default_memory_limit() -> str:
    """70% of the binding limit, as a DuckDB size string.

    Not 80%: the budget has to cover the Python parent, the ONNX session's own allocations and
    the OS, none of which are inside DuckDB's accounting. ItalyPowerDemand reached 25.7 GB on a
    box with no limit set at all and took the machine down with it.
    """
    total, _ = binding_memory_bytes()
    return f"{max(int(total * 0.70) // (1024 ** 3), 1)}GB"


def build_sql(config: RocketPFNConfig, meta: dict, outdir: Path, threads: int,
              memory_limit: str, temp_dir: Path) -> str:
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
        # Thread count is set explicitly rather than inherited from the visible core count.
        # On a 112-core pod, four concurrent runs each sized their own pool from that number,
        # on top of ONNX's per-session threads, and every run died near completion with no
        # error message at all. A container's visible core count is not its budget, especially
        # when several of these run side by side.
        f"SET threads = {threads};",
        # Same reasoning as the thread count, for memory. Without an explicit limit DuckDB sizes
        # itself against RAM it cannot actually have, and a temp directory is what turns the
        # overflow into a slow query instead of a dead process.
        f"SET memory_limit = '{memory_limit}';",
        f"SET temp_directory = '{temp_dir.as_posix()}';",
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

    sql = build_sql(config, meta, workdir, args.threads, memory_limit, workdir)
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
