"""Phase 4's other exit criterion: does the extension beat Python on throughput?

Conformance says the C++ is *right*; this says whether it was worth writing. Both halves are
needed, and the plan is explicit that the answer decides whether Phase 4 was justified at all.

    uv run python scripts/benchmark_transform.py
    uv run python scripts/benchmark_transform.py --kernels 1000 --series 200

Three implementations are timed on identical inputs:

  * **python**  — the Phase 1 oracle, numpy-vectorised over series
  * **cpp**     — `rocket_transform` in the built extension
  * **sql**     — the pure-SQL macro, if `--with-sql` is passed. Off by default because at any
                  size worth benchmarking it does not finish; `sql_rocket_check.py` measured
                  ~4e5x slower than Python on 8 kernels, which is the number that retired the
                  "maybe we can stop at pure SQL" branch.

**These are local Windows timings and belong in no table.** PLAN.md is explicit that the RTX
3060 box is for correctness only and every reported number comes from a pod -- and on
Windows/WDDM a memory problem shows up as a silent 6x slowdown rather than an OOM, so local
timings mislead even directionally. What this script supports is the comparative claim
"C++ beats Python by roughly X on the same machine", which is what Phase 4 actually needs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.rocket import generate_kernels, transform  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


def time_python(series: np.ndarray, kernels_per_group: int, seed: int, first_kernel: int,
                repeats: int) -> tuple[float, np.ndarray]:
    best = float("inf")
    features = None
    for _ in range(repeats):
        started = time.perf_counter()
        kernels = generate_kernels(
            seed, series.shape[1], kernels_per_group, first_kernel=first_kernel
        )
        features = transform(series, kernels)
        best = min(best, time.perf_counter() - started)
    return best, features


def time_cpp(duckdb_path: Path, series: np.ndarray, kernels_per_group: int, seed: int,
             first_kernel: int, repeats: int) -> tuple[float, np.ndarray]:
    """Time the extension, excluding process startup and Parquet read.

    The series go in as a Parquet file rather than as SQL literals: at benchmark sizes the
    literal form is megabytes of text, and parsing it would be timed as if it were transform
    work. The query is run once to warm up, then timed inside DuckDB itself, so the number
    reflects the function rather than the harness.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        data_path = tmpdir / "series.parquet"
        pq.write_table(
            pa.table(
                {
                    "series_id": pa.array(np.arange(series.shape[0]), type=pa.int64()),
                    "values": pa.array(list(series), type=pa.list_(pa.float64())),
                }
            ),
            data_path,
        )
        out_path = (tmpdir / "features.json").as_posix()
        timing_path = (tmpdir / "timing.json").as_posix()

        call = (
            f"rocket_transform(values, {kernels_per_group}, {seed}, {first_kernel})"
        )
        # `.timer on` reports DuckDB's own execution time; the repeated CREATE TABLE runs are
        # what we actually measure, via explicit epoch stamps around them.
        runs = "\n".join(
            f"CREATE OR REPLACE TABLE bench_{i} AS "
            f"SELECT series_id, {call} AS features FROM read_parquet('{data_path.as_posix()}');"
            for i in range(repeats)
        )
        sql = f"""
CREATE OR REPLACE TABLE warmup AS
    SELECT series_id, {call} AS features
    FROM read_parquet('{data_path.as_posix()}');

CREATE OR REPLACE TABLE t0 AS SELECT epoch_ns(current_timestamp) AS ns;
{runs}
CREATE OR REPLACE TABLE t1 AS SELECT epoch_ns(current_timestamp) AS ns;

.mode json
.once '{timing_path}'
SELECT ((SELECT ns FROM t1) - (SELECT ns FROM t0)) / 1e9 / {repeats} AS seconds;

.once '{out_path}'
SELECT series_id, features FROM bench_0 ORDER BY series_id;
"""
        with tempfile.NamedTemporaryFile(
            "w", suffix=".sql", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(sql)
            script = fh.name
        try:
            proc = subprocess.run(
                [str(duckdb_path), "-f", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
            )
        finally:
            Path(script).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"duckdb failed: {(proc.stderr or '')[:2000]}")

        seconds = float(json.loads(Path(timing_path).read_text(encoding="utf-8"))[0]["seconds"])
        payload = json.loads(Path(out_path).read_text(encoding="utf-8"))

    features = np.asarray([r["features"] for r in payload], dtype=np.float64)
    return seconds, features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=ROOT / "build" / "release" / "duckdb.exe")
    parser.add_argument("--kernels", type=int, default=250,
                        help="kernels per group; 250 is this project's group size")
    parser.add_argument("--series", type=int, default=200)
    parser.add_argument("--timepoints", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-kernel", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "benchmark.json")
    args = parser.parse_args()

    if not args.duckdb.exists():
        print(f"no such shell: {args.duckdb}", file=sys.stderr)
        return 1

    rng = np.random.RandomState(7)
    series = rng.randn(args.series, args.timepoints)

    print(
        f"config: {args.series} series x {args.timepoints} timepoints, "
        f"{args.kernels} kernels (= {args.kernels * 2} features), "
        f"best of {args.repeats}\n"
    )

    python_seconds, python_features = time_python(
        series, args.kernels, args.seed, args.first_kernel, args.repeats
    )
    print(f"  python : {python_seconds:8.3f}s")

    cpp_seconds, cpp_features = time_cpp(
        args.duckdb, series, args.kernels, args.seed, args.first_kernel, args.repeats
    )
    print(f"  cpp    : {cpp_seconds:8.3f}s")

    # A benchmark that does not check its own output can report a wonderful time for the wrong
    # answer, so the two feature matrices are compared here as well.
    max_diff = float(np.abs(cpp_features - python_features).max())
    agree = max_diff < 1e-9
    speedup = python_seconds / cpp_seconds if cpp_seconds > 0 else float("inf")

    print(f"\n  speedup: {speedup:.1f}x")
    print(f"  outputs agree: {agree} (max abs diff {max_diff:.3e})")

    report = {
        "config": {
            "series": args.series,
            "timepoints": args.timepoints,
            "kernels_per_group": args.kernels,
            "features": args.kernels * 2,
            "seed": args.seed,
            "repeats": args.repeats,
        },
        "python_seconds": round(python_seconds, 4),
        "cpp_seconds": round(cpp_seconds, 4),
        "speedup": round(speedup, 1),
        "outputs_agree": agree,
        "max_abs_diff": max_diff,
        "caveat": "local Windows timings, correctness-oriented hardware; PLAN.md requires "
                  "reported numbers to come from a pod",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
