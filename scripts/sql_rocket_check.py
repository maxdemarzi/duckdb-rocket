"""Check `sql/rocket.sql` against the Phase 1 oracle, and measure what it costs.

PLAN.md Phase 4 opens with a pure-SQL macro and one decision attached to it: *"Expect it to be
too slow -- but ... If it is merely 5-10x slow rather than 1000x, seriously consider stopping
here."* This script produces the number that decision needs, and checks the SQL is right first,
because a fast wrong answer settles nothing.

    uv run python scripts/sql_rocket_check.py                    # correctness + timing
    uv run python scripts/sql_rocket_check.py --kernels 16 --series 4

Correctness is checked against `duckdb_rocket/`, not against the golden vectors, so that a
mismatch points at this file rather than at a stale fixture.
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
DUCKDB = ROOT / "tools" / "duckdb.exe"
ROCKET_SQL = (ROOT / "sql" / "rocket.sql").as_posix()


def run_sql(sql: str, timeout: int = 3600) -> tuple[bool, str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False, encoding="utf-8") as fh:
        fh.write(sql)
        script = fh.name
    try:
        proc = subprocess.run(
            [str(DUCKDB), "-f", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    finally:
        Path(script).unlink(missing_ok=True)
    return proc.returncode == 0, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def split_seed(seed: int) -> tuple[int, int]:
    """The SQL side carries a u64 as two 32-bit halves; see sql/rocket.sql."""
    return (seed >> 32) & 0xFFFFFFFF, seed & 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=int, default=16)
    parser.add_argument("--first-kernel", type=int, default=0)
    parser.add_argument("--series", type=int, default=4)
    parser.add_argument("--timepoints", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "sql_rocket_check.json")
    args = parser.parse_args()

    rng = np.random.RandomState(12345)
    series = rng.randn(args.series, args.timepoints)

    print(
        f"config: {args.kernels} kernels from index {args.first_kernel}, "
        f"{args.series} series x {args.timepoints} timepoints, seed {args.seed}"
    )

    # --- the oracle ---------------------------------------------------------------------
    started = time.perf_counter()
    kernels = generate_kernels(
        args.seed, args.timepoints, args.kernels, first_kernel=args.first_kernel
    )
    expected = transform(series, kernels)
    python_seconds = time.perf_counter() - started
    print(f"\npython: {expected.shape} in {python_seconds:.3f}s")

    # --- the SQL path -------------------------------------------------------------------
    hi, lo = split_seed(args.seed)
    # repr() of a numpy scalar is `np.float64(...)`, which DuckDB parses as a function call.
    # `float(...)` first, and `.17g` so the literal round-trips to the same double the oracle
    # used -- a shortened literal would show up as a spurious conformance failure.
    values = ",\n    ".join(
        f"({i}, {t}, {float(series[i, t]):.17g})"
        for i in range(args.series)
        for t in range(args.timepoints)
    )
    sql = f"""
.read {ROCKET_SQL}

CREATE OR REPLACE TABLE series(series_id BIGINT, t BIGINT, value DOUBLE);
INSERT INTO series VALUES
    {values};

-- Guard the fixed draw grid rather than trusting it: too few accepted pairs would silently
-- produce short weight vectors instead of an error.
CREATE OR REPLACE TABLE shortfall AS
SELECT count(*) AS n FROM (
    SELECT kernel_id, count(*) AS got
    FROM rocket_kernels({hi}, {lo}, {args.first_kernel}, {args.kernels}, {args.timepoints})
    GROUP BY kernel_id
) WHERE got NOT IN (7, 9, 11);

.mode json
.once '{{OUT}}'
SELECT series_id, kernel_id, max_feature, ppv_feature
FROM rocket_features({hi}, {lo}, {args.first_kernel}, {args.kernels},
                     {args.timepoints}, series)
ORDER BY series_id, kernel_id;

.once '{{SHORT}}'
SELECT n FROM shortfall;
"""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = (Path(tmp) / "features.json").as_posix()
        short_path = (Path(tmp) / "shortfall.json").as_posix()
        started = time.perf_counter()
        ok, stdout, stderr = run_sql(sql.replace("{OUT}", out_path).replace("{SHORT}", short_path))
        sql_seconds = time.perf_counter() - started

        if not ok:
            print(f"\nSQL FAILED after {sql_seconds:.1f}s", file=sys.stderr)
            print(stderr[:3000], file=sys.stderr)
            return 1

        rows = json.loads(Path(out_path).read_text(encoding="utf-8"))
        shortfall = json.loads(Path(short_path).read_text(encoding="utf-8"))[0]["n"]

    print(f"sql:    {len(rows)} rows in {sql_seconds:.3f}s")
    if shortfall:
        print(f"  FAIL: {shortfall} kernels had an unexpected weight count", file=sys.stderr)

    # --- compare ------------------------------------------------------------------------
    got = np.zeros_like(expected)
    for row in rows:
        s, k = int(row["series_id"]), int(row["kernel_id"])
        got[s, 2 * k] = float(row["max_feature"])
        got[s, 2 * k + 1] = float(row["ppv_feature"])

    diff = np.abs(got - expected)
    max_diff = float(diff.max()) if diff.size else 0.0
    mismatches = int((diff > args.tolerance).sum())

    print(f"\n  max absolute difference: {max_diff:.3e}  (tolerance {args.tolerance:.0e})")
    print(f"  values outside tolerance: {mismatches} / {expected.size}")

    # PPV is a ratio of small integers and must be exact; a mismatch there means the
    # convolution disagreed about a sign, not about the last ulp.
    ppv_diff = float(np.abs(got[:, 1::2] - expected[:, 1::2]).max()) if expected.size else 0.0
    print(f"  max PPV difference:      {ppv_diff:.3e}  (should be 0)")

    slowdown = sql_seconds / python_seconds if python_seconds > 0 else float("inf")
    print(f"\n  SQL is {slowdown:.1f}x slower than Python at this size")

    report = {
        "config": {
            "kernels": args.kernels,
            "first_kernel": args.first_kernel,
            "series": args.series,
            "timepoints": args.timepoints,
            "seed": args.seed,
        },
        "correct": mismatches == 0 and shortfall == 0,
        "max_abs_diff": max_diff,
        "max_ppv_diff": ppv_diff,
        "values_outside_tolerance": mismatches,
        "kernels_with_bad_weight_count": shortfall,
        "timing": {
            "python_seconds": round(python_seconds, 4),
            "sql_seconds": round(sql_seconds, 4),
            "slowdown": round(slowdown, 1),
            "note": "one DuckDB process launch is included in sql_seconds; at this size that "
            "is a meaningful share of it, so the ratio flatters Python at small inputs "
            "and is best read at the largest size that finishes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if report["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
