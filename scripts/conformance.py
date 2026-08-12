"""Conformance: does the C++ extension reproduce the golden vectors?

This is the test PLAN.md's whole Phase 1 PRNG decision exists to make possible. A portable,
explicitly-specified stream plus fixed golden vectors is what turns "the C++ port looks right"
into something that can actually fail.

    uv run python scripts/conformance.py
    uv run python scripts/conformance.py --duckdb build/release/duckdb.exe

Two fixtures are checked, and the second is the interesting one. `features_offset` starts at
global kernel index 9,000, exercising the property the entire group design rests on: kernel `i`
is a pure function of `(seed, i)`, so group `g` is addressable without generating groups
`0..g-1`. A port that quietly treats `first_kernel` as an offset into a freshly-seeded stream
passes the base fixture and fails this one.

Tolerance is tight but non-zero by design. SPEC.md 4 fixes the accumulation order but
floating-point addition is not associative, and `log`/`pow` are not correctly-rounded across
platforms, so last-ulp drift is expected and bit-identity is not the requirement. PPV is held to
exact equality separately: it is a ratio of small integers, so any difference there means the
two implementations disagreed about the sign of a convolution output, which is a real bug
rather than rounding.
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

from duckdb_rocket import golden  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


def run_extension(duckdb_path: Path, series: np.ndarray, num_kernels: int, seed: int,
                  first_kernel: int, timeout: int = 1800) -> np.ndarray:
    """Call `rocket_transform` once per series and return the feature matrix."""
    rows = []
    for i, row in enumerate(series):
        # .17g so each literal round-trips to the exact double the oracle used; a shortened
        # literal would read as a conformance failure that is really a formatting bug.
        literals = ", ".join(f"{float(v):.17g}" for v in row)
        rows.append(
            f"SELECT {i} AS series_id, "
            f"rocket_transform([{literals}]::DOUBLE[], {num_kernels}, {seed}, {first_kernel}) "
            f"AS features"
        )
    query = "\nUNION ALL\n".join(rows)

    with tempfile.TemporaryDirectory() as tmp:
        out = (Path(tmp) / "features.json").as_posix()
        sql = f".mode json\n.once '{out}'\nSELECT series_id, features FROM ({query}) ORDER BY series_id;\n"
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
                timeout=timeout,
            )
        finally:
            Path(script).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"duckdb failed: {(proc.stderr or '')[:2000]}")
        payload = json.loads(Path(out).read_text(encoding="utf-8"))

    return np.asarray([r["features"] for r in payload], dtype=np.float64)


def check(name: str, expected: np.ndarray, got: np.ndarray, tolerance: float) -> dict:
    if got.shape != expected.shape:
        print(f"  {name}: FAIL — shape {got.shape}, expected {expected.shape}", file=sys.stderr)
        return {"fixture": name, "ok": False, "reason": "shape mismatch",
                "got_shape": list(got.shape), "expected_shape": list(expected.shape)}

    diff = np.abs(got - expected)
    max_diff = float(diff.max())
    outside = int((diff > tolerance).sum())
    # Columns 1, 3, 5, ... are PPV (SPEC.md 5: features are interleaved).
    ppv_diff = float(np.abs(got[:, 1::2] - expected[:, 1::2]).max())
    max_diff_ulps = float((diff / np.maximum(np.abs(expected), 1e-300)).max())

    ok = outside == 0 and ppv_diff == 0.0
    status = "OK" if ok else "FAIL"
    print(f"  {name}: {status}")
    print(f"    shape {got.shape}, max abs diff {max_diff:.3e}, "
          f"max rel diff {max_diff_ulps:.3e}")
    print(f"    values outside tolerance: {outside}/{expected.size}; "
          f"max PPV diff {ppv_diff:.3e} (must be 0)")

    return {
        "fixture": name,
        "ok": ok,
        "shape": list(got.shape),
        "max_abs_diff": max_diff,
        "max_rel_diff": max_diff_ulps,
        "values_outside_tolerance": outside,
        "max_ppv_diff": ppv_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb",
        type=Path,
        default=ROOT / "build" / "release" / "duckdb.exe",
        help="shell with the rocket extension linked in",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "conformance.json")
    args = parser.parse_args()

    if not args.duckdb.exists():
        print(f"no such shell: {args.duckdb}\nBuild it with scripts/build_extension.bat",
              file=sys.stderr)
        return 1

    series = golden.golden_input()
    print(f"golden input: {series.shape}, seed {golden.GOLDEN_INPUT_SEED:#x}")
    print(f"tolerance: {args.tolerance:.0e}\n")

    fixtures = [
        ("features_base", golden.GOLDEN_NUM_KERNELS, golden.GOLDEN_FIRST_KERNEL),
        ("features_offset", golden.GOLDEN_OFFSET_NUM_KERNELS, golden.GOLDEN_OFFSET_FIRST_KERNEL),
    ]

    results = []
    started = time.perf_counter()
    for name, num_kernels, first_kernel in fixtures:
        _, _, expected = golden.build_golden(
            seed=golden.GOLDEN_SEED, num_kernels=num_kernels, first_kernel=first_kernel
        )
        got = run_extension(
            args.duckdb, series, num_kernels, golden.GOLDEN_SEED, first_kernel
        )
        results.append(check(name, expected, got, args.tolerance))
    elapsed = time.perf_counter() - started

    passed = all(r["ok"] for r in results)
    print(f"\n{'ALL FIXTURES PASS' if passed else 'CONFORMANCE FAILED'}  ({elapsed:.1f}s)")

    report = {
        "duckdb": str(args.duckdb),
        "tolerance": args.tolerance,
        "seed": golden.GOLDEN_SEED,
        "passed": passed,
        "fixtures": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
