"""Does the C++ multivariate overload match the Python oracle? (SPEC.md 7)

Separate from `conformance.py` because it checks a different claim. That one compares against
committed golden vectors, which are univariate and must never change. This one compares the two
implementations directly across a range of channel counts, since the multivariate fixtures do
not exist yet and freezing them before the spec has been exercised would freeze in any mistake.

    uv run python scripts/conformance_multivariate.py
    uv run python scripts/conformance_multivariate.py --channels 2,3,6,12

The most valuable case is `--channels 1`. SPEC.md 7.1 promises that a single-channel series
produces byte-identical kernels whether it goes through the univariate or the multivariate path,
and that promise is what keeps every committed golden vector valid. It is checked here on both
sides of the language boundary.
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
from duckdb_rocket.shells import built_shell  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


def literal(values: np.ndarray) -> str:
    """`.17g` so every literal round-trips to the exact double the oracle used."""
    return "[" + ", ".join(f"{float(v):.17g}" for v in values) + "]"


def run_extension(shell: Path, x: np.ndarray, num_kernels: int, seed: int,
                  first_kernel: int, timeout: int = 1800) -> np.ndarray:
    """Call the multivariate overload once per series. `x` is (n_series, n_channels, n)."""
    rows = []
    for i, series in enumerate(x):
        channels = ", ".join(literal(channel) for channel in series)
        rows.append(
            f"SELECT {i} AS series_id, rocket_transform([{channels}]::DOUBLE[][], "
            f"{num_kernels}, {seed}, {first_kernel}) AS features"
        )
    query = "\nUNION ALL\n".join(rows)

    with tempfile.TemporaryDirectory() as tmp:
        out = (Path(tmp) / "features.json").as_posix()
        sql = (f".mode json\n.once '{out}'\n"
               f"SELECT series_id, features FROM ({query}) ORDER BY series_id;\n")
        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(sql)
            script = fh.name
        try:
            proc = subprocess.run([str(shell), "-f", script], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
        finally:
            Path(script).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"duckdb failed: {(proc.stderr or '')[:2000]}")
        payload = json.loads(Path(out).read_text(encoding="utf-8"))

    return np.asarray([r["features"] for r in payload], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=built_shell())
    parser.add_argument("--channels", default="1,2,3,6,12")
    parser.add_argument("--kernels", type=int, default=32)
    parser.add_argument("--series", type=int, default=4)
    parser.add_argument("--timepoints", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--first-kernel", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reference" / "conformance_multivariate.json")
    args = parser.parse_args()

    if not args.duckdb.exists():
        print(f"no such shell: {args.duckdb}", file=sys.stderr)
        return 1

    channel_counts = [int(c) for c in args.channels.split(",") if c.strip()]
    rng = np.random.RandomState(4242)
    results = []
    started = time.perf_counter()

    for n_channels in channel_counts:
        x = rng.randn(args.series, n_channels, args.timepoints)
        bank = generate_kernels(args.seed, args.timepoints, args.kernels,
                                first_kernel=args.first_kernel, n_channels=n_channels)
        expected = transform(x, bank)
        got = run_extension(args.duckdb, x, args.kernels, args.seed, args.first_kernel)

        if got.shape != expected.shape:
            print(f"  C={n_channels}: FAIL — shape {got.shape}, expected {expected.shape}",
                  file=sys.stderr)
            results.append({"channels": n_channels, "ok": False, "reason": "shape mismatch"})
            continue

        diff = np.abs(got - expected)
        max_diff = float(diff.max())
        outside = int((diff > args.tolerance).sum())
        ppv_diff = float(np.abs(got[:, 1::2] - expected[:, 1::2]).max())
        ok = outside == 0 and ppv_diff == 0.0

        # SPEC.md 7.1, checked across the language boundary: a single-channel series must give
        # the same features through the multivariate overload as through the univariate one.
        matches_univariate = None
        if n_channels == 1:
            uni_bank = generate_kernels(args.seed, args.timepoints, args.kernels,
                                        first_kernel=args.first_kernel)
            uni = transform(x[:, 0, :], uni_bank)
            matches_univariate = bool(np.array_equal(uni, expected))
            ok = ok and matches_univariate

        print(f"  C={n_channels}: {'OK' if ok else 'FAIL'}  max abs diff {max_diff:.3e}, "
              f"PPV diff {ppv_diff:.3e}, outside tolerance {outside}/{expected.size}"
              + ("" if matches_univariate is None
                 else f", identical to the univariate path: {matches_univariate}"))

        results.append({
            "channels": n_channels,
            "ok": ok,
            "max_abs_diff": max_diff,
            "max_ppv_diff": ppv_diff,
            "values_outside_tolerance": outside,
            "matches_univariate_path": matches_univariate,
        })

    elapsed = time.perf_counter() - started
    passed = all(r["ok"] for r in results)
    print(f"\n{'ALL CHANNEL COUNTS PASS' if passed else 'MULTIVARIATE CONFORMANCE FAILED'} "
          f"({elapsed:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "duckdb": str(args.duckdb),
        "seed": args.seed,
        "kernels": args.kernels,
        "tolerance": args.tolerance,
        "passed": passed,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
