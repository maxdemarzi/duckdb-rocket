"""The breadth sweep: every runnable UCR dataset in the subset, several seeds, one report.

This is what the pod is for. Locally the subset had to be cut to five small datasets and three
seeds, which left the noise floor resting on a single dataset (Beef) — every other dataset
saturated and reproduced exactly, so the subset could not resolve anything.

    uv run python scripts/pod/sweep.py                       # all runnable datasets, 3 seeds
    uv run python scripts/pod/sweep.py --seeds 5 --datasets Beef,OSULeaf

Each (dataset, seed) is a separate `phase5_pipeline.py` process. That is deliberate: ONNX
Runtime's API is process-global, a crash in one combination must not take the sweep with it, and
a pod that dies three hours in should leave behind every result it had already earned. Results
are written incrementally for the same reason.

**Every accuracy number here is `tabicl-v2`**, because `tabpfn-v2-5` does not load in the
`anofox_tabfm` build the community repository serves (fixed upstream in `v2026.08.11`, not yet
propagated). When it lands, re-run this with `--model tabpfn-v2-5` for the paper's actual model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from duckdb_rocket.datasets import RUNNABLE_SUBSET, UCR_SUBSET, describe  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", help="comma-separated; default is every runnable one")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--n-groups", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "pod_sweep.json")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per run")
    args = parser.parse_args()

    if args.datasets:
        wanted = {n.strip() for n in args.datasets.split(",")}
        specs = [d for d in UCR_SUBSET if d.name in wanted]
        missing = wanted - {d.name for d in specs}
        if missing:
            parser.error(f"unknown dataset(s): {', '.join(sorted(missing))}")
        skipped = [d for d in specs if not d.runnable]
        specs = [d for d in specs if d.runnable]
    else:
        specs = list(RUNNABLE_SUBSET)
        skipped = [d for d in UCR_SUBSET if not d.runnable]

    for spec in skipped:
        print(f"SKIP  {describe(spec)}", file=sys.stderr)

    runs: list[dict] = []
    started_all = time.perf_counter()

    for spec in specs:
        print(f"\n=== {describe(spec)}", flush=True)
        for seed in range(args.seeds):
            out = ROOT / "reference" / "pod" / f"{spec.name}_seed{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
                "--dataset", spec.name,
                "--num-kernels", str(args.num_kernels),
                "--n-groups", str(args.n_groups),
                "--seed", str(seed),
                "--out", str(out),
            ]
            started = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  seed={seed}  TIMEOUT after {args.timeout}s", flush=True)
                runs.append({"dataset": spec.name, "seed": seed, "error": "timeout"})
                continue
            elapsed = time.perf_counter() - started

            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or proc.stdout or "")[-400:]
                print(f"  seed={seed}  FAILED ({elapsed:.0f}s): {tail}", flush=True)
                runs.append({"dataset": spec.name, "seed": seed, "error": tail})
                continue

            report = json.loads(out.read_text(encoding="utf-8"))
            runs.append(
                {
                    "dataset": spec.name,
                    "seed": seed,
                    "accuracy": report["accuracy"],
                    "seconds": report["seconds"],
                    "n_test": report["shape"]["n_test"],
                    "row_alignment_ok": not report["failures"],
                }
            )
            print(
                f"  seed={seed}  acc={report['accuracy']:.4f}  {report['seconds']:.0f}s"
                + ("" if not report["failures"] else f"  ALIGNMENT FAILURES: {report['failures']}"),
                flush=True,
            )

            # Written after every run, not at the end: a pod that dies late should not take the
            # results it already earned with it.
            _write(args.out, specs, runs, args, time.perf_counter() - started_all)

    total = time.perf_counter() - started_all
    _write(args.out, specs, runs, args, total)

    ok = [r for r in runs if "accuracy" in r]
    if ok:
        print(f"\nmean accuracy over {len(ok)} runs: "
              f"{statistics.fmean(r['accuracy'] for r in ok):.4f}")
    print(f"total wall clock: {total / 60:.1f} min")
    print(f"wrote {args.out}")
    return 0 if len(ok) == len(runs) else 1


def _write(path: Path, specs, runs: list[dict], args, elapsed: float) -> None:
    per_dataset = {}
    for spec in specs:
        accs = [r["accuracy"] for r in runs if r["dataset"] == spec.name and "accuracy" in r]
        if accs:
            per_dataset[spec.name] = {
                "mean": statistics.fmean(accs),
                # sd is None rather than 0.0 for a single sample: "variance not measured" and
                # "no variance" get read very differently by whoever opens this file later.
                "sd": statistics.stdev(accs) if len(accs) > 1 else None,
                "n": len(accs),
                "accuracies": accs,
            }
    ok = [r for r in runs if "accuracy" in r]
    sds = [v["sd"] for v in per_dataset.values() if v["sd"] is not None]

    payload = {
        "model": "tabicl-v2",
        "note": "tabpfn-v2-5 does not load in the community anofox_tabfm build (bc6d8af); "
                "fixed upstream in v2026.08.11 but not yet propagated. These are substitute-"
                "backbone numbers.",
        "config": {
            "num_kernels": args.num_kernels,
            "n_groups": args.n_groups,
            "seeds": args.seeds,
        },
        "elapsed_seconds": round(elapsed, 1),
        "per_dataset": per_dataset,
        "noise_floor": {
            "max_per_dataset_sd": max(sds) if sds else None,
            "mean_per_dataset_sd": statistics.fmean(sds) if sds else None,
            "datasets_with_variance": sum(1 for s in sds if s > 0),
            "datasets_measured": len(sds),
        },
        "overall_mean_accuracy": statistics.fmean(r["accuracy"] for r in ok) if ok else None,
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
