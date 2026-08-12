"""Accuracy and noise-floor harness for the Phase 1 oracle.

    # smoke test -- small, CPU-friendly, proves the wiring
    uv run python scripts/accuracy.py --smoke

    # the real thing, on a pod
    uv run python scripts/accuracy.py --seeds 5 --out reference/accuracy.json

    # what the SQL path costs: paper e=8 against anofox-reachable e=1, paired
    uv run python scripts/accuracy.py --compare-estimators 8,1 --seeds 5

**Measure the noise floor before believing any comparison.** The tabicl fork found that pairing
-- same seed, same data, one setting changed -- tightened its estimate roughly eightfold, from a
5-point resolution to 0.6. A gap smaller than the floor is not a result. This harness therefore
reports the spread of absolute accuracy AND, when comparing, the spread of *paired* gaps, which
are different numbers and the second is the one that matters.

Every run archives its environment tuple via `doctor.py`, because a number without one is not
attributable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.datasets import RUNNABLE_SUBSET, UCR_SUBSET, describe, load  # noqa: E402
from duckdb_rocket.pipeline import RocketPFN, RocketPFNConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doctor import doctor  # noqa: E402


def run_one(spec, config: RocketPFNConfig) -> dict:
    """Fit and score one dataset under one config."""
    started = time.perf_counter()
    x_train, y_train = load(spec.name, "train")
    x_test, y_test = load(spec.name, "test")
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model = RocketPFN(config).fit(x_train, y_train)
    predictions = model.predict_proba(x_test)
    fit_seconds = time.perf_counter() - started

    labels = predictions.labels
    accuracy = float((labels == y_test).mean())

    # Per-group accuracy is nearly free here and answers "is the ensembling doing anything?"
    # without a second run. If the mean of the per-group accuracies equals the ensembled
    # accuracy, the averaging bought nothing on this dataset.
    per_group = [
        float((predictions.group_labels(g) == y_test).mean())
        for g in range(config.n_groups)
    ]

    return {
        "dataset": spec.name,
        "accuracy": accuracy,
        "per_group_accuracy": per_group,
        "per_group_mean": statistics.fmean(per_group),
        "ensembling_gain": accuracy - statistics.fmean(per_group),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_timepoints": int(x_train.shape[1]),
        "load_seconds": round(load_seconds, 2),
        "fit_predict_seconds": round(fit_seconds, 2),
    }


def summarise(values: list[float]) -> dict:
    """Mean and spread. `sd` is None for a single sample rather than 0.0.

    Reporting 0.0 there would read as "no variance measured" when the truth is "variance not
    measured", and those get treated very differently by whoever reads the file later.
    """
    return {
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", help="comma-separated names; default is the subset")
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds per config")
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compare-estimators",
        help="two comma-separated values, e.g. 8,1 -- runs both and reports paired gaps",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny configuration for wiring checks; NOT a source of reportable numbers",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.smoke:
        args.num_kernels, args.n_groups, args.n_estimators = 200, 2, 1
        if not args.datasets:
            args.datasets = "ItalyPowerDemand,GunPoint"

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

    estimator_values = (
        [int(v) for v in args.compare_estimators.split(",")]
        if args.compare_estimators
        else [args.n_estimators]
    )
    if args.compare_estimators and len(estimator_values) != 2:
        parser.error("--compare-estimators takes exactly two values, e.g. 8,1")

    def make_config(seed: int, n_estimators: int) -> RocketPFNConfig:
        return RocketPFNConfig(
            num_kernels=args.num_kernels,
            n_groups=args.n_groups,
            seed=seed,
            n_estimators=n_estimators,
            device=args.device,
        )

    report = {
        "config": {
            "num_kernels": args.num_kernels,
            "n_groups": args.n_groups,
            "n_estimators": estimator_values,
            "device": args.device,
            "seeds": args.seeds,
            "smoke": args.smoke,
        },
        "environment": doctor(),
        "skipped": [asdict(d) for d in skipped],
        "runs": [],
    }

    if args.smoke:
        print(
            "SMOKE MODE: reduced kernels and e=1. Results are for wiring only and must not "
            "be reported.\n",
            file=sys.stderr,
        )

    for spec in specs:
        print(f"\n=== {describe(spec)}", flush=True)
        for seed in range(args.seeds):
            for n_estimators in estimator_values:
                config = make_config(seed, n_estimators)
                try:
                    result = run_one(spec, config)
                except Exception as exc:  # noqa: BLE001 - one dataset must not kill the sweep
                    print(f"  seed={seed} e={n_estimators}  FAILED: {exc}", flush=True)
                    report["runs"].append(
                        {
                            "dataset": spec.name,
                            "seed": seed,
                            "n_estimators": n_estimators,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                result["seed"] = seed
                result["n_estimators"] = n_estimators
                report["runs"].append(result)
                print(
                    f"  seed={seed} e={n_estimators}  acc={result['accuracy']:.4f}"
                    f"  (groups mean {result['per_group_mean']:.4f},"
                    f" ensembling {result['ensembling_gain']:+.4f})"
                    f"  {result['fit_predict_seconds']:.1f}s",
                    flush=True,
                )

    ok = [r for r in report["runs"] if "accuracy" in r]

    # Per-dataset noise floor: spread of absolute accuracy across seeds at fixed config.
    per_dataset = {}
    for spec in specs:
        for n_estimators in estimator_values:
            values = [
                r["accuracy"]
                for r in ok
                if r["dataset"] == spec.name and r["n_estimators"] == n_estimators
            ]
            if values:
                per_dataset[f"{spec.name}/e={n_estimators}"] = summarise(values)
    report["per_dataset"] = per_dataset

    if estimator_values and len(estimator_values) == 2:
        # Paired gaps: same dataset, same seed, only n_estimators differs. This is the
        # comparison that the noise floor should be read against.
        hi, lo = estimator_values
        gaps = []
        for spec in specs:
            for seed in range(args.seeds):
                a = [
                    r["accuracy"]
                    for r in ok
                    if r["dataset"] == spec.name and r["seed"] == seed and r["n_estimators"] == hi
                ]
                b = [
                    r["accuracy"]
                    for r in ok
                    if r["dataset"] == spec.name and r["seed"] == seed and r["n_estimators"] == lo
                ]
                if a and b:
                    gaps.append(a[0] - b[0])
        report["paired_gap"] = {
            "description": f"accuracy(e={hi}) - accuracy(e={lo}), paired on (dataset, seed)",
            "gaps": gaps,
            **summarise(gaps),
        }
        if gaps:
            g = report["paired_gap"]
            print(
                f"\nPaired gap e={hi} vs e={lo}: mean {g['mean']:+.4f}"
                + (f", sd {g['sd']:.4f}" if g["sd"] is not None else ", sd unmeasured (n=1)")
                + f", over {len(gaps)} pairs"
            )

    if ok:
        overall = summarise([r["accuracy"] for r in ok])
        report["overall"] = overall
        print(f"\nMean accuracy across {overall['n']} runs: {overall['mean']:.4f}")
        if args.seeds == 1:
            print(
                "Noise floor NOT measured (one seed). Re-run with --seeds >= 3 before "
                "treating any difference as real."
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")

    failures = len(report["runs"]) - len(ok)
    return 1 if failures and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
