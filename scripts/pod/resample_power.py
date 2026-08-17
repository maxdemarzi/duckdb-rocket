"""How many resamples, and how many datasets, would it take to call a half-point?

RESULTS.md's top "Not done" entry says the noise floor is the binding constraint and that fixing
it "is a campaign rather than a run". This is the run that sizes the campaign. It does not try to
settle any comparison; it measures how much noise there is and where the noise lives, which is the
one thing that decides whether the campaign is affordable and what it should buy.

**The decomposition is the whole point.** A paired A-vs-B comparison over D datasets and R
resamples has two independent noise terms, and they respond to completely different money:

    SE(grand mean delta)  =  sqrt( between/D  +  within/(D*R) )

* `within` is split luck -- the same dataset and the same two configs disagreeing from one
  resample to the next. **More resamples shrink this.**
* `between` is real heterogeneity -- the effect genuinely differing by dataset. **More resamples
  do NOTHING to this.** Only more datasets touch it.

So the answer "run 30 resamples" is only correct if `within` dominates, and RESULTS.md already
hints it may not: the same 500-vs-10,000 comparison at G=40 gives -0.0038 over 28 datasets,
-0.0001 over the 24 with cubes, and the sign flips with the subset. A sign that flips with the
dataset subset is between-dataset variance, and thirty resamples per dataset would buy an
extremely precise estimate of a quantity that still changes when you swap the datasets.

Every run is a separate `phase5_pipeline.py` process, for the reasons sweep.py already gives, and
results are written after each one so a pod that dies late keeps what it earned.

    uv run python scripts/pod/resample_power.py --dry-run              # what it would run
    uv run python scripts/pod/resample_power.py --datasets Beef,OSULeaf --resamples 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from duckdb_rocket.datasets import RUNNABLE_SUBSET, UCR_SUBSET, describe  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent

#: The comparison this pilot is sized around: the cheap corner against the archived baseline.
#: Chosen because it is the one with real money behind it -- a 20x kernel cut and a 4x group cut
#: measured -0.0003 on a single split, and if that is genuinely zero the student gets 20x cheaper
#: for nothing. It is also the comparison whose sign already flips with the dataset subset, which
#: is what makes it the right probe for where the variance lives.
#:
#: kernels-per-group is held at 250 across both arms deliberately. A group is 2 features per
#: kernel, so changing kernels while holding groups would change the feature width the model sees,
#: and the comparison would confound the bank size with the input width.
CONFIG_A = ("--num-kernels", "10000", "--n-groups", "40")   # baseline: 250 kernels/group
CONFIG_B = ("--num-kernels", "2500", "--n-groups", "10")    # cheap corner: same 250/group, 4x fewer


def one_run(args, dataset: str, resample: int, arm: str, config: tuple[str, ...]) -> dict:
    out = ROOT / "reference" / "resample" / f"{dataset}_r{resample}_{arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
        "--dataset", dataset,
        # The resample index, NOT --seed. Both arms of a pair get the same split -- that is what
        # makes the difference paired, and pairing is worth far more here than any number of
        # extra runs: it cancels the dataset's own difficulty, which is the largest term of all.
        "--resample", str(resample),
        "--test-chunk", str(args.test_chunk),
        "--threads", str(args.threads),
        "--out", str(out),
        *config,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"dataset": dataset, "resample": resample, "arm": arm, "error": "timeout"}
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out.exists():
        return {"dataset": dataset, "resample": resample, "arm": arm, "seconds": elapsed,
                "error": (proc.stderr or proc.stdout or "")[-400:]}
    report = json.loads(out.read_text(encoding="utf-8"))
    return {"dataset": dataset, "resample": resample, "arm": arm,
            "accuracy": report["accuracy"], "seconds": report["seconds"],
            "n_test": report["shape"]["n_test"], "failures": report["failures"]}


def analyse(runs: list[dict], target: float) -> dict:
    """Split the noise into the part resamples fix and the part only datasets fix."""
    by_key = {(r["dataset"], r["resample"], r["arm"]): r["accuracy"]
              for r in runs if "accuracy" in r}
    # A pair is complete only when BOTH arms of the SAME resample finished. A half-finished pair
    # contributes nothing: taking arm A's accuracy against some other resample's arm B would
    # reintroduce exactly the split noise the pairing exists to remove.
    deltas: dict[str, dict[int, float]] = {}
    for (ds, k, arm) in list(by_key):
        if arm != "A":
            continue
        if (ds, k, "B") in by_key:
            deltas.setdefault(ds, {})[k] = by_key[(ds, k, "B")] - by_key[(ds, k, "A")]

    per_dataset = {}
    within_vars = []
    for ds, ks in sorted(deltas.items()):
        vals = [ks[k] for k in sorted(ks)]
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        per_dataset[ds] = {"n_resamples": len(vals), "mean_delta": statistics.fmean(vals),
                           "sd_within": sd, "deltas": {str(k): ks[k] for k in sorted(ks)}}
        if sd is not None:
            within_vars.append(sd ** 2)

    D = len(per_dataset)
    if D == 0:
        return {"per_dataset": per_dataset, "note": "no complete pairs yet"}

    means = [v["mean_delta"] for v in per_dataset.values()]
    R = statistics.fmean([v["n_resamples"] for v in per_dataset.values()])
    within = statistics.fmean(within_vars) if within_vars else 0.0

    # The variance of the per-dataset MEANS already contains within/R, so the between-dataset
    # component is what is left after removing it. It can come out negative when the true
    # heterogeneity is near zero and D is small; that is an estimate of zero, not a defect, and
    # clamping it silently would turn "we cannot tell them apart" into "there is no heterogeneity".
    var_of_means = statistics.variance(means) if D > 1 else float("nan")
    between_raw = var_of_means - within / R if R else float("nan")
    between = max(between_raw, 0.0)

    def se(d: int, r: int) -> float:
        return math.sqrt(between / d + within / (d * r)) if d and r else float("inf")

    # What would it take to resolve `target`? Detecting an effect of that size at conventional
    # power wants roughly SE <= target / 2.8 (1.96 + 0.84, two-sided 5% at 80%).
    want_se = target / 2.8
    plans = []
    for d in (D, 24, 28, 40, 60, 100):
        r_needed = None
        for r in range(1, 201):
            if se(d, r) <= want_se:
                r_needed = r
                break
        plans.append({"datasets": d, "resamples_needed": r_needed,
                      "runs": (2 * d * r_needed) if r_needed else None,
                      "se_at_30": se(d, 30)})

    return {
        "per_dataset": per_dataset,
        "datasets": D,
        "mean_resamples": R,
        "grand_mean_delta": statistics.fmean(means),
        "se_observed": se(D, int(R)) if R else None,
        "var_within": within,
        "var_between": between,
        "var_between_raw": between_raw,
        # The number that decides what to buy. Resamples cannot push SE below this, however many
        # are run, because between-dataset heterogeneity does not average out over resamples.
        "se_floor_at_this_D": math.sqrt(between / D) if D else None,
        "target_effect": target,
        "se_needed_for_target": want_se,
        "plans": plans,
    }


def report(a: dict) -> None:
    if "datasets" not in a:
        print(f"  {a.get('note')}")
        return
    print(f"\n  {'dataset':<26} {'R':>3} {'mean delta':>11} {'sd across resamples':>21}")
    for ds, v in a["per_dataset"].items():
        sd = f"{v['sd_within']:.4f}" if v["sd_within"] is not None else "  (one resample)"
        print(f"  {ds:<26} {v['n_resamples']:>3} {v['mean_delta']:>+11.4f} {sd:>21}")

    print(f"\n  grand mean delta  {a['grand_mean_delta']:+.4f}   "
          f"SE {a['se_observed']:.4f}  over {a['datasets']} datasets x {a['mean_resamples']:.0f}")
    print(f"  variance within (split luck, resamples fix this)   {a['var_within']:.6f}")
    print(f"  variance between (dataset effect, they do not)     {a['var_between']:.6f}"
          + ("   [raw estimate was negative -> effectively zero]"
             if a["var_between_raw"] < 0 else ""))
    print(f"\n  SE floor at {a['datasets']} datasets, however many resamples: "
          f"{a['se_floor_at_this_D']:.4f}")
    print(f"  to resolve {a['target_effect']:.4f} you need SE <= {a['se_needed_for_target']:.4f}")

    print(f"\n  {'datasets':>9} {'resamples':>10} {'total runs':>11} {'SE at R=30':>11}")
    for p in a["plans"]:
        need = str(p["resamples_needed"]) if p["resamples_needed"] else ">200"
        runs = str(p["runs"]) if p["runs"] else "-"
        print(f"  {p['datasets']:>9} {need:>10} {runs:>11} {p['se_at_30']:>11.4f}")
    if all(p["resamples_needed"] is None for p in a["plans"]):
        print("\n  No row reaches the target: the between-dataset term dominates, and this "
              "\n  comparison cannot be resolved by resampling at any affordable scale. That is "
              "\n  a result about the effect, not a failure of the harness.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", help="comma-separated; default is every runnable one")
    parser.add_argument("--resamples", type=int, default=5,
                        help="resample indices 1..N. The pilot's job is to estimate variance, "
                             "not to settle the comparison, so this is small by design.")
    parser.add_argument("--test-chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--target", type=float, default=0.005,
                        help="the effect size worth resolving. 0.005 is the scale of every "
                             "result RESULTS.md currently reports as undetectable.")
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "resample_power.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run list and the cost, and spend nothing")
    parser.add_argument("--analyse-only", action="store_true",
                        help="re-read --out and re-run the statistics, with no pod time at all")
    args = parser.parse_args()

    if args.analyse_only:
        if not args.out.exists():
            parser.error(f"no {args.out} to analyse")
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        report(analyse(prior["runs"], args.target))
        return 0

    if args.datasets:
        wanted = {n.strip() for n in args.datasets.split(",")}
        specs = [d for d in UCR_SUBSET if d.name in wanted]
        missing = wanted - {d.name for d in specs}
        if missing:
            parser.error(f"unknown dataset(s): {', '.join(sorted(missing))}")
        specs = [d for d in specs if d.runnable]
    else:
        specs = list(RUNNABLE_SUBSET)
        for d in (x for x in UCR_SUBSET if not x.runnable):
            print(f"SKIP  {describe(d)}", file=sys.stderr)

    jobs = [(s.name, k, arm, cfg)
            for s, k, (arm, cfg) in itertools.product(
                specs, range(1, args.resamples + 1),
                (("A", CONFIG_A), ("B", CONFIG_B)))]

    print(f"{len(specs)} datasets x {args.resamples} resamples x 2 arms = {len(jobs)} runs")
    print(f"  A {' '.join(CONFIG_A)}\n  B {' '.join(CONFIG_B)}")
    if args.dry_run:
        # Costing off the archived wall clocks rather than a guess: those are the same pipeline on
        # the same datasets, which is the only honest estimate available before spending anything.
        known, unknown = [], 0
        for s in specs:
            f = ROOT / "reference" / f"phase5_{s.name}.json"
            if f.exists():
                known.append(json.loads(f.read_text(encoding="utf-8"))["seconds"])
            else:
                unknown += 1
        if known:
            per = statistics.fmean(known)
            hours = len(jobs) * per / 3600 / max(1, args.jobs)
            print(f"\n  archived mean wall clock {per:.0f}s over {len(known)} datasets"
                  f"{f' ({unknown} unmeasured)' if unknown else ''}")
            print(f"  ~{hours:.1f} h at --jobs {args.jobs}, and arm B is cheaper than its "
                  f"archived A, so this is an upper bound")
        for ds, k, arm, cfg in jobs[:6]:
            print(f"    {ds} r{k} arm {arm}: {' '.join(cfg)}")
        if len(jobs) > 6:
            print(f"    ... and {len(jobs) - 6} more")
        return 0

    runs: list[dict] = []
    started_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(one_run, args, ds, k, arm, cfg): (ds, k, arm)
                   for ds, k, arm, cfg in jobs}
        for future in as_completed(futures):
            ds, k, arm = futures[future]
            r = future.result()
            runs.append(r)
            if "accuracy" in r:
                print(f"  {ds} r{k} {arm}  acc={r['accuracy']:.4f}  {r['seconds']:.0f}s"
                      + (f"  ALIGNMENT {r['failures']}" if r.get("failures") else ""), flush=True)
            else:
                print(f"  {ds} r{k} {arm}  FAILED: {str(r.get('error'))[:200]}", flush=True)
            args.out.write_text(json.dumps({
                "arms": {"A": list(CONFIG_A), "B": list(CONFIG_B)},
                "resamples": args.resamples, "runs": runs,
                "elapsed_seconds": round(time.perf_counter() - started_all, 1),
                "analysis": analyse(runs, args.target),
            }, indent=2), encoding="utf-8")

    report(analyse(runs, args.target))
    print(f"\nwrote {args.out}  ({(time.perf_counter() - started_all) / 60:.1f} min)")
    return 0 if all("accuracy" in r for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
