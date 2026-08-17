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

#: The GROUP-COUNT lever: 40 groups against 10, at 250 kernels per group in both. This is
#: RESULTS.md's "10,000 kernels, G=10" cell, which measured **-0.0033** on the archived single
#: split -- a 4x cheaper student, for a cost that may or may not be real. That is the target
#: number this pilot has to decide is resolvable.
#:
#: kernels-per-group is held at 250 across both arms deliberately, so this is a clean group-count
#: comparison rather than a confounded one. A group is 2 features per kernel; changing the bank
#: size while holding the group count would change the feature width the model sees, and the
#: comparison would then mix the bank size with the input width.
#:
#: A consequence worth stating, because it makes the estimate specific rather than general: arm B
#: is a strict PREFIX of arm A. Kernel i is a pure function of (seed, i), so groups 0-9 of a
#: 40-group run are exactly a 10-group run, and RESULTS.md relies on the same property to read its
#: group sweep off archived cubes. The two arms therefore share their first ten groups, which
#: makes the pair more tightly coupled than two unrelated configurations -- so `within` measured
#: here is a floor for nested comparisons, not a universal constant to quote at unrelated ones.
CONFIG_A = ("--num-kernels", "10000", "--n-groups", "40")   # 40 groups x 250 kernels
CONFIG_B = ("--num-kernels", "2500", "--n-groups", "10")    # its first 10 groups, exactly

#: Named comparisons, so the paired machinery below is reused rather than copied. Every one holds
#: everything except the single thing under test -- that is what makes the difference paired, and it
#: is the property a new entry has to preserve to belong here.
ARM_SETS = {
    #: Phase 7's group lever: 40 groups against its own first 10.
    "groups": (CONFIG_A, CONFIG_B),
    #: Phase 7b': feature CONCATENATION, which is where 7a's negative points. 7a established that
    #: no rule can SELECT between arms -- averaging, margin-routing, surest-arm and a supervised
    #: stacker all failed -- so the remaining move is to stop treating families as separate arms and
    #: hand both to one model. 500 ROCKET + 116 statistics = 616 columns, which stays inside
    #: tabicl-v2's 512-per-estimator budget at G=40. Archived on six hard datasets at +0.0088,
    #: 4 wins to 2, on one split; this is what tests it properly.
    "features": (("--num-kernels", "10000", "--n-groups", "40", "--features", "rocket"),
                 ("--num-kernels", "10000", "--n-groups", "40", "--features", "both")),
}


def one_run(args, dataset: str, resample: int, arm: str, config: tuple[str, ...]) -> dict:
    out = (ROOT / "reference" / "resample" / args.arms /
           f"{dataset}_r{resample}_{arm}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Already done: return it rather than recompute. This is what makes the campaign growable --
    # run --resamples 1, look at the answer, then --resamples 3 and pay only for the two new ones.
    # On a small box that is the difference between committing to eight hours up front and
    # deciding after two whether the rest is worth it.
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
            return {"dataset": dataset, "resample": resample, "arm": arm, "cached": True,
                    "accuracy": prior["accuracy"], "seconds": prior["seconds"],
                    "n_test": prior["shape"]["n_test"], "failures": prior["failures"]}
        except (ValueError, KeyError):
            pass          # truncated by a kill mid-write; fall through and redo it
    cmd = [
        sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
        "--dataset", dataset,
        # The resample index, NOT --seed. Both arms of a pair get the same split -- that is what
        # makes the difference paired, and pairing is worth far more here than any number of
        # extra runs: it cancels the dataset's own difficulty, which is the largest term of all.
        "--resample", str(resample),
        "--test-chunk", str(args.test_chunk),
        "--threads", str(args.threads),
        # Passed explicitly, never left to the pipeline's own default, because that default sizes
        # from the visible core count and every concurrent job would size from the SAME number.
        # That is the failure PLAN.md records: four concurrent runs on a 112-core pod each built a
        # pool from 112, and all four died near completion with no error message at all. With
        # --jobs J the box carries J x threads x onnx_threads at once.
        "--onnx-threads", str(args.onnx_threads),
        # Same trap as the thread pools, in its memory form and with a worse failure. Left to
        # itself the pipeline sets memory_limit to 70% of the cgroup, which is correct for one run
        # and catastrophic for J of them: at --jobs 6 on a 64 GB cgroup, six runs each claimed
        # 44.8 GB and the kernel took them at exit -9, with no DuckDB error and no traceback --
        # 12 of the first 92 runs died this way. Divided here because only the driver knows J.
        "--memory-limit", args.memory_limit,
        # One directory per (dataset, resample, arm). Without this, --jobs > 1 has concurrent runs
        # of the same dataset writing and reading one raw.parquet: observed as "No magic bytes
        # found at end of file" on the first launch, which is the loud version. The quiet version
        # is one resample reading the split another just wrote.
        "--workdir", str(ROOT / "data" / "resample" / args.arms /
                         f"{dataset}_r{resample}_{arm}"),
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
        # The stderr tail is useless on its own: ONNX Runtime prints tens of thousands of
        # "Trying to register schema with name ..." lines, so the last 400 characters are always
        # that, whatever actually went wrong. The first diagnosis of an OOM kill here read
        # "defs.cc line 927" and pointed at a schema registration that was fine. Pull the
        # pipeline's own verdict out instead, and keep the exit code, which is what says -9.
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        signal = [ln for ln in blob.splitlines()
                  if ln.strip() and "schema error" not in ln.lower()
                  and "Trying to register schema" not in ln
                  and "registered from" not in ln]
        verdict = next((ln for ln in reversed(signal) if "FAILED" in ln or "Error" in ln), "")
        return {"dataset": dataset, "resample": resample, "arm": arm, "seconds": elapsed,
                "returncode": proc.returncode,
                "error": (verdict or "\n".join(signal[-4:]))[:400]}
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
    # With one dataset there is no between-dataset variance to estimate, and every quantity below
    # is NaN. NaN compares False against everything, so `se(d, r) <= want_se` fails for every plan
    # and the report used to conclude "the between-dataset term dominates; this cannot be resolved
    # at any affordable scale" -- a verdict assembled entirely from missing data, and one that
    # reads exactly like a finding. Say what is actually true instead.
    if D < 2:
        return {"per_dataset": per_dataset, "datasets": D, "mean_resamples": R,
                "grand_mean_delta": statistics.fmean(means), "var_within": within,
                "note": f"{D} dataset with complete pairs: the between-dataset term cannot be "
                        f"estimated at all from one dataset, so no campaign can be sized yet. "
                        f"Needs at least 2, and realistically 8-10 for a usable estimate."}

    var_of_means = statistics.variance(means)
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
    if "note" in a:
        # Covers both "nothing complete yet" and "too few datasets to decompose". Printed instead
        # of the plan table, never alongside it: a table of NaN beside a caveat still gets read as
        # a table.
        for ds, v in a.get("per_dataset", {}).items():
            print(f"  {ds:<26} R={v['n_resamples']} mean delta {v['mean_delta']:+.4f}")
        print(f"\n  {a['note']}")
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
    parser.add_argument("--threads", type=int, default=2,
                        help="DuckDB threads per run. Small by default because these runs are "
                             "largely serial -- 40 classify calls one after another -- so the "
                             "parallelism worth having is across jobs, not inside one.")
    parser.add_argument("--onnx-threads", type=int, default=2,
                        help="ONNX intra-op threads per run. jobs x threads x onnx-threads is "
                             "what the box actually carries; size it to the CGROUP quota, which "
                             "on a RunPod GPU host is nothing like what nproc reports.")
    parser.add_argument("--arms", default="groups", choices=sorted(ARM_SETS),
                        help="which paired comparison to run. 'groups' is 40 groups vs its own "
                             "first 10. 'features' is ROCKET vs ROCKET+statistics concatenated, "
                             "which is where Phase 7a's negative points.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--memory-limit", default=None,
                        help="DuckDB memory_limit PER RUN, e.g. '8GB'. Defaults to 60%% of the "
                             "cgroup limit divided by --jobs, because the pipeline's own default "
                             "is 70%% of the cgroup and every concurrent job would claim that "
                             "same share. An OOM kill here leaves exit -9, no DuckDB error and no "
                             "traceback, which looks exactly like a hang.")
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
    arm_a, arm_b = ARM_SETS[args.arms]

    if args.analyse_only:
        # Before the memory arithmetic, which is about running and has no business printing
        # "memory_limit 8GB per run x 1 jobs" above a table of results computed weeks ago.
        if not args.out.exists():
            parser.error(f"no {args.out} to analyse")
        report(analyse(json.loads(args.out.read_text(encoding="utf-8"))["runs"], args.target))
        return 0

    if not args.memory_limit:
        # The cgroup, not free(3): inside a container those differ and it is the cgroup that
        # kills you. 60% rather than the pipeline's 70% because the model allocates OUTSIDE
        # DuckDB's buffer manager, so the limit governs only part of a run's footprint.
        total = None
        for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                raw = Path(f).read_text().strip()
                if raw != "max":
                    total = int(raw)
                    break
            except (OSError, ValueError):
                continue
        if total and total < (1 << 62):          # v1 reports a sentinel when unlimited
            per = int(total * 0.6 / max(1, args.jobs))
            args.memory_limit = f"{max(1, per // (1 << 30))}GB"
        else:
            args.memory_limit = "8GB"
        print(f"memory_limit {args.memory_limit} per run x {args.jobs} jobs", flush=True)

    if args.datasets:
        # Validated against the full 112 equal-length univariate UCR archive, not against
        # UCR_SUBSET. That subset is the ten datasets the breadth sweep uses and nine of its ten
        # sit at 0.94-1.00, which makes it exactly the wrong population for a variance pilot: a
        # saturated dataset has no room for a resample to move it, so it contributes a `within` of
        # zero and drags the estimate toward "resampling is free".
        wanted = [n.strip() for n in args.datasets.split(",") if n.strip()]
        try:
            from aeon.datasets.tsc_datasets import univariate_equal_length
            known = set(univariate_equal_length)
        except ImportError:                       # aeon absent: fall back to the curated subset
            known = {d.name for d in UCR_SUBSET}
        missing = [n for n in wanted if n not in known]
        if missing:
            parser.error(f"unknown dataset(s): {', '.join(sorted(missing))}")
        names = wanted
    else:
        names = [d.name for d in RUNNABLE_SUBSET]
        for d in (x for x in UCR_SUBSET if not x.runnable):
            print(f"SKIP  {describe(d)}", file=sys.stderr)

    # Cheapest dataset first, and both arms of a pair adjacent. A run on a small box may have to
    # be stopped before it finishes, and the analysis drops half-finished pairs -- so the order
    # decides whether an early stop leaves usable pairs or fragments. Cost comes from the archived
    # wall clocks; datasets with no archived run sort last, since an unknown cost is the one most
    # likely to be large.
    def archived_cost(name: str) -> float:
        for pat in (f"phase5_{name}.json", f"phase5_{name}_cpu.json", f"phase5_{name}_gpu.json"):
            p = ROOT / "reference" / pat
            if p.exists():
                try:
                    return float(json.loads(p.read_text(encoding="utf-8"))["seconds"])
                except (ValueError, KeyError):
                    continue
        return float("inf")

    ordered = sorted(names, key=archived_cost)
    jobs = [(n, k, arm, cfg)
            for n, k, (arm, cfg) in itertools.product(
                ordered, range(1, args.resamples + 1),
                (("A", arm_a), ("B", arm_b)))]

    print(f"{len(names)} datasets x {args.resamples} resamples x 2 arms = {len(jobs)} runs")
    print(f"comparison '{args.arms}':\n  A {' '.join(arm_a)}\n  B {' '.join(arm_b)}")
    if args.dry_run:
        # Costing off the archived wall clocks rather than a guess: those are the same pipeline on
        # the same datasets, which is the only honest estimate available before spending anything.
        known, unknown = [], 0
        for n in names:
            # The archive names the same run three ways depending on which sweep produced it --
            # phase5_X.json, phase5_X_cpu.json, phase5_X_ts.json -- and checking only the first
            # costed a 16-dataset pilot off ONE dataset while reporting "15 unmeasured" in small
            # print. An estimate that thin is worse than none, because it still gets quoted.
            hit = next((p for p in (ROOT / "reference").glob(f"phase5_{n}.json")), None) \
                or next((p for p in (ROOT / "reference").glob(f"phase5_{n}_cpu.json")), None)
            if hit:
                known.append(json.loads(hit.read_text(encoding="utf-8"))["seconds"])
            else:
                unknown += 1
        if known:
            per = statistics.fmean(known)
            hours = len(jobs) * per / 3600 / max(1, args.jobs)
            print(f"\n  archived mean wall clock {per:.0f}s over {len(known)} datasets"
                  f"{f' ({unknown} unmeasured)' if unknown else ''}")
            # Whether the archived time over- or under-states the total depends on the
            # comparison, and saying "upper bound" unconditionally would be wrong for half of
            # them: `groups` arm B runs a quarter of the groups, `features` arm B carries 616
            # columns against 500 and is DEARER than the archived run it is costed from.
            direction = ("and arm B runs fewer groups, so this is an upper bound"
                         if args.arms == "groups" else
                         "and arm B carries MORE columns than the archived run, so this is a "
                         "LOWER bound")
            print(f"  ~{hours:.1f} h at --jobs {args.jobs}, {direction}")
        for ds, k, arm, cfg in jobs[:6]:
            print(f"    {ds} r{k} arm {arm}: {' '.join(cfg)}")
        if len(jobs) > 6:
            print(f"    ... and {len(jobs) - 6} more")
        return 0

    # Pull every dataset into the cache SERIALLY before any job starts. The archive is downloaded
    # and extracted on first use, and that is not safe to do from several processes at once: the
    # pilot's only failure in 160 runs was the FIRST job to touch InsectEPGSmallTrain, which died
    # in 1.2 s reading a half-written file ("array at index 0 has 1 dimension(s), index 1 has 2").
    # Every later run of that dataset succeeded, because by then the cache was complete.
    #
    # Rare here and not rare in the campaign: a fresh pod running 40 datasets faces this 40 times,
    # and each hit quietly costs one resample of one dataset rather than failing the run.
    print(f"warming the dataset cache for {len(names)} datasets", flush=True)
    from duckdb_rocket.datasets import load as _load
    for n in names:
        try:
            _load(n, "train"), _load(n, "test")
        except Exception as exc:                  # report and carry on; the run will fail loudly
            print(f"  {n}: FAILED TO LOAD -- {type(exc).__name__}: {exc}", file=sys.stderr)

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
                "comparison": args.arms,
                "arms": {"A": list(arm_a), "B": list(arm_b)},
                "resamples": args.resamples, "runs": runs,
                "elapsed_seconds": round(time.perf_counter() - started_all, 1),
                "analysis": analyse(runs, args.target),
            }, indent=2), encoding="utf-8")

    report(analyse(runs, args.target))
    print(f"\nwrote {args.out}  ({(time.perf_counter() - started_all) / 60:.1f} min)")
    return 0 if all("accuracy" in r for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
