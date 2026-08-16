"""Routing at serving time: deploy the artifacts, serve a batch, report where the time went.

`distill_gate.py --route` measured routing by sorting a whole test set and escalating its least
confident fraction. Nothing serving a request can do that. This is the same rule as a system would
actually run it -- a threshold on one row's margin, fixed before the batch arrives -- executed
end to end through the real extension, so the numbers include the parts an analysis skips.

    uv run python scripts/route_serve.py deploy --dataset ScreenType --target 0.20
    uv run python scripts/route_serve.py serve  --dataset ScreenType --batch 128

**What is deployed.** Three things, and the first is the one worth noticing:

* the ROCKET features of the labelled training rows -- which serve double duty as the matrix the
  ridge was fit on AND as the teacher's in-context training table. The teacher has no trained weights
  for your task; `tabfm_classify` takes the labelled rows as context on every call, so deploying it
  means deploying the training data with it.
* the ridge head: a scaler and a coefficient matrix, kilobytes.
* one float: the margin threshold, taken as a quantile of out-of-fold margins on the train split.

**One feature computation serves both models.** The teacher's groups of 250 kernels are slices
[250g, 250(g+1)) of exactly the bank the student's ridge uses -- verified to 1.8e-15 against
`rocket_transform(values, 250, seed, offset)`. So the student reads every feature and the teacher
reads 500 at a time, from one transform. `n_kernels` and `n_groups` are therefore one decision, and
`deploy` derives the first from the second rather than taking both.

**The default is 10 groups, not the 40 every archived run used.** Cost is exactly linear in the
group count, and over 24 datasets G=10 costs -0.0033 routed against G=40 -- inside what this
harness can resolve, which is about half a point at a 20% budget. That is a 4x cut on the expensive
path. The archived pipeline keeps 40 so its numbers stay comparable; this is the serving path,
which has no archive to protect. See RESULTS.md, "The teacher runs 40 passes and needs about ten"
and "Both levers at once, and a noise floor worth naming".

**Why the teacher call reuses `phase5_pipeline.build_sql`.** The escalated rows are just a small test
split against the same training context, so the generated pipeline is exactly right for them -- and
it carries the id-recovery key, the kernel-bank fingerprint and the `features_check` guard, every one
of which exists because something silently produced a plausible wrong answer without it. Writing a
leaner serving query would drop those.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402
from duckdb_rocket.shells import built_shell  # noqa: E402

import phase5_pipeline as p5  # noqa: E402
from distill_gate import ALPHAS, decision_margin, oof_margins  # noqa: E402


#: Kernels per teacher group, and the one number that must not drift. 250 kernels is 500 features
#: per `tabfm_classify` call, which is what every archived accuracy was measured at and what fits
#: `tabicl-v2`'s 512-column cap. `n_kernels` and `n_groups` are two ways of saying the same thing
#: and are checked against each other rather than set independently.
KERNELS_PER_GROUP = 250

#: Ten groups, not forty. Cost is exactly linear in the group count and 24 datasets put G=10 at
#: -0.0033 routed against G=40, which no test here can detect (reference/RESULTS.md, "The teacher
#: runs 40 passes and needs about ten"). The archived runs stay at 40 so their numbers remain
#: comparable; this is the serving path, which has no archive to protect.
DEFAULT_GROUPS = 10


def deploy(dataset: str, target: float, n_groups: int, seed: int, folds: int,
           out: Path, n_kernels: int | None = None) -> dict:
    """Fit the student, choose the threshold, write everything a server needs.

    The kernel bank follows the group count rather than being set beside it: the teacher's groups
    are slices [250g, 250(g+1)) of the student's bank, so `n_kernels` and `n_groups` are one
    decision. Passing them separately is how a serve at G=40 against a 2,500-kernel deploy would
    silently run 62-kernel groups -- 124 features against the 500 every measurement used -- and
    still produce plausible answers.
    """
    n_kernels = n_kernels if n_kernels is not None else n_groups * KERNELS_PER_GROUP
    if n_kernels != n_groups * KERNELS_PER_GROUP:
        raise ValueError(
            f"{n_kernels} kernels over {n_groups} groups is "
            f"{n_kernels / n_groups:.1f} kernels per group; every accuracy here was measured at "
            f"{KERNELS_PER_GROUP} (500 features per call). Scale them together.")
    xtr, ytr = load(dataset, "train")
    xtr = normalize_series(xtr)
    bank = generate_kernels(seed, xtr.shape[-1], n_kernels)
    ftr = transform(xtr, bank)
    scaler = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(scaler.transform(ftr), ytr)

    # The threshold comes from margins the model produced on rows it had NOT seen. A fitted model is
    # systematically surer of its own training rows, so an in-sample quantile sits too high and the
    # server would escalate too little.
    # n_kernels, not the function's default: the threshold is a quantile of THIS model's margins,
    # and a bank of a different size has a different decision scale.
    margins = oof_margins(dataset, "rocket+ridge", seed, folds,
                          str(ROOT / "data" / "oof_margins"), n_kernels=n_kernels)
    threshold = float(np.quantile(margins, target))

    out.mkdir(parents=True, exist_ok=True)
    (out / "student.pkl").write_bytes(pickle.dumps({"scaler": scaler, "clf": clf}))
    # n_groups is deployed, not chosen again at serve time. It is half of the same decision as
    # n_kernels, and a server that took it as a separate flag could not be checked.
    meta = {"dataset": dataset, "seed": seed, "n_kernels": n_kernels, "n_groups": n_groups,
            "n_timepoints": int(xtr.shape[-1]), "target": target, "threshold": threshold,
            "folds": folds, "n_train": int(len(ytr)), "classes": [str(c) for c in clf.classes_]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"deployed {dataset} -> {out}")
    print(f"  student: {n_kernels} kernels, {ftr.shape[1]} features, ridge over "
          f"{len(clf.classes_)} classes")
    print(f"  teacher: {n_groups} groups x {KERNELS_PER_GROUP} kernels = "
          f"{KERNELS_PER_GROUP * 2} features per call")
    print(f"  threshold {threshold:.4f} at a {target:.0%} target, from {folds}-fold out-of-fold "
          f"margins over {len(margins)} training rows")
    return meta


def student_predict(meta: dict, art: Path, x: np.ndarray):
    """Predictions and margins. This is the whole 80% path: a transform and a matrix multiply."""
    d = pickle.loads((art / "student.pkl").read_bytes())
    bank = generate_kernels(meta["seed"], meta["n_timepoints"], meta["n_kernels"])
    f = d["scaler"].transform(transform(x, bank))
    dec = d["clf"].decision_function(f)
    return d["clf"].predict(f), decision_margin(dec)


def teacher_predict(dataset: str, idx: np.ndarray, workdir: Path, n_groups: int,
                    num_kernels: int, seed: int, shell: Path,
                    memory_limit: str | None = None) -> np.ndarray:
    """The teacher on the escalated rows only, through the real extension.

    Built as a one-off dataset whose test split IS the escalated batch, so `build_sql` produces
    exactly the right pipeline: 40 groups against the same labelled context, probabilities averaged,
    argmax. Its correctness guards come along -- the full-vector id recovery, the bank fingerprint,
    and the features_check that catches a silently dropped feature name.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    xtr, ytr = load(dataset, "train")
    xte, _ = load(dataset, "test")
    xtr, xte = normalize_series(xtr), normalize_series(xte)
    xq = xte[idx]
    n_train, n_q = len(ytr), len(xq)

    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / "raw.parquet"
    pq.write_table(pa.table({
        "id": pa.array(np.arange(n_train + n_q), type=pa.int64()),
        "split": pa.array(["train"] * n_train + ["test"] * n_q),
        "label": pa.array([str(v) for v in ytr] + ["?"] * n_q),
        "values": pa.array(list(xtr) + list(xq), type=pa.list_(pa.float64())),
    }), raw)

    cfg = RocketPFNConfig(num_kernels=num_kernels, n_groups=n_groups, seed=seed, n_estimators=1)
    cfg.validate()
    meta = {"dataset": dataset, "n_train": n_train, "n_test": n_q, "n_channels": 1,
            "n_timepoints": int(xtr.shape[-1]), "multivariate": False,
            "raw_parquet": raw.as_posix()}
    # Not a hardcoded "8GB", which is what this was and which is an order of magnitude under
    # phase5's own cgroup-aware default (44GB on the dev box, ~87GB on the 124GB pod). Worth
    # matching the rest of the codebase regardless -- but note what it did NOT fix, below.
    sql = p5.build_sql(cfg, meta, workdir, 4, memory_limit or p5.default_memory_limit(),
                       workdir, 128, 4, device="cpu")
    (workdir / "serve.sql").write_text(sql, encoding="utf-8")
    # encoding is explicit: DuckDB's box-drawing output is UTF-8 and Windows would otherwise decode
    # it as cp1252 and raise mid-run, after the work has been done.
    r = subprocess.run([str(shell), "-c", f".read {(workdir / 'serve.sql').as_posix()}"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    pred_path = workdir / "predictions.json"
    if not pred_path.exists():
        # The exit code, the group it reached, and the whole stdout on disk -- none of which this
        # reported before, which is why an 8GB memory limit got diagnosed as the cause of a failure
        # that recurred at 87GB. A negative code is a signal (-11 is SIGSEGV, -9 an OOM kill); a
        # positive one is DuckDB refusing something and the message will be in the log.
        crash = workdir / "crash.log"
        crash.write_text(f"returncode={r.returncode}\n\n=== stdout ===\n{r.stdout}\n"
                         f"\n=== stderr ===\n{r.stderr}", encoding="utf-8")
        groups = r.stdout.count("] group ")
        raise RuntimeError(
            f"the teacher produced no predictions: the shell exited {r.returncode} after scoring "
            f"{groups} of {cfg.n_groups} groups on {n_q} test rows. Full output in {crash}.\n"
            f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    by_id = {int(p["id"]): str(p["yhat"])
             for p in json.loads(pred_path.read_text(encoding="utf-8"))}
    # Test row k is id n_train + k, asserted rather than assumed: the same offset that took three
    # attempts to get right in the pipeline itself.
    missing = [k for k in range(n_q) if (n_train + k) not in by_id]
    if missing:
        raise RuntimeError(f"the teacher returned no prediction for {len(missing)} escalated "
                           f"row(s), first at id {n_train + missing[0]}")
    return np.array([by_id[n_train + k] for k in range(n_q)])


def group_seconds(workdir: Path) -> tuple[float, float, float] | None:
    """(transform total, classify total, slowest/median group) from the run's own timings.json.

    The fit below uses the TOTAL, not the median group: the total is what the call actually costs,
    and a fixed-plus-marginal model built from medians would not add back up to it. The spread comes
    back alongside because a single stalled group -- 12.96x the median in the worst archived run --
    is the one thing that would make the total a bad summary, and it should be visible when it does.
    """
    p = workdir / "timings.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text(encoding="utf-8"))
    if not t:
        return None
    cl = sorted(float(r["classify_seconds"]) for r in t)
    med = cl[len(cl) // 2]
    return sum(float(r["transform_seconds"]) for r in t), sum(cl), (cl[-1] / med if med else 1.0)


def steadiness(workdir: Path) -> tuple[float, float] | None:
    """(spread of the repeated groups, group 0's warm-up factor) -- a contention detector.

    The 40 groups are the same computation 40 times over, so on an idle box their times are nearly
    identical: measured at +/-1% across the last five groups of every run here. Background load does
    not arrive uniformly, so it shows up as scatter. That makes this the check `serve` used to say
    it could not do -- it printed "this script cannot tell whether it is running clean" while
    writing the evidence to timings.json.

    Group 0 is excluded from the spread and reported separately: it carries the ONNX session
    warm-up and ran 1.2-1.4x the rest on every clean run, so folding it in would make a clean run
    look contended.
    """
    p = workdir / "timings.json"
    if not p.exists():
        return None
    t = sorted(json.loads(p.read_text(encoding="utf-8")), key=lambda r: r["grp"])
    if len(t) < 3:
        return None
    cl = [float(r["classify_seconds"]) for r in t]
    rest = cl[1:]
    mean = sum(rest) / len(rest)
    if mean <= 0:
        return None
    var = sum((x - mean) ** 2 for x in rest) / len(rest)
    return (var ** 0.5) / mean, cl[0] / mean


def cost_model(small: Path, big: Path, n_small: int, n_big: int, n_groups: int,
               wall_small: float, wall_big: float) -> None:
    """Split the teacher's cost into the part routing can avoid and the part it cannot.

    Two batch sizes of the same dataset give two points on `seconds_per_group = a + b*n`, and the
    split matters more than the total. **b*n is the only part escalating fewer rows removes.** `a`
    is the pass over the labelled context, which tabfm_classify redoes on every call because the
    teacher has no trained weights -- so it is paid once per group whether one row is escalated or
    all of them.

    Read off the run's own per-group timings rather than off wall clock, so that DuckDB startup and
    the ROCKET transform are not silently attributed to the model. On the first measurement those
    came to 0.5 s of a 63.2 s call -- small, but assuming it would have been an assumption.
    """
    a_small, a_big = group_seconds(small), group_seconds(big)
    if a_small is None or a_big is None or n_big == n_small:
        return
    (tr_s, cl_s, sp_s), (tr_b, cl_b, sp_b) = a_small, a_big
    # Per group, from the totals: sum / n_groups.
    per_s, per_b = cl_s / n_groups, cl_b / n_groups
    b = (per_b - per_s) / (n_big - n_small)
    a = per_s - b * n_small
    if a <= 0 or b <= 0:
        print("  (the two batch sizes do not separate a fixed and a marginal cost here)")
        return
    fixed, marginal = a * n_groups, b * n_groups
    print(f"\n  cost of a classify call, fitted on {n_small} and {n_big} rows:")
    print(f"    {a:.3f} s fixed per group + {b * 1000:.1f} ms per query row")
    print(f"    over {n_groups} groups: {fixed:.1f} s that escalating cannot avoid, "
          f"+ {marginal * 1000:.0f} ms per escalated row")
    print(f"    startup and transform, outside the model: {wall_small - cl_s - tr_s:.1f} s of the "
          f"{wall_small:.1f} s call ({tr_s:.1f} s of it the ROCKET transform)")
    if max(sp_s, sp_b) > 3:
        print(f"    CAUTION: one group ran {max(sp_s, sp_b):.1f}x the median, so the totals these "
              f"are fitted on are not a steady rate")
    # The number that decides whether routing is worth anything on this shape of batch.
    print(f"    so escalating {n_small}/{n_big} rows costs "
          f"{(fixed + marginal * n_small) / (fixed + marginal * n_big):.0%} of teacher-everywhere, "
          f"not {n_small / n_big:.0%}: the fixed pass is {fixed / (fixed + marginal * n_big):.0%} "
          f"of the full-batch cost and routing does not touch it")


def serve(dataset: str, art: Path, batch: int, n_groups: int | None, seed: int, shell: Path,
          workdir: Path, compare: bool = False, memory_limit: str | None = None) -> int:
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    # The group count is part of what was deployed, so it is read rather than re-chosen. An
    # override is still allowed -- it is how the group sweep was run -- but it has to keep
    # kernels-per-group at the width every measurement used, or the call silently changes shape.
    deployed = meta.get("n_groups")
    if n_groups is None:
        n_groups = deployed if deployed else meta["n_kernels"] // KERNELS_PER_GROUP
    if meta["n_kernels"] % n_groups or meta["n_kernels"] // n_groups != KERNELS_PER_GROUP:
        raise ValueError(
            f"serving {meta['n_kernels']} deployed kernels over {n_groups} groups is "
            f"{meta['n_kernels'] / n_groups:.1f} kernels per group, not {KERNELS_PER_GROUP}. "
            f"Re-deploy with --n-groups {n_groups} instead of overriding it here.")
    xte, yte = load(dataset, "test")
    xte_n = normalize_series(xte)
    take = min(batch, len(yte))
    x, y = xte_n[:take], yte[:take]

    t0 = time.perf_counter()
    spred, margin = student_predict(meta, art, x)
    t_student = time.perf_counter() - t0

    esc = margin < meta["threshold"]
    idx = np.nonzero(esc)[0]
    print(f"\nbatch of {take} rows from {dataset}")
    print(f"  student answered in {t_student * 1000:.0f} ms "
          f"({t_student / take * 1000:.2f} ms/row, features + ridge)")
    print(f"  escalating {esc.sum()}/{take} = {esc.mean():.1%} "
          f"(threshold {meta['threshold']:.4f}, target {meta['target']:.0%})")

    final = np.asarray(spred, dtype=object).copy()
    t_teacher = 0.0
    if len(idx):
        t0 = time.perf_counter()
        tpred = teacher_predict(dataset, idx, workdir, n_groups, meta["n_kernels"], seed, shell,
                                memory_limit)
        t_teacher = time.perf_counter() - t0
        final[idx] = tpred
        print(f"  teacher answered {len(idx)} rows in {t_teacher:.1f} s "
              f"({t_teacher / len(idx) * 1000:.0f} ms/row, {n_groups} groups)")
    else:
        print("  no row fell below the threshold, so this batch cost the student alone")

    truth = np.asarray(y, dtype=object)
    acc_routed = float((final == truth).mean())
    acc_student = float((np.asarray(spred, dtype=object) == truth).mean())
    total = t_student + t_teacher
    print(f"\n  routed   {acc_routed:.4f}   {total:.1f} s total")
    print(f"  student  {acc_student:.4f}   {t_student:.1f} s   (what you would have had for free)")

    # The cost claim needs all three arms on ONE box at ONE time. Assembling it from an archived
    # run instead is how a 27 s figure from a 96-core CUDA node got compared against a contended
    # 8-core CPU measurement and produced a "138x" that meant nothing.
    if compare and len(idx):
        t0 = time.perf_counter()
        tall = teacher_predict(dataset, np.arange(take), workdir / "all", n_groups,
                               meta["n_kernels"], seed, shell, memory_limit)
        t_all = time.perf_counter() - t0
        acc_teacher = float((np.asarray(tall, dtype=object) == truth).mean())
        print(f"  teacher  {acc_teacher:.4f}   {t_all:.1f} s   (every row, same box, same moment)")
        print(f"\n  routing spent {total / t_all:.0%} of the teacher-everywhere time for "
              f"{(acc_routed - acc_student) / (acc_teacher - acc_student):.0%} of its accuracy gain"
              if acc_teacher != acc_student else "\n  the teacher and student tie on this batch")
        print(f"  per-row: {t_student / take * 1000:.1f} ms student, "
              f"{t_teacher / len(idx) * 1000:.0f} ms teacher on {len(idx)} escalated rows, "
              f"{t_all / take * 1000:.0f} ms teacher on all {take}")
        print(f"  the teacher's per-row cost falls {(t_teacher / len(idx)) / (t_all / take):.1f}x "
              f"going from {len(idx)} rows to {take} -- its context pass is fixed per call, so a "
              f"small escalation batch amortises it over fewer rows")
        cost_model(workdir, workdir / "all", len(idx), take, n_groups, t_teacher, t_all)
    elif len(idx):
        print(f"  the escalated {esc.mean():.0%} of rows took {t_teacher / total:.0%} of the time")
        print("  (--compare runs the teacher on every row too, which is the only honest way to "
              "state a cost ratio)")
    # Not "timings mean nothing on a contended box, and this cannot tell" -- it can, from the
    # repeated groups it already timed.
    st = steadiness(workdir)
    if st is None:
        print("\n  No per-group timings, so nothing here says whether the box was contended.")
    else:
        cv, warm = st
        verdict = ("the box looks idle" if cv < 0.05 else
                   "SOMETHING ELSE WAS RUNNING; treat these timings as upper bounds")
        print(f"\n  Per-group spread {cv:.1%} (group 0 warm-up {warm:.2f}x): {verdict}.")
        print("  The groups are the same computation repeated, so on a quiet box they land within "
              "a percent or two of each other and background load shows up as scatter.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("deploy", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True)
        s.add_argument("--artifacts", type=Path)
        s.add_argument("--seed", type=int, default=0)
        if name == "deploy":
            s.add_argument("--target", type=float, default=0.20)
            s.add_argument("--n-groups", type=int, default=DEFAULT_GROUPS,
                           help=f"teacher groups; the student's bank follows at "
                                f"{KERNELS_PER_GROUP} kernels each (default {DEFAULT_GROUPS}, "
                                f"which is 4x cheaper than 40 for -0.0033 routed)")
            s.add_argument("--n-kernels", type=int,
                           help="override the student's bank size. Must equal n_groups x "
                                f"{KERNELS_PER_GROUP}; it exists to make that explicit, not to "
                                "let the two drift apart.")
            s.add_argument("--folds", type=int, default=5)
        else:
            s.add_argument("--batch", type=int, default=128)
            s.add_argument("--n-groups", type=int,
                           help="override the deployed group count (default: whatever deploy "
                                "recorded)")
            s.add_argument("--compare", action="store_true",
                           help="also run the teacher on every row, on this box at this moment, so the cost ratio is measured rather than assembled from runs on different hardware")
            s.add_argument("--shell", type=Path, default=built_shell())
            s.add_argument("--memory-limit",
                           help="DuckDB memory_limit for the teacher call, e.g. '20GB'. Defaults "
                                "to phase5's own cgroup-aware budget.")
    args = ap.parse_args()
    art = args.artifacts or (ROOT / "data" / "serve" / args.dataset)

    if args.cmd == "deploy":
        deploy(args.dataset, args.target, args.n_groups, args.seed, args.folds, art,
               args.n_kernels)
        return 0
    return serve(args.dataset, art, args.batch, args.n_groups, args.seed, args.shell,
                 ROOT / "data" / "serve" / args.dataset / "work", args.compare,
                 args.memory_limit)


if __name__ == "__main__":
    sys.exit(main())
