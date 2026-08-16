"""The two knobs that set what routing costs, measured rather than assumed.

Routing pays for two things per served batch. Every row goes through the student -- a ROCKET
transform of `num_kernels` kernels and one matrix multiply -- and the escalated fraction goes on to
the teacher, which is `n_groups` separate in-context calls. Both defaults were inherited from the
accuracy work and neither was ever chosen for cost:

* **the teacher's group count, G = 40.** Cost is linear in G by construction: each group is its own
  `tabfm_classify` pass over the same rows. If G = 10 answers as well, three quarters of the
  expensive path is waste.
* **the student's kernel count, 10,000** -- 20,000 features for every row, on the cheap path that
  every row takes. Routing does not need the student's best accuracy, it needs its ORDERING to be
  right, and those are not the same requirement.

    uv run python scripts/perf_levers.py --groups  --pergroup data/pergroup --out out.json
    uv run python scripts/perf_levers.py --kernels --from-gate reference/distill_gate.json

**Why the group sweep costs one run and not four.** Group g covers kernel indices
[250g, 250(g+1)) whenever kernels_per_group is 250, and the prediction is the argmax of the MEAN of
the groups' probabilities. So a G-group run reads exactly groups 0..G-1 of the bank the 40-group run
reads, and averaging the first G groups of one archived run reproduces it -- exactly, not
approximately. `phase5_pipeline.py --per-group-soft` writes the cube that makes this possible.

The trap that makes this exact rather than sloppy: `kernels_per_group = num_kernels // n_groups`, so
the run being reproduced is `--n-groups 10 --num-kernels 2500`, NOT `--n-groups 10` at the default
10,000 kernels. The latter makes each group 1000 kernels -- 2000 features against tabicl's 512 cap
-- and is a different experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402

from distill_gate import (  # noqa: E402
    gate_selection, load_soft, route_curve, rocket_ridge_scored, sign_test, teacher_labels,
    teacher_reports,
)

#: Fixed escalation budgets, in the range the product question lives in. Not the peak of the curve:
#: the peak is chosen on the same split it is read off, which is how an "oracle" number got quoted
#: as a shippable one once already.
BUDGETS = (0.10, 0.20, 0.30)

KERNEL_GRID = (250, 500, 1_000, 2_000, 5_000, 10_000)


# ---------------------------------------------------------------------------- the teacher's groups

def prefix_predictions(cube: np.ndarray, classes: list[str]) -> list[np.ndarray]:
    """Predictions after averaging the first G groups, for every G from 1 to n_groups.

    Cumulative sum rather than n_groups separate means: the same arithmetic, but it makes it obvious
    that group order is the only thing this depends on, and group order is the kernel-bank order.
    """
    running = np.cumsum(cube, axis=0)
    cls = np.asarray(classes, dtype=object)
    return [cls[np.argmax(running[g], axis=1)] for g in range(cube.shape[0])]


def load_pergroup(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d["proba"] = np.asarray(d["proba"], dtype=np.float64)
    if d["proba"].shape[0] != d["n_groups"]:
        raise ValueError(f"{path.name}: {d['proba'].shape[0]} groups of probabilities but "
                         f"n_groups={d['n_groups']}")
    return d


def run_groups(args) -> int:
    """Teacher accuracy, and routing gain, as a function of how many groups the teacher runs."""
    files = sorted(Path(args.pergroup).glob("*_pergroup.json"))
    if not files:
        print(f"no *_pergroup.json under {args.pergroup}; run phase5_pipeline.py --per-group-soft")
        return 1
    rows = []
    for f in files:
        d = load_pergroup(f)
        name = d["dataset"]
        try:
            _, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: cannot load test split ({type(e).__name__}); skipped")
            continue
        n_train, n_test = d["n_train"], d["n_test"]
        # Test row k is id n_train + k. Asserted, not assumed -- the offset the sidecar records
        # precisely because a consumer that rediscovers it eventually gets it wrong.
        want = [n_train + k for k in range(n_test)]
        if d["ids"] != want:
            raise ValueError(f"{f.name}: ids are not arange({n_train}, {n_train + n_test})")
        if len(yte) != n_test:
            raise ValueError(f"{f.name}: {n_test} rows archived but the test split has {len(yte)}")

        truth = np.asarray(yte, dtype=object)
        preds = prefix_predictions(d["proba"], d["classes"])
        tacc = [float((p == truth).mean()) for p in preds]
        # Check the cube against ITS OWN run's report -- the sibling file, not `reference/`. Both
        # numbers are then the same argmax over the same rows of the same execution, so a mismatch
        # is a bug in the export rather than device drift, and 1e-9 is the right tolerance.
        # Comparing against an archived run instead would fold in a cpu-vs-cuda difference and make
        # a real export bug look like acceptable noise.
        sib = f.with_name(f.name.replace("_pergroup.json", ".json"))
        if sib.exists():
            ref = json.loads(sib.read_text(encoding="utf-8")).get("accuracy")
            if ref is not None and abs(tacc[-1] - float(ref)) > 1e-9:
                print(f"  {name}: the cube averages to {tacc[-1]:.4f} but its own run reported "
                      f"{float(ref):.4f}; SKIPPED")
                continue

        # The student's ordering is fixed across G -- it never sees the teacher -- so one scoring
        # serves the whole sweep.
        cache = Path(args.route_cache) / f"{name}__rocket_ridge__seed{args.seed}.json"
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            spred, sconf = np.asarray(c["pred"], dtype=object), np.asarray(c["conf"], dtype=float)
        else:
            xtr, ytr = load(name, "train")
            xte, _ = load(name, "test")
            sp, sc = rocket_ridge_scored(normalize_series(xtr), ytr, normalize_series(xte),
                                         seed=args.seed)
            spred, sconf = np.asarray([str(v) for v in sp], dtype=object), np.asarray(sc)

        routed = {}
        for b in BUDGETS:
            routed[b] = [route_curve(spred, sconf, p, yte, [b])[0][1] for p in preds]
        sacc = float((spred == truth).mean())

        rows.append({"dataset": name, "n_test": n_test, "n_groups": d["n_groups"],
                     "kernels_per_group": d["kernels_per_group"], "student": sacc,
                     "teacher_by_g": tacc, "routed_by_g": {str(b): routed[b] for b in BUDGETS}})
        shown = [g for g in (1, 10, len(tacc)) if g <= len(tacc)]
        print(f"  {name:28s} " + "  ".join(f"G={g} {tacc[g - 1]:.4f}" for g in shown), flush=True)

    if not rows:
        return 1
    report_groups(rows)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "design": "prefix averages of one 40-group run; averaging the first G groups is exactly "
                      "what --n-groups G --num-kernels 250*G computes",
            "budgets": list(BUDGETS), "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def report_groups(rows: list[dict]) -> None:
    # gmax is the shortest cube on offer, and it is always the last row: it is the baseline every
    # other row is differenced against, so a grid that stops short of it (1, 2 for a 4-group cube)
    # would print differences against a column that is not shown.
    gmax = min(r["n_groups"] for r in rows)
    grid = sorted({g for g in (1, 2, 5, 10, 20, 40) if g < gmax} | {gmax})
    print(f"\nTEACHER GROUPS -- {len(rows)} datasets, cost is linear in G\n")
    print(f"  {'G':>4s} {'teacher':>9s} {f'vs G={gmax}':>9s} {'p':>7s}   "
          + "  ".join(f"route@{int(b * 100)}%" for b in BUDGETS))
    full = np.array([r["teacher_by_g"][gmax - 1] for r in rows])
    for g in grid:
        acc = np.array([r["teacher_by_g"][g - 1] for r in rows])
        d = acc - full
        cells = []
        for b in BUDGETS:
            rb = np.array([r["routed_by_g"][str(b)][g - 1] for r in rows])
            cells.append(f"{rb.mean():9.4f}")
        p = sign_test(d) if g != gmax else float("nan")
        print(f"  {g:4d} {acc.mean():9.4f} {d.mean():+9.4f} {p:7.4f}   " + "  ".join(cells))

    # The product question: the cheapest G that is not measurably worse. Stated as the first G whose
    # loss against G=40 is both small and not significant, rather than as the first that "looks
    # fine" -- with 28 datasets a 0.005 mean difference is well inside the noise.
    print("\n  the routed columns are what a served system would get: the student everywhere, the")
    print("  teacher at G groups on the least-confident rows only.")


# ------------------------------------------------------------------------- the student's kernels

def _kernel_worker(t):
    name, n_kernels, seed = t
    try:
        xtr, ytr = load(name, "train")
        xte, _ = load(name, "test")
        xtr, xte = normalize_series(xtr), normalize_series(xte)
        t0 = time.perf_counter()
        pred, conf = rocket_ridge_scored(xtr, ytr, xte, n_kernels=n_kernels, seed=seed)
        fit_s = time.perf_counter() - t0
        # Transform time alone, on the test rows: the per-row serving cost, separated from the fit
        # that happens once at deploy.
        bank = generate_kernels(seed, xtr.shape[-1], n_kernels,
                                n_channels=xtr.shape[1] if xtr.ndim == 3 else 1)
        t0 = time.perf_counter()
        transform(xte, bank)
        tx_s = time.perf_counter() - t0
        return name, n_kernels, [str(p) for p in pred], [float(c) for c in conf], fit_s, tx_s, ""
    except Exception as e:  # noqa: BLE001
        return name, n_kernels, None, None, 0.0, 0.0, f"{type(e).__name__}: {e}"[:140]


def run_kernels(args) -> int:
    """Does routing need the student's full kernel bank, or only its ordering?"""
    reports = teacher_reports(args.teacher)
    wanted = (gate_selection(args.from_gate, args.max_student) if args.from_gate
              else (args.datasets or sorted(reports)))
    names = [n for n in wanted if n in reports and load_soft(args.teacher, n) is not None]
    if not names:
        print("no dataset has both a report and a soft-label sidecar")
        return 1
    grid = [int(k) for k in (args.kernel_grid or KERNEL_GRID)]
    print(f"student kernels: {len(names)} datasets x {len(grid)} sizes, teacher fixed at its "
          f"archived 40-group labels\n")

    # Warm the dataset cache SERIALLY before forking. aeon downloads and extracts on first use, and
    # the job list puts one worker per (dataset, size) -- so six workers reach for the same cold
    # archive at once and read each other's half-written files. That is not hypothetical: the first
    # run of this on a fresh pod lost 44 of 168 fits to "Inconsistent number of dimensions in case
    # 33" and "zero-size array to reduction operation maximum", which look like modelling failures
    # and are not. It also left each row of the table averaged over a different subset of datasets.
    missing = []
    for n in names:
        try:
            load(n, "train")
            load(n, "test")
        except Exception as e:  # noqa: BLE001
            missing.append((n, f"{type(e).__name__}: {e}"[:90]))
    for n, err in missing:
        print(f"  {n}: will not load even serially -- {err}")
    names = [n for n in names if n not in {m for m, _ in missing}]
    if not names:
        print("no dataset loads")
        return 1
    print(f"  dataset cache warm for {len(names)} datasets\n")

    jobs = [(n, k, args.seed) for n in names for k in grid]
    got: dict[tuple[str, int], tuple] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_kernel_worker, j) for j in jobs]), 1):
            n, k, pred, conf, fit_s, tx_s, err = fut.result()
            if pred is None:
                print(f"  {n} k={k} failed: {err}", flush=True)
            else:
                got[(n, k)] = (pred, conf, fit_s, tx_s)
            if i % 20 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} fits", flush=True)

    rows = []
    for name in names:
        soft = load_soft(args.teacher, name)
        _, yte = load(name, "test")
        tpred = teacher_labels(soft, len(yte))
        truth = np.asarray(yte, dtype=object)
        for k in grid:
            if (name, k) not in got:
                continue
            pred, conf, fit_s, tx_s = got[(name, k)]
            spred = np.asarray(pred, dtype=object)
            routed = {str(b): route_curve(spred, np.asarray(conf), tpred, yte, [b])[0][1]
                      for b in BUDGETS}
            rows.append({"dataset": name, "n_kernels": k, "n_test": int(len(yte)),
                         "student": float((spred == truth).mean()), "routed": routed,
                         "fit_seconds": fit_s, "transform_seconds": tx_s})

    if not rows:
        return 1
    report_kernels(rows, grid)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"design": "one teacher (40 groups, archived) against students of varying kernel count",
             "budgets": list(BUDGETS), "grid": grid, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def report_kernels(rows: list[dict], grid: list[int]) -> None:
    by = {}
    for r in rows:
        by.setdefault(r["n_kernels"], []).append(r)
    full = {r["dataset"]: r for r in by[grid[-1]]}
    names = sorted(full)
    print(f"\nSTUDENT KERNELS -- {len(names)} datasets at the full bank, teacher unchanged\n")
    # `n` is printed per row and is not decoration. When a fit fails, that row is averaged over a
    # different subset from the others, and a table without this column reads as though every row
    # covered the same datasets.
    print(f"  {'kernels':>8s} {'n':>3s} {'student':>9s} {'vs full':>9s} {'p':>7s}   "
          + "  ".join(f"route@{int(b * 100)}%" for b in BUDGETS) + f"  {'tx ms/row':>10s}")
    for k in grid:
        sub = {r["dataset"]: r for r in by.get(k, [])}
        common = [n for n in names if n in sub]
        s = np.array([sub[n]["student"] for n in common])
        d = s - np.array([full[n]["student"] for n in common])
        cells = [f"{np.mean([sub[n]['routed'][str(b)] for n in common]):9.4f}" for b in BUDGETS]
        ms = np.mean([sub[n]["transform_seconds"] / sub[n]["n_test"] * 1000 for n in common])
        p = sign_test(d) if k != grid[-1] else float("nan")
        print(f"  {k:8d} {len(common):3d} {s.mean():9.4f} {d.mean():+9.4f} {p:7.4f}   "
              + "  ".join(cells) + f"  {ms:10.2f}")
    if len({len([n for n in names if n in {r['dataset'] for r in by.get(k, [])}])
            for k in grid}) > 1:
        print("\n  WARNING: the rows do not cover the same datasets, so the columns are not "
              "directly comparable down the table. The 'vs full' column still is -- it pairs by "
              "dataset name -- but the means are over different subsets.")

    # Routing gain is the thing that must survive, not student accuracy: a smaller bank can lose a
    # little accuracy and still route as well, and that trade is worth taking on the path every row
    # walks. Reported against the same budget at the full bank so the comparison is like for like.
    print("\n  routing gain at each size, against the SAME budget at the full bank:")
    for b in BUDGETS:
        line = []
        for k in grid:
            sub = {r["dataset"]: r for r in by.get(k, [])}
            common = [n for n in names if n in sub]
            # Both sides indexed by the SAME dataset name. Slicing the full-bank array to
            # len(common) instead would silently pair dataset i of one list with dataset i of the
            # other whenever a fit failed at some size, and the failures are exactly the datasets
            # where the two lists stop agreeing.
            d = np.array([sub[n]["routed"][str(b)] - full[n]["routed"][str(b)] for n in common])
            line.append(f"{k}: {d.mean():+.4f} (p={sign_test(d):.3f})" if k != grid[-1]
                        else f"{k}: --")
        print(f"    escalate {b:.0%}   " + "   ".join(line))


# -------------------------------------------------------------------- both levers at once

def _margin_worker(t):
    """Student predictions AND margins at one kernel count. `--kernels` keeps only accuracies."""
    name, n_kernels, seed = t
    try:
        xtr, ytr = load(name, "train")
        xte, _ = load(name, "test")
        pred, conf = rocket_ridge_scored(normalize_series(xtr), ytr, normalize_series(xte),
                                         n_kernels=n_kernels, seed=seed)
        return name, n_kernels, [str(p) for p in pred], [float(c) for c in conf], ""
    except Exception as e:  # noqa: BLE001
        return name, n_kernels, None, None, f"{type(e).__name__}: {e}"[:140]


def run_joint(args) -> int:
    """Do the two levers compose, or was each measured against the other's default?

    The group sweep held the student at 10,000 kernels; the kernel sweep routed against the
    archived 40-group teacher. Neither says what happens when both are cut, which is the
    configuration worth shipping -- so this crosses them directly on the same datasets.
    """
    cubes = {}
    for f in sorted(Path(args.pergroup).glob("*_pergroup.json")):
        d = load_pergroup(f)
        cubes[d["dataset"]] = d
    if not cubes:
        print(f"no cubes under {args.pergroup}")
        return 1
    grid = [int(k) for k in (args.kernel_grid or (500, 2_000, 5_000, 10_000))]
    gs = [int(g) for g in (args.group_grid or (5, 10, 20, 40))]
    names = sorted(cubes)
    print(f"joint: {len(names)} datasets x {len(grid)} kernel sizes x {len(gs)} group counts\n")

    for n in names:  # serial warm, as in run_kernels
        load(n, "train"), load(n, "test")

    jobs = [(n, k, args.seed) for n in names for k in grid]
    students: dict[tuple[str, int], tuple] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_margin_worker, j) for j in jobs]), 1):
            n, k, pred, conf, err = fut.result()
            if pred is None:
                print(f"  {n} k={k} failed: {err}", flush=True)
            else:
                students[(n, k)] = (np.asarray(pred, dtype=object), np.asarray(conf))
            if i % 20 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} student fits", flush=True)

    rows = []
    for n in names:
        _, yte = load(n, "test")
        preds_by_g = prefix_predictions(cubes[n]["proba"], cubes[n]["classes"])
        for k in grid:
            if (n, k) not in students:
                continue
            spred, sconf = students[(n, k)]
            for g in gs:
                if g > len(preds_by_g):
                    continue
                acc = route_curve(spred, sconf, preds_by_g[g - 1], yte, [args.budget])[0][1]
                rows.append({"dataset": n, "n_kernels": k, "groups": g, "routed": acc})
    if not rows:
        return 1
    report_joint(rows, grid, gs, args.budget)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"design": "student kernel count crossed with teacher group count, one budget",
             "budget": args.budget, "grid": grid, "groups": gs, "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def report_joint(rows, grid, gs, budget) -> None:
    by = {(r["dataset"], r["n_kernels"], r["groups"]): r["routed"] for r in rows}
    names = sorted({r["dataset"] for r in rows})
    base = [by.get((n, grid[-1], gs[-1])) for n in names]
    if any(b is None for b in base):
        names = [n for n, b in zip(names, base) if b is not None]
        base = [b for b in base if b is not None]
    base = np.array(base)
    print(f"\nBOTH LEVERS -- routed accuracy at a {budget:.0%} budget, {len(names)} datasets")
    print(f"baseline is {grid[-1]} kernels x G={gs[-1]}: {base.mean():.4f}\n")
    print(f"  {'kernels':>8s} " + "  ".join(f"G={g:<9d}" for g in gs))
    for k in grid:
        cells = []
        for g in gs:
            vals = np.array([by.get((n, k, g), np.nan) for n in names])
            ok = ~np.isnan(vals)
            d = vals[ok] - base[ok]
            cells.append(f"{vals[ok].mean():.4f} ({d.mean():+.4f})")
        print(f"  {k:8d} " + "  ".join(cells))
    # The shippable question: is the cheap corner distinguishable from the expensive one?
    print(f"\n  vs the {grid[-1]}x{gs[-1]} baseline, sign test over datasets:")
    for k in grid:
        for g in gs:
            if (k, g) == (grid[-1], gs[-1]):
                continue
            vals = np.array([by.get((n, k, g), np.nan) for n in names])
            ok = ~np.isnan(vals)
            d = vals[ok] - base[ok]
            if k in (grid[0], grid[-1]) or g in (gs[0], gs[-1]):
                print(f"    {k:5d} kernels x G={g:<3d} {d.mean():+.4f}  p={sign_test(d):.3f}  "
                      f"{int((d >= 0).sum())}/{len(d)} not worse")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", action="store_true", help="teacher group count, from archived cubes")
    ap.add_argument("--kernels", action="store_true", help="student kernel count")
    ap.add_argument("--joint", action="store_true",
                    help="cross the two: the group sweep held kernels at 10,000 and the kernel "
                         "sweep routed against a 40-group teacher, so the cheap corner is unmeasured")
    ap.add_argument("--group-grid", nargs="*", type=int)
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--pergroup", type=Path, default=ROOT / "data" / "pergroup")
    ap.add_argument("--teacher", type=Path, default=ROOT / "reference")
    ap.add_argument("--route-cache", type=Path, default=ROOT / "data" / "route_cache")
    ap.add_argument("--from-gate", type=Path)
    ap.add_argument("--max-student", type=float, default=0.90)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--kernel-grid", nargs="*", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not (args.groups or args.kernels or args.joint):
        ap.error("pick --groups, --kernels or --joint")
    rc = 0
    if args.groups:
        rc |= run_groups(args)
    if args.kernels:
        rc |= run_kernels(args)
    if args.joint:
        rc |= run_joint(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
