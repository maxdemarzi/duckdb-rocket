"""Is the teacher good enough to teach? The gate from docs/DISTILLATION_PLAN.md, third design.

Distillation only earns its place if a student trained on teacher-pseudo-labelled unlabelled data
beats what you would have done with the labels you already had. That needs the teacher to be *better
than the student* -- otherwise its pseudo-labels are no better than the student's own guesses.

**Two earlier designs of this gate were wrong, in different ways, and both are worth stating because
the shape of the mistake recurs.**

*First design: it measured `C - A`* -- fit on context+pool with real labels, against fit on context
alone. That is how much room exists, and says nothing about whether the teacher can reach any of it.
On the saturated subset `C - A` was under a point and the gate said stop, correctly but by luck; on
the hard datasets `C - A` is +0.09 and the same rule would have said go while the teacher was barely
ahead of the student.

*Second design: it measured `T - A` on half a test set, against the better of two students.* Three
faults compounded:

* **Six datasets.** The same sample size that produced a feature "shortlist" indistinguishable from
  noise elsewhere in this project.
* **Halving the test set.** Herring's holdout was 32 rows, so one row is 3.1 accuracy points and its
  -0.1250 against MultiRocketHydra was four rows. The teacher's own holdout accuracy differed from
  its full-test accuracy by up to 5 points -- InlineSkate 0.4400 against 0.4909 -- pure noise.
* **A max over two students.** The max of two noisy estimates exceeds either one's expectation, so
  reducing the baseline to "the better learner" is biased in the baseline's favour by construction.

Decomposed, the second design's verdict was almost entirely those artifacts:

    vs ridge      4/6 wins, mean +0.0115
    vs mr-hydra   3/6 wins, mean -0.0129
    vs max-of-two 2/6 wins, mean -0.0281   <- reported as "the teacher is behind on 4 of 6"

On the *same datasets and models* scored on full test sets, the sign on mr-hydra flips: 6/6 and
+0.0328 against ridge, 3/4 and +0.0182 against mr-hydra.

**This design.** The gate question needs no pool at all -- both teacher and student train on the
train split, and both are scored on the whole test set:

    T   the teacher's accuracy on the FULL test split, read from its archived pipeline report
    A   each student, trained on train, scored on the FULL test split
    gate:  T - A, reported PER LEARNER, never as a max

Only arm B -- the actual distillation -- needs unlabelled data, so it keeps a pool/holdout split, and
it takes `--repeats` so per-dataset split noise is averaged rather than believed.

**What the wide gate found, and why arm B runs where it does.** Over 67 archived teachers the teacher
is level with both students (+0.0085 and +0.0019, against a detectable shift of 0.0140) because 28 of
them have a student at 0.95 or better and no room to move. Below 0.90 the teacher leads ridge on 21 of
29 (+0.0294, p=0.0125) and below 0.75 on 11 of 11 (+0.0572, p=0.0010). So the gate opens on a
subgroup, and arm B is asked the same question on the same subgroup rather than on the archive:

    uv run python scripts/distill_gate.py --gate                       # T vs A, all archived teachers
    uv run python scripts/distill_gate.py --gate --learners rocket+ridge
    uv run python scripts/distill_gate.py --arm-b --from-gate reference/distill_gate.json \\
        --max-student 0.90 --repeats 3 --jobs 7
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

from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402

ALPHAS = np.logspace(-3, 3, 10)


def rocket_ridge(xtr, ytr, xte, n_kernels: int = 10_000, seed: int = 0):
    """The pipeline's own feature family, classified by ridge instead of an in-context model."""
    nch = xtr.shape[1] if xtr.ndim == 3 else 1
    bank = generate_kernels(seed, xtr.shape[-1], n_kernels, n_channels=nch)
    ftr, fte = transform(xtr, bank), transform(xte, bank)
    sc = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(ftr), ytr)
    return clf.predict(sc.transform(fte))


def mr_hydra(xtr, ytr, xte, seed: int = 0):
    """aeon's MultiRocketHydra -- the intended CPU student, and the stronger label-only baseline."""
    from aeon.classification.convolution_based import MultiRocketHydraClassifier

    a = xtr[:, None, :] if xtr.ndim == 2 else xtr
    b = xte[:, None, :] if xte.ndim == 2 else xte
    return MultiRocketHydraClassifier(random_state=seed).fit(a, ytr).predict(b)


LEARNERS = {"rocket+ridge": rocket_ridge, "mr-hydra": mr_hydra}



def _student_cache_path(cache: Path, name: str, learner: str, seed: int) -> Path:
    return cache / f"{name}__{learner.replace('+', '_')}__seed{seed}.json"


def student_accuracy(name: str, learner: str, seed: int, cache: Path | None) -> float | None:
    """One student's accuracy on the full test split, cached per (dataset, learner, seed).

    Cached because the gate is re-run every time another teacher report lands, and the student side
    has not changed. The key includes the seed so a different bank is never served from cache.
    """
    if cache is not None:
        cp = _student_cache_path(cache, name, learner, seed)
        if cp.exists():
            try:
                return float(json.loads(cp.read_text(encoding="utf-8"))["accuracy"])
            except Exception:  # noqa: BLE001
                pass
    try:
        xtr, ytr = load(name, "train")
        xte, yte = load(name, "test")
    except Exception:  # noqa: BLE001
        return None
    xtr, xte = normalize_series(xtr), normalize_series(xte)
    acc = float((LEARNERS[learner](xtr, ytr, xte, seed=seed) == yte).mean())
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        _student_cache_path(cache, name, learner, seed).write_text(
            json.dumps({"dataset": name, "learner": learner, "seed": seed, "accuracy": acc}),
            encoding="utf-8")
    return acc


def _student_worker(t):
    name, learner, seed, cache = t
    try:
        return name, learner, student_accuracy(name, learner, seed, Path(cache) if cache else None)
    except Exception:  # noqa: BLE001
        return name, learner, None

def teacher_reports(directory: Path) -> dict[str, dict]:
    """Archived pipeline reports, keyed by dataset: the teacher's FULL-test accuracy.

    A run that recorded failures is skipped rather than used -- an accuracy computed over a broken row
    alignment is not the teacher's accuracy, and that is exactly the class of number this project has
    had to retract before.
    """
    out: dict[str, dict] = {}
    for p in sorted(directory.glob("phase5_*.json")):
        if p.name.endswith("_soft.json") or "_both" in p.name:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "accuracy" not in d or "shape" not in d:
            continue
        if d.get("failures"):
            print(f"  skipping {d['dataset']}: its run recorded {len(d['failures'])} failure(s)")
            continue
        out.setdefault(d["dataset"], d)
    return out


def load_soft(directory: Path, dataset: str) -> dict | None:
    # _cpu_soft too: the sweep archives by recorded device, so a dataset scored on CPU is filed
    # under _cpu and would otherwise be invisible to arm B.
    for stem in (f"phase5_{dataset}_gpu_soft.json", f"phase5_{dataset}_cpu_soft.json",
                 f"phase5_{dataset}_soft.json"):
        p = directory / stem
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def teacher_label_conf(soft: dict, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Teacher argmax and its confidence per test row, in the dataset's own test order.

    The pipeline lays its ids out as arange(n_train + n_test) with train first, so test row k is id
    n_train + k. The sidecar records n_train so that offset is asserted rather than rediscovered.

    Confidence is the winning posterior, renormalised. Arm Bc uses it to keep only the teacher's most
    confident pseudo-labels, which is the standard answer to hard-label distillation diluting the
    training set with the teacher's own mistakes.
    """
    off, mean_p = soft["n_train"], soft["mean_proba"]
    if soft["n_test"] != n_test:
        raise ValueError(f"teacher ran on {soft['n_test']} test rows, the loader gives {n_test}")
    lab, conf = [], []
    for k in range(n_test):
        row = mean_p.get(str(off + k))
        if row is None:
            raise ValueError(f"teacher has no probabilities for test row {k} (id {off + k})")
        best = max(row, key=row.get)
        lab.append(best)
        conf.append(row[best] / (sum(row.values()) or 1.0))
    return np.asarray(lab), np.asarray(conf, dtype=float)


def teacher_labels(soft: dict, n_test: int) -> np.ndarray:
    return teacher_label_conf(soft, n_test)[0]


def sign_test(diffs: np.ndarray) -> float:
    """Two-sided sign test on the paired differences: P(this many wins or more, if it were a coin).

    Distribution-free on purpose. Accuracy differences across datasets are not normal, not equally
    variable -- a 30-row test set and a 4500-row one are not comparable draws -- and the question is
    only "does the teacher win more often than not".
    """
    from math import comb

    nz = diffs[diffs != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    w = int((nz > 0).sum())
    k = max(w, n - w)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def run_gate(args) -> int:
    reports = teacher_reports(args.teacher)
    names = [n for n in (args.datasets or sorted(reports)) if n in reports]
    if not names:
        print(f"no archived teacher reports in {args.teacher}")
        return 1

    learners = {k: v for k, v in LEARNERS.items() if k in args.learners}
    print(f"gate: teacher vs student on the FULL test split, {len(names)} datasets, "
          f"{len(learners)} learner(s)\n")
    print(f"{'dataset':24s} {'n_test':>6s} {'teacher':>8s} "
          + " ".join(f"{k:>13s}" for k in learners) + "   T-A per learner")

    jobs = [(n, l, args.seed, str(args.cache) if args.cache else None)
            for n in names for l in learners]
    results: dict[tuple[str, str], float] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [ex.submit(_student_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            n, l, acc = fut.result()
            if acc is not None:
                results[(n, l)] = acc
            if i % 10 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} student fits done", flush=True)

    rows: list[dict] = []
    for name in names:
        rep = reports[name]
        T, n_test = rep["accuracy"], rep["shape"]["n_test"]
        accs = {l: results[(name, l)] for l in learners if (name, l) in results}
        if not accs:
            continue
        deltas = {l: T - a for l, a in accs.items()}
        print(f"{name:24s} {n_test:6d} {T:8.4f} "
              + " ".join(f"{accs.get(k, float('nan')):13.4f}" for k in learners)
              + "   " + "  ".join(f"{k}: {v:+.4f}" for k, v in deltas.items()))
        rows.append({"dataset": name, "n_test": n_test, "teacher": T,
                     "students": accs, "delta": deltas})

    if not rows:
        return 1

    print(f"\nTHE GATE -- per learner, no max over learners, full test sets:")
    verdicts = {}
    for lname in learners:
        d = np.array([r["delta"][lname] for r in rows if lname in r["delta"]])
        if not len(d):
            continue
        p = sign_test(d)
        verdicts[lname] = (len(d), int((d > 0).sum()), float(d.mean()), p)
        print(f"  vs {lname:14s} {int((d > 0).sum()):3d}/{len(d)} wins   mean {d.mean():+.4f}   "
              f"median {float(np.median(d)):+.4f}   sign test p = {p:.4f}")

    # The decision is per learner because a student the teacher cannot beat is a student not worth
    # distilling INTO -- it is not evidence about the others, and collapsing them with a max was the
    # previous design's central error.
    print()
    for lname, (n, wins, mean, p) in verdicts.items():
        if mean > 0 and p < 0.05:
            print(f"  {lname}: GO -- teacher ahead by {mean:+.4f} over {n} datasets, p={p:.4f}")
        elif mean > 0:
            print(f"  {lname}: teacher ahead by {mean:+.4f} but p={p:.4f} over {n} datasets; "
                  f"underpowered, not a negative")
        else:
            print(f"  {lname}: STOP -- teacher behind by {mean:+.4f} over {n} datasets, p={p:.4f}")

    # Power, stated rather than assumed, because "no significant difference" at n=6 means nothing.
    n = len(rows)
    if n:
        sd = float(np.std([r["delta"][list(verdicts)[0]] for r in rows], ddof=1)) if n > 1 else 0.0
        det = 2.8 * sd / max(1, n) ** 0.5
        print(f"\n  n={n}, sd of the paired difference {sd:.4f}, so this can detect a mean shift of "
              f"about {det:.4f} at 80% power. A null result below that size is not evidence.")

    if args.out:
        args.out.write_text(json.dumps(
            {"design": "T vs A, full test split, per learner",
             "n_datasets": len(rows), "rows": rows,
             "verdicts": {k: {"n": v[0], "wins": v[1], "mean": v[2], "sign_p": v[3]}
                          for k, v in verdicts.items()}}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


ARMS = ("A", "B", "C", "Bc", "Bs")
ARM_DOC = {"A": "train labels only", "B": "train + teacher-labelled pool",
           "C": "train + truly-labelled pool (the ceiling)",
           "Bc": "train + the most confident half of the teacher-labelled pool",
           "Bs": "train + the pool under the teacher's full distribution (soft targets, ridge only)"}


def noise_rate(arm: str) -> float | None:
    """`N20` -> 0.20. The break-even sweep: the pool with TRUE labels corrupted at a known rate.

    Arm B answers "does this teacher work". These answer the question that governs the whole family
    of pseudo-labelling ideas -- ensembles, better teachers, anything -- which is *how accurate would
    a labeller have to be*. Once the crossing point is known, every candidate labeller is judged by
    one comparison against a number already measured, instead of by another run of arm B.

    N00 is not offered: it is arm C, which is already cached on the same split.
    """
    if len(arm) == 3 and arm[0] == "N" and arm[1:].isdigit():
        return int(arm[1:]) / 100.0
    return None


def known_arm(arm: str) -> bool:
    return arm in ARM_DOC or noise_rate(arm) is not None


def teacher_proba(soft: dict, n_test: int) -> tuple[list[str], np.ndarray]:
    """The teacher's full distribution over the test split, as (classes, n_test x n_classes)."""
    classes = list(soft["classes"])
    off, mean_p = soft["n_train"], soft["mean_proba"]
    if soft["n_test"] != n_test:
        raise ValueError(f"teacher ran on {soft['n_test']} test rows, the loader gives {n_test}")
    out = np.zeros((n_test, len(classes)), dtype=float)
    for k in range(n_test):
        row = mean_p.get(str(off + k))
        if row is None:
            raise ValueError(f"teacher has no probabilities for test row {k} (id {off + k})")
        tot = sum(row.values()) or 1.0
        for j, c in enumerate(classes):
            out[k, j] = row.get(c, 0.0) / tot
    return classes, out


def soft_target_ridge(xtr, ytr, xpool, ppool, classes, xte, n_kernels: int = 10_000, seed: int = 0):
    """Distillation with the textbook soft-target loss: ridge REGRESSION onto the teacher's
    probability vectors, argmax at prediction time.

    Hard argmax discards everything the teacher knew about the rows it was unsure of, which on a
    dataset where it is 60% accurate is most of what it knew. Regressing on the distribution keeps it,
    and a wrong-but-hedged pseudo-label then costs the fit far less than a wrong-but-confident one.
    Without this arm, a negative arm B bounds hard-label distillation and not distillation.
    """
    from sklearn.linear_model import RidgeCV

    nch = xtr.shape[1] if xtr.ndim == 3 else 1
    bank = generate_kernels(seed, xtr.shape[-1], n_kernels, n_channels=nch)
    ftr = transform(np.concatenate([xtr, xpool]), bank)
    sc = StandardScaler().fit(ftr)
    idx = {c: i for i, c in enumerate(classes)}
    missing = sorted({y for y in ytr} - set(idx))
    if missing:
        raise ValueError(f"train labels {missing} are absent from the teacher's class list")
    targets = np.zeros((len(ytr) + len(ppool), len(classes)), dtype=float)
    targets[np.arange(len(ytr)), [idx[y] for y in ytr]] = 1.0
    targets[len(ytr):] = ppool
    reg = RidgeCV(alphas=ALPHAS).fit(sc.transform(ftr), targets)
    return np.asarray(classes)[reg.predict(sc.transform(transform(xte, bank))).argmax(1)]


def pool_holdout(yte: np.ndarray, seed: int, rep: int):
    """The split, deterministic in (dataset, seed, rep) -- which is what makes the cache safe.

    Stratified so a small holdout still contains every class; unstratified only when some class has
    a single member and stratification is impossible.
    """
    idx = np.arange(len(yte))
    try:
        return train_test_split(idx, test_size=0.5, random_state=seed + rep, stratify=yte)
    except ValueError:
        return train_test_split(idx, test_size=0.5, random_state=seed + rep)


def _armb_cache_path(cache: Path, name: str, learner: str, seed: int, rep: int) -> Path:
    return cache / f"{name}__{learner.replace('+', '_')}__seed{seed}__r{rep}.json"


def arm_split(name: str, learner: str, seed: int, rep: int, arms: tuple[str, ...],
              teacher_dir: str, cache: str | None) -> dict[str, float]:
    """Every requested arm on ONE pool/holdout split, cached per split and merged per arm.

    Cached per arm rather than per split so that adding an arm later -- or another repeat -- costs
    only the new fits. The split is a pure function of (dataset, seed, rep), so a cached A from an
    earlier run is the same A the new arms are being compared against.
    """
    cp = _armb_cache_path(Path(cache), name, learner, seed, rep) if cache else None
    have: dict[str, float] = {}
    if cp is not None and cp.exists():
        try:
            have = dict(json.loads(cp.read_text(encoding="utf-8"))["arms"])
        except Exception:  # noqa: BLE001
            have = {}
    todo = [a for a in arms if a not in have]
    if not todo:
        return have

    soft = load_soft(Path(teacher_dir), name)
    if soft is None:
        raise ValueError("no soft-label sidecar")
    xtr, ytr = load(name, "train")
    xte, yte = load(name, "test")
    xtr, xte = normalize_series(xtr), normalize_series(xte)
    tlab, tconf = teacher_label_conf(soft, len(yte))
    pool_i, hold_i = pool_holdout(yte, seed, rep)
    hx, hy = xte[hold_i], yte[hold_i]
    fn = LEARNERS[learner]

    def score(x, y) -> float:
        return float((fn(x, y, hx, seed=seed) == hy).mean())

    out = dict(have)
    for a in todo:
        e = noise_rate(a)
        if a == "A":
            out["A"] = score(xtr, ytr)
        elif a == "B":
            out["B"] = score(np.concatenate([xtr, xte[pool_i]]),
                             np.concatenate([ytr, tlab[pool_i]]))
        elif a == "C":
            out["C"] = score(np.concatenate([xtr, xte[pool_i]]),
                             np.concatenate([ytr, yte[pool_i]]))
        elif a == "Bc":
            order = np.argsort(-tconf[pool_i], kind="stable")
            keep = pool_i[order[: max(1, len(pool_i) // 2)]]
            out["Bc"] = score(np.concatenate([xtr, xte[keep]]),
                              np.concatenate([ytr, tlab[keep]]))
        elif a == "Bs":
            if learner != "rocket+ridge":
                continue  # soft targets need a regressor; aeon's classifier takes hard labels only
            classes, proba = teacher_proba(soft, len(yte))
            pred = soft_target_ridge(xtr, ytr, xte[pool_i], proba[pool_i], classes, hx, seed=seed)
            out["Bs"] = float((pred == hy).mean())
        elif e is not None:
            # Seeded on (seed, rep, rate) so each rate is an independent corruption of the same
            # split, and so a cached rate is reproducible on its own rather than only as part of a
            # sweep run in one particular order.
            rng = np.random.default_rng([seed, rep, int(round(e * 100))])
            ylab = yte[pool_i].copy()
            classes = np.unique(np.concatenate([ytr, yte]))
            for i in np.nonzero(rng.random(len(ylab)) < e)[0]:
                alt = classes[classes != ylab[i]]
                if len(alt):
                    ylab[i] = rng.choice(alt)
            out[a] = score(np.concatenate([xtr, xte[pool_i]]), np.concatenate([ytr, ylab]))
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"dataset": name, "learner": learner, "seed": seed, "repeat": rep,
                                  "n_pool": int(len(pool_i)), "n_hold": int(len(hold_i)),
                                  "arms": out}), encoding="utf-8")
    return out


def _armb_worker(t):
    name, learner, seed, rep, arms, teacher_dir, cache = t
    try:
        return name, learner, rep, arm_split(name, learner, seed, rep, arms, teacher_dir, cache), ""
    except Exception as e:  # noqa: BLE001
        return name, learner, rep, None, f"{type(e).__name__}: {e}"[:120]


def break_even(points: list[tuple[float, float]]) -> float | None:
    """Where the gain from the pool crosses zero, linearly interpolated between swept noise rates.

    `points` is [(label error rate, mean gain over arm A)] ascending in error, starting at 0.0 where
    the gain is the true-label headroom. Returns None when there is no crossing inside the range --
    either the pool never paid at all, or it still pays at the highest rate swept, and calling either
    of those a break-even would invent a number the sweep did not measure.
    """
    for (e0, d0), (e1, d1) in zip(points, points[1:]):
        if d0 > 0 >= d1:
            return e0 + (e1 - e0) * d0 / (d0 - d1)
    return None


def gate_selection(path: Path, max_student: float) -> list[str]:
    """Datasets from a gate run whose best student is below `max_student`.

    The subgroup is read off the gate's own output rather than pasted in, so what was tested is
    recoverable from the command line. docs/DISTILLATION_PLAN.md argues for this cut before any of
    these numbers existed: a dataset a label-only student already solves has no headroom to distil
    into, and 28 of the gate's 67 datasets are in that state.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    return [r["dataset"] for r in d["rows"]
            if r.get("students") and max(r["students"].values()) < max_student]


def run_arm_b(args) -> int:
    """Arms A/B/C/Bc on pool/holdout splits -- the only part that needs unlabelled data.

    Averaged over `--repeats` splits: one 50/50 split of a small test set is a very noisy estimate,
    and believing a single one is what broke the second design of the gate.
    """
    reports = teacher_reports(args.teacher)
    if args.from_gate:
        wanted = gate_selection(args.from_gate, args.max_student)
        print(f"subgroup from {args.from_gate}: {len(wanted)} datasets with every student "
              f"below {args.max_student}")
    else:
        wanted = args.datasets or sorted(reports)
    names = [n for n in wanted if n in reports]
    no_soft = [n for n in names if load_soft(args.teacher, n) is None]
    names = [n for n in names if n not in no_soft]
    if no_soft:
        print(f"  {len(no_soft)} skipped for want of a soft-label sidecar: {', '.join(no_soft)}")
    if not names:
        print("nothing to run")
        return 1

    learners = {k: v for k, v in LEARNERS.items() if k in args.learners}
    bad = [a for a in args.arms if not known_arm(a)]
    if bad:
        print(f"unknown arm(s) {bad}; known: {', '.join(ARM_DOC)} and N05/N10/... noise rates")
        return 1
    arms = tuple(dict.fromkeys(args.arms))
    cache = str(args.armb_cache) if args.armb_cache else None
    print(f"\narms {'/'.join(arms)} over {args.repeats} split(s), {len(names)} datasets, "
          f"{len(learners)} learner(s), {args.jobs} worker(s)")
    for a in arms:
        e = noise_rate(a)
        print(f"    {a:3s} " + (ARM_DOC[a] if e is None else
                                f"train + truly-labelled pool corrupted at {e:.0%}"))

    # Repeat-major, biggest-first within a repeat. Two reasons, and the ordering is worth the line:
    # a fan-out's tail is set by its slowest job, so the 760-row datasets must not start last; and
    # finishing every dataset's repeat 0 before starting any repeat 1 means the cache holds a
    # *complete* lower-repeat answer at all times, so a run this long can be reported on -- or
    # interrupted -- at any point rather than only at the end.
    jobs = sorted(((n, l, args.seed, r, arms, str(args.teacher), cache)
                   for n in names for l in learners for r in range(args.repeats)),
                  key=lambda j: (j[3], -reports[j[0]]["shape"]["n_test"]))
    got: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [ex.submit(_armb_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            n, l, rep, res, err = fut.result()
            if res is None:
                print(f"  {n} {l} r{rep} failed: {err}", flush=True)
            else:
                got.setdefault((n, l), {})[rep] = res
            if i % 20 == 0 or i == len(jobs):
                el = (time.perf_counter() - t0) / 60
                print(f"  ... {i}/{len(jobs)} splits, {el:.1f} min elapsed, "
                      f"~{el / i * (len(jobs) - i):.1f} min left", flush=True)

    print(f"\n{'dataset':30s} {'learner':13s} " + " ".join(f"{a:>7s}" for a in arms)
          + "".join(f"{a + '-A':>9s}" for a in arms if a != "A"))
    rows = []
    for name in names:
        for lname in learners:
            per = got.get((name, lname), {})
            if not per:
                continue
            mean = {a: float(np.mean([v[a] for v in per.values() if a in v]))
                    for a in arms if any(a in v for v in per.values())}
            if "A" not in mean:
                continue
            print(f"{name:30s} {lname:13s} " + " ".join(f"{mean.get(a, float('nan')):7.4f}" for a in arms)
                  + "".join(f"{mean[a] - mean['A']:+9.4f}" for a in arms
                            if a != "A" and a in mean))
            rows.append({"dataset": name, "learner": lname, "repeats": len(per),
                         "teacher": reports[name]["accuracy"], "mean": mean,
                         "per_split": {str(k): v for k, v in sorted(per.items())}})

    if not rows:
        return 1

    print("\nDISTILLATION -- paired across datasets, per learner:")
    verdicts: dict[str, dict] = {}
    for lname in learners:
        sub = [r for r in rows if r["learner"] == lname]
        if not sub:
            continue
        for a in arms:
            if a == "A":
                continue
            d = np.array([r["mean"][a] - r["mean"]["A"] for r in sub if a in r["mean"]])
            if not len(d):
                continue
            p = sign_test(d)
            verdicts[f"{lname}:{a}"] = {"n": len(d), "wins": int((d > 0).sum()),
                                        "mean": float(d.mean()),
                                        "median": float(np.median(d)), "sign_p": p}
            print(f"  {lname:13s} {a + '-A':6s} {int((d > 0).sum()):3d}/{len(d)} wins   "
                  f"mean {d.mean():+.4f}   median {float(np.median(d)):+.4f}   p = {p:.4f}")
        d = np.array([r["mean"]["B"] - r["mean"]["A"] for r in sub if "B" in r["mean"]])
        if len(d) > 1:
            print(f"  {lname:13s} {'':6s} sd {float(np.std(d, ddof=1)):.4f}, so n={len(d)} detects "
                  f"a shift of about {2.8 * float(np.std(d, ddof=1)) / len(d) ** 0.5:.4f} at 80% power")

    # The mechanism check. Distillation can only add what the teacher knows and the student does
    # not, so B-A should track the teacher's own edge. If it does not, any aggregate win is luck.
    if "B" in arms and "C" in arms:
        print("\n  by the teacher's edge over the student on the same splits (T - A):")
        for lname in learners:
            sub = [r for r in rows if r["learner"] == lname and "B" in r["mean"]]
            if len(sub) < 4:
                continue
            edge = np.array([r["teacher"] - r["mean"]["A"] for r in sub])
            ba = np.array([r["mean"]["B"] - r["mean"]["A"] for r in sub])
            ca = np.array([r["mean"]["C"] - r["mean"]["A"] for r in sub])
            for label, m in (("teacher ahead", edge > 0), ("teacher behind", edge <= 0)):
                if m.sum():
                    print(f"    {lname:13s} {label:15s} n={int(m.sum()):3d}   "
                          f"B-A {ba[m].mean():+.4f} ({int((ba[m] > 0).sum())} wins)   "
                          f"C-A {ca[m].mean():+.4f}   p = {sign_test(ba[m]):.4f}")

    swept = sorted((noise_rate(a) for a in arms if noise_rate(a) is not None))
    if swept and "C" in arms:
        print("\n  BREAK-EVEN -- how accurate a labeller would have to be for the pool to pay:")
        print(f"    {'dataset':30s} {'learner':13s} {'C-A':>8s} {'e*':>7s} {'teacher err':>12s} "
              f"{'verdict':>9s}")
        be_rows = []
        for r in rows:
            pts = [(0.0, r["mean"]["C"] - r["mean"]["A"])] + [
                (e, r["mean"][f"N{int(round(e * 100)):02d}"] - r["mean"]["A"])
                for e in swept if f"N{int(round(e * 100)):02d}" in r["mean"]]
            if len(pts) < 2:
                continue
            e_star = break_even(pts)
            terr = 1.0 - r["teacher"]
            # Three outcomes, and collapsing the first two would be a reporting error of the exact
            # kind this project has had to retract: a pool that never paid even with TRUE labels is
            # silent about label noise, and calling it "tolerates more than 40%" reads as the
            # opposite of what it is.
            if pts[0][1] <= 0:
                shown, ok, e_star = "n/a", "no headroom", None
            elif e_star is None:
                shown, ok = f">{max(swept):.0%}", "PAYS"
            else:
                shown, ok = f"{e_star:.1%}", ("PAYS" if terr < e_star else "no")
            print(f"    {r['dataset']:30s} {r['learner']:13s} {pts[0][1]:+8.4f} {shown:>7s} "
                  f"{terr:12.1%} {ok:>11s}")
            be_rows.append({"dataset": r["dataset"], "learner": r["learner"],
                            "headroom": pts[0][1], "break_even": e_star, "teacher_error": terr,
                            "outcome": ok})
        crossed = [b for b in be_rows if b["break_even"] is not None]
        if crossed:
            es = np.array([b["break_even"] for b in crossed])
            te = np.array([b["teacher_error"] for b in crossed])
            print(f"\n    median break-even {float(np.median(es)):.1%} against a median teacher error "
                  f"of {float(np.median(te)):.1%}; the teacher is accurate enough on "
                  f"{int((te < es).sum())} of {len(crossed)}")
        never = [b for b in be_rows if b["outcome"] == "PAYS" and b["break_even"] is None]
        dead = [b for b in be_rows if b["outcome"] == "no headroom"]
        if never:
            print(f"    {len(never)} case(s) still paid at {max(swept):.0%} noise, so their "
                  f"break-even is above the swept range")
        if dead:
            print(f"    {len(dead)} case(s) had no headroom even with TRUE labels and say nothing "
                  f"about label quality")
        for b in be_rows:
            verd = verdicts.setdefault("break_even", {})
            verd[f"{b['dataset']}:{b['learner']}"] = b

    print("\nArm B uses the teacher's hard argmax. A soft-label student can exceed it, so a negative "
          "here bounds hard-label distillation and not distillation. C is the same pool with real "
          "labels: where C-A is itself near zero, no labelling of the pool could have helped and the "
          "dataset says nothing about the teacher.")
    if args.out:
        args.out.write_text(json.dumps(
            {"design": "arms A/B/C/Bc on 50/50 pool/holdout, averaged over repeats",
             "arms": {a: ARM_DOC[a] for a in arms}, "repeats": args.repeats, "seed": args.seed,
             "selection": (f"best student < {args.max_student} in {args.from_gate}"
                           if args.from_gate else "explicit"),
             "n_datasets": len({r['dataset'] for r in rows}), "verdicts": verdicts, "rows": rows},
            indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="T vs A on full test splits (the gate)")
    ap.add_argument("--arm-b", action="store_true", help="arms A/B/C on pool/holdout splits")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--teacher", type=Path, default=ROOT / "reference",
                    help="directory of archived pipeline reports and soft-label sidecars")
    ap.add_argument("--learners", nargs="*", default=list(LEARNERS))
    ap.add_argument("--repeats", type=int, default=5, help="pool/holdout splits to average (arm B)")
    ap.add_argument("--arms", nargs="*", default=["A", "B", "C"],
                    help=f"which arms to score: {', '.join(f'{k} = {v}' for k, v in ARM_DOC.items())}"
                         "; plus N05/N10/N20/... = the pool with TRUE labels corrupted at that rate, "
                         "which with C measures how accurate any labeller would have to be")
    ap.add_argument("--from-gate", type=Path,
                    help="select datasets from a gate run instead of naming them, so the subgroup "
                         "under test is recoverable from the command line")
    ap.add_argument("--max-student", type=float, default=0.90,
                    help="with --from-gate, keep datasets whose BEST student is below this; a "
                         "dataset a label-only student already solves has no headroom to distil into")
    ap.add_argument("--armb-cache", type=Path, default=ROOT / "data" / "armb_splits",
                    help="per-(dataset, learner, seed, repeat) arm scores, merged per arm, so "
                         "adding an arm or a repeat later costs only the new fits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel student fits; the gate is embarrassingly parallel across datasets")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "gate_students",
                    help="per-(dataset, learner, seed) student accuracies, so re-running the "
                         "gate as more teacher reports land costs nothing already computed")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if not (args.gate or args.arm_b):
        args.gate = True
    t0 = time.perf_counter()
    rc = run_gate(args) if args.gate else 0
    if args.arm_b:
        rc = run_arm_b(args) or rc
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
