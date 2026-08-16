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


def load_soft(directory: Path, dataset: str, model: str | None = None) -> dict | None:
    """One model's soft-label sidecar, or None.

    `model=None` means the original teacher, whose reports predate `--model` and carry no model in
    their name. The archived spelling is preserved rather than migrated: renaming those files would
    orphan reference/distill_gate.json and everything built on it. `_cpu_soft` is included because
    the sweep archives by recorded device, so a dataset scored on CPU is filed under `_cpu` and
    would otherwise be invisible.
    """
    stems = ([f"phase5_{dataset}_{model}_soft.json"] if model and model != "tabicl-v2" else
             [f"phase5_{dataset}_gpu_soft.json", f"phase5_{dataset}_cpu_soft.json",
              f"phase5_{dataset}_soft.json"])
    for stem in stems:
        p = directory / stem
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_ensemble_soft(directory: Path, dataset: str, models: list[str]) -> dict | None:
    """Several labellers averaged into one, or None if any of them is missing this dataset.

    Averaging probabilities rather than voting on argmax, for the same reason arm Bs exists: on a
    dataset where each model is 60% accurate, most of what they collectively know is in how they
    hedge, and a majority vote throws it away before the ensemble can use it.

    Returning None when one model is absent is deliberate. Silently averaging whichever subset
    happened to have run would make the ensemble's membership vary by dataset, so a per-dataset
    accuracy would not be comparable to any other and the aggregate would describe no fixed system.
    """
    parts = []
    for m in models:
        s = load_soft(directory, dataset, m)
        if s is None:
            return None
        parts.append(s)
    base = parts[0]
    classes = list(base["classes"])
    for s in parts[1:]:
        if list(s["classes"]) != classes:
            raise ValueError(f"{dataset}: {s.get('model')} has classes {s['classes']}, "
                             f"{base.get('model')} has {classes}")
        if s["n_train"] != base["n_train"] or s["n_test"] != base["n_test"]:
            raise ValueError(f"{dataset}: {s.get('model')} ran on a different split")
    merged: dict[str, dict[str, float]] = {}
    for key in base["mean_proba"]:
        rows = []
        for s in parts:
            row = s["mean_proba"].get(key)
            if row is None:
                raise ValueError(f"{dataset}: {s.get('model')} has no row {key}")
            tot = sum(row.values()) or 1.0
            rows.append({c: row.get(c, 0.0) / tot for c in classes})
        merged[key] = {c: sum(r[c] for r in rows) / len(rows) for c in classes}
    return {"dataset": dataset, "model": "+".join(models), "n_train": base["n_train"],
            "n_test": base["n_test"], "classes": classes, "mean_proba": merged}


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


def arm_doc(arm: str) -> str:
    """One description per arm, for printing and for the report.

    A function rather than `ARM_DOC[arm]` because the noise arms are generated from their names and
    have no dict entry, and indexing the dict for them crashed the run that produced the whole
    break-even table -- after the analysis had printed, while writing the JSON.
    """
    e = noise_rate(arm)
    return ARM_DOC[arm] if e is None else f"train + truly-labelled pool corrupted at {e:.0%}"


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


def noise_curve_at(curve: list[tuple[float, float]], e: float) -> tuple[float, bool]:
    """The gain a pool would give at label-error rate `e`, read off the measured curve.

    Interpolating rather than snapping to the nearest swept rate, because snapping is not neutral:
    the swept rates sit BELOW the teacher's error on nine of ten datasets here, so the nearest arm
    is a less corrupted pool than the teacher's, and every comparison against it would flatter the
    random-noise side. Returns whether the answer required extrapolating past the last rate swept.
    """
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= e <= x1:
            return (y0 if x1 == x0 else y0 + (y1 - y0) * (e - x0) / (x1 - x0)), False
    (x0, y0), (x1, y1) = curve[-2], curve[-1]
    return (y1 + (y1 - y0) * (e - x1) / (x1 - x0)), True


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
        print(f"    {a:3s} {arm_doc(a)}")

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

        # The question the break-even alone cannot answer, and the one that turned out to matter.
        # If a pool tolerates 25% random noise and the teacher errs at 22%, arm B should have paid.
        # It did not, so the RATE is not what governs -- and the comparison below is the test:
        # the teacher's own labels against a random corruption of the TRUE labels at the teacher's
        # exact error rate, on the same split, same pool rows, same student.
        if "B" in arms:
            print("\n  STRUCTURED vs RANDOM error, matched on rate:")
            print(f"    {'dataset':26s} {'learner':13s} {'T err':>7s} {'B-A':>8s} "
                  f"{'random@Terr':>12s} {'gap':>8s}")
            gaps, exact = [], []
            for r in rows:
                if "B" not in r["mean"] or "C" not in r["mean"]:
                    continue
                have = [e for e in swept if f"N{int(round(e * 100)):02d}" in r["mean"]]
                if len(have) < 2:
                    continue
                curve = [(0.0, r["mean"]["C"] - r["mean"]["A"])] + [
                    (e, r["mean"][f"N{int(round(e * 100)):02d}"] - r["mean"]["A"]) for e in have]
                terr = 1.0 - r["teacher"]
                ny, extrapolated = noise_curve_at(curve, terr)
                gap = ny - (r["mean"]["B"] - r["mean"]["A"])
                gaps.append(gap)
                if not extrapolated:
                    exact.append(gap)
                print(f"    {r['dataset']:26s} {r['learner']:13s} {terr:7.1%} "
                      f"{r['mean']['B'] - r['mean']['A']:+8.4f} {ny:+12.4f} {gap:+8.4f}"
                      + ("  (extrapolated)" if extrapolated else ""))
            if gaps:
                g = np.array(gaps)
                print(f"\n    random noise at the teacher's OWN error rate beats the teacher's own "
                      f"labels on {int((g > 0).sum())}/{len(g)}, mean {g.mean():+.4f}, "
                      f"p = {sign_test(g):.4f}")
                if exact and len(exact) < len(gaps):
                    x = np.array(exact)
                    print(f"    without the extrapolated rows: {int((x > 0).sum())}/{len(x)}, "
                          f"mean {x.mean():+.4f}, p = {sign_test(x):.4f}")
                print("    A teacher's mistakes are not noise. They are systematic -- concentrated "
                      "on the same ambiguous rows, and consistent -- so the student learns a "
                      "coherent wrong rule, where random errors of the same size largely cancel. "
                      "That, not the error rate, is what closes hard-label distillation here, and "
                      "it is why an ensemble has to decorrelate error STRUCTURE and not merely "
                      "lower the rate.")
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
             "arms": {a: arm_doc(a) for a in arms}, "repeats": args.repeats, "seed": args.seed,
             "selection": (f"best student < {args.max_student} in {args.from_gate}"
                           if args.from_gate else "explicit"),
             "n_datasets": len({r['dataset'] for r in rows}), "verdicts": verdicts, "rows": rows},
            indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def most_confident_pick(softs: dict[str, dict], n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the prediction of whichever model is most confident about it, and who was picked.

    Averaging cannot reach complementary information: where exactly one model is right and the other
    is confidently wrong, the mean lands on the wrong one. Selecting can, in principle -- so this is
    the cheap test of whether the gap between "at least one right" and the average single model is
    reachable by any rule that does not already know the answer.

    The caveat is calibration. Confidences from different models are not on a common scale, and a
    systematically bolder model wins every row regardless of being right, which is why the caller is
    given the pick distribution rather than just the accuracy.
    """
    names = list(softs)
    labs, confs = {}, {}
    for m in names:
        labs[m], confs[m] = teacher_label_conf(softs[m], n_test)
    pick = np.vstack([confs[m] for m in names]).argmax(axis=0)
    return np.array([labs[names[p]][i] for i, p in enumerate(pick)]), pick


def error_overlap(wrong: dict[str, np.ndarray]) -> dict:
    """Do these labellers make the SAME mistakes? The question an ensemble lives or dies by.

    Accuracy is not what governs pseudo-labelling here: a teacher's errors beat random errors of the
    same rate by five points, because they concentrate on the same ambiguous rows and point the same
    way. An ensemble that is more accurate while wrong in the same places inherits exactly that, so
    the number to look at is not the rate.

    * `joint / independent` -- how much more often two models are wrong together than chance would
      give. 1.0 is independence; above 1.0 they share a failure mode.
    * `all_wrong` -- rows no model gets right. No combination rule can ever recover these, so this is
      the hard floor on any ensemble built from this set.
    * `any_right` -- the accuracy a perfect oracle picking among these models would reach. The gap
      between it and the best single model is all the complementary information there is.
    """
    names = list(wrong)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ja, jb = wrong[a], wrong[b]
            joint = float((ja & jb).mean())
            indep = float(ja.mean() * jb.mean())
            pairs.append({"a": a, "b": b, "joint": joint, "independent": indep,
                          "ratio": (joint / indep) if indep > 0 else float("nan")})
    stacked = np.vstack([wrong[n] for n in names])
    return {"pairs": pairs, "all_wrong": float(stacked.all(axis=0).mean()),
            "any_right": float(1.0 - stacked.all(axis=0).mean()),
            "mean_single": float(1.0 - stacked.mean())}


def run_labellers(args) -> int:
    """How accurate is each labeller, and their average, on the rows a pool would be drawn from?

    This costs no fits at all -- every number comes from archived sidecars and the true test labels
    -- and it is what prices an ensemble before one is built. The break-even sweep says how accurate
    a labeller must be for the pool to pay; this says how accurate each candidate actually is. If
    the best candidate's error sits above the break-even, no amount of arm B will change it, and the
    comparison is one column against another rather than another run.
    """
    models = list(args.labellers)
    reports = teacher_reports(args.teacher)
    if args.from_gate:
        wanted = gate_selection(args.from_gate, args.max_student)
    else:
        wanted = args.datasets or sorted(reports)
    names = [n for n in wanted if n in reports]

    print(f"labeller accuracy on the full test split, {len(models)} model(s): {', '.join(models)}\n")
    print(f"{'dataset':30s} " + " ".join(f"{m:>12s}" for m in models)
          + f"{'ensemble':>12s}{'best single':>13s}")
    rows, missing, overlaps, picks = [], {m: 0 for m in models}, [], []
    for name in names:
        _, yte = load(name, "test")
        accs = {}
        for m in models:
            s = load_soft(args.teacher, name, m)
            if s is None:
                missing[m] += 1
                continue
            accs[m] = float((teacher_labels(s, len(yte)) == yte).mean())
        ens = load_ensemble_soft(args.teacher, name, models) if len(accs) == len(models) else None
        ens_acc = float((teacher_labels(ens, len(yte)) == yte).mean()) if ens else float("nan")
        conf_acc = float("nan")
        if len(accs) == len(models) and len(models) > 1:
            softs = {m: load_soft(args.teacher, name, m) for m in models}
            wrong = {m: (teacher_labels(softs[m], len(yte)) != yte) for m in models}
            overlaps.append({"dataset": name, **error_overlap(wrong)})
            pred, pick = most_confident_pick(softs, len(yte))
            conf_acc = float((pred == yte).mean())
            picks.append({"dataset": name, "accuracy": conf_acc,
                          "share": {m: float((pick == i).mean()) for i, m in enumerate(models)}})
        print(f"{name:30s} " + " ".join(f"{accs.get(m, float('nan')):12.4f}" for m in models)
              + f"{ens_acc:12.4f}{(max(accs.values()) if accs else float('nan')):13.4f}")
        rows.append({"dataset": name, "per_model": accs, "ensemble": ens_acc,
                     "most_confident": conf_acc,
                     "best_single": max(accs.values()) if accs else None})
    for m, k in missing.items():
        if k:
            print(f"  {m}: no sidecar for {k} of {len(names)} datasets")

    full = [r for r in rows if r["ensemble"] == r["ensemble"] and r["best_single"] is not None]
    if full:
        print(f"\n  over the {len(full)} datasets every model ran:")
        for m in models:
            v = np.array([r["per_model"][m] for r in full])
            print(f"    {m:14s} mean {v.mean():.4f}   mean error {1 - v.mean():.1%}")
        e = np.array([r["ensemble"] for r in full])
        b = np.array([r["best_single"] for r in full])
        print(f"    {'ensemble':14s} mean {e.mean():.4f}   mean error {1 - e.mean():.1%}")
        print(f"\n    ensemble vs the best single model: {e.mean() - b.mean():+.4f} "
              f"({int((e > b).sum())}/{len(full)} wins, p = {sign_test(e - b):.4f})")
        print("    'best single' is chosen per dataset on the test set, so it is an oracle: a real "
              "system must pick one model in advance, and the ensemble is what avoids that choice.")

    if overlaps:
        print(f"\n  ERROR OVERLAP over {len(overlaps)} datasets -- whether these models fail in "
              f"the same places:")
        ratios: dict[tuple[str, str], list[float]] = {}
        for o in overlaps:
            for pr in o["pairs"]:
                if pr["ratio"] == pr["ratio"]:
                    ratios.setdefault((pr["a"], pr["b"]), []).append(pr["ratio"])
        for (a, b), v in sorted(ratios.items()):
            arr = np.array(v)
            print(f"    {a:12s} vs {b:12s} wrong together {arr.mean():.2f}x as often as "
                  f"independence would give")
        allw = np.array([o["all_wrong"] for o in overlaps])
        anyr = np.array([o["any_right"] for o in overlaps])
        single = np.array([o["mean_single"] for o in overlaps])
        print(f"\n    every model wrong on {allw.mean():.1%} of rows -- the floor no combination "
              f"rule can go below")
        print(f"    at least one right on {anyr.mean():.1%}, against {single.mean():.1%} for the "
              f"average single model: {anyr.mean() - single.mean():+.1%} of complementary "
              f"information exists to be exploited")
        if picks:
            pa = np.array([p["accuracy"] for p in picks])
            en = np.array([r["ensemble"] for r in rows if r["dataset"]
                           in {p["dataset"] for p in picks}])
            print(f"\n    picking the MORE CONFIDENT model per row: {pa.mean():.4f}, "
                  f"against {en.mean():.4f} for averaging and {anyr.mean():.1%} for the oracle")
            for m in models:
                sh = np.array([p["share"][m] for p in picks])
                print(f"      {m:12s} wins the confidence comparison on {sh.mean():.1%} of rows")
            print("      Confidences are not on a common scale across models, so a systematically "
                  "bolder model wins rows regardless of being right; the shares above are what "
                  "makes that visible rather than assumed.")
        print("    A ratio near 1.0 means the errors are independent and an ensemble can average "
              "them away. Well above 1.0 means the models share a failure mode -- on these features, "
              "on these rows -- and a more accurate ensemble would still be wrong in the same "
              "places, which is the thing that closed hard-label distillation.")
    if args.out and rows:
        args.out.write_text(json.dumps({"models": models, "rows": rows,
                                        "overlaps": overlaps, "picks": picks}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def decision_margin(d: np.ndarray) -> np.ndarray:
    """How far the student was from changing its mind: |distance| binary, top1 - top2 multiclass."""
    if d.ndim == 1:
        return np.abs(d)
    s = np.sort(d, axis=1)
    return s[:, -1] - s[:, -2]


def rocket_ridge_scored(xtr, ytr, xte, n_kernels: int = 10_000, seed: int = 0):
    nch = xtr.shape[1] if xtr.ndim == 3 else 1
    bank = generate_kernels(seed, xtr.shape[-1], n_kernels, n_channels=nch)
    sc = StandardScaler().fit(transform(xtr, bank))
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(transform(xtr, bank)), ytr)
    F = sc.transform(transform(xte, bank))
    return clf.predict(F), decision_margin(clf.decision_function(F))


def mr_hydra_scored(xtr, ytr, xte, seed: int = 0):
    """MultiRocketHydra's predictions and its ridge's decision margin.

    aeon exposes no confidence: its `predict_proba` falls back to one-hot for a RidgeClassifierCV
    backbone, which is exactly no information. The margin is reachable only by reproducing the
    private transform pipeline, so the reconstruction is checked against `predict()` before its
    margins are used -- a wrong reconstruction returns plausible confidences and would route the
    wrong rows, which is the failure this whole file exists to avoid.
    """
    from aeon.classification.convolution_based import MultiRocketHydraClassifier

    a = xtr[:, None, :] if xtr.ndim == 2 else xtr
    b = xte[:, None, :] if xte.ndim == 2 else xte
    m = MultiRocketHydraClassifier(random_state=seed).fit(a, ytr)
    xt = np.concatenate((m._scale_hydra.transform(m._transform_hydra.transform(b)),
                         m._scale_multirocket.transform(m._transform_multirocket.transform(b))),
                        axis=1)
    pred, direct = m.classifier.predict(xt), m.predict(b)
    if not np.array_equal(pred, direct):
        raise RuntimeError("the reconstructed mr-hydra pipeline disagrees with predict(); its "
                           "internals have changed and the margins would be meaningless")
    return direct, decision_margin(m.classifier.decision_function(xt))


SCORERS = {"rocket+ridge": rocket_ridge_scored, "mr-hydra": mr_hydra_scored}

#: Which scorers have a kernel bank whose size changes their margin scale. mr-hydra's transform is
#: fixed by aeon, so its margins carry no bank size and its cache key must not pretend otherwise.
KERNEL_SCORERS = frozenset({"rocket+ridge"})


def route_curve_random(spred, tpred, y, fracs, seed: int = 0, draws: int = 20
                       ) -> list[tuple[float, float]]:
    """The control: escalate a RANDOM fraction instead of the least-confident one.

    Escalating any rows at all to a teacher that is better on average buys something, so a routing
    curve that rises proves nothing by itself -- the claim is that the STUDENT'S UNCERTAINTY picks
    better rows than chance. Without this line the gain and the signal are not separable, and the
    obvious reading of the result would be the wrong one.

    Averaged over `draws` because a single random subset of a 100-row test set is very noisy.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(spred, dtype=object)
    t = np.asarray(tpred, dtype=object)
    truth = np.asarray(y, dtype=object)
    n = len(truth)
    out = []
    for f in fracs:
        k = int(round(f * n))
        accs = []
        for _ in range(draws):
            pred = s.copy()
            if k:
                idx = rng.choice(n, size=k, replace=False)
                pred[idx] = t[idx]
            accs.append(float((pred == truth).mean()))
        out.append((f, float(np.mean(accs))))
    return out


def route_curve(spred, sconf, tpred, y, fracs) -> list[tuple[float, float]]:
    """Accuracy when the student's least-confident `f` of the rows are handed to the teacher.

    f=0 is the student alone and f=1 the teacher alone, so the curve only says something new if it
    rises above both ends: that is the claim that the two models are wrong on different rows and
    that the student knows which ones are its own.
    """
    order = np.argsort(sconf, kind="stable")
    s = np.asarray(spred, dtype=object)
    t = np.asarray(tpred, dtype=object)
    truth = np.asarray(y, dtype=object)
    out = []
    for f in fracs:
        pred = s.copy()
        k = int(round(f * len(truth)))
        if k:
            pred[order[:k]] = t[order[:k]]
        out.append((f, float((pred == truth).mean())))
    return out


def _route_worker(t):
    name, learner, seed, teacher_dir, cache = t
    try:
        cp = Path(cache) / f"{name}__{learner.replace('+', '_')}__seed{seed}.json" if cache else None
        if cp is not None and cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            return name, learner, d["pred"], d["conf"], ""
        xtr, ytr = load(name, "train")
        xte, _ = load(name, "test")
        pred, conf = SCORERS[learner](normalize_series(xtr), ytr, normalize_series(xte), seed=seed)
        pred, conf = [str(p) for p in pred], [float(c) for c in conf]
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"dataset": name, "learner": learner, "seed": seed,
                                      "pred": pred, "conf": conf}), encoding="utf-8")
        return name, learner, pred, conf, ""
    except Exception as e:  # noqa: BLE001
        return name, learner, None, None, f"{type(e).__name__}: {e}"[:140]


def oof_margins(name: str, learner: str, seed: int, folds: int, cache: str | None,
                n_kernels: int = 10_000) -> np.ndarray:
    """Out-of-fold decision margins on the TRAIN split.

    **`n_kernels` must match the model the threshold will be applied to.** A margin is a distance in
    the fitted model's decision space, so a bank of a different size produces a different scale, and
    a quantile taken from the wrong one sets the escalation rate to something nobody chose. It is
    also part of the cache key for the same reason -- a hit that ignored it would return margins
    from another model and look like a fast path.

    A served system cannot sort a batch it has not received, so the escalation rule has to be a
    THRESHOLD on one row's margin rather than a fraction of a set. The threshold has to come from
    data that exists before any test row does, which means the training split -- and it has to come
    from margins the model produced on rows it had not seen, since a fitted model is systematically
    more confident on its own training rows and an in-sample threshold would sit far too low.

    Cross-validated rather than a single holdout because these training sets are small: ArrowHead has
    36 rows, so a 25% holdout would estimate a 20% quantile from nine numbers. Every row gets a
    margin this way, which is the most calibration data the problem allows.
    """
    stem = f"{name}__{learner.replace('+', '_')}__seed{seed}__k{folds}"
    keyed = stem if learner not in KERNEL_SCORERS else f"{stem}__n{n_kernels}"
    if cache:
        cp = Path(cache) / f"{keyed}.json"
        # The un-suffixed name is the pre-existing cache, which was always computed at 10,000
        # kernels. Read it only for that size; for any other, a hit on it would be the exact
        # mismatch the suffix exists to stop.
        legacy = Path(cache) / f"{stem}.json"
        for path in ((cp, legacy) if n_kernels == 10_000 else (cp,)):
            if path.exists():
                try:
                    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["margins"],
                                      dtype=float)
                except Exception:  # noqa: BLE001
                    pass
    from sklearn.model_selection import StratifiedKFold, KFold

    xtr, ytr = load(name, "train")
    xtr = normalize_series(xtr)
    k = max(2, min(folds, int(np.bincount(np.unique(ytr, return_inverse=True)[1]).min())))
    try:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        parts = list(splitter.split(xtr, ytr))
    except ValueError:
        parts = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(xtr))
    out = np.zeros(len(ytr), dtype=float)
    for fit_i, held_i in parts:
        kw = {"n_kernels": n_kernels} if learner in KERNEL_SCORERS else {}
        _, conf = SCORERS[learner](xtr[fit_i], ytr[fit_i], xtr[held_i], seed=seed, **kw)
        out[held_i] = conf
    if cache:
        cp = Path(cache) / f"{keyed}.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"dataset": name, "learner": learner, "seed": seed, "folds": k,
                                  "n_kernels": n_kernels, "margins": out.tolist()}),
                      encoding="utf-8")
    return out


def _oof_worker(t):
    name, learner, seed, folds, cache = t
    try:
        return name, learner, oof_margins(name, learner, seed, folds, cache).tolist(), ""
    except Exception as e:  # noqa: BLE001
        return name, learner, None, f"{type(e).__name__}: {e}"[:140]


def run_calibrate(args) -> int:
    """Turn the escalation FRACTION into a servable THRESHOLD, and measure what that costs.

    The routing result was produced by sorting a whole test set and escalating its least confident
    fraction. No serving system can do that. This picks the threshold from out-of-fold margins on the
    training split -- data available before deployment -- and then applies it row by row, which is
    what a served system would actually run.

    Two things can go wrong and both are reported rather than assumed: the realised escalation rate
    can miss its target, because train and test margins are not identically distributed; and the
    accuracy can fall short of the sorted-fraction version even when the rate is right.
    """
    reports = teacher_reports(args.teacher)
    wanted = (gate_selection(args.from_gate, args.max_student) if args.from_gate
              else (args.datasets or sorted(reports)))
    names = [n for n in wanted if n in reports and load_soft(args.teacher, n) is not None]
    learners = [k for k in SCORERS if k in args.learners]
    targets = [float(t) for t in args.targets]
    print(f"calibrating a margin threshold from {args.folds}-fold out-of-fold margins on the train "
          f"split\n{len(names)} datasets, {len(learners)} learner(s), targets "
          f"{', '.join(f'{t:.0%}' for t in targets)}\n")

    jobs = sorted(((n, l, args.seed, args.folds, str(args.oof_cache))
                   for n in names for l in learners),
                  key=lambda j: -reports[j[0]]["shape"]["n_test"])
    oof: dict[tuple[str, str], np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_oof_worker, j) for j in jobs]), start=1):
            n, l, m, err = fut.result()
            if m is None:
                print(f"  {n} {l} failed: {err}", flush=True)
            else:
                oof[(n, l)] = np.asarray(m, dtype=float)
            if i % 10 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} calibration sets built", flush=True)

    print(f"\n{'dataset':26s} {'learner':13s} {'target':>7s} {'actual':>7s} "
          f"{'threshold':>10s} {'acc':>8s} {'sorted':>8s} {'gap':>8s}")
    rows = []
    for name in names:
        soft = load_soft(args.teacher, name)
        _, yte = load(name, "test")
        tpred = teacher_labels(soft, len(yte))
        for lname in learners:
            if (name, lname) not in oof:
                continue
            got = _route_worker((name, lname, args.seed, str(args.teacher), str(args.route_cache)))
            if got[2] is None:
                continue
            spred, sconf = np.asarray(got[2], dtype=object), np.asarray(got[3], dtype=float)
            for t in targets:
                thr = float(np.quantile(oof[(name, lname)], t))
                esc = sconf < thr
                pred = spred.copy()
                pred[esc] = np.asarray(tpred, dtype=object)[esc]
                acc = float((pred == np.asarray(yte, dtype=object)).mean())
                # The same budget spent by sorting the test set, which is the number the routing
                # result reported and the thing a threshold has to be judged against.
                sorted_acc = route_curve(spred, sconf, tpred, yte, [t])[0][1]
                print(f"{name:26s} {lname:13s} {t:7.0%} {esc.mean():7.1%} {thr:10.4f} "
                      f"{acc:8.4f} {sorted_acc:8.4f} {acc - sorted_acc:+8.4f}")
                rows.append({"dataset": name, "learner": lname, "target": t,
                             "realised": float(esc.mean()), "threshold": thr, "accuracy": acc,
                             "sorted_accuracy": sorted_acc,
                             "student": route_curve(spred, sconf, tpred, yte, [0.0])[0][1]})

    if not rows:
        return 1
    print("\nCALIBRATION -- does a threshold chosen before deployment hit its budget and its number?")
    for lname in learners:
        for t in targets:
            sub = [r for r in rows if r["learner"] == lname and r["target"] == t]
            if not sub:
                continue
            rate = np.array([r["realised"] for r in sub])
            gap = np.array([r["accuracy"] - r["sorted_accuracy"] for r in sub])
            gain = np.array([r["accuracy"] - r["student"] for r in sub])
            print(f"  {lname:13s} target {t:4.0%}: realised {rate.mean():5.1%} "
                  f"(median {float(np.median(rate)):5.1%}, {rate.min():4.1%}-{rate.max():5.1%})   "
                  f"gain over the student {gain.mean():+.4f} (p = {sign_test(gain):.4f})   "
                  f"vs sorting the batch {gap.mean():+.4f}")
    print("\n  The spread on the realised rate is the cost of not having the batch: a threshold set "
          "on training margins spends more or less than its budget on any given dataset, because "
          "train and test margins are not identically distributed. The gain column is what a served "
          "system would actually get.")
    if args.out:
        args.out.write_text(json.dumps({"folds": args.folds, "targets": targets, "rows": rows},
                                       indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def run_route(args) -> int:
    """Route, do not distil: run the student, escalate only the rows it is unsure of.

    Distillation needs the teacher's labels to be RIGHT, which on a hard dataset they mostly are not.
    Routing needs something weaker and quite different -- that the teacher be right on the rows the
    student gets wrong, and that the student know which those are. It also never contaminates a
    training set, so a teacher error costs one row instead of biasing a fit.
    """
    reports = teacher_reports(args.teacher)
    if args.from_gate:
        wanted = gate_selection(args.from_gate, args.max_student)
    else:
        wanted = args.datasets or sorted(reports)
    names = [n for n in wanted if n in reports and load_soft(args.teacher, n) is not None]
    if not names:
        print("no dataset has both a report and a soft-label sidecar")
        return 1
    learners = [k for k in SCORERS if k in args.learners]
    fracs = [i / 20 for i in range(21)]
    print(f"routing: {len(names)} datasets, {len(learners)} learner(s), "
          f"escalating the least-confident 0..100% to the teacher\n")

    jobs = sorted(((n, l, args.seed, str(args.teacher), str(args.route_cache))
                   for n in names for l in learners),
                  key=lambda j: -reports[j[0]]["shape"]["n_test"])
    got: dict[tuple[str, str], tuple[list, list]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_route_worker, j) for j in jobs]), start=1):
            n, l, pred, conf, err = fut.result()
            if pred is None:
                print(f"  {n} {l} failed: {err}", flush=True)
            else:
                got[(n, l)] = (pred, conf)
            if i % 10 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} students scored", flush=True)

    print(f"\n{'dataset':30s} {'learner':13s} {'student':>8s} {'teacher':>8s} {'best':>8s} "
          f"{'at':>5s} {'gain':>8s}")
    rows = []
    for name in names:
        soft = load_soft(args.teacher, name)
        _, yte = load(name, "test")
        tpred = teacher_labels(soft, len(yte))
        for lname in learners:
            if (name, lname) not in got:
                continue
            pred, conf = got[(name, lname)]
            curve = route_curve(pred, np.asarray(conf), tpred, yte, fracs)
            control = route_curve_random(pred, tpred, yte, fracs, seed=args.seed)
            acc = [a for _, a in curve]
            best_i = int(np.argmax(acc))
            gain = acc[best_i] - max(acc[0], acc[-1])
            print(f"{name:30s} {lname:13s} {acc[0]:8.4f} {acc[-1]:8.4f} {acc[best_i]:8.4f} "
                  f"{fracs[best_i]:5.0%} {gain:+8.4f}")
            rows.append({"dataset": name, "learner": lname, "student": acc[0], "teacher": acc[-1],
                         "best": acc[best_i], "best_frac": fracs[best_i], "gain_over_both": gain,
                         "curve": curve, "control": control})

    if not rows:
        return 1
    print("\nROUTING -- does escalating the student's least-confident rows beat either model alone?")
    for lname in learners:
        sub = [r for r in rows if r["learner"] == lname]
        if not sub:
            continue
        g = np.array([r["gain_over_both"] for r in sub])
        print(f"  {lname:13s} beats both ends on {int((g > 0).sum())}/{len(g)}   "
              f"mean {g.mean():+.4f}   median {float(np.median(g)):+.4f}   p = {sign_test(g):.4f}")
        # The product question is not the peak, it is the price: what a fixed escalation budget buys
        # against the student alone, since the teacher costs ~14x the student per row.
        for f in (0.10, 0.20, 0.30, 0.50):
            j = fracs.index(f)
            d = np.array([r["curve"][j][1] - r["student"] for r in sub])
            c = np.array([r["control"][j][1] - r["student"] for r in sub])
            edge = d - c
            print(f"    escalate {f:4.0%}: {d.mean():+.4f} over the student alone "
                  f"({int((d > 0).sum())}/{len(d)} datasets, p = {sign_test(d):.4f})   "
                  f"| random rows {c.mean():+.4f}, so the confidence signal is worth "
                  f"{edge.mean():+.4f} (p = {sign_test(edge):.4f})")
    # Selecting the best fraction per dataset on the same data it is measured on is an oracle and
    # cannot be shipped; it is reported to bound what a tuned rule could reach, never as the result.
    print("\n  The random-rows column is the control that makes the rest readable: escalating\n  ANY rows to a teacher that is better on average buys something, so a rising curve is not\n  evidence that the student knows what it does not know. The difference between the two is.")
    print("\n  'best' selects the escalation fraction on the test set itself, so it is an oracle "
          "bound on what a tuned rule could reach, not an achievable number. The fixed-budget rows "
          "above are the ones a product could actually run.")
    if args.out:
        args.out.write_text(json.dumps({"design": "escalate the least-confident fraction",
                                        "fracs": fracs, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="T vs A on full test splits (the gate)")
    ap.add_argument("--arm-b", action="store_true", help="arms A/B/C on pool/holdout splits")
    ap.add_argument("--labeller-accuracy", action="store_true",
                    help="how accurate each labeller and their average are; costs no fits and is "
                         "what prices an ensemble against the break-even")
    ap.add_argument("--labellers", nargs="*", default=["tabicl-v2"],
                    help="models whose archived soft labels to read and average")
    ap.add_argument("--calibrate", action="store_true",
                    help="turn the escalation fraction into a servable margin threshold, chosen "
                         "from out-of-fold margins on the train split, and measure what it costs")
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-validation folds for the out-of-fold calibration margins")
    ap.add_argument("--targets", nargs="*", default=[0.10, 0.20, 0.30],
                    help="escalation budgets to calibrate a threshold for")
    ap.add_argument("--oof-cache", type=Path, default=ROOT / "data" / "oof_margins",
                    help="per-(dataset, learner, seed, folds) out-of-fold margins")
    ap.add_argument("--route", action="store_true",
                    help="escalate the student's least-confident rows to the teacher instead of "
                         "distilling from it")
    ap.add_argument("--route-cache", type=Path, default=ROOT / "data" / "route_students",
                    help="per-(dataset, learner, seed) student predictions and margins")
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

    if not (args.gate or args.arm_b or args.route or args.labeller_accuracy
            or args.calibrate):
        args.gate = True
    t0 = time.perf_counter()
    rc = run_gate(args) if args.gate else 0
    if args.arm_b:
        rc = run_arm_b(args) or rc
    if args.route:
        rc = run_route(args) or rc
    if args.labeller_accuracy:
        rc = run_labellers(args) or rc
    if args.calibrate:
        rc = run_calibrate(args) or rc
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
