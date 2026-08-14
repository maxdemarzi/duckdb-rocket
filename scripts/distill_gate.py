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

    uv run python scripts/distill_gate.py --gate                       # T vs A, all archived teachers
    uv run python scripts/distill_gate.py --gate --learners rocket+ridge
    uv run python scripts/distill_gate.py --arm-b --teacher reference --repeats 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    for stem in (f"phase5_{dataset}_gpu_soft.json", f"phase5_{dataset}_soft.json"):
        p = directory / stem
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def teacher_labels(soft: dict, n_test: int) -> np.ndarray:
    """Teacher argmax per test row, in the dataset's own test order.

    The pipeline lays its ids out as arange(n_train + n_test) with train first, so test row k is id
    n_train + k. The sidecar records n_train so that offset is asserted rather than rediscovered.
    """
    off, mean_p = soft["n_train"], soft["mean_proba"]
    if soft["n_test"] != n_test:
        raise ValueError(f"teacher ran on {soft['n_test']} test rows, the loader gives {n_test}")
    out = []
    for k in range(n_test):
        row = mean_p.get(str(off + k))
        if row is None:
            raise ValueError(f"teacher has no probabilities for test row {k} (id {off + k})")
        out.append(max(row, key=row.get))
    return np.asarray(out)


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

    rows: list[dict] = []
    for name in names:
        rep = reports[name]
        T, n_test = rep["accuracy"], rep["shape"]["n_test"]
        try:
            xtr, ytr = load(name, "train")
            xte, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"{name:24s} load failed: {str(e)[:44]}")
            continue
        if len(yte) != n_test:
            print(f"{name:24s} SKIPPED: report says {n_test} test rows, loader gives {len(yte)}")
            continue
        xtr, xte = normalize_series(xtr), normalize_series(xte)

        accs, deltas = {}, {}
        for lname, fn in learners.items():
            try:
                accs[lname] = float((fn(xtr, ytr, xte, seed=args.seed) == yte).mean())
            except Exception as e:  # noqa: BLE001
                print(f"{name:24s} {lname} failed: {str(e)[:40]}")
                continue
            deltas[lname] = T - accs[lname]
        if not deltas:
            continue
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


def run_arm_b(args) -> int:
    """Arms A, B and C on a pool/holdout split -- the only part that needs unlabelled data.

    Averaged over `--repeats` splits: one 50/50 split of a small test set is a very noisy estimate,
    and believing a single one is what broke the previous design.
    """
    reports = teacher_reports(args.teacher)
    names = [n for n in (args.datasets or sorted(reports)) if n in reports]
    learners = {k: v for k, v in LEARNERS.items() if k in args.learners}
    print(f"arms A/B/C over {args.repeats} split(s) per dataset\n")
    print(f"{'dataset':22s} {'learner':13s} {'A':>7s} {'B':>7s} {'C':>7s} {'B-A':>8s} {'C-A':>8s}")

    rows = []
    for name in names:
        soft = load_soft(args.teacher, name)
        if soft is None:
            continue
        xtr, ytr = load(name, "train")
        xte, yte = load(name, "test")
        xtr, xte = normalize_series(xtr), normalize_series(xte)
        try:
            tlab = teacher_labels(soft, len(yte))
        except ValueError as e:
            print(f"{name:22s} {e}")
            continue

        for lname, fn in learners.items():
            A, B, C = [], [], []
            for rep in range(args.repeats):
                idx = np.arange(len(yte))
                try:
                    pool_i, hold_i = train_test_split(
                        idx, test_size=0.5, random_state=args.seed + rep, stratify=yte)
                except ValueError:
                    pool_i, hold_i = train_test_split(
                        idx, test_size=0.5, random_state=args.seed + rep)
                hx, hy = xte[hold_i], yte[hold_i]
                bx = np.concatenate([xtr, xte[pool_i]])
                A.append(float((fn(xtr, ytr, hx, seed=args.seed) == hy).mean()))
                B.append(float((fn(bx, np.concatenate([ytr, tlab[pool_i]]), hx,
                                   seed=args.seed) == hy).mean()))
                C.append(float((fn(bx, np.concatenate([ytr, yte[pool_i]]), hx,
                                   seed=args.seed) == hy).mean()))
            a, b, c = float(np.mean(A)), float(np.mean(B)), float(np.mean(C))
            print(f"{name:22s} {lname:13s} {a:7.4f} {b:7.4f} {c:7.4f} {b - a:+8.4f} {c - a:+8.4f}")
            rows.append({"dataset": name, "learner": lname, "A": a, "B": b, "C": c})

    if rows:
        for lname in learners:
            sub = [r for r in rows if r["learner"] == lname]
            if not sub:
                continue
            ba = np.array([r["B"] - r["A"] for r in sub])
            print(f"\n  {lname}: B-A {int((ba > 0).sum())}/{len(ba)} wins, mean {ba.mean():+.4f}, "
                  f"sign test p = {sign_test(ba):.4f}")
        print("\nArm B uses the teacher's hard argmax. A soft-label student can exceed it, so a "
              "negative here bounds hard-label distillation and not distillation.")
    if args.out and rows:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
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
    ap.add_argument("--seed", type=int, default=0)
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
