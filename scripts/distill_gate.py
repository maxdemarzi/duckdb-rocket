"""The gate from docs/DISTILLATION_PLAN.md: arms A, C, T and B.

Distillation only earns its place if a student trained on teacher-pseudo-labelled unlabelled data
beats what you would have done with the real labels you already had.

    context = the real train split          (real labels; what the teacher uses in-context)
    pool    = half the test split           (labels discarded for arm B; kept for arm C)
    holdout = the other half of the test    (labels used only for scoring)

    A  fit on context                       -> what you do WITHOUT the teacher
    C  fit on context + pool, real labels    -> the ceiling arm B is chasing
    T  the teacher itself, scored on holdout -> can it label the pool better than A can?
    B  fit on context + pool, TEACHER labels -> the actual proposition

**The first version of this gate measured `C - A` and called that the decision. That was the wrong
quantity.** `C - A` says how much room there is; it says nothing about whether the teacher can
reach any of it. A teacher no better than the student produces pseudo-labels no better than the
student's own guesses, and then a large `C - A` is simply unreachable. On the saturated subset
`C - A` was under 1 point and the gate correctly said stop -- but on the hard datasets `C - A` is
+0.09 and the same gate would have said go, while the teacher beats arm A by only ~0.02. `T - A` is
the quantity that decides it, and it was never computed.

So `T` is the gate now, and `B` is measured rather than inferred:

    T - A <= 0    the teacher cannot label the pool better than the student already can -- stop
    T - A >  0    there is something to inherit; B says how much of it survives the transfer

Arm T needs the real teacher (ROCKET features + `tabicl-v2` in-context), which lives in DuckDB and
wants a GPU, so it is not recomputed here. It is read from the soft-label sidecar that
`phase5_pipeline.py` writes next to its report (`--teacher DIR`). Those probabilities are also what
arm B trains on, which is the point: a student fit on the teacher's *distribution* inherits its
uncertainty, not just its decisions.

Two learners per arm, because arm A has to be the *best* label-only option rather than a convenient
one: ROCKET+ridge (the pipeline's own feature family) and MultiRocketHydra (aeon's stronger
convolutional classifier, and the intended CPU student).

    uv run python scripts/distill_gate.py
    uv run python scripts/distill_gate.py --datasets ECG5000 ItalyPowerDemand
    uv run python scripts/distill_gate.py --datasets Haptics --teacher reference/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402

# pool/context ratio is what should drive any gain, so the candidates are ordered by it.
# SyntheticControl is deliberately included at 0.5x as a control that should show ~nothing.
CANDIDATES = ("ItalyPowerDemand", "ECG5000", "OSULeaf", "SyntheticControl")


def rocket_ridge(xtr, ytr, xte, n_kernels: int = 10_000):
    """The pipeline's own feature family, classified by ridge instead of an in-context model."""
    nch = xtr.shape[1] if xtr.ndim == 3 else 1
    bank = generate_kernels(0, xtr.shape[-1], n_kernels, n_channels=nch)
    ftr, fte = transform(xtr, bank), transform(xte, bank)
    sc = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(sc.transform(ftr), ytr)
    return clf.predict(sc.transform(fte))


def mr_hydra(xtr, ytr, xte):
    """aeon's MultiRocketHydra -- the intended CPU student, and the stronger label-only baseline."""
    from aeon.classification.convolution_based import MultiRocketHydraClassifier

    # aeon wants (n, channels, timepoints); our univariate arrays are (n, timepoints).
    a = xtr[:, None, :] if xtr.ndim == 2 else xtr
    b = xte[:, None, :] if xte.ndim == 2 else xte
    clf = MultiRocketHydraClassifier(random_state=0).fit(a, ytr)
    return clf.predict(b)


LEARNERS = {"rocket+ridge": rocket_ridge, "mr-hydra": mr_hydra}


def load_teacher(directory: Path, dataset: str) -> dict | None:
    """The soft-label sidecar `phase5_pipeline.py` writes beside its report.

    Arm T is the real pipeline -- ROCKET features through `tabicl-v2` in-context -- which needs
    DuckDB, the anofox extension and (in practice) a GPU. It is not recomputed here; it is read.
    """
    for stem in (f"phase5_{dataset}_gpu_soft.json", f"phase5_{dataset}_soft.json"):
        p = directory / stem
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def teacher_labels(teacher: dict, n_test: int) -> np.ndarray:
    """Teacher argmax per test row, in the dataset's own test order.

    The pipeline lays its id space out as arange(n_train + n_test) with the train rows first, so
    test row k is id n_train + k. Recovering that offset by inspection would be the kind of guess
    that produces a plausible wrong number, so the sidecar records `n_train` and this asserts on it.
    """
    off, mean_p = teacher["n_train"], teacher["mean_proba"]
    if teacher["n_test"] != n_test:
        raise ValueError(f"teacher ran on {teacher['n_test']} test rows, the loader gives {n_test}")
    out = []
    for k in range(n_test):
        row = mean_p.get(str(off + k))
        if row is None:
            raise ValueError(f"teacher has no probabilities for test row {k} (id {off + k})")
        out.append(max(row, key=row.get))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", type=Path,
                    help="directory holding phase5_<dataset>[_gpu]_soft.json; enables arms T and B")
    args = ap.parse_args()

    print(f"{'dataset':18s} {'learner':13s} {'ctx':>5s} {'pool':>5s} {'hold':>5s} "
          f"{'A':>7s} {'T':>7s} {'B':>7s} {'C':>7s} {'T-A':>7s} {'B-A':>7s} {'C-A':>7s} {'secs':>6s}")
    rows = []
    for name in args.datasets:
        try:
            xtr, ytr = load(name, "train")
            xte, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"{name:18s} load failed: {str(e)[:50]}")
            continue
        xtr, xte = normalize_series(xtr), normalize_series(xte)

        # Teacher labels are indexed by position in the test split, so the split has to be done on
        # INDICES rather than on the arrays -- otherwise there is no way to say which teacher label
        # belongs to which pooled row. The arrays are then taken by those indices, so arms A and C
        # see exactly the same split they saw before this was added.
        tlab = None
        if args.teacher:
            teacher = load_teacher(args.teacher, name)
            if teacher is None:
                print(f"{name:18s} no teacher sidecar in {args.teacher} -- T and B unavailable")
            else:
                tlab = teacher_labels(teacher, len(yte))

        idx = np.arange(len(yte))
        # Stratify so a class cannot land entirely in one half -- with heavily imbalanced sets
        # (ECG5000) an unstratified split would make the arms incomparable rather than noisy.
        try:
            pool_i, hold_i = train_test_split(
                idx, test_size=0.5, random_state=args.seed, stratify=yte)
        except ValueError:
            pool_i, hold_i = train_test_split(idx, test_size=0.5, random_state=args.seed)

        pool_x, pool_y = xte[pool_i], yte[pool_i]
        hold_x, hold_y = xte[hold_i], yte[hold_i]

        ctx_x, ctx_y = xtr, ytr
        both_x = np.concatenate([ctx_x, pool_x])
        both_y = np.concatenate([ctx_y, pool_y])

        # Arm T: the teacher's own accuracy on the holdout. Scored on the holdout and not on the
        # pool, so it is directly comparable with A, B and C rather than nearly comparable.
        #
        # The sidecar comes from a run that scored the WHOLE test split, and slicing the holdout out
        # of it afterwards is exact, not an approximation: an in-context learner treats each test
        # row as an independent query against the train context, so a row's prediction cannot
        # depend on which other rows shared its call. That was verified directly -- GunPoint,
        # 150/150 rows identical across chunk sizes -- before the pipeline began chunking at all.
        # It also means one pipeline run per dataset serves every arm here.
        t_acc = float((tlab[hold_i] == hold_y).mean()) if tlab is not None else float("nan")
        # How good the pseudo-labels arm B trains on actually are. Not a gate, a diagnostic: if
        # B underperforms while this is high, the loss is in the transfer and not in the teacher.
        t_pool = float((tlab[pool_i] == pool_y).mean()) if tlab is not None else float("nan")

        for lname, fn in LEARNERS.items():
            t0 = time.perf_counter()
            try:
                a = float((fn(ctx_x, ctx_y, hold_x) == hold_y).mean())
                c = float((fn(both_x, both_y, hold_x) == hold_y).mean())
                if tlab is None:
                    b = float("nan")
                else:
                    # Arm B: the real proposition. Same rows as C, teacher labels instead of real
                    # ones on the pool half. Hard argmax labels -- see the note in the summary
                    # below about what a soft-label student would add and why it is not measured
                    # here.
                    b = float((fn(both_x, np.concatenate([ctx_y, tlab[pool_i]]),
                                  hold_x) == hold_y).mean())
            except Exception as e:  # noqa: BLE001
                print(f"{name:18s} {lname:13s} FAILED {str(e)[:44]}")
                continue
            secs = time.perf_counter() - t0
            print(f"{name:18s} {lname:13s} {len(ctx_y):5d} {len(pool_y):5d} {len(hold_y):5d} "
                  f"{a:7.4f} {t_acc:7.4f} {b:7.4f} {c:7.4f} "
                  f"{t_acc - a:+7.4f} {b - a:+7.4f} {c - a:+7.4f} {secs:6.1f}")
            rows.append((name, lname, a, t_acc, b, c, t_pool))

    if not rows:
        return 0

    have_teacher = any(not np.isnan(r[3]) for r in rows)

    print("\nheadroom (C - A), the most arm B could ever recover:")
    for name, lname, a, _, _, c, _ in rows:
        print(f"  {name:18s} {lname:13s} {c - a:+.4f}"
              f"   {'room exists' if c - a >= 0.01 else 'no headroom'}")

    if not have_teacher:
        print("\nNO VERDICT. C - A alone does not decide this -- it says how much room there is,\n"
              "not whether the teacher can reach any of it. Re-run with --teacher DIR to get T.")
        return 0

    print("\nthe gate (T - A): can the teacher label the pool better than the student already can?")
    reachable = []
    for name, lname, a, t, b, c, t_pool in rows:
        ta = t - a
        if ta <= 0:
            note = "STOP -- the teacher is no better than this student; its labels add nothing"
        elif ta < 0.01:
            note = "marginal -- under 1 point of usable signal"
        else:
            note = "go -- there is something to inherit"
        print(f"  {name:18s} {lname:13s} T-A {ta:+.4f}  B-A {b - a:+.4f}  "
              f"(teacher on pool {t_pool:.4f})  {note}")
        if ta > 0:
            reachable.append((name, lname, ta, b - a))

    # The comparison that matters is against the BEST label-only option, not against a convenient
    # one. A teacher that beats ridge and loses to mr-hydra has not earned a distillation pipeline,
    # and reporting only the ridge column would hide that.
    print("\nagainst the best label-only learner per dataset:")
    for name in dict.fromkeys(r[0] for r in rows):
        per = [r for r in rows if r[0] == name]
        best_a = max(r[2] for r in per)
        best_l = max(per, key=lambda r: r[2])[1]
        t = per[0][3]
        # Three ways, not two: a tie is not "behind". On a saturated dataset every arm reaches the
        # same number and calling that a loss for the teacher misreads a ceiling as a defeat.
        if t > best_a:
            verdict = "teacher ahead"
        elif t == best_a:
            verdict = "tied -- nothing to distil either way"
        else:
            verdict = "TEACHER BEHIND"
        print(f"  {name:18s} best A {best_a:.4f} ({best_l})  T {t:.4f}  T-A {t - best_a:+.4f}"
              f"   {verdict}")

    if reachable:
        print("\nArm B above uses the teacher's hard argmax. A soft-label student (KLDivLoss on the\n"
              "distribution, which `phase5_pipeline.py` now writes) can exceed it, because equal\n"
              "argmax accuracy still carries unequal uncertainty -- so a negative B - A here bounds\n"
              "hard-label distillation, not distillation. Neither of the two learners here can take\n"
              "soft targets; that needs the neural student (LITE / InceptionTimePlus).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
