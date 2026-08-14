"""The gate from docs/DISTILLATION_PLAN.md, arms A and C.

Distillation only earns its place if a student trained on teacher-pseudo-labelled unlabelled data
beats what you would have done with the real labels you already had. This measures the two arms
that need no teacher at all, because if they come out equal there is no headroom and arm B cannot
help:

    context = the real train split          (real labels; what the teacher would use in-context)
    pool    = half the test split           (labels discarded for arm B; kept for arm C)
    holdout = the other half of the test    (labels used only for scoring)

    A  fit on context           -> what you do WITHOUT the teacher
    C  fit on context + pool    -> the ceiling arm B is chasing, using real labels for the pool

`C - A` is the headroom. Arm B can only ever recover part of it, because the teacher's own
accuracy caps the pseudo-labels. If C - A is ~0, stop.

Two learners per arm, because arm A has to be the *best* label-only option rather than a
convenient one: ROCKET+ridge (the pipeline's own feature family) and MultiRocketHydra (aeon's
stronger convolutional classifier, and the intended CPU student).

    uv run python scripts/distill_gate.py
    uv run python scripts/distill_gate.py --datasets ECG5000 ItalyPowerDemand
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"{'dataset':18s} {'learner':13s} {'ctx':>5s} {'pool':>5s} {'hold':>5s} "
          f"{'A':>7s} {'C':>7s} {'C-A':>8s} {'secs':>6s}")
    rows = []
    for name in args.datasets:
        try:
            xtr, ytr = load(name, "train")
            xte, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"{name:18s} load failed: {str(e)[:50]}")
            continue
        xtr, xte = normalize_series(xtr), normalize_series(xte)

        # Stratify so a class cannot land entirely in one half -- with heavily imbalanced sets
        # (ECG5000) an unstratified split would make the arms incomparable rather than noisy.
        try:
            pool_x, hold_x, pool_y, hold_y = train_test_split(
                xte, yte, test_size=0.5, random_state=args.seed, stratify=yte)
        except ValueError:
            pool_x, hold_x, pool_y, hold_y = train_test_split(
                xte, yte, test_size=0.5, random_state=args.seed)

        ctx_x, ctx_y = xtr, ytr
        both_x = np.concatenate([ctx_x, pool_x])
        both_y = np.concatenate([ctx_y, pool_y])

        for lname, fn in LEARNERS.items():
            t0 = time.perf_counter()
            try:
                a = float((fn(ctx_x, ctx_y, hold_x) == hold_y).mean())
                c = float((fn(both_x, both_y, hold_x) == hold_y).mean())
            except Exception as e:  # noqa: BLE001
                print(f"{name:18s} {lname:13s} FAILED {str(e)[:44]}")
                continue
            secs = time.perf_counter() - t0
            print(f"{name:18s} {lname:13s} {len(ctx_y):5d} {len(pool_y):5d} {len(hold_y):5d} "
                  f"{a:7.4f} {c:7.4f} {c - a:+8.4f} {secs:6.1f}")
            rows.append((name, lname, a, c, c - a))

    if rows:
        print("\nheadroom (C - A), the most arm B could ever recover:")
        for name, lname, _, _, d in rows:
            verdict = "worth trying" if d >= 0.01 else "no headroom"
            print(f"  {name:18s} {lname:13s} {d:+.4f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
