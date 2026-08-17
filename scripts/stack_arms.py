"""Phase 7a — can ANY function of the arms' posteriors reach the oracle?

RESULTS.md's feature-route section closes with a precise claim: the per-row oracle over the four
arms is **0.8483** against a best-achieved 0.7686, and reaching it "needs a signal that knows which
arm is right, and no arm's own confidence is that signal." Three routes were tried and all failed —
averaging (0.7619, below best-single-arm), margin-routing to another representation (+0.0042,
p=0.27), and picking the surest arm per row (−0.0032).

This asks the question those three leave open, with the most permissive signal available: a model
fitted on **every arm's full posterior vector** with access to **real labels**. If that cannot reach
the oracle, nothing reading these posteriors can, and the whole collect-the-diversity direction is
closed. It costs no pod time — all four arms are archived for 17 of 17 datasets.

**This is deliberately an upper bound, not a deployable design.** The stacker is cross-validated
*within the test rows*, so it trains on labels a real system would not have. A shippable version
would fit on out-of-fold posteriors over the train split, which nothing has produced. The point is
that an upper bound is exactly what a gate needs: a negative here is conclusive, and a positive
tells you the shippable version is worth building.

**The control is the whole experiment.** The arms differ in competence by 3.4 points, so a stacker
that learns nothing but "usually pick tabicl-v2" would post a healthy-looking number while
collecting no diversity at all. `pick-best-arm-by-fold` is that rule made explicit: inside each CV
training fold, find the single most accurate arm and apply it to the held-out fold. It sees arm
competence and no per-row information. **Stacker minus that control is the per-row signal**, and it
is the only number here worth reading.

    uv run python scripts/stack_arms.py
    uv run python scripts/stack_arms.py --folds 10 --out reference/stack_arms.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckdb_rocket.datasets import load  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: The four arms, and how each is spelled on disk. `tabicl-v2` predates `--model` in the report
#: names, so it is filed under the device that ran it; `ts` is the same backbone on
#: anofox_forecast's 116 statistics rather than 500 ROCKET features, which is the arm that made
#: feature diversity measurable in the first place.
ARMS = {
    "tabicl-v2": ["phase5_{ds}_gpu_soft.json", "phase5_{ds}_cpu_soft.json", "phase5_{ds}_soft.json"],
    "tabpfn-v2": ["phase5_{ds}_tabpfn-v2_soft.json"],
    "orion-bix": ["phase5_{ds}_orion-bix_soft.json"],
    "tabicl-v2/ts": ["phase5_{ds}_ts_soft.json"],
}
PRIMARY = "tabicl-v2"


def read_arm(directory: Path, dataset: str, stems: list[str]) -> dict | None:
    for stem in stems:
        p = directory / stem.format(ds=dataset)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def arm_matrix(soft: dict, classes: list[str], n_test: int) -> np.ndarray:
    """One arm's posteriors as (n_test, n_classes), aligned to a shared class order.

    Aligned by class LABEL, never by position: the arms are separate runs and nothing guarantees
    they enumerate classes the same way. A positional read would silently permute one arm's
    columns, which looks like that arm being incompetent rather than like a bug.
    """
    off = soft["n_train"]
    out = np.zeros((n_test, len(classes)), dtype=float)
    idx = {c: j for j, c in enumerate(classes)}
    for k in range(n_test):
        row = soft["mean_proba"].get(str(off + k))
        if row is None:
            raise ValueError(f"no posterior for test row {k} (id {off + k})")
        for c, p in row.items():
            if str(c) in idx:
                out[k, idx[str(c)]] = float(p)
        s = out[k].sum()
        if s > 0:
            out[k] /= s
    return out


def stratified_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Fold assignment, stratified, without sklearn's minimum-class-count objection.

    UCR test splits get small -- Beef has 30 rows over 5 classes -- so a class can have fewer
    members than folds. Round-robin within each class degrades gracefully there instead of raising,
    and keeps every fold's class mix as close to the whole as the counts allow.
    """
    rng = np.random.default_rng(seed)
    assign = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        assign[idx] = np.arange(len(idx)) % k
    return [np.nonzero(assign == f)[0] for f in range(k)]


def evaluate(dataset: str, arms: dict[str, np.ndarray], y: np.ndarray,
             classes: list[str], folds: int, seed: int) -> dict:
    names = list(arms)
    per_arm_pred = {n: np.asarray([classes[j] for j in arms[n].argmax(1)]) for n in names}
    per_arm_acc = {n: float((per_arm_pred[n] == y).mean()) for n in names}

    stacked = np.hstack([arms[n] for n in names])          # (n_test, n_arms * n_classes)
    mean_pred = np.asarray([classes[j] for j in
                            np.mean([arms[n] for n in names], axis=0).argmax(1)])

    # The oracle: right if ANY arm is right. This is the bound the whole exercise is aimed at.
    right = np.vstack([per_arm_pred[n] == y for n in names])
    oracle = float(right.any(0).mean())
    none_right = float((~right.any(0)).mean())

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # THREE stackers, not one, so a negative cannot be blamed on the learner. They differ in
    # variance, which matters because a UCR test split can be 30 rows:
    #   logistic  -- 4 x n_classes coefficients, the standard stacking choice
    #   weights   -- ONE coefficient per arm, so it can only reweight the arms. Far lower
    #                variance, and on 30 rows that may beat the flexible model outright.
    #   gbm       -- trees over the same features, the only one that can express
    #                "trust arm B when arm A is unsure", which is the interaction the whole
    #                collect-the-diversity idea is banking on.
    def fit_logistic(xtr, ytr, xte):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        return m.fit(xtr, ytr).predict(xte)

    def fit_gbm(xtr, ytr, xte):
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                           learning_rate=0.1, random_state=seed)
        return m.fit(xtr, ytr).predict(xte)

    def fit_weights(xtr_idx, ytr, xte_idx):
        """Learn one non-negative weight per arm by grid search on the training folds.

        Deliberately crude -- a coarse simplex grid rather than an optimiser -- because with a
        handful of folds the search space is the thing to keep small, not the objective exact.
        """
        best_w, best_acc = None, -1.0
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        from itertools import product
        for w in product(grid, repeat=len(names)):
            if sum(w) == 0:
                continue
            mix = sum(wi * arms[n] for wi, n in zip(w, names))
            pred = np.asarray([classes[j] for j in mix.argmax(1)])
            acc = (pred[xtr_idx] == ytr).mean()
            if acc > best_acc:
                best_acc, best_w = acc, w
        mix = sum(wi * arms[n] for wi, n in zip(best_w, names))
        return np.asarray([classes[j] for j in mix.argmax(1)])[xte_idx]

    parts = stratified_folds(y, folds, seed)
    preds = {k: np.empty(len(y), dtype=object)
             for k in ("logistic", "gbm", "weights", "control")}
    for f, held in enumerate(parts):
        if len(held) == 0:
            continue
        tr = np.concatenate([p for g, p in enumerate(parts) if g != f]) if folds > 1 else held
        if len(np.unique(y[tr])) < 2:
            # Cannot fit a classifier on one class; fall back to that class, which is what any
            # honest learner would do and keeps the fold in the denominator rather than dropping it.
            for k in preds:
                preds[k][held] = y[tr][0]
            continue
        preds["logistic"][held] = fit_logistic(stacked[tr], y[tr], stacked[held])
        preds["gbm"][held] = fit_gbm(stacked[tr], y[tr], stacked[held])
        preds["weights"][held] = fit_weights(tr, y[tr], held)

        # THE CONTROL: the best arm as judged on the training folds only, applied to the held-out
        # fold. Sees competence, sees nothing per-row. Ties broken by the fixed ARMS order so the
        # control is deterministic rather than quietly favouring whichever arm numpy saw first.
        best = max(names, key=lambda n: ((per_arm_pred[n][tr] == y[tr]).mean(), -names.index(n)))
        preds["control"][held] = per_arm_pred[best][held]

    acc = {k: float((v == y).mean()) for k, v in preds.items()}
    return {
        "dataset": dataset, "n_test": int(len(y)), "n_classes": len(classes),
        "per_arm": per_arm_acc,
        "primary": per_arm_acc.get(PRIMARY),
        "best_single_on_test": max(per_arm_acc.values()),
        "mean_ensemble": float((mean_pred == y).mean()),
        "control_pick_best_arm": acc["control"],
        "stack_logistic": acc["logistic"], "stack_gbm": acc["gbm"],
        "stack_weights": acc["weights"],
        # The upper bound this gate is actually about: the best any of the three managed.
        "stacker": max(acc["logistic"], acc["gbm"], acc["weights"]),
        "oracle": oracle, "none_right": none_right,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlap", type=Path, default=ROOT / "reference" / "error_overlap.json",
                    help="which datasets to use; defaults to the 17 the overlap result covers")
    ap.add_argument("--reference", type=Path, default=ROOT / "reference")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "reference" / "stack_arms.json")
    args = ap.parse_args()

    names = [r["dataset"] for r in json.loads(args.overlap.read_text(encoding="utf-8"))["rows"]]
    rows, skipped = [], []
    for ds in names:
        softs = {a: read_arm(args.reference, ds, stems) for a, stems in ARMS.items()}
        missing = [a for a, s in softs.items() if s is None]
        if missing:
            skipped.append((ds, f"missing {', '.join(missing)}"))
            continue
        try:
            _, y_test = load(ds, "test")
        except Exception as exc:                                          # noqa: BLE001
            skipped.append((ds, f"{type(exc).__name__}: {exc}"))
            continue
        y = np.asarray([str(v) for v in y_test])
        # Union over arms, sorted, so every arm is read into the same column order.
        classes = sorted({c for s in softs.values() for c in s["classes"]})
        try:
            mats = {a: arm_matrix(s, classes, len(y)) for a, s in softs.items()}
        except ValueError as exc:
            skipped.append((ds, str(exc)))
            continue
        rows.append(evaluate(ds, mats, y, classes, args.folds, args.seed))
        r = rows[-1]
        print(f"  {ds:<30} primary {r['primary']:.4f}  ctrl {r['control_pick_best_arm']:.4f}  "
              f"stack {r['stacker']:.4f}  oracle {r['oracle']:.4f}", flush=True)

    for ds, why in skipped:
        print(f"  SKIP {ds}: {why}", file=sys.stderr)
    if not rows:
        print("no dataset had all four arms")
        return 1

    def m(key):
        return statistics.fmean(r[key] for r in rows)

    prim, ctrl, orc = m("primary"), m("control_pick_best_arm"), m("oracle")
    best, mean_ens = m("best_single_on_test"), m("mean_ensemble")
    print(f"\n=== {len(rows)} datasets, {args.folds}-fold CV within the test rows\n")
    print(f"  {'primary (tabicl-v2 alone)':<36} {prim:.4f}")
    print(f"  {'mean of the four posteriors':<36} {mean_ens:.4f}   {mean_ens - prim:+.4f}")
    print(f"  {'best arm, chosen honestly by fold':<36} {ctrl:.4f}   {ctrl - prim:+.4f}")
    for key, label in (("stack_weights", "stacker: learned arm weights"),
                       ("stack_logistic", "stacker: logistic on posteriors"),
                       ("stack_gbm", "stacker: gradient boosting")):
        print(f"  {label:<36} {m(key):.4f}   {m(key) - prim:+.4f}")
    print(f"  {'best single arm (chosen ON TEST)':<36} {best:.4f}   {best - prim:+.4f}")
    print(f"  {'per-row ORACLE over the arms':<36} {orc:.4f}   {orc - prim:+.4f}")

    # Judged against the PRIMARY, not against the best-arm control. The control turned out to be
    # worse than simply always using tabicl-v2 -- choosing an arm on a handful of folds is noisy
    # enough to lose ground -- so `stacker - control` would credit the stacker for avoiding the
    # control's own mistakes. The primary is the honest floor: it requires no choice at all, it is
    # what the pipeline already ships, and every gain has to be measured from there.
    from distill_gate import sign_test
    print(f"\n  {'stacker':<22} {'vs primary':>11} {'of oracle':>10} {'win/lose/tie':>14} {'p':>8}")
    verdicts = {}
    for key, label in (("stack_weights", "arm weights"), ("stack_logistic", "logistic"),
                       ("stack_gbm", "gradient boosting"),
                       # An ORACLE OVER LEARNERS: the max of three, chosen per dataset on the very
                       # rows it is scored on. Printed because leaving it out invites someone to
                       # compute it, and labelled because at +0.0158/p=0.049 it is the one number
                       # here that would otherwise get quoted as a result. The verdict below uses
                       # the best single learner by MEAN, which is a choice a product could make.
                       ("stacker", "ORACLE over learners")):
        gain = m(key) - prim
        frac = 100 * gain / (orc - prim) if orc > prim else float("nan")
        d = np.asarray([r[key] - r["primary"] for r in rows])
        w, l = int((d > 0).sum()), int((d < 0).sum())
        p = sign_test(d)
        verdicts[key] = {"gain_over_primary": gain, "oracle_fraction": frac / 100,
                         "wins": w, "losses": l, "p": p}
        print(f"  {label:<22} {gain:>+11.4f} {frac:>9.1f}% "
              f"{f'{w}/{l}/{len(rows) - w - l}':>14} {p:>8.4f}")

    bestkey = max(("stack_weights", "stack_logistic", "stack_gbm"),
                  key=lambda k: verdicts[k]["gain_over_primary"])
    bv = verdicts[bestkey]
    reached = bv["gain_over_primary"] > 0 and bv["p"] < 0.05
    print(f"\n  headroom to the oracle from the primary   {orc - prim:+.4f}")
    print(f"  best learner collects                     {bv['gain_over_primary']:+.4f} "
          f"({100 * bv['oracle_fraction']:.1f}% of it), p = {bv['p']:.4f}")
    verdict = (
        f"a learner with real labels and every posterior collects "
        f"{100 * bv['oracle_fraction']:.0f}% of the oracle. "
        + ("Enough to be worth building the deployable version."
           if reached and bv["oracle_fraction"] > 0.25 else
           "That is a NEGATIVE for this direction: the oracle is not reachable from these "
           "posteriors even with labels, so no confidence rule, average or router will find it."))
    print(f"\n  VERDICT: {verdict}")

    args.out.write_text(json.dumps({
        "design": {"folds": args.folds, "seed": args.seed, "arms": list(ARMS),
                   "note": "CV within test rows: an UPPER BOUND, not a deployable design. The "
                           "control picks the best arm using training folds only, so "
                           "stacker - control is the per-row signal."},
        "n_datasets": len(rows),
        "means": {"primary": prim, "mean_ensemble": mean_ens, "control": ctrl,
                  "stack_weights": m("stack_weights"), "stack_logistic": m("stack_logistic"),
                  "stack_gbm": m("stack_gbm"), "best_single_on_test": best, "oracle": orc},
        "headroom_to_oracle": orc - prim,
        "verdicts": verdicts,
        "best_learner": bestkey,
        "rows": rows, "skipped": [{"dataset": d, "why": w} for d, w in skipped],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
