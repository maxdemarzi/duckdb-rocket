"""Route between representations, not between models.

The overlap result says feature diversity is a real axis where model diversity is not: one backbone
over two feature families fails together 2.99x more than independence, where two backbones over one
family reach 3.74-3.78x. It also says **averaging cannot collect that** — the mean of four arms
scores 0.7619 against 0.7686 for the best single one, because the ts arm is 3.4 points weaker and
drags the mean toward itself.

An average is the wrong rule for arms of unequal competence. This project already has the right one:
escalate on a decision margin, and spend the second call only where the first was unsure. Routing was
built to escalate a *student* to a *teacher*; here the same rule escalates one **representation** to
another, and the control is built in — routing to `tabpfn-v2` is the identical rule spending the
identical budget on a different *model* instead.

    uv run python scripts/feature_route.py
    uv run python scripts/feature_route.py --primary tabicl-v2 --out reference/feature_route.json

**This is an analysis, not a serving rule, and the distinction is the same one `route_serve.py`
exists to make.** The budget is spent by sorting a whole test set and taking its least-confident
fraction, which nothing answering a request can do — a server fixes a threshold beforehand, from
out-of-fold margins on the training split. Those margins do not exist for the teacher arms here, so
the numbers below are what the rule would get with a perfectly-placed threshold and no more. The
same caveat applies to every routing curve in `distill_gate.py --route`, and the gap between an
analysis and a served batch was measured there at a few tenths of a point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duckdb_rocket.datasets import load  # noqa: E402
from error_overlap import aligned, discover  # noqa: E402

#: Escalation budgets, matching the ones every other routing table here is reported at.
BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


def margins(p: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability, per row.

    The same quantity `distill_gate.decision_margin` takes from a ridge's decision function, read
    off a probability vector instead. It is the arm's own confidence and needs no labels, which is
    what makes it usable as a router.
    """
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def route(primary: np.ndarray, alternate: np.ndarray, budget: float) -> np.ndarray:
    """Predictions from `primary`, with the least-confident `budget` fraction taken from `alternate`.

    Ties in the margin are broken by row order, which matters only when a whole batch is equally
    confident -- `Earthquakes`, where every arm predicts the majority class everywhere.
    """
    n = len(primary)
    k = int(round(budget * n))
    out = primary.argmax(1)
    if k <= 0:
        return out
    escalate = np.argsort(margins(primary), kind="stable")[:k]
    out[escalate] = alternate.argmax(1)[escalate]
    return out


def one_dataset(name: str, arms: dict[str, tuple[Path, Path]], primary: str) -> dict | None:
    if primary not in arms or len(arms) < 2:
        return None
    try:
        _, ytest = load(name, "test")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: cannot load test split ({type(e).__name__}); skipped")
        return None
    truth = np.asarray([str(v) for v in ytest])

    softs = {a: json.loads(p.read_text(encoding="utf-8")) for a, (p, _) in arms.items()}
    classes = sorted({str(c) for s in softs.values() for c in s["classes"]})
    proba = {}
    for a, s in softs.items():
        p = aligned(s, classes)
        if p is None or p.shape[0] != len(truth):
            print(f"  {name}: {a} does not align to the test split; skipped")
            return None
        proba[a] = p
    cls = np.asarray(classes)

    curves = {}
    for alt in sorted(a for a in arms if a != primary):
        curves[alt] = {f"{b:g}": float((cls[route(proba[primary], proba[alt], b)] == truth).mean())
                       for b in BUDGETS}

    # A budget-free selector, because the routed curves above only ever consult the PRIMARY's
    # confidence -- they ask "was I unsure?", never "is someone else surer?". Picking the arm with
    # the largest margin on each row is the standard way to ask the second question, and it is the
    # obvious remaining candidate for reaching the oracle. Reported over all arms and over the two
    # that differ only in representation.
    order = sorted(arms)
    def select(names: list[str]) -> float:
        m = np.stack([margins(proba[a]) for a in names])          # (arms, rows)
        pick = m.argmax(0)
        pred = np.stack([proba[a].argmax(1) for a in names])      # (arms, rows)
        return float((cls[pred[pick, np.arange(len(truth))]] == truth).mean())
    family = [a for a in order if a.split("/")[0] == primary.split("/")[0]]
    any_right = np.zeros(len(truth), dtype=bool)
    for a in order:
        any_right |= cls[proba[a].argmax(1)] == truth
    return {"dataset": name, "n_test": len(truth), "primary": primary,
            "primary_accuracy": float((cls[proba[primary].argmax(1)] == truth).mean()),
            "curves": curves,
            "max_margin_all": select(order),
            "max_margin_family": select(family) if len(family) > 1 else None,
            "best_single": max(float((cls[proba[a].argmax(1)] == truth).mean()) for a in order),
            "oracle": float(any_right.mean())}


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """(better, worse, two-sided p) over datasets, ties excluded -- the test used throughout here."""
    from scipy.stats import binomtest
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    return better, worse, float(p)


def report(rows: list[dict], primary: str) -> None:
    if not rows:
        print(f"no dataset carries {primary} alongside another arm")
        return
    alts = sorted({a for r in rows for a in r["curves"]})
    base = float(np.mean([r["primary_accuracy"] for r in rows]))
    print(f"\nROUTING BETWEEN ARMS -- {len(rows)} datasets, primary {primary} ({base:.4f})\n")
    print(f"  {'escalate to':26s} " + " ".join(f"{b:>8.0%}" for b in BUDGETS))
    for alt in alts:
        cells = []
        for b in BUDGETS:
            v = [r["curves"][alt][f"{b:g}"] for r in rows if alt in r["curves"]]
            cells.append(f"{np.mean(v):8.4f}")
        kind = "features" if alt.split("/")[0] == primary.split("/")[0] else "model"
        print(f"  {alt:20s} {kind:>5s} " + " ".join(cells))

    print(f"\n  vs the primary alone, at each budget (mean delta, sign test over datasets):")
    for alt in alts:
        line = []
        for b in BUDGETS[1:]:
            d = [r["curves"][alt][f"{b:g}"] - r["primary_accuracy"]
                 for r in rows if alt in r["curves"]]
            bt, ws, p = sign_test(d)
            line.append(f"{np.mean(d):+.4f} (p={p:.2f})")
        print(f"    {alt:24s} " + "  ".join(line))

    # The comparison the overlap result predicts: at one budget, does spending it on a different
    # REPRESENTATION beat spending it on a different MODEL?
    same_model = [a for a in alts if a.split("/")[0] == primary.split("/")[0]]
    other_model = [a for a in alts if a.split("/")[0] != primary.split("/")[0]]
    if same_model and other_model:
        print("\n  the same budget, spent two ways:")
        for b in BUDGETS[1:]:
            f = np.mean([[r["curves"][a][f"{b:g}"] for a in same_model if a in r["curves"]]
                         for r in rows])
            m = np.mean([[r["curves"][a][f"{b:g}"] for a in other_model if a in r["curves"]]
                         for r in rows])
            print(f"    {b:>4.0%}  a different representation {f:.4f}   "
                  f"a different model {m:.4f}   {f - m:+.4f}")

    # Does asking "is someone else surer?" do better than "was I unsure?" -- and does either reach
    # what a per-row oracle would?
    mean = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))  # noqa: E731
    print("\n  picking the surest arm per row, against the bounds:")
    print(f"    the primary alone                 {base:.4f}")
    fam = [r for r in rows if r.get("max_margin_family") is not None]
    if fam:
        print(f"    surest of the two representations {mean('max_margin_family'):.4f} "
              f"({mean('max_margin_family') - base:+.4f})")
    print(f"    surest of all arms                {mean('max_margin_all'):.4f} "
          f"({mean('max_margin_all') - base:+.4f})")
    print(f"    best single arm per dataset       {mean('best_single'):.4f}")
    print(f"    a per-row oracle over the arms    {mean('oracle'):.4f}")
    d = [r["max_margin_all"] - r["primary_accuracy"] for r in rows]
    bt, ws, p = sign_test(d)
    print(f"    surest-of-all vs the primary: {np.mean(d):+.4f}, {bt} better / {ws} worse, p={p:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=ROOT / "reference")
    ap.add_argument("--primary", default="tabicl-v2",
                    help="the arm that answers by default; the others are escalation targets")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    found = discover(args.reference)
    rows = [r for name, arms in sorted(found.items())
            if (r := one_dataset(name, arms, args.primary))]
    report(rows, args.primary)
    if args.out:
        args.out.write_text(json.dumps({"primary": args.primary, "budgets": list(BUDGETS),
                                        "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
