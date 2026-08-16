"""Do two backbones fail in the same places?

Every accuracy number in this project comes from one in-context model, and the reason is that
`tabicl-v2` is what loaded. Routing already captures about 63% of that model's advantage over the
student, so the remaining headroom is the *teacher's* ceiling -- and the standard escape, an
ensemble of labellers, only helps if the labellers are wrong in different places.

**That makes overlap the measurement, not accuracy.** Two models of one `icl-transformer` family
scoring one set of ROCKET features can be more accurate and still fail together on the same rows,
and an ensemble that inherits the failure mode inherits the whole problem. So this reports:

* how much more often two models are wrong *together* than independence would predict,
* how many rows **no** model gets right -- the floor an ensemble cannot go below,
* how far a perfect oracle over them would reach -- the ceiling it cannot exceed,
* and what averaging their probabilities actually gets, which lies somewhere between.

    uv run python scripts/error_overlap.py
    uv run python scripts/error_overlap.py --reference reference --out reference/error_overlap.json

Reads the `*_soft.json` sidecars that `phase5_pipeline.py` already writes, so it costs nothing to
run and needs no pod. The soft labels are per-row probabilities averaged over the groups, and
`note` in each file fixes the alignment: test row k of the dataset's test split is id n_train + k.

**Configurations are checked rather than assumed.** A comparison between a 40-group run and a
10-group one would measure the group count, not the backbone, so pairs whose `num_kernels`,
`n_groups` or `seed` differ are dropped with a message instead of quietly averaged in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckdb_rocket.datasets import load  # noqa: E402

#: The fields that have to agree before two runs can be compared as backbones. `seed` is in here
#: because the kernel bank depends on it, so two seeds are two different feature sets.
COMPARED_CONFIG = ("num_kernels", "n_groups", "kernels_per_group", "n_estimators", "seed")

SOFT = re.compile(r"^phase5_(?P<dataset>.+?)_(?P<tag>[A-Za-z0-9.\-]+)_soft\.json$")


def discover(reference: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """{dataset: {model: (soft, report)}}, keyed on the run's own `model` field.

    Not on the filename tag: `_cpu` and `_gpu` are both `tabicl-v2` and differ only in device, so
    keying on the tag would compare a model against itself and call it an ensemble.
    """
    out: dict[str, dict[str, tuple[Path, Path]]] = {}
    for soft in sorted(reference.glob("phase5_*_soft.json")):
        m = SOFT.match(soft.name)
        if not m:
            continue
        report = soft.with_name(soft.name.replace("_soft.json", ".json"))
        if not report.exists():
            continue
        d = json.loads(soft.read_text(encoding="utf-8"))
        model = d.get("model")
        if not model:
            continue
        # A dataset can hold the same model twice (cpu and gpu). Keep one; they are the same
        # computation and RESULTS.md records them agreeing.
        out.setdefault(m.group("dataset"), {}).setdefault(model, (soft, report))
    return out


def aligned(soft: dict, classes: list[str]) -> np.ndarray | None:
    """This run's probabilities as (n_test, len(classes)), aligned by id and by class name.

    `mean_proba` is {row id: {class: probability}}, and both keys are used rather than either
    being taken as positional. Two models can enumerate their classes differently -- the order is
    read off each model's own output -- and indexing one model's probabilities with another's
    ordering permutes the labels while leaving every shape intact, which is the kind of wrong that
    still produces a plausible accuracy. Row ids are checked against n_train + k for the same
    reason: that offset took three attempts to get right in the pipeline itself.
    """
    mp = soft["mean_proba"]
    if not isinstance(mp, dict):
        return None
    n_train, n_test = int(soft["n_train"]), int(soft["n_test"])
    want = [str(n_train + k) for k in range(n_test)]
    if set(mp) != set(want):
        return None
    out = np.zeros((n_test, len(classes)), dtype=np.float64)
    for k, key in enumerate(want):
        row = mp[key]
        for j, c in enumerate(classes):
            out[k, j] = float(row.get(c, 0.0))
    return out


def one_dataset(name: str, entries: dict[str, tuple[Path, Path]]) -> dict | None:
    models = sorted(entries)
    if len(models) < 2:
        return None
    softs = {m: json.loads(entries[m][0].read_text(encoding="utf-8")) for m in models}
    reports = {m: json.loads(entries[m][1].read_text(encoding="utf-8")) for m in models}

    base = reports[models[0]].get("config", {})
    for m in models[1:]:
        cfg = reports[m].get("config", {})
        diff = [k for k in COMPARED_CONFIG if base.get(k) != cfg.get(k)]
        if diff:
            print(f"  {name}: {models[0]} and {m} differ on {', '.join(diff)}; skipped")
            return None

    try:
        _, ytest = load(name, "test")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: cannot load test split ({type(e).__name__}); skipped")
        return None
    truth = np.asarray([str(v) for v in ytest])

    n = len(truth)
    for m in models:
        if int(softs[m]["n_test"]) != n:
            print(f"  {name}: {m} covers {softs[m]['n_test']} rows, not the {n}-row test split; "
                  f"skipped")
            return None

    classes = sorted({str(c) for m in models for c in softs[m]["classes"]})
    proba = {}
    for m in models:
        p = aligned(softs[m], classes)
        if p is None:
            print(f"  {name}: {m}'s soft labels are not keyed by the expected row ids; skipped")
            return None
        proba[m] = p
    pred = {m: np.asarray(classes)[proba[m].argmax(1)] for m in models}
    right = {m: pred[m] == truth for m in models}

    # The two bounds an ensemble lives between: every row some model gets right, and every row
    # none does. Neither depends on how the ensemble combines them.
    any_right = np.zeros(n, dtype=bool)
    for m in models:
        any_right |= right[m]

    # And what averaging actually gets, which is the cheapest combination rule there is.
    mean_proba = np.mean([proba[m] for m in models], axis=0)
    ens = np.asarray(classes)[mean_proba.argmax(1)] == truth

    pairs = []
    for a, b in combinations(models, 2):
        wa, wb = ~right[a], ~right[b]
        both = float((wa & wb).mean())
        indep = float(wa.mean() * wb.mean())
        pairs.append({
            "models": [a, b],
            "accuracy": [float(right[a].mean()), float(right[b].mean())],
            "both_wrong": both,
            "both_wrong_if_independent": indep,
            # >1 means correlated failure, which is what makes an ensemble disappointing.
            "excess": float(both / indep) if indep > 0 else None,
            "disagree": float((pred[a] != pred[b]).mean()),
        })
    return {
        "dataset": name, "n_test": n, "models": models, "n_classes": len(classes),
        "accuracy": {m: float(right[m].mean()) for m in models},
        "oracle": float(any_right.mean()),
        "none_right": float(1 - any_right.mean()),
        "mean_proba_ensemble": float(ens.mean()),
        "best_single": max(float(right[m].mean()) for m in models),
        "pairs": pairs,
    }


def report(rows: list[dict]) -> None:
    if not rows:
        print("no dataset has two comparable models; nothing to compare")
        return
    models = sorted({m for r in rows for m in r["models"]})
    print(f"\nERROR OVERLAP -- {len(rows)} datasets, {len(models)} backbones "
          f"({', '.join(models)})\n")
    print(f"{'dataset':30s} " + " ".join(f"{m:>11s}" for m in models)
          + f" {'ensemble':>9s} {'oracle':>7s} {'none':>6s}")
    for r in sorted(rows, key=lambda r: r["dataset"]):
        cells = " ".join(f"{r['accuracy'].get(m, float('nan')):11.4f}" for m in models)
        print(f"{r['dataset'][:30]:30s} {cells} {r['mean_proba_ensemble']:9.4f} "
              f"{r['oracle']:7.4f} {r['none_right']:6.4f}")
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    print(f"\n{'mean':30s} " + " ".join(
        f"{np.mean([r['accuracy'][m] for r in rows if m in r['accuracy']]):11.4f}" for m in models)
        + f" {mean('mean_proba_ensemble'):9.4f} {mean('oracle'):7.4f} {mean('none_right'):6.4f}")

    print("\n  what an ensemble has to beat, and what it cannot reach:")
    best = float(np.mean([r["best_single"] for r in rows]))
    print(f"    best single backbone per dataset  {best:.4f}")
    print(f"    averaging their probabilities     {mean('mean_proba_ensemble'):.4f} "
          f"({mean('mean_proba_ensemble') - best:+.4f})")
    print(f"    a perfect oracle over them        {mean('oracle'):.4f} "
          f"({mean('oracle') - best:+.4f} of headroom)")
    print(f"    rows no backbone gets right       {mean('none_right'):.4f}")

    # Per pair, not pooled. With three backbones the pooled number hides the only comparison that
    # separates "these architectures are alike" from "the ROCKET features are the ceiling": if a
    # third architecture overlaps the first two as much as they overlap each other, the common
    # element is the features they all read, not the lineage two of them share.
    print("\n  are they wrong in the same places?  (per pair, over the datasets holding both)")
    print(f"    {'pair':34s} {'both wrong':>10s} {'if indep.':>10s} {'excess':>8s} "
          f"{'disagree':>9s}  n")
    by_pair: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        for p in r["pairs"]:
            by_pair.setdefault(tuple(p["models"]), []).append(p)
    for pair, ps in sorted(by_pair.items()):
        ex = [p["excess"] for p in ps if p["excess"] is not None]
        print(f"    {' vs '.join(pair)[:34]:34s} "
              f"{np.mean([p['both_wrong'] for p in ps]):10.4f} "
              f"{np.mean([p['both_wrong_if_independent'] for p in ps]):10.4f} "
              f"{np.mean(ex) if ex else float('nan'):7.2f}x "
              f"{np.mean([p['disagree'] for p in ps]):9.4f}  {len(ps)}")
    print("\n  An excess above 1 means the failures are correlated: the second model is most often"
          "\n  wrong exactly where the first one is, which is the case that makes an ensemble"
          "\n  disappointing however it combines them.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=ROOT / "reference")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    found = discover(args.reference)
    rows = [r for name, e in sorted(found.items()) if (r := one_dataset(name, e))]
    report(rows)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
