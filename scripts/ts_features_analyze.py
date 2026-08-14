"""Which of the 116 statistics survive a null, across every dataset the screen ran?

The screen deliberately does not select. This does, and it does it against an explicit null, because
the first attempt at selection had none and produced eleven names that were pure chance:

    six datasets, top-12 of 116     observed 11 features at >=2   null mean 14.0   P = 0.94

The null here is the same and stays stated: if a dataset's top-K were K names drawn uniformly from
N, each feature appears in Binomial(D, K/N) of the D lists. A feature that genuinely matters on many
datasets exceeds that; one that matters on two does not, and cannot be distinguished at any D.

Three readings, weakest to strongest:

    count      how many datasets put the feature in their single-fit top-K
    stability  the same, but averaged over bootstrap resamples -- a fit is one draw
    binomial   the tail probability of that count under the null, Benjamini-Hochberg corrected
               across all N features, because testing 116 hypotheses at 0.05 buys ~6 free winners

Only the third is a result. The other two are reported so the gap between "looks important" and
"survives testing" stays visible.

    uv run python scripts/ts_features_analyze.py
    uv run python scripts/ts_features_analyze.py --in-dir data/ts_screen --top 12 --fdr 0.05
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact sum; n is at most a few hundred here."""
    if k <= 0:
        return 1.0
    return float(sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def benjamini_hochberg(pvals: list[float], fdr: float) -> tuple[list[bool], float]:
    """Which hypotheses survive at the given false-discovery rate, and the p threshold used.

    BH rather than Bonferroni: with 116 correlated features Bonferroni is needlessly harsh, and the
    question here is "give me a shortlist worth implementing", which is exactly an FDR question.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh = 0.0
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= fdr * rank / m:
            cutoff, thresh = rank, pvals[i]
    keep = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= cutoff:
            keep[i] = True
    return keep, thresh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data" / "ts_screen")
    ap.add_argument("--top", type=int, default=12, help="K for the per-dataset top-K rule")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    recs = []
    for p in sorted(args.in_dir.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "skipped" not in r and "coef_magnitude" in r:
            recs.append(r)
    if not recs:
        print(f"no usable records in {args.in_dir}")
        return 1

    names = recs[0]["ts_feature_names"]
    N, D, K = len(names), len(recs), args.top
    # Every record must describe the same feature space, or a count is summing different things.
    for r in recs:
        if r["ts_feature_names"] != names:
            print(f"{r['dataset']} has a different feature list; refusing to pool")
            return 1

    print(f"{D} datasets, {N} features, top-{K} per dataset")
    print(f"null: each feature in Binomial({D}, {K}/{N}) lists, mean {D * K / N:.1f}\n")

    mag = np.array([r["coef_magnitude"] for r in recs])          # D x N
    stab = np.array([r.get("bootstrap_topk_freq", [0.0] * N) for r in recs])
    counts = np.zeros(N, dtype=int)
    for row in mag:
        counts[np.argsort(row)[::-1][:K]] += 1

    p = K / N
    pvals = [binom_sf(int(c), D, p) for c in counts]
    keep, thresh = benjamini_hochberg(pvals, args.fdr)
    order = np.argsort(counts)[::-1]

    print(f"{'feature':38s} {'count':>6s} {'expect':>7s} {'stability':>10s} {'p':>10s}  survives")
    shown = 0
    for i in order:
        if shown >= args.show and not keep[i]:
            continue
        print(f"{names[i]:38s} {counts[i]:6d} {D * p:7.1f} {stab[:, i].mean():10.3f} "
              f"{pvals[i]:10.2e}  {'YES' if keep[i] else ''}")
        shown += 1

    n_keep = sum(keep)
    print(f"\n{n_keep} of {N} features survive BH at FDR {args.fdr}"
          f" (p <= {thresh:.2e})" if n_keep else
          f"\nNOTHING survives BH at FDR {args.fdr}: no feature's count exceeds chance")
    if n_keep:
        print("shortlist, most-selected first:")
        print("  " + ", ".join(names[i] for i in order if keep[i]))
        print("\nThis is a shortlist worth implementing. It is not a claim that the others are\n"
              "useless -- BH controls false discoveries, not false negatives -- and a feature that\n"
              "matters on three datasets out of a hundred cannot be found by this test at all.")

    # Accuracy summary, when the rocket arm ran. Independent of the selection above: these are
    # measured accuracies, and they were never the part at risk from a bad selection rule.
    have = [r for r in recs if "rocket" in r["acc"]]
    if have:
        rk = np.array([r["acc"]["rocket"] for r in have])
        print(f"\naccuracy over {len(have)} datasets, every arm against rocket:")
        print(f"  {'arm':13s} {'wins':>5s} {'ties':>5s} {'loss':>5s} {'mean':>9s} {'median':>8s} "
              f"{'best':>8s} {'worst':>8s}")
        # Worst-case matters as much as the mean here: the tuned and stacked arms buy their extra
        # wins with much larger losses, and a mean alone hides that entirely.
        for arm in ("ts", "both", "both_scaled", "both_tuned", "stack"):
            if arm not in have[0]["acc"]:
                continue
            d = np.array([r["acc"][arm] for r in have]) - rk
            print(f"  {arm:13s} {int((d > 0).sum()):5d} {int((d == 0).sum()):5d} "
                  f"{int((d < 0).sum()):5d} {d.mean():+9.4f} {float(np.median(d)):+8.4f} "
                  f"{d.max():+8.4f} {d.min():+8.4f}")

        # What CV chose is the most informative number in the whole run: it says how often the
        # correct answer is "ignore the statistics", which the accuracies only imply.
        if "chosen_block_weight" in have[0]:
            w = collections.Counter(round(r["chosen_block_weight"], 1) for r in have)
            a = collections.Counter(r["chosen_stack_alpha"] for r in have)
            print(f"  CV chose block weight {dict(sorted(w.items()))}")
            print(f"  CV chose stack alpha  {dict(sorted(a.items()))}")

        # The size of the prize, if the arm were chosen per dataset on the training split.
        arms = ("rocket", "ts", "both", "both_scaled", "both_tuned", "stack")
        present = [k for k in arms if k in have[0]["acc"]]
        oracle = np.array([max(r["acc"][k] for k in present) for r in have])
        best_of = collections.Counter(
            max(present, key=lambda k: r["acc"][k]) for r in have) if present else {}
        print(f"  oracle arm choice: mean {oracle.mean():.4f} vs rocket {rk.mean():.4f} "
              f"({oracle.mean() - rk.mean():+.4f}) -- the headroom per-dataset selection could reach")
        print(f"  best arm per dataset: {dict(best_of.most_common())}")

    if args.out:
        args.out.write_text(json.dumps({
            "n_datasets": D, "n_features": N, "top_k": K, "fdr": args.fdr,
            "null_mean_count": D * p,
            "features": [{"name": names[i], "count": int(counts[i]),
                          "stability": float(stab[:, i].mean()), "p": pvals[i],
                          "survives": bool(keep[i])} for i in order],
            "datasets": [r["dataset"] for r in recs],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
