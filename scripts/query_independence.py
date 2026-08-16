"""Is a query row's prediction independent of the other query rows in its call?

That is the property a cached support set needs. If it holds, the support half of the forward pass
can be computed once and reused across calls, because nothing in it depends on which queries arrive.
If it fails, no cache is exact -- and neither is chunking the query set, which anofox-tabfm already
relies on.

Trained weights, via the sklearn API. An earlier version of this used a randomly initialised model
and every difference came out at exactly 0.0 -- including the control -- because a random TabICL
emits a constant vector regardless of input. A degenerate model passes an independence test
trivially, so the controls below are the point of the script, not decoration.

    uv run --with tabicl python query_independence.py
"""

from __future__ import annotations

import numpy as np
from tabicl import TabICLClassifier


def main() -> int:
    rng = np.random.default_rng(0)
    S, Q, H = 64, 12, 8
    xtr = rng.normal(size=(S, H))
    ytr = (xtr[:, 0] + 0.5 * xtr[:, 1] > 0).astype(int)
    xq = rng.normal(size=(Q, H))

    clf = TabICLClassifier(random_state=0, device="cpu").fit(xtr, ytr)
    whole = clf.predict_proba(xq)

    # --- controls: the model must actually be using its inputs -------------------------------
    varies_by_row = float(np.abs(whole - whole[0]).max())
    other_support = TabICLClassifier(random_state=0, device="cpu").fit(
        rng.normal(size=(S, H)), ytr).predict_proba(xq)
    support_matters = float(np.abs(whole - other_support).max())
    flipped_labels = TabICLClassifier(random_state=0, device="cpu").fit(
        xtr, 1 - ytr).predict_proba(xq)
    labels_matter = float(np.abs(whole - flipped_labels).max())

    # --- the claim ---------------------------------------------------------------------------
    split = np.vstack([clf.predict_proba(xq[:7]), clf.predict_proba(xq[7:])])
    d_split = float(np.abs(whole - split).max())
    d_solo = float(np.abs(whole[3] - clf.predict_proba(xq[3:4])[0]).max())

    print(f"controls  predictions vary across query rows : {varies_by_row:.3e}")
    print(f"          a different support set changes them: {support_matters:.3e}")
    print(f"          flipped support labels change them  : {labels_matter:.3e}")
    print(f"claim     12 queries in one call vs split 7+5 : {d_split:.3e}")
    print(f"          query #3 in a batch of 12 vs alone  : {d_solo:.3e}")

    ok = (varies_by_row > 1e-3 and support_matters > 1e-3 and labels_matter > 1e-3
          and d_split < 1e-5 and d_solo < 1e-5)
    print(f"\n=> {'query rows are independent given the support' if ok else 'INCONCLUSIVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
