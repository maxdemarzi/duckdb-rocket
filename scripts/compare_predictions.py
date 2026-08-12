"""Compare two `predictions.json` files row by row, and exit non-zero if they differ.

    uv run python scripts/compare_predictions.py before.json after.json

Equal accuracy is the weaker claim and it is not the one worth making. Two runs can reach an
identical accuracy while disagreeing about *which* rows they got right, so equal accuracy is
consistent with a change having silently altered the answer. This compares the prediction for
every id.

Written because the same check has now been needed twice for the same reason -- once to show
that splitting the test set across several `tabfm_classify` calls preserves every prediction,
and once to show that replaying one prepared plan preserves them too. Both were changes whose
whole defence was "this cannot change the answer", which is exactly the kind of claim that
should be measured rather than argued.

Ids are compared as a set first: a change that drops or duplicates rows shows up there, and
would otherwise hide inside a per-row loop that only visits the ids both files happen to share.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[int, str]:
    # utf-8-sig, not utf-8: it decodes both, and a BOM is easy to acquire moving a file between
    # the Windows box and the pod. Plain utf-8 rejects one outright.
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    by_id: dict[int, str] = {}
    for r in rows:
        rid = int(r["id"])
        if rid in by_id:
            sys.exit(f"{path}: id {rid} appears more than once")
        by_id[rid] = r["yhat"]
    return by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--show", type=int, default=5, help="disagreements to print")
    args = parser.parse_args()

    a, b = load(args.before), load(args.after)
    print(f"  {args.before.name}: {len(a)} rows")
    print(f"  {args.after.name}: {len(b)} rows")

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"  ID SETS DIFFER: {len(only_a)} only in before, {len(only_b)} only in after")
        print(f"    before-only: {only_a[:args.show]}")
        print(f"    after-only:  {only_b[:args.show]}")
        return 1

    disagree = [k for k in sorted(a) if a[k] != b[k]]
    if disagree:
        print(f"  {len(disagree)}/{len(a)} rows DISAGREE")
        for k in disagree[:args.show]:
            print(f"    id {k}: {a[k]!r} -> {b[k]!r}")
        return 1

    print(f"  IDENTICAL: {len(a)}/{len(a)} rows agree, same id set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
