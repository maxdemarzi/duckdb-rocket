"""Run the teacher (the DuckDB pipeline) across as many datasets as a time budget allows.

The distillation gate compares the teacher against a student on the full test split, so it needs one
archived pipeline report per dataset. Six was not enough to conclude anything -- the same sample size
that produced a feature shortlist indistinguishable from noise elsewhere in this project -- and the
fix is more datasets, not a cleverer statistic.

**Ordering is ascending by test-set size, on purpose.** Inference cost is roughly linear in test rows
(ECG5000's 4500 rows took 18.6 minutes on an A40, so about 0.25 s/row), and the gate's power comes
from the NUMBER of datasets rather than their size. Cheapest-first therefore buys the most statistical
power per pod-hour, and because each dataset writes its own report the sweep can be stopped at any
point and resumed later without losing anything.

Only datasets within `tabicl-v2`'s 10-class cap are eligible; above that the model cannot represent
the label space at all (`max_classes: 10` in its export report), which is a property of the teacher
and not something a longer run fixes.

    uv run python scripts/teacher_sweep.py --plan                    # what would run, and the cost
    uv run python scripts/teacher_sweep.py --budget-min 240 --device cuda \\
        --anofox-extension EXT --register-model-dir DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckdb_rocket.datasets import load  # noqa: E402

#: tabicl-v2's hard limit, from its export report. Not a tuning knob.
MAX_CLASSES = 10

#: Measured on an A40: ECG5000, 4500 test rows, 18.6 minutes end to end.
SECONDS_PER_TEST_ROW = 0.25


def candidates(cache_only: bool) -> list[dict]:
    """Eligible datasets with their shapes, cheapest first.

    Loading each dataset to count classes is the only reliable way to apply the cap -- the aeon
    metadata tables do not carry it -- so this is slow the first time and cached by aeon after.
    """
    from aeon.datasets.tsc_datasets import univariate_equal_length

    out = []
    for name in sorted(univariate_equal_length):
        try:
            xtr, ytr = load(name, "train")
            _, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:26s} unavailable: {str(e)[:50]}")
            continue
        n_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
        if n_classes > MAX_CLASSES:
            continue
        out.append({"dataset": name, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "n_timepoints": int(xtr.shape[-1]), "n_classes": n_classes})
    out.sort(key=lambda d: d["n_test"])
    return out


def already_done(outdir: Path, name: str) -> bool:
    """A report that recorded failures does not count as done: its accuracy is not the teacher's."""
    for p in (outdir / f"phase5_{name}_gpu.json", outdir / f"phase5_{name}.json"):
        if p.exists():
            try:
                if not json.loads(p.read_text(encoding="utf-8")).get("failures"):
                    return True
            except Exception:  # noqa: BLE001
                return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="list what would run and stop")
    ap.add_argument("--budget-min", type=float, default=240.0,
                    help="stop launching new datasets once this many minutes have elapsed")
    ap.add_argument("--max-test-rows", type=int, default=0,
                    help="skip datasets with more test rows than this (0 = no limit)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reference")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--anofox-extension", type=Path)
    ap.add_argument("--register-model-dir", type=Path)
    ap.add_argument("--test-chunk", type=int, default=128)
    ap.add_argument("--timeout-min", type=float, default=90.0,
                    help="per-dataset timeout; a single pathological dataset must not eat the budget")
    args = ap.parse_args()

    cands = candidates(cache_only=False)
    if args.max_test_rows:
        cands = [c for c in cands if c["n_test"] <= args.max_test_rows]
    todo = [c for c in cands if not already_done(args.out_dir, c["dataset"])]
    have = len(cands) - len(todo)

    est = sum(c["n_test"] for c in todo) * SECONDS_PER_TEST_ROW / 60
    print(f"{len(cands)} datasets within the {MAX_CLASSES}-class cap; {have} already have a clean "
          f"report, {len(todo)} to run")
    print(f"estimated {est:.0f} min for all of them at {SECONDS_PER_TEST_ROW} s/test-row; "
          f"budget is {args.budget_min:.0f} min")

    if args.plan:
        print(f"\n{'dataset':26s} {'n_test':>7s} {'n_tp':>6s} {'cls':>4s} {'est min':>8s} {'cum':>7s}")
        cum = 0.0
        for c in todo:
            m = c["n_test"] * SECONDS_PER_TEST_ROW / 60
            cum += m
            flag = "" if cum <= args.budget_min else "  (past budget)"
            print(f"{c['dataset']:26s} {c['n_test']:7d} {c['n_timepoints']:6d} "
                  f"{c['n_classes']:4d} {m:8.1f} {cum:7.1f}{flag}")
        return 0

    t0 = time.perf_counter()
    ran = ok = 0
    for c in todo:
        elapsed = (time.perf_counter() - t0) / 60
        if elapsed >= args.budget_min:
            print(f"\nbudget reached after {elapsed:.0f} min; {len(todo) - ran} datasets left. "
                  f"Re-run to resume -- nothing is recomputed.")
            break
        name = c["dataset"]
        cmd = [sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
               "--dataset", name, "--device", args.device,
               "--test-chunk", str(args.test_chunk),
               "--out", str(args.out_dir / f"phase5_{name}_gpu.json")]
        if args.anofox_extension:
            cmd += ["--anofox-extension", str(args.anofox_extension)]
        if args.register_model_dir:
            cmd += ["--register-model-dir", str(args.register_model_dir)]
        print(f"\n[{ran + 1}/{len(todo)}] {name}  n_test={c['n_test']} classes={c['n_classes']} "
              f"(elapsed {elapsed:.0f}/{args.budget_min:.0f} min)", flush=True)
        try:
            r = subprocess.run(cmd, timeout=args.timeout_min * 60,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {args.timeout_min:.0f} min", flush=True)
            ran += 1
            continue
        ran += 1
        tail = [l for l in r.stdout.splitlines() if "accuracy" in l or "row alignment" in l]
        for l in tail:
            print("  " + l.strip(), flush=True)
        if r.returncode == 0:
            ok += 1
        else:
            print(f"  rc={r.returncode}; last stderr: {(r.stderr or '').splitlines()[-1:]}",
                  flush=True)

    print(f"\n{ok}/{ran} clean in {(time.perf_counter() - t0) / 60:.0f} min; "
          f"reports in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
