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

sys.path.insert(0, str(ROOT / "scripts"))
from phase5_pipeline import LABELLERS, MODEL  # noqa: E402

#: The hard limit, and it is not tabicl's alone: every one of the six registered models reports
#: max_classes = 10. v2026.08.15 made the ceiling per-model rather than hardcoded, which changed
#: nothing here -- Phoneme's 39 classes are out of reach for the whole family, not for one model.
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


def report_name(name: str, model: str, features: str = "rocket") -> str:
    """Where a run's report lands, kept distinct per model.

    Every archived report so far is tabicl-v2 and is named without a model, so that spelling is
    preserved: renaming them would orphan reference/distill_gate.json and every result built on it.
    A second labeller gets its own suffix instead, and the two never collide.
    """
    if features != "rocket":
        # The feature family is part of the run's identity, not a variation on it: a ts arm and a
        # rocket arm of the SAME model are the comparison, so naming them alike would have the
        # second overwrite the first. The existing phase5_<name>_both.json files fix the spelling.
        return (f"phase5_{name}_{features}.json" if model == MODEL
                else f"phase5_{name}_{model}_{features}.json")
    return f"phase5_{name}_gpu.json" if model == MODEL else f"phase5_{name}_{model}.json"


def already_done(outdir: Path, name: str, model: str = MODEL,
                 features: str = "rocket") -> bool:
    """A report that recorded failures does not count as done: its accuracy is not the teacher's."""
    cands = ((outdir / report_name(name, model, features),) if model != MODEL or features != "rocket"
             else (outdir / f"phase5_{name}_gpu.json", outdir / f"phase5_{name}.json"))
    for p in cands:
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
    ap.add_argument("--model", default=MODEL, choices=LABELLERS,
                    help="which in-context model to run as the teacher")
    ap.add_argument("--datasets", nargs="*",
                    help="restrict to these datasets; without it, everything inside the class cap")
    ap.add_argument("--from-gate", type=Path,
                    help="restrict to a gate run's unsaturated subgroup, so a second labeller is "
                         "measured on exactly the datasets the first one's gate opened on")
    ap.add_argument("--max-student", type=float, default=0.90)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--anofox-extension", type=Path)
    ap.add_argument("--register-model-dir", type=Path)
    ap.add_argument("--test-chunk", type=int, default=128)
    # Passed through rather than left at phase5's defaults because those defaults size themselves
    # from the VISIBLE cores, and several of these running at once each claiming the whole box is
    # what killed a pod sweep before. Running N shards concurrently means setting these so that
    # N x threads x onnx-threads lands near the vCPU count, not above it.
    ap.add_argument("--threads", type=int, help="DuckDB threads per phase5 run")
    ap.add_argument("--onnx-threads", type=int, help="ONNX intra-op threads per session")
    # Memory has exactly the same trap as threads and cost more. phase5's default is 70% of the
    # box, so N concurrent shards ask for 70N% of it: four shards on a 124GB pod were killed by the
    # OOM killer (exit -9) in 18 seconds each, while a single run of the same dataset succeeded in
    # 598. Note the limit only governs DuckDB -- ONNX allocates outside it -- so this bounds the
    # part that can be bounded and the shard count has to do the rest.
    ap.add_argument("--memory-limit", help="DuckDB memory_limit per phase5 run, e.g. '24GB'. "
                                           "Set this whenever more than one sweep runs at once.")
    ap.add_argument("--features", default="rocket", choices=("rocket", "ts", "both"),
                    help="which feature families the classifier sees. 'ts' is anofox_forecast's "
                         "116 statistics and needs --n-groups 1, since there is no kernel bank to "
                         "slice and 40 identical groups would be 40x the cost for one answer.")
    ap.add_argument("--n-groups", type=int,
                    help="override phase5's 40 groups. Required to be 1 for --features ts.")
    ap.add_argument("--per-group-soft", action="store_true",
                    help="archive each group's own probabilities, which makes the group-count "
                         "sweep exact from a single run (scripts/perf_levers.py --groups)")
    ap.add_argument("--timeout-min", type=float, default=90.0,
                    help="per-dataset timeout; a single pathological dataset must not eat the budget")
    args = ap.parse_args()
    # phase5 refuses n_groups != 1 for ts features, and it is right to: the groups exist to slice a
    # kernel bank, and there is no bank here. Caught before the sweep rather than once per dataset.
    if args.features == "ts" and (args.n_groups or 40) != 1:
        ap.error("--features ts needs --n-groups 1: the 116 statistics do not come from a kernel "
                 "bank, so 40 groups would compute the same features 40 times for one answer")

    cands = candidates(cache_only=False)
    if args.from_gate:
        rows = json.loads(args.from_gate.read_text(encoding="utf-8"))["rows"]
        keep = {r["dataset"] for r in rows
                if r.get("students") and max(r["students"].values()) < args.max_student}
        cands = [c for c in cands if c["dataset"] in keep]
    if args.datasets:
        want = set(args.datasets)
        cands = [c for c in cands if c["dataset"] in want]
    if args.max_test_rows:
        cands = [c for c in cands if c["n_test"] <= args.max_test_rows]
    todo = [c for c in cands
            if not already_done(args.out_dir, c["dataset"], args.model, args.features)]
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
               "--model", args.model,
               "--test-chunk", str(args.test_chunk),
               "--out", str(args.out_dir / report_name(name, args.model, args.features))]
        if args.features != "rocket":
            cmd += ["--features", args.features]
        if args.n_groups:
            cmd += ["--n-groups", str(args.n_groups)]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        if args.onnx_threads:
            cmd += ["--onnx-threads", str(args.onnx_threads)]
        if args.memory_limit:
            cmd += ["--memory-limit", args.memory_limit]
        if args.per_group_soft:
            cmd += ["--per-group-soft"]
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
        # "FAILED" is in this filter because phase5 prints the one line that identifies a crash --
        # "FAILED after 18.4s (exit -9)" -- to stdout, and a filter of accuracy/row-alignment drops
        # exactly the runs that have neither. That turned a SIGKILL into "rc=1; last stderr: ['']"
        # twice, and both times the visible evidence pointed somewhere else.
        tail = [l for l in r.stdout.splitlines()
                if "accuracy" in l or "row alignment" in l or "FAILED" in l]
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
