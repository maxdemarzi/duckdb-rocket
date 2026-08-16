"""Does the teacher need the whole training set as context, or just some of it?

**This is the only lever that cuts the dominant cost without an upstream change.** `tabfm_classify`
has no trained weights for the task, so every call re-encodes the labelled rows it is given --
measured at ~14 ms per training row per group, and 71-80% of a full-batch teacher call
(RESULTS.md, "What routing actually costs"). The model weights are cached by `tabfm_load`; the
*encoded context* is not, and cannot be from SQL, because `tabfm_classify(train, y, test := ...)`
takes both halves in one call and there is no prepare-then-query split to cache between.

So until upstream exposes one, the way to pay less for the context is to send less of it.

    uv run python scripts/context_sweep.py --datasets ScreenType Computers --fracs 0.25 0.5 1.0

Two things this can settle at once. Whether accuracy survives a smaller context is the question it
is for. But the same knob is also the only fix available for the datasets that cannot run at all:
a 500-to-600-row context needs more than the 29.8 GiB a CPU pod gives, so `EthanolLevel` and both
`*OutlineCorrect` datasets are unmeasurable at full context and may be measurable at half.

Reports go to distinct paths per (dataset, context size) so nothing overwrites the full-context
archive in `reference/`.
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
sys.path.insert(0, str(ROOT / "scripts"))

from duckdb_rocket.datasets import load  # noqa: E402

#: Fractions of the labelled context. 1.0 is included and run rather than read from the archive:
#: the archived numbers come from other hardware, and a timing comparison across machines is the
#: mistake this whole line of work exists to stop repeating.
FRACS = (0.25, 0.5, 1.0)


def run_one(dataset: str, max_rows: int, outdir: Path, groups: int, threads: int,
            onnx_threads: int, memory_limit: str, timeout_min: float) -> dict:
    tag = "full" if max_rows == 0 else f"ctx{max_rows}"
    out = outdir / f"phase5_{dataset}_{tag}.json"
    if out.exists():
        try:
            d = json.loads(out.read_text(encoding="utf-8"))
            if not d.get("failures"):
                d["cached"] = True
                return d
        except Exception:  # noqa: BLE001
            pass
    cmd = [sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
           "--dataset", dataset, "--model", "tabicl-v2", "--device", "cpu",
           "--n-groups", str(groups), "--num-kernels", str(250 * groups),
           "--threads", str(threads), "--onnx-threads", str(onnx_threads),
           "--memory-limit", memory_limit, "--test-chunk", "128",
           "--out", str(out)]
    if max_rows:
        cmd += ["--max-train-rows", str(max_rows)]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, timeout=timeout_min * 60, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"dataset": dataset, "failed": f"timeout after {timeout_min:.0f} min"}
    wall = time.perf_counter() - t0
    if r.returncode != 0 or not out.exists():
        # phase5 prints the exit code on the FAILED line; carry it rather than "rc=1", which is
        # what made two crashes here look like three different problems.
        why = next((l.strip() for l in r.stdout.splitlines() if "FAILED" in l), f"rc={r.returncode}")
        return {"dataset": dataset, "failed": why, "wall_seconds": wall}
    d = json.loads(out.read_text(encoding="utf-8"))
    d["wall_seconds"] = wall
    return d


def run(args) -> int:
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        try:
            _, ytr = load(name, "train")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: will not load ({type(e).__name__}); skipped")
            continue
        n_train = len(ytr)
        n_classes = len(np.unique(ytr))
        for f in args.fracs:
            # 0 means "everything" to phase5, which is not the same as asking for n_train rows --
            # it skips the subsample path entirely, so the full-context run is the unmodified one.
            max_rows = 0 if f >= 1.0 else max(n_classes, int(round(f * n_train)))
            got = max_rows or n_train
            print(f"\n  {name}  context {got}/{n_train} rows ({f:.0%})", flush=True)
            d = run_one(name, max_rows, outdir, args.groups, args.threads, args.onnx_threads,
                        args.memory_limit, args.timeout_min)
            if d.get("failed"):
                print(f"    FAILED: {d['failed']}", flush=True)
                rows.append({"dataset": name, "frac": f, "context_rows": got,
                             "n_train": n_train, "failed": d["failed"]})
                continue
            ts = d.get("time_split") or {}
            rows.append({"dataset": name, "frac": f, "context_rows": got, "n_train": n_train,
                         "accuracy": d["accuracy"], "wall_seconds": d.get("wall_seconds"),
                         "classify_seconds": ts.get("classify_seconds"),
                         "cached": bool(d.get("cached"))})
            print(f"    accuracy {d['accuracy']:.4f}   classify {ts.get('classify_seconds', 0):.0f}s"
                  f"{'  (cached)' if d.get('cached') else ''}", flush=True)

    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps({"design": "teacher at reduced labelled context",
                                              "fracs": list(args.fracs), "rows": rows}, indent=2),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def report(rows: list[dict]) -> None:
    ok = [r for r in rows if "accuracy" in r]
    if not ok:
        print("\nnothing completed")
        return
    full = {r["dataset"]: r for r in ok if r["frac"] >= 1.0}
    print(f"\nCONTEXT SIZE -- {len(full)} datasets with a full-context run to compare against\n")
    print(f"  {'dataset':28s} {'ctx':>6s} {'rows':>6s} {'accuracy':>9s} {'vs full':>9s} "
          f"{'classify s':>11s} {'vs full':>8s}")
    for name in sorted({r["dataset"] for r in ok}):
        base = full.get(name)
        for r in sorted((x for x in ok if x["dataset"] == name), key=lambda x: x["frac"]):
            da = f"{r['accuracy'] - base['accuracy']:+9.4f}" if base else " " * 9
            cs = r.get("classify_seconds") or 0
            dt = (f"{cs / base['classify_seconds']:7.2f}x"
                  if base and base.get("classify_seconds") else " " * 8)
            print(f"  {name:28s} {r['frac']:6.0%} {r['context_rows']:6d} {r['accuracy']:9.4f} "
                  f"{da} {cs:11.0f} {dt}")

    # The two questions, separated: does it get cheaper in proportion, and does it stay as accurate.
    for f in sorted({r["frac"] for r in ok if r["frac"] < 1.0}):
        sub = [r for r in ok if r["frac"] == f and r["dataset"] in full]
        if not sub:
            continue
        da = np.array([r["accuracy"] - full[r["dataset"]]["accuracy"] for r in sub])
        speed = np.array([full[r["dataset"]]["classify_seconds"] / r["classify_seconds"]
                          for r in sub if r.get("classify_seconds")
                          and full[r["dataset"]].get("classify_seconds")])
        print(f"\n  at {f:.0%} context, over {len(sub)} datasets:")
        print(f"    accuracy {da.mean():+.4f} mean, {da.min():+.4f} worst, "
              f"{int((da >= 0).sum())}/{len(da)} not worse")
        if len(speed):
            print(f"    classify {speed.mean():.2f}x faster (perfect proportionality would be "
                  f"{1 / f:.2f}x)")

    failed = [r for r in rows if r.get("failed")]
    if failed:
        print(f"\n  {len(failed)} run(s) did not complete:")
        for r in failed:
            print(f"    {r['dataset']} at {r['frac']:.0%}: {r['failed']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--fracs", nargs="*", type=float, default=list(FRACS))
    ap.add_argument("--groups", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--onnx-threads", type=int, default=14)
    ap.add_argument("--memory-limit", default="12GB")
    ap.add_argument("--timeout-min", type=float, default=40.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "context")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.fracs = sorted(args.fracs)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
