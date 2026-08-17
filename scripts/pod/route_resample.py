"""Does routing's advantage survive resampling, or was it one lucky split?

`reference/distill_route.json` says escalating the least-confident 20% of rows to the teacher beats
the student by **+0.0200** (p=0.004), and beats random-row escalation by **+0.0135** (p=0.013) —
which is the number that matters, because escalating *any* rows to a better model buys something.
Every one of those figures comes from a single train/test split, and the resample pilot measured
split luck at sd **0.0173** per dataset. So the result needs re-testing across splits before it is
quoted further.

**Why this needs a pod at all**, when the students are cheap sklearn fits: routing needs the
TEACHER's prediction on every test row, and a resampled test set contains rows the archive put in
the train half, for which no archived teacher prediction exists. So each resample needs a fresh
`tabfm_classify` pass over all 29 datasets. That is the whole cost; the students are minutes.

Two passes per resample, in order:

1. **teacher** — `phase5_pipeline.py --resample K` per dataset, in parallel, writing the
   soft-label sidecar `distill_gate.load_soft` reads.
2. **routing** — `distill_gate.py --route --resample K` against that directory, serial, cheap.

The sidecar records its resample and `assert_same_split` refuses a mismatch, so a directory mix-up
fails loudly instead of scoring the student against another split's teacher. Sizes alone cannot
catch that: a resample preserves n_train and n_test exactly.

    uv run python scripts/pod/route_resample.py --dry-run
    uv run python scripts/pod/route_resample.py --resamples 3 --jobs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def gate_datasets(gate: Path, max_student: float) -> list[str]:
    from distill_gate import gate_selection
    return sorted(gate_selection(gate, max_student))


def teacher_run(args, dataset: str, resample: int) -> dict:
    """One teacher pass. Writes phase5_<ds>.json and phase5_<ds>_soft.json into the resample's dir."""
    outdir = ROOT / "reference" / f"route_r{resample}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"phase5_{dataset}.json"
    if out.exists() and (outdir / f"phase5_{dataset}_soft.json").exists():
        return {"dataset": dataset, "resample": resample, "cached": True}
    cmd = [
        sys.executable, str(ROOT / "scripts" / "phase5_pipeline.py"),
        "--dataset", dataset, "--resample", str(resample),
        "--num-kernels", str(args.num_kernels), "--n-groups", str(args.n_groups),
        "--test-chunk", str(args.test_chunk),
        # Every one of these exists because --jobs > 1 stops a run being alone on the box. The
        # pipeline sizes threads and memory from machine-wide numbers, and shares one working
        # directory per dataset; see docs/POD.md. The memory one killed 38 of 160 runs once.
        "--threads", str(args.threads), "--onnx-threads", str(args.onnx_threads),
        "--memory-limit", args.memory_limit,
        "--workdir", str(ROOT / "data" / "route_resample" / f"{dataset}_r{resample}"),
        "--out", str(out),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"dataset": dataset, "resample": resample, "error": "timeout"}
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out.exists():
        # ONNX Runtime prints tens of thousands of "Trying to register schema" lines, so the raw
        # stderr tail is always that whatever went wrong -- it once made an OOM kill look like a
        # schema-registration bug. Keep the pipeline's own verdict and the exit code, which is
        # what says -9.
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        sig = [ln for ln in blob.splitlines()
               if ln.strip() and "Trying to register schema" not in ln
               and "schema error" not in ln.lower() and "registered from" not in ln]
        verdict = next((ln for ln in reversed(sig) if "FAILED" in ln or "Error" in ln), "")
        return {"dataset": dataset, "resample": resample, "seconds": elapsed,
                "returncode": proc.returncode, "error": (verdict or "\n".join(sig[-3:]))[:300]}
    return {"dataset": dataset, "resample": resample, "seconds": elapsed,
            "accuracy": json.loads(out.read_text(encoding="utf-8"))["accuracy"]}


def route_pass(args, resample: int) -> dict:
    """Score routing on one resample. Cheap: sklearn students plus the archived-format sidecars."""
    teacher_dir = ROOT / "reference" / f"route_r{resample}" if resample else ROOT / "reference"
    out = ROOT / "reference" / f"route_r{resample}" / "route.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "distill_gate.py"), "--route",
        "--from-gate", str(args.gate), "--max-student", str(args.max_student),
        "--resample", str(resample), "--teacher", str(teacher_dir),
        "--route-cache", str(ROOT / "data" / "route_students"),
        "--jobs", str(args.jobs), "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stdout or "")[-600:] + (proc.stderr or "")[-600:]
        return {"resample": resample, "error": tail[-600:]}
    return {"resample": resample, "report": str(out)}


def _at(pairs: list, frac: float) -> float | None:
    """Accuracy at an escalation fraction, from a [[frac, acc], ...] curve."""
    for f, a in pairs:
        if abs(f - frac) < 1e-9:
            return float(a)
    return None


def summarise(resamples: list[int], budget: float = 0.20) -> dict:
    """Pool the fixed-budget CONFIDENCE SIGNAL across resamples.

    The signal -- escalating by the student's own uncertainty MINUS escalating random rows -- is the
    only figure worth pooling. The raw gain over the student is not: handing any rows to a teacher
    that is better on average buys something, so that column rises even when the student has no idea
    which rows it is getting wrong. The whole claim rests on the difference.

    Read off the per-dataset curves rather than from any pre-aggregated field, because the report
    format is per-dataset `curve`/`control` pairs and an aggregate read from the wrong key would
    silently pool nothing.
    """
    per, rows = {}, []
    for k in resamples:
        f = (ROOT / "reference" / f"route_r{k}" / "route.json" if k
             else ROOT / "reference" / "distill_route.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        by_learner: dict[str, list[float]] = {}
        for r in d.get("rows", []):
            curve, control = r.get("curve"), r.get("control")
            if not curve or not control:
                continue
            base, esc = _at(curve, 0.0), _at(curve, budget)
            rbase, resc = _at(control, 0.0), _at(control, budget)
            if None in (base, esc, rbase, resc):
                continue
            by_learner.setdefault(r["learner"], []).append((esc - base) - (resc - rbase))
        for learner, sigs in by_learner.items():
            rows.append({"resample": k, "learner": learner, "n_datasets": len(sigs),
                         "signal": statistics.fmean(sigs),
                         "wins": sum(1 for s in sigs if s > 0)})
    for learner in sorted({r["learner"] for r in rows}):
        sig = [r["signal"] for r in rows if r["learner"] == learner]
        if sig:
            per[learner] = {
                "resamples": len(sig), "mean_signal": statistics.fmean(sig),
                "sd_signal": statistics.stdev(sig) if len(sig) > 1 else None,
                "min": min(sig), "max": max(sig),
                "all_positive": all(s > 0 for s in sig),
            }
    return {"budget": budget, "rows": rows, "per_learner": per}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=3, help="score resamples 1..N (0 is archived)")
    ap.add_argument("--gate", type=Path, default=ROOT / "reference" / "distill_gate.json")
    ap.add_argument("--max-student", type=float, default=0.90)
    ap.add_argument("--num-kernels", type=int, default=10_000)
    ap.add_argument("--n-groups", type=int, default=40)
    ap.add_argument("--test-chunk", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--onnx-threads", type=int, default=3)
    ap.add_argument("--memory-limit", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=10_800)
    ap.add_argument("--out", type=Path, default=ROOT / "reference" / "route_resample.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise-only", action="store_true",
                    help="re-read the per-resample reports and pool them; no pod time")
    args = ap.parse_args()

    names = gate_datasets(args.gate, args.max_student)
    ks = list(range(1, args.resamples + 1))

    if args.summarise_only:
        print(json.dumps(summarise([0] + ks), indent=2)[:4000])
        return 0

    if not args.memory_limit:
        total = None
        for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                raw = Path(f).read_text().strip()
                if raw != "max":
                    total = int(raw)
                    break
            except (OSError, ValueError):
                continue
        args.memory_limit = (f"{max(1, int(total * 0.6 / max(1, args.jobs)) // (1 << 30))}GB"
                             if total and total < (1 << 62) else "8GB")

    print(f"{len(names)} datasets x {len(ks)} resamples = {len(names) * len(ks)} teacher runs")
    print(f"memory_limit {args.memory_limit} per run x {args.jobs} jobs")
    if args.dry_run:
        known, unknown = [], 0
        for n in names:
            hit = next((p for p in (ROOT / "reference").glob(f"phase5_{n}.json")), None) \
                or next((p for p in (ROOT / "reference").glob(f"phase5_{n}_cpu.json")), None)
            if hit:
                known.append(json.loads(hit.read_text(encoding="utf-8"))["seconds"])
            else:
                unknown += 1
        if known:
            per = statistics.fmean(known)
            print(f"  archived mean {per:.0f}s over {len(known)} datasets"
                  f"{f' ({unknown} unmeasured)' if unknown else ''}")
            print(f"  ~{len(names) * len(ks) * per / 3600 / max(1, args.jobs):.1f} h "
                  f"at --jobs {args.jobs}")
        print(f"  then {len(ks)} routing passes, minutes each")
        return 0

    # The archive downloads and extracts on first use and that is not safe from several processes
    # at once; the symptom only ever hits the FIRST run of each dataset, which a fresh pod always
    # performs. Serially, before anything is dispatched.
    print(f"warming the dataset cache for {len(names)} datasets", flush=True)
    from duckdb_rocket.datasets import load as _load
    for n in names:
        try:
            _load(n, "train"), _load(n, "test")
        except Exception as exc:                                          # noqa: BLE001
            print(f"  {n}: FAILED TO LOAD -- {type(exc).__name__}: {exc}", file=sys.stderr)

    teacher, started = [], time.perf_counter()
    jobs = [(n, k) for k in ks for n in names]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futs = {pool.submit(teacher_run, args, n, k): (n, k) for n, k in jobs}
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            teacher.append(r)
            tag = "cached" if r.get("cached") else (
                f"acc={r['accuracy']:.4f} {r['seconds']:.0f}s" if "accuracy" in r
                else f"FAILED: {str(r.get('error'))[:160]}")
            print(f"  [{i}/{len(jobs)}] {r['dataset']} r{r['resample']}  {tag}", flush=True)
            args.out.write_text(json.dumps({"teacher": teacher}, indent=2), encoding="utf-8")

    bad = [r for r in teacher if "error" in r]
    print(f"\nteacher passes: {len(teacher) - len(bad)} ok, {len(bad)} failed "
          f"({(time.perf_counter() - started) / 60:.0f} min)")

    routing = []
    for k in ks:
        print(f"\nrouting on resample {k}", flush=True)
        r = route_pass(args, k)
        routing.append(r)
        print(f"  {'wrote ' + r['report'] if 'report' in r else 'FAILED: ' + str(r.get('error'))[:400]}")
        args.out.write_text(json.dumps({"teacher": teacher, "routing": routing}, indent=2),
                            encoding="utf-8")

    summary = summarise([0] + ks)
    args.out.write_text(json.dumps({"teacher": teacher, "routing": routing, "summary": summary},
                                   indent=2), encoding="utf-8")
    print(f"\n=== the 20% signal, per resample")
    for row in sorted(summary["rows"], key=lambda r: (r["learner"], r["resample"])):
        s = row["signal"]
        print(f"  {row['learner']:<14} r{row['resample']}  signal "
              f"{s:+.4f}" if s is not None else f"  {row['learner']} r{row['resample']}  --")
    for learner, v in summary["per_learner"].items():
        sd = f"{v['sd_signal']:.4f}" if v["sd_signal"] is not None else "n/a"
        print(f"\n  {learner}: mean signal {v['mean_signal']:+.4f} over {v['resamples']} "
              f"resamples, sd {sd}, range {v['min']:+.4f}..{v['max']:+.4f}, "
              f"all positive: {v['all_positive']}")
    print(f"\nwrote {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
