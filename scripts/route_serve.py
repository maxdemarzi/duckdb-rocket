"""Routing at serving time: deploy the artifacts, serve a batch, report where the time went.

`distill_gate.py --route` measured routing by sorting a whole test set and escalating its least
confident fraction. Nothing serving a request can do that. This is the same rule as a system would
actually run it -- a threshold on one row's margin, fixed before the batch arrives -- executed
end to end through the real extension, so the numbers include the parts an analysis skips.

    uv run python scripts/route_serve.py deploy --dataset ScreenType --target 0.20
    uv run python scripts/route_serve.py serve  --dataset ScreenType --batch 128

**What is deployed.** Three things, and the first is the one worth noticing:

* the ROCKET features of the labelled training rows -- which serve double duty as the matrix the
  ridge was fit on AND as the teacher's in-context training table. The teacher has no trained weights
  for your task; `tabfm_classify` takes the labelled rows as context on every call, so deploying it
  means deploying the training data with it.
* the ridge head: a scaler and a coefficient matrix, kilobytes.
* one float: the margin threshold, taken as a quantile of out-of-fold margins on the train split.

**One feature computation serves both models.** The teacher's 40 groups of 250 kernels are slices
[0,250) ... [9750,10000) of exactly the 10,000-kernel bank the student's ridge uses -- verified to
1.8e-15 against `rocket_transform(values, 250, seed, offset)`. So the student reads all 20,000
features and the teacher reads 500 at a time, from one transform.

**Why the teacher call reuses `phase5_pipeline.build_sql`.** The escalated rows are just a small test
split against the same training context, so the generated pipeline is exactly right for them -- and
it carries the id-recovery key, the kernel-bank fingerprint and the `features_check` guard, every one
of which exists because something silently produced a plausible wrong answer without it. Writing a
leaner serving query would drop those.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402
from duckdb_rocket.shells import built_shell  # noqa: E402

import phase5_pipeline as p5  # noqa: E402
from distill_gate import ALPHAS, decision_margin, oof_margins  # noqa: E402


def deploy(dataset: str, target: float, n_kernels: int, seed: int, folds: int,
           out: Path) -> dict:
    """Fit the student, choose the threshold, write everything a server needs."""
    xtr, ytr = load(dataset, "train")
    xtr = normalize_series(xtr)
    bank = generate_kernels(seed, xtr.shape[-1], n_kernels)
    ftr = transform(xtr, bank)
    scaler = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(scaler.transform(ftr), ytr)

    # The threshold comes from margins the model produced on rows it had NOT seen. A fitted model is
    # systematically surer of its own training rows, so an in-sample quantile sits too high and the
    # server would escalate too little.
    margins = oof_margins(dataset, "rocket+ridge", seed, folds, str(ROOT / "data" / "oof_margins"))
    threshold = float(np.quantile(margins, target))

    out.mkdir(parents=True, exist_ok=True)
    (out / "student.pkl").write_bytes(pickle.dumps({"scaler": scaler, "clf": clf}))
    meta = {"dataset": dataset, "seed": seed, "n_kernels": n_kernels, "n_timepoints":
            int(xtr.shape[-1]), "target": target, "threshold": threshold, "folds": folds,
            "n_train": int(len(ytr)), "classes": [str(c) for c in clf.classes_]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"deployed {dataset} -> {out}")
    print(f"  student: {n_kernels} kernels, {ftr.shape[1]} features, ridge over "
          f"{len(clf.classes_)} classes")
    print(f"  threshold {threshold:.4f} at a {target:.0%} target, from {folds}-fold out-of-fold "
          f"margins over {len(margins)} training rows")
    return meta


def student_predict(meta: dict, art: Path, x: np.ndarray):
    """Predictions and margins. This is the whole 80% path: a transform and a matrix multiply."""
    d = pickle.loads((art / "student.pkl").read_bytes())
    bank = generate_kernels(meta["seed"], meta["n_timepoints"], meta["n_kernels"])
    f = d["scaler"].transform(transform(x, bank))
    dec = d["clf"].decision_function(f)
    return d["clf"].predict(f), decision_margin(dec)


def teacher_predict(dataset: str, idx: np.ndarray, workdir: Path, n_groups: int,
                    num_kernels: int, seed: int, shell: Path) -> np.ndarray:
    """The teacher on the escalated rows only, through the real extension.

    Built as a one-off dataset whose test split IS the escalated batch, so `build_sql` produces
    exactly the right pipeline: 40 groups against the same labelled context, probabilities averaged,
    argmax. Its correctness guards come along -- the full-vector id recovery, the bank fingerprint,
    and the features_check that catches a silently dropped feature name.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    xtr, ytr = load(dataset, "train")
    xte, _ = load(dataset, "test")
    xtr, xte = normalize_series(xtr), normalize_series(xte)
    xq = xte[idx]
    n_train, n_q = len(ytr), len(xq)

    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / "raw.parquet"
    pq.write_table(pa.table({
        "id": pa.array(np.arange(n_train + n_q), type=pa.int64()),
        "split": pa.array(["train"] * n_train + ["test"] * n_q),
        "label": pa.array([str(v) for v in ytr] + ["?"] * n_q),
        "values": pa.array(list(xtr) + list(xq), type=pa.list_(pa.float64())),
    }), raw)

    cfg = RocketPFNConfig(num_kernels=num_kernels, n_groups=n_groups, seed=seed, n_estimators=1)
    cfg.validate()
    meta = {"dataset": dataset, "n_train": n_train, "n_test": n_q, "n_channels": 1,
            "n_timepoints": int(xtr.shape[-1]), "multivariate": False,
            "raw_parquet": raw.as_posix()}
    sql = p5.build_sql(cfg, meta, workdir, 4, "8GB", workdir, 128, 4, device="cpu")
    (workdir / "serve.sql").write_text(sql, encoding="utf-8")
    # encoding is explicit: DuckDB's box-drawing output is UTF-8 and Windows would otherwise decode
    # it as cp1252 and raise mid-run, after the work has been done.
    r = subprocess.run([str(shell), "-c", f".read {(workdir / 'serve.sql').as_posix()}"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    pred_path = workdir / "predictions.json"
    if not pred_path.exists():
        raise RuntimeError(f"the teacher produced no predictions.\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    by_id = {int(p["id"]): str(p["yhat"])
             for p in json.loads(pred_path.read_text(encoding="utf-8"))}
    # Test row k is id n_train + k, asserted rather than assumed: the same offset that took three
    # attempts to get right in the pipeline itself.
    missing = [k for k in range(n_q) if (n_train + k) not in by_id]
    if missing:
        raise RuntimeError(f"the teacher returned no prediction for {len(missing)} escalated "
                           f"row(s), first at id {n_train + missing[0]}")
    return np.array([by_id[n_train + k] for k in range(n_q)])


def serve(dataset: str, art: Path, batch: int, n_groups: int, seed: int, shell: Path,
          workdir: Path, compare: bool = False) -> int:
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    xte, yte = load(dataset, "test")
    xte_n = normalize_series(xte)
    take = min(batch, len(yte))
    x, y = xte_n[:take], yte[:take]

    t0 = time.perf_counter()
    spred, margin = student_predict(meta, art, x)
    t_student = time.perf_counter() - t0

    esc = margin < meta["threshold"]
    idx = np.nonzero(esc)[0]
    print(f"\nbatch of {take} rows from {dataset}")
    print(f"  student answered in {t_student * 1000:.0f} ms "
          f"({t_student / take * 1000:.2f} ms/row, features + ridge)")
    print(f"  escalating {esc.sum()}/{take} = {esc.mean():.1%} "
          f"(threshold {meta['threshold']:.4f}, target {meta['target']:.0%})")

    final = np.asarray(spred, dtype=object).copy()
    t_teacher = 0.0
    if len(idx):
        t0 = time.perf_counter()
        tpred = teacher_predict(dataset, idx, workdir, n_groups, meta["n_kernels"], seed, shell)
        t_teacher = time.perf_counter() - t0
        final[idx] = tpred
        print(f"  teacher answered {len(idx)} rows in {t_teacher:.1f} s "
              f"({t_teacher / len(idx) * 1000:.0f} ms/row, {n_groups} groups)")
    else:
        print("  no row fell below the threshold, so this batch cost the student alone")

    truth = np.asarray(y, dtype=object)
    acc_routed = float((final == truth).mean())
    acc_student = float((np.asarray(spred, dtype=object) == truth).mean())
    total = t_student + t_teacher
    print(f"\n  routed   {acc_routed:.4f}   {total:.1f} s total")
    print(f"  student  {acc_student:.4f}   {t_student:.1f} s   (what you would have had for free)")

    # The cost claim needs all three arms on ONE box at ONE time. Assembling it from an archived
    # run instead is how a 27 s figure from a 96-core CUDA node got compared against a contended
    # 8-core CPU measurement and produced a "138x" that meant nothing.
    if compare and len(idx):
        t0 = time.perf_counter()
        tall = teacher_predict(dataset, np.arange(take), workdir / "all", n_groups,
                               meta["n_kernels"], seed, shell)
        t_all = time.perf_counter() - t0
        acc_teacher = float((np.asarray(tall, dtype=object) == truth).mean())
        print(f"  teacher  {acc_teacher:.4f}   {t_all:.1f} s   (every row, same box, same moment)")
        print(f"\n  routing spent {total / t_all:.0%} of the teacher-everywhere time for "
              f"{(acc_routed - acc_student) / (acc_teacher - acc_student):.0%} of its accuracy gain"
              if acc_teacher != acc_student else "\n  the teacher and student tie on this batch")
        print(f"  per-row: {t_student / take * 1000:.1f} ms student, "
              f"{t_teacher / len(idx) * 1000:.0f} ms teacher on {len(idx)} escalated rows, "
              f"{t_all / take * 1000:.0f} ms teacher on all {take}")
        print(f"  the teacher's per-row cost falls {(t_teacher / len(idx)) / (t_all / take):.1f}x "
              f"going from {len(idx)} rows to {take} -- its context pass is fixed per call, so a "
              f"small escalation batch amortises it over fewer rows")
    elif len(idx):
        print(f"  the escalated {esc.mean():.0%} of rows took {t_teacher / total:.0%} of the time")
        print("  (--compare runs the teacher on every row too, which is the only honest way to "
              "state a cost ratio)")
    print("\n  Timings mean nothing on a contended box. RESULTS.md measures ~1.8x inflation here "
          "from background load alone, and this script cannot tell whether it is running clean.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("deploy", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True)
        s.add_argument("--artifacts", type=Path)
        s.add_argument("--seed", type=int, default=0)
        if name == "deploy":
            s.add_argument("--target", type=float, default=0.20)
            s.add_argument("--n-kernels", type=int, default=10_000)
            s.add_argument("--folds", type=int, default=5)
        else:
            s.add_argument("--batch", type=int, default=128)
            s.add_argument("--n-groups", type=int, default=40)
            s.add_argument("--compare", action="store_true",
                           help="also run the teacher on every row, on this box at this moment, so the cost ratio is measured rather than assembled from runs on different hardware")
            s.add_argument("--shell", type=Path, default=built_shell())
    args = ap.parse_args()
    art = args.artifacts or (ROOT / "data" / "serve" / args.dataset)

    if args.cmd == "deploy":
        deploy(args.dataset, args.target, args.n_kernels, args.seed, args.folds, art)
        return 0
    return serve(args.dataset, art, args.batch, args.n_groups, args.seed, args.shell,
                 ROOT / "data" / "serve" / args.dataset / "work", args.compare)


if __name__ == "__main__":
    sys.exit(main())
