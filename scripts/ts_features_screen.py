"""Screen `anofox_forecast`'s 116 statistical features against ROCKET, across many datasets.

RESULTS.md's uncomfortable finding is that the *feature extractor* is the live lever, not the
classifier -- MultiRocketHydra beat the whole in-context pipeline on Herring (0.7344 vs 0.6406) with
a linear head on better features. `anofox_forecast` ships a second family in-database:
`ts_features_by` returns one row per series with 116 numeric columns, exactly the shape our
classifier path consumes.

**This uses that extension as a BLACK BOX, on purpose.** anofox_forecast is BSL 1.1: its Additional
Use Grant permits production use but forbids offering it "to third parties on a hosted or embedded
basis", so `rocket` must never depend on it. The point is to find which statistics carry signal and
reimplement only those -- from the tsfresh catalogue (MIT) or the underlying mathematics, which is
where these statistics come from -- and NOT from reading DataZooDE's Rust. Screening with someone's
binary and deriving from their source are not the same act.

Three accuracies per dataset, same head (`RidgeClassifierCV`) so features are the only variable:

    ts        the 116 statistical features
    rocket    10,000 ROCKET features
    both      concatenated

**Why this runs on 112 datasets and not six.** The first version ran six and produced an eleven-name
"shortlist" of features appearing in more than one dataset's top-12. That was noise: six datasets
drawing 12 of 116 names produce ~14 such coincidences by chance, and the screen produced 11 -- below
the null mean, P(null >= observed) = 0.94. Six datasets cannot identify a feature. At 112 the same
null has a mean of ~11.6 appearances per feature, so one that genuinely matters separates from it;
the test acquires its power purely from the number of datasets.

Two design consequences of that lesson:

* the **full** 116 coefficient magnitudes are recorded per dataset, not a top-K slice, so any
  selection rule can be re-evaluated later without recomputing anything;
* a **bootstrap stability** score accompanies them -- how often each feature lands in the top-K
  across resamples of the training set -- because a ranking from a single fit is one draw, and
  reading one noisy ranking as a result is exactly the mistake being corrected.

Selection itself is deliberately NOT done here. `scripts/ts_features_analyze.py` does it, against an
explicit null with multiple-testing correction.

    uv run python scripts/ts_features_screen.py --datasets Herring MiddlePhalanxTW
    uv run python scripts/ts_features_screen.py --all --jobs 16 --out-dir data/ts_screen
    uv run python scripts/ts_features_screen.py --all --no-rocket --jobs 16   # ts arm only, fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from duckdb_rocket.datasets import load  # noqa: E402
from duckdb_rocket.rocket import generate_kernels, normalize_series, transform  # noqa: E402

SHELL = ROOT / "build" / "release" / ("duckdb.exe" if sys.platform == "win32" else "duckdb")

# The six hard datasets the first screen ran, kept as the default so a quick run stays cheap.
HARD = ("Herring", "MiddlePhalanxTW", "RefrigerationDevices", "Haptics", "ScreenType", "InlineSkate")

ALPHAS = np.logspace(-3, 3, 10)


def all_datasets() -> list[str]:
    """The 112 equal-length univariate UCR datasets.

    Equal-length because `rocket_transform` draws dilation and padding against a reference length,
    and unequal-length data needs a decision about that which this screen has no business making.
    Univariate because `ts_features_by` takes one (group, time, value) triple.
    """
    from aeon.datasets.tsc_datasets import univariate_equal_length

    return sorted(univariate_equal_length)


def ts_features(x: np.ndarray, shell: Path) -> tuple[np.ndarray, list[str], int]:
    """The 116 features, computed by anofox_forecast, for one array of series.

    `ts_features_by` takes long format -- (group, time, value) -- while our arrays are one series
    per row, so the series are unnested WITH ORDINALITY on the way in.
    """
    if x.ndim != 2:
        raise ValueError(f"ts_features is univariate only, got shape {x.shape}")

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.parquet"
        out = Path(td) / "feat.parquet"
        pq.write_table(
            pa.table({
                "id": pa.array(np.arange(len(x)), type=pa.int64()),
                "values": pa.array([list(map(float, s)) for s in x], type=pa.list_(pa.float64())),
            }),
            raw,
        )
        sql = f"""
INSTALL anofox_forecast FROM community;
LOAD anofox_forecast;
CREATE TABLE long AS
  SELECT id, u.i AS ts, u.v AS v
  FROM read_parquet('{raw.as_posix()}'), unnest(values) WITH ORDINALITY AS u(v, i);
CREATE TABLE feat AS SELECT * FROM ts_features_by('long', id, ts, v);
COPY (SELECT * FROM feat ORDER BY id) TO '{out.as_posix()}' (FORMAT parquet);
"""
        r = subprocess.run([str(shell), "-noheader", "-list", "-c", sql],
                           capture_output=True, text=True)
        if not out.exists():
            raise RuntimeError(f"ts_features_by produced nothing.\n{r.stdout[-800:]}\n{r.stderr[-800:]}")

        tbl = pq.read_table(out)
        names = [n for n in tbl.column_names if n != "id"]
        ids = tbl.column("id").to_numpy()
        # ORDER BY in the COPY is not a guarantee once the file is read back, and a silently permuted
        # feature matrix would train against the wrong labels and still look plausible.
        if not np.array_equal(ids, np.arange(len(x))):
            raise RuntimeError("feature rows are not the input series in order")
        f = np.column_stack([tbl.column(n).to_numpy().astype(np.float64) for n in names])

    # Unbounded statistics on real data: a near-constant series makes a variance-normalised feature
    # non-finite. Left as NaN they poison the ridge silently, so they are zeroed and counted.
    bad = int((~np.isfinite(f)).sum())
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), names, bad


def fit_score(ftr, ytr, fte, yte) -> float:
    sc = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(ftr), ytr)
    return float((clf.predict(sc.transform(fte)) == yte).mean())


def coef_magnitude(f, y) -> np.ndarray:
    """Mean |coefficient| per feature on standardised inputs, from one fit.

    Crude, and treated as crude: a ranking signal to be tested against a null downstream, not a
    measure of importance. Recorded in full so a better rule can be applied without recomputing.
    """
    sc = StandardScaler().fit(f)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(f), y)
    return np.abs(np.atleast_2d(clf.coef_)).mean(axis=0)


def bootstrap_stability(f, y, top_k: int, draws: int, seed: int) -> np.ndarray:
    """How often each feature lands in the top-K across bootstrap resamples of the training set.

    Cheap here because the ts matrix is 116 columns wide -- the expensive arm is ROCKET, which this
    does not touch.

    A resample that loses a class entirely is redrawn: RidgeClassifierCV cannot fit a single class,
    and silently dropping such draws would bias the counts toward features that serve the majority
    classes. The attempt cap keeps a pathological label distribution from looping forever.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_classes = len(np.unique(y))
    hits = np.zeros(f.shape[1], dtype=np.int64)
    done = 0
    for _ in range(draws * 10):
        if done >= draws:
            break
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < n_classes:
            continue
        hits[np.argsort(coef_magnitude(f[idx], y[idx]))[::-1][:top_k]] += 1
        done += 1
    return hits / done if done else np.zeros(f.shape[1])


def run_one(name: str, kernels: int, top_k: int, draws: int, shell: str,
            do_rocket: bool, seed: int) -> dict:
    """One dataset, start to finish."""
    t0 = time.perf_counter()
    xtr, ytr = load(name, "train")
    xte, yte = load(name, "test")
    if xtr.ndim != 2:
        return {"dataset": name, "skipped": "multivariate; ts_features_by takes one channel"}
    xtr, xte = normalize_series(xtr), normalize_series(xte)

    ttr, names, bad_tr = ts_features(xtr, Path(shell))
    tte, _, bad_te = ts_features(xte, Path(shell))

    rec: dict = {
        "dataset": name,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "n_timepoints": int(xtr.shape[-1]), "n_classes": int(len(np.unique(ytr))),
        "n_ts_features": len(names), "ts_feature_names": names,
        "nonfinite": bad_tr + bad_te,
        "acc": {"ts": fit_score(ttr, ytr, tte, yte)},
        # Full ranking and stability, so selection stays a post-hoc decision.
        "coef_magnitude": [float(v) for v in coef_magnitude(ttr, ytr)],
        "bootstrap_topk_freq": [float(v) for v in bootstrap_stability(ttr, ytr, top_k, draws, seed)],
        "bootstrap_top_k": top_k, "bootstrap_draws": draws,
    }

    if do_rocket:
        # One bank for both splits, as the pipeline does: kernels depend on series length, so a
        # per-split bank would make the columns mean different things across train and test.
        bank = generate_kernels(seed, xtr.shape[-1], kernels, n_channels=1)
        rtr, rte = transform(xtr, bank), transform(xte, bank)
        rec["acc"]["rocket"] = fit_score(rtr, ytr, rte, yte)
        rec["acc"]["both"] = fit_score(np.hstack([rtr, ttr]), ytr, np.hstack([rte, tte]), yte)
        rec["n_rocket_features"] = int(rtr.shape[1])

    rec["secs"] = round(time.perf_counter() - t0, 1)
    return rec


def _worker(args_tuple) -> tuple[str, dict | None, str | None]:
    name = args_tuple[0]
    try:
        return name, run_one(*args_tuple), None
    except Exception:  # noqa: BLE001
        return name, None, traceback.format_exc(limit=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true",
                    help="the 112 equal-length univariate UCR datasets")
    ap.add_argument("--kernels", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=12, help="K for the bootstrap top-K stability count")
    ap.add_argument("--draws", type=int, default=25, help="bootstrap resamples per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-rocket", action="store_true",
                    help="skip the 10,000-feature arm; the ts arm alone is far cheaper")
    ap.add_argument("--shell", type=Path, default=SHELL)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ts_screen",
                    help="one JSON per dataset, so a killed run resumes instead of restarting")
    ap.add_argument("--redo", action="store_true", help="recompute datasets already on disk")
    args = ap.parse_args()

    if not args.shell.exists():
        print(f"no duckdb shell at {args.shell}")
        return 1

    names = all_datasets() if args.all else list(args.datasets or HARD)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resume rather than restart. A wide run is long enough that a dropped connection or a killed pod
    # would otherwise throw away hours, which is how this project has lost time before.
    todo = [n for n in names if args.redo or not (args.out_dir / f"{n}.json").exists()]
    print(f"{len(names)} datasets: {len(names) - len(todo)} already on disk, {len(todo)} to run, "
          f"{args.jobs} job(s), rocket arm {'off' if args.no_rocket else 'on'}", flush=True)
    if not todo:
        print("nothing to do; pass --redo to recompute")
        return 0

    payload = [(n, args.kernels, args.top, args.draws, str(args.shell),
                not args.no_rocket, args.seed) for n in todo]

    done = failed = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = {ex.submit(_worker, p): p[0] for p in payload}
        for fut in as_completed(futures):
            name, rec, err = fut.result()
            if err:
                failed += 1
                print(f"  FAILED {name}: {err.splitlines()[-1][:90]}", flush=True)
                (args.out_dir / f"{name}.error.txt").write_text(err, encoding="utf-8")
                continue
            (args.out_dir / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
            done += 1
            if "skipped" in rec:
                print(f"  skip   {name}: {rec['skipped']}", flush=True)
            else:
                a = rec["acc"]
                extra = f" rocket {a['rocket']:.4f} both {a['both']:.4f}" if "rocket" in a else ""
                print(f"  [{done + failed:3d}/{len(todo)}] {name:26s} ts {a['ts']:.4f}{extra}"
                      f"  {rec['secs']:6.1f}s", flush=True)

    print(f"\n{done} done, {failed} failed, {(time.perf_counter() - t0) / 60:.1f} min wall clock")
    print(f"records in {args.out_dir}; analyse with scripts/ts_features_analyze.py")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
