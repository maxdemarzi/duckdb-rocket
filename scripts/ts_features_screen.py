"""Screen `anofox_forecast`'s 116 statistical features: do they carry signal ROCKET misses?

RESULTS.md's uncomfortable finding is that the *feature extractor* is the live lever, not the
classifier -- MultiRocketHydra beat the whole in-context pipeline on Herring (0.7344 vs 0.6406) with
a linear head on better features. So a second, orthogonal feature family is worth measuring, and
`anofox_forecast` ships one in-database: `ts_features_by` returns one row per series with 116
numeric columns, which is exactly the shape our classifier path already consumes.

**This script uses that extension as a BLACK BOX, on purpose.** anofox_forecast is BSL 1.1: its
Additional Use Grant permits production use but forbids offering it "to third parties on a hosted or
embedded basis", so `rocket` must never depend on it. The plan is to find which of the 116 features
carry signal here and reimplement only those -- from the tsfresh catalogue (MIT) or the underlying
mathematics, which is where these statistics come from in the first place, and NOT from reading
DataZooDE's Rust. Screening with someone's binary is fine; deriving from their source is not the
same act.

Three numbers per dataset, same head (RidgeClassifierCV) so the features are the only variable:

    ts        116 statistical features
    rocket    10,000 ROCKET features -- the baseline already in reference/RESULTS.md
    both      concatenated, which is the question worth asking: do 116 interpretable features
              add anything to 10,000 random convolutions?

    uv run python scripts/ts_features_screen.py --datasets Herring MiddlePhalanxTW
    uv run python scripts/ts_features_screen.py --top 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
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

# The four hard datasets with a published-vs-ROCKET gap, plus the two verified late.
HARD = ("Herring", "MiddlePhalanxTW", "RefrigerationDevices", "Haptics", "ScreenType", "InlineSkate")


def ts_features(x: np.ndarray, shell: Path) -> tuple[np.ndarray, list[str], int]:
    """The 116 features, computed by anofox_forecast, for one array of series.

    Univariate only. `ts_features_by` takes long format -- (group, time, value) -- while our arrays
    are one series per row, so the series are unnested WITH ORDINALITY on the way in. Multivariate
    input would need one call per channel and a decision about how to combine them; that is a
    different experiment and this raises rather than guessing.
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
        # ORDER BY id in the COPY is not a guarantee once the file is read back, and a silently
        # permuted feature matrix would train against the wrong labels and still look plausible.
        if not np.array_equal(ids, np.arange(len(x))):
            raise RuntimeError("feature rows are not the input series in order")
        f = np.column_stack([tbl.column(n).to_numpy().astype(np.float64) for n in names])

    # These are unbounded statistics on real data: a constant series makes a variance-normalised
    # feature non-finite. Left as NaN they poison the ridge silently, so they are zeroed and counted.
    bad = int((~np.isfinite(f)).sum())
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f, names, bad


def fit_score(ftr, ytr, fte, yte) -> float:
    sc = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(sc.transform(ftr), ytr)
    return float((clf.predict(sc.transform(fte)) == yte).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(HARD))
    ap.add_argument("--kernels", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=12, help="how many features to name per dataset")
    ap.add_argument("--shell", type=Path, default=SHELL)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if not args.shell.exists():
        print(f"no duckdb shell at {args.shell}")
        return 1

    print(f"{'dataset':22s} {'ts':>7s} {'rocket':>7s} {'both':>7s} "
          f"{'ts-rock':>8s} {'both-rock':>10s} {'nonfinite':>10s} {'secs':>6s}")
    rows = []
    for name in args.datasets:
        t0 = time.perf_counter()
        try:
            xtr, ytr = load(name, "train")
            xte, yte = load(name, "test")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} load failed: {str(e)[:44]}")
            continue
        xtr, xte = normalize_series(xtr), normalize_series(xte)
        if xtr.ndim != 2:
            print(f"{name:22s} skipped: multivariate, ts_features_by takes one channel")
            continue

        try:
            ttr, names, bad_tr = ts_features(xtr, args.shell)
            tte, _, bad_te = ts_features(xte, args.shell)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} ts_features failed: {str(e)[:44]}")
            continue

        # One bank for both splits, as the pipeline does: kernels depend on series length, so a
        # per-split bank would make the columns mean different things across train and test.
        bank = generate_kernels(0, xtr.shape[-1], args.kernels, n_channels=1)
        rtr, rte = transform(xtr, bank), transform(xte, bank)

        a_ts = fit_score(ttr, ytr, tte, yte)
        a_rk = fit_score(rtr, ytr, rte, yte)
        a_bo = fit_score(np.hstack([rtr, ttr]), ytr, np.hstack([rte, tte]), yte)
        secs = time.perf_counter() - t0
        print(f"{name:22s} {a_ts:7.4f} {a_rk:7.4f} {a_bo:7.4f} "
              f"{a_ts - a_rk:+8.4f} {a_bo - a_rk:+10.4f} {bad_tr + bad_te:10d} {secs:6.1f}")

        # Which features to reimplement, if any. Coefficient magnitude on standardised features is
        # a crude ranking and is reported as one: it is a screen to shorten a list of 116, not a
        # claim about which statistics matter in general.
        sc = StandardScaler().fit(ttr)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(sc.transform(ttr), ytr)
        mag = np.abs(np.atleast_2d(clf.coef_)).mean(axis=0)
        order = np.argsort(mag)[::-1][:args.top]
        print(f"    top {args.top}: " + ", ".join(names[i] for i in order))
        rows.append({"dataset": name, "ts": a_ts, "rocket": a_rk, "both": a_bo,
                     "n_ts_features": len(names), "nonfinite": bad_tr + bad_te,
                     "top_features": [names[i] for i in order]})

    if rows:
        print(f"\nts vs rocket:  {sum(r['ts'] > r['rocket'] for r in rows)} wins from {len(rows)}, "
              f"mean {np.mean([r['ts'] - r['rocket'] for r in rows]):+.4f}")
        print(f"both vs rocket: {sum(r['both'] > r['rocket'] for r in rows)} wins from {len(rows)}, "
              f"mean {np.mean([r['both'] - r['rocket'] for r in rows]):+.4f}")
        # A feature that ranks highly on one dataset is a coincidence; one that ranks highly on
        # several is a candidate worth writing in C++.
        tally: dict[str, int] = {}
        for r in rows:
            for f in r["top_features"]:
                tally[f] = tally.get(f, 0) + 1
        repeated = sorted((v, k) for k, v in tally.items() if v > 1)[::-1]
        # NOT a shortlist. Measured against a uniform-random null this rule yields about 14
        # features at >=2 of 6 by chance, and the real screen yielded 11 -- below the null mean,
        # P(null >= observed) = 0.94. So the count is printed with the number chance predicts
        # beside it: an earlier version called this "the reimplement shortlist" and it was
        # believed. Identifying individual features needs many more datasets, stability
        # selection over bootstraps, or family-level ablation. See reference/RESULTS.md.
        n_feat, d = rows[0]["n_ts_features"], len(rows)
        pk = args.top / n_feat
        # max(0,...) because with one dataset the expression is degenerate and prints -0.0,
        # which reads like a computed value rather than 'not applicable'.
        exp2 = max(0.0, n_feat * (1 - (1 - pk) ** d - d * pk * (1 - pk) ** (d - 1)))
        print(f"\nin the top-{args.top} of more than one dataset: {len(repeated)} features "
              f"(uniform-random null gives {exp2:.1f} -- so this is NOT a shortlist)")
        for v, name_ in repeated:
            print(f"  {v}/{len(rows)}  {name_}")

    if args.out and rows:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
