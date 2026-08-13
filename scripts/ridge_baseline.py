"""Is ROCKET + ridge actually better? The README asserts it; measure it.

Same features the pipeline uses (the Python oracle, which is bit-comparable to the
extension), same train/test splits, 10,000 kernels. The only thing that changes is
what consumes the features: RidgeClassifierCV instead of an in-context model.
"""
import sys, time
sys.path.insert(0, r"C:\Users\maxde\duckdb-rocket")
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from duckdb_rocket.datasets import UCR_SUBSET, load
from duckdb_rocket.rocket import generate_kernels, transform, normalize_series

PIPELINE = {  # measured, reference/RESULTS.md
    "BasicMotions": 1.0000, "Coffee": 1.0000, "Trace": 1.0000, "GunPoint": 0.9933,
    "SyntheticControl": 0.9867, "FaceFour": 0.9773, "ItalyPowerDemand": 0.9718,
    "OSULeaf": 0.9711, "ECG5000": 0.9480, "Beef": 0.7667,
}
print(f"{'dataset':18s} {'ridge':>8s} {'pipeline':>9s} {'delta':>8s} {'ridge s':>8s}")
rows = []
for spec in UCR_SUBSET:
    name = spec.name
    try:
        xtr, ytr = load(name, "train"); xte, yte = load(name, "test")
    except Exception as e:
        print(f"{name:18s} load failed: {str(e)[:40]}"); continue
    xtr, xte = normalize_series(xtr), normalize_series(xte)
    t0 = time.perf_counter()
    # One bank for both splits, exactly as the pipeline does: kernels depend on series length,
    # so a per-split bank would make the columns mean different things across train and test.
    nch = xtr.shape[1] if xtr.ndim == 3 else 1
    bank = generate_kernels(0, xtr.shape[-1], 10_000, n_channels=nch)
    ftr = transform(xtr, bank)
    fte = transform(xte, bank)
    sc = StandardScaler().fit(ftr)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(sc.transform(ftr), ytr)
    acc = float((clf.predict(sc.transform(fte)) == yte).mean())
    secs = time.perf_counter() - t0
    p = PIPELINE.get(name)
    d = f"{acc - p:+.4f}" if p else "—"
    print(f"{name:18s} {acc:8.4f} {p if p else float('nan'):9.4f} {d:>8s} {secs:8.1f}")
    rows.append((name, acc, p, secs))
ok = [(a, p) for _, a, p, _ in rows if p]
print(f"\nmean ridge {np.mean([a for a,_ in ok]):.4f}   mean pipeline {np.mean([p for _,p in ok]):.4f}")
print(f"ridge wins {sum(1 for a,p in ok if a>p)}, pipeline wins {sum(1 for a,p in ok if p>a)}, tie {sum(1 for a,p in ok if a==p)}")
print(f"total ridge feature+fit time: {sum(s for _,_,_,s in rows):.1f}s for {len(rows)} datasets")
