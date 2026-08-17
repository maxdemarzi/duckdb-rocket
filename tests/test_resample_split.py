"""`--resample` must change the split and nothing else.

Every accuracy in RESULTS.md comes from one train/test split -- the one the archive ships -- and
the effects being chased are smaller than what one split can resolve. Beef has 30 test rows, so a
single row moves accuracy by 0.0333, while G=10 costs -0.0033 and the best ensemble rule found
gains +0.0024. The paper averages 30 resamples for exactly this reason, and `resample_split` is
what makes that possible here.

Which puts a lot of weight on it. A resampler that quietly loses rows, or drifts the class balance
of the context, or is not reproducible, produces numbers that look like the campaign succeeded --
more splits, tighter-looking intervals -- while measuring the sampler. Every property below is one
that would fail that way rather than loudly:

* resample 0 must be the archive split BYTE FOR BYTE, or nothing already archived reproduces.
* The pooled set must survive exactly: no row lost, none duplicated, none invented.
* Per-class train counts must be preserved, not merely the total. An in-context model reads the
  training rows as its entire signal, so a context whose minority class thinned by luck is a
  different treatment, and two resamples disagreeing about it would be measuring the draw.
* resample k must be the same split every time it is asked for, independent of `--seed`.

The multivariate case is checked because the pooling is an axis-0 concatenate over a 3-D array
there, which is the shape most likely to be silently wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phase5_pipeline import resample_split  # noqa: E402


def make_split(n_train_per_class=(20, 12, 8), n_test_per_class=(30, 18, 6),
               timepoints=16, channels=0, seed=1234):
    """An imbalanced dataset whose train and test class ratios deliberately DIFFER.

    Equal ratios would make "preserves per-class train counts" and "preserves the total" pass or
    fail together, so the test could not tell them apart -- which is the whole distinction.
    """
    rng = np.random.default_rng(seed)
    shape = (timepoints,) if not channels else (channels, timepoints)
    xs, ys, split = [], [], []
    for cls, (n_tr, n_te) in enumerate(zip(n_train_per_class, n_test_per_class)):
        for _ in range(n_tr):
            xs.append(rng.normal(size=shape)); ys.append(f"c{cls}"); split.append("train")
        for _ in range(n_te):
            xs.append(rng.normal(size=shape)); ys.append(f"c{cls}"); split.append("test")
    x, y, split = np.array(xs), np.array(ys), np.array(split)
    return x[split == "train"], y[split == "train"], x[split == "test"], y[split == "test"]


def row_keys(x):
    """A hashable identity per series, so pooled membership can be compared as a multiset."""
    return sorted(hash(np.asarray(r).tobytes()) for r in x)


def test_resample_zero_is_the_archive_split_untouched():
    xtr, ytr, xte, yte = make_split()
    a, b, c, d = resample_split(xtr, ytr, xte, yte, 0)
    # `is` where possible: resample 0 must not even copy, so there is no chance of a transform
    # sneaking in on the default path that every archived result was produced by.
    assert a is xtr and c is xte
    assert np.array_equal(b, ytr) and np.array_equal(d, yte)


@pytest.mark.parametrize("channels", [0, 3], ids=["univariate", "multivariate"])
def test_sizes_and_pooled_membership_survive(channels):
    xtr, ytr, xte, yte = make_split(channels=channels)
    for k in (1, 2, 7, 29):
        a, b, c, d = resample_split(xtr, ytr, xte, yte, k)
        assert a.shape[0] == xtr.shape[0] and c.shape[0] == xte.shape[0]
        assert len(b) == len(ytr) and len(d) == len(yte)
        assert a.shape[1:] == xtr.shape[1:], "series shape changed"
        # Every original row appears exactly once across the new split, and nothing else does.
        assert row_keys(np.concatenate([a, c])) == row_keys(np.concatenate([xtr, xte]))
        assert sorted(np.concatenate([b, d])) == sorted(np.concatenate([ytr, yte]))


def test_per_class_train_counts_are_preserved_not_just_the_total():
    xtr, ytr, xte, yte = make_split()
    want = {c: int(np.sum(ytr == c)) for c in np.unique(ytr)}
    assert len(set(want.values())) > 1, "the fixture must be imbalanced or this proves nothing"
    for k in (1, 5, 13):
        _, b, _, _ = resample_split(xtr, ytr, xte, yte, k)
        assert {c: int(np.sum(b == c)) for c in np.unique(b)} == want


def test_labels_stay_attached_to_their_series():
    """The re-split indexes x and y with one index array; a bug here decorrelates them.

    Nothing else in this file would catch that -- sizes, counts and pooled membership all still
    pass with labels shuffled against the series, and the pipeline would simply report a chance
    accuracy on every resample, which reads as the model failing rather than the sampler.
    """
    xtr, ytr, xte, yte = make_split()
    # Stamp each series with its own label so the pairing is checkable after the fact.
    truth = {}
    for x, y in ((xtr, ytr), (xte, yte)):
        for i, cls in enumerate(y):
            x[i, 0] = float(cls[1:])
            truth[hash(x[i].tobytes())] = cls
    a, b, c, d = resample_split(xtr, ytr, xte, yte, 3)
    for x, y in ((a, b), (c, d)):
        for i, cls in enumerate(y):
            assert truth[hash(x[i].tobytes())] == cls


def test_the_same_resample_is_the_same_split_and_different_ones_differ():
    xtr, ytr, xte, yte = make_split()
    a1, b1, _, _ = resample_split(xtr, ytr, xte, yte, 4)
    a2, b2, _, _ = resample_split(xtr, ytr, xte, yte, 4)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    a3, _, _, _ = resample_split(xtr, ytr, xte, yte, 5)
    assert not np.array_equal(a1, a3), "resamples 4 and 5 drew the same split"


def test_it_actually_moves_rows_across_the_boundary():
    """Otherwise every property above is satisfiable by returning the input unchanged."""
    xtr, ytr, xte, yte = make_split()
    before = row_keys(xtr)
    a, _, _, _ = resample_split(xtr, ytr, xte, yte, 6)
    moved = len(set(before) - set(row_keys(a)))
    assert moved > 0.2 * len(before), f"only {moved} of {len(before)} train rows changed"


def test_write_raw_parquet_returns_the_resampled_y_test_not_the_archive_one():
    """The integration point, and the one that would corrupt every number in the campaign.

    `write_raw_parquet` writes the labels into the parquet AND returns `y_test` separately, and
    accuracy is scored against the returned copy. If the resample reached the file but not the
    return -- or the two came out in different orders -- every resample would score against the
    archive's labels and report something near chance, which reads as the model failing on
    resampled data rather than as the harness misaligning it.

    Checked as an equality between the two, rather than by re-deriving what either ought to be:
    re-deriving would reproduce whatever mistake the function makes.
    """
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from phase5_pipeline import write_raw_parquet

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            for k in (0, 1, 2, 0):
                meta, y_test = write_raw_parquet("Coffee", Path(tmp) / f"r{k}", True, 0, 0, k)
                table = pq.read_table(Path(tmp) / f"r{k}" / "raw.parquet")
                labels = [str(v) for v in table.column("label").to_pylist()]
                first = [v[0] for v in table.column("values").to_pylist()]
                # The returned y_test must BE the test half of what was written, in that order.
                assert list(y_test) == labels[meta["n_train"]:]
                seen.append((labels, first, meta["n_train"], meta["n_test"]))
    except (FileNotFoundError, OSError) as exc:      # the archive is not present in this checkout
        pytest.skip(f"Coffee not available: {exc}")

    r0a, r1, r2, r0b = seen
    assert r0a == r0b, "resample 0 is not reproducible"
    assert all(s[2:] == r0a[2:] for s in seen), "the split sizes moved"
    assert r1[1] != r0a[1] and r2[1] != r1[1], "resamples 1 and 2 did not change the split"
    assert sorted(r1[1]) == sorted(r0a[1]), "the pooled series changed"


def test_a_class_the_archive_puts_only_in_test_stays_out_of_the_context():
    """A real property of some UCR splits, and it must not become a shape change.

    The row-alignment assertions downstream are stated in terms of n_train and n_test, so a
    resample that quietly rebalanced such a class would break them somewhere far from here.
    """
    xtr, ytr, xte, yte = make_split(n_train_per_class=(20, 12, 0), n_test_per_class=(30, 18, 6))
    a, b, c, d = resample_split(xtr, ytr, xte, yte, 2)
    assert a.shape[0] == xtr.shape[0] and c.shape[0] == xte.shape[0]
    assert "c2" not in set(b), "a class absent from the archive's context reappeared in it"
    assert int(np.sum(d == "c2")) == 6
