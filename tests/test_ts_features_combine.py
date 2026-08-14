"""The decision-value plumbing in the stacking arm, which can fail silently.

`RidgeClassifierCV.decision_function` returns signed distance for binary problems and one column per
class for multiclass. Combining them means reimplementing sklearn's argmax rule, and getting the
binary case wrong inverts every prediction while still producing a plausible accuracy near 1-p. So
the reimplementation is checked against `predict` itself on both shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ts_features_screen as s  # noqa: E402


def _fit(n_classes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 120, 8
    y = np.array([str(i % n_classes) for i in range(n)])
    # Signal, so the classifier is not fitting noise and predict() means something.
    x = rng.normal(size=(n, p)) + np.array([[float(v)] for v in y], dtype=float)
    sc, clf = s._fit(x, y)
    return sc, clf, x, y


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_argmax_labels_reproduces_predict(n_classes):
    sc, clf, x, y = _fit(n_classes)
    xt = sc.transform(x)
    mine = s._argmax_labels(clf, s._decide(clf, xt))
    assert np.array_equal(mine, clf.predict(xt)), (
        "the decision-value rule disagrees with sklearn; on binary data this inverts every "
        "prediction and still reports a plausible accuracy"
    )


def test_decide_is_always_two_dimensional():
    for n_classes in (2, 4):
        sc, clf, x, _ = _fit(n_classes)
        d = s._decide(clf, sc.transform(x))
        assert d.ndim == 2
        assert d.shape[1] == (1 if n_classes == 2 else n_classes)


def test_block_weight_of_one_matches_naive_concatenation():
    # both_scaled with weight 1.0 IS `both`, so the scaled path must reduce to the baseline. If it
    # does not, the two arms are not comparable and the whole comparison is meaningless.
    rng = np.random.default_rng(1)
    ytr = np.array([str(i % 3) for i in range(90)])
    yte = np.array([str(i % 3) for i in range(45)])
    off_tr = np.array([[float(v)] for v in ytr])
    off_te = np.array([[float(v)] for v in yte])
    rtr, rte = rng.normal(size=(90, 40)) + off_tr, rng.normal(size=(45, 40)) + off_te
    ttr, tte = rng.normal(size=(90, 7)) + off_tr, rng.normal(size=(45, 7)) + off_te

    scaled = s.block_scaled_score(rtr, ttr, ytr, rte, tte, yte, 1.0)
    # `both` standardises the concatenation as one block; block_scaled standardises each block. With
    # weight 1.0 those differ only by per-column scaling, which a standardised ridge is invariant to.
    naive = s.fit_score(np.hstack([rtr, ttr]), ytr, np.hstack([rte, tte]), yte)
    assert scaled == pytest.approx(naive, abs=0.05)


def test_cv_folds_refuses_to_stratify_a_singleton_class():
    # A class with one member cannot be stratified. Returning None makes the caller fall back to a
    # fixed weight; silently using unstratified folds would tune one arm and not the other.
    y = np.array(["a"] * 30 + ["b"] * 30 + ["c"])
    assert s._cv_folds(y, 5, 0) is None
    assert s._cv_folds(np.array(["a"] * 30 + ["b"] * 30), 5, 0) is not None
