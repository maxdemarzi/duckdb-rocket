"""The arm-B plumbing, whose failures are all silent.

Three things here produce a plausible number while being wrong, and none of them raises:

* **The id offset.** The pipeline writes ids as `arange(n_train + n_test)`, so test row k is id
  `n_train + k`. Reading the sidecar from id 0 instead yields a full-length label vector of the
  teacher's *training* predictions -- an accuracy, just not the teacher's.
* **The cache.** Arms are cached per `(dataset, learner, seed, repeat)` and merged per arm, so that
  adding an arm later costs only the new fits. If the split were not a pure function of that key,
  a cached A would be compared against a B measured on different rows.
* **Confidence.** Arm Bc keeps the most confident half of the pool. Sorting the wrong way keeps the
  teacher's *least* confident labels, which still trains, still scores, and still looks like a
  result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import distill_gate as dg  # noqa: E402


def _sidecar(n_train: int, n_test: int, probs: list[dict]) -> dict:
    return {"dataset": "Fake", "model": "tabicl-v2", "n_train": n_train, "n_test": n_test,
            "classes": ["1", "2"], "mean_proba": {str(n_train + k): p for k, p in enumerate(probs)}}


class TestTeacherLabels:
    def test_reads_from_the_train_offset_not_from_zero(self):
        # Ids 0..2 are train rows and must never be read. If they were, every label would be "2".
        soft = _sidecar(3, 2, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}])
        soft["mean_proba"]["0"] = {"1": 0.0, "2": 1.0}
        soft["mean_proba"]["1"] = {"1": 0.0, "2": 1.0}
        soft["mean_proba"]["2"] = {"1": 0.0, "2": 1.0}
        lab, conf = dg.teacher_label_conf(soft, 2)
        assert list(lab) == ["1", "2"]
        assert conf == pytest.approx([0.9, 0.8])

    def test_row_count_disagreement_raises(self):
        soft = _sidecar(3, 2, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}])
        with pytest.raises(ValueError, match="2 test rows"):
            dg.teacher_label_conf(soft, 5)

    def test_a_missing_row_raises_rather_than_shifting_the_rest(self):
        # Dropping a row and carrying on would misalign every later label by one, which is exactly
        # the class of bug the full-vector id recovery was built to prevent.
        soft = _sidecar(3, 3, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}, {"1": 0.6, "2": 0.4}])
        del soft["mean_proba"]["4"]
        with pytest.raises(ValueError, match="test row 1"):
            dg.teacher_label_conf(soft, 3)

    def test_confidence_is_renormalised(self):
        soft = _sidecar(0, 1, [{"1": 0.6, "2": 0.2}])
        assert dg.teacher_label_conf(soft, 1)[1] == pytest.approx([0.75])

    def test_labels_wrapper_matches(self):
        soft = _sidecar(1, 2, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}])
        assert list(dg.teacher_labels(soft, 2)) == ["1", "2"]


class TestPoolHoldout:
    def test_deterministic_in_seed_and_repeat(self):
        y = np.array(["1"] * 20 + ["2"] * 20)
        a = dg.pool_holdout(y, 0, 1)
        b = dg.pool_holdout(y, 0, 1)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert not np.array_equal(dg.pool_holdout(y, 0, 2)[0], a[0])

    def test_partitions_the_test_split(self):
        y = np.array(["1"] * 20 + ["2"] * 20)
        pool, hold = dg.pool_holdout(y, 0, 0)
        assert sorted(np.concatenate([pool, hold]).tolist()) == list(range(40))

    def test_falls_back_when_a_class_has_one_member(self):
        # Stratifying is impossible here; refusing the dataset would silently drop it from the arm.
        y = np.array(["1"] * 19 + ["2"] * 19 + ["3"] * 1 + ["4"] * 1)
        pool, hold = dg.pool_holdout(y, 0, 0)
        assert len(pool) + len(hold) == 40


class _Recorder:
    """A learner that costs nothing and counts how often it was asked to fit."""

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, xtr, ytr, xte, seed=0):
        self.calls.append(len(ytr))
        return np.array([ytr[0]] * len(xte))


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    xtr, xte = rng.normal(size=(12, 30)), rng.normal(size=(20, 30))
    ytr = np.array([str(1 + i % 2) for i in range(12)])
    yte = np.array([str(1 + i % 2) for i in range(20)])
    monkeypatch.setattr(dg, "load", lambda name, split: (xtr, ytr) if split == "train" else (xte, yte))
    rec = _Recorder()
    monkeypatch.setattr(dg, "LEARNERS", {"fake": rec})
    # Confidence descends with the row index, so "the most confident half" is a known set.
    probs = [{"1": 1.0 - k / 100.0, "2": k / 100.0} for k in range(20)]
    (tmp_path / "phase5_Fake_soft.json").write_text(json.dumps(_sidecar(12, 20, probs)))
    return tmp_path, rec


class TestArmSplitCache:
    def test_a_second_run_adds_only_the_new_arm(self, fake):
        tmp, rec = fake
        first = dg.arm_split("Fake", "fake", 0, 0, ("A",), str(tmp), str(tmp / "cache"))
        assert set(first) == {"A"} and len(rec.calls) == 1

        rec.calls.clear()
        second = dg.arm_split("Fake", "fake", 0, 0, ("A", "B", "C"), str(tmp), str(tmp / "cache"))
        assert set(second) == {"A", "B", "C"}
        assert second["A"] == first["A"], "the cached A must be reused, not refitted"
        assert len(rec.calls) == 2, "only B and C should have been fitted"

        rec.calls.clear()
        assert dg.arm_split("Fake", "fake", 0, 0, ("A", "B"), str(tmp), str(tmp / "cache")) == {
            "A": second["A"], "B": second["B"], "C": second["C"]}
        assert rec.calls == [], "a fully cached split must fit nothing"

    def test_arms_train_on_the_sizes_they_claim(self, fake):
        tmp, rec = fake
        dg.arm_split("Fake", "fake", 0, 0, ("A", "B", "C", "Bc"), str(tmp), None)
        # 12 train rows; a 20-row test split halves to a 10-row pool; Bc keeps 5 of those.
        assert rec.calls == [12, 22, 22, 17]

    def test_bc_keeps_the_teachers_most_confident_pool_rows(self, fake):
        tmp, _ = fake
        soft = dg.load_soft(tmp, "Fake")
        _, conf = dg.teacher_label_conf(soft, 20)
        pool, _ = dg.pool_holdout(np.array([str(1 + i % 2) for i in range(20)]), 0, 0)
        keep = pool[np.argsort(-conf[pool], kind="stable")[: len(pool) // 2]]
        assert conf[keep].min() >= conf[np.setdiff1d(pool, keep)].max()

    def test_a_dataset_without_a_sidecar_raises(self, fake):
        tmp, _ = fake
        with pytest.raises(ValueError, match="sidecar"):
            dg.arm_split("Missing", "fake", 0, 0, ("A",), str(tmp), None)


class TestGateSelection:
    def test_selects_on_the_best_student_not_the_teacher(self, tmp_path):
        p = tmp_path / "gate.json"
        p.write_text(json.dumps({"rows": [
            {"dataset": "Easy", "teacher": 0.50, "students": {"a": 0.99, "b": 0.60}},
            {"dataset": "Hard", "teacher": 0.99, "students": {"a": 0.60, "b": 0.70}},
        ]}))
        assert dg.gate_selection(p, 0.90) == ["Hard"]
        assert dg.gate_selection(p, 1.00) == ["Easy", "Hard"]


class TestSignTest:
    def test_all_wins_is_the_two_sided_coin_tail(self):
        assert dg.sign_test(np.ones(6)) == pytest.approx(2 / 2**6)

    def test_ties_are_dropped_not_counted(self):
        # A tied dataset is no evidence either way. Counting it as a loss would make the median-0
        # saturated subgroup look like a defeat.
        assert dg.sign_test(np.array([1.0, 1.0, 0.0, 0.0])) == dg.sign_test(np.array([1.0, 1.0]))

    def test_no_information_is_p_one(self):
        assert dg.sign_test(np.zeros(5)) == 1.0
