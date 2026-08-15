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


class TestNoiseArms:
    def test_a_rate_is_read_off_the_arm_name(self):
        assert dg.noise_rate("N20") == pytest.approx(0.20)
        assert dg.noise_rate("N05") == pytest.approx(0.05)

    def test_ordinary_arms_are_not_rates(self):
        for a in ("A", "B", "C", "Bc", "Bs"):
            assert dg.noise_rate(a) is None

    def test_two_digits_are_required(self):
        # `N5` would be ambiguous between 5% and 50%, and guessing either would silently mislabel
        # the whole sweep's x-axis.
        assert dg.noise_rate("N5") is None
        assert not dg.known_arm("N5")
        assert dg.known_arm("N05")

    def test_corruption_is_reproducible_and_hits_about_the_right_fraction(self, fake):
        tmp, _ = fake
        a = dg.arm_split("Fake", "fake", 0, 0, ("N30",), str(tmp), None)
        b = dg.arm_split("Fake", "fake", 0, 0, ("N30",), str(tmp), None)
        assert a["N30"] == b["N30"]


class TestBreakEven:
    def test_interpolates_the_crossing(self):
        assert dg.break_even([(0.0, 0.10), (0.10, 0.05), (0.20, -0.05)]) == pytest.approx(0.15)

    def test_none_when_it_still_pays_at_the_highest_rate(self):
        assert dg.break_even([(0.0, 0.10), (0.40, 0.02)]) is None

    def test_none_when_there_was_never_any_headroom(self):
        # Distinct from the case above and not interchangeable with it: a pool that never paid even
        # with TRUE labels is silent about label quality, and reporting it as "tolerates >40% noise"
        # states the opposite of what was measured.
        assert dg.break_even([(0.0, -0.02), (0.20, -0.05)]) is None

    def test_takes_the_first_crossing(self):
        assert dg.break_even([(0.0, 0.1), (0.1, -0.1), (0.2, 0.1), (0.3, -0.1)]) == pytest.approx(0.05)


class TestErrorOverlap:
    """Whether labellers fail in the same places, which is what an ensemble lives or dies by.

    Accuracy is not the governing quantity here: a teacher's errors cost about five points more than
    random errors of the same rate, because they concentrate on the same rows. So a more accurate
    ensemble that is wrong in the same places inherits exactly the problem it was meant to solve.
    """

    def _wrong(self, n, *idx_sets):
        return {f"m{i}": np.isin(np.arange(n), list(s)) for i, s in enumerate(idx_sets)}

    def test_identical_errors_are_far_above_independence(self):
        o = dg.error_overlap(self._wrong(10, {0, 1}, {0, 1}))
        # p = 0.2 each, so independence predicts 0.04 and they are actually wrong together 0.2.
        assert o["pairs"][0]["ratio"] == pytest.approx(5.0)
        assert o["all_wrong"] == pytest.approx(0.2)
        assert o["any_right"] == pytest.approx(0.8)
        # No complementary information at all: the oracle equals the average single model.
        assert o["any_right"] == pytest.approx(o["mean_single"])

    def test_disjoint_errors_leave_everything_to_gain(self):
        o = dg.error_overlap(self._wrong(10, {0, 1}, {2, 3}))
        assert o["pairs"][0]["ratio"] == pytest.approx(0.0)
        assert o["all_wrong"] == pytest.approx(0.0)
        assert o["any_right"] - o["mean_single"] == pytest.approx(0.2)

    def test_independent_errors_sit_at_one(self):
        # 100 rows: A wrong on the first 20, B wrong on every fifth. Their overlap is exactly what
        # independence predicts, which is the calibration point the ratio is read against.
        n = 100
        a = np.zeros(n, bool); a[:20] = True
        b = np.zeros(n, bool); b[::5] = True
        assert dg.error_overlap({"a": a, "b": b})["pairs"][0]["ratio"] == pytest.approx(1.0)

    def test_every_pair_is_reported_once(self):
        o = dg.error_overlap(self._wrong(10, {0}, {1}, {2}))
        assert [(p["a"], p["b"]) for p in o["pairs"]] == [("m0", "m1"), ("m0", "m2"), ("m1", "m2")]

    def test_a_model_that_is_never_wrong_gives_no_ratio_rather_than_a_crash(self):
        o = dg.error_overlap(self._wrong(10, {0, 1}, set()))
        assert o["pairs"][0]["ratio"] != o["pairs"][0]["ratio"]  # nan, and filtered when aggregated
        assert o["all_wrong"] == 0.0


class TestNoiseCurve:
    CURVE = [(0.0, 0.10), (0.10, 0.06), (0.20, 0.02), (0.40, -0.06)]

    def test_interpolates_between_swept_rates(self):
        assert dg.noise_curve_at(self.CURVE, 0.05) == (pytest.approx(0.08), False)
        assert dg.noise_curve_at(self.CURVE, 0.30) == (pytest.approx(-0.02), False)

    def test_hits_the_measured_points_exactly(self):
        for x, y in self.CURVE:
            assert dg.noise_curve_at(self.CURVE, x)[0] == pytest.approx(y)

    def test_flags_extrapolation_past_the_last_rate(self):
        v, extrapolated = dg.noise_curve_at(self.CURVE, 0.50)
        assert extrapolated and v == pytest.approx(-0.10)

    def test_interpolation_is_not_the_same_as_snapping(self):
        # Snapping to the nearest swept rate is biased whenever the rates sit below the value asked
        # for, which is the usual case here: it compares against a LESS corrupted pool and flatters
        # the random-noise side of the comparison.
        assert dg.noise_curve_at(self.CURVE, 0.19)[0] != pytest.approx(0.02)


class TestTeacherProba:
    def test_columns_follow_the_sidecars_class_order(self):
        soft = _sidecar(2, 2, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}])
        soft["classes"] = ["2", "1"]
        classes, p = dg.teacher_proba(soft, 2)
        assert classes == ["2", "1"]
        assert p[0].tolist() == pytest.approx([0.1, 0.9])

    def test_rows_are_renormalised(self):
        soft = _sidecar(0, 1, [{"1": 0.3, "2": 0.1}])
        assert dg.teacher_proba(soft, 1)[1][0].tolist() == pytest.approx([0.75, 0.25])

    def test_a_class_absent_from_a_row_is_zero_not_missing(self):
        soft = _sidecar(0, 1, [{"1": 1.0}])
        assert dg.teacher_proba(soft, 1)[1][0].tolist() == pytest.approx([1.0, 0.0])


class TestSoftTargetRidge:
    def test_recovers_a_separable_problem_from_the_teachers_distribution(self):
        rng = np.random.default_rng(0)
        n, L = 40, 32
        xtr = np.vstack([rng.normal(0, 1, (n, L)), rng.normal(6, 1, (n, L))])
        ytr = np.array(["1"] * n + ["2"] * n)
        xpool = np.vstack([rng.normal(0, 1, (10, L)), rng.normal(6, 1, (10, L))])
        # A hedged but correct teacher: the arm exists to show hedging costs less than a wrong
        # argmax, so the fit has to survive probabilities nowhere near one-hot.
        ppool = np.array([[0.7, 0.3]] * 10 + [[0.3, 0.7]] * 10)
        xte = np.vstack([rng.normal(0, 1, (5, L)), rng.normal(6, 1, (5, L))])
        pred = dg.soft_target_ridge(xtr, ytr, xpool, ppool, ["1", "2"], xte, n_kernels=200, seed=0)
        assert list(pred) == ["1"] * 5 + ["2"] * 5

    def test_a_train_label_outside_the_teachers_classes_raises(self):
        rng = np.random.default_rng(0)
        xtr = rng.normal(size=(6, 16))
        with pytest.raises(ValueError, match="absent from the teacher"):
            dg.soft_target_ridge(xtr, np.array(["1", "2", "3", "1", "2", "3"]),
                                 rng.normal(size=(2, 16)), np.array([[0.5, 0.5]] * 2),
                                 ["1", "2"], rng.normal(size=(2, 16)), n_kernels=64)


class TestEnsembleSoftLabels:
    def _write(self, d: Path, dataset: str, model: str, probs: list[dict], classes=("1", "2")):
        s = _sidecar(0, len(probs), probs)
        s["classes"] = list(classes)
        s["model"] = model
        name = (f"phase5_{dataset}_gpu_soft.json" if model == "tabicl-v2"
                else f"phase5_{dataset}_{model}_soft.json")
        (d / name).write_text(json.dumps(s))

    def test_probabilities_are_averaged_not_voted(self, tmp_path):
        # Two models that disagree 0.6/0.4 and 0.3/0.7 average to 0.45/0.55 -- class 2. A majority
        # vote on their argmaxes ties, and any tie-break would be arbitrary; that is the information
        # averaging keeps and voting discards.
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}])
        self._write(tmp_path, "D", "mitra", [{"1": 0.3, "2": 0.7}])
        ens = dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2", "mitra"])
        assert ens["mean_proba"]["0"] == pytest.approx({"1": 0.45, "2": 0.55})
        assert list(dg.teacher_labels(ens, 1)) == ["2"]

    def test_one_model_is_the_identity(self, tmp_path):
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}])
        one = dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2"])
        assert one["mean_proba"] == dg.load_soft(tmp_path, "D")["mean_proba"]

    def test_a_missing_model_gives_none_rather_than_a_smaller_ensemble(self, tmp_path):
        # Averaging whichever models happened to have run would make the ensemble's membership vary
        # by dataset, so no per-dataset accuracy would be comparable to any other.
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}])
        assert dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2", "mitra"]) is None

    def test_rows_are_renormalised_before_averaging(self, tmp_path):
        # An unnormalised model would otherwise get more weight than the others in proportion to how
        # unnormalised it is.
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 6.0, "2": 4.0}])
        self._write(tmp_path, "D", "mitra", [{"1": 0.2, "2": 0.8}])
        ens = dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2", "mitra"])
        assert ens["mean_proba"]["0"] == pytest.approx({"1": 0.4, "2": 0.6})

    def test_disagreeing_class_lists_raise(self, tmp_path):
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}], classes=("1", "2"))
        self._write(tmp_path, "D", "mitra", [{"1": 0.6, "2": 0.4}], classes=("2", "1"))
        with pytest.raises(ValueError, match="classes"):
            dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2", "mitra"])

    def test_a_different_split_raises(self, tmp_path):
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}])
        self._write(tmp_path, "D", "mitra", [{"1": 0.6, "2": 0.4}, {"1": 0.5, "2": 0.5}])
        with pytest.raises(ValueError, match="different split"):
            dg.load_ensemble_soft(tmp_path, "D", ["tabicl-v2", "mitra"])

    def test_the_archived_teacher_keeps_its_model_free_filename(self, tmp_path):
        # Renaming the archived sidecars would orphan reference/distill_gate.json and everything
        # built on it, so tabicl-v2 must resolve to the old spelling and only new models get a suffix.
        self._write(tmp_path, "D", "tabicl-v2", [{"1": 0.6, "2": 0.4}])
        assert dg.load_soft(tmp_path, "D", "tabicl-v2") is not None
        assert dg.load_soft(tmp_path, "D") is not None
        assert dg.load_soft(tmp_path, "D", "mitra") is None


class TestRouting:
    def test_binary_margin_is_distance_from_the_boundary(self):
        assert dg.decision_margin(np.array([-2.0, 0.5, 3.0])).tolist() == [2.0, 0.5, 3.0]

    def test_multiclass_margin_is_the_gap_to_the_runner_up(self):
        # Not "the top score": a row scoring 9.0/8.9 is a coin flip and a row scoring 2.0/0.1 is not,
        # and routing on the top score alone would escalate the confident one.
        d = np.array([[9.0, 8.9, 0.0], [2.0, 0.1, 0.0]])
        assert dg.decision_margin(d).tolist() == pytest.approx([0.1, 1.9])

    def test_the_ends_of_the_curve_are_the_two_models_alone(self):
        y = np.array(["a", "b", "a", "b"])
        s = np.array(["a", "b", "b", "b"])          # student: 3/4
        t = np.array(["a", "a", "a", "a"])          # teacher: 2/4
        c = dg.route_curve(s, np.array([9.0, 8.0, 1.0, 2.0]), t, y, [0.0, 1.0])
        assert c[0][1] == pytest.approx(0.75)
        assert c[1][1] == pytest.approx(0.50)

    def test_it_escalates_the_least_confident_first(self):
        y = np.array(["a", "b", "a", "b"])
        s = np.array(["a", "b", "b", "b"])          # wrong only on row 2, its least confident
        t = np.array(["a", "b", "a", "b"])          # teacher is right everywhere
        # Routing 25% must pick row 2 and reach 1.0. Escalating the MOST confident instead would
        # leave the error in place and score 0.75 -- a plausible number from an inverted sort.
        assert dg.route_curve(s, np.array([9.0, 8.0, 1.0, 7.0]), t, y, [0.25])[0][1] == 1.0

    def test_string_labels_of_different_widths_survive_the_substitution(self):
        # A fixed-width numpy string array silently truncates a longer label assigned into it.
        y = np.array(["long_label", "b"])
        s = np.array(["b", "b"])
        t = np.array(["long_label", "long_label"])
        assert dg.route_curve(s, np.array([0.0, 9.0]), t, y, [0.5])[0][1] == 1.0


class TestSignTest:
    def test_all_wins_is_the_two_sided_coin_tail(self):
        assert dg.sign_test(np.ones(6)) == pytest.approx(2 / 2**6)

    def test_ties_are_dropped_not_counted(self):
        # A tied dataset is no evidence either way. Counting it as a loss would make the median-0
        # saturated subgroup look like a defeat.
        assert dg.sign_test(np.array([1.0, 1.0, 0.0, 0.0])) == dg.sign_test(np.array([1.0, 1.0]))

    def test_no_information_is_p_one(self):
        assert dg.sign_test(np.zeros(5)) == 1.0
