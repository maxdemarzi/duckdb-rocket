"""The overlap measurement is only as good as its alignment.

Both axes are looked up by name -- rows by `n_train + k`, classes by label -- because getting
either wrong produces a plausible accuracy rather than an error. A permuted class order would
report two models disagreeing everywhere while every array shape stayed valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from error_overlap import aligned, discover, one_dataset  # noqa: E402


def _soft(n_train, rows, classes, model="m"):
    """rows: list of {class: prob} in test order."""
    return {"dataset": "GunPoint", "model": model, "n_train": n_train, "n_test": len(rows),
            "classes": list(classes),
            "mean_proba": {str(n_train + k): r for k, r in enumerate(rows)}}


def test_aligned_follows_class_names_not_their_order():
    """Two runs listing the classes in opposite orders must produce the same matrix."""
    a = _soft(5, [{"1": 0.9, "2": 0.1}, {"1": 0.2, "2": 0.8}], ["1", "2"])
    b = _soft(5, [{"2": 0.1, "1": 0.9}, {"2": 0.8, "1": 0.2}], ["2", "1"])
    shared = ["1", "2"]
    assert np.allclose(aligned(a, shared), aligned(b, shared))
    # And the shared order is what indexes the answer, so argmax must name class "1" first.
    assert np.array(shared)[aligned(b, shared).argmax(1)].tolist() == ["1", "2"]


def test_aligned_reads_rows_by_id_not_by_position():
    """Test row k is id n_train + k. A file that starts at 0 is not this dataset's test split."""
    good = _soft(5, [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], ["a", "b"])
    assert aligned(good, ["a", "b"]) is not None
    bad = dict(good, mean_proba={"0": {"a": 1.0, "b": 0.0}, "1": {"a": 0.0, "b": 1.0}})
    assert aligned(bad, ["a", "b"]) is None, "ids starting at 0 should be refused, not shifted"


def test_aligned_refuses_a_short_file():
    d = _soft(5, [{"a": 1.0}], ["a"])
    d["n_test"] = 2                       # claims two rows, carries one
    assert aligned(d, ["a"]) is None


def test_aligned_zero_fills_a_class_the_model_never_saw():
    """One model can have a class the other lacks; the shared order is the union."""
    d = _soft(0, [{"a": 0.7, "b": 0.3}], ["a", "b"])
    out = aligned(d, ["a", "b", "c"])
    assert out.shape == (1, 3) and out[0, 2] == 0.0


def _write(tmp_path, dataset, model, tag, rows, classes, n_train, config):
    (tmp_path / f"phase5_{dataset}_{tag}_soft.json").write_text(
        json.dumps(dict(_soft(n_train, rows, classes, model), dataset=dataset)), encoding="utf-8")
    (tmp_path / f"phase5_{dataset}_{tag}.json").write_text(
        json.dumps({"dataset": dataset, "model": model, "config": config}), encoding="utf-8")


CFG = {"num_kernels": 10_000, "n_groups": 40, "kernels_per_group": 250, "n_estimators": 1,
       "seed": 0}


def test_discover_keys_on_the_model_not_the_filename_tag(tmp_path):
    """`_cpu` and `_gpu` are both tabicl-v2; keying on the tag compares a model with itself."""
    rows = [{"1": 1.0, "2": 0.0}]
    _write(tmp_path, "GunPoint", "tabicl-v2", "cpu", rows, ["1", "2"], 50, CFG)
    _write(tmp_path, "GunPoint", "tabicl-v2", "gpu", rows, ["1", "2"], 50, CFG)
    found = discover(tmp_path)
    assert list(found["GunPoint"]) == ["tabicl-v2"], "one model appeared twice"
    assert one_dataset("GunPoint", found["GunPoint"]) is None, "a model was compared with itself"


def test_one_backbone_over_two_feature_families_is_two_arms(tmp_path):
    """Same model, different features, is the comparison -- not a duplicate to be collapsed.

    Keying on the model alone would merge the rocket and ts runs of one backbone and silently keep
    whichever was read first, reporting "one arm" for the experiment that exists to contrast them.
    """
    rows = [{"1": 1.0, "2": 0.0}]
    _write(tmp_path, "GunPoint", "tabicl-v2", "cpu", rows, ["1", "2"], 50, dict(CFG))
    _write(tmp_path, "GunPoint", "tabicl-v2", "ts", rows, ["1", "2"], 50,
           dict(CFG, n_groups=1, num_kernels=0, kernels_per_group=0, features="ts"))
    # The ts report has to declare its family; the writer above puts config only, so state it.
    p = tmp_path / "phase5_GunPoint_ts.json"
    p.write_text(json.dumps({"dataset": "GunPoint", "model": "tabicl-v2", "features": "ts",
                             "config": dict(CFG, n_groups=1, num_kernels=0,
                                            kernels_per_group=0)}), encoding="utf-8")
    assert sorted(discover(tmp_path)["GunPoint"]) == ["tabicl-v2", "tabicl-v2/ts"]


def test_feature_families_are_not_refused_for_differing_on_the_kernel_bank(tmp_path):
    """ts features come from no kernel bank, so n_groups and num_kernels cannot match -- and
    requiring them would refuse the one comparison this tool exists to make."""
    from duckdb_rocket.datasets import load
    _, y = load("GunPoint", "test")
    rows = [{str(v): 1.0, ("2" if str(v) == "1" else "1"): 0.0} for v in y]
    _write(tmp_path, "GunPoint", "tabicl-v2", "cpu", rows, ["1", "2"], 50, dict(CFG))
    (tmp_path / "phase5_GunPoint_ts_soft.json").write_text(
        json.dumps(dict(_soft(50, rows, ["1", "2"], "tabicl-v2"), dataset="GunPoint")),
        encoding="utf-8")
    (tmp_path / "phase5_GunPoint_ts.json").write_text(
        json.dumps({"dataset": "GunPoint", "model": "tabicl-v2", "features": "ts",
                    "config": {"num_kernels": 0, "n_groups": 1, "kernels_per_group": 0,
                               "n_estimators": 1, "seed": 0}}), encoding="utf-8")
    r = one_dataset("GunPoint", discover(tmp_path)["GunPoint"])
    assert r is not None, "the feature-family comparison was refused on the kernel bank"
    assert sorted(r["models"]) == ["tabicl-v2", "tabicl-v2/ts"]
    # A differing seed is still fatal, because the two arms would not be over the same split.
    (tmp_path / "phase5_GunPoint_ts.json").write_text(
        json.dumps({"dataset": "GunPoint", "model": "tabicl-v2", "features": "ts",
                    "config": {"num_kernels": 0, "n_groups": 1, "kernels_per_group": 0,
                               "n_estimators": 1, "seed": 7}}), encoding="utf-8")
    assert one_dataset("GunPoint", discover(tmp_path)["GunPoint"]) is None


def test_a_config_mismatch_is_refused_rather_than_averaged(tmp_path, capsys):
    """A 40-group run against a 10-group one measures the group count, not the backbone."""
    rows = [{"1": 1.0, "2": 0.0}]
    _write(tmp_path, "GunPoint", "tabicl-v2", "cpu", rows, ["1", "2"], 50, CFG)
    _write(tmp_path, "GunPoint", "tabpfn-v2", "tabpfn-v2", rows, ["1", "2"], 50,
           dict(CFG, n_groups=10))
    assert one_dataset("GunPoint", discover(tmp_path)["GunPoint"]) is None
    assert "differ on n_groups" in capsys.readouterr().out


def test_overlap_counts_a_shared_failure_as_shared(tmp_path):
    """The whole point: two models wrong on the same row, against independence.

    GunPoint's test split is 150 rows; only the first four are given distinctive probabilities and
    the rest are made correct, so the arithmetic is checkable by hand.
    """
    from duckdb_rocket.datasets import load
    _, y = load("GunPoint", "test")
    truth = [str(v) for v in y]
    other = {"1": "2", "2": "1"}
    # Row 0: both wrong (the shared failure). Row 1: only A wrong. Rest: both right.
    a_rows, b_rows = [], []
    for k, t in enumerate(truth):
        wrong_a = k in (0, 1)
        wrong_b = k == 0
        a_rows.append({other[t] if wrong_a else t: 1.0, t if wrong_a else other[t]: 0.0})
        b_rows.append({other[t] if wrong_b else t: 1.0, t if wrong_b else other[t]: 0.0})
    _write(tmp_path, "GunPoint", "tabicl-v2", "cpu", a_rows, ["1", "2"], 50, CFG)
    _write(tmp_path, "GunPoint", "tabpfn-v2", "tabpfn-v2", b_rows, ["1", "2"], 50, CFG)
    r = one_dataset("GunPoint", discover(tmp_path)["GunPoint"])
    n = len(truth)
    assert r["accuracy"]["tabicl-v2"] == pytest.approx(1 - 2 / n)
    assert r["accuracy"]["tabpfn-v2"] == pytest.approx(1 - 1 / n)
    assert r["none_right"] == pytest.approx(1 / n), "the shared failure is the floor"
    assert r["oracle"] == pytest.approx(1 - 1 / n), "row 1 is recoverable, row 0 is not"
    pair = r["pairs"][0]
    assert pair["both_wrong"] == pytest.approx(1 / n)
    assert pair["both_wrong_if_independent"] == pytest.approx((2 / n) * (1 / n))
    # One shared failure out of a possible 2x1/n is a large excess, which is the number the
    # ensemble argument turns on.
    assert pair["excess"] == pytest.approx(n / 2)
