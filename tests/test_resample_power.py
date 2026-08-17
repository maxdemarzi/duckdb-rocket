"""The variance decomposition must say what to buy: more resamples, or more datasets.

`resample_power.analyse` exists to answer one question -- how big a campaign would it take to
resolve a half-point -- and a wrong answer here is expensive in a way that is hard to notice. If it
understates the between-dataset term it will recommend 30 resamples for a comparison that no number
of resamples can settle, and the campaign comes back with a tight interval around a quantity that
still flips sign when the datasets change. RESULTS.md already records that flip happening.

So the tests below build data whose truth is known by construction and check that the split comes
out the right way round:

* noise entirely WITHIN datasets  -> between ~ 0, and resamples buy precision
* noise entirely BETWEEN datasets -> within ~ 0, an SE floor resamples cannot cross
* a known true effect             -> recovered by the grand mean
* half-finished pairs             -> dropped, not silently mixed across resamples

The last one matters more than it looks: a pod that dies mid-sweep leaves exactly that state, and
pairing arm A of resample 3 against arm B of resample 7 would put the split noise straight back
into the number the pairing exists to remove.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pod"))

from resample_power import analyse  # noqa: E402


def runs_from(effects: dict[str, list[float]], base: float = 0.75) -> list[dict]:
    """One (dataset, resample) pair per delta, as the driver would have collected them."""
    out = []
    for ds, deltas in effects.items():
        for k, d in enumerate(deltas, start=1):
            out += [
                {"dataset": ds, "resample": k, "arm": "A", "accuracy": base},
                {"dataset": ds, "resample": k, "arm": "B", "accuracy": base + d},
            ]
    return out


def test_noise_purely_within_datasets_is_attributed_to_resamples():
    rng = np.random.default_rng(0)
    # Every dataset has the SAME true effect of zero; all spread is split luck.
    effects = {f"d{i}": list(rng.normal(0.0, 0.02, 12)) for i in range(8)}
    a = analyse(runs_from(effects), target=0.005)

    assert a["var_within"] == pytest.approx(0.02 ** 2, rel=0.4)
    assert a["var_between"] < 0.1 * a["var_within"], "split luck was blamed on the datasets"
    # With no true heterogeneity there is no floor worth speaking of, so resampling works.
    assert a["se_floor_at_this_D"] < a["se_needed_for_target"]
    assert any(p["resamples_needed"] for p in a["plans"])


def test_noise_purely_between_datasets_cannot_be_resampled_away():
    # Each dataset has its own fixed effect and NO split noise at all: every resample of a given
    # dataset returns the identical delta. This is the case where 30 resamples buy nothing.
    rng = np.random.default_rng(1)
    effects = {f"d{i}": [float(v)] * 12 for i, v in enumerate(rng.normal(0.0, 0.02, 8))}
    a = analyse(runs_from(effects), target=0.005)

    assert a["var_within"] == pytest.approx(0.0, abs=1e-12)
    assert a["var_between"] == pytest.approx(0.02 ** 2, rel=0.6)
    # The floor is the whole point: SE cannot go below it however large R gets.
    floor = a["se_floor_at_this_D"]
    assert floor > 0
    plan_here = next(p for p in a["plans"] if p["datasets"] == a["datasets"])
    assert plan_here["se_at_30"] == pytest.approx(floor, rel=1e-6), \
        "SE at 30 resamples should already be sitting on the floor"


def test_a_known_effect_is_recovered():
    rng = np.random.default_rng(2)
    truth = 0.012
    effects = {f"d{i}": list(rng.normal(truth, 0.01, 20)) for i in range(10)}
    a = analyse(runs_from(effects), target=0.005)
    assert a["grand_mean_delta"] == pytest.approx(truth, abs=0.005)
    # An effect of 0.012 against this noise should be comfortably resolvable.
    assert a["se_observed"] < truth / 2.8


def test_more_datasets_is_the_only_lever_on_the_between_term():
    rng = np.random.default_rng(3)
    effects = {f"d{i}": [float(v)] * 6 for i, v in enumerate(rng.normal(0.0, 0.03, 6))}
    a = analyse(runs_from(effects), target=0.005)
    by_d = {p["datasets"]: p["se_at_30"] for p in a["plans"]}
    # Strictly decreasing in D, and flat in R (checked above). That asymmetry IS the recommendation.
    ds = sorted(by_d)
    assert all(by_d[ds[i]] > by_d[ds[i + 1]] for i in range(len(ds) - 1))


def test_half_finished_pairs_are_dropped_rather_than_mixed():
    runs = runs_from({"d0": [0.01, 0.02, 0.03]})
    # Lose arm B of resample 2 -- exactly what a pod dying mid-sweep leaves behind.
    runs = [r for r in runs if not (r["resample"] == 2 and r["arm"] == "B")]
    a = analyse(runs, target=0.005)
    assert a["per_dataset"]["d0"]["n_resamples"] == 2
    assert sorted(a["per_dataset"]["d0"]["deltas"]) == ["1", "3"]


def test_a_failed_run_carries_no_accuracy_and_is_ignored():
    runs = runs_from({"d0": [0.01, 0.02]})
    runs.append({"dataset": "d0", "resample": 9, "arm": "A", "error": "timeout"})
    runs.append({"dataset": "d0", "resample": 9, "arm": "B", "error": "timeout"})
    a = analyse(runs, target=0.005)
    assert a["per_dataset"]["d0"]["n_resamples"] == 2


def test_no_complete_pairs_says_so_instead_of_dividing_by_zero():
    a = analyse([{"dataset": "d0", "resample": 1, "arm": "A", "accuracy": 0.5}], target=0.005)
    assert "note" in a and "plans" not in a


def test_one_dataset_refuses_to_size_a_campaign_rather_than_returning_nan():
    """A single dataset cannot estimate between-dataset variance, and NaN reads as a finding.

    This is not hypothetical: the smoke run of this pilot was one dataset, and the report
    concluded "the between-dataset term dominates, and this comparison cannot be resolved by
    resampling at any affordable scale" -- because `statistics.variance` of one value is NaN, NaN
    fails every `<=` comparison, so every plan came back unreachable. The most confident sentence
    in the output was the one with no data behind it.
    """
    a = analyse(runs_from({"d0": [0.01, 0.02, 0.03]}), target=0.005)
    assert "note" in a
    assert "plans" not in a, "sized a campaign from one dataset"
    assert not any(isinstance(v, float) and math.isnan(v)
                   for v in a.values() if isinstance(v, float)), "NaN escaped into the report"
    assert a["datasets"] == 1 and a["var_within"] > 0   # what IS knowable is still reported


def test_a_negative_between_estimate_is_reported_rather_than_hidden():
    """Small D and near-zero heterogeneity legitimately give a negative estimate.

    Clamping it in silence would turn "we cannot distinguish this from zero" into the much
    stronger "there is no heterogeneity", which is the kind of claim this project keeps having to
    retract. The clamped value is used for arithmetic; the raw one is kept for the reader.
    """
    rng = np.random.default_rng(4)
    effects = {f"d{i}": list(rng.normal(0.0, 0.05, 3)) for i in range(3)}
    a = analyse(runs_from(effects), target=0.005)
    assert a["var_between"] >= 0.0
    if a["var_between_raw"] < 0:
        assert a["var_between"] == 0.0
