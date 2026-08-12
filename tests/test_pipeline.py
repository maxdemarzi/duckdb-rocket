"""Tests for the RocketPFN pipeline's structure and guards.

These deliberately avoid running TabPFN. Model inference is slow, needs downloaded weights, and
is exercised by the accuracy harness instead; what is worth testing cheaply and constantly is
the arithmetic around it -- group partitioning, the feature cap, probability averaging -- since
that is where a silent error produces a plausible-looking wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest

from duckdb_rocket.pipeline import (
    TABPFN_V2_5_MAX_FEATURES,
    GroupPredictions,
    RocketPFN,
    RocketPFNConfig,
)


class TestConfig:
    def test_paper_defaults(self):
        cfg = RocketPFNConfig()
        assert cfg.num_kernels == 10_000
        assert cfg.n_groups == 10
        assert cfg.kernels_per_group == 1_000
        # The arithmetic the paper's design is built around.
        assert cfg.features_per_group == TABPFN_V2_5_MAX_FEATURES

    def test_defaults_pin_the_paper_s_model_and_ensemble_size(self):
        cfg = RocketPFNConfig()
        assert cfg.model_version == "v2.5", "tabpfn 8.2 would otherwise default to v3"
        assert cfg.n_estimators == 8, "the paper specifies e=8"

    def test_rejects_uneven_group_split(self):
        with pytest.raises(ValueError, match="not divisible"):
            RocketPFNConfig(num_kernels=10_000, n_groups=3).validate()

    def test_rejects_exceeding_the_feature_cap(self):
        with pytest.raises(ValueError, match="exceeds TabPFN v2.5"):
            RocketPFNConfig(num_kernels=10_000, n_groups=2).validate()

    def test_cap_can_be_overridden_deliberately(self):
        RocketPFNConfig(num_kernels=10_000, n_groups=2, ignore_feature_cap=True).validate()

    def test_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="must both be positive"):
            RocketPFNConfig(num_kernels=0).validate()

    def test_is_frozen_so_a_reported_config_cannot_drift(self):
        cfg = RocketPFNConfig()
        with pytest.raises(Exception):
            cfg.seed = 5  # type: ignore[misc]


class TestGroupPredictions:
    def _make(self):
        classes = np.array(["a", "b"])
        per_group = np.array(
            [
                [[0.9, 0.1], [0.4, 0.6]],
                [[0.3, 0.7], [0.2, 0.8]],
            ]
        )
        return GroupPredictions(classes=classes, per_group=per_group)

    def test_mean_proba_averages_over_groups(self):
        gp = self._make()
        assert np.allclose(gp.mean_proba, [[0.6, 0.4], [0.3, 0.7]])

    def test_labels_are_argmax_of_the_average(self):
        assert list(self._make().labels) == ["a", "b"]

    def test_averaging_can_overturn_a_single_group(self):
        """The reason the paper averages probabilities rather than voting.

        Group 0 calls row 0 'a' with high confidence; group 1 calls it 'b' with lower
        confidence. Probability averaging keeps 'a'; a majority vote over two groups could
        not distinguish the case at all.
        """
        gp = self._make()
        assert gp.group_labels(0)[0] == "a"
        assert gp.group_labels(1)[0] == "b"
        assert gp.labels[0] == "a"

    def test_probabilities_stay_normalised_under_averaging(self):
        assert np.allclose(self._make().mean_proba.sum(axis=1), 1.0)


class TestRocketPFNGuards:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            RocketPFN().predict_proba(np.zeros((2, 64)))

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="but y has"):
            RocketPFN().fit(np.zeros((3, 64)), np.array([0, 1]))

    def test_fit_rejects_non_2d(self):
        with pytest.raises(ValueError, match="expected 2-D"):
            RocketPFN().fit(np.zeros(64), np.array([0]))

    def test_warns_above_the_ten_class_limit(self):
        x = np.zeros((11, 64))
        y = np.arange(11)
        with pytest.warns(UserWarning, match="10-class limit"):
            RocketPFN().fit(x, y)

    def test_predict_rejects_a_test_length_mismatch(self):
        model = RocketPFN(RocketPFNConfig(num_kernels=4, n_groups=2)).fit(
            np.zeros((2, 64)), np.array([0, 1])
        )
        with pytest.raises(ValueError, match="does not match the fitted"):
            model.predict_proba(np.zeros((2, 32)))

    def test_group_features_are_disjoint_slices_of_one_bank(self):
        """Guards `_features`' use of `first_kernel`.

        If every group generated from index 0, all G groups would produce identical features
        and the ensemble would be G copies of one classifier -- which would still run, still
        average, and still report a plausible accuracy.
        """
        cfg = RocketPFNConfig(num_kernels=8, n_groups=2)
        model = RocketPFN(cfg)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 64))

        f0 = model._features(x, 0, 64)
        f1 = model._features(x, 1, 64)
        assert f0.shape == f1.shape == (3, 8)
        assert not np.array_equal(f0, f1), "groups produced identical features"
