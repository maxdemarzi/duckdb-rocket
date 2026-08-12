"""RocketPFN — the Phase 1 oracle end to end.

Composes ROCKET feature extraction with TabPFN classification, exactly as the paper describes:
10,000 kernels split into G=10 groups of 1,000, each group producing 2,000 features (TabPFN
v2.5's column cap), each group classified independently, and the resulting class probabilities
averaged.

Every accuracy number the project reports is measured against this module, so its defaults are
chosen for fidelity to the paper rather than for speed or for whatever the libraries happen to
do on their own. Two of those library defaults are actively wrong for us and are overridden
here; both are documented at the point of override, because a silently-inherited default is
the failure mode this project has already been bitten by once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .rocket import generate_kernels, normalize_series, transform

# TabPFN v2.5's stated ceiling. The paper's G=10 x 1,000 kernels x 2 features is built to land
# exactly on it, which is why the group count is not a free parameter in disguise.
TABPFN_V2_5_MAX_FEATURES = 2_000


@dataclass(frozen=True)
class RocketPFNConfig:
    """Everything that affects a reported number, in one place.

    Deliberately a frozen dataclass rather than loose keyword arguments: this object is what
    gets serialised alongside an accuracy figure, and a config you cannot write down is a
    number you cannot attribute.
    """

    num_kernels: int = 10_000
    n_groups: int = 10
    seed: int = 0

    n_estimators: int = 8
    """TabPFN's internal ensemble size. The paper specifies e=8, which is also TabPFN's own
    default -- but **anofox_tabfm hard-throws on n_estimators > 1**, so the DuckDB pipeline
    cannot reach this setting (see PLAN.md Phase 2, finding 3). Run both e=8 and e=1 and report
    them side by side: the difference is precisely what the SQL path costs us, and measuring it
    is cheaper than arguing about it."""

    model_version: str = "v2.5"
    """The paper uses TabPFN v2.5. **tabpfn 8.2.0 defaults to v3**, so leaving this unset would
    silently benchmark a different, newer model than the one being reproduced."""

    device: str = "cpu"
    normalize: bool = True

    ignore_feature_cap: bool = False
    """Escape hatch for deliberately exceeding v2.5's 2,000-column limit. Off by default so
    that exceeding it is a decision rather than an accident."""

    extra_tabpfn_kwargs: dict = field(default_factory=dict)

    @property
    def kernels_per_group(self) -> int:
        return self.num_kernels // self.n_groups

    @property
    def features_per_group(self) -> int:
        return self.kernels_per_group * 2

    def validate(self) -> None:
        if self.num_kernels <= 0 or self.n_groups <= 0:
            raise ValueError("num_kernels and n_groups must both be positive")
        if self.num_kernels % self.n_groups:
            raise ValueError(
                f"num_kernels={self.num_kernels} is not divisible by n_groups="
                f"{self.n_groups}; groups must partition the bank evenly"
            )
        if not self.ignore_feature_cap and self.features_per_group > TABPFN_V2_5_MAX_FEATURES:
            raise ValueError(
                f"{self.features_per_group} features per group exceeds TabPFN v2.5's "
                f"{TABPFN_V2_5_MAX_FEATURES}-column cap. The paper's configuration lands "
                f"exactly on it; pass ignore_feature_cap=True to override deliberately."
            )


def build_classifier(config: RocketPFNConfig):
    """Construct a TabPFNClassifier with the two dangerous defaults pinned.

    Imported lazily so that the ROCKET half of this project stays usable -- and testable --
    without torch present.
    """
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import resolve_model_path

    # --- Override 1: precision -------------------------------------------------------------
    # inference_precision defaults to "auto", which resolves PER DEVICE: autocast is ON for any
    # CUDA device (in fp16), and on CPU whenever the hardware has native bf16 (Intel AMX /
    # AVX512-BF16, AMD Zen 4+). Passing an explicit torch.dtype takes a different branch
    # entirely and forces use_autocast_ = False regardless of device.
    #
    # This is not a micro-optimisation. The tabicl fork measured 7.3 AUC lost to bf16 autocast,
    # "harmless where the model is confident and expensive where it is not" -- backwards from
    # where precision is wanted -- and re-measured every GPU number it had. GPU autocast here
    # is fp16, whose range is narrower still. Note also that a CPU-vs-GPU agreement check does
    # NOT catch this on a Zen 4 / Sapphire Rapids pod, where both sides are reduced precision
    # and can agree while both are wrong.
    precision = torch.float32

    # --- Override 2: model version ---------------------------------------------------------
    # tabpfn 8.2.0's settings default is ModelVersion.V3. The version is resolved from the
    # model path's *filename*, so pinning means resolving a concrete v2.5 checkpoint rather
    # than passing a bare version string.
    model_paths, *_ = resolve_model_path(None, "classifier", version=config.model_version)
    model_path = model_paths[0] if isinstance(model_paths, list) else model_paths

    return TabPFNClassifier(
        model_path=model_path,
        n_estimators=config.n_estimators,
        inference_precision=precision,
        device=config.device,
        random_state=config.seed,
        **config.extra_tabpfn_kwargs,
    )


@dataclass
class GroupPredictions:
    """Per-group probabilities, kept rather than collapsed.

    `predict_proba` could return only the average, but the per-group matrix is what makes the
    interesting questions answerable -- how much the ensembling actually buys, whether one
    group is an outlier, whether G=10 is more than needed -- and it costs nothing to retain.
    """

    classes: np.ndarray
    """Shape (n_classes,). Verified identical across groups."""

    per_group: np.ndarray
    """Shape (n_groups, n_test, n_classes)."""

    @property
    def mean_proba(self) -> np.ndarray:
        """The paper's ensembling: a plain mean of class probabilities across groups."""
        return self.per_group.mean(axis=0)

    @property
    def labels(self) -> np.ndarray:
        return self.classes[np.argmax(self.mean_proba, axis=1)]

    def group_labels(self, group: int) -> np.ndarray:
        return self.classes[np.argmax(self.per_group[group], axis=1)]


class RocketPFN:
    """Training-free time-series classification: ROCKET features into TabPFN.

    There is no `fit` in the gradient sense -- TabPFN is in-context, so the labelled series are
    context rows rather than training data. `fit` therefore just retains them.
    """

    def __init__(self, config: RocketPFNConfig | None = None) -> None:
        self.config = config or RocketPFNConfig()
        self.config.validate()
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RocketPFN:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected 2-D (n_series, n_timepoints), got shape {x.shape}")
        y = np.asarray(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} series but y has {y.shape[0]} labels")
        n_classes = len(np.unique(y))
        if n_classes > 10:
            warnings.warn(
                f"{n_classes} classes exceeds TabPFN v2.5's stated 10-class limit; "
                f"results are outside the model's validated range",
                stacklevel=2,
            )
        self._x, self._y = x, y
        return self

    def _features(self, x: np.ndarray, group: int, n_timepoints: int) -> np.ndarray:
        cfg = self.config
        kernels = generate_kernels(
            cfg.seed,
            n_timepoints,
            cfg.kernels_per_group,
            first_kernel=group * cfg.kernels_per_group,
        )
        return transform(x, kernels)

    def predict_proba(self, x: np.ndarray) -> GroupPredictions:
        """Run all G groups and return their probabilities, un-averaged."""
        if self._x is None or self._y is None:
            raise RuntimeError("fit() must be called before predict_proba()")

        cfg = self.config
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected 2-D (n_series, n_timepoints), got shape {x.shape}")
        if x.shape[1] != self._x.shape[1]:
            raise ValueError(
                f"test series length {x.shape[1]} does not match the fitted "
                f"{self._x.shape[1]}"
            )

        x_train = normalize_series(self._x) if cfg.normalize else self._x
        x_test = normalize_series(x) if cfg.normalize else x
        n_timepoints = x_train.shape[1]

        classes: np.ndarray | None = None
        groups = []

        for g in range(cfg.n_groups):
            f_train = self._features(x_train, g, n_timepoints)
            f_test = self._features(x_test, g, n_timepoints)

            clf = build_classifier(cfg)
            clf.fit(f_train, self._y)
            proba = np.asarray(clf.predict_proba(f_test), dtype=np.float64)

            if classes is None:
                classes = np.asarray(clf.classes_)
            elif not np.array_equal(classes, np.asarray(clf.classes_)):
                # Averaging probabilities positionally across groups is only meaningful if the
                # class ordering is identical. It should be -- TabPFN derives it from the same
                # y every time -- but a silent reordering would corrupt every result while
                # leaving the output well-formed, so it is checked rather than assumed.
                raise RuntimeError(
                    f"group {g} reported classes {clf.classes_!r}, expected {classes!r}"
                )

            groups.append(proba)

        return GroupPredictions(classes=classes, per_group=np.stack(groups))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).labels

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y)))
