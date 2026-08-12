"""The 10-dataset UCR subset every comparison in this project is read against.

Fixed here rather than chosen per experiment. The tabicl fork recorded three separate
occasions where a single-dataset result failed to survive a second dataset, so the subset
exists specifically to make "it helped on one task" an unavailable conclusion.

Selection criteria, in order of how much they mattered:

1. **Series length spread.** Dilation is drawn against series length, so ROCKET's behaviour is
   not length-invariant. The subset runs from 24 to 470 timepoints.
2. **Class-count spread.** TabPFN v2.5 is capped at 10 classes; several UCR datasets exceed it
   and are excluded for that reason rather than by preference.
3. **Train-set size spread.** TabPFN is in-context, so the labelled row count is the context
   size -- a variable that does not exist for a conventionally-trained classifier and is
   therefore easy to forget to vary.
4. **One multivariate dataset**, which the univariate transform cannot yet run. It is listed
   anyway, marked, and skipped with a reason -- an unmet requirement in plain sight is worth
   more than a subset that looks complete.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_timepoints: int
    n_classes: int
    n_train: int
    multivariate: bool = False
    note: str = ""

    @property
    def runnable(self) -> bool:
        """False for anything the end-to-end pipeline cannot handle yet.

        Multivariate *kernels* exist now (SPEC.md 7, in both the Python oracle and the
        extension's `DOUBLE[][]` overload), so this is no longer a statement about the
        transform. What is still missing is the pipeline around it: `phase5_pipeline.py` writes
        one `DOUBLE[]` per row and `load()` squeezes a single channel, so a 6-channel dataset
        has nowhere to go between the loader and `tabfm_classify`. Kept as a marked skip with a
        reason rather than quietly dropped.
        """
        return not self.multivariate


# Sizes are the standard UCR/UEA train splits, recorded so a load can be sanity-checked
# without a network round trip -- and so a silently-changed archive is detectable.
UCR_SUBSET: tuple[DatasetSpec, ...] = (
    DatasetSpec("ItalyPowerDemand", 24, 2, 67, note="shortest series in the subset"),
    DatasetSpec("SyntheticControl", 60, 6, 300, note="multi-class, small context"),
    DatasetSpec("ECG5000", 140, 5, 500, note="largest test split; heavily imbalanced"),
    DatasetSpec("GunPoint", 150, 2, 50, note="tiny context, the in-context stress case"),
    DatasetSpec("Coffee", 286, 2, 28, note="smallest train set in the subset"),
    DatasetSpec("Trace", 275, 4, 100, note="near-separable; a sanity ceiling"),
    DatasetSpec("FaceFour", 350, 4, 24, note="long series, very small context"),
    DatasetSpec("OSULeaf", 427, 6, 200, note="long series, many classes"),
    DatasetSpec("Beef", 470, 5, 30, note="longest series; classically hard"),
    DatasetSpec(
        "BasicMotions",
        100,
        4,
        40,
        multivariate=True,
        note="6 channels; kernels handle this (SPEC.md 7) but the pipeline does not yet",
    ),
)

RUNNABLE_SUBSET = tuple(d for d in UCR_SUBSET if d.runnable)


def load(name: str, split: str):
    """Load one UCR/UEA dataset via `aeon`, returning `(x, y)`.

    `x` comes back as `(n_series, n_timepoints)` for univariate data. aeon returns a 3-D
    `(n_series, n_channels, n_timepoints)` array, and the single channel is squeezed here so
    callers do not each rediscover that.
    """
    from aeon.datasets import load_classification

    x, y = load_classification(name, split=split)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    return x, y


def describe(spec: DatasetSpec) -> str:
    bits = [
        f"{spec.name}",
        f"n={spec.n_timepoints}",
        f"{spec.n_classes} classes",
        f"{spec.n_train} train",
    ]
    if spec.multivariate:
        bits.append("MULTIVARIATE")
    return "  ".join(bits) + (f"  -- {spec.note}" if spec.note else "")
