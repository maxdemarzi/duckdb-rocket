"""ROCKET kernel generation and the max/PPV transform — the Phase 1 oracle.

Correctness matters more than speed here: this module defines what the C++ extension must
reproduce, and its output becomes the golden vectors in `reference/golden/`. Where a clearer
formulation and a faster one conflict, take the clearer one.

Reference: Dempster, Petitjean & Webb, "ROCKET: Exceptionally fast and accurate time series
classification using random convolutional kernels" (2020). The kernel-generation scheme below
follows that paper; the PRNG underneath does not follow its reference implementation, for the
reasons in `prng.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .prng import SplitMix64, kernel_seed

# ROCKET draws kernel length uniformly from this set. Fixed by the paper, not a tunable.
KERNEL_LENGTHS = (7, 9, 11)

# Each kernel contributes exactly 2 features (global max and PPV). The paper pairs this with
# 1,000 kernels per group to land on 2,000 features, TabPFN v2.5's stated column cap -- but see
# pipeline.TABPFN_MAX_FEATURES_PER_ESTIMATOR: a single estimator only ever sees 500 of them, so
# this project groups 250 kernels instead. The per-kernel count is fixed by the paper either way.
FEATURES_PER_KERNEL = 2


@dataclass(frozen=True)
class Kernels:
    """A generated kernel bank, in the flat layout the C++ extension will use.

    Weights are stored end-to-end in one array with an offsets index rather than as a list of
    per-kernel arrays. That is slightly less convenient in Python and considerably more
    convenient in C++, and keeping one layout across both removes a translation step -- and
    with it a place for the two implementations to disagree without the golden vectors
    noticing.
    """

    n_timepoints: int
    """Series length the dilations were drawn against. See `transform`'s length check."""

    lengths: np.ndarray  # int64, shape (num_kernels,)
    weights: np.ndarray  # float64, shape (sum(lengths),)
    offsets: np.ndarray  # int64, shape (num_kernels + 1,) -- weights[offsets[i]:offsets[i+1]]
    biases: np.ndarray  # float64, shape (num_kernels,)
    dilations: np.ndarray  # int64, shape (num_kernels,)
    paddings: np.ndarray  # int64, shape (num_kernels,)

    @property
    def num_kernels(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def num_features(self) -> int:
        return self.num_kernels * FEATURES_PER_KERNEL


def generate_kernel(seed: int, index: int, n_timepoints: int) -> tuple:
    """Generate kernel `index` in isolation.

    **The order of draws below is part of the specification.** Reordering them, or inserting a
    draw, silently changes every kernel and invalidates every golden vector. See SPEC.md.

    Returns `(length, weights, bias, dilation, padding)` with `weights` a plain list.
    """
    rng = SplitMix64(kernel_seed(seed, index))

    # 1. Length, uniform over {7, 9, 11}.
    length = KERNEL_LENGTHS[rng.next_below(len(KERNEL_LENGTHS))]

    # 2. Weights, standard normal, then mean-centred. The centring is not cosmetic: it makes
    #    each kernel's response invariant to a constant offset in the series, so PPV measures
    #    shape rather than level.
    raw = [rng.next_normal() for _ in range(length)]
    mean = sum(raw) / length
    weights = [w - mean for w in raw]

    # 3. Bias, uniform on [-1, 1). Shifts the threshold PPV counts against.
    bias = rng.next_uniform(-1.0, 1.0)

    # 4. Dilation, log-uniform. The upper bound is the largest dilation for which the kernel's
    #    full span still fits inside the series, so `output_length` below stays >= 1 by
    #    construction rather than by a runtime guard.
    upper = math.log2((n_timepoints - 1) / (length - 1))
    dilation = int(math.floor(2.0 ** rng.next_uniform(0.0, upper)))

    # 5. Padding, present or absent with equal probability. When present it is exactly enough
    #    to centre the kernel, so the series' first and last points get the same coverage as
    #    its middle.
    padding = ((length - 1) * dilation) // 2 if rng.next_below(2) == 1 else 0

    return length, weights, bias, dilation, padding


def generate_kernels(
    seed: int,
    n_timepoints: int,
    num_kernels: int = 10_000,
    *,
    first_kernel: int = 0,
) -> Kernels:
    """Generate `num_kernels` kernels starting at global index `first_kernel`.

    `first_kernel` is how groups are expressed. Group `g` of `G`, with `k` kernels each, is
    `generate_kernels(seed, n, k, first_kernel=g * k)` -- and because each kernel is a pure
    function of `(seed, global index)`, that produces exactly the kernels a single
    `generate_kernels(seed, n, G * k)` call would have put in those positions. The groups are
    a partition of one bank, not ten unrelated banks.
    """
    if num_kernels <= 0:
        raise ValueError(f"num_kernels must be positive, got {num_kernels}")
    if n_timepoints < max(KERNEL_LENGTHS):
        raise ValueError(
            f"n_timepoints={n_timepoints} is shorter than the longest kernel "
            f"({max(KERNEL_LENGTHS)}); ROCKET has no meaningful output here"
        )

    lengths = np.empty(num_kernels, dtype=np.int64)
    biases = np.empty(num_kernels, dtype=np.float64)
    dilations = np.empty(num_kernels, dtype=np.int64)
    paddings = np.empty(num_kernels, dtype=np.int64)
    weight_blocks = []

    for i in range(num_kernels):
        length, weights, bias, dilation, padding = generate_kernel(
            seed, first_kernel + i, n_timepoints
        )
        lengths[i] = length
        biases[i] = bias
        dilations[i] = dilation
        paddings[i] = padding
        weight_blocks.append(np.asarray(weights, dtype=np.float64))

    offsets = np.zeros(num_kernels + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    return Kernels(
        n_timepoints=n_timepoints,
        lengths=lengths,
        weights=np.concatenate(weight_blocks),
        offsets=offsets,
        biases=biases,
        dilations=dilations,
        paddings=paddings,
    )


def apply_kernel(x: np.ndarray, weights, length, bias, dilation, padding):
    """Convolve one kernel over a batch of series; return `(max, ppv)` per series.

    `x` is 2-D, `(n_series, n_timepoints)`. Batching over series here rather than looping in
    the caller is what keeps the oracle usable on real UCR data -- the per-kernel Python
    overhead is paid 10,000 times instead of 10,000 x n_series times.
    """
    n_series, n = x.shape
    output_length = n + 2 * padding - (length - 1) * dilation
    if output_length <= 0:
        raise ValueError(
            f"kernel spans {(length - 1) * dilation + 1} points but the series has {n} "
            f"(padding={padding}); this kernel was generated against a longer series"
        )

    if padding:
        # Zero-padding both ends is exactly equivalent to the reference implementation's
        # skip-out-of-range-taps loop, since a skipped tap contributes nothing to the sum.
        padded = np.zeros((n_series, n + 2 * padding), dtype=np.float64)
        padded[:, padding : padding + n] = x
    else:
        padded = x

    # Accumulate one tap at a time. The loop runs at most 11 times and each step is a full
    # vectorised multiply-add over every series at once.
    out = np.full((n_series, output_length), bias, dtype=np.float64)
    for j in range(length):
        start = j * dilation
        out += weights[j] * padded[:, start : start + output_length]

    return out.max(axis=1), (out > 0).sum(axis=1) / output_length


def transform(x: np.ndarray, kernels: Kernels) -> np.ndarray:
    """Apply a kernel bank to `(n_series, n_timepoints)`, returning `(n_series, 2 * K)`.

    **Feature layout:** kernel `i` occupies columns `2i` (global max) and `2i + 1` (PPV). Kept
    interleaved rather than blocked -- all maxima then all PPVs -- so that slicing a contiguous
    column range yields whole kernels, which is what group extraction needs.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (n_series, n_timepoints), got shape {x.shape}")

    n_series, n = x.shape
    if n != kernels.n_timepoints:
        # Not a soft warning. Dilations were drawn against a specific series length, so a
        # different length here silently produces features that are not comparable with the
        # ones the classifier was given as context -- the kind of mistake that shows up as a
        # mediocre accuracy number rather than as an error. Variable-length support is a
        # deliberate later step, not something to allow by accident.
        raise ValueError(
            f"series length {n} does not match the {kernels.n_timepoints} these kernels were "
            f"generated for; regenerate the bank or resample the series"
        )

    features = np.empty((n_series, kernels.num_features), dtype=np.float64)
    for i in range(kernels.num_kernels):
        lo, hi = kernels.offsets[i], kernels.offsets[i + 1]
        maxima, ppv = apply_kernel(
            x,
            kernels.weights[lo:hi],
            int(kernels.lengths[i]),
            float(kernels.biases[i]),
            int(kernels.dilations[i]),
            int(kernels.paddings[i]),
        )
        features[:, 2 * i] = maxima
        features[:, 2 * i + 1] = ppv

    return features


def normalize_series(x: np.ndarray) -> np.ndarray:
    """Per-series zero-mean, unit-variance normalisation.

    Kept as an explicit step the caller opts into rather than folded into `transform`. Most UCR
    datasets ship already normalised, and normalising twice is harmless while normalising
    neither time is not -- so the decision belongs where someone can see it, next to the data
    loading, not buried in the transform.

    Constant series are left centred rather than divided by their standard deviation.

    The guard below tests against the data's own floating-point noise floor, not against zero,
    and that distinction is the whole point. `np.std` of a genuinely constant series is not
    exactly 0.0 -- for `full(128, 4.2)` it comes out near 8.9e-16, since the mean is not
    representable and the residuals are not identically zero. A `std > 0` guard therefore
    passes, and the division computes noise divided by noise: every value in the series
    becomes exactly -1.0 or +1.0. That is far worse than not normalising, because the result
    is finite, well-formed, and completely fabricated -- ROCKET would then extract features
    from pure rounding error and the series would look like strong signal to the classifier.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    # Below roughly this, `std` is indistinguishable from the error of having computed it.
    scale = np.maximum(np.abs(mean), np.abs(x).max(axis=1, keepdims=True))
    noise_floor = 8.0 * np.finfo(np.float64).eps * np.where(scale > 0.0, scale, 1.0)

    return (x - mean) / np.where(std > noise_floor, std, 1.0)
