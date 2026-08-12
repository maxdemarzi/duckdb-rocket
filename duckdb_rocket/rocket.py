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
    weights: np.ndarray  # float64, shape (sum(lengths * channels_used),)
    offsets: np.ndarray  # int64, shape (num_kernels + 1,) -- weights[offsets[i]:offsets[i+1]]
    biases: np.ndarray  # float64, shape (num_kernels,)
    dilations: np.ndarray  # int64, shape (num_kernels,)
    paddings: np.ndarray  # int64, shape (num_kernels,)

    n_channels: int = 1
    """Channels in the series these kernels were drawn against. 1 means univariate."""

    channels: np.ndarray | None = None
    """int64, flat, the channel indices each kernel reads. `None` for univariate.

    Kernel `i` uses `channels[channel_offsets[i]:channel_offsets[i+1]]`, sorted ascending, and
    its weights are laid out channel-major in the same order.
    """

    channel_offsets: np.ndarray | None = None  # int64, shape (num_kernels + 1,)

    @property
    def num_kernels(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def num_features(self) -> int:
        # Two per kernel regardless of how many channels it reads: the channels are summed
        # inside one convolution rather than producing features of their own.
        return self.num_kernels * FEATURES_PER_KERNEL

    @property
    def is_multivariate(self) -> bool:
        return self.n_channels > 1

    def channels_for(self, index: int) -> np.ndarray:
        """The channel indices kernel `index` reads."""
        if self.channels is None or self.channel_offsets is None:
            return np.zeros(1, dtype=np.int64)
        start, stop = self.channel_offsets[index], self.channel_offsets[index + 1]
        return self.channels[start:stop]


def select_channels(rng: SplitMix64, n_channels: int, n_selected: int) -> list[int]:
    """`n_selected` distinct channels from `[0, n_channels)`, by partial Fisher-Yates.

    Exactly `n_selected` draws, whatever comes up -- no rejection and no data-dependent draw
    count, for the same reason `next_below` does not reject (SPEC.md 1.4): a port that consumes
    a different number of draws desynchronises the whole stream from that point on.

    The result is **sorted**. Selection order carries no meaning, and sorting means a channel's
    weight block is located by position in a stable order rather than by remembering whichever
    order the swaps happened to produce.
    """
    perm = list(range(n_channels))
    for k in range(n_selected):
        j = k + rng.next_below(n_channels - k)
        perm[k], perm[j] = perm[j], perm[k]
    return sorted(perm[:n_selected])


def generate_kernel(seed: int, index: int, n_timepoints: int, n_channels: int = 1) -> tuple:
    """Generate kernel `index` in isolation.

    **The order of draws below is part of the specification.** Reordering them, or inserting a
    draw, silently changes every kernel and invalidates every golden vector. See SPEC.md.

    Returns `(length, weights, bias, dilation, padding, channels)`, `weights` a plain list laid
    out channel-major -- all of the first selected channel's `length` values, then the next.
    """
    rng = SplitMix64(kernel_seed(seed, index))

    # 1. Length, uniform over {7, 9, 11}.
    length = KERNEL_LENGTHS[rng.next_below(len(KERNEL_LENGTHS))]

    # 2. Channels -- but only when there is a choice to make. At n_channels == 1 the subset is
    #    forced, so spending a draw on it would shift every later value in the stream and
    #    invalidate every committed golden vector in exchange for deciding nothing. See
    #    SPEC.md 7.1: the univariate stream is exactly as it was before multivariate existed.
    if n_channels > 1:
        upper_channels = math.log2(n_channels + 1)
        n_selected = int(math.floor(2.0 ** rng.next_uniform(0.0, upper_channels)))
        # The bound already gives 1 <= n_selected <= n_channels; clamped because a silently
        # out-of-range subset would be far harder to notice than a redundant min().
        n_selected = max(1, min(n_selected, n_channels))
        channels = select_channels(rng, n_channels, n_selected)
    else:
        n_selected = 1
        channels = [0]

    # 3. Weights, standard normal, then mean-centred **per channel**. The centring is not
    #    cosmetic: it makes each kernel's response invariant to a constant offset in the series,
    #    so PPV measures shape rather than level. Per channel rather than globally because each
    #    channel carries its own offset, and one global mean would only remove their average.
    weights: list[float] = []
    for _ in range(n_selected):
        raw = [rng.next_normal() for _ in range(length)]
        mean = sum(raw) / length
        weights.extend(w - mean for w in raw)

    # 4. Bias, uniform on [-1, 1). Shifts the threshold PPV counts against.
    bias = rng.next_uniform(-1.0, 1.0)

    # 5. Dilation, log-uniform. The upper bound is the largest dilation for which the kernel's
    #    full span still fits inside the series, so `output_length` below stays >= 1 by
    #    construction rather than by a runtime guard.
    upper = math.log2((n_timepoints - 1) / (length - 1))
    dilation = int(math.floor(2.0 ** rng.next_uniform(0.0, upper)))

    # 6. Padding, present or absent with equal probability. When present it is exactly enough
    #    to centre the kernel, so the series' first and last points get the same coverage as
    #    its middle.
    padding = ((length - 1) * dilation) // 2 if rng.next_below(2) == 1 else 0

    return length, weights, bias, dilation, padding, channels


def generate_kernels(
    seed: int,
    n_timepoints: int,
    num_kernels: int = 10_000,
    *,
    first_kernel: int = 0,
    n_channels: int = 1,
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

    if n_channels < 1:
        raise ValueError(f"n_channels must be positive, got {n_channels}")

    lengths = np.empty(num_kernels, dtype=np.int64)
    biases = np.empty(num_kernels, dtype=np.float64)
    dilations = np.empty(num_kernels, dtype=np.int64)
    paddings = np.empty(num_kernels, dtype=np.int64)
    weight_blocks = []
    channel_blocks = []
    channel_counts = np.empty(num_kernels, dtype=np.int64)

    for i in range(num_kernels):
        length, weights, bias, dilation, padding, channels = generate_kernel(
            seed, first_kernel + i, n_timepoints, n_channels
        )
        lengths[i] = length
        biases[i] = bias
        dilations[i] = dilation
        paddings[i] = padding
        channel_counts[i] = len(channels)
        weight_blocks.append(np.asarray(weights, dtype=np.float64))
        channel_blocks.append(np.asarray(channels, dtype=np.int64))

    # Weight blocks are `length * channels_used` long, which reduces to `length` when
    # univariate -- so the layout is unchanged for the existing golden vectors.
    offsets = np.zeros(num_kernels + 1, dtype=np.int64)
    np.cumsum(lengths * channel_counts, out=offsets[1:])

    channel_offsets = np.zeros(num_kernels + 1, dtype=np.int64)
    np.cumsum(channel_counts, out=channel_offsets[1:])

    return Kernels(
        n_timepoints=n_timepoints,
        lengths=lengths,
        weights=np.concatenate(weight_blocks),
        offsets=offsets,
        biases=biases,
        dilations=dilations,
        paddings=paddings,
        n_channels=n_channels,
        channels=np.concatenate(channel_blocks) if n_channels > 1 else None,
        channel_offsets=channel_offsets if n_channels > 1 else None,
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


def apply_kernel_multivariate(x, weights, length, bias, dilation, padding, channels):
    """The multivariate convolution: one output, summed across the kernel's channels.

    `x` is 3-D, `(n_series, n_channels, n_timepoints)`. The channels are summed *inside* one
    convolution rather than producing separate outputs, which is why a kernel still yields
    exactly 2 features however many channels it reads (SPEC.md 7.5).
    """
    n_series, _, n = x.shape
    output_length = n + 2 * padding - (length - 1) * dilation
    if output_length <= 0:
        raise ValueError(
            f"kernel spans {(length - 1) * dilation + 1} points but the series has {n} "
            f"(padding={padding}); this kernel was generated against a longer series"
        )

    out = np.full((n_series, output_length), bias, dtype=np.float64)
    # Channel-major, and within a channel one tap at a time -- SPEC.md 7.5 fixes this order
    # because floating-point addition is not associative and the golden vectors are compared
    # at a tight tolerance.
    for ci, channel in enumerate(channels):
        xc = x[:, channel, :]
        if padding:
            padded = np.zeros((n_series, n + 2 * padding), dtype=np.float64)
            padded[:, padding : padding + n] = xc
        else:
            padded = xc
        block = weights[ci * length : (ci + 1) * length]
        for j in range(length):
            start = j * dilation
            out += block[j] * padded[:, start : start + output_length]

    return out.max(axis=1), (out > 0).sum(axis=1) / output_length


def transform(x: np.ndarray, kernels: Kernels) -> np.ndarray:
    """Apply a kernel bank, returning `(n_series, 2 * K)`.

    Accepts `(n_series, n_timepoints)` for univariate and
    `(n_series, n_channels, n_timepoints)` for multivariate; the bank decides which is expected,
    since kernels drawn against one channel count cannot be applied to another.

    **Feature layout:** kernel `i` occupies columns `2i` (global max) and `2i + 1` (PPV). Kept
    interleaved rather than blocked -- all maxima then all PPVs -- so that slicing a contiguous
    column range yields whole kernels, which is what group extraction needs.
    """
    x = np.asarray(x, dtype=np.float64)

    if kernels.is_multivariate:
        return _transform_multivariate(x, kernels)

    # A 3-D array with a single channel is accepted and squeezed: SPEC.md 7.1 guarantees the
    # kernels are identical either way, so rejecting it would be pedantry rather than safety.
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (n_series, n_timepoints), got shape {x.shape}")

    n = x.shape[1]
    if n != kernels.n_timepoints:
        # Not a soft warning. Dilations were drawn against a specific series length, so a
        # different length here silently produces features that are not comparable with the
        # ones the classifier was given as context -- the kind of mistake that shows up as a
        # mediocre accuracy number rather than as an error. For genuinely ragged data use
        # `transform_variable`, which makes the reference length explicit (SPEC.md 8).
        raise ValueError(
            f"series length {n} does not match the {kernels.n_timepoints} these kernels were "
            f"generated for; regenerate the bank, resample the series, or use "
            f"transform_variable for ragged input"
        )

    return _transform_fixed(x, kernels)


def _transform_fixed(x: np.ndarray, kernels: Kernels) -> np.ndarray:
    """The univariate transform proper, with the length check already done."""
    n_series = x.shape[0]
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


def _transform_multivariate(x: np.ndarray, kernels: Kernels) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(
            f"these kernels were drawn for {kernels.n_channels} channels, so a 3-D "
            f"(n_series, n_channels, n_timepoints) array is expected; got shape {x.shape}"
        )
    n_series, n_channels, n = x.shape
    if n_channels != kernels.n_channels:
        # Channel indices were drawn against a specific count, so a different one here does not
        # merely reshape the problem -- kernel 7 would be reading a different physical signal
        # than the one it selected.
        raise ValueError(
            f"series has {n_channels} channels but these kernels were drawn for "
            f"{kernels.n_channels}; regenerate the bank"
        )
    if n != kernels.n_timepoints:
        raise ValueError(
            f"series length {n} does not match the {kernels.n_timepoints} these kernels were "
            f"generated for; regenerate the bank or resample the series"
        )

    features = np.empty((n_series, kernels.num_features), dtype=np.float64)
    for i in range(kernels.num_kernels):
        lo, hi = kernels.offsets[i], kernels.offsets[i + 1]
        maxima, ppv = apply_kernel_multivariate(
            x,
            kernels.weights[lo:hi],
            int(kernels.lengths[i]),
            float(kernels.biases[i]),
            int(kernels.dilations[i]),
            int(kernels.paddings[i]),
            kernels.channels_for(i),
        )
        features[:, 2 * i] = maxima
        features[:, 2 * i + 1] = ppv

    return features


def transform_variable(series: list, kernels: Kernels) -> np.ndarray:
    """Apply one bank to series of **differing lengths** (SPEC.md 8).

    `series` is a list of 1-D arrays (univariate) or 2-D `(n_channels, n_timepoints)` arrays
    (multivariate). The bank is used unchanged for every one of them, which is the whole point:
    a classifier compares feature `j` across rows, so column `j` has to come from the same
    kernel every time. Generating a bank per row from that row's own length -- the obvious
    implementation, since each series carries its length -- produces a well-formed matrix in
    which every row was measured with a different instrument.

    Every series must be at least `kernels.n_timepoints` long. That is what makes the bank safe
    to apply: the dilation bound guarantees the kernel span fits inside `n_timepoints`, so any
    longer series admits `output_length >= 1` structurally. Shorter series are rejected rather
    than padded, because padding would fabricate data and change what the features measure.

    **`max` is biased upward by series length** and PPV is not; see SPEC.md 8.3. If length
    correlates with the label, half the features carry that correlation directly.
    """
    if not series:
        raise ValueError("no series given")

    features = np.empty((len(series), kernels.num_features), dtype=np.float64)
    for row, item in enumerate(series):
        x = np.asarray(item, dtype=np.float64)
        if kernels.is_multivariate:
            if x.ndim != 2:
                raise ValueError(
                    f"series {row} has shape {x.shape}; a multivariate bank expects 2-D "
                    f"(n_channels, n_timepoints) per series"
                )
            n = x.shape[1]
        else:
            if x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            if x.ndim != 1:
                raise ValueError(
                    f"series {row} has shape {x.shape}; a univariate bank expects 1-D per series"
                )
            n = x.shape[0]

        if n < kernels.n_timepoints:
            raise ValueError(
                f"series {row} has {n} timepoints but the bank was drawn against "
                f"{kernels.n_timepoints}; a shorter series is rejected rather than padded "
                f"(SPEC.md 8.2). Regenerate the bank against the shortest series"
            )

        # One row at a time: the arrays are ragged, so there is nothing to batch over.
        batched = x[None, :, :] if kernels.is_multivariate else x[None, :]
        single = (
            _transform_multivariate(batched, _rebound(kernels, n))
            if kernels.is_multivariate
            else _transform_fixed(batched, _rebound(kernels, n))
        )
        features[row] = single[0]

    return features


def _rebound(kernels: Kernels, n_timepoints: int) -> Kernels:
    """The same bank, relabelled for a longer series.

    Only `n_timepoints` changes, and it is used purely for the length check in the transform --
    every kernel parameter is carried over untouched. This is what applying one bank to a
    longer series means, and doing it by copy rather than by loosening the check keeps the
    check itself strict everywhere else.
    """
    return Kernels(
        n_timepoints=n_timepoints,
        lengths=kernels.lengths,
        weights=kernels.weights,
        offsets=kernels.offsets,
        biases=kernels.biases,
        dilations=kernels.dilations,
        paddings=kernels.paddings,
        n_channels=kernels.n_channels,
        channels=kernels.channels,
        channel_offsets=kernels.channel_offsets,
    )


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

    # Always the time axis, which is the last one either way: (n_series, n_timepoints) for
    # univariate and (n_series, n_channels, n_timepoints) for multivariate. Normalising a
    # multivariate series **per channel** is the only defensible reading -- channels are
    # different physical quantities in different units, and pooling them would let a
    # large-amplitude channel set the scale for a small one.
    axis = x.ndim - 1
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)

    # Below roughly this, `std` is indistinguishable from the error of having computed it.
    scale = np.maximum(np.abs(mean), np.abs(x).max(axis=axis, keepdims=True))
    noise_floor = 8.0 * np.finfo(np.float64).eps * np.where(scale > 0.0, scale, 1.0)

    return (x - mean) / np.where(std > noise_floor, std, 1.0)
