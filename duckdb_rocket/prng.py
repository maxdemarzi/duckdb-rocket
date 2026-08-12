"""SplitMix64 — the project's portable pseudo-random stream.

The single most consequential decision in Phase 1. The C++ extension must reproduce
byte-identical kernels from the same seed, and replicating NumPy's Mersenne/PCG stream in C++
is painful enough that people give up and accept "close enough" -- at which point the
conformance test in Phase 4 stops being able to fail. So the stream is defined here, in ~100
lines that port to C++ in an afternoon.

Everything is specified in SPEC.md. This module is the executable copy of that document; if
they disagree, SPEC.md is wrong and should be fixed to match, since the golden vectors are
generated from this code.

Why SplitMix64: 64-bit state, no arrays, no warm-up, and the whole generator is three
multiply-xor-shift rounds. Its statistical quality is far beyond what random convolutional
kernels need. Reference: Steele, Lea & Flood, "Fast splittable pseudorandom number
generators" (OOPSLA 2014).
"""

from __future__ import annotations

import math

_MASK64 = 0xFFFFFFFFFFFFFFFF

# SplitMix64's three magic constants. GOLDEN_GAMMA is the odd 64-bit approximation of
# 2**64 / phi; the two mixing multipliers are from the reference implementation.
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB

# 2**-53. Converting a u64 to a double uses the top 53 bits -- exactly the mantissa width --
# so every representable value in [0, 1) is produced with uniform probability and no value is
# produced twice as often as its neighbour. Using all 64 bits would be worse, not better: the
# low 11 bits cannot survive the conversion and merely introduce rounding.
_TWO_POW_NEG_53 = 1.0 / (1 << 53)


def _mix(z: int) -> int:
    """SplitMix64's output mixing function, applied to a raw state value."""
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
    return z ^ (z >> 31)


def kernel_seed(master_seed: int, index: int) -> int:
    """Seed for kernel `index`, derived from `master_seed` in O(1).

    This exists to make Phase 4 possible. A DuckDB extension wants to generate group 7's
    kernels without first generating groups 0-6, and to fan kernels across threads with no
    shared generator state. A single sequential stream forbids both: you would have to replay
    the stream from the start to reach kernel 9,000.

    So each kernel gets its own independent substream, addressed directly. The construction is
    SplitMix64's own: its state advances by GOLDEN_GAMMA per step and its output is `_mix` of
    that state, so the state at step `index` is available in closed form. This is the generator
    used as its designers intended -- as a splittable seed source -- not an ad-hoc scheme.

    Note `index + 1`: kernel 0 must not be seeded with the unmixed `master_seed` state itself.

    Consequence worth stating plainly: **kernel i is a pure function of (master_seed, i)**. Two
    groups never interfere, kernel count can change without renumbering the kernels that were
    already there, and any kernel is reproducible in isolation for debugging.
    """
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    return _mix((master_seed + (index + 1) * _GOLDEN_GAMMA) & _MASK64)


class SplitMix64:
    """A deterministic 64-bit stream.

    Draw order is part of the specification. Any change to the sequence of calls made while
    generating kernels invalidates every golden vector on disk, so treat the call order in
    `rocket.py` as load-bearing rather than incidental.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        # Accept any Python int -- including negatives and values well past 2**64 -- by
        # reducing into the state's natural width, so callers never have to think about it.
        self._state = seed & _MASK64

    @property
    def state(self) -> int:
        """Current raw state. Exposed for tests and for cross-checking a C++ port."""
        return self._state

    def next_u64(self) -> int:
        """Advance the stream and return the next 64-bit value."""
        self._state = (self._state + _GOLDEN_GAMMA) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
        return z ^ (z >> 31)

    def next_double(self) -> float:
        """Uniform in [0, 1), using the top 53 bits."""
        return (self.next_u64() >> 11) * _TWO_POW_NEG_53

    def next_uniform(self, low: float, high: float) -> float:
        """Uniform in [low, high)."""
        return low + (high - low) * self.next_double()

    def next_below(self, n: int) -> int:
        """Uniform integer in [0, n).

        Multiply-and-floor rather than rejection sampling. This is biased, but the bias is
        bounded by n * 2**-53 -- for the n <= 3 this project actually uses, that is roughly
        one part in 3e15, i.e. unobservable next to the randomness of the kernels themselves.
        Rejection sampling would be unbiased but introduces a data-dependent number of draws,
        which is one more thing a C++ port can get subtly wrong. Determinism is worth more
        here than the last 2**-53 of uniformity.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return int(self.next_double() * n)

    def next_normal(self) -> float:
        """Standard normal, via the Marsaglia polar method.

        Chosen over Box-Muller specifically to avoid `sin`/`cos`. Neither libm nor MSVC's CRT
        guarantees correctly-rounded trigonometric functions, so a trig-based transform can
        differ in the last ulp between our Python oracle and the C++ extension -- exactly the
        kind of drift the golden vectors exist to detect, showing up as noise rather than as
        the real mismatch it would be masking. The polar method needs only `log` and `sqrt`;
        `sqrt` is correctly rounded by IEEE-754 mandate, and `log` is far more consistent
        across implementations than trig.

        The polar method naturally produces two independent normals per accepted pair. **We
        discard the second one.** That wastes roughly half the arithmetic and is entirely
        deliberate: caching the spare makes the value returned by any given call depend on
        whether the count of preceding calls was odd or even, which is a stateful subtlety
        every reimplementation has to reproduce exactly. One draw, one loop, no hidden state.
        """
        while True:
            u = 2.0 * self.next_double() - 1.0
            v = 2.0 * self.next_double() - 1.0
            s = u * u + v * v
            # s == 0 is astronomically unlikely but would divide by zero; s >= 1 is the
            # ordinary ~21% rejection for falling outside the unit disc.
            if 0.0 < s < 1.0:
                return u * math.sqrt(-2.0 * math.log(s) / s)
