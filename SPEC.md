# duckdb-rocket — Kernel Generation Specification

Normative description of the pseudo-random stream, kernel generation, and the ROCKET
transform, written so the C++ extension can be implemented from this document alone.

The Python implementation in `duckdb_rocket/` is the reference. Golden vectors in
`reference/golden/` are generated from it. **If this document and the code disagree, the code
is right and this document is a bug** — the golden vectors are what the conformance test
actually checks against.

All arithmetic is IEEE-754 `binary64` (C `double`) unless stated otherwise. All integer
arithmetic marked *u64* is unsigned 64-bit with wrapping overflow.

---

## 1. The pseudo-random stream: SplitMix64

Chosen over NumPy's generators specifically because it must be reimplemented in C++ exactly.
Reference: Steele, Lea & Flood, *Fast splittable pseudorandom number generators* (OOPSLA 2014).

### 1.1 Constants

| Name | Value |
|---|---|
| `GOLDEN_GAMMA` | `0x9E3779B97F4A7C15` |
| `MIX_A` | `0xBF58476D1CE4E5B9` |
| `MIX_B` | `0x94D049BB133111EB` |

### 1.2 The mixing function

```
mix(z: u64) -> u64:
    z = (z XOR (z >> 30)) * MIX_A     # wrapping
    z = (z XOR (z >> 27)) * MIX_B     # wrapping
    return z XOR (z >> 31)
```

`>>` is a *logical* (zero-filling) shift.

### 1.3 The generator

State is a single u64. A seed of any width is reduced modulo 2^64.

```
next_u64(state) -> u64:
    state = state + GOLDEN_GAMMA      # wrapping
    return mix(state)
```

**Conformance vectors.** Seeded with `0`, the first five outputs are:

```
0xE220A8397B1DCDAF
0x6E789E6AA1B965F4
0x06C45D188009454F
0xF88BB8A8724C81EC
0x1B39896A51A8749B
```

These are the published reference values. A port that reproduces them has `next_u64` right.

### 1.4 Derived draws

```
next_double() -> double:
    return (next_u64() >> 11) * 2^-53
```

Uses the top 53 bits — exactly the `binary64` mantissa width. Result is in `[0, 1)`.

```
next_uniform(lo, hi) -> double:
    return lo + (hi - lo) * next_double()
```

```
next_below(n) -> integer:            # n > 0
    return floor(next_double() * n)
```

`next_below` is **deliberately biased**, by at most `n * 2^-53`. Rejection sampling is not
used: it would consume a data-dependent number of draws, which is an easy thing for a port to
get subtly wrong, and at the `n <= 3` used here the bias is about one part in `3e15`.

### 1.5 Normal draws — Marsaglia polar

```
next_normal() -> double:
    loop:
        u = 2 * next_double() - 1
        v = 2 * next_double() - 1
        s = u*u + v*v
        if 0 < s < 1:
            return u * sqrt(-2 * ln(s) / s)
```

Two properties are normative and easy to get wrong:

1. **The second normal of the pair is discarded.** The polar method naturally yields
   `v * sqrt(...)` as well. Do not cache and return it on the next call. Caching would make
   each call's result depend on the parity of the number of preceding calls — hidden state
   every reimplementation would have to match.
2. **Both `u` and `v` are drawn on every iteration**, including rejected ones. A rejected
   iteration consumes exactly two `next_double()` calls.

Box-Muller is **not** used, to avoid `sin`/`cos`: neither glibc nor the MSVC CRT guarantees
correctly-rounded trigonometric functions, so a trig-based transform can differ in the last
ulp between platforms. The polar method needs only `sqrt` (correctly rounded by IEEE-754
mandate) and `ln`.

---

## 2. Per-kernel seeding

Kernel `i` is a **pure function of `(master_seed, i)`**. There is no sequential stream running
across kernels.

```
kernel_seed(master_seed: u64, index: integer >= 0) -> u64:
    return mix(master_seed + (index + 1) * GOLDEN_GAMMA)    # wrapping
```

Kernel `i` is then generated from a generator seeded with `kernel_seed(master_seed, i)` —
meaning its first `next_u64()` is `mix(kernel_seed(...) + GOLDEN_GAMMA)`, not
`kernel_seed(...)` itself.

**Why this construction.** It is SplitMix64 used as its designers intended: the state after
`index + 1` steps is available in closed form, so any kernel is addressable in O(1). That is
what lets the extension generate group 7 without generating groups 0–6, and fan kernels across
threads with no shared generator state.

**The `index + 1` is required.** With `index`, kernel 0's seed would be `mix(master_seed)`,
colliding with the unmixed state.

**A tempting wrong version:** seeding kernel `i` with `master_seed + i * GOLDEN_GAMMA`
directly, *without* `mix`. Since the generator advances its state by exactly `GOLDEN_GAMMA` per
call, kernel `i`'s stream would be kernel `i+1`'s stream shifted by one step, and neighbouring
kernels would share nearly all their randomness.

---

## 3. Kernel generation

Given `master_seed`, global kernel index `i`, and series length `n`:

```
rng = SplitMix64(kernel_seed(master_seed, i))

length   = LENGTHS[rng.next_below(3)]         where LENGTHS = (7, 9, 11)
raw[j]   = rng.next_normal()                  for j = 0 .. length-1
weights  = raw - mean(raw)
bias     = rng.next_uniform(-1, 1)
dilation = floor(2 ^ rng.next_uniform(0, log2((n - 1) / (length - 1))))
padding  = ((length - 1) * dilation) / 2      if rng.next_below(2) == 1 else 0
                                              (integer division)
```

**The order of these draws is normative.** Reordering them, or inserting a draw, changes every
kernel and invalidates every golden vector.

Notes:

- `mean(raw)` is the plain arithmetic mean, `sum(raw) / length`, summed in index order.
  Mean-centring makes each kernel's response invariant to a constant offset in the series, so
  PPV measures shape rather than level.
- The `dilation` bound guarantees `(length - 1) * dilation <= n - 1`, which is what makes
  `output_length >= 1` structural rather than a runtime check.
- `dilation >= 1` always, since the exponent is `>= 0`.

---

## 4. The transform

For one series `x` of length `n` and one kernel:

```
output_length = n + 2 * padding - (length - 1) * dilation

conv[k] = bias + SUM over j = 0 .. length-1 of
              weights[j] * xpad[k + j * dilation]
          for k = 0 .. output_length - 1
```

where `xpad` is `x` with `padding` zeros prepended and `padding` zeros appended. Equivalently,
and as the reference implementation phrases it: iterate `i` from `-padding`, index `x` directly
at `i + j * dilation`, and skip taps falling outside `[0, n)`. The two are identical because a
skipped tap contributes zero.

Two features per kernel:

```
max = maximum over k of conv[k]
ppv = count(conv[k] > 0) / output_length        # strictly greater than zero
```

**Accumulation order.** The reference accumulates one tap at a time across all output
positions — that is, `j` is the outer loop. Floating-point addition is not associative, so a
different accumulation order can differ in the last ulp. The conformance test uses a tight but
non-zero tolerance for exactly this reason; bit-identical output is not required, and should
not be assumed.

---

## 5. Feature layout

For a bank of `K` kernels, the feature vector has `2K` entries:

| Column | Content |
|---|---|
| `2i` | kernel `i`'s **max** |
| `2i + 1` | kernel `i`'s **PPV** |

Interleaved, not blocked. This is what makes a contiguous column range correspond to a whole
set of kernels, which is what group extraction needs.

---

## 6. Groups

The paper's configuration is 10,000 kernels as `G = 10` groups of 1,000, each group yielding
`1000 * 2 = 2000` features — which the paper matches to TabPFN v2.5's 2,000-column cap.

**This project uses `G = 40` groups of 250 instead**, for 500 features per group. 2,000 is
TabPFN v2.5's *input* ceiling, but a single estimator only ever sees 500 features
(`max_features_per_estimator`); wider groups are subsampled per estimator and need `e >= 4` to
be covered at all, which `anofox_tabfm` — capped at `e = 1` — cannot supply. Narrowing the
group preserves the 10,000-kernel budget and the averaging structure while keeping one
estimator's view complete. See `reference/PHASE2_FINDINGS.md`.

Nothing in this section depends on which of the two splits is used; `G` is a parameter.

Group `g` consists of global kernel indices `[g * k, (g + 1) * k)` where `k = K / G`. Because
kernel `i` is a pure function of `(master_seed, i)`, **the groups are a partition of one bank,
not `G` independent banks**: generating group `g` alone yields exactly the kernels a full
10,000-kernel generation would have placed at those indices.

Each group is classified independently and the resulting class probabilities are averaged.

---

## 7. What is not yet specified

Deliberately open; these are later phases and will be added here before being implemented.

- **Multivariate series.** The paper assigns each kernel a random subset of `K` channels with
  independent weights per channel, still producing 2 features per kernel. The draw order for
  channel selection is not yet fixed.
- **Variable-length series.** Dilations are drawn against a specific `n`. The reference
  implementation rejects a length mismatch rather than silently producing incomparable
  features.
- **Series normalisation.** `normalize_series` is an explicit caller-side step, not part of
  the transform. Note its guard is against the floating-point noise floor rather than against
  zero: `std` of a constant series is near `8.9e-16`, not `0.0`, and dividing by it turns
  rounding error into a series of exactly ±1.
