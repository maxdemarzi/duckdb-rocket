# duckdb-rocket

Training-free time-series classification in DuckDB.

This project brings [RocketPFN](https://arxiv.org/abs/2606.21786) into DuckDB by building the
**feature-extraction half** — a `rocket_transform()` extension — and composing it with
[`anofox_tabfm`](https://github.com/DataZooDE/anofox-tabfm), which ships the `tabpfn-v2-5` and
`tabicl-v2` tabular foundation models.

```
series ──▶ rocket_transform()  ──▶ tabfm_classify()  ──▶ average probs ──▶ label
           (this project)          (anofox_tabfm)        (plain SQL)
```

No gradient descent, no training loop: ROCKET projects each series onto 10,000 random
convolutional kernels, and an in-context tabular model classifies the resulting features
directly. Each kernel contributes 2 features — global max and proportion of positive values —
and the kernels are split into groups classified independently, with the class probabilities
averaged.

The paper's reference result is **0.900 mean accuracy across 92 UCR datasets** at a median of
~30 s per fold.

## Status

The pipeline runs end to end, entirely inside DuckDB. Phases 1–5 of [PLAN.md](PLAN.md) are done:
the ten-dataset UCR subset is measured, **mean accuracy 0.9630**, every dataset reproducing its
local accuracy exactly. That is a subset chosen for spread, not the paper's 92-dataset protocol,
and the two numbers measure different things — see [reference/RESULTS.md](reference/RESULTS.md).

The extension builds on **nine platforms** — Linux (amd64/arm64), macOS (amd64/arm64), Windows
(MSVC and mingw) and all three wasm targets — and is
[submitted to community-extensions](https://github.com/duckdb/community-extensions/pull/2497).

```sql
SELECT rocket_transform(values, 250, 0, 0) FROM series;   -- 500 features from 250 kernels
```

Two findings from Phase 2 changed the design, and anyone reading the paper alongside this repo
should know about them:

**Groups are 250 kernels, not 1,000.** The paper pairs 1,000 kernels per group with TabPFN
v2.5's 2,000-column cap. But 2,000 is the model's *input* ceiling, not the width one estimator
sees — that is 500 (`max_features_per_estimator`), above which features are subsampled per
estimator. Covering a 2,000-feature group needs at least 4 estimators, and `anofox_tabfm` caps
estimators at 1. So this project uses **G=40 groups of 250 kernels** (500 features each), which
keeps the paper's 10,000-kernel budget and its averaging structure while letting a single
estimator see a whole group.

**`tabpfn-v2-5` does not load through the `anofox_tabfm` build the community repository serves**
(`bc6d8af` / `v2026.08.07`), so `tabicl-v2` is the pipeline's default backbone. This is **fixed
upstream in `v2026.08.11`** — the fix has simply not reached the community build yet, and the
checkpoint also needs running through their `convert_weights.py`. Details in
[reference/PHASE2_FINDINGS.md](reference/PHASE2_FINDINGS.md).

### What has been measured

| | |
|---|---|
| Conformance (C++ vs. Python oracle) | max abs diff **1.8e-15**, PPV differences exactly 0 |
| End-to-end predictions, C++ vs. Python features | **150/150 identical** rows (GunPoint) |
| `rocket_transform` vs. numpy oracle | **7.1×** faster at 4,000 series, 1.7× at 200 |
| Pure-SQL ROCKET | correct, and ~**4×10⁵** slower than Python — a fallback, not an option |
| Extension test suite | 12 assertions via DuckDB's sqllogictest runner |

Accuracy on the full ten-dataset UCR/UEA subset, G=40, e=1, through the DuckDB pipeline
(TabICL v2), run on pods:

| Dataset | Test rows | Channels | Accuracy |
|---|---|---|---|
| BasicMotions | 40 | **6** | 1.0000 |
| Coffee | 28 | 1 | 1.0000 |
| Trace | 100 | 1 | 1.0000 |
| GunPoint | 150 | 1 | 0.9933 |
| SyntheticControl | 300 | 1 | 0.9867 |
| FaceFour | 88 | 1 | 0.9773 |
| ItalyPowerDemand | **1029** | 1 | 0.9718 |
| OSULeaf | 242 | 1 | 0.9711 |
| ECG5000 | **4500** | 1 | 0.9480 |
| Beef | 30 | 1 | 0.7667 |

Every dataset that had also been run locally reproduced its accuracy **exactly** on the pod —
two machines, two operating systems, and for GunPoint three different batching configurations.

Where the Python oracle (TabPFN v2.5, pinned fp32) has also been run, it agrees exactly on four
of five datasets; the one difference is two test rows out of thirty on Beef, against a
seed-to-seed sd of 0.0509 on that same dataset — so this subset cannot tell the two backbones
apart. Do not read the mean against the paper's 0.900: that is 92 datasets over 30 resamples,
this is one split of ten. Full detail and caveats in
[reference/RESULTS.md](reference/RESULTS.md).

Accuracies above are from a pod; each report carries an `environment` block recording what it
observed, so provenance is measured rather than asserted. The `rocket_transform` micro-benchmarks
in the table above are still local Windows numbers and are marked as such in
[reference/RESULTS.md](reference/RESULTS.md) — the local box runs this pipeline roughly 1.8×
slower than the pod, so its timings understate it in the direction that flatters it.

## When to use this

Training-free classification is a trade, not a free lunch. What you buy is that **there is no
model**: no fit step, no artefact to store or version, no Python in the serving path. What you pay
is that **every prediction pays full price** — there is nothing to amortise, and inference is
**93.7%** of wall clock in the measured runs.

### It fits well when

- **You have no training pipeline and don't want one.** A new dataset is a `SELECT`, not a project.
- **The data is already in DuckDB.** Features and classification are both SQL; nothing leaves the
  database, and the composition is four lines.
- **Test sets are small to moderate** — hundreds of rows, not tens of thousands. Coffee (28 test
  rows) finishes in 64 s; GunPoint (150) in 185 s.
- **Labelled examples are scarce.** In-context learning needs no gradient steps, so 28 training
  rows is a normal amount rather than a problem.
- **Series are short-to-medium and multivariate is fine.** BasicMotions (6 channels) scores 1.0000.
- **You want a strong baseline fast**, to find out whether a problem is hard before investing in it.

### It fits badly when

- **The test set is large.** Cost is linear in test rows with no amortisation: ItalyPowerDemand's
  1,029 rows took 1,010 s on 16 vCPU. ECG5000's 4,500 rows never finished on CPU at all — four
  attempts, >5h46m, ~44 GB — and needed a GPU to land in 18m39s.
- **The training context is large.** Memory scales with context rows × features and lives outside
  DuckDB's `memory_limit`, so it cannot be bounded by configuration — only by `--test-chunk`, and
  only for the test half. A 500-row context is already heavy.
- **You will classify repeatedly.** A trained model turns training cost into near-free inference.
  This does the opposite. If you are going to run it a thousand times, train something.
- **You need low latency.** Per-call cost is seconds, not milliseconds.
- **Licences matter.** The model weights are third-party and some are non-commercial; see below.

### Against the alternatives

**ROCKET + ridge regression** — the original 2020 pipeline — is the honest comparison, and it has
now been measured on these same ten datasets rather than asserted (`scripts/ridge_baseline.py`):

| | mean accuracy | wins | total time, 10 datasets |
|---|---|---|---|
| ROCKET + ridge | **0.9636** | 3 | **262 s** |
| this pipeline | 0.9615 | 4 | ~3,741 s |
| ties | | 3 | |

**Accuracy is a coin flip; cost is not** — roughly 14×, and that is against our slower Python
feature extractor, not the C++ one. If you have labels and can run scikit-learn, do that. This
project is not faster and is not more accurate.

What it offers instead is the *shape*: no training step to run or schedule, no model artefact to
store, version or serve, and no process boundary between your data and your predictions. That is
worth something when the alternative is standing up a training pipeline for a question you might
ask once.

**The thing ridge cannot do is train without labels**, and that is where the two compose rather
than compete: use this pipeline once to label an unlabelled pool, then distil into a cheap student
that serves in milliseconds. The soft per-class probabilities needed for that already exist inside
the pipeline (averaged over 40 kernel groups). Design, protocol and kill criteria in
[docs/DISTILLATION_PLAN.md](docs/DISTILLATION_PLAN.md) — proposed, not built.

**A trained deep model** (InceptionTime, ROCKET+ridge at scale) will beat this on accuracy-per-
dollar whenever you have enough labels and enough runs to amortise training. Nothing here competes
with that.

**Accuracy, stated plainly.** Mean **0.9630** over the ten-dataset subset, ranging from 1.0000 on
four of them to **0.7667** on Beef. That is a subset chosen for spread rather than difficulty, and
it is *not* comparable to the paper's 0.900 over 92 datasets — different datasets, different
backbone, one split instead of 30 resamples. Treat it as evidence the pipeline works, not as a
benchmark result.

## Try it

```bash
uv sync
scripts\build_extension.bat                    # MSVC Build Tools required
uv run python scripts/conformance.py           # C++ vs. the golden vectors
uv run python scripts/phase5_pipeline.py --dataset GunPoint
```

`build_extension.bat` builds DuckDB v1.5.5 from the pinned submodule along with the extension,
which takes a while the first time. It produces `build/release/duckdb.exe` with the extension
linked in, plus a loadable `rocket.duckdb_extension`.

## `rocket_transform`

```sql
rocket_transform(series DOUBLE[],   kernels_per_group, seed, first_kernel[, n_reference]) -> DOUBLE[]
rocket_transform(series DOUBLE[][], kernels_per_group, seed, first_kernel[, n_reference]) -> DOUBLE[]
```

Returns `kernels_per_group * 2` features, interleaved: element `2i` is kernel `i`'s max and
`2i+1` its PPV.

The second overload is the multivariate one — outer list channels, inner list timepoints. Each
kernel draws a random subset of channels with independent weights per channel, and still yields
exactly 2 features, because the selected channels are summed inside a single convolution. A
one-channel series gives byte-identical kernels either way, which is what keeps the univariate
golden vectors valid (SPEC.md §7.1).

`first_kernel` is what makes groups work. Kernel `i` is a pure function of `(seed, i)`, so group
`g` is global kernel indices `[g*k, (g+1)*k)` and can be generated without generating the groups
before it:

```sql
-- these are the same kernels
SELECT rocket_transform(s, 4, 7, 4) = rocket_transform(s, 8, 7, 0)[9:16] FROM ...;  -- true
```

`n_reference` matters only for **variable-length** data, and it matters a lot. Kernel weights
and lengths do not depend on series length, but dilation and padding do — 64 timepoints and 65
already give different kernels. Without an explicit reference each row draws its bank from its
own length, so column *j* comes from a different kernel in every row while the output stays
perfectly well-formed. Pass `n_reference` (the shortest series in the dataset) whenever lengths
vary; series shorter than it are rejected rather than padded.

Be aware that `max` is biased upward by series length — measured at **+43%** from n=64 to n=512
on pure noise — while PPV is not. If length correlates with the label, half the features carry
that correlation directly. See [SPEC.md](SPEC.md) §8.

The pseudo-random stream is SplitMix64, specified byte-for-byte in [SPEC.md](SPEC.md) so the
C++, the Python reference, and the pure-SQL implementation in [sql/rocket.sql](sql/rocket.sql)
all produce identical kernels.

## Development

Requires [`uv`](https://docs.astral.sh/uv/), CMake, Ninja, and — on Windows — MSVC Build Tools
(clang alone cannot link the extension without the Windows SDK).

```bash
uv run pytest                              # 94 tests, no model weights needed
uv run python scripts/doctor.py            # record the environment tuple
uv run python scripts/emit_golden.py       # regenerate conformance fixtures
uv run python scripts/probe_anofox.py      # re-run the Phase 2 probes
uv run python scripts/accuracy.py --smoke  # oracle accuracy harness
```

The DuckDB CLI under `tools/` and the `duckdb` submodule are both pinned to **v1.5.5**
(`d8cdaa33`) because extension ABI is version-bound.

### Model weights require third-party licences

Two separate gates, and clearing one does not clear the other.

**TabPFN** (for the Python oracle) is gated behind an accepted Prior Labs licence. Register at
<https://ux.priorlabs.ai/account>, accept the licence, then:

```bash
export TABPFN_TOKEN=...      # cached to ~/.cache/tabpfn/auth_token after first use
```

**anofox_tabfm** downloads its own weights and needs its own acknowledgement:

```sql
SET anofox_tabfm_accept_hf_license = true;
FROM tabfm_download('classification', model := 'tabicl-v2');
```

Everything that does not touch model weights — the ROCKET transform, the golden vectors, the
conformance test, the whole test suite — runs without either.

## License

MIT — see [LICENSE](LICENSE). The license matches `anofox_tabfm` to keep the door open for
upstreaming.
