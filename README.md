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

Where that 92 comes from is worth knowing, because it bounds this project too. The standard UCR
bake-off archive has **112** datasets; the tabular foundation models cap at **10 classes**
(`max_classes: 10` in every `anofox_tabfm` model, and the exported graph's class head is literally
10 wide). Filtering 112 to ≤10 classes leaves **exactly 92** — so the paper's protocol is not a
curated selection, it is the subset that fits under the class ceiling. The 20 it excludes are the
many-class datasets: ShapesAll (60), the three Pig sets (52), FiftyWords (50), Phoneme (39), Adiac
(37), down to InsectWingbeatSound (11). Nobody's published 0.900 includes them either.

## Status

The pipeline runs end to end, entirely inside DuckDB, and has now been measured on the paper's own
92-dataset protocol (the reachable 92 of the UCR archive's 112 equal-length univariate datasets —
the other 20 exceed every `anofox_tabfm` model's 10-class cap): **mean accuracy 0.8770**, against
the paper's reported **0.900**. Not resampled — one split per dataset, the archive's own, which is
the honest form of a number this cheap to get (6.3 GPU-hours) rather than their 30-resample
average. Full table, and the two ways this run's configuration differs from the paper's, in
[reference/RESULTS.md](reference/RESULTS.md#the-papers-92-dataset-protocol-2026-08-21).

A ten-dataset subset chosen for spread rather than difficulty — Phases 1–5 of [PLAN.md](PLAN.md) —
measures **0.9630**, every dataset reproducing its local accuracy exactly; the two numbers measure
different things and are not the same claim. See
[reference/RESULTS.md](reference/RESULTS.md) for both.

The extension builds on **nine platforms** — Linux (amd64/arm64), macOS (amd64/arm64), Windows
(MSVC and mingw) and all three wasm targets — and is **merged into community-extensions**
([duckdb/community-extensions#2497](https://github.com/duckdb/community-extensions/pull/2497),
merged 2026-08-18). No build required:

```sql
INSTALL rocket FROM community;
LOAD rocket;
```

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
is that **inference is 93.7% of wall clock**, and there is no trained artefact to amortise it
against — a new set of labelled examples pays full price every time.

One narrower thing *can* now be amortised, and it is worth knowing the size of it. Where many calls
share the same labelled context, the context only has to be encoded once
([anofox-tabfm#40](https://github.com/DataZooDE/anofox-tabfm/pull/40), unreleased). Measured through
this pipeline that is **1.85x** on a dataset with nine test chunks per group, 1.13x at two chunks,
and **0.64-0.73x — a net loss — at one**, because a single chunk pays the cold encode and gets no
reuse. Single-chunk is the most common shape in this project's own archive, so on a full sweep it is
worth about 7%. It changes the constant, not the trade: see
[reference/RESULTS.md](reference/RESULTS.md) for the numbers, and
[docs/DISTILLATION_PLAN.md](docs/DISTILLATION_PLAN.md) for the approach that would change the trade
itself.

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

- **The test set is large, on CPU.** Cost is linear in test rows with no amortisation:
  ItalyPowerDemand's 1,029 rows took 1,010 s on 16 vCPU. ECG5000's 4,500 rows never finished on
  CPU at all — four attempts, >5h46m, ~44 GB — and needed a GPU to land in 18m39s (17m04s in the
  92-dataset run). A GPU build is no longer something you have to self-build:
  [this repo publishes one](https://github.com/maxdemarzi/duckdb-rocket/releases/tag/prebuilt)
  (`anofox_tabfm-cuda-*`, linux_x86_64), since none exists upstream.
- **The training context is large.** Memory scales with context rows × features and lives outside
  DuckDB's `memory_limit`, so it cannot be bounded by configuration — only by `--test-chunk`, and
  only for the test half. A 500-row context is already heavy.
- **You will classify repeatedly.** A trained model turns training cost into near-free inference.
  This does the opposite. If you are going to run it a thousand times, train something.
- **You need low latency.** Per-call cost is seconds, not milliseconds.
- **Licences matter.** The model weights are third-party and some are non-commercial; see below.

### Your table is large: what actually helps

**The premise needs correcting first.** `rocket_transform` itself is not what gets slow — it is a
convolution over your series, embarrassingly parallel across rows, and every timing this project
has recorded puts it under 2% of wall clock. That matches the literature, not just our own
numbers: [MiniRocket](https://arxiv.org/abs/2012.08791) transformed and classified all 109 UCR
datasets combined in under ten minutes and was validated up to MosquitoSound's 139,780 training
series with no reported scaling breakdown — ROCKET-family feature extraction is linear in series
count and length, full stop. What dominates is `tabfm_classify`: it is an
in-context model with no trained weights for your task, so **every call re-encodes your entire
labelled training set before it looks at a single query row**. That re-encoding is measured at
71-80% of a call's cost ([docs/ROUTING.md](docs/ROUTING.md)) — a fixed fee, paid again on every
call, whether or not the context changed since the last one. "Large table" almost always means
"many rows to classify," and the fixed fee is what makes that expensive, not the convolution.

That reframes "should I sample?" into two different questions with different answers:

- **Sampling the *test* rows** — the ones you're classifying — doesn't save you anything you
  wanted: you'd just get predictions for fewer rows. What you actually want is to batch them, not
  drop them (below).
- **Sampling the *training* context** — the labelled rows the model conditions on — is possible
  (`--max-train-rows`, stratified by class) but **measured to be a bad trade for speed**: halving
  the context bought 1.48x, not 2x, because the fixed per-call fee doesn't shrink — only the
  per-row term does. That 1.48x cost −0.0168 accuracy, worse than the −0.0033 that reducing the
  group count buys for a 4x speedup (below). Shrink the context only when a run doesn't fit in
  memory otherwise — it's a memory lever, not a speed lever. It is worth doing regardless past a
  point, though: TabPFN v2's own suggested regime tops out at 10,000 rows / 500 features / 10
  classes, enforced as a hard `ValueError` in the reference implementation
  ([PriorLabs/TabPFN#115](https://github.com/PriorLabs/TabPFN/issues/115)), and real-world reports
  past that ceiling describe a cliff, not graceful decay. `anofox_tabfm` doesn't expose that guard,
  so `phase5_pipeline.py` does: it warns (not errors — the ONNX path might still be fine) when your
  training context exceeds 10,000 rows, pointing at `--max-train-rows`. **Don't bother building
  anything smarter than stratified random
  sampling to pick which rows survive the cap**: a dedicated study
  ([arXiv:2607.26628](https://arxiv.org/abs/2607.26628)) tested K-Means and farthest-point
  selection against plain random sampling for TabPFN context and found them statistically tied —
  what predicts accuracy is the context's diversity/coverage, not how cleverly it's chosen, and
  forcing a context to match the full training distribution's feature means measured *worse* than
  random. Cheap random sampling already covers the space in expectation. If you're capping a
  context from a genuinely huge table with no measurement of your own to go on yet, TabICL's own
  practitioner guidance for million-row tables — subsample the context to somewhere in **500-5,000
  rows** ([arXiv:2502.05564](https://arxiv.org/pdf/2502.05564)) — is a reasonable starting point,
  well inside TabPFN's 10,000-row ceiling.

In the order they're worth trying, measured on 28-29 hard UCR datasets
([docs/ROUTING.md](docs/ROUTING.md), [reference/RESULTS.md](reference/RESULTS.md)):

1. **Reduce `--n-groups`.** Cost is exactly linear in the group count, and G=10 against the
   paper's G=40 measured **3.7-3.8x faster** for −0.0033 mean accuracy (not statistically
   different from zero). This is the single biggest lever with the smallest accuracy cost, and
   it's why `--n-groups 10` is the shipped default rather than 40.
2. **Batch calls; don't shrink them.** The context-encoding fee is per *call*, not per row, so one
   big `--test-chunk` amortises it and many small ones re-pay it every time — chunking finer than
   necessary measured **2.18x slower** on whole-dataset runs. Set `--test-chunk` to the largest
   batch your memory allows (see "It fits badly when" above for the memory math), not the smallest
   that feels safe. TabPFN's own guidance for large test sets converges on the same idea from the
   opposite direction — chunk test inference into batches on the order of 1,000 rows rather than
   one call per row — which is a floor to start from, not a ceiling to stay under.
3. **Route: run a cheap model on everything, escalate only what it's unsure of.** This is the
   lever aimed specifically at "many rows." Train a ROCKET-features-plus-ridge (or
   `MultiRocketHydraClassifier`) student — milliseconds per row — and send only its
   least-confident rows (by decision margin, not top score) to the teacher. At a 20% escalation
   budget: **+0.0200** over the student alone (ridge) for **26%** of the teacher-on-everything
   cost. The student's own uncertainty picks better-than-random rows to escalate — about 3x the
   signal of escalating a random 20% (p≈0.01), which is the actual claim, not just "escalating
   helps." Full method, the confidence-margin code, and the "why this works but distillation
   doesn't" analysis: [docs/ROUTING.md](docs/ROUTING.md). Tooling:
   `scripts/distill_gate.py --route` and `scripts/route_serve.py`.
4. **Distillation — replacing the teacher entirely — is the one that sounds obvious and measured
   negative.** Label an unlabelled pool once with the teacher, train a cheap student on those
   pseudo-labels, then never run the teacher again. Gated on 67 datasets and it does not clear the
   bar: on datasets with real headroom the teacher's own soft labels recovered only 25% of what
   real labels would have bought (+0.0119 against a +0.0474 ceiling, p=0.13), and its hard labels
   recovered 2%. **The reason isn't the teacher's error rate** — it tolerates 25.6% label noise
   and the teacher's error rate is 21.6%, comfortably inside that — it's that a teacher's mistakes
   aren't noise: they land on the same ambiguous rows every time, so a student trained on them
   learns a coherent wrong rule instead of one that averages out. Swapping the same error rate for
   *random* errors is worth +0.0516. This is why routing (above) wins where distillation loses:
   routing never asks a wrong-but-confident prediction to be trusted, it just costs one escalated
   row. Full gate, the label-noise sweep that found this, and what would make it worth revisiting:
   [docs/DISTILLATION_PLAN.md](docs/DISTILLATION_PLAN.md).

**One more idea, checked and left unimplemented on purpose.** "Bag the context" — run several
calls with different random subsamples of a large training set and average, recovering some of
what any one small context misses. It's a real, named pattern elsewhere (ConTextTab uses 8-fold
context bagging at inference, [arXiv:2506.10707](https://arxiv.org/pdf/2506.10707)), but nothing
found ties a number to how much it recovers versus just picking one larger context, and the cost
is linear in the number of resamples — on top of a cost this pipeline already spends most of its
time on. Not implemented here; flagged rather than built, per this project's own rule of not
shipping a lever before it's measured.

**What none of this fixes.** The context-encoding cost is architectural — `tabfm_classify` takes
train and test in one call with no prepare-then-query split to cache between, so there's nothing
to expose from SQL. An unreleased upstream context cache
([anofox-tabfm#40](https://github.com/DataZooDE/anofox-tabfm/pull/40)) helps only across chunks
that already share a context (1.85x at nine chunks, a net loss at one — see `--context-cache`
above), and doesn't touch the per-call fee itself. If your table is large enough that none of the
above gets you to an acceptable cost, this pipeline is the wrong tool for serving it — train
something, per "It fits badly when" above.

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

**The thing ridge cannot do is train without labels**, which suggests composing them: label an
unlabelled pool with this pipeline, then distil into a cheap student. That was designed, gated and
**measured — and the gate failed**. Even handing a student *real* labels for the pool buys under
one point over training on the context alone (ECG5000: +0.0027 with 2,250 extra labelled rows), so
pseudo-labels have nothing to recover. These datasets are near ceiling and a ROCKET-family
classifier already extracts what is there. Full numbers and what the result does *not* say in
[docs/DISTILLATION_PLAN.md](docs/DISTILLATION_PLAN.md); `scripts/distill_gate.py` reproduces it in
about seven minutes.

**A trained deep model** (InceptionTime, ROCKET+ridge at scale) will beat this on accuracy-per-
dollar whenever you have enough labels and enough runs to amortise training. Nothing here competes
with that.

**Accuracy, stated plainly.** Mean **0.9630** over the ten-dataset subset, ranging from 1.0000 on
four of them to **0.7667** on Beef. That is a subset chosen for spread rather than difficulty, and
it is *not* comparable to the paper's 0.900 over 92 datasets — different datasets, different
backbone, one split instead of 30 resamples. Treat it as evidence the pipeline works, not as a
benchmark result.

That last gap is now measured rather than merely admitted. A 160-run pilot put split luck at **8x**
the between-dataset effect (sd 0.0173 against 0.0061), which means a proper protocol here is
**40 datasets x 4 resamples — 320 runs**, not the 1,440 that copying 24 x 30 would cost. It also
found the single split pointing the wrong way on the group-count comparison: +0.0025 measured
against −0.0033 archived, both indistinguishable from zero. `--resample` exists; the campaign has
not been run.

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
uv run pytest                              # 320 tests, no model weights needed
uv run python scripts/doctor.py            # record the environment tuple
uv run python scripts/emit_golden.py       # regenerate conformance fixtures
uv run python scripts/probe_anofox.py      # re-run the Phase 2 probes
uv run python scripts/accuracy.py --smoke  # oracle accuracy harness
```

The pipeline harness takes three flags worth knowing about:

```bash
# A different train/test split. 0 (the default) is the archive's own, so every archived
# result reproduces; 1..N are stratified re-splits at the same per-class sizes. NOT the
# same as --seed, which varies the kernel bank while the split stays put.
uv run python scripts/phase5_pipeline.py --dataset GunPoint --resample 3

# Encode the labelled context once per support set instead of once per classify call.
# Needs an unreleased anofox_tabfm build and a model directory carrying the split graph
# pair, and it refuses to start without them rather than running uncached in silence.
# Pays off only across --test-chunk chunks: 1.85x at nine chunks per group, a LOSS at one.
uv run python scripts/phase5_pipeline.py --dataset OSULeaf --test-chunk 128 \
    --anofox-extension PATH --register-model-dir DIR --context-cache

# Where raw.parquet and predictions.json go. Defaults to data/phase5/<dataset>, which is
# shared by every run of that dataset -- any driver running jobs in parallel must give
# each one its own, or concurrent runs overwrite each other's predictions.
uv run python scripts/phase5_pipeline.py --dataset Beef --workdir data/scratch/beef
```

The DuckDB CLI under `tools/` and the `duckdb` submodule are both pinned to **v1.5.5**
(`d8cdaa33`) because extension ABI is version-bound.

### Timings come from pods, not this workstation

Accuracy reproduces locally; wall clock does not. Local Windows timings mislead because WDDM spills
to host memory instead of failing, so a run that would be OOM-killed in a container merely gets
slow. Every timing in [reference/RESULTS.md](reference/RESULTS.md) came off a rented pod, and
[docs/POD.md](docs/POD.md) is how — the launchers, the drivers, and the several ways a container
lies about how large it is.

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
