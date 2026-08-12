# duckdb-rocket — Implementation Plan

Bring [RocketPFN](https://arxiv.org/abs/2606.21786) (training-free time-series classification)
into DuckDB by building the **feature-extraction half** as a focused extension and composing
it with [`anofox_tabfm`](https://github.com/DataZooDE/anofox-tabfm), which already ships
`tabpfn-v2-5` **and** `tabicl-v2`.

## The target composition

```
series ──▶ rocket_transform()  ──▶ tabfm_classify()  ──▶ average probs ──▶ label
           (this project)          (anofox_tabfm)        (plain SQL)
```

Per the paper: 10,000 kernels split into **G=10 groups of 1,000**. Each kernel yields 2
features (global max + PPV), so each group is 2,000 features — exactly TabPFN v2.5's cap.
Each group is classified independently and the **class probabilities are averaged**.

> **Phase 2 corrected this.** 2,000 is TabPFN v2.5's *input* ceiling, not the width a single
> estimator sees — that is **500** (`max_features_per_estimator`). A 2,000-feature group is
> only fully covered at e≥4, and `anofox_tabfm` caps estimators at 1. **This project therefore
> runs G=40 groups of 250 kernels (500 features each)**, which keeps the paper's 10,000 kernels
> and its average-across-groups ensembling while making e=1 honest and the SQL path
> reproducible. See `reference/PHASE2_FINDINGS.md`.

**Design consequence:** the pipeline needs probability output, not labels. Verifying that
`anofox_tabfm` can provide it is Phase 2, and it gates everything after it.

## Reference constraints (from the paper)

| Item | Value |
|---|---|
| Feature extractor | Rocket (primary); MiniRocket / MultiRocket also evaluated |
| Kernels | 10,000 total = G=10 groups × 1,000 |
| Features | 2 per kernel (global max, PPV) → 2,000 per group |
| Classifier | TabPFN v2.5, e=8 internal estimators — **not reachable via anofox-tabfm**, see Phase 2 finding 3 |
| TabPFN v2.5 limits | 50,000 samples, 2,000 features, ≤10 classes |
| Ensembling | Average probability estimates across the G groups |
| Multivariate | Each kernel gets a random subset of K channels, independent weights per channel; still 2 features per kernel |
| Reported result | 0.900 mean accuracy on 92 UCR datasets (30-resample), median ~30s/fold |

---

## Prior art in this workspace — read before starting

Two local repos already solved problems this project would otherwise rediscover.

### `C:\Users\maxde\Repositories\tabicl` (fork of soda-inria/tabicl, branch `scaling-upgrades`)

A deep TabICL v2 codebase with `src/tabicl/scaling/` and a measurement discipline documented
across `STATUS.md` / `DESIGN.md` / `PERFORMANCE.md`. Four findings bear directly on us:

1. **AMP costs 7.3 AUC.** `use_amp=True` is the default and scored 73.61 against CPU's 80.93
   on rel-event; `use_amp=False` reproduces CPU exactly. It is *harmless where the model is
   confident and expensive where it is not* — exactly backwards from where you want
   precision. **Check TabPFN v2.5's AMP default before recording a single accuracy number.**
   This is the highest-value thing carried over, and it cost that project a re-measurement of
   every GPU result in its file. — **Done:** see "The AMP default, resolved" in Phase 1. The
   default is device-dependent and on by default on any CUDA device.
2. **Pair everything; know your noise floor.** Pairing (same seed, same data, one setting
   changed) tightened their estimate ~8×, from a ±5-point resolution to ±0.6. We need our own
   floor for UCR accuracy before any comparison is meaningful.
3. **No lever goes in as a recommendation on one task.** Three separate times a single-dataset
   result failed to survive a second dataset. Our 10-dataset subset exists for this reason.
4. **Width sensitivity is a property of the backbone's prior.** On TabICL v2, adding 150
   columns cost −3.36, and replacing raw columns with PCA components *at equal width* lost
   3–6 points on 10 of 10 arm-task pairs. The stated reason: "a principal component is not a
   column — it has no stable identity to attend over across rows, no marginal distribution
   the pretraining prior recognises, no semantics." **ROCKET features are exactly that kind
   of object.** See the optional Phase 3b experiment below.

Also relevant: `python -m tabicl.scaling.build_native` builds a C++17 pybind11 extension on
Windows, and its notes record that **MSVC Build Tools are required — clang alone cannot link
a CPython extension** (no Windows SDK). Precedent for our Phase 4 build.

### `C:\Users\maxde\Repositories\swan` — a shipped DuckDB extension that already uses `anofox_tabfm`

The single most relevant repo in the workspace, and the reason Phase 2 is mostly confirmation
rather than discovery. swan is a semantic-modeling/query-optimization DuckDB extension whose
predictive reasoner drives `tabicl-v2` through `anofox_tabfm` in production. It supplies:

- **The template decision** for Phase 4 — it is built from the C++ `extension-template`.
- **Phase 2's answers**, including the probability gate. See "What swan already established".
- `scripts/vendor_anofox_tabfm.sh` — pinned-tag vendoring, reusable as-is.
- `scripts/sql_only_feature_probe.py` — prior art for Phase 4's pure-SQL macro step.
- `scripts/windows_path_repro.ps1` / `windows_simd_repro.ps1` — Windows problems already hit once.
- `docs/dev/TABICL_LIMITATIONS_ROADMAP.md` — maps the TabICLv2 paper's stated limitations onto
  a real integration. Relevant to Phase 3b, and a good model for how to write up our own.

Its documentation is unusually candid about bugs that were **only** found by running against
live weights — collapse-to-mean, target leakage through a rowid, an unsorted quantile head.
Treat that as a standing warning: this SQL surface does not fail loudly.

### `C:\Users\maxde\black_swan` — the RunPod lane

`docs/RUNPOD.md` + `scripts/cloud/runpod_launch.py` (stdlib only, so it runs on a pod
diagnosing itself). `check` / `gpus` / `plan` are read-only; `create` requires
`--yes-i-will-pay`. `tabicl/scaling/pod_runner.py` is the other harness — it gates hosts on
CUDA-from-Python *and* a 5 MB/s download floor before scheduling work, after one host with a
healthy `nvidia-smi` and 270 kB/s cost a whole session.

**Reuse the pattern rather than writing a third one.** Standing rules from those docs:

- **The account is shared** with several RelationalAI people; pods belonging to others show
  up in `check`. Confirm a pod is yours before stopping it.
- A forgotten L40S is **$0.79/hr → $19/day → $570/month**. `stop` keeps billing the volume.
- Archive the environment tuple `(GPU, VRAM, compute capability, glibc, library versions)`
  alongside every result — `doctor.json` exists because a number without it is not attributable.
- Never mix GPU types inside one measured comparison; re-run the control on the treatment's card.

**Local RTX 3060 (12 GB) is for smoke tests and correctness only. Every number that goes in a
table comes from a pod.** Note the 3060 is also on Windows/WDDM, which *silently spills to
host RAM instead of raising OOM* — so a local memory problem shows up as a 6× slowdown rather
than an error, and local timings are not trustworthy even directionally.

---

## Phase 0 — Toolchain and skeleton

**Goal:** a working build/test environment. No project logic yet.

- [x] CMake 4.4.2 and Ninja installed (winget)
- [x] DuckDB CLI v1.5.5 → `tools/duckdb.exe` (pinned; extension ABI is version-bound)
- [x] Confirmed: VS Build Tools 2022 17.14 present (needs a vcvars64 shell for `cl`)
- [x] Local GPU: RTX 3060 12 GB — smoke tests only
- [x] `uv` project + light deps: `numpy`, `scikit-learn`, `pyarrow`, `duckdb`, `aeon`
- [x] `torch` 2.13.0+**cpu** and `tabpfn` 8.2.0 (local install is for smoke tests only, so the
      CPU wheel is deliberate — it also sidesteps the WDDM spill trap below)
- [x] `git init`; MIT license (matches anofox_tabfm, keeps the door open for upstreaming);
      public repo at https://github.com/maxdemarzi/duckdb-rocket
- [ ] Port the RunPod launcher pattern from `black_swan/scripts/cloud/runpod_launch.py`
- [ ] A `doctor.py` equivalent recording the environment tuple for every run

**Exit:** `cmake`, `ninja`, `duckdb --version` all work; `uv run python -c "import tabpfn"` succeeds.

---

## Phase 1 — Python reference implementation (the oracle)

**Goal:** a faithful, seeded RocketPFN in Python. Everything downstream is validated against
this, so correctness here matters more than speed.

- [ ] Implement ROCKET kernel generation and the max/PPV transform
- [ ] **Use a portable, explicitly-specified PRNG (splitmix64 or PCG32) — not `np.random`.**
      The single most important decision in the phase. The C++ extension must reproduce
      identical kernels from the same seed, and replicating NumPy's stream in C++ is painful.
      Define the stream ourselves, in a written spec, from day one.
- [ ] Implement the G-group split and probability averaging over TabPFN v2.5
- [ ] **Pass `inference_precision=torch.float32` to every `TabPFNClassifier` used for an
      accuracy run.** Never rely on the default. See "The AMP default, resolved" below for why
      the obvious CPU-vs-GPU sanity check does *not* catch this.
- [ ] Run on a **10-dataset UCR subset** (mix of short/long series, 2-class and multi-class,
      one multivariate)
- [ ] **Establish the noise floor**: same config, N seeds, report sd of paired gaps and of
      absolute accuracy. Every later comparison is read against this number.
- [ ] Record per-dataset accuracy → `reference/accuracy.json`
- [ ] Emit **golden vectors**: fixed seed + fixed input series → exact feature matrix in
      `reference/golden/*.parquet`. This is the C++ conformance test.
- [ ] Write `SPEC.md` documenting kernel generation (lengths, dilations, padding, biases,
      weight normalization) precisely enough to reimplement from text alone

### ~~BLOCKED~~ CLEARED: TabPFN v2.5 weights are license-gated

> **Cleared 2026-08-11.** The licence was accepted and an API key saved to `token.txt`
> (gitignored). `tabpfn` cached it to `~/.cache/tabpfn/auth_token`, so local runs no longer
> need `TABPFN_TOKEN` in the environment — **but pods still do**, and the key must never be
> committed. Note `anofox_tabfm` is a *separate* gate with its own licence flag
> (`SET anofox_tabfm_accept_hf_license = true`), as predicted.

**Nothing in Phase 1 that needs model weights can run until this is cleared, and it needs a
human.** `tabpfn` 8.2.0 will not download v2.5 weights without an accepted Prior Labs licence.
The first attempt opens a browser and waits for an interactive login, which fails outright in a
non-interactive shell (`WinError 10038` — a socket the harness cannot service).

To clear it, once:

1. Register / log in at <https://ux.priorlabs.ai/account>
2. Accept the licence at <https://ux.priorlabs.ai/account/licenses>
3. Copy the API key

Then make it available non-interactively. `tabpfn` resolves a token in this order
(`browser_auth.py:76-95`):

1. the **`TABPFN_TOKEN`** environment variable
2. `~/.cache/tabpfn/auth_token`
3. `~/.tabpfn/token` (tabpfn-client's own cache)

**This is not only a local-setup annoyance — it is a pod problem and a reproducibility
problem.** Every pod needs the token injected, so it belongs in the RunPod launcher's
environment alongside the rest of the run configuration, and it must never be committed. It is
also worth stating in the README that reproducing this project's numbers requires accepting a
third-party licence, since that is a real constraint on anyone else running it.

*Unaffected and already verified:* dataset loading via `aeon` (the smoke run reached the model
step, so the UCR download path works), and everything in Phases 3–4 that touches DuckDB rather
than PyTorch. `anofox_tabfm` ships its own ONNX weights via `tabfm_download` and is a separate
gate — do not assume this token clears that one too.

### The AMP default, resolved (tabpfn 8.2.0)

Answered by reading the installed package, so the tabicl lesson does not have to be relearned
at the cost of re-measuring every result.

`TabPFNClassifier(inference_precision=...)` defaults to `"auto"`, which resolves **per device**:

| Device | `auto` resolves to | Precision |
|---|---|---|
| CUDA | autocast **on** — `is_autocast_available` is true for any CUDA device | **fp16** |
| CPU | autocast on **iff** `_cpu_supports_fast_bf16()` — Intel AMX / AVX512-BF16, AMD Zen 4+ | bf16 |

Two consequences:

1. **GPU autocast is fp16, not bf16.** Narrower range than the bf16 that cost tabicl 7.3 AUC,
   so treat that figure as a floor on the risk rather than an estimate of it.
2. **The CPU baseline is not automatically trustworthy.** This plan previously proposed
   verifying that CPU and GPU agree before trusting GPU numbers. That check is void on a
   Zen 4 / Sapphire Rapids pod, where the *CPU* run is bf16 autocast too — both sides are
   then reduced precision and can be wrong together, in agreement. **CPU≡GPU agreement is
   evidence only when both sides are pinned to fp32.**

Passing a `torch.dtype` takes a different branch entirely (`base.py:267`), setting
`use_autocast_ = False` and forcing the dtype regardless of device. That is the lever.

Local dev box is incidentally safe: Comet Lake (Family 6 Model 165) has no AVX512-BF16, so
`_cpu_supports_fast_bf16()` is `False` and CPU runs here are genuine fp32. **Do not generalize
that to pods** — record `_cpu_supports_fast_bf16()` in `doctor.py`'s environment tuple.

**Exit:** accuracy in the neighborhood of the paper's per-dataset numbers, a measured noise
floor, and golden vectors on disk.

---

## Phase 2 — Probe `anofox_tabfm` (GATE — largely pre-answered by swan)

**Goal:** find out whether the intended composition is possible at all. Cheapest phase,
highest information value. **Run it concurrently with Phase 1.**

> **The gate is already open.** `maxdemarzi/swan` ships a production `anofox_tabfm` integration
> (`docs/dev/PREDICTIVE_TABICL.md`, `python/pyrel_duckdb/reasoners/predictive/predictive_tabicl.py`).
> Its findings are recorded in "What swan already established" below and turn most of this
> phase from discovery into confirmation. **Read that section before running anything.** Its
> bugs were found empirically against live weights, not by reading — several are invisible to
> static inspection and expensive to rediscover.

> **Phase 2 is done.** Findings, with reproduction, in **`reference/PHASE2_FINDINGS.md`**;
> raw probe output in `reference/anofox_probe_*.json`; the harness is
> `scripts/probe_anofox.py`. Probed against **`anofox_tabfm bc6d8af`** / DuckDB v1.5.5. The
> headline: the composition works, `tabpfn-v2-5` does not load at all in this build, and the
> 500-feature-per-estimator ceiling forces G=40 rather than G=10.

- [x] `INSTALL anofox_tabfm FROM community; LOAD anofox_tabfm;` then `tabfm_download(...)` —
      note the real signature is `tabfm_download('classification', model := '...')`, and
      non-commercial weights need `SET anofox_tabfm_accept_hf_license = true`
- [x] **`tabpfn-v2-5` is unusable in `bc6d8af`** — its published checkpoint no longer matches
      anofox's bundled ONNX graph, and re-downloading (which the error advises) cannot fix it.
      `tabicl-v2` works, which promotes optional Phase 3b to the default path
- [x] ~~Does `tabfm_classify()` return class probabilities or only a label?~~ **Yes — `proba`,
      a per-class map.** Confirm it holds for `tabpfn-v2-5` as well as swan's `tabicl-v2`
- [x] ~~Confirm the train/test convention~~ **Explicit `test := <view>`; single-table mode is
      unsafe.** See finding 4
- [x] ~~Can it accept 2,000 feature columns?~~ **Yes.** The advertised 500 limit is a
      configurable guard: `SET anofox_tabfm_max_features = 4000`. The contemplated upstream PR
      is unnecessary
- [x] ~~Does it accept a `LIST`/`ARRAY`-valued column?~~ **No, and it crashes** —
      `INTERNAL Error: Run() called with null input buffers`. An upstream bug report
- [x] ~~Does it expose an AMP / precision setting?~~ **No.** Valid options are exactly
      `task, n_estimators, seed, output_mode, context_rows, softmax_temperature, model`. The
      exported graph's precision is unknown *and unsettable* — recorded as a result
- [x] **The `e=8` question is settled by arithmetic, not preference** — one estimator sees 500
      features, so at 500-feature groups e=1 is not a compromise but the correct setting. G=40
      replaces G=10; no custom ensembling layer needed
- [x] ~~Row identity across G groups~~ **Deterministic ordering works** — output contains only
      test rows, in the test view's own order, stably across calls. swan's rowid hack is not
      needed. **Phase 3 must assert this rather than assume it** (verified at 40 rows only)
- [ ] Measure: latency at realistic UCR row counts (40 s for one 2,000-column call on 100 rows
      is already a Phase 5 warning; the curve is worse than linear in width)

### What swan already established

Sourced from `docs/dev/PREDICTIVE_TABICL.md` and the call site at
`predictive_tabicl.py:388-400`. swan drives `tabicl-v2`; we drive `tabpfn-v2-5` through the
same SQL surface, so confirm each of these still holds for our model rather than assuming.

**1. `tabfm_classify` returns `proba`. The gate is GO.** The literal shape:

```sql
SELECT "__tabicl_rowid", yhat, yhat_score, proba
FROM tabfm_classify('<train_view>', '<target>', test := '<test_view>',
                    model := '<model>', features := [...], opts := {...})
```

`proba` arrives as a **map keyed by class label** (consumed as a Python dict,
`predictive_tabicl.py:729`). That is better than a bare array for us — averaging across the G
groups can key on the class rather than trusting positional alignment.

**2. Never average `yhat_score`.** It is confidence in *whichever* class was predicted, not
P(class). swan hit a real bug where accuracy measured 1.0 while AUROC came out **below chance**,
because a confidently-negative row carries a high `yhat_score`. Average `proba`, always.

**3. `opts['n_estimators'] > 1` hard-throws `NotImplementedException`** — gated on anofox's own
roadmap milestone M3, no ETA. **This directly contradicts our reference constraints table**,
which specifies TabPFN v2.5 with `e=8` internal estimators. We cannot reproduce the paper's
configuration through anofox-tabfm today. Options: accept `e=1` and expect to land below the
paper's 0.900, or rebuild ensembling as our own orchestration layer the way swan's
`predict_ensemble()` does. **Decide this explicitly and record it next to any accuracy number** —
an unexplained gap against the paper is otherwise attributable to our ROCKET implementation
when the real cause is a missing classifier setting.

**4. Use explicit `test := <view>`, never the single-table `target IS NULL` convention.**
swan confirmed empirically that `tabfm_regress`'s single-table mode **silently collapses to
predicting the training mean for every row**. Classification measured correctly in isolation,
but swan routes both through the explicit-view path rather than trusting one to behave
differently. This answers the plan's train/test-convention question.

**5. There is no passthrough/id column.** `tabfm_classify` echoes back the target plus exactly
what is named in `features := [...]`; a PK not listed there is **silently dropped**. This is a
hard problem for us specifically: averaging across G=10 groups means joining rows across ten
separate classify calls, so we need stable row identity. swan's workaround is to inject a rowid
*as a model feature*, and it took two empirically-found bugs to get right:

| Attempt | Failure |
|---|---|
| `hash(pk)` | High-entropy feature; model collapsed to predicting the training mean. A 60-row perfectly-linear signal went from ~0.3 error to ~50 |
| `ROW_NUMBER() OVER (ORDER BY pk)` | Smooth and bounded, but leaks target-correlated signal whenever PK assignment correlates with the target |
| **`ROW_NUMBER() OVER (ORDER BY hash(pk))`** | **Shipped.** Smooth value, hash-scrambled ordering |

Compute it **once**, in the base view that train/test views derive from — never recomputed
inside the derived views, or the same PK maps to different rowids across them.

*Our exposure is lower but not zero:* swan notes the collapse-to-mean bug was invisible in its
own flatten-compiler tests because many real features dilute one noisy one, and we will have
2,000. But we also have an alternative swan lacked — the group index is already a
`rocket_transform()` argument, so a deterministic row ordering may be reconstructable without
injecting anything. **Prefer that if it works; fall back to swan's rowid if it doesn't.**

**6. Expect 2,000 scalar columns, not one `LIST` column.** swan always passes N scalar names in
`features := [...]`. Separately, its own flatten compiler *prunes* `DOUBLE[]`/`FLOAT[]` columns
entirely rather than feeding them to the model. That is swan's layer, not anofox's, so it is not
proof — but it is the only evidence available, and it points away from the list-valued
convenience this plan hoped for. **Keep the plan's original question live and test it directly.**

**7. ONNX Runtime ABI hazard.** Any process that resolves one ONNX Runtime build is permanently
bound to it (`Ort::InitApi()` is process-global). Loading anofox-tabfm's debug build afterward
fails with a raw dynamic-linker symbol-version error, after which every entry point reports
"extension not available." Vendoring anofox in **release** mode (statically-linked ONNX Runtime)
is the real fix. We have no ONNX Runtime of our own, so this bites us only if we load swan and
anofox in one process — but the failure mode is opaque enough to be worth recognizing on sight.

**Pin the version.** `INSTALL ... FROM community` is fine for the initial probe, but swan's
`scripts/vendor_anofox_tabfm.sh` records the reason not to leave it there: `anofox-tabfm` is
**pre-1.0 and tags near-daily**, so an unpinned dependency means Phase 3's accuracy numbers are
not reproducible next week. swan builds it from a pinned tag (`ANOFOX_TABFM_TAG`, at
`v2026.07.17`) as an *independent* artifact rather than a CMake source-level dependency —
deliberately decoupled from its own vendored DuckDB version. Record whatever tag we probe
against alongside the findings note, and reuse that script rather than writing a second one.

**Exit:** a written findings note, plus GitHub issues filed on `anofox-tabfm` for any gaps.

**Branch on the outcome:** the probability branch is settled — `proba` exists, so proceed to
Phase 3 as designed and drop the majority-vote fallback. The live branch is now the column
convention: *if 2,000 names in `features := [...]` proves unworkable and no list-valued form
exists*, the upstream PR moves from "nice to have" to a prerequisite. Consider offering to write
it. A second upstream candidate is `n_estimators` (their milestone M3) — but that is their
roadmap already, so ask before building.

---

## Phase 3 — SQL composition prototype

**Goal:** prove the DuckDB half end-to-end while ROCKET still lives in Python. Isolates
composition risk from C++ risk.

- [ ] Compute ROCKET features in Python (Phase 1 code), write to Parquet
- [ ] Load into DuckDB; run G classify calls; average probabilities in SQL; argmax
- [ ] **Reproduce Phase 1 accuracy exactly** — same features in, same predictions out
- [ ] Record wall-clock for the DuckDB-side classification

**Exit:** one `.sql` script from feature Parquet to predictions, matching the oracle.

**This is the real milestone.** The idea is proven and everything after is performance
engineering. It is also a reasonable place to stop and publish if Phase 4 stops looking worth it.

### Phase 3b (optional) — TabICL v2 as the backbone

Not required for the extension, and genuinely novel. `anofox_tabfm` ships `tabicl-v2`
alongside `tabpfn-v2-5`, so swapping the backbone is a one-argument change once Phase 3 works.

**The prediction, from the tabicl fork's own measurements:** TabICL v2 should do *relatively
badly* here. Its prior is width-sensitive and rotation-hostile — PCA components at equal width
lost 3–6 points on 10 of 10 pairs — and ROCKET features are synthetic projections with no
stable per-column identity, no recognisable marginal, and no semantics. Yet TabICL v2 beats
TabPFN on tabular benchmarks (TALENT average rank 2.12 vs TabPFN v2.6's 2.74).

So the two facts point opposite ways, which is what makes it worth running: it isolates
whether "width sensitivity" is about *column count* or about *column meaningfulness*. A
2,000-feature ROCKET block is the cleanest available test, and the fork is the only place with
the prior measurements to interpret the result against. Cheap — one argument, same harness.

---

## Phase 4 — ROCKET inside DuckDB

**Goal:** replace the Python feature step.

- [ ] **First: a pure-SQL macro** over DuckDB `LIST` operations. Expect it to be too slow —
      but it establishes the semantics in-database and gives a zero-build fallback. If it is
      merely 5–10× slow rather than 1000×, seriously consider stopping here.
- [ ] Otherwise scaffold from a template — **which one is a real decision; see below**
- [ ] Implement `rocket_transform(series, kernels, seed, group)` → `FLOAT[]`
- [ ] **Conformance test against Phase 1 golden vectors** (tight float tolerance)
- [ ] Multivariate support (random channel subsets per kernel)
- [ ] Variable-length series handling
- [ ] Parallelize across kernels/rows using DuckDB's execution model
- [ ] Benchmark vs. the Python implementation

**Build note:** MSVC Build Tools required on Windows; see `tabicl/scaling/build_native.py` for
a working precedent.

### Template: the C++ one, following `maxdemarzi/swan`

DuckDB ships three official templates — C++
([`extension-template`](https://github.com/duckdb/extension-template)), C API
([`extension-template-c`](https://github.com/duckdb/extension-template-c), experimental, needs
no DuckDB build), and Rust
([`extension-template-rs`](https://github.com/duckdb/extension-template-rs), experimental).

**Use the C++ template**, because [`maxdemarzi/swan`](https://github.com/maxdemarzi/swan)
already does and it is a working DuckDB extension on this exact Windows/MSVC toolchain. A
proven local build path outweighs the C API's theoretical advantages here: on paper the C API
looks attractive for a one-scalar-function extension (no engine build, stable ABI), but that
argument is worth less than a repo on this machine that already compiles.

Copy swan's layout:

```
duckdb/                  submodule, pinned to a release branch
extension-ci-tools/      submodule, branch main
extension_config.cmake   duckdb_extension_load(rocket SOURCE_DIR ...)
vcpkg.json               dependency manifest
Makefile                 include extension-ci-tools/makefiles/duckdb_extension.Makefile
src/  test/sql/  cmake/
```

```bash
# "Use this template" on GitHub, then:
git clone --recurse-submodules https://github.com/maxdemarzi/<repo>.git
python3 ./scripts/bootstrap-template.py rocket
```

Two places we should **diverge** from swan:

- **Pin `duckdb` to v1.5.5, not v1.5.4.** swan's `.gitmodules` tracks `branch = v1.5.4`; our
  `tools/duckdb.exe` is v1.5.5. Extension ABI is version-bound, so a mismatch here produces a
  load failure, not a warning. Pick one and make both match.
- **`vcpkg.json` should be near-empty.** swan pulls `openssl` and `highs`, and its
  `extension_config.cmake` carries a lot of hard-won ONNX Runtime and Emscripten
  platform-detection logic. **None of that applies to us** — `rocket_transform` is pure
  arithmetic with no external dependencies. Copy the skeleton, not the payload; every vcpkg
  dependency we don't take is a Windows build failure we don't get.

Also worth reading before Phase 4 starts: `scripts/windows_path_repro.ps1` and
`scripts/windows_simd_repro.ps1` in swan are Windows-specific reproductions of problems already
hit once, and `scripts/sql_only_feature_probe.py` is prior art for this phase's pure-SQL-macro
step.
**Exit:** `rocket_transform()` matches golden vectors and beats Python on throughput.

---

## Phase 5 — Full pipeline and benchmark

- [ ] Whole pipeline in SQL: raw series table → predictions, no Python anywhere
- [ ] Run the 10-dataset subset on a pod; accuracy must match Phase 1
- [ ] Expand toward the paper's 92-dataset / 30-resample protocol if timing permits
- [ ] Compare wall-clock against the paper's ~30s/fold median
- [ ] Every result archived with its environment tuple

---

## Phase 6 — Upstream and release

- [ ] Open the `anofox-tabfm` PRs identified in Phase 2 (list-valued features; probability
      output if missing). **Open an issue describing the use case before writing a large PR** —
      DataZoo GmbH has no CONTRIBUTING guide and `anofox` is a commercial product, so confirm
      appetite first.
- [ ] Package `rocket` for the DuckDB community-extensions repo (metadata, CI matrix, docs)
- [ ] README with the composition example front and center

---

## Standing risks

| Risk | Phase | Mitigation |
|---|---|---|
| ~~`tabfm_classify` won't return probabilities~~ | 2 | **Retired** — swan confirms `proba` exists |
| ~~`e=8` unreachable~~ | 2, 5 | **Retired** — at 500-feature groups one estimator sees everything, so e=1 is correct rather than a compromise. G=40 |
| ~~Row identity lost across the G classify calls~~ | 2, 3 | **Retired, with a caveat** — output is test-rows-only in test-view order, stable across calls. Verified at 40 rows; Phase 3 asserts it |
| **`tabpfn-v2-5` does not load in `anofox_tabfm bc6d8af`** | 2, 3, 5 | Checkpoint/graph mismatch upstream. Use `tabicl-v2`; Phase 3b becomes the default path, not an experiment |
| **A requested `n_estimators` is silently raised** | 1, 3 | TabPFN auto-scales e to cover 500-feature-wide estimators. `auto_scale_n_estimators=False` is pinned; harness records `covers_all_features` and `anofox_reachable` |
| No precision control on the ONNX path | 2, 5 | No such option exists; the graph's precision is unknown and unsettable. Local fp32 TabPFN and DuckDB `tabicl-v2` are **not** precision-comparable |
| Averaging `yhat_score` instead of `proba` | 3 | Silently produces accuracy 1.0 with sub-chance AUROC; swan hit this for real |
| **AMP silently corrupts accuracy numbers** | 1 | Resolved: default is `"auto"` → fp16 on any CUDA device. Force `inference_precision=torch.float32` everywhere; CPU≡GPU checks count only with both pinned to fp32 |
| Kernel-generation spec mismatch Python↔C++ | 1→4 | Portable PRNG spec + golden vectors, decided in Phase 1 |
| 2,000-column SQL calling convention unusable | 2 | Upstream list-valued features PR |
| TabPFN inference dominates runtime, making C++ ROCKET pointless | 3 | Phase 3 measures this *before* any C++ is written |
| Conclusions drawn from one dataset | 1, 5 | 10-dataset subset; noise floor measured first |
| Local Windows timings mislead (WDDM spills instead of OOM) | all | Numbers come from pods, not the 3060 |
| **TabPFN v2.5 weights need an accepted Prior Labs licence** | 1, 5 | One-time browser acceptance, then `TABPFN_TOKEN` in the environment; inject into every pod, never commit it |
| A forgotten pod bills silently | all | `check` before and after every session; shared account |
| DuckDB extension ABI churn | 0, 6 | Pinned to v1.5.5; C++ template CI matrix. **Keep the `duckdb` submodule and `tools/duckdb.exe` on the same version** — swan pins v1.5.4, we use v1.5.5 |
| `anofox_tabfm` moves under us mid-measurement | 2, 3 | Pin a tag; it is pre-1.0 and tags near-daily (see below) |

## Guiding principle

Each phase produces something that either works or kills an assumption cheaply. Phases 2 and 3
retire nearly all the risk, and neither requires writing a line of C++.
