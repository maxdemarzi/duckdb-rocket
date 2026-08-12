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

**Design consequence:** the pipeline needs probability output, not labels. Verifying that
`anofox_tabfm` can provide it is Phase 2, and it gates everything after it.

## Reference constraints (from the paper)

| Item | Value |
|---|---|
| Feature extractor | Rocket (primary); MiniRocket / MultiRocket also evaluated |
| Kernels | 10,000 total = G=10 groups × 1,000 |
| Features | 2 per kernel (global max, PPV) → 2,000 per group |
| Classifier | TabPFN v2.5, e=8 internal estimators |
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

## Phase 2 — Probe `anofox_tabfm` (GO / NO-GO GATE)

**Goal:** find out whether the intended composition is possible at all. Cheapest phase,
highest information value. **Run it concurrently with Phase 1.**

- [ ] `INSTALL anofox_tabfm FROM community; LOAD anofox_tabfm;` then `tabfm_download('tabpfn-v2-5')`
- [ ] **Does `tabfm_classify()` return class probabilities or only a label?** ← the gate
- [ ] Can it accept 2,000 feature columns? Is `features := [...]` with 2,000 names workable?
- [ ] Does it accept a `LIST`/`ARRAY`-valued column instead of N scalar columns?
- [ ] Does it expose an AMP / precision setting? (See Phase 1.)
- [ ] Measure: latency for one 2,000-feature classify call at realistic UCR row counts
- [ ] Confirm the train/test convention — TabPFN is in-context, so how are labeled context
      rows vs. query rows expressed in the SQL API?

**Pin the version.** `INSTALL ... FROM community` is fine for the initial probe, but swan's
`scripts/vendor_anofox_tabfm.sh` records the reason not to leave it there: `anofox-tabfm` is
**pre-1.0 and tags near-daily**, so an unpinned dependency means Phase 3's accuracy numbers are
not reproducible next week. swan builds it from a pinned tag (`ANOFOX_TABFM_TAG`, at
`v2026.07.17`) as an *independent* artifact rather than a CMake source-level dependency —
deliberately decoupled from its own vendored DuckDB version. Record whatever tag we probe
against alongside the findings note, and reuse that script rather than writing a second one.

**Exit:** a written findings note, plus GitHub issues filed on `anofox-tabfm` for any gaps.

**Branch on the outcome:**
- *Probabilities available* → proceed to Phase 3 as designed.
- *Labels only* → majority-vote across G groups is the fallback (some accuracy loss vs.
  probability averaging; quantify it in Phase 1's harness before committing). Open an upstream
  issue requesting probability output — small, in-scope, generically useful.
- *No list-valued features and 2,000 columns is unworkable* → that upstream PR moves from
  "nice to have" to a prerequisite. Consider offering to write it.

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
| `tabfm_classify` won't return probabilities | 2 | Gate the project on it; majority-vote fallback, quantified in Python first |
| **AMP silently corrupts accuracy numbers** | 1 | Resolved: default is `"auto"` → fp16 on any CUDA device. Force `inference_precision=torch.float32` everywhere; CPU≡GPU checks count only with both pinned to fp32 |
| Kernel-generation spec mismatch Python↔C++ | 1→4 | Portable PRNG spec + golden vectors, decided in Phase 1 |
| 2,000-column SQL calling convention unusable | 2 | Upstream list-valued features PR |
| TabPFN inference dominates runtime, making C++ ROCKET pointless | 3 | Phase 3 measures this *before* any C++ is written |
| Conclusions drawn from one dataset | 1, 5 | 10-dataset subset; noise floor measured first |
| Local Windows timings mislead (WDDM spills instead of OOM) | all | Numbers come from pods, not the 3060 |
| A forgotten pod bills silently | all | `check` before and after every session; shared account |
| DuckDB extension ABI churn | 0, 6 | Pinned to v1.5.5; C++ template CI matrix. **Keep the `duckdb` submodule and `tools/duckdb.exe` on the same version** — swan pins v1.5.4, we use v1.5.5 |
| `anofox_tabfm` moves under us mid-measurement | 2, 3 | Pin a tag; it is pre-1.0 and tags near-daily (see below) |

## Guiding principle

Each phase produces something that either works or kills an assumption cheaply. Phases 2 and 3
retire nearly all the risk, and neither requires writing a line of C++.
