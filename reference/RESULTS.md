# Results

What has actually been measured, with the caveats that make each number readable. Raw output is
in the JSON files beside this one; every script named here regenerates its own.

**Environment.** Two, and which one a number came from matters:

- **Local** — Windows 11, RTX 3060 (unused; `torch` is the CPU wheel by design), DuckDB v1.5.5
  (`d8cdaa33`), `anofox_tabfm` `bc6d8af`, `tabpfn` 8.2.0, `tabicl-v2` as the DuckDB-side
  backbone. Phases 1–4 and the earlier Phase 5 numbers.
- **Pod** — RunPod CPU instance, 16 vCPU, Linux, same DuckDB and `anofox_tabfm`. Full tuple in
  `pod_doctor.json`. The Phase 5 breadth table.

Each Phase 5 report now carries its own `environment` block and a `caveat` that is `null` on a
pod and populated off-pod, so provenance is something the run observed rather than something
asserted for it. That was not always true: the string was previously hardcoded, and every pod
run archived itself as "local Windows timing on a contended box" — exactly backwards, on the
runs that existed to be reportable.

**Local timings here are contended, and understate the pipeline by roughly 1.8x.** The box was
simultaneously running an unrelated `finetune.py` training job with two worker processes, plus
this project's own background runs. Measured against the pod: Beef 129 s local against 67 s,
Coffee 128 s against 64 s. PLAN.md requires reported figures to come from a pod, and the reason
is visible in that gap.

---

## Phase 1 — the oracle, and the noise floor

`scripts/accuracy.py`, TabPFN v2.5 pinned to fp32, `auto_scale_n_estimators=False`, G=40 groups
of 250 kernels (500 features — one estimator's full view), e=1, 3 seeds.

| Dataset | seed 0 | seed 1 | seed 2 | sd | mean ensembling gain |
|---|---|---|---|---|---|
| Coffee | 1.0000 | 1.0000 | 1.0000 | 0.0000 | +0.0000 |
| Trace | 1.0000 | 1.0000 | 1.0000 | 0.0000 | +0.0000 |
| GunPoint | 0.9933 | 1.0000 | 1.0000 | 0.0038 | +0.0032 |
| FaceFour | 0.9773 | 0.9773 | 0.9773 | 0.0000 | +0.0299 |
| Beef | 0.8333 | 0.8667 | 0.7667 | **0.0509** | **+0.0792** |

Mean accuracy across 15 runs: **0.9595**.

**The noise floor is 0.0509**, and it is not evenly distributed — it is *entirely* Beef. Four of
the five datasets are at or near ceiling and reproduce exactly across seeds, so they cannot
discriminate between anything. Any claimed effect smaller than ~5 points on Beef, or smaller
than ~0.4 points on GunPoint, is not a result. This is the number PLAN.md requires before an
accuracy comparison means anything, and it should be read as "one dataset can move, the rest are
saturated" rather than as a single global tolerance.

**Ensembling earns its keep exactly where the task is hard.** The gain from averaging across the
40 groups is +0.0792 on Beef and +0.0299 on FaceFour, and identically zero on the three datasets
that were already perfect. On Beef the per-group mean accuracy is ~0.74 while the ensemble
reaches 0.77–0.87 — the averaging is doing real work, not decoration. That is worth knowing
because it is the part of the paper's design that survived the move to G=40 unchanged.

### Does the backbone swap cost anything?

`tabpfn-v2-5` will not load in DuckDB, so the oracle runs TabPFN v2.5 and the pipeline runs
TabICL v2. Same ROCKET features, same seed, same grouping:

| Dataset | TabPFN v2.5 (oracle) | TabICL v2 (DuckDB) | delta | seed sd |
|---|---|---|---|---|
| Coffee | 1.0000 | 1.0000 | +0.0000 | 0.0000 |
| Trace | 1.0000 | 1.0000 | +0.0000 | 0.0000 |
| GunPoint | 0.9933 | 0.9933 | +0.0000 | 0.0038 |
| FaceFour | 0.9773 | 0.9773 | +0.0000 | 0.0000 |
| Beef | 0.8333 | 0.7667 | **−0.0667** | 0.0509 |

**Four of five are identical, and the one difference is the size of its own noise.** Beef's
−0.0667 is two test rows out of thirty, against a seed-to-seed sd of 0.0509 on that same
dataset — so this subset cannot distinguish the two backbones, and it would be wrong to report
that TabICL is worse.

This bears on PLAN.md's Phase 3b prediction, which expected TabICL v2 to do *relatively badly*
on ROCKET features because its prior is width-sensitive and rotation-hostile, and ROCKET
features are synthetic projections with no stable per-column identity. At **500-feature groups**
that penalty does not appear. The honest reading is narrow: the tabicl fork measured its width
penalty at much greater widths, and this configuration deliberately keeps every group inside one
estimator's 500-feature budget — so the experiment as run does not put the prediction under
strain. Testing it properly needs the wide-group configuration, which anofox cannot reach at
e=1.

## The pod run that failed

Worth recording in full, because the next person to reach for a pod will otherwise repeat it.

**Outcome: 140 minutes of RTX 6000 Ada (~$1.75) and zero accuracy results.** What it did produce
was three findings, all of which change how the next run should be set up.

**1. A GPU buys nothing here.** `tabfm_devices()` on the Linux pod reported only
`CPUExecutionProvider`, exactly as on Windows — the community `anofox_tabfm` build ships
CPU-only ONNX Runtime despite the repository advertising CUDA/ROCm. The GPU sat idle for the
whole run. **A CPU pod is the correct instance type**, and the tooling should support one.

> **Amended 2026-08-13, and the reason is stronger than the original.** The community build
> being CPU-only is a packaging choice and could change. It turns out not to matter: building
> the `cuda` flavor from source and running it on an A40 (sm_86, CUDA 12.8), **`tabicl-v2` —
> our backbone — cannot execute on CUDA at all.** It fails inside the graph at a `ScatterND`
> node. `mitra` runs fine on the same build and device, and `tabicl-v2` runs fine on the CPU
> EP, so this is specific to that model on that provider. Zeroing the test rows gets past the
> first such node and into a second one ~1500 nodes later, so there is no shape we can feed it
> to avoid this.
>
> So "a CPU pod is the correct instance type" holds for a firmer reason than packaging: **as
> shipped, `tabicl-v2` has no GPU path to buy.** Reported upstream in
> [anofox-tabfm#21](https://github.com/DataZooDE/anofox-tabfm/issues/21).
>
> **A workaround exists, and it reopens the GPU question.** The CUDA `Slice` computing the
> ScatterND's `indices` returns its input untrimmed — `[0..T-1]` rather than `[0..S-1]` —
> because the CPU-side buffer holding `S` is recycled before the CUDA kernel reads it. Naming
> the two tensors that carry `S` (`sym_size_int_56`, `val_95`) as extra graph outputs pins
> their buffers, and the **full 4526-node graph then runs on CUDA** at every shape tried,
> matching the CPU EP to 3.5e-4 relative on identical inputs.
>
> **Confirmed with the real checkpoint (2026-08-13).** Built the `cuda` flavor from stock
> upstream on an A40 and registered the stock and patched graphs from the same real weights
> and tensor map, so the graph was the only variable. Stock on CUDA fails; patched on CUDA
> scores all 60 rows with **0/60 label mismatches** against the CPU baseline (max softmax
> delta 4.6e-3, ordinary fp32 CPU/GPU divergence), and patched on CPU is **bit-identical** to
> stock. So GPU is genuinely available to this pipeline, and the CPU numbers below are
> unaffected either way.
>
> Still true: it masks an ONNX Runtime bug (present in 1.26.0 and 1.28.0), not a fix, and it
> is unmerged upstream ([#23](https://github.com/DataZooDE/anofox-tabfm/pull/23)). We do not
> need the merge — `tabfm_register_model` points the stock extension at our own graph — but we
> do need a self-built `cuda` flavor, because **no GPU build is published for any platform**:
> the `ext.anofox.com` host named in anofox's own error message does not resolve (NXDOMAIN
> from both Windows and Linux).
>
> **Not on Windows.** GPU here is Linux-only for now — `ProbeCudaDevices` was compiled out on
> Windows, so `tabfm_devices()` returned the cpu row alone even with a working card. Ported to
> NVML-via-`LoadLibrary` and verified against the local RTX 3060 (`cuda:0`, sm_86, 12 GB,
> driver 610.74); the full Windows CUDA build is the outstanding piece.
>
> Sizing note if a GPU run is ever planned: the local card is 12 GB against ECG5000's ~44 GB
> CPU peak. A GPU changes the speed, not the context-row scaling, so the large datasets still
> need the test-set chunking.
>
> **The CPU path is not implicated, and this is now settled rather than assumed.** Reading the
> shipped graph statically, `ScatterND`'s `indices` is `Unsqueeze(Slice(Range(0,T,1), 0, S))`
> — length S — and its `updates` is `Min(rows, train)` — also S. They agree by construction,
> so the export is correct and the CPU EP is not tolerating anything out of spec. (An earlier
> revision of this note claimed the graph was out of spec. It is not; ONNX simply cannot infer
> `min(T,S)` symbolically, which makes the operands *look* mismatched.) What goes wrong is
> that under CUDA the value arriving at `indices` has length T instead of S. Our accuracy
> numbers run entirely on the CPU EP and are unaffected.
>
> Worth recording because it cost a wrong hypothesis: a hand-built minimal graph carrying that
> exact `Range`→`Slice` pattern runs **correctly** on CUDA, on ORT 1.26.0 and 1.28.0 alike. The
> trigger needs the full ~4500-node graph — partitioning or folding — not the op pattern. Also:
> when the CUDA EP fails to load, ORT silently falls back to CPU and a "CUDA" run reports a
> clean pass, so `get_providers()` has to be asserted, not assumed.
>
> Separately, and unrelated to accuracy: ORT 1.26.0 was initially thought incompatible with
> `anofox_tabfm`. It is not — that was an env-ordering bug on anofox's side, fixed in
> [anofox-tabfm#22](https://github.com/DataZooDE/anofox-tabfm/pull/22).

**2. The cross-platform build works, and conformance holds.** The extension built cleanly under
gcc 11 on an EPYC 7453 and reproduced the golden vectors at **the same 1.776e-15** as the
Windows/MSVC build, with PPV differences of exactly 0. That is the one durable result of the
run, and it is not nothing: it says SPEC.md is portable rather than accidentally
Windows-shaped.

**3. Every sweep run was killed by the OOM killer. The pods needed more RAM, not more cores.**

The exit status is `-9` — SIGKILL. Establishing that took a second pod and three wrong
hypotheses, all recorded here because each one *sounded* right:

| Hypothesis | Killed by |
|---|---|
| Thread explosion from 112 visible cores | The CPU pod had **16** cores and failed identically |
| Concurrency — 4 runs at once | A single sequential run failed too |
| Feature width, 500 columns | `probe_anofox.py` ran 500 columns on the same pod in 4.0 s |
| Dataset or configuration | The same dataset and config run clean locally |

What actually distinguishes the failures is **context rows**. `tabfm_classify`'s memory scales
with training rows × features, and the CPU pod's container limit was 32 GB:

| Dataset | Train rows | On the pod |
|---|---|---|
| Coffee | 28 | works |
| SyntheticControl | 300 | **OOM-killed** |
| OSULeaf | 200 | **OOM-killed** |
| ItalyPowerDemand | 67 (but 1,029 test) | **OOM-killed** |

The local box has more memory than either pod's container limit, which is the entire reason
every one of these runs fine here.

**Before the next pod run:** size the instance on **RAM**, not vCPU or GPU. 32 GB is not enough
for 300 context rows at 500 features, and `--jobs N` multiplies the requirement by N.

Note that `exit -9` was only visible *because* of the stderr fallback added after the first pod
run. Before that, the ONNX-noise filter swallowed the whole message and the failure reported
itself as a bare "FAILED after Ns" — which is what sent the first three hypotheses down the
wrong path.

Two process mistakes made this worse and are worth naming:

- `pkill -f phase5_pipeline` issued over SSH **matched the SSH command line itself**, killing
  the remote shell before the rest of the command ran. Cleanups silently did nothing, stale
  sweeps accumulated, and several sweeps ran concurrently writing the same output files. Use a
  self-excluding pattern (`[p]hase5_pipeline`).
- `phase5_pipeline.py` filters ONNX's schema-registration noise out of stderr, which is
  necessary — but when the real failure produces *no* other stderr, the filter leaves an empty
  message and the script reports a bare "FAILED after Ns". The harness hid the very thing it
  was meant to surface.

### A rented GPU can be broken in a way every check we had said was fine (2026-08-14)

Two consecutive community L40S pods, both on machine `kldbzozpc4vi`, ran no CUDA at all. Everything
we normally trust said otherwise:

    nvidia-smi              NVIDIA L40S, driver 550.163.01, CUDA Version 12.8
    tabfm_devices()         cuda:0  CUDAExecutionProvider  sm_89  47.8 GB free  usable = true
    cudaGetDeviceCount      rc=999  count=-1        <- plain ctypes, system libcudart
    cuInit(0)               rc=999

`tabfm_devices()` is built from NVML, which talks to `/dev/nvidiactl` and never creates a CUDA
context, so it is perfectly happy on a host where no CUDA program can run. Our cache smoke test
called exactly that function and passed the pod through to a run that died three minutes later
inside ORT session creation, behind a 40-line C++ stack naming `cudaSetDevice` — which reads like
our bug, and is not.

`scripts/pod/anofox_cuda.sh --preflight` now calls `cuInit` through `libcuda.so.1` before the
clone, the 110 MB checkpoint or any build. On the second bad pod it returned in about ten seconds
rather than three minutes. A `SECURE`-cloud pod on a different machine passed it immediately and
ran clean. Cost of the lesson: two pods, roughly $0.30.

The general form is one this project keeps rediscovering: **a probe that answers from a different
subsystem than the one you depend on is not a check.** NVML answering for CUDA is the same shape as
`hardware_concurrency()` answering for a cgroup quota.

## Phase 2 — `anofox_tabfm` probe

Full write-up in [PHASE2_FINDINGS.md](PHASE2_FINDINGS.md). Two results reshaped the design.

**`tabpfn-v2-5` does not load.** Its published checkpoint no longer matches anofox's bundled
ONNX graph; re-downloading returns a byte-identical file and the same failure. `tabicl-v2` works.

**A TabPFN estimator sees 500 features, not 2,000.** 2,000 is the input ceiling; above 500,
features are subsampled per estimator (`preprocessing/configs.py:115`). Covering a
2,000-feature group needs e≥4, and anofox caps e at 1. Hence **G=40 groups of 250 kernels**
rather than the paper's G=10 × 1,000 — same kernel budget, same averaging, but one estimator
now sees a whole group.

The trap worth remembering: `tabpfn` silently auto-scales a requested `e=1` up to `e=4` to reach
coverage, so a local run labelled e=1 would really have been e=4 while the DuckDB path really
was e=1. `auto_scale_n_estimators=False` is now pinned.

| Feature columns | Default guard | Guard raised | Wall clock (60 train / 40 test) |
|---|---|---|---|
| 100 | OK | — | 4.1 s |
| 500 | OK | — | 6.8 s |
| 512 | rejected | OK | 7.0 s |
| 1,000 | rejected | OK | 17.2 s |
| 2,000 | rejected | OK | 41.4 s |

The 500-column limit is a configurable guard (`SET anofox_tabfm_max_features`), not a model
limit — so the upstream PR the plan contemplated for wide feature lists is unnecessary. Note the
cost grows faster than the width.

## Phase 3 — SQL composition, Python features

`scripts/phase3_sql.py`. GunPoint, `tabicl-v2`, e=1, G=40.

| | |
|---|---|
| Accuracy | **0.9933** |
| SQL argmax vs. numpy argmax | **150/150** identical |
| Averaged probabilities sum to 1 | yes |
| Row alignment | 150/150 ids, 6,000 group-rows, 40–40 groups per row |
| Wall clock (40 classify calls) | 287 s |

The plan's original exit criterion — "reproduce Phase 1 accuracy exactly" — assumed both sides
run the same classifier, which is impossible now that `tabpfn-v2-5` will not load in DuckDB. So
the *plumbing* is what is checked exactly: the same per-group probabilities averaged and
argmaxed in SQL and in numpy must agree, isolating the composition from the backbone.

Row identity does not need swan's rowid-as-feature hack. `tabfm_classify` returns test rows
only, in the test view's order, stably — but rows are still recovered by joining on an echoed
feature value, and the join is asserted total, because a positional join that silently slipped
would corrupt every number while leaving the output well-formed.

## Phase 4 — `rocket_transform` in C++

Conformance against the golden vectors, `scripts/conformance.py`:

| Fixture | Shape | Max abs diff | Max PPV diff | Outside 1e-9 |
|---|---|---|---|---|
| `features_base` | (8, 128) | 1.776e-15 | **0** | 0 / 1024 |
| `features_offset` | (8, 32) | 1.776e-15 | **0** | 0 / 256 |

`features_offset` starts at global kernel index 9,000 and is the one that matters: it proves
`first_kernel` addresses into a single bank rather than reseeding, which is the property the
whole group design rests on. A SQL test asserts the same thing directly —
`rocket_transform(s, 4, 7, 4) = rocket_transform(s, 8, 7, 0)[9:16]`.

PPV differences are held to exact zero separately from the tolerance: PPV is a ratio of small
integers, so any difference there means the two implementations disagreed about the *sign* of a
convolution output, which is a real bug rather than rounding.

Throughput, `scripts/benchmark_transform.py`, 250 kernels, 150 timepoints:

| Series | Python (numpy) | C++ extension | Speedup |
|---|---|---|---|
| 200 | 0.108 s | 0.065 s | 1.7× |
| 4,000 | 7.961 s | 1.117 s | **7.1×** |

The gap widens with row count because parallelism comes from DuckDB scheduling the scalar
function across chunks, and 200 rows is a single chunk. The oracle is numpy-vectorised, so this
is C++-against-BLAS-ish rather than C++-against-interpreted-Python.

### Pure SQL: correct, and not an option

`sql/rocket.sql` implements the whole spec — SplitMix64 over 32-bit halves in BIGINT, the polar
method's rejection loop as a fixed draw grid plus a running count of accepted pairs, and the
convolution as a join.

| | |
|---|---|
| Max abs diff vs. oracle | 1.776e-15 |
| Max PPV diff | 0 |
| 8 kernels, 2 series, 48 timepoints | Python 0.001 s, **SQL 342 s** |

That is ~4×10⁵ slower, which closes PLAN.md's "if it is merely 5–10× slow, seriously consider
stopping here" branch. It stays as an executable statement of the spec and a zero-build fallback
for tiny inputs.

Two bugs found on the way, both of which failed far from their cause: partial products need
HUGEINT because two 32-bit factors reach 2^64 and wrap BIGINT into negatives, and three of the
six SplitMix64 constants had been transcribed from hex incorrectly.

## The hard datasets: where the in-context model actually earns its place (2026-08-14)

The ten-dataset subset was chosen for *spread*, and nine of its ten sit at 0.94-1.00. At that
altitude ROCKET+ridge ties this pipeline (0.9636 vs 0.9615, 3 wins to 4 with 3 ties) at ~14x less
cost, which reads as "the model adds nothing". It was the wrong place to look.

Of the 112 bake-off datasets, 13 have a best published accuracy below 0.75. Six of those are within
`tabicl-v2`'s 10-class cap. On an A40, patched graph, G=40, e=1, `--test-chunk 128`:

| dataset | **pipeline** | our ridge | our mr-hydra | published ROCKET | published best |
|---|---|---|---|---|---|
| Herring | 0.6406 | 0.6250 | **0.7344** | 0.594 | 0.734 |
| **MiddlePhalanxTW** | **0.6104** | 0.5325 | 0.5130 | 0.539 | *0.578* |
| RefrigerationDevices | 0.5573 | 0.5307 | 0.5173 | 0.512 | 0.600 |
| Haptics | 0.5552 | 0.5357 | 0.5260 | 0.526 | 0.571 |
| ScreenType | 0.5200 | 0.4773 | — | 0.467 | 0.595 |
| InlineSkate | 0.4909 | 0.4764 | — | — | 0.544 |

**Against ridge on the same features: 6 wins from 6, mean +0.033.** Deltas +0.0156, +0.0779,
+0.0266, +0.0195, +0.0427, +0.0145. The mr-hydra column still covers the first four only. Against
published ROCKET: +0.047, +0.071, +0.045, +0.029, and +0.053 on ScreenType, which is one of the
datasets where ROCKET specifically trails the field. On MiddlePhalanxTW the pipeline **beats the
best published result** (0.6104 vs 0.578 across ROCKET, HC2, InceptionTime, Hydra-MR, 1NN-DTW and
FreshPRINCE). InlineSkate has no published ROCKET figure recorded here, so that cell is empty
rather than filled with a guess.

The ridge figures for ScreenType and InlineSkate arrived later than the rest, from the feature screen
below, which needed the same baseline. That the in-context model wins all six on identical features
is the claim the original ten-dataset subset could not make -- there, at 0.94-1.00, ridge tied it.

So the earlier "ridge ties it" finding was an artefact of testing on saturated problems. The
in-context model does extract more from ROCKET features than a linear head — it just cannot show
that where a linear head already scores 0.99.

`mr-hydra` splits 3-1 against the pipeline and wins Herring outright (0.7344 vs 0.6406), so a
better *feature extractor* is competitive with a better *classifier*. Both beat ridge.

The last two rows were held back for two days and three attempts at the id recovery, because both
first came out of a run whose row alignment was wrong. They are reported now, at `rc=0`, with
`min_groups_per_row = max_groups_per_row = 40` on both — 15,000 group-rows over 375 ids and 22,000
over 550. `reference/phase5_ScreenType_gpu.json`, `reference/phase5_InlineSkate_gpu.json`.

### Widened to 67 datasets: that +0.033 is a hard-dataset margin, not a general one (2026-08-14)

The six rows above are six rows, and this project has now had three six-dataset conclusions fail to
survive widening. So the teacher was run across every dataset inside `tabicl-v2`'s 10-class cap —
75 archived reports, 67 of them with a clean run and a loadable test split — and each was compared
against the same two students on the **same full test split**, per learner, never as a max over
learners. `scripts/distill_gate.py --gate`, `reference/distill_gate.json`.

| subgroup | n | vs ROCKET+ridge | vs mr-hydra |
|---|---|---|---|
| every archived teacher | 67 | 30/67, **+0.0085**, p=0.69 | 25/67, **+0.0019**, p=0.68 |
| best student < 0.90 | 29 | 21/29, **+0.0294**, p=0.0125 | 17/29, **+0.0198**, p=0.25 |
| best student < 0.75 | 11 | 11/11, **+0.0572**, p=0.0010 | 8/11, **+0.0326**, p=0.23 |
| best student ≥ 0.95 | 28 | 6/28, −0.0057, p=0.17 | 6/28, −0.0102, p=0.12 |

Two-sided sign tests on the paired per-dataset differences — distribution-free on purpose, because a
30-row test set and a 4,500-row one are not comparable draws.

**Over the whole archive the pipeline and both students are level.** +0.0085 and +0.0019 are below
the 0.0140 shift that n=67 detects at 80% power, and the median difference is exactly 0.0000: **28 of
the 67 datasets have a student at 0.95 or better**, and on those neither model can move.

**On the datasets with headroom the margin is real.** Below 0.90 the pipeline leads ridge by 2.9
points on 21 of 29; below 0.75 it wins **11 of 11** at p=0.0010, which survives correcting for the
four subgroups looked at. That subgroup was specified in `docs/DISTILLATION_PLAN.md` before any of
these numbers existed, on the argument that a saturated dataset cannot demonstrate anything.

**Against `mr-hydra` the sign is positive everywhere and significant nowhere** (+0.0198 and +0.0326,
p=0.25 and p=0.23). That is a power statement, not a tie: only eleven datasets in the whole archive
are genuinely hard, so widening further cannot fix it — the archive runs out of hard problems before
the test runs out of appetite.

So the six-dataset "+0.033, 6 of 6" above is a **correct measurement of a subgroup, reported as if it
were general**. The corrected claim: this pipeline beats a ridge head on the same features where the
problem is hard, ties it where the problem is easy, and its advantage over a stronger convolutional
student is unresolved and will stay unresolved on UCR.

The gate cost 56.9 minutes locally at 7 workers, with per-`(dataset, learner, seed)` student
accuracies cached under `data/gate_students/` so that re-running it as more teacher reports land
costs only the new fits.

### Distillation: the teacher's error *rate* was never the problem (2026-08-15)

The gate above opens on 29 datasets — those where a label-only student is still below 0.90. Arm B
asks the actual question there: train a student on the train split plus a pool of test rows the
teacher has labelled, and score it on a held-out half. `scripts/distill_gate.py --arm-b`, 28 datasets
with soft-label sidecars, one 50/50 split each, `rocket+ridge`.

| arm | what the pool carries | wins | mean | p | share of the ceiling |
|---|---|---|---|---|---|
| **C** | the **true** labels | 21/28 | **+0.0474** | 0.0001 | — (this *is* the ceiling) |
| **Bs** | the teacher's full distribution | 15/28 | +0.0119 | 0.13 | 25.1% |
| **Bc** | argmax, most confident half | 14/28 | +0.0064 | 0.19 | 13.5% |
| **B** | argmax, whole pool | 12/28 | +0.0011 | 1.00 | 2.4% |

`Bs − B` is +0.0107 at p=0.0525, paired on identical splits — the one comparison here worth trusting,
since it changes nothing but how the pool's labels are represented. Note that `Bs − A` at +0.0119 is
**below** the 0.0163 this design can detect at 80% power, so "soft-target distillation works" is not
established. What is established is that soft targets beat hard argmax, and that the ceiling is four
times either.

**The break-even sweep, which was meant to close the question and did the opposite.** Rather than
test candidate labellers one at a time, corrupt the pool's *true* labels at 5/10/20/30/40% and find
where the gain crosses zero. That gives the error rate any labeller must beat, and then every
candidate — ensembles included — is one number against another. Ten datasets with ≥5 points of
headroom, both learners:

    median break-even   25.6%      the pool tolerates a quarter of its labels being wrong
    median teacher err  21.6%      the teacher is inside that on 9 of 13 measurable cases
    7 of 20 cases still paid at 40% noise

By that reading arm B should have paid. It returned +0.0011.

**Both are right, and the contradiction is the result.** Matching each dataset's synthetic noise to
its teacher's exact error rate — same split, same pool rows, same student, interpolating the measured
curve rather than snapping to the nearest swept rate — replacing the teacher's labels with *random*
labels of the same error rate is worth about five points:

| | wins for random noise | mean | p |
|---|---|---|---|
| all 20 (dataset, learner) | 15/20 | +0.0401 | 0.0414 |
| no extrapolation (T err ≤ 40%) | 10/12 | **+0.0516** | 0.0386 |

So the error *rate* is not what governs. **A teacher's mistakes are not noise**: they concentrate on
the same ambiguous rows and point the same way, so the student learns a coherent wrong rule, where
random errors of the same size largely cancel. That also explains the arm ordering above — soft
targets and confidence filtering both work by marking exactly the rows the teacher got systematically
wrong, which is why they recover 25% and 14% of the ceiling against hard argmax's 2%.

The interpolation matters and is not a detail: the swept rates sit *below* the teacher's error on
nine of ten datasets, so snapping to the nearest one compares against a **less** corrupted pool and
flatters the random side. Correcting that bias made the effect larger, not smaller.

`reference/distill_armb.json`, `reference/distill_breakeven.json`.

#### Routing works where distilling does not (2026-08-15)

If a teacher's errors are systematic, the fix is not to make its labels better but to stop putting
them in a training set. Route instead: run the cheap student on everything, sort by its decision
margin, and hand only the rows it is least sure of to the teacher. A teacher error then costs one
row instead of biasing every coefficient, and the requirement weakens from "the teacher's labels are
right" to "the teacher is right where the student is unsure".

`scripts/distill_gate.py --route`, the same 28 datasets, full test splits, escalation fraction fixed
in advance rather than tuned:

| escalate | rocket+ridge | random rows | **signal** | mr-hydra | random rows | **signal** |
|---|---|---|---|---|---|---|
| 10% | +0.0095 (p=0.019) | +0.0033 | **+0.0061** (p=0.013) | +0.0083 (p=0.029) | +0.0025 | **+0.0059** (p=0.036) |
| **20%** | **+0.0200** (p=0.004) | +0.0065 | **+0.0135** (p=0.013) | **+0.0145** (p=0.015) | +0.0044 | **+0.0101** (p=0.036) |
| 30% | +0.0223 (p=0.019) | +0.0084 | +0.0139 (p=0.013) | +0.0201 (p=0.052) | +0.0054 | +0.0146 (p=0.013) |
| 50% | +0.0283 (p=0.006) | +0.0154 | +0.0129 (p=0.036) | +0.0230 (p=0.052) | +0.0104 | +0.0127 (p=0.013) |

**The "random rows" column is the control, and without it the rest is unreadable.** Escalating *any*
rows to a teacher that is better on average buys something, so a rising curve is not evidence that
the student knows what it does not know — the difference between the two columns is. At a 20% budget
the student's own uncertainty picks rows about three times better than chance, significantly, for
both students and at every budget.

Set against distillation on the identical datasets — hard argmax +0.0011 (p=1.00), soft targets
+0.0119 (p=0.13) — **escalating 20% of rows beats every distillation arm, and does so significantly.**

**What routing does not do is beat the teacher.** At the same fixed 20% budget, against running the
teacher on every row:

    rocket+ridge    6/28   mean -0.0116   p = 0.0227      significantly behind
    mr-hydra       11/28   mean -0.0060   p = 0.8388      level

So this is a cost trade, not a free win: it captures 63% of the teacher's advantage for ridge
(+0.0200 of +0.0316) and 71% for `mr-hydra` (+0.0145 of +0.0205). ~~The teacher costs ~14x the
student per row (262 s against 3,741 s over ten datasets, `DISTILLATION_PLAN.md`), so a 20%
escalation runs at ~3.6x a student-only system rather than 14x.~~ **Superseded — that cost claim is
wrong; see "What routing actually costs" below.** It divided totals from different hardware by row
counts, which assumes a per-row cost the teacher does not have. Measured, the ratio is 55-69x per
row and a 20% escalation costs 25-83% of teacher-everywhere depending on test-set size, never the
20% a per-row model predicts. The *accuracy* rows of this table are unaffected.

The curve is concave, which is what makes a small budget worth having: the first 10% of escalated
rows buys a third of the total gain and the last 50% buys almost none. On `mr-hydra` the 50% point is
*above* running the teacher on everything (+0.0230 against +0.0205), because past some budget the
escalation starts handing over rows the student was getting right.

`docs/ROUTING.md` writes this up for both a user and an implementer.

There is a figure in the per-dataset output that looks stronger — "beats both ends" on 12/28 and
20/28 — and it should not be quoted as this one. It uses the `best` column, whose escalation
fraction is chosen per dataset **on the same test split it is scored on**. That is an oracle bound on
what a tuned rule could reach. It was labelled as an oracle in the output and then repeated in prose
here as though it were achievable, which is the same mistake as the six-dataset feature shortlist,
in a new place.

Two things this table does not claim. The `best` column in the per-dataset output selects the
escalation fraction on the same test split it is scored on, so it is an oracle bound on what a tuned
rule could reach and is labelled as such; the fixed budgets above are what a product could run. And
`mr-hydra` exposes no usable confidence — aeon's `predict_proba` returns one-hot for a
`RidgeClassifierCV` backbone — so its decision margin is recovered by reproducing its private
transform pipeline, which is checked against `predict()` on every fit and raises if they disagree.
A wrong reconstruction would return entirely plausible margins and route the wrong rows.

`reference/distill_route.json`.

#### What routing actually costs: the 3.6x above is wrong (2026-08-15)

The paragraph above says the teacher costs ~14x the student per row, so a 20% escalation runs at
~3.6x a student-only system. **Both halves are wrong, and the second is wrong in a way that matters
more than the first.** They were assembled from per-dataset totals on different hardware and then
divided by row counts, which assumes the teacher's cost is proportional to how many rows you send
it. It is not.

Measured properly for the first time: one rented AMD EPYC 4564P, 16 cores, 124 GB, nothing else
running, all three arms in the same process invocation at the same moment
(`scripts/route_serve.py serve --compare`).

| | Herring | ScreenType |
|---|---|---|
| train rows (the teacher's context) | 64 | 375 |
| student, per row | 23.3 ms | 30.3 ms |
| teacher on the escalated rows | 63.2 s for 14 | 223.6 s for 22 |
| teacher on the whole batch | 82.4 s for 64 | 267.9 s for 128 |
| **teacher / student, per row** | **55x** | **69x** |
| rows escalated | 22% | 17% |
| **cost of escalating them** | **77%** | **83%** |

Escalating a fifth of the rows costs four fifths of running the teacher on all of them. The reason
is in `tabfm_classify`'s contract rather than in this pipeline: the teacher has **no trained weights
for your task**, so every call re-encodes the labelled training context from scratch. Fitting
`seconds_per_group = a + b*n` on the two batch sizes each run already produces:

    Herring      1.398 s per group + 9.0 ms per query row   ->  55.9 s fixed over 40 groups
    ScreenType   5.053 s per group + 9.7 ms per query row   -> 202.1 s fixed over 40 groups

The fixed term is 71% and 80% of the respective full-batch costs, and routing does not touch it.
Three independent estimates of the marginal term agree — 9.0 ms, 9.7 ms, and 9.3 ms from a
regression of per-group seconds on `n_train` and `n_test` across 51 archived CPU runs — while the
fixed term tracks the *training* set at roughly 14 ms per labelled row per group.

Read off the per-group timings, not the wall clock, which is how it is known that DuckDB startup is
0.5 s of a 63.2 s call rather than part of the fixed term.

**So what routing saves is calls, not rows.** The context pass is paid once per group per
`--test-chunk`, so escalation only saves anything when it drops the chunk count. Applying the fitted
model across the 28 subgroup datasets at the 128-row chunk these runs use:

| test-set size | datasets | escalating 20% costs, of teacher-everywhere |
|---|---|---|
| <= 128 rows | 7 | 77% |
| 129-384 | 16 | 40% |
| >= 385 | 5 | 25% |

against the 20% that a per-row cost model predicts. Herring and ScreenType are both in the top row,
so the two measured datasets are the **worst** case rather than the typical one — but no dataset
reaches 20%, and the ~3.6x claim above is optimistic by roughly an order of magnitude on a small
batch.

Two consequences. **Batch the escalated rows**: everything above is per call, so a server that
escalates one row at a time pays the entire fixed cost per request. And the fixed term is per
*group*, exactly linear in `G` — verified rather than assumed, since after group 0's ONNX warm-up
(1.22-1.36x) the remaining groups run flat to +/-1% on all four clean runs. That makes the group
count the lever worth pulling and the student's kernel count nearly irrelevant on a routed path: the
student is 1.7% of a routed ScreenType batch, so cutting its 10,000 kernels to 2,000 — a genuine 5x
on the transform, which is exactly linear in kernels at 0.93 ms/row for 250 and 37.37 ms/row for
10,000 — saves under 2% of the request. On a student-only path that same cut is the whole bill.

Not established here: `SemgHandMovementCh2`'s full-batch arm failed on both attempts, dying after
group 1 of 40 on 128 test rows while its 24-row escalated arm succeeded every time. An 8 GB DuckDB
memory limit was blamed and then exonerated when the failure recurred at ~87 GB; `route_serve.py`
now reports the shell's exit code and archives a `crash.log`, which is what should have been
reported instead of a guess. **Resolved on 2026-08-16** — exit `-9`, an OOM kill against a 32 GB
cgroup cap, fixed by `--test-chunk 32` rather than by any memory setting, and *still* not fixed by
the one that was built for it (see "`anofox_tabfm_max_memory` guards residency, not allocation").
See "The shipped default,
actually served".

#### The teacher runs 40 passes and needs about ten (2026-08-15)

`G` was inherited from the paper's configuration and never chosen for cost. The section above
establishes that the teacher's cost is exactly linear in it — each group is its own
`tabfm_classify` call carrying its own context pass, and after group 0's ONNX warm-up the
remaining groups run flat to +/-1%. So the question is only whether accuracy survives.

**One 40-group run answers it for every G at once.** Group *g* covers kernel indices
[250g, 250(g+1)) and the prediction is the argmax of the *mean* of the groups' probabilities, so
averaging the first G groups is exactly what a G-group run computes — provided `--num-kernels` is
scaled with `--n-groups` to hold kernels-per-group at 250. `--per-group-soft` archives the
unaveraged cube; `perf_levers.py --groups` averages prefixes of it. Each cube is checked against
its own run's reported accuracy to 1e-9 before it is used.

24 datasets, full test splits, `tabicl-v2` on CPU:

| G | teacher alone | vs G=40 | p | **routed @20%** | vs G=40 | p | not worse |
|---|---|---|---|---|---|---|---|
| 1 | 0.7214 | −0.0172 | **0.012** | 0.7325 | −0.0019 | 0.38 | 11/24 |
| 2 | 0.7273 | −0.0113 | 0.115 | 0.7294 | −0.0049 | 0.33 | 13/24 |
| 5 | 0.7299 | −0.0087 | 0.167 | 0.7325 | −0.0018 | 0.81 | 14/24 |
| 10 | 0.7348 | −0.0038 | 0.648 | 0.7310 | −0.0033 | 1.00 | 17/24 |
| 20 | 0.7385 | −0.0001 | 1.000 | 0.7345 | +0.0002 | 0.11 | 22/24 |
| 40 | 0.7386 | — | — | 0.7343 | — | — | 24/24 |

(student alone 0.7205, so routing at 20% is worth +0.0138 here.)

**G=20 is free: half the cost, +0.0002.** G=10 is a 4x cut for a loss no test detects. Only G=1 is
measurably worse, and only as the teacher — its *routed* deficit is −0.0019, because a fifth of the
rows reach the teacher and a 0.0172 deficit arrives diluted fivefold.

That dilution is why the recommendation is **G=10 rather than G=1**, even though the routed column
barely separates them. At G=1 the teacher itself is significantly worse, so the routed number is
being carried by the escalation rate rather than by the model; raise the budget later and it
degrades. G=10 is safe on both columns at once.

Against the measured cost model, a routed 128-row ScreenType batch at G=10 falls from **227.5 s to
~60 s, 3.8x, for −0.0033**. That is the largest speedup available here without an upstream change.

Reproduction is worth recording separately: all 24 datasets returned **bit-identical accuracies to
the archived runs** on different hardware with a rebuilt extension.

##### Completed to 29 datasets, and the answer does not move (2026-08-16)

The five missing above — `EthanolLevel`, both `*OutlineCorrect`, both `SemgHand*` — were written
off as too large for the box. They are not: they need **more RAM, not a smaller call**. Re-run on a
64 GB pod, all five completed at the archived `--test-chunk 128`, so the recovered cubes are the
same configuration as the other 24 and not a chunked approximation of it. `DistalPhalanxOutlineCorrect`
peaked at 39.9 GB — above the 29.8 GiB the earlier pod allowed, below the new ceiling. The chunk
ladder built for this never fired.

| G | teacher alone | vs G=40 | p | **routed @20%** | vs G=40 | p | not worse |
|---|---|---|---|---|---|---|---|
| 1 | 0.7253 | −0.0289 | **0.0009** | 0.7384 | −0.0052 | 0.076 | 11/29 |
| 2 | 0.7368 | −0.0174 | **0.015** | 0.7375 | −0.0061 | 0.267 | 14/29 |
| 5 | 0.7432 | −0.0110 | 0.152 | 0.7413 | −0.0023 | 1.00 | 17/29 |
| 10 | 0.7499 | −0.0043 | 0.405 | 0.7403 | −0.0033 | 0.815 | 19/29 |
| 20 | 0.7538 | −0.0004 | 1.000 | 0.7436 | +0.0000 | 0.267 | 25/29 |
| 40 | 0.7542 | — | — | 0.7436 | — | — | 29/29 |

**The G=10 routed cost is −0.0033 on 24 datasets and −0.0033 on 29** — the same to four decimal
places, on a fifth more data including the five hardest. G=20 goes from +0.0002 to exactly 0.0000.
Nothing about the recommendation moves.

What the five *do* change is the case against G=1, which is the one conclusion that was ever
significant. They are datasets where the teacher earns most — mean teacher accuracy rises from
0.7386 to 0.7542 — so the cost of collapsing it to a single group deepens from −0.0172 to −0.0289,
and p from 0.012 to 0.0009. G=2 becomes significant as well (p=0.015), which it was not at 24.
Per-dataset at G=10 the five split both ways: −0.0275 (`MiddlePhalanxOutlineCorrect`), −0.0140,
−0.0036, 0.0000, +0.0111 (`SemgHandSubjectCh2`).

One control worth keeping. Chunking a call changes the probabilities by up to **9e-6**, and that is
chunking alone: the same dataset at the same chunk size on a 16-vCPU and a 32-vCPU box is
**bit-identical**, max|d| exactly 0. But the cubes store six decimals, so 9e-6 is nine units in the
last stored place, every observed difference is a whole multiple of 1e-6, and prefix-averaged
predictions agree 100% at G=1, 5, 10, 20 and 40 — the closest decision anywhere is 4.4e-3, or 489x
the difference. So chunking cannot move a reported number, and a first version of this check that
compared at 1e-6 was testing below its own storage precision and cried wolf.

`reference/perf_groups_29.json`, and the 29 cubes in `reference/pergroup_cubes.tar.gz` (4.4 MB).
`reference/perf_groups.json` is the 24-dataset original, kept because the kernel and joint sweeps
are stated against it.

#### The student's kernel bank, and why it is the wrong thing to cut (2026-08-15)

The other inherited default: 10,000 kernels, 20,000 features for every row on the path every row
takes. Routing does not need the student's best accuracy, it needs its *ordering* — and those are
different requirements, so this is worth asking separately. 28 datasets, teacher fixed at its
archived 40-group labels:

| kernels | student | vs full | p | routed @20% | vs full | p |
|---|---|---|---|---|---|---|
| 250 | 0.7102 | −0.0113 | 0.169 | 0.7301 | −0.0115 | 0.189 |
| 500 | 0.7104 | −0.0111 | 0.029 | 0.7377 | −0.0038 | 0.424 |
| 1,000 | 0.7141 | −0.0074 | 0.169 | 0.7348 | −0.0068 | 0.286 |
| 2,000 | 0.7161 | −0.0054 | 0.108 | 0.7397 | −0.0019 | 0.541 |
| 5,000 | 0.7179 | −0.0036 | 0.308 | 0.7381 | −0.0034 | 0.027 |
| 10,000 | 0.7215 | — | — | 0.7416 | — | — |

The transform is exactly linear in kernel count — 0.93 ms/row at 250 against 37.37 at 10,000,
measured single-job on an idle box — so 2,000 kernels is a 5x cheaper student for −0.0019 routed.

**And it is worth almost nothing, because the student is not the bill.** On the ScreenType batch
measured above the student was 30.3 ms/row of a 1,777 ms/row routed request: **1.7%**. Cutting it
fivefold saves under 2% of a routed request. The same cut is the *entire* saving for a
student-only system, where 37.4 ms/row becomes 7.3 for −0.0054 of student accuracy — a real trade,
in the one deployment that has no teacher in it.

So the two levers are not comparable in value: the group count divides the dominant term and the
kernel count divides a term that rounds to nothing. Note the p-column is a sign test and is
sensitive to direction rather than size — 5,000 kernels reads as significant at 0.027 while 2,000
reads as 0.541, on effect sizes of −0.0034 and −0.0019. Neither is a real difference between those
two rows.

An earlier version of this table was **retracted**: its first run put one worker per (dataset,
kernel size), so six workers pulled the same cold dataset archive concurrently, 44 of 168 fits died
on half-written files, and each row averaged over a different subset of datasets while reporting a
single count. The table above is 28 datasets at every size with no failures.
`reference/perf_kernels.json`.

#### Sending the teacher a smaller context: do not, except to make a run possible (2026-08-15)

The context is the third lever and the only one that attacks the fixed term without an upstream
change. `tabfm_classify` re-encodes the labelled rows on every call, so sending half of them should
halve the dominant cost. It does not, and it is not free.

Six datasets at G=10, `--max-train-rows` with a stratified draw and a floor of one row per class:

| context | classify | accuracy vs full | worst | not worse |
|---|---|---|---|---|
| 25% | **1.94x** faster (of a possible 4x) | −0.0632 | −0.1776 | 1/4 |
| 50% | **1.48x** faster (of a possible 2x) | −0.0168 | −0.0526 | 2/4 |

**Sub-proportional and expensive.** Sub-proportional because only the fixed term shrinks — the
~9 ms per query row is untouched and becomes the floor, exactly as the cost model predicts (176 s
-> 119 -> 89 on ScreenType against a predicted 187/111/74). Expensive because the context is the
model's *only* knowledge of the task, unlike the groups, which are an ensemble over feature subsets
and are genuinely redundant.

The damage tracks **examples per class**, not the fraction removed:

| dataset | classes | rows at 25% | per class | accuracy |
|---|---|---|---|---|
| MedicalImages | 10 | 95 | **9.5** | **−0.1776** |
| ProximalPhalanxTW | 6 | 100 | 16.7 | +0.0049 |
| ScreenType | 3 | 94 | 31.3 | −0.0400 |
| Computers | 2 | 62 | 31.0 | −0.0400 |

So the comparison across all three levers is not close. **The group count divides the dominant term
four times over for −0.0033; the context divides it 1.48x for −0.0168 and can cost 0.18 on a
many-class dataset; the kernel count divides 1.7% of the bill.** Cut groups.

The one thing this lever does that nothing else does is make an impossible run possible.
`EthanolLevel` and `DistalPhalanxOutlineCorrect` cannot run at full context on a **29.8 GiB** CPU
pod at all — their 504 and 600 row contexts exceed it — and both completed at 25% and 50%
(0.5880/0.6120 and 0.7754/0.7862). Those numbers confound the context cut with G=10 and are not
comparable to the archived full-context values; what they establish is feasibility, not accuracy.

**Superseded as advice (2026-08-16).** Both ran at full context, full test set and the archived
`--test-chunk 128` on a 64 GB pod — `DistalPhalanxOutlineCorrect` peaked at 39.9 GB, just over the
old ceiling. So the honest ordering is: rent the memory, then chunk the call, and only subsample the
context if neither is available. Subsampling is the only one of the three that changes the model's
answer.

`reference/perf_context.json`. The 29 per-group cubes are archived compressed as
`reference/pergroup_cubes.tar.gz` (4.4 MB), so the group analysis can be redone or extended
without renting anything.

#### Both levers at once, and a noise floor worth naming (2026-08-16)

The two sweeps above each held the other lever at its default: the group sweep ran a
10,000-kernel student, and the kernel sweep routed against the archived 40-group teacher. **The
cheap corner was therefore never measured**, and "cut both" was being quoted as though the two
compose. Crossing them on the 24 datasets that have cubes, routed accuracy at a 20% budget:

| kernels | G=5 | G=10 | G=20 | G=40 |
|---|---|---|---|---|
| 500 | 0.7336 (−0.0007) | 0.7341 (−0.0003) | 0.7362 (+0.0019) | 0.7363 (+0.0020) |
| 2,000 | 0.7328 (−0.0015) | 0.7329 (−0.0014) | 0.7359 (+0.0015) | 0.7340 (−0.0003) |
| 5,000 | 0.7307 (−0.0037) | 0.7298 (−0.0045) | 0.7322 (−0.0022) | 0.7314 (−0.0029) |
| 10,000 | 0.7325 (−0.0018) | 0.7310 (−0.0033) | 0.7345 (+0.0002) | **0.7343** |

**They compose without penalty — but the stronger reading is that none of this grid is measurable.**
Every cell is within ±0.005 of the baseline, no comparison reaches significance (p = 0.11 to 1.00),
and the sign test would need about 18 of 24 datasets in one direction where every cell lands at
11-15. A 20x cut in kernels and an 8x cut in groups together cost −0.0003.

Two things say plainly that these numbers are noise rather than small effects:

* **The grid is not monotone.** 5,000 kernels is the *worst* row at every group count, worse than
  both 2,000 and 10,000. No mechanism makes a middle bank size worse than the ones either side.
* **The sign flips with the dataset subset.** The same 500-vs-10,000 comparison at G=40 gives
  −0.0038 over the kernel sweep's 28 datasets, −0.0001 over the 24 of those that have cubes, and
  +0.0020 in this run over the same 24 against a freshly measured teacher. Three estimates of one
  quantity, straddling zero.

So the honest statement is **not** "500 kernels is as good as 10,000". It is that this
28-dataset harness cannot resolve differences of about half a point at a 20% escalation budget, and
every configuration tried sits inside that. Anyone wanting to *establish* equivalence needs more
datasets or more seeds, not a better reading of these.

What that supports in practice: the accuracy data gives no reason to keep 10,000 kernels, and the
cost data says cutting them matters once the teacher path is cheap — the student is 1.7% of a
routed request today but 43% of one at G=10 with an upstream context cache. Cut them, and state the
tolerance (~0.005) rather than claiming the cut is free.

`reference/perf_joint.json`.

#### The shipped default, actually served (2026-08-16)

Everything above about G=10 was offline: per-group cubes averaged at a prefix, plus a linear cost
model. `route_serve` was switched to G=10 / 2,500 kernels on the strength of it and then never run,
so the "227.5 s → ~60 s" in circulation was a division, not a measurement. This is that
configuration served — three arms, one 16-vCPU pod, **both group counts on the same box inside one
hour** (`scripts/pod/serve_compare_cpu.sh`).

**First, the archived cost measurements reproduce to under a percent on different hardware.** The
G=40 arms against the 2026-08-15 pod, which is the only reason to trust anything below:

| | Herring, escalated | Herring, all | ScreenType, escalated | ScreenType, all |
|---|---|---|---|---|
| 2026-08-15 | 63.2 s | 82.4 s | 223.6 s | 267.9 s |
| 2026-08-16 | 63.7 s | 82.8 s | 222.6 s | 266.1 s |

and the fitted costs with them: 1.398 s + 9.0 ms → **1.411 s + 9.0 ms**, 5.053 s + 9.7 ms →
**5.033 s + 9.5 ms**. A wall-clock measurement of an in-context model reproducing within 1% on a
different rented box, a day apart, is not what this project expected going in.

**The default is 3.8x cheaper, not 4x:**

| | Herring | ScreenType | Semg (chunk 32) |
|---|---|---|---|
| teacher-everywhere, G=40 | 82.8 s | 266.1 s | 1013.5 s |
| teacher-everywhere, G=10 | 21.4 s | 70.4 s | 258.5 s |
| | **3.87x** | **3.78x** | **3.92x** |
| whole routed request, G=40 | 65.2 s | 226.3 s | 280.8 s |
| whole routed request, G=10 | 17.6 s | 60.1 s | 75.4 s |
| | **3.70x** | **3.77x** | **3.72x** |

The shortfall from 4x is the per-run work that does not scale with `G` — DuckDB startup and the
ROCKET transform, 0.5-2.4 s of each call. The student's own cut *is* 4x, as linear-in-kernels
requires: 22.49 → 5.69, 29.37 → 7.26, 58.07 → 14.86 ms/row (3.95x, 4.05x, 3.91x).

Accuracy on these batches, all six arms:

| | routed G=40 | routed G=10 | teacher G=40 | teacher G=10 | student 10k | student 2.5k |
|---|---|---|---|---|---|---|
| Herring (64 rows) | 0.6406 | 0.6719 | 0.6562 | 0.6406 | 0.6250 | 0.6562 |
| ScreenType (128) | 0.5312 | 0.5391 | 0.5938 | 0.6094 | 0.4922 | 0.4766 |
| Semg (128) | 0.7422 | 0.7266 | 0.8281 | 0.8281 | 0.6953 | 0.6641 |

Three batches of 64-128 rows settle nothing on their own — one row is 0.8-1.6 points, and the
differences sign-flip exactly as the 24-dataset noise floor above says they should. What matters is
that they do not contradict it, and that Semg's teacher scores **identically** at both group counts
for a quarter of the time.

**One caveat the offline analysis could not have produced.** The realized escalation rate moved:
Herring escalated 34.4% of its batch at G=10 against 21.9% at G=40, both aiming at 20%. The
threshold is a quantile of out-of-fold margins on the *training* rows, and a 2,500-kernel student's
test margins do not fall the same way a 10,000-kernel one's do. ScreenType and Semg stayed near
target (17.2% both, 21.9% vs 18.8%), so it is not universal — but **the 3.8x is on the teacher
call, not guaranteed on the request**, and a batch that escalates half again as many rows hands
some of it back. Herring still came in at 3.70x because escalating more rows is cheap and calling
the teacher at all is not.

**SemgHandMovementCh2, finally diagnosed.** It has failed its full-batch arm on every previous
attempt; an 8 GB memory limit was blamed and then exonerated at ~87 GB. The exit code says it
plainly: **-9, SIGKILL, after scoring 1 group of 40 on 128 test rows** — an OOM kill against the
container's 32 GB cgroup cap, on a host whose `free` reports 124 GB. It reproduces identically at
G=10 (1 of 10), which rules the group count out: the allocation is per *call*, and the only thing
that sizes a call is how many rows it is handed. At `--test-chunk 32` the dataset completed for the
first time, at both group counts. DuckDB sat at 26-27 GB RSS throughout, against a 29.8 GiB ceiling.

**And it puts the cost thesis at its limit.** Semg's context is 450 labelled rows of 1500
timepoints, and the fit at G=40 reads:

    6.111 s fixed per group per call + 0.1 ms per query row

The fixed pass is **100%** of the full-batch cost, to the printed precision. Query rows are free;
the labelled context is the entire bill. This is also the one dataset where routing looks good on
cost — 25-29% of teacher-everywhere — and the reason is not that it escalated fewer rows but that
the full batch needed four calls where the escalated batch needed one. *Routing saves calls.*

That number only exists because `cost_model` learned about chunking first. The old two-point fit
would have charged the full arm's three extra context passes to its 104 extra rows and reported
**~178 ms per query row against a true 0.1** — a wrong answer, in the confident direction, on the
one quantity the whole routing argument turns on.

**The run also caught the contention detector crying wolf at the count we had just shipped.** The
same box, minutes apart, read 2.6-3.9% per-group spread at G=40 ("the box looks idle") and 5.1-6.5%
at G=10 ("SOMETHING ELSE WAS RUNNING"). Nothing else was running in either case. The cause is a
single excursion — Herring ran group 9 at 1.19x and 1.28x in the two arms of the *same* run,
ScreenType ran group 1 at 1.16x and 1.17x in both — which is reproducible across arms and therefore
not background load. One 20% outlier divided over 39 samples adds ~3% to a standard deviation and
passes; over the 9 that a default serve leaves, it adds ~7% and fails. `steadiness` now sets the
single slowest group aside and reports it; the archived contended fixture, which has four of nine
elevated, still reads 28%.

`reference/serve_compare.json` carries all eight runs and the per-group timings behind the last
paragraph. Not established: whether the 3.8x holds on datasets whose test sets are large enough to
chunk at 128, where the group count and the chunk count multiply.

#### What this leaves, and what it costs to find out

An ensemble of labellers is the standard escape, and after `scripts/convert_model_weights.sh` there
are four to build one from — `tabicl-v2` (BSD-3), `mitra` (Apache-2.0), `tabpfn-v2` (Apache-2.0 +
attribution) and `orion-bix` (MIT), all verified classifying through the real extension on CPU.
`tabpfn-v2-5` and `tabpfn-v3` also exist and are **non-commercial**, so they are research instruments
and not shippable.

But the result above changes what an ensemble has to prove. Lowering the error *rate* is not the
goal, because the rate was never binding. Six models of one `icl-transformer` family scoring one set
of ROCKET features can be wrong in the same places, and an ensemble that is more accurate while
sharing the failure mode inherits the whole problem. So the measurement is error **overlap** — how
much more often two models are wrong together than independence would give, how many rows no model
gets right, and how far a perfect oracle over them would reach.

One practical note for anyone re-running this: `mitra` costs about **five times** what the others do
per call. It declares `max_features = 100` against `tabicl`'s 512 and the engine covers a
500-feature group by raising the estimator count rather than truncating — so it does read all 500
(verified: it classifies correctly with the only informative feature at index 499, 9 of 10, ~1% by
chance) and pays five times over for it. On 16 vCPU that is ~100 minutes for a 61-row test set, and
it timed out on 28 of 29 datasets at the 30-minute limit that suits the other three.

##### The overlap, measured: the two backbones fail together (2026-08-16)

It cost nothing to answer. `tabpfn-v2` had already been run on 18 datasets at exactly the archived
configuration — 10,000 kernels, G=40, 250 per group, e=1, seed 0, `--test-chunk 128` — and
`phase5_pipeline.py` writes per-row soft labels beside every report. 17 of those datasets also have
`tabicl-v2` soft labels, so the comparison was sitting in `reference/` unread.
`scripts/error_overlap.py`, no pod:

| | mean over 17 datasets |
|---|---|
| `tabicl-v2` alone | 0.7579 |
| `tabpfn-v2` alone | 0.7568 |
| averaging their probabilities | 0.7598 |
| **a perfect oracle over the two** | **0.8080** |
| rows neither gets right | **0.1920** |

**Their errors are not independent, by a factor of 3.74.** Both are wrong on the same row 19.20% of
the time against the 6.86% independence would give — median 3.30x, and every one of the 17 datasets
is above 1 (range 1.87 to 9.52). They disagree on only 11.1% of rows.

So the concern the section above raised is the case that obtains. Averaging buys **+0.0019** over
`tabicl-v2` and +0.0030 over `tabpfn-v2`, which is inside this harness's ~0.005 noise floor, and it
is *behind* picking the better backbone per dataset (0.7651) — though that is itself an oracle over
model choice and not a deployable strategy. Meanwhile a per-row oracle would reach 0.8080. **There
is 4.3 points of headroom between the two models and no combination rule tried reaches any of it**,
because the rows where one is right and the other wrong are exactly the 11% they disagree on, and
the 19% neither gets is a floor no ensemble of these two can go below.

`Earthquakes` is the degenerate case worth naming: both score 0.7482, the oracle is also 0.7482 and
they never disagree. That is the majority-class rate on a heavily imbalanced set — two models
agreeing perfectly because both have given up.

**What this suggests, and does not establish.** Both models read the same 500 ROCKET features per
call. Correlated failure is what you would expect if the *features* — not the models — are what
discards the signal on those 19% of rows, and the ts-features result above already shows 116
statistics beating 10,000 ROCKET features on three of six hard datasets under a ridge. That points
at the feature family rather than the model zoo as the axis with headroom on it. Two backbones of
broadly one lineage cannot separate those explanations: `orion-bix` (MIT, and cheap, unlike `mitra`)
on the same datasets would, since a comparable 3.7x excess across a third architecture is evidence
for the features and a materially lower one is evidence for the models.

##### A third architecture answers it: it is the features (2026-08-16)

`orion-bix` on the same 17 datasets, same G=40, same 250-kernel groups, one 32-vCPU pod. It is a
*cheaper* labeller than the teacher, not a dearer one — 0.83x and 0.92x of `tabicl-v2` on the two
probe datasets, nothing like `mitra`'s 5x — so this cost about 40 minutes.

| pair | both wrong | if independent | **excess** | disagree |
|---|---|---|---|---|
| `orion-bix` vs `tabicl-v2` | 0.2010 | 0.0736 | **3.75x** | 0.1174 |
| `orion-bix` vs `tabpfn-v2` | 0.2022 | 0.0741 | **3.78x** | 0.1175 |
| `tabicl-v2` vs `tabpfn-v2` | 0.1920 | 0.0686 | **3.74x** | 0.1111 |

**Three architectures, three pairs, and the excess is the same to within 1%.** A third model
overlaps the other two exactly as much as they overlap each other. Whatever produces the correlated
failure is not the lineage two of them share — it is what all three are given, which is 500 ROCKET
features per call. `Earthquakes` makes the point in miniature: all three score 0.7482, the oracle is
0.7482, and no pair ever disagrees. Three architectures giving up on the same rows.

Adding the third model makes the ensemble **worse**, not better: averaging falls from 0.7598 to
0.7544, because `orion-bix` is the weakest of the three (0.7371) and averaging pulls toward it. The
oracle rises (0.8080 to 0.8254, now +0.0586 over the best single backbone) and the rows nobody gets
right fall from 0.1920 to 0.1746 — so a third architecture does recover a few rows in principle, and
no combination rule tried reaches them.

**So the model zoo is the wrong axis.** More labellers of this kind cost linearly and buy nothing
an averaging rule can collect; ~17% of these rows are unclassifiable from the features they are all
being handed. The ts-features result above — 116 statistics beating 10,000 ROCKET features on three
of six hard datasets under a ridge — is the direction with headroom, and `phase5` already supports
`--features both`.

Caveats: 17 datasets, one seed, and all three read the same feature construction by design, which is
the hypothesis rather than a controlled variable — the decisive follow-up is one backbone against
*two* feature families, not more backbones. `reference/error_overlap.json`, cubes in
`reference/pergroup_cubes_orion.tar.gz`.

##### The controlled version: change the features, not the model (2026-08-16)

Same `tabicl-v2`, same 17 datasets, same rows, reading `anofox_forecast`'s 116 statistics instead of
500 ROCKET features. One call per dataset rather than forty, because there is no kernel bank to
slice: **17/17 in two minutes.** Six pairs now, and they separate cleanly by what varies:

| pair | what differs | both wrong | if independent | **excess** | disagree |
|---|---|---|---|---|---|
| `tabicl-v2` vs `tabpfn-v2` | model | 0.1920 | 0.0686 | 3.74x | 0.1111 |
| `orion-bix` vs `tabicl-v2` | model | 0.2010 | 0.0736 | 3.75x | 0.1174 |
| `orion-bix` vs `tabpfn-v2` | model | 0.2022 | 0.0741 | 3.78x | 0.1175 |
| `orion-bix` vs `tabicl-v2/ts` | model **and features** | 0.1865 | 0.0824 | **2.67x** | 0.1860 |
| `tabicl-v2/ts` vs `tabpfn-v2` | model **and features** | 0.1824 | 0.0768 | **2.83x** | 0.1732 |
| **`tabicl-v2` vs `tabicl-v2/ts`** | **features only** | 0.1906 | 0.0763 | **2.99x** | 0.1545 |

**One model with two feature families overlaps less (2.99x) than two different models with one
family (3.74x).** That is the controlled comparison, and it settles the question the three-backbone
result could only pose: the representation is doing more of the work than the architecture. Every
cross-feature pair sits at 2.67-2.99x against 3.74-3.78x for every same-feature pair, and
disagreement rises from ~11% to 15-19%.

It moves the bounds too. Against the three ROCKET arms alone, adding the ts arm takes the oracle
from 0.8254 to **0.8483** and the rows nobody gets right from 0.1746 to **0.1517**. A third
*architecture* was worth +0.0174 of oracle; a second *feature family* is worth +0.0229 while being
the weakest arm on the board.

**And averaging still cannot collect it.** The mean of the four probabilities is 0.7619 against
0.7686 for the best single arm per dataset, because the ts arm averages 0.7236 — 3.4 points behind
`tabicl-v2` — and the mean is pulled toward it. Per dataset the ts arm swings hard both ways:
+0.0164 on `Lightning2` and +0.0195 on `ProximalPhalanxTW`, −0.1299 on `Worms` and −0.1200 on
`ArrowHead`.

So the conclusion is not "use both feature families in an ensemble". It is that **feature diversity
is a real axis where model diversity is not, and collecting it needs a selective rule rather than an
average** — which is the mechanism this project already has. Routing escalates on a student's
decision margin; the same margin could choose between representations. That is the experiment this
points to, and it is not run here.

Caveats: 17 datasets, one seed, and `anofox_forecast` is BSL 1.1 — production use permitted,
offering it hosted or embedded to third parties is not — so the ts family is a research instrument
here and never a dependency. `Earthquakes` is unmoved by any of it: all four arms score 0.7482, the
oracle is 0.7482, and no pair ever disagrees. `reference/error_overlap.json`.

##### The selective rule does not collect it either (2026-08-16)

The obvious next move, and the one the section above points at: an average is the wrong rule for arms
of unequal competence, and this project already has the right one. Escalate on a decision margin and
spend the second call only where the first was unsure — except that here the escalation target is a
different **representation** rather than a bigger model, with routing to `tabpfn-v2` as the control
spending the identical budget on a different *model*. `scripts/feature_route.py`, no pod, primary
`tabicl-v2` at 0.7579.

| escalate to | | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| `tabicl-v2/ts` | features | +0.0026 | +0.0001 | −0.0056 | −0.0179 | −0.0344 |
| `tabpfn-v2` | model | +0.0042 | +0.0025 | +0.0006 | +0.0007 | −0.0011 |
| `orion-bix` | model | −0.0022 | −0.0105 | −0.0125 | −0.0177 | −0.0209 |

**Nothing here is significant** — the best cell is +0.0042 at p=0.27, and every other positive is
above p=0.5. Against a per-row oracle at 0.8483, i.e. **+0.0904 over the primary**, a margin-routed
rule collects essentially none of it.

Nor does asking the other question. The curves above only ever consult the primary's own
confidence — "was I unsure?" — never "is someone else surer?". Picking the arm with the largest
margin on each row, which is the standard way to ask the second question and needs no budget at all:

    the primary alone                 0.7579
    surest of the two representations 0.7500  (-0.0079)
    surest of all four arms           0.7548  (-0.0032, 6 better / 7 worse, p=1.00)
    best single arm per dataset       0.7686
    a per-row oracle over the arms    0.8483

**So the headroom is real and confidence cannot reach it.** The mechanism is the overlap number
itself: a margin identifies rows where the primary is unsure, and at 2.99x correlated failure those
are disproportionately rows where *every* arm is unsure and wrong. Cutting the excess from 3.74x to
2.99x shifts those odds without shifting them enough for a confidence signal to exploit — the arms
disagree on 15% of rows, and knowing *that* they disagree is not knowing *which* is right.

One asymmetry survives and is worth recording, because it is the overlap result showing through:
at a 10-20% budget, spending it on a different representation beats spending it on a different model
(+0.0017 and +0.0041 in a like-for-like comparison), and at 50-100% it loses badly (−0.0094,
−0.0234) as the ts arm's 3.4-point competence gap swamps its diversity. Feature diversity helps
exactly where you escalate few rows. None of it clears the noise floor.

**What this closes.** The teacher's ceiling is not reachable by more models, by more feature
families, or by any confidence-based selection among them. Three routes tried, all negative, and the
oracle says +0.09 is sitting there. Reaching it needs a signal that knows which arm is right, and no
arm's own confidence is that signal. `reference/feature_route.json`.

##### Phase 7a: labels do not reach the oracle either, so selection is closed (2026-08-17)

The section above closes by saying the +0.09 oracle "needs a signal that knows which arm is right,
and no arm's own confidence is that signal." This grants the most permissive signal there is — a
learner fitted on **every arm's full posterior vector with access to the real labels** — and asks
whether even that reaches it. `scripts/stack_arms.py`, the same 17 datasets, no pod: all four arms
were already archived, so this is a fit over files on disk.

Cross-validated **within the test rows**, which makes it an upper bound rather than a design: it
trains on labels a deployed system would not have. That is what a gate wants. A negative here
cannot be blamed on the rule being too weak, because no rule gets more than this one.

| | accuracy | vs primary | of the oracle | wins | p |
|---|---|---|---|---|---|
| primary (`tabicl-v2` on ROCKET) | 0.7579 | — | — | — | — |
| mean of the four posteriors | 0.7619 | +0.0039 | 4.3% | — | — |
| best arm, chosen honestly by fold | 0.7481 | **−0.0098** | — | — | — |
| stacker: learned arm weights | 0.7564 | −0.0015 | −1.6% | 7/8 | 1.00 |
| **stacker: logistic on all posteriors** | **0.7632** | **+0.0053** | **5.8%** | 11/6 | 0.33 |
| stacker: gradient boosting | 0.7158 | −0.0421 | −46.6% | 3/13 | **0.02** |
| per-row **oracle** over the arms | 0.8483 | +0.0903 | 100% | — | — |

**Three learners, spanning the whole flexibility range, and the answer is the same.** The best
collects **5.8% of the oracle at p=0.33**. The lowest-variance form — one weight per arm, four
parameters — is flat. The most flexible is *significantly worse* than doing nothing at all
(−0.0421, p=0.02), which is a 30-row test split overfitting exactly as it should be expected to.
There is no capacity sweet spot in between: both ends fail.

**So the direction is closed, and closed more firmly than before.** Averaging failed, margin-routing
failed, picking the surest arm failed — and now a supervised learner with the labels in hand fails
too. The oracle is not a target that better *selection* can reach; the information about which arm
is right is not present in the arms' posteriors. Four routes tried, four negatives, one afternoon
and no pod time for the last one.

Two things worth recording about how this was measured, since both would have manufactured a result.
The obvious control — "pick the best arm, judged per fold" — came out at 0.7481, **worse than simply
always using the primary**, because choosing an arm on a handful of folds is noisy enough to lose
ground. Scoring the stacker against that control would have credited it +0.0151 for avoiding the
control's own mistakes; the primary is the honest floor because it requires no choice at all. And
taking the best of the three learners per dataset gives +0.0158 at p=0.049 — the one number here
that looks like a finding, and it is an oracle over learners, chosen on the rows it is scored on. It
is printed and labelled as such rather than omitted, because leaving it out invites recomputing it.

**What survives is concatenation, not selection.** If the arms cannot be chosen between, the
remaining move is to stop treating them as arms: feed both feature families to *one* model.
`--features both` (500 ROCKET + 116 statistics = 616 columns, inside `tabicl-v2`'s 512-per-estimator
budget at G=40) is already archived for the six hard datasets:

| dataset | rocket | both | |
|---|---|---|---|
| ScreenType | 0.5200 | 0.5547 | **+0.0347** |
| InlineSkate | 0.4909 | 0.5145 | +0.0236 |
| RefrigerationDevices | 0.5573 | 0.5787 | +0.0214 |
| Herring | 0.6406 | 0.6562 | +0.0156 |
| Haptics | 0.5552 | 0.5390 | −0.0162 |
| MiddlePhalanxTW | 0.6104 | 0.5844 | −0.0260 |

**+0.0088 mean, 4 wins to 2** — six datasets and one split, so this establishes nothing on its own
(the sign test is p≈0.69 and the resample pilot puts split luck at sd 0.0173). But it is the only
positive direction left standing, it needs no selection rule, and it is the case RESULTS.md
predicted when it noted that "a ridge on 500 standardised random features drowns 116 statistics, and
an in-context model need not." `reference/stack_arms.json`.

#### The context cache is 7x on repeated calls, and 2.5x slower on the first (2026-08-16)

The 4-11x this file has been quoting for the context cache came from a PyTorch prototype in the
exporter README. This is the same thing measured **through DuckDB**, on real `tabicl-classifier-v2`
weights, with the unreleased `anofox_tabfm_context_cache` build from
[#40](https://github.com/DataZooDE/anofox-tabfm/pull/40).

#37's escalation shape: 375 labelled context rows, 500 features, 22 query rows per call, and twelve
calls arriving one after another against the **same** context — which is what `--test-chunk` does.
Both arms in one DuckDB session so they see identical data. 16 vCPU pod, `scripts/pod/context_cache_cpu.sh`.

| | cache off | cache on | |
|---|---|---|---|
| call 0 (context is new) | 7.39 s | 18.62 s | **0.40x** — 2.5x slower |
| calls 1-11 (context reused) | 73.00 s | 10.17 s | **7.18x** |
| all twelve | 80.39 s | 28.79 s | 2.79x |

Per call that is ~6.6 s against ~0.92 s. The first call pays for a context it has not reused yet —
the prepare pass plus one pass to score the 375 context rows — and costs an extra 11.2 s; each
later call saves ~5.7 s, so **the break-even is between the third and fourth call**. One-shot
scoring is worse off, which is why the setting is off by default.

Three runs of the steady state gave 7.89x, 5.47x and 7.18x; the 7.18x is the twelve-call one and
the best sampled. The cache-hit calls are short enough that contention moves them around by ~50%,
while the cache-off calls sit steadily at 6-7 s.

**The answers are identical where it matters.** Over 264 scored rows: zero label disagreements and a
maximum `yhat_score` difference of **0.0** — not "close", the same. The 375 context rows also agree
on every label but differ by up to 0.338 in score, which is the documented consequence of the query
half having no label path: a context row is no longer scored knowing its own label. Both numbers
reproduced exactly across all three runs.

What this changes here: nothing yet, and not what we assumed. Our per-group arms each carry their
own context, so they would land on the cold side of that table and get *slower*. The pattern that
gains is one context scored repeatedly — `--test-chunk` runs, and any serving path that fits once
and scores many times. That is worth having, but it is a narrower win than "4-11x on the escalation
case" implied.

*(Half right, and measured the next day: the per-group arms **do** gain, because within a group
every chunk shares one context. See "The context cache is worth 1.85x, 1.13x or −56%" below.)*

Three separate runs of this measurement produced confident-looking results with nothing running at
all: a signature-blocked extension (`-unsigned` missing) reported 1.00x, a CDN-cached copy of the
script silently re-ran the old version, and a binder error gave six 0.01 s "calls". Two arms of zero
divide to a perfectly plausible 1.00x. The script now asserts the extension loads and carries the
setting before either arm runs, and marks each arm in the output rather than counting backwards from
the end — an off-by-one there had already produced one wrong table that looked entirely reasonable.

#### The context cache is worth 1.85x, 1.13x or −56%, and chunks per group decide which (2026-08-16)

The measurement above is a call pair. This is the same setting through the **whole phase-5
pipeline** — raw series, 40 groups, `rocket_transform`, `tabfm_classify`, argmax — which is the
number that decides whether to use it. `scripts/pod/context_cache_phase5_cpu.sh`, G=40, e=1,
`--test-chunk 128`, `tabicl-v2` registered from a local model directory carrying the split pair.

A phase-5 run makes 40 group calls and **every group's context is different** — different kernels,
different feature columns — so per group the cache starts cold. It can only win *inside* a group,
across the chunks that share one support set. That makes chunks-per-group the whole question, so
four datasets were run either side of the predicted break-even rather than one:

| Dataset | Context rows | Test rows | Chunks/group | Cache off | Cache on | | Rows | Disagree |
|---|---|---|---|---|---|---|---|---|
| ItalyPowerDemand | 67 | 1029 | **9** | 1608.5 s | **870.2 s** | **1.85x** | 1029 | **0** |
| OSULeaf | 200 | 242 | 2 | 528.9 s | 469.8 s | 1.13x | 242 | **0** |
| Coffee | 28 | 28 | 1 | 86.7 s | 118.8 s | **0.73x** | 28 | **0** |
| Beef | 30 | 30 | 1 | 92.2 s | 144.0 s | **0.64x** | 30 | **0** |

Classify seconds, not total: the cache cannot touch the ROCKET transform, which is ~6 s of a
1600 s run and would only dilute the number.

**It is the same answer, faster.** Zero label disagreements on all 1329 test rows, and all four
accuracies identical between arms to full precision. The `memcmp` context match survives DuckDB's
parallel scan of `train_cur`, which was the one thing between this design and a null result — the
table is filled once per group and never touched while its chunks run, so the support rows really
do arrive byte-identical every time.

**Chunks decide it, not context size, and one dataset could not have shown that.** OSULeaf carries
three times ItalyPowerDemand's context (200 rows against 67) and a far better context-to-query
ratio (1.56 against 0.52), so on "how much is there to amortise" it should win comfortably. It
reuses twice instead of nine times and gets 1.13x against 1.85x. The two variables point opposite
ways and the chunk count wins.

**1.85x is well under the call-pair 7.18x, and the gap is the shape.** That bench ran 375 context
rows against 22 query rows — a ratio of 17. ItalyPowerDemand runs 67 against 128, a ratio of 0.52.
The cache saves the context encode, so less context per unit of query is less to save. Working
backwards from the single-chunk rows, the cold call costs ~1.4-1.6x a combined call rather than the
bench's 2.5x, which puts the warm call near 2.4-3.6x rather than 7.2x. The mechanism transfers; the
magnitudes do not.

**The single-chunk loss is measured, not predicted.** Beef and Coffee were run precisely because a
recommendation *not* to switch something on deserves the same evidence as the recommendation to
switch it on. At one chunk per group every group pays a cold prepare and gets nothing back: 56% and
37% slower. That is the largest bucket in this archive — of 125 archived chunked runs, **45 are
single-chunk and 49 are two-chunk**, so 75% of runs sit at 1.13x or worse:

| chunks/group | 1 | 2 | 3 | 4-9 | 36 |
|---|---|---|---|---|---|
| runs | **45** | 49 | 21 | 9 | 1 |
| hours | 1.67 | 4.95 | 2.94 | 1.34 | 0.31 |

So the honest summary is that the cache is a real win on a rare shape. Applying the measured
speedups to the archive, a full sweep pass moves from 11.2 h to about 10.4 h — 7%, not 85%. It
should stay opt-in and be switched on per dataset, not globally, and `--context-cache` in
`phase5_pipeline.py` is off by default for that reason. The break-even is between two and three
chunks per group, which at `--test-chunk 128` means roughly **384 test rows**.

Two traps worth recording. The flag is **silently inert** unless the model directory carries all
four split artifacts beside the combined graph — the engine falls back and the run still reports a
normal accuracy, so the timing would be filed as "with the cache" having cached nothing; the
pipeline now refuses to start instead. And the mechanism check in the script originally compared
the lowest-chunk dataset against the highest and printed "as predicted" whenever the second came
out higher — on the Beef/Coffee run, where both have one chunk, it duly manufactured a verdict from
two datasets that differ only by luck. It now reports "not testable" when there is no gradient.

*Also settled on the way: `--register-model-dir` had never been exercised on CPU — both archived
reports show `registered_graph: null`, it existed for the CUDA ScatterND workaround. All four
cache-off arms reproduce their archived accuracies exactly (ItalyPowerDemand 0.9718172983 to ten
decimals), so a locally registered split-export model is numerically identical to the shipped
`tabicl-v2` on this path.*

Environment: RTX 6000 Ada community pod in US, `--device cpu`. CPU pods were unavailable at 16, 32
and 64 vCPU, so this is a GPU box running the CPU path; the comparison is paired within one host,
which is what it needs to be. **`nproc` reported 112 and the cgroup quota was 11.9** — pools sized
4 DuckDB x 2 ONNX from the quota, because sizing from `nproc` would have put 112 threads on 12
cores, which is the failure this file already records twice. Reports in `reference/ctxcache/`.

#### Split luck is 8x the dataset effect, so buy breadth (2026-08-16)

The noise floor has been this file's binding constraint for two days: every recent result lands
inside it, and the entry above says fixing that "is a campaign rather than a run". **This is the
run that sizes the campaign**, and it changes what the campaign should be.

`scripts/pod/resample_power.py`, 16 unsaturated datasets x 5 resamples x 2 arms = 160 runs, 57
minutes on a 32-vCPU pod. Arm A is G=40 at 250 kernels per group; arm B is **exactly its first ten
groups** — kernel *i* is a pure function of `(seed, i)`, which is the same property this file uses
to read its group sweep off archived cubes. That is RESULTS.md's own `10,000 kernels, G=10` cell,
which measured **−0.0033** on the archived single split. Both arms see the same resample, so every
difference is paired and the dataset's own difficulty cancels.

| | variance | sd | |
|---|---|---|---|
| **within** — same dataset, same configs, different split | 0.000300 | **0.0173** | resamples shrink this |
| **between** — the effect genuinely differing by dataset | 0.000037 | **0.0061** | only more datasets touch this |

**Split luck is 8x the dataset effect, and this contradicts what the entry above predicted.** The
argument there was that the sign flipping with the dataset subset pointed at between-dataset
heterogeneity, which no amount of resampling can touch. For this comparison that is wrong: `within`
dominates, so resamples *are* the right purchase — but so is breadth, and breadth is cheaper:

| datasets | resamples needed | total runs |
|---|---|---|
| 16 (this pilot) | 23 | 736 |
| 24 | 8 | 384 |
| 28 | 6 | 336 |
| **40** | **4** | **320** |
| 60 | 2 | 240 |

The paper's protocol at this project's usual breadth — 24 datasets x 30 resamples — is 1,440 runs.
**40 datasets x 4 resamples is 320 and resolves the same 0.005**, because more datasets shrink
*both* terms while more resamples shrink only one. That is the practical finding: 4.5x less pod
time than copying the protocol, and it comes from measuring where the noise lives rather than
assuming.

**The single split had the sign backwards.** Over 16 datasets and 5 splits the mean delta is
**+0.0025** — ten groups scoring *higher* than forty — against the archived single split's −0.0033.
It is 0.94 SE from zero, so the honest statement remains "not distinguishable from zero", which is
exactly the point: the archived −0.0033 was never distinguishable from zero either, and it was the
opposite sign. Per-dataset sd across five splits alone runs 0.0040 to 0.0333; Beef's +0.0267 mean
sits on an sd of 0.0279.

*Caveat that keeps this number honest: arm B is a strict PREFIX of arm A, so the two arms share
their first ten groups and are more tightly coupled than two unrelated configurations. The
`within` measured here is therefore a floor for nested comparisons and must not be quoted at
unrelated ones — a kernel-bank change, or a different backbone, would have more.*

**Five concurrency bugs, in one session, all the same bug.** Every one is a process reading a
machine-wide number and assuming it is alone, and PLAN.md already records two prior instances:

| | how it announced itself |
|---|---|
| ONNX pools sized from `nproc` | the GPU host said 112 cores and had 11.9 |
| shared `data/phase5/<dataset>` | "No magic bytes found at end of file" — a parquet read mid-write |
| `memory_limit` at 70% of the cgroup, per run | **38 of 160 runs at exit −9**, no error, no traceback |
| the dataset cache, on first download | one run dead in 1.2 s on a half-written archive |
| `sweep.py` had the first three too | nothing yet — latent, and it corrupts rather than crashes |

The memory one is the expensive one: at `--jobs 6` on a 64 GB cgroup, six runs each claimed 44.8
GB. Only the driver knows the job count, so the driver now divides. The `sweep.py` case is the one
worth remembering — `predictions.json` lives in the shared directory and is read back to compute
accuracy, so two concurrent seeds can each report the other's number with nothing crashing. No
archived result came through that path, so it is latent rather than historical.

And the first diagnosis of the OOM was wrong in a way worth recording: ONNX Runtime prints tens of
thousands of `Trying to register schema` lines, so the captured stderr tail was always that. The
failure read as `defs.cc line 927`, pointing at a schema registration that was perfectly fine. The
driver now filters that noise and keeps the exit code, which is what says −9.

*The pilot's own analysis also had to be fixed before it could be trusted: run against one dataset
it reported "the between-dataset term dominates, and this comparison cannot be resolved by
resampling at any affordable scale." Nothing supported that. `statistics.variance` of one value is
NaN, NaN fails every comparison, so every plan came back unreachable and the code printed the
sentence it prints when the floor is genuinely too high — the most confident line in the output was
the one with no data behind it. It now refuses to size a campaign below two datasets.*

Reports in `reference/resample_pilot/`. 159 of 160 runs completed.

#### `anofox_tabfm_max_memory` guards residency, not allocation (2026-08-16)

Upstream [#36](https://github.com/DataZooDE/anofox-tabfm/pull/36) adds a setting that reads `VmRSS`
from `/proc/self/status` and refuses a predict call above a ceiling. It is merged and in **no
released build** — the community extension is pinned at 2026.08.14 — so this was measured against
the CI artifact from `main` at `e5021d9`, which needs no build: the pipeline uploads a
`linux_amd64` extension per run, and it matches our DuckDB v1.5.5.

**#36 is ours, and its own comment already said this.** The setting was written here and the code
carries the sentence "a watchdog on CURRENT resident memory, not an estimate of what the call about
to run will cost". So the finding below is not a surprise discovered in someone else's work — it is
the measurement of what that sentence costs in practice, and the first framing of it in this file
as a limitation found in upstream code was wrong.

Tested where it should matter most. `SemgHandMovementCh2` at `--test-chunk 128` on a 32 GB
container, the failure that cost several sessions and one withdrawn diagnosis:

| ceiling | outcome |
|---|---|
| none | `FAILED after 43.4s (exit -9)` — the kernel, no message |
| 28 GB | `FAILED after 38.9s (exit -9)` |
| 24 GB | `FAILED after 23.3s (exit -9)` |
| 18 GB | `FAILED after 20.7s (exit -9)` |
| 16 GB | `exit 1` with a message naming the number, the ceiling and three remedies |

Only the last one is clean, and the reason is the mechanism rather than the tuning: **the check runs
before a call and compares the process's current RSS against the ceiling.** Resident memory here is
~17.9 GB before the classify call — the ROCKET features and the loaded model — so a 16 GB ceiling is
already exceeded and the call is refused, while any ceiling above 17.9 GB passes the check and the
call then allocates its way past the 32 GB cgroup cap and is killed exactly as before.

So it is a **residency** guard, not an **allocation** guard, and its own error text says as much
when it suggests `tabfm_unload(...)`: it is built to catch memory accumulated across calls and
models, which is a real failure mode and not ours. Ours is one call that is too large, and
`--test-chunk` remains the only lever for that. The setting is still worth having — an `exit 1`
carrying `resident memory (17910112256 bytes) is already at or above anofox_tabfm_max_memory
(16000000000 bytes)` is a different world from `exit -9` and an empty stderr — but it does not make
a too-large call safe, and a ceiling low enough to fire on this dataset would refuse legitimate work.

`phase5_pipeline.py --tabfm-max-memory` emits it, opt-in and only when asked: the default upstream
is empty, and a released build that has never heard of the parameter fails the whole script on the
`SET`. `scripts/pod/max_memory_cpu.sh` runs the arms in the only order that proves anything — the
no-ceiling arm must still die, or a clean refusal downstream would mean nothing.

The follow-up is upstream as [#39](https://github.com/DataZooDE/anofox-tabfm/pull/39): sample RSS
either side of the forward pass, record the growth against `(model, device, T, H)`, and refuse the
next call of a shape already measured to breach the ceiling. No formula and no per-model constant —
it works because the failure repeats, once per group and once per chunk. It cannot protect the
first call of a new shape, which is the limit of measuring rather than modelling.

##### Most of that +0.09 was never there (2026-08-16)

Three negative results measured against one number deserve a look at the number. **An oracle over N
arms rises mechanically with N**: on a row where every arm is near-guessing, "someone was right"
approaches certain, and the row is counted as recoverable headroom that nothing could
systematically recover. Decomposing it — of the 626 rows the primary gets wrong, how many of the
other three arms are right:

| alternates right | rows | | |
|---|---|---|---|
| 0 of 3 | 401 | **64.1%** | no arm gets it; the floor |
| 1 of 3 | 125 | **20.0%** | needs picking one dissenter out of three, with no signal to do it |
| 2 of 3 | 76 | 12.1% | a majority vote finds these |
| 3 of 3 | 24 | 3.8% | a majority vote finds these |

So two thirds of the primary's errors are unrecoverable by any rule over these arms, and a further
fifth are recoverable only by identifying a single correct dissenter — which is precisely the thing
the confidence experiments above establish there is no signal for. **Only 15.9% of the primary's
errors are reachable by consensus**, and the cheapest rule that exploits consensus, a majority vote
of all four arms, gets **+0.0024** — because it also overturns rows the primary had right.

And on binary datasets the oracle is not a competence measure at all. **If the primary is wrong on a
2-class problem, any arm that merely disagrees is automatically right.** Six of these 17 datasets
are 2-class, and there 43.7% of primary-wrong rows count as "recoverable" — a disagreement rate
wearing a competence costume, and the arms disagree on 11-19% of rows by construction.

**This reframes the three negative results as correct rather than as failures of technique.** The
accessible headroom is not +0.09; it is about +0.002 to +0.004, which is exactly what every rule
tried returned and exactly what this harness cannot resolve. The ensemble line closes here, and it
closes cleanly: not "we could not find the rule" but "the target was mostly an artefact of counting
four chances to be right".

This is an analysis and not a serving rule, the same distinction `route_serve.py` exists to make:
the budget is spent by sorting a whole test set and taking its least-confident fraction, which
nothing answering a request can do. That makes these numbers the best a perfectly-placed threshold
would get — which only strengthens a negative result.

Two things about the run itself, since both cost real time. `orion-bix` is published as a torch zip
whose pickle stream uses an opcode the checkpoint reader does not implement, so `tabfm_download`
reports success and inference then dies with `unsupported pickle opcode 0x65`;
`scripts/convert_model_weights.sh` is what fixes it, and RESULTS.md's "all four verified classifying
through the real extension" was always a post-conversion claim. And `Earthquakes` and `Computers`
both died with an *empty* stderr while four shards ran concurrently on a 64 GB box, then both
completed in 6 minutes each run alone — the same concurrency-versus-memory trade that left five
datasets without cubes for days.

### The second feature family, measured on all 112 datasets (2026-08-14)

`anofox_forecast` -- the same vendor's forecasting extension -- exposes `ts_features_by(table, group,
time, value)`, which returns one row per series with **116 numeric feature columns**: the tsfresh
catalogue of statistics, computed in-database. (`ts_features_list()` has 117 rows, but those are
feature *definitions*, some parameterised; the emitted table carries 116, read from `DESCRIBE`.) That
is exactly the shape our classifier path consumes, so it is a second feature family for the cost of
an `unnest ... WITH ORDINALITY`.

Same head (`RidgeClassifierCV`) throughout, so the features are the only variable.
`scripts/ts_features_screen.py`, records in `reference/ts_screen/`, analysis in
`reference/ts_features_analysis.json`. 112 equal-length univariate UCR datasets, 32 vCPU, 53 minutes,
0 failures.

**This section replaces a six-dataset version that was wrong in its headline.** That screen reported
"3 wins from 6, mean +0.0025" and I described the two families as complementary. Six datasets chosen
for being hard is a sample selected for the regime where the statistics do well:

| | 6 hard datasets | **all 112** |
|---|---|---|
| ts vs rocket | 3 wins, mean **+0.0025** | **12 wins of 112**, mean **-0.0795**, median **-0.0488** |

At archive scale the 116 statistics **lose to 10,000 ROCKET features by about 8 accuracy points on
average**, with losses reaching -0.70 (`PigCVP`: 0.2308 against 0.9327). The wins are real but rare
-- RefrigerationDevices +0.0720, MiddlePhalanxTW +0.0714, ScreenType +0.0693, and interestingly
`PowerCons` +0.0611 and `FordA` +0.0591 -- twelve datasets out of a hundred and twelve.

**And the mechanism I proposed for those wins is wrong.** I claimed the statistics help where ROCKET
is weak, because max/PPV pooling cannot express repetitiveness and the winning datasets were
quantised appliance traces. Measured:

    corr(rocket accuracy, ts - rocket) = +0.015

No relationship at all. Broken out, the hard end is where the statistics do *least badly* rather
than best, and two of their largest wins are on datasets where ROCKET already scored 0.93 and 0.94:

| ROCKET accuracy band | n | mean ts - rocket | mean both - rocket |
|---|---|---|---|
| < 0.60 (hard) | 10 | -0.0425 | +0.0009 |
| 0.60 - 0.80 | 19 | -0.0861 | +0.0029 |
| 0.80 - 0.95 | 43 | **-0.1014** | +0.0024 |
| >= 0.95 (saturated) | 40 | -0.0621 | +0.0001 |

#### Selection works at 112 and did not at 6

The point of widening was to make feature selection possible. The null is unchanged -- if a dataset's
top-K were K names drawn uniformly from N, each feature appears in Binomial(D, K/N) of the D lists --
but its power comes entirely from D. At D=6 the null mean was 14 of 116 features at ">= 2 lists" and
the screen produced 11, below chance. At D=112 the null mean is 11.6 appearances per feature, and
**22 of 116 clear Benjamini-Hochberg at FDR 0.05** (19 on the first 112-dataset run; the shortlist is stable under a 110/112 resample) (BH across all 116, because 116 hypotheses at 0.05
buys about six free winners on their own):

| feature | count | chance | p |
|---|---|---|---|
| `fft_coefficient_6_abs` | 32 | 11.6 | 5.5e-08 |
| `fft_coefficient_1_abs` | 30 | 11.6 | 6.5e-07 |
| `fft_coefficient_5_abs` | 30 | 11.6 | 6.5e-07 |
| `number_peaks` | 29 | 11.6 | 2.1e-06 |
| `permutation_entropy` | 29 | 11.6 | 2.1e-06 |
| `binned_entropy` | 26 | 11.6 | 5.4e-05 |
| `sample_entropy` | 22 | 11.6 | 2.1e-03 |
| `longest_strike_above_mean` | 21 | 11.6 | 5.2e-03 |

**14 of the 19 survivors are Fourier coefficients.** The entropy hypothesis is partly vindicated --
`permutation_entropy`, `binned_entropy`, `sample_entropy` and `number_peaks` all survive -- but the
dominant family is spectral, which the six-dataset story did not predict. BH controls false
discoveries and not false negatives, so this is a shortlist worth implementing rather than a claim
that the other 97 are useless; a feature that matters on three datasets of a hundred is invisible to
this test at any D.

#### Naive concatenation is the wrong combination rule

`both` against rocket over 112 datasets: **30 wins, 56 exact ties, mean +0.0015, median 0.0000**. It
almost never hurts (worst -0.0137 on `Lightning7`) and occasionally helps (+0.0561 on `FordA`), but
the 56 *exact* ties are the tell: the ridge is usually ignoring the 116 columns outright.

That is a fact about the ridge rather than about the features. A single global L2 penalty over 10,116
standardised columns cannot shrink two blocks differently, and the statistics are outnumbered 86 to
1, so their coefficients are pushed to nothing before they can contribute. Through the in-context
model on the six hard datasets the same concatenation gained +0.0089 (4 of 6), which is consistent
with the drowning being the ridge's doing.

So the open question is not "do the two families combine" but "does any combination rule let the
smaller block contribute". Rules worth testing, all cheap once the features exist: per-block scaling
so each family's total variance is comparable, a separate penalty per block, and stacking two
independently-tuned heads and combining their decision values.

**Licence, and why nothing depends on this.** `anofox_forecast` is BSL 1.1; its Additional Use Grant
permits production use but forbids offering the work "to third parties on a hosted or embedded
basis", converting to MPL 2.0 after five years. So it is used strictly as a black box to find out
which statistics matter, and the 19 above would be reimplemented from the tsfresh catalogue (MIT) or
the underlying mathematics -- these are standard statistics, not DataZooDE's invention -- rather than
from reading their Rust. `rocket` must not depend on it.

**What the 112-dataset result means for that work.** Implementing 19 features to buy +0.0015 mean
under a ridge is not justified. What keeps it open is conditional rather than general: the
combination never loses much, gains several points on a minority of datasets, and has not yet been
tried with a combination rule that gives the smaller block a chance.

### Can a smarter combination rule stop ROCKET drowning the statistics? (2026-08-14)

Naive concatenation tied *exactly* with ROCKET on 56 of 112 datasets, which is one global L2 penalty
being unable to shrink two blocks differently while the smaller is outnumbered 86 to 1. Four rules,
110 of 112 datasets (the two exceptions below), same `RidgeClassifierCV` head throughout:

    both          naive concatenation
    both_scaled   each block divided by sqrt(its column count) AFTER standardising, equalising the
                  total variance each family contributes to the shared penalty
    both_tuned    that block weight chosen by stratified 5-fold CV on the TRAIN split only
    stack         two independently tuned ridges, decision values z-scored and mixed with a weight
                  also chosen on train; no shared penalty, so neither block can drown the other

| arm | wins | ties | losses | mean | median | best | **worst** |
|---|---|---|---|---|---|---|---|
| ts alone | 12 | 5 | 93 | −0.0792 | −0.0488 | +0.0720 | −0.7019 |
| **both** | 28 | 56 | 26 | **+0.0013** | 0.0000 | +0.0561 | **−0.0137** |
| both_scaled | 26 | 15 | 69 | −0.0100 | −0.0039 | +0.0960 | −0.1106 |
| both_tuned | 28 | 49 | 33 | −0.0010 | 0.0000 | **+0.1120** | −0.1500 |
| stack | 32 | 25 | 53 | −0.0112 | 0.0000 | +0.0717 | −0.1886 |

**No smarter rule beats naive concatenation on average, and the naive one has by far the smallest
downside.** What the smarter rules do is raise the ceiling: the best single gain goes from +0.0561 to
+0.1120, and `stack` wins on more datasets than any other arm (32) while also losing on more (53).

**The drowning was mostly the ridge being right.** What CV picked is the clearest evidence:

    block weight   0.5 on 54 of 110 datasets   (the FLOOR of the grid)
    stack alpha    0.0 on 56 of 110            (literally "use ROCKET only")

On half the archive the correct answer is to ignore the 116 statistics, and cross-validation on the
training split finds that unaided. So the 56 exact ties were not a defect to be engineered away --
they were the penalty doing approximately the right thing. Where the rules go wrong is the other
half: CV sometimes picks a non-zero weight that does not survive the test split, which is how
`both_tuned` reaches −0.1500 and `stack` −0.1886.

**One of those failures is the harness's fault, not the method's.** The weight grid starts at 0.5 and
CV chose that floor 54 times, so the optimum frequently lies *below* the grid -- and 0.0 is not on
it, so `both_tuned` cannot switch the statistics off and fall back to ROCKET. Adding 0.0 would bound
its worst case by ROCKET's own. `stack` does have 0.0 available and still reaches −0.1886, so that
one is genuine CV overfitting on small training splits rather than a grid problem.

#### Where this leaves the second feature family

**19 of 110 datasets have some combination beating ROCKET by more than 0.02**, and the winning arm
differs each time:

| dataset | rocket | best arm | gain |
|---|---|---|---|
| ScreenType | 0.4773 | both_tuned 0.5893 | **+0.1120** |
| CinCECGTorso | 0.8268 | both_scaled 0.9051 | +0.0783 |
| OliveOil | 0.9000 | stack 0.9667 | +0.0667 |
| PowerCons | 0.9333 | stack 0.9944 | +0.0611 |
| FordA | 0.9409 | both_scaled 1.0000 | +0.0591 |
| BeetleFly | 0.9000 | stack 0.9500 | +0.0500 |
| ArrowHead | 0.8229 | stack 0.8686 | +0.0457 |

Picking the best arm per dataset with an oracle gives **+0.0116 mean over ROCKET** (0.8710 against
0.8593), and ROCKET alone is still the best arm on 57 of 110. That +0.0116 is the honest size of the
prize, and it is only reachable with per-dataset model selection -- which is legitimate to do on the
training split, and is what `both_tuned` and `stack` already attempt imperfectly.

So the answer to "is there a smarter way to combine them" is **yes for a minority of datasets and no
in general**. Naive concatenation is the right default: +0.0013 mean and a worst case of −0.0137. The
gains live in per-dataset selection, not in a single better rule.

**Two datasets are missing and the reason is a design flaw worth recording.** `Crop` (24,000 series,
24 classes) and `ElectricDevices` (16,637 series) were still running after 85 minutes and the pod was
terminated with 110 of 112 done. The CV arms fit 7 weights x 5 folds plus 7 alphas x 5 folds x 2
heads -- around 105 ridge fits -- and `RidgeClassifierCV` on n≈8,000 by p=10,116 with 24 one-vs-rest
problems is expensive enough that this multiplies into hours. The tuning cost should scale with
`n_train * n_classes` and does not; a fold subsample or a coarser grid on large datasets would fix
it. 110 of 112 does not change any conclusion above, but the two omitted are both large-n datasets
and that is not a random 2% .

### The id-recovery key: three wrong answers, all of them the same wrong answer

`anofox_tabfm` echoes back only the target and the columns named in `features`, so a plain `id`
column is dropped and scored rows must be rejoined to their ids on feature values. The key was
`f0` alone. It measured **zero collisions across all ten datasets of the original subset**,
ECG5000's 4500 rows included — which is precisely why it survived — and then collided on two of the
first six hard datasets tried, scoring rows by up to 75 and 80 groups instead of 40.

Then it was widened to four columns, which fixed neither dataset. Then to sixteen, which fixed
InlineSkate and left **five distinct ScreenType series still sharing a key**. Three guesses, each
one the previous guess with a bigger number, and the third was still wrong.

The key is now the **entire 500-column feature vector**, as a `DOUBLE[]`. That is not a fourth
guess; it is the end of the question. Two rows share it only if they *are* the same feature vector,
and identical vectors get identical predictions, so collapsing them (`any_value(proba)` under
`GROUP BY grp, id`) is exact rather than merely tolerable. Verified on v1.5.5 at the real width
before being relied on: a `DOUBLE[]` equality plans as a `HASH_JOIN` rather than a nested loop, the
3890-byte key survives a `PREPARE`, and DuckDB holds `NaN = NaN` so a non-finite feature cannot
silently drop a row out of the join.

The two failure modes were never the same problem, which is why one fix kept not covering both:

| | test rows | distinct series | cause |
|---|---|---|---|
| ScreenType | 375 | 375 | genuine feature collisions — quantised electricity data makes ROCKET's max/PPV coincide across *different* series |
| InlineSkate | 550 | 521 | 29 byte-identical series — **no** key of any width can separate these, and none should |

The collision column would now be a tautology, so it reports duplicate test series instead —
descriptive, not asserted. It reads 29 on InlineSkate and **0 on ScreenType**, confirming after the
fact that ScreenType's five were genuine key collisions and not duplicates. It is also `max()` and
not `sum()`: every group writes the same count, so summing published InlineSkate's 29 as 1160.

Worth noting what caught the original bug: not the accuracy, which looked plausible. The
row-alignment assertion counted 15,070 group-rows where 375 x 40 = 15,000 were expected. And worth
noting what *nearly* didn't — the assertion only checked `min_groups_per_row`, and the fan-out
moved `max` (to 75 and 80) while `min` stayed at 40. The proxy collision count is what actually
fired. `max_groups_per_row` is now checked directly.

ScreenType's accuracy is **0.5200 both before and after** the fix. The five contaminated rows
happened not to change the answer. The number was right by luck and is now right by construction,
and the difference between those two is the entire point of the assertion.

## Phase 5 — the whole pipeline in DuckDB

`scripts/phase5_pipeline.py`. Raw series → `rocket_transform` → 500 scalar columns →
`tabfm_classify` → average `proba` → argmax, all in the database.

GunPoint, G=40, e=1:

| | |
|---|---|
| Accuracy (C++ features) | **0.9933** |
| Accuracy (Phase 3, Python features) | 0.9933 |
| **Per-row prediction agreement** | **150/150** |
| Wall clock | 261.5 s |

The per-row number is the claim worth making. Equal accuracy is the weaker statement — two runs
can match on accuracy while disagreeing about which rows they get right, so equal accuracy is
consistent with the transform being subtly wrong. Identical predictions on every row is what
says the C++ port is interchangeable with the oracle in the real pipeline, not merely conformant
on fixtures.

The loadable extension was also verified against the **stock upstream v1.5.5 CLI** in `tools/`,
not only the shell built here — the check that the pinned ABI genuinely matches rather than
being self-consistent.

### Breadth: nine datasets on a pod, G=40, e=1, `tabicl-v2`

RunPod CPU instance, 16 vCPU, `--test-chunk 128`, `memory_limit` from the cgroup. Environment
tuple in `pod_doctor.json`; each report carries its own `environment` block and a `caveat` of
`null`, meaning it observed itself running containerised rather than asserting it.

| Dataset | Test rows | Channels | Timepoints | Accuracy | Wall clock | Local run |
|---|---|---|---|---|---|---|
| BasicMotions | 40 | **6** | 100 | 1.0000 | 79 s | 1.0000 |
| Coffee | 28 | 1 | 286 | 1.0000 | 64 s | 1.0000 |
| Trace | 100 | 1 | 275 | 1.0000 | 133 s | 1.0000 |
| GunPoint | 150 | 1 | 150 | 0.9933 | 185 s | 0.9933 |
| SyntheticControl | 300 | 1 | 60 | 0.9867 | 653 s | 0.9867 |
| FaceFour | 88 | 1 | 350 | 0.9773 | 87 s | 0.9773 |
| ItalyPowerDemand | **1029** | 1 | 24 | 0.9718 | 1010 s | — |
| OSULeaf | 242 | 1 | 427 | 0.9711 | 355 s | 0.9711 |
| Beef | 30 | 1 | 470 | 0.7667 | 67 s | 0.7667 |

Row alignment was total on every one: 40 of 40 groups scored every row, no duplicates, no drops.
Zero `f0` collisions across all 40 groups of every dataset, so the id recovery — which joins
scored rows back on a feature value, because `anofox_tabfm` echoes back only the columns named
in `features` — never fanned out.

**Every dataset that had a local number reproduced it exactly**, across two machines, two
operating systems, and (for GunPoint) three different chunk configurations. The pod is also
~1.8x faster than the contended local box: Beef 67 s against 129 s, Coffee 64 s against 128 s.
That matters for reading the wall-clock column at all — the local timings understated the
pipeline by nearly half, in the direction that flatters it.

`ItalyPowerDemand` is new here. It could not be run before: at 1,029 test rows in a single
`tabfm_classify` call it took 25.7 GB and killed the local machine, then was OOM-killed twice on
the pod. See "The classify call's memory" below.

`BasicMotions` is the multivariate case, and the first multivariate prediction this project has
produced. It exercises the whole SPEC.md 7 path — per-kernel channel subsets, per-channel
mean-centring, the `DOUBLE[][]` overload — and 1.0000 is where published ROCKET results sit on
it. `SyntheticControl` and `OSULeaf` are the two the failed pod run was meant to add, run
locally instead; `SyntheticControl` is incidentally the dataset that died on the pod every time,
so running it clean here is what proved that failure environmental.

**Mean accuracy 0.9630 — and it should not be compared to the paper's 0.900.** That figure is a
mean over 92 datasets with 30 resamples; this is a single split of nine datasets chosen for
*spread* rather than difficulty, using a different backbone (`tabicl-v2`, because `tabpfn-v2-5`
will not load) at e=1 rather than e=8. The two numbers measure different things and the
resemblance is not evidence of anything.

It moved from the eight-dataset 0.9619 only because `ItalyPowerDemand` (0.9718) happens to sit
above that mean. A mean over nine hand-picked datasets moves with *which* dataset you add, not
with how good the method is — which is the same reason it cannot be read against the paper's.

### Where the wall clock goes, and three things that move it

All measured on one pod, one commit. `time_split` in each `phase5_*.json` carries the first;
`threads_ab/` carries the third.

**1. Inference is 96–99% of it. The C++ transform is ~1.7%.**

| | transform | classify | share |
|---|---:|---:|---:|
| Eight datasets (SyntheticControl excluded, see below) | 35.7 s | 2085 s | **98.3% classify** |
| OSULeaf, the most transform-heavy in the subset | 13.9 s | 357.9 s | 96.3% |

This settles PLAN.md's standing risk, *"TabPFN inference dominates runtime, making C++ ROCKET
pointless"*, in the affirmative on the speed axis. `rocket_transform` is 7.1× faster than the
NumPy oracle and that buys under 2% of the pipeline.

It does **not** make the extension pointless, but it relocates the justification. Pure SQL
measured ~4×10⁵ slower — not viable — so C++ is what makes ROCKET possible *inside the database
at all*, which was the goal. The 7.1× is real and nearly irrelevant to the pipeline, and should
not be quoted as though it moved it.

**2. Chunking the test set costs 2.18×, and preserves every prediction.**

ItalyPowerDemand, same pod, back to back:

| | calls | wall clock | classify |
|---|---:|---:|---:|
| `--test-chunk 128` | 360 | 1074.0 s | 1065.9 s |
| unchunked | 40 | **493.3 s** | **479.8 s** |

**1029/1029 predictions identical, same id set.** Each chunk re-sends the whole train context, so
splitting multiplies work the model has already done. Chunking was never free — it is the price
of fitting a small box. Chunk as coarsely as memory allows, not as finely as convenient.

The row-pass model predicted 1.49× (65,280 → 43,840 rows through the model); the measurement is
2.22× on classify. Per-call overhead therefore costs more than row count alone implies.

**3. The ONNX thread default is sized from the host, not the container: 2.85–5.3×.**

`anofox_tabfm`'s intra-op default is `hardware_concurrency() / 2`, which reads the *host*. On a
64-core cpuset inside a 256-core host that is **128 threads per session**, and DuckDB builds one
per concurrent task — 132 threads in one process, load average 143 against 64 usable cores.

Paired and alternating (128, 16, 128, 16), Coffee, one pod:

| `anofox_tabfm_threads` | runs | median | spread |
|---:|---|---:|---:|
| 128 (the default here) | 211.0 s, 564.8 s | 387.9 s | **2.68×** |
| 16 (cores ÷ duckdb threads) | 74.0 s, 71.4 s | 72.7 s | 1.04× |

Median ratio **5.34×**; worst case for the smaller setting is still **2.85×**. Accuracy identical
across all four.

The variance is half the result: at 16 threads the runs differ by 4%, at 128 by 2.7×.
Oversubscription costs predictability as well as throughput, which is why a single run of either
arm would have proved nothing. Filed upstream as
[anofox-tabfm#3 on the fork](https://github.com/maxdemarzi/anofox-tabfm/issues/3).

**A caveat on all timings here.** The pod is shared and its other tenants are invisible, so
"benchmark on an idle machine, and prove it was idle" cannot be satisfied. `SyntheticControl` took
2984 s where the same dataset took 653 s on the smaller pod, while `OSULeaf` minutes before and
`ItalyPowerDemand` minutes after were within 6% of their earlier numbers. It ran at 16 threads,
and the 16-thread arm above is stable, so thread contention does not explain it. **It is excluded
from the aggregate rather than averaged in, and it remains unexplained.** Accuracy is unaffected —
the pipeline is deterministic, and every dataset reproduced its accuracy exactly.

### The classify call's memory

The earlier note here said the two missing datasets were "slow rather than unsupported". That was
wrong, and the correction is the useful part.

`tabfm_classify`'s memory scales with the rows in one call and lives **outside** DuckDB's buffer
manager, so `SET memory_limit` does not contain it. Four hypotheses were tested against the
failure; three were wrong:

| Hypothesis | Test | Result |
|---|---|---|
| SQL text too large | 18.7 MB -> 7.6 MB | failure moved 18.3 s -> 17.2 s. No |
| DuckDB's own budget | cap 20 GB -> 8 GB | plateau moved 28.73 -> 28.73 GB. No |
| Test rows per call | chunk 128 -> 32 | same 28.7 GB plateau; only the early spike changed. No |
| **Train context size** | GunPoint 50 rows: 11.75 GB. ECG5000 500 rows: 28.7 GB | **yes** |

`--test-chunk` splits the test rows across several calls, which is what made `ItalyPowerDemand`
runnable — its context is only 67 rows, so bounding the chunk bounds the call. It is
identity-preserving, and that was verified rather than argued: GunPoint chunked against
unchunked, same pod, same commit, **150/150 rows identical**. `scripts/compare_predictions.py`
is that check.

`ECG5000` is the one dataset that never produced a number. Its **train context is 500 rows**,
which every call must carry, so its floor is ~501 rows per call however finely the test rows are
split. Four attempts:

| pod | config | outcome |
|---|---|---|
| 29 GB | chunk 128 | OOM at 17.2 s |
| 29 GB | chunk 32 | OOM at 923 s, plateau 28.7 GB against a 29.8 GB ceiling |
| 119 GB | chunk 128, 128 ONNX threads | killed at 4 h, no progress signal |
| 119 GB | chunk 128, 16 ONNX threads | terminated at 5 h 46 m, still running |

Established: **~44 GB peak, more than 5h46m of wall clock**, and memory that no `memory_limit` can
contain because it is ONNX's rather than DuckDB's. It was never slow *because* it was too big —
memory and time are separate walls and it hit both.

One observation left unexplained rather than rationalised: the last attempt sustained **247% CPU**
where `ItalyPowerDemand` on the same pod and settings ran at **1531%** — six times less
parallelism, 5.9 MB of spill, no identified cause. That is where a retry should look first.

### ECG5000, obtained on a GPU (2026-08-13)

**Accuracy 0.9480**, in **18 m 39 s**. Five attempts, and the one that worked changed the device
rather than the query.

| | |
|---|---|
| accuracy | **0.9480** |
| wall clock | **1108.2 s** (transform 70.1 s, classify 1035.0 s = 93.7%) |
| per-group classify | min 25.0 s, median 25.8 s, max 27.3 s (1.1x median) |
| row alignment | 4500/4500 ids, 180,000 group-rows, 40 groups per row |
| device | A40 (sm_86), `anofox_tabfm_device = 'cuda'`, `--test-chunk 128` |
| peak VRAM | ~42.3 GB of 46 GB |

Archived as `reference/phase5_ECG5000_gpu.json`. **This closes Phase 5's tenth dataset.**

Two things make it comparable to the nine CPU rows rather than a separate result. First,
`GunPoint` was run on the same GPU, same build, same patched graph immediately before, and
returned **0.9933 — identical to its recorded CPU accuracy** (`reference/phase5_GunPoint_gpu.json`).
That is the control: the GPU path is not a different numerical pipeline. Second, the patched graph
is bit-identical to the shipped one on CPU, verified separately.

Note what did *not* fix it: chunking was already at 128, and the CPU attempts had 119 GB of host
RAM. The wall was ONNX's own allocation and its speed, and both moved at once on a GPU — 42 GB of
VRAM in place of ~44 GB of host RAM, and >5h46m becoming 18m39s. The 247% CPU anomaly above is
therefore still unexplained; the GPU run routed around it rather than answering it.

Cost: one A40 pod, about 70 minutes end to end, of which ~50 was building the two extensions
because **no GPU build of `anofox_tabfm` is published for any platform**
([anofox-tabfm#25](https://github.com/DataZooDE/anofox-tabfm/issues/25)) and the rocket extension
is ours. The inference itself was 18 minutes.

What the table *is* good for is a sanity check, and it passes one: these values sit where
published ROCKET results sit on these datasets, including Beef being much the hardest — the
subset notes flagged it as "classically hard" before any of it was run. A ROCKET implementation
that was subtly wrong would be unlikely to land in the right neighbourhood on five datasets at
once while also matching the Python oracle bit-for-bit.

Timings are contended and include process startup and Parquet I/O per dataset; they scale with
test rows, as expected when the classify calls dominate.

**What Python still does:** downloads the dataset, writes Parquet, and generates the SQL text.
It computes none of the result. The generation is mechanical templating that cannot currently be
avoided, because `tabfm_classify` needs N named scalar columns — the `LIST` form crashes the
engine — so a 40-group run is ~1.1 MB of SQL.

## Phase 7b′ — does concatenating feature families raise the ceiling? (2026-08-18)

**Yes, by about nine tenths of an accuracy point, and it is not free.**

`--features rocket` (500 ROCKET columns) against `--features both` (500 ROCKET + 116
`anofox_forecast` statistics = 616 columns), paired on the same resampled split per dataset, over the
29 hard datasets. RTX A6000 secure pod running `--device cpu`, 7.65-core cgroup, 46.6 GB, 843.6 min.

**27 datasets paired: mean +0.00923, SE 0.00368, t = 2.51 on df 26, p ~ 0.019.**
B wins 15, A wins 6, 6 exact ties. Net **+57 correct rows out of 6665**.

| dataset | rocket | both | delta | in test rows |
|---|---|---|---|---|
| Beef | 0.7333 | 0.8000 | +0.0667 | +2 of 30 |
| RefrigerationDevices | 0.7787 | 0.8293 | +0.0507 | +19 of 375 |
| Ham | 0.7714 | 0.8000 | +0.0286 | +3 of 105 |
| Lightning7 | 0.7534 | 0.7808 | +0.0274 | +2 of 73 |
| ScreenType | 0.6667 | 0.6880 | +0.0213 | +8 of 375 |
| SemgHandSubjectCh2 | 0.9333 | 0.9533 | +0.0200 | +9 of 450 |
| SmallKitchenAppliances | 0.8240 | 0.8427 | +0.0187 | +7 of 375 |
| InlineSkate | 0.4891 | 0.5073 | +0.0182 | +10 of 550 |
| DistalPhalanxOutlineAgeGroup | 0.8273 | 0.8417 | +0.0144 | +2 of 139 |
| MiddlePhalanxTW | 0.5649 | 0.5779 | +0.0130 | +2 of 154 |
| WormsTwoClass | 0.7922 | 0.8052 | +0.0130 | +1 of 77 |
| MedicalImages | 0.8447 | 0.8553 | +0.0105 | +8 of 760 |
| DistalPhalanxTW | 0.7554 | 0.7626 | +0.0072 | +1 of 139 |
| ProximalPhalanxTW | 0.8098 | 0.8146 | +0.0049 | +1 of 205 |
| SemgHandMovementCh2 | 0.8400 | 0.8422 | +0.0022 | +1 of 450 |
| Herring | 0.6719 | 0.6719 | +0.0000 | +0 of 64 |
| Lightning2 | 0.7541 | 0.7541 | +0.0000 | +0 of 61 |
| ACSF1 | 0.8400 | 0.8400 | +0.0000 | +0 of 100 |
| Haptics | 0.6136 | 0.6136 | +0.0000 | +0 of 308 |
| Earthquakes | 0.7482 | 0.7482 | +0.0000 | +0 of 139 |
| MiddlePhalanxOutlineAgeGroup | 0.6948 | 0.6948 | +0.0000 | +0 of 154 |
| ProximalPhalanxOutlineAgeGroup | 0.8683 | 0.8634 | -0.0049 | -1 of 205 |
| ArrowHead | 0.8857 | 0.8800 | -0.0057 | -1 of 175 |
| LargeKitchenAppliances | 0.9280 | 0.9200 | -0.0080 | -3 of 375 |
| Worms | 0.7662 | 0.7532 | -0.0130 | -1 of 77 |
| EthanolLevel | 0.7980 | 0.7820 | -0.0160 | -8 of 500 |
| Computers | 0.8840 | 0.8640 | -0.0200 | -5 of 250 |

**Read the last column.** Six datasets change no prediction at all, and several deltas are one or two
rows — Beef's +0.0667, the largest, is two rows out of thirty. The result is real but it is assembled
from small margins, and the single-split `+0.0088, 4 wins to 2` this replaced was made of the same
rows without the error bar.

**The cost, in seconds.** 616 columns are slower than 500 on every dataset measured:

| dataset | rocket | both | extra |
|---|---|---|---|
| RefrigerationDevices | 1,278 s | 1,678 s | +400 s |
| LargeKitchenAppliances | 1,297 s | 1,675 s | +378 s |
| SmallKitchenAppliances | 1,268 s | 1,638 s | +370 s |
| ProximalPhalanxTW | 842 s | 1,079 s | +237 s |

Four to seven extra minutes per dataset for +0.009 accuracy. Timings are comparable only *within* a
dataset — the campaign was restarted mid-flight from `--jobs 2` to `--jobs 1`, so seconds do not
compare across that boundary. Accuracy does: it is deterministic given the split and the seed.

**Also: `ts` features are BSL 1.1 (`anofox_forecast`) and cannot ship.** A +0.009 gain from columns
this project cannot distribute is a research finding, not a product change. `catch22` (22 columns, in
`aeon`, unrestricted) is the shippable version of the same idea.

**Plumbing done, campaign not run.** `--features catch22` (rocket dropped, `--n-groups 1`, same
restriction `ts` carries) and `--features both22` (rocket+catch22, the shippable analogue of `both`)
are wired into `phase5_pipeline.py` — `catch22_feature_names()`/`write_catch22_parquet()` compute the
22 statistics in Python via aeon's `Catch22` transformer (no DuckDB extension exists for it, so it is
loaded with `read_parquet` exactly where `ts` mode would `LOAD anofox_forecast`) and joined the same
way `tsfeat` is. Smoke-tested end to end against the real extension on Coffee: `catch22` alone and
`both22` both scored 1.0000 with clean integrity checks (`c22_check`, `prime_check`,
`features_check` all 0) and exact row alignment. **Not yet measured**: whether `catch22` recovers any
of the `+0.0092` `both` gained on the 27-dataset resample campaign — that needs the same pod pass
`--features both` got, at `--features both22`.

### Two datasets are missing, asymmetrically

`MiddlePhalanxOutlineCorrect` and `DistalPhalanxOutlineCorrect` — both **600 training rows**, the
largest in the set — completed arm A (1,494 s) and had arm B killed by the kernel at 132 s and 34 s,
under `--jobs 1`, so one run alone exceeded 46.6 GB. B is the wider arm, so **memory pressure removes
pairs exactly where B is most expensive**, and the direction of the resulting bias is unknown. 2 of 29
does not overturn the result; omitting the mechanism from the write-up would misrepresent it.

`--test-chunk` is not the fix: it sizes the query side only, and a group's chunks share one context,
so the context encode runs at full width regardless.

### The harness printed a false verdict, and now refuses to

The R=1 run reported *"the between-dataset term dominates, and this comparison cannot be resolved by
resampling at any affordable scale."* That was an artifact of R=1, not a finding:

| | R=1 asserted | corrected with the pilot's measured within |
|---|---|---|
| `var_within` (split luck) | 0.000000 | 0.000300 (pilot at R=4.94) |
| `var_between` | 0.000365 | **0.000065** |
| SE floor at D=27 | 0.0037 "however many resamples" | **0.00155** |
| runs to reach SE ≤ 0.0018 | unreachable at any scale | **R=14, 756 runs** |

With one resample per dataset nothing can observe within-dataset variance, so `within` defaults to
0.0; the decomposition then assigns the whole spread to `between`, `se(d, r)` loses its dependence on
`r`, and the "cannot be resolved" line prints for **every** input. The pilot had already measured
within at **8×** between — the term zeroed out was the larger one. `analyse()` now carries an `R < 2`
guard beside the existing `D < 2` one: mean and SE are returned (both valid at R=1), the decomposition
and every plan sized from it are withheld. Resampling here would nearly halve the SE (R=3 → 0.0025,
108 new runs, ~28 h), the opposite of what was printed.

## Not done

- **A noise floor worth the name, and it is now the binding constraint.** One was measured (0.0509)
  on a five-dataset subset where four saturate, so it had almost no resolving power. The subset
  problem is fixed — 28-29 hard datasets, chosen for being unsaturated — but the floor is not: at a
  20% escalation budget this harness cannot resolve about half a point, and **every result of the
  last two days lands inside it.** G=10 costs −0.0033; a 20x kernel cut costs −0.0003; the best
  ensemble rule found gains +0.0024. Each is reported as undetectable rather than as equivalent,
  which is honest and unsatisfying in the same measure. Converting any of them into a claim needs
  more seeds or more datasets, and that is a campaign rather than a run.

  **Now sized, and the campaign is 4.5x cheaper than the paper's protocol.** `phase5_pipeline.py
  --resample K` re-splits the pooled train and test at the same per-class sizes (K=0 is the
  archive's own split, so nothing archived changes), and a 160-run pilot through
  `scripts/pod/resample_power.py` measured where the noise actually lives: **split luck is 8x the
  dataset effect** (sd 0.0173 against 0.0061). That contradicts what this entry used to predict
  from the subset sign-flip. Since more datasets shrink both terms and more resamples only one,
  **40 datasets x 4 resamples = 320 runs** resolves 0.005 where 24 x 30 would be 1,440. The pilot
  also put the sign the other way from the archived split: +0.0025 against −0.0033, both
  indistinguishable from zero, which is the entry's own point made with data. See "Split luck is
  8x the dataset effect, so buy breadth" above. **What remains is running it.**
- **e=8 versus e=1.** `--compare-estimators 8,1` exists and was never run: at 500-feature
  groups e=1 already covers every feature, so the interesting comparison is the paper's
  2,000-feature groups at e=8 against this configuration — which is a different experiment than
  the flag performs, and an expensive one.
- **The 30-resample protocol.** Every result here is a single train/test split. Phase 5 now covers
  59 datasets rather than the five this entry used to describe, which fixes the *breadth* problem
  and not the *resampling* one: the paper's protocol averages 30 resamples per dataset, and one
  split cannot separate a real half-point from a lucky one. This is the same gap the noise-floor
  entry names from the other end, and it is why every recent result here is reported as "inside
  what this harness can resolve" rather than as an effect.
- **The upstream context cache, in the extension.** Both halves are now written.
  [PR #38](https://github.com/DataZooDE/anofox-tabfm/pull/38) (merged) exports the split graphs, and
  [PR #40](https://github.com/DataZooDE/anofox-tabfm/pull/40) (open) is the runtime handle that holds
  a prepared support set across calls, behind `anofox_tabfm_context_cache`. So this is now blocked on
  **review, then a release, then a re-export of the published weights** — the pair is discovered on
  disk, and no published model ships one.

  Now measured twice: through DuckDB on real weights as a call pair (**7.18x** repeated, 0.40x
  cold), and then through the **whole phase-5 pipeline**, which is the number that decides
  anything. End to end it is **1.85x** at nine chunks per group, 1.13x at two, and **0.64-0.73x at
  one** — a loss, and one chunk is the largest bucket in this archive. Same answer throughout: zero
  disagreements on 1329 test rows. See "The context cache is worth 1.85x, 1.13x or −56%" above.
  Break-even is around 384 test rows, so it belongs on per dataset rather than globally, and a full
  sweep pass would gain about 7%.
- **`tabpfn-v2-5` and `tabpfn-v3`, the paper's own backbones.** Both download and neither loads,
  by two different faults with one remedy: `tabpfn-v2-5`'s checkpoint is published under tensor
  names the exported graph was not built against (missing 248 of 250), and `tabpfn-v3` is a torch
  zip whose pickle stream uses opcode 0x65, which the checkpoint reader does not implement. Either
  way the engine prefers a sibling `model.safetensors`, and the one-time conversion
  `scripts/convert_model_weights.sh` performs writes one — for the two models licensed
  commercially, deliberately not for these. So this is a **licence decision, not a blocker**, and
  it is much less alarming than it was: `tabicl-v2`, `tabpfn-v2` and `orion-bix` score 0.7579,
  0.7568 and 0.7371 on the same 17 datasets and fail on the same rows, so the backbone is not
  where the results come from.
*(Resolved during the session: `test/sql/rocket.test` now runs through DuckDB's own
sqllogictest runner — **12 assertions, all passing** — via `scripts/build_extension.bat tests`.
Two traps were worth fixing on the way: the runner registers tests under a canonical
forward-slash path, so a native Windows path matches nothing, and it then exits 0 having run
nothing, which reads exactly like success. The script now checks for the pass line rather than
trusting the exit code.)*
