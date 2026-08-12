# Results

What has actually been measured, with the caveats that make each number readable. Raw output is
in the JSON files beside this one; every script named here regenerates its own.

**Environment.** Windows 11, RTX 3060 (unused — `torch` is the CPU wheel by design), DuckDB
v1.5.5 (`d8cdaa33`), `anofox_tabfm` `bc6d8af`, `tabpfn` 8.2.0, `tabicl-v2` as the DuckDB-side
backbone.

**Every timing here is contended.** The box was simultaneously running an unrelated
`finetune.py` training job with two worker processes, plus this project's own background runs.
PLAN.md already requires reported figures to come from a pod; these are correctness-oriented
numbers with a comparative reading at best.

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

### Breadth: seven datasets, G=40, e=1, `tabicl-v2`

| Dataset | Test rows | Timepoints | Accuracy | Wall clock |
|---|---|---|---|---|
| Coffee | 28 | 286 | 1.0000 | 128 s |
| Trace | 100 | 275 | 1.0000 | 260 s |
| GunPoint | 150 | 150 | 0.9933 | 258 s |
| SyntheticControl | 300 | 60 | 0.9867 | 674 s |
| FaceFour | 88 | 350 | 0.9773 | 172 s |
| OSULeaf | 242 | 427 | 0.9711 | 449 s |
| Beef | 30 | 470 | 0.7667 | 129 s |

Row alignment was total on every one: 40 of 40 groups scored every row, no duplicates, no drops.

`SyntheticControl` and `OSULeaf` are the two the failed pod run was meant to add; they were run
locally instead, at full configuration, after the pod was stopped. `SyntheticControl` is
incidentally the dataset that failed on the pod every time — running clean here is what
identified that failure as environmental rather than anything to do with the data.

**Mean accuracy 0.9564 — and it should not be compared to the paper's 0.900.** That figure is a
mean over 92 datasets with 30 resamples; this is a single split of seven datasets chosen for
*spread* rather than difficulty, using a different backbone (`tabicl-v2`, because `tabpfn-v2-5`
will not load) at e=1 rather than e=8. The two numbers measure different things and the
resemblance is not evidence of anything.

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

## Not done

- **A noise floor worth the name.** One was measured (0.0509) but it rests on a single dataset:
  four of the five saturate and reproduce exactly, so the subset has almost no resolving power.
  Three seeds on one informative dataset is a weak floor. Widening the subset toward datasets
  that are hard but not ceiling-bound would buy more than more seeds on these.
- **e=8 versus e=1.** `--compare-estimators 8,1` exists and was never run: at 500-feature
  groups e=1 already covers every feature, so the interesting comparison is the paper's
  2,000-feature groups at e=8 against this configuration — which is a different experiment than
  the flag performs, and an expensive one.
- **Anything measured on a pod.** One was run and it produced **no usable results**; see
  "The pod run that failed" below. Every number above is local.
- **Multivariate and variable-length series.** Still unspecified in SPEC.md §7, so
  `BasicMotions` remains skipped rather than silently mishandled.
- **The 92-dataset / 30-resample protocol.** Phase 5 covers five datasets on a single split.
  That is enough to rule out a one-dataset fluke and nowhere near enough to compare against the
  paper. The multivariate case is excluded by construction, and the larger datasets
  (`ECG5000`, `ItalyPowerDemand`, `OSULeaf`, `SyntheticControl`) were skipped for runtime.
- **Re-measuring on the correct backbone.** Every accuracy number here uses `tabicl-v2`, because
  `tabpfn-v2-5` would not load. That is fixed in `anofox_tabfm v2026.08.11`, which the community
  repository does not serve yet. Once it lands, the whole subset is worth re-running on the
  paper's actual model — and `tabpfn-v3` becomes available too. **Anything measured before then
  is measured on a substitute.**
*(Resolved during the session: `test/sql/rocket.test` now runs through DuckDB's own
sqllogictest runner — **12 assertions, all passing** — via `scripts/build_extension.bat tests`.
Two traps were worth fixing on the way: the runner registers tests under a canonical
forward-slash path, so a native Windows path matches nothing, and it then exits 0 having run
nothing, which reads exactly like success. The script now checks for the pass line rather than
trusting the exit code.)*
