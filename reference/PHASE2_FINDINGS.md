# Phase 2 findings — probing `anofox_tabfm`

**Probed:** `anofox_tabfm` **`bc6d8af`** (community repository) against DuckDB **v1.5.5**,
Windows 11, CPU execution provider. Reproduce with `uv run python scripts/probe_anofox.py`;
raw output in `anofox_probe_*.json` beside this file.

> Pin the version with every number. anofox-tabfm is pre-1.0 and tags near-daily.

## The short version

The gate is open, but not on the terms the plan assumed. Two findings reshape the design:

1. **`tabpfn-v2-5` does not run at all in `bc6d8af`.** The published checkpoint no longer
   matches anofox's bundled ONNX graph. `tabicl-v2` works and is the de facto backbone.
2. **A TabPFN estimator only ever sees 500 features, not 2,000.** The paper's
   2,000-feature groups need ≥4 estimators to be covered — and anofox caps estimators at 1.
   The plan's "2,000 features, exactly TabPFN v2.5's cap" reads an input ceiling as a
   per-estimator budget.

Together these mean the intended composition works, but **the group width must come down from
2,000 features to 500** for the SQL path to measure the same thing the Python oracle does.

---

## 1. `tabpfn-v2-5` is broken in this build — **fixed upstream in `v2026.08.11`**

> **Update (2026-08-12).** This is fixed, and it was fixed the day before we hit it. Release
> [`v2026.08.11`](https://github.com/DataZooDE/anofox-tabfm/releases/tag/v2026.08.11) — *"Fixed:
> three models could not load real weights"* — reports that `tabpfn-v2`, `tabpfn-v2-5` and
> `orion-bix` all failed against real weights while passing their fixture tests: the converter
> writes `model.safetensors` but the manifests declare the downloadable `model.ckpt`, and
> nothing pointed at the converter's output. The engine now prefers a sibling
> `model.safetensors`, and the misleading *"corrupted — re-download it"* message below has been
> replaced by one that names `convert_weights.py`. No issue was filed, because there was nothing
> left to report.
>
> **We do not have the fix yet.** The community repository still serves `bc6d8af`
> (`v2026.08.07`), so `FORCE INSTALL ... FROM community` is a no-op. Getting `tabpfn-v2-5`
> requires either waiting for the community rebuild or vendoring the tag from source, and the
> checkpoint additionally has to be run through `convert_weights.py`.
>
> **The same release adds `tabpfn-v3`** as a seventh built-in model, with a stated export parity
> of 2.4e-07 for classification.
>
> Everything below stands as the record of what `bc6d8af` does, which is still what an
> installing user gets today.

`tabfm_download('classification', model := 'tabpfn-v2-5')` succeeds (42,935,499 bytes from
`Prior-Labs/tabpfn_2_5@main`), but any `tabfm_classify` against it fails:

```
IO Error: anofox_tabfm: the checkpoint for 'classification' is corrupted or does not match the
bundled model graph. ORT said: graph.cc:3660 ReplaceInitializedTensorImpl Failed to find
existing initializer with name m.transformer_encoder.layers.14.self_attn_between_items._w_qkv
```

`tabfm_remove` + re-download returns a byte-identical file and the same failure (the named
missing initializer varies between runs — layer 14, then layer 7 — but the cause does not).
So this is **not** a corrupt download, despite what the error message advises. anofox pins the
HuggingFace ref to `@main`, and upstream has moved past the graph anofox was built against.

**Consequence:** the model this project was designed around is unavailable through DuckDB
today. `tabicl-v2` (BSD-3, `jingang/TabICL@main`, 110,368,038 bytes) loads and runs, which
makes PLAN.md's optional **Phase 3b** — TabICL v2 as the backbone — the *default* path rather
than an experiment. Worth an upstream issue; see "For upstream" below.

## 2. The 500-feature-per-estimator ceiling — the important one

`tabfm_list_models()` advertises `max_features = 500` for both `tabpfn-v2-5` and `tabicl-v2`,
which contradicts the plan's 2,000-feature groups. Chasing that down through the local `tabpfn`
package rather than guessing:

- `preprocessing/configs.py:115` — `max_features_per_estimator: int = 500`
- `preprocessing/ensemble.py:1220` — `scale_n_estimators_for_feature_coverage()` raises
  `n_estimators` to `ceil(n_total_features / 500)` so every feature is seen at least once

So **2,000 is the widest input TabPFN v2.5 accepts; 500 is the widest view a single estimator
gets.** Above 500, features are subsampled per estimator. A 2,000-feature group is covered only
at e≥4, and the paper's e=8 covers it with redundancy to spare.

This was caught because a local run requesting `n_estimators=1` emitted:

```
UserWarning: Auto-scaling n_estimators from 1 to 4 so every feature is included in at least one
ensemble member (n_total_features=2000, max_features_per_estimator=500)
```

**The measurement trap:** anofox hard-caps `n_estimators=1` and has no auto-scaling. Left
alone, a local run *labelled* e=1 would really be e=4, compared against a DuckDB path that
really is e=1 — a gap that would have looked like a ROCKET implementation bug. `RocketPFNConfig`
now sets `auto_scale_n_estimators=False` so a requested e is the e that runs, and exposes
`covers_all_features` / `anofox_reachable`, which `scripts/accuracy.py` records beside every
accuracy number.

**Design consequence — the recommended change:** narrow the groups. At **250 kernels per group
→ 500 features**, one estimator sees the whole group, e=1 is honest, and the two paths compare.
Keeping the paper's 10,000 kernels means **G=40** rather than G=10. This preserves the kernel
budget and the average-probabilities-across-groups structure; only the split changes. Phase 1's
current run uses this configuration.

## 3. The 2,000-column calling convention works — question retired

The 500-column limit is a **configurable guard, not a model limit**. anofox rejects wider calls
with a Binder Error that names the setting to change:

```
Binder Error: tabfm_predict_agg: 2000 feature columns exceed anofox_tabfm_max_features (500).
Raise it with SET anofox_tabfm_max_features = 2000;
```

With `SET anofox_tabfm_max_features = 4000`, all widths bind and run:

| Feature columns | Default guard | Guard raised | Wall clock (60 train / 40 test) |
|---|---|---|---|
| 100 | OK | — | 4.1 s |
| 500 | OK | — | 6.8 s |
| 512 | rejected | OK | 7.0 s |
| 1,000 | rejected | OK | 17.2 s |
| 2,000 | rejected | OK | 41.4 s |

So `features := [...]` with 2,000 names is workable, and **the upstream PR the plan
contemplated for this is unnecessary.** Note the cost curve is worse than linear (≈6× the width
for ≈10× the time) on a trivially small problem — a latency warning for Phases 3 and 5.

The accuracy column from that sweep is *not* reported here on purpose: the probe is a
deliberately adversarial "one signal column among N noise columns" dilution test on 44 test
rows, where a single row is 2.3%. It cannot separate width sensitivity from noise, and reading
it as evidence about ROCKET features would be exactly the single-dataset overreach the plan
warns about.

## 4. Row identity is free — swan's rowid hack is not needed

The plan's fallback was swan's `ROW_NUMBER() OVER (ORDER BY hash(pk))` injected as a model
feature, which cost swan two empirically-found bugs. It is unnecessary here:

- **Only test rows are returned.** With an explicit `test := <view>`, the result contains
  exactly the test rows (40 in, 40 out; 0 rows with `is_training = true`).
- **Output order is deterministic** — two identical calls agreed on all 40 rows.
- **Output order matches the test view's order** — 0 mismatches against
  `row_number() OVER (ORDER BY id)`.

So the G groups can be joined **positionally**, which PLAN.md hoped for ("prefer that if it
works"). Two caveats before relying on it in Phase 3: this was 40 rows on one thread, and
DuckDB may reorder under parallelism at UCR scale; and it is an undocumented behaviour, not a
guarantee. **Phase 3 must assert the join rather than assume it** — a positional join that
silently slips would corrupt every downstream number while leaving the output well-formed.

## 5. Answers to the remaining Phase 2 questions

| Question | Answer |
|---|---|
| Does `tabfm_classify` return probabilities? | **Yes.** `proba` is `MAP(VARCHAR, DOUBLE)` keyed by class label — averaging across groups can key on the class rather than trusting position. Confirms swan. |
| Does `n_estimators > 1` still throw? | **Yes.** `Not implemented Error: n_estimators > 1 (ensemble) is not available yet — lands with milestone M3; use n_estimators = '1'`. Confirms swan. |
| Is there a precision / AMP lever? | **No.** The binder enumerates every valid option: `task, n_estimators, seed, output_mode, context_rows, softmax_temperature, model`. There is no precision control on the ONNX path, so TabPFN's `inference_precision` lever does not exist here. What precision the exported graph uses is **unknown and unsettable** — recorded as a result, per the plan. |
| Do `LIST`/`ARRAY` columns work instead of N scalar columns? | **No — and it crashes.** `features := ['feats']` over a `DOUBLE[]` column fails with `INTERNAL Error: anofox_tabfm: Run() called with null input buffers`, which DuckDB reports as an assertion failure. Not a clean rejection; a genuine bug. |
| Is there a passthrough id column? | **No.** The result echoes the target, the named features, `yhat`, `yhat_score`, `is_training`, and `proba`. An `id` column not named in `features` is silently dropped. Confirms swan. |

Undocumented but useful: **`is_training`** distinguishes context rows from scored rows, and
`context_rows` / `softmax_temperature` / `seed` are settable options the plan never listed.

## 6. Two Windows-specific traps

- **ONNX Runtime prints thousands of `Schema error: Trying to register schema ...` lines to
  stderr on every model load.** They are harmless duplicate-registration warnings and they bury
  the one line that matters — the real error above was found under ~20,000 characters of it.
  `scripts/probe_anofox.py` filters them; anything driving this extension should too. This is a
  milder relative of the ONNX ABI hazard at PLAN.md Phase 2, finding 7.
- **Generated wide-column SQL must go through a script file, not `duckdb -c`.** A 500-column
  feature list already exceeds the Windows 32,767-character command-line limit, and the failure
  is `FileNotFoundError: [WinError 206] The filename or extension is too long` — an error
  naming neither SQL nor length. Phases 3 and 4 inherit this.

## For upstream

1. ~~**`tabpfn-v2-5` checkpoint/graph mismatch**~~ — **already fixed** in `v2026.08.11`, along
   with the misleading error message. Not filed; there was nothing left to report. Checking the
   release notes before writing the issue is what caught this.
2. **`Run() called with null input buffers`** on a `LIST`-valued feature column (finding 5) —
   **filed as [anofox-tabfm#17](https://github.com/DataZooDE/anofox-tabfm/issues/17)**.
   Confirmed still open before filing: `src/tabfm_preprocess.cpp` and `src/tabfm_ort_engine.cpp`
   are byte-identical between `bc6d8af` and `v2026.08.11`. The cause is `GetFeatureKind`'s
   categorical fallback, which is correct for scalar unknowns like `BLOB` but swallows nested
   types, so the failure only surfaces in the ONNX engine long after the type is gone.

The two upstream PRs PLAN.md anticipated are both moot: list-valued features are a bug report
rather than a feature request, and probability output already exists. `n_estimators` is
anofox's own milestone M3 — their roadmap, so ask before building.
