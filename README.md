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

The pipeline runs end to end, entirely inside DuckDB. Phases 1–4 of [PLAN.md](PLAN.md) are
done; Phase 5 is measured on a subset, not yet at the paper's 92-dataset protocol.

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

**`tabpfn-v2-5` does not currently load through `anofox_tabfm`** (version `bc6d8af`): the
published checkpoint no longer matches anofox's bundled ONNX graph. `tabicl-v2` works and is
what the pipeline uses by default. Details and reproduction in
[reference/PHASE2_FINDINGS.md](reference/PHASE2_FINDINGS.md).

### What has been measured

| | |
|---|---|
| Conformance (C++ vs. Python oracle) | max abs diff **1.8e-15**, PPV differences exactly 0 |
| `rocket_transform` vs. numpy oracle | **7.1×** faster at 4,000 series, 1.7× at 200 |
| Pure-SQL ROCKET | correct, and ~**4×10⁵** slower than Python — a fallback, not an option |
| GunPoint, end to end | see `reference/phase5_GunPoint.json` |

All timings are local Windows numbers on a contended box and belong in no table — PLAN.md
requires reported figures to come from a pod.

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
rocket_transform(series DOUBLE[], kernels_per_group BIGINT, seed BIGINT, first_kernel BIGINT)
  -> DOUBLE[]
```

Returns `kernels_per_group * 2` features, interleaved: element `2i` is kernel `i`'s max and
`2i+1` its PPV.

`first_kernel` is what makes groups work. Kernel `i` is a pure function of `(seed, i)`, so group
`g` is global kernel indices `[g*k, (g+1)*k)` and can be generated without generating the groups
before it:

```sql
-- these are the same kernels
SELECT rocket_transform(s, 4, 7, 4) = rocket_transform(s, 8, 7, 0)[9:16] FROM ...;  -- true
```

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
