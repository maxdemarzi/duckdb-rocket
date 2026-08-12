# duckdb-rocket

Training-free time-series classification in DuckDB.

This project brings [RocketPFN](https://arxiv.org/abs/2606.21786) into DuckDB by building the
**feature-extraction half** — a `rocket_transform()` extension — and composing it with
[`anofox_tabfm`](https://github.com/DataZooDE/anofox-tabfm), which already ships the
`tabpfn-v2-5` and `tabicl-v2` tabular foundation models.

```
series ──▶ rocket_transform()  ──▶ tabfm_classify()  ──▶ average probs ──▶ label
           (this project)          (anofox_tabfm)        (plain SQL)
```

No gradient descent, no training loop: ROCKET projects each series onto 10,000 random
convolutional kernels, and an in-context tabular model classifies the resulting features
directly. Per the paper, the kernels are split into **G=10 groups of 1,000** — each kernel
contributes 2 features (global max and proportion of positive values), so each group is exactly
2,000 features, TabPFN v2.5's column cap. Every group is classified independently and the class
probabilities are averaged.

The reference result is **0.900 mean accuracy across 92 UCR datasets** at a median of ~30s per
fold.

## Status

Early. Phase 0 of [PLAN.md](PLAN.md) — toolchain and skeleton. Nothing here classifies anything
yet.

The two phases that matter most are cheap and require no C++: Phase 2 probes whether
`anofox_tabfm` can return class probabilities at all (a GO/NO-GO gate on the whole design), and
Phase 3 proves the composition end-to-end with ROCKET still living in Python. See
[PLAN.md](PLAN.md) for the full sequence and the standing risks.

## Development

Requires [`uv`](https://docs.astral.sh/uv/), CMake, Ninja, and — on Windows — MSVC Build Tools
(clang alone cannot link a CPython extension without the Windows SDK).

```bash
uv sync
uv run pytest                              # 94 tests, no model weights needed
uv run python scripts/doctor.py            # record the environment tuple
uv run python scripts/emit_golden.py       # regenerate conformance fixtures
```

The DuckDB CLI is pinned to v1.5.5 under `tools/` because extension ABI is version-bound.

### TabPFN weights require a third-party licence

Reproducing the accuracy numbers needs TabPFN v2.5 weights, which Prior Labs gates behind an
accepted licence. Register at <https://ux.priorlabs.ai/account>, accept the licence, and put
the API key in the environment:

```bash
export TABPFN_TOKEN=...      # or let the first interactive run cache it
```

Everything that does not touch model weights — the ROCKET transform, the golden vectors, the
whole test suite — runs without it.

## License

MIT — see [LICENSE](LICENSE). The license matches `anofox_tabfm` to keep the door open for
upstreaming.
