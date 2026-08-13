# Open tasks — handoff at 2026-08-13 20:00

Phase 5 is complete: **ten of ten datasets**. The previous version of this file was written when
eight had run and two looked unreachable on any machine, so most of its "blocker" sections are now
history. What survives below is the traps, because those cost real time and will cost it again.

## State

- **Phase 5: done.** Nine datasets from one CPU pod at one config; ECG5000 from a GPU
  (0.9480, 18m39s). `reference/RESULTS.md` carries both, plus the control that makes them
  comparable — `GunPoint` re-run on the GPU returned its recorded CPU accuracy exactly.
- **No pods of ours are running.** Verified with `runpod_gpu.py check`. Two `eval-*` A40s belong to
  someone else on the shared account; leave them.
- **`description.yml` is pinned** to `e325474a7cab58bc11a30ebba61604002cb8fe55`, and the breadth
  gate in its own header is met. **Submission is the owner's decision and has not been made.**

## Open

1. **Submit to community-extensions** — or decide not to. Everything mechanical is done. Re-pin
   `ref` only if `src/`, `extension_config.cmake`, `CMakeLists.txt` or the duckdb submodule
   changes; doc and script commits do not alter what is built.
2. **Publish a prebuilt shell** so `scripts/pod/bootstrap.sh`'s caching path activates. The code is
   in place and unused. Today's GPU run spent ~50 of its ~70 minutes building two extensions; one
   of them is this.
3. **Upstream review is out of our hands**: [#22](https://github.com/DataZooDE/anofox-tabfm/pull/22)
   (Ort::Env ordering), [#23](https://github.com/DataZooDE/anofox-tabfm/pull/23) (ScatterND graph
   workaround), [#24](https://github.com/DataZooDE/anofox-tabfm/pull/24) (Windows CUDA discovery),
   [#25](https://github.com/DataZooDE/anofox-tabfm/issues/25) (dead `ext.anofox.com`).
   [#19](https://github.com/DataZooDE/anofox-tabfm/pull/19) merged and shipped in `v2026.08.13`.

## GPU: works, but only if you build it

Reversed from the previous handoff, which recorded GPU as closed. It is reachable — it is just not
*downloadable*.

- **No GPU build is published for any platform.** `ext.anofox.com` — the host named in
  `anofox_tabfm`'s own error message, in its `anofox_tabfm_device` setting description and in its
  README — has no DNS record. Filed upstream as
  [#25](https://github.com/DataZooDE/anofox-tabfm/issues/25). Building the cuda flavor takes ~25
  minutes.
- **`tabicl-v2` cannot run on CUDA with the shipped graph.** It fails at a `ScatterND` node because
  the CUDA `Slice` computing that node's `indices` returns its input untrimmed — an ONNX Runtime
  bug, present in 1.26.0 and 1.28.0. The workaround is a graph edit
  ([#23](https://github.com/DataZooDE/anofox-tabfm/pull/23)) and is bit-identical on CPU. Point a
  run at the patched graph with `--register-model-dir`.
- **The pipeline supports this directly**: `--device cuda --anofox-extension <path>
  --register-model-dir <dir>`. `--device cuda` without `--anofox-extension` is rejected, because
  the installed community build is CPU-only and would run on the CPU while reporting success.
- **Windows GPU additionally needs [#24](https://github.com/DataZooDE/anofox-tabfm/pull/24)** —
  device discovery was compiled out there. With that patch it works on the local RTX 3060, and the
  CUDA 12 runtime must be on `PATH`.
- **`anofox_tabfm_gpu_precision` defaults to bf16.** It reads as ROCm/MIGraphX-only, and CUDA
  reproducing CPU accuracy exactly is evidence it did not apply — but that was not confirmed from
  the code path. Before any ROCm run, pin `fp32` and re-verify accuracy *before* believing a speed
  number.

## Traps worth keeping

**Container-blindness, four instances.** Every one presented as something else.

| what read the host instead of the container | symptom |
|---|---|
| DuckDB `memory_limit` from `free` | OOM kill, no error message |
| DuckDB `threads` from `nproc` | 132 threads, load average 143 |
| ONNX intra-op from `hardware_concurrency()` | same, per session; fixed upstream in #19 |
| `ninja` job count vs the cgroup ceiling | `cc1plus` killed — reads as a compiler bug |

Set `threads`, `anofox_tabfm_threads`, `memory_limit` and `CMAKE_BUILD_PARALLEL_LEVEL` explicitly.
`duckdb_rocket/budget.py` derives the first three; 6 is the working build parallelism against the
79 MB `tabfm_bundled_resources.cpp`.

**Never `PREPARE` against an empty source table.** DuckDB fixes a filter's selectivity from the
source's statistics at prepare time, so a statement prepared against an empty table has its
predicate pruned to always-false — permanently, silently. The ordering in `build_sql` is
load-bearing and `tests/test_phase5_sql.py` pins it.

**Chunking is identity-preserving, but verify it.** `--test-chunk N` issues one `tabfm_classify`
per N test rows, so peak memory is a function of N rather than of the dataset. An in-context
learner treats each test row as an independent query against the train context, so a row's
prediction cannot depend on which rows shared its call — verified rather than argued: GunPoint at
`--test-chunk 50` against its own unchunked run, **150/150 ids, 0 rows disagreeing**. Accuracy
alone would not have settled it; two runs can match on accuracy and disagree on which rows they got
right. `scripts/compare_predictions.py` is that check. It is also nearly free: 248.7 s chunked vs
258 s unchunked, 3x the calls, because the model load amortises.

**`pgrep -f` / `pkill -f` match themselves.** Hit five times in one day despite the warning being
quoted in advance. Use `pkill -x`, or a `[b]racket` pattern — which still does not protect against
another process that happens to contain the string.

**Gate a pod on download speed before using it.** One pod spent 29 minutes cloning at 0.08 MB/s.
`runpod_cpu.py gate` is read-only and exists for this. Also: `curl` without `-L` reads as 0 B/s on
a perfectly healthy host.

**A "GPU" run that silently ran on the CPU looks like a success.** Assert `tabfm_devices()` shows a
usable cuda row before believing any GPU number, and assert a patched graph actually differs from
the stock one — a failed `curl` otherwise yields a plausible result from the wrong graph.

**`SET custom_extension_repository` is global.** It redirects *every* install, including the
`httpfs` autoload, which then fails first with an unrelated message. And `INSTALL` on an
already-installed extension appears to succeed while leaving the old one in place; only
`FORCE INSTALL` surfaces the error.

**Do not pipe a build through `tail`.** It keeps the summary and discards the error. Write the full
log to a file and grep the *first* error with context. The same applies to `git push | tail`, which
hides a failed push behind a zero exit status.
