# Open tasks — handoff at 2026-08-12 11:50 (machine restart)

## ⚠ LIVE POD — `khv4iojz2zbphj` is billing right now

    python scripts/pod/runpod_cpu.py check
    python scripts/pod/runpod_cpu.py ssh khv4iojz2zbphj
    python scripts/pod/runpod_cpu.py terminate khv4iojz2zbphj --yes-destroy-the-volume

`duckdb-rocket-cpu-128g`, 64 vCPU, **119 GB** cgroup, ~$2.24/hr, host `38.80.152.147`.
**Terminate it when done.** `volumeInGb: 0`, so nothing on it survives — fetch before terminating.

Exists only for ECG5000, which needs ~29 GB and did not fit the first pod's 29.8 GB ceiling.
The nine other datasets are fetched and committed already.

### Gate every pod on download speed before bootstrapping it

Learned expensively today, and it is already in PLAN.md as black_swan's rule (a host with healthy
specs and 270 kB/s cost them a whole session). Four pods were created before one passed:

| vCPU | host | download | outcome |
|---|---|---|---|
| 32 | 157.157.221.177 | 0.09 MB/s | 29 min stuck cloning DuckDB, terminated |
| 32 | 157.157.221.177 | 0.09 MB/s | gate fail |
| 32 | 157.157.221.177 | 0.08 MB/s | gate fail |
| 32 | 157.157.221.177 | 0.08 MB/s | gate fail |
| **64** | **38.80.152.147** | **34.24 MB/s** | **gate pass** |

Every 32-vCPU request landed on the same slow host; 64 placed elsewhere. The gate is one curl:

    curl -sL -o /dev/null -w "%{speed_download}" --max-time 60 \
      https://github.com/duckdb/duckdb/releases/download/v1.5.5/duckdb_cli-linux-amd64.zip

**Use `-L`.** That URL 302-redirects, and without `-L` curl reports 0 B/s on a *healthy* host —
which looks identical to the failure it is meant to detect.

`vcpuCount` must be a power of 2. 24 and 20 are rejected with HTTP 500, so the sizes are
16 (~29 GB), 32 (~60 GB), 64 (~119 GB).

**Terminated today, do not look for them:** `hwkx2c4ceogn71` (16 vCPU, after its nine results
were fetched), `7lewu2k1gpbrj5`, `nu6613z93o7brh`, `wzf9y402jbjz1g`, `t5qlpktoa1bmjh`,
`vv13qnibro08l3`.

### Before terminating — nothing on this pod survives it (`volumeInGb: 0`)

1. **Fetch the reports.** They are the only copy of the pod runs:
   `scp -P <port> root@<ip>:/workspace/duckdb-rocket/reference/phase5_*.json reference/`
   Ten files. `data/` is gitignored, so `predictions.json` is not archived and does not need
   fetching; the reports carry accuracy, timing, row alignment and the environment block.
2. **Archive the environment tuple.** `bootstrap.sh` runs `scripts/doctor.py` without `--json`,
   so the pod's tuple only ever went to `/root/bootstrap.log` and dies with the box. Run
   `uv run python scripts/doctor.py --json reference/pod_doctor.json` on the pod and fetch it.
   PLAN.md: "a number without it is not attributable".
3. **Update `reference/RESULTS.md`.** Its Phase 5 section still says "Breadth: eight datasets"
   with local timings and the old environment header. Ten datasets now, from the pod.
4. **Update PLAN.md's Phase 5 results table** — provenance column and ECG5000's row.
5. `python scripts/pod/runpod_cpu.py terminate hwkx2c4ceogn71 --yes-destroy-the-volume`
6. `python scripts/pod/runpod_cpu.py check` — confirm nothing of ours is still billing. The
   three `pattern-arm-*` GPU pods are **not ours**; leave them.

### Scripts left running on the pod

`/root/finish.sh` -> `/root/finish.log`. Waits for `/root/regen.log` to say `REGEN COMPLETE`,
re-runs any report carrying a failure entry (see below), then retries ECG5000. Ends with
`ALL COMPLETE`. Per-dataset logs are `/root/run_*.log`, `/root/regen_*.log`,
`/root/recheck_*.log`, `/root/ecg.log`.

Two facts about it that were not what the recipe asked for:

- **32 GB RAM, and ItalyPowerDemand measured 25.7 GB.** `runpod_cpu.py` picks compute-optimised
  flavors first (`cpu5c`), ~2 GB/vCPU. Its comment claims the sweep is "not memory-bound"; the
  25.7 GB measurement falsifies that. ECG5000 is 4.4x larger and will not fit — it needs either a
  general-purpose flavor (`cpu*g`, ~4 GB/vCPU), more vCPUs, or DuckDB spilling to disk via an
  explicit `memory_limit` + `temp_directory`.
- **`volumeInGb: 0`** — the requested 40 GB volume was not attached. Only the 60 GB container
  disk exists, so **nothing on this pod survives termination**. Copy results off before
  terminating, or they are gone.


Written mid-task because the box ran out of memory. Everything below is verified state, not plans.

## Safe: committed and pushed

`main` is in sync with `origin/main`. Nothing is uncommitted.

- `78d630a` — descriptor + pure-SQL scope docs (recovered from a hung session, see below)
- `b3429fd` — hoisted the reference-length flatten out of the per-row loop in
  `src/rocket_extension.cpp`. Build clean (9/9), 114 python tests, 25 sqllogictest assertions
- `7944d44` — `.gitattributes` now pins the working tree to LF, not just the index. 30 files had
  drifted to CRLF; `build_extension.bat` had drifted the *other* way and was violating its own
  `eol=crlf` rule. All verified with `git check-attr` and a byte-level check

## BLOCKER: Phase 5's last two datasets cannot run on this machine

Phase 5 needs 10 datasets (`UCR_SUBSET` in `duckdb_rocket/datasets.py`). Eight are done and
archived in `reference/phase5_*.json`. The two remaining are **ItalyPowerDemand** and **ECG5000**.

**ItalyPowerDemand was attempted locally and consumed 25.7 GB before being killed.** It died in
step `[3/3] running the whole pipeline in DuckDB` — the DuckDB child process
(`build/release/duckdb.exe -f data/phase5/ItalyPowerDemand/pipeline.sql`), not Python. No
`predictions.json` was produced. Both datasets already have `pipeline.sql` and `raw.parquet`
staged on disk from an earlier session; only execution is missing.

Why this is a wall and not bad luck:

| | test rows | in-context train rows | result |
|---|---|---|---|
| Largest completed (SyntheticControl) | 300 | 300 | 674s, fine |
| ItalyPowerDemand | 1029 | 67 | **25.7 GB, killed** |
| ECG5000 | 4500 | 500 | not attempted |

Every completed dataset is ≤300 test rows. ItalyPowerDemand is the first past that line and it
blew up 3.4× beyond the largest one that worked.

**Where the memory actually goes — corrected.** The first guess written here was "the 40
per-group feature tables". That was wrong by three orders of magnitude: 500 features x 1029 rows
is ~4 MB. The memory is inside the single `tabfm_classify` call, in ONNX allocations DuckDB's
buffer manager never sees — which is why `SET memory_limit` and `--threads 1` changed nothing.
The 6 GB limit run died *faster* (10.2s) than the 20 GB one (25.9s).

## RESOLVED — chunk the classify call

`--test-chunk N` issues one `tabfm_classify` per N test rows instead of one for the whole split.
Peak memory becomes a function of N rather than of the dataset.

**It is identity-preserving, and that was verified rather than argued.** An in-context learner
treats each test row as an independent query against the train context, so a row's prediction
cannot depend on which other rows shared its call. The control: GunPoint at `--test-chunk 50`
against its own unchunked run, same pod, same commit — **150/150 ids, 0 rows disagreeing**.
Accuracy alone would not have settled it; two runs can match on accuracy and disagree on which
rows they got right.

**It is nearly free.** Chunked GunPoint 248.7s vs 258s unchunked. 3x the classify calls, no
measurable cost — the model load amortises across calls.

**This supersedes the "no pod size fixes ECG5000" claim made earlier in this session.** That
extrapolated ~120 GB for ECG5000 from a two-point linear fit. With the peak bounded by chunk
size instead, ECG5000's 4500 rows are ~36 calls per group at the same ~12 GB ceiling, and the
29 GB pod is sufficient. Runtime, not memory, is now the open question for it.

Note the axis. swan's `predict_ensemble()` caps `context_rows` — the *train* side — which does
change predictions and is why they call it an ensemble. This chunks the *test* side, which does
not. Our train contexts are 50-67 rows across these datasets while test rows go 150 -> 4500, so
the test axis was the one that mattered here; swan's lever would not have helped.

### DECIDED: pod only. Do not run ItalyPowerDemand or ECG5000 locally again.

Owner's instruction, 2026-08-12. This matches PLAN.md line 496 (`Run the 10-dataset subset on a
pod`) and the standing rule that every number in a table comes from a pod, not the 3060.

Use `scripts/pod/runpod_cpu.py` — a CPU pod is correct here, not GPU: `anofox_tabfm`'s ONNX
Runtime is CPU-only, and the project already paid 140 minutes for an idle GPU card learning that.

    python scripts/pod/runpod_cpu.py check                    # read-only, run before AND after
    python scripts/pod/runpod_cpu.py plan                     # read-only
    python scripts/pod/runpod_cpu.py create --yes-i-will-pay  # billable
    python scripts/pod/runpod_cpu.py ssh POD_ID               # prints the bootstrap line
    python scripts/pod/runpod_cpu.py terminate POD_ID --yes-destroy-the-volume

Defaults: 16 vCPU, ~$0.56/hr, 60 GB container + 40 GB volume. Credentials are present
(`~/.runpod/token.txt`, ssh pubkey). `TABPFN_TOKEN` must be injected into the pod — local runs
read a cached token that pods do not have.

**Open question for the pod run:** ItalyPowerDemand needed 25.7 GB. Confirm the chosen CPU flavor
has headroom before starting ECG5000, which is 4.4x larger. Check RAM on the pod first rather
than discovering it the way this machine did.

**Not chosen:** making the pipeline stream groups rather than materialize all 40
(`scripts/phase5_pipeline.py`). Only revisit if the pod lane turns out to be blocked — it is real
work and it changes what is being measured.

### `check` output at handoff time

    vii1yh4zx07q6d   pattern-arm-D          RUNNING   GPU   $0.44/hr
    4f1zckwdzemdfy   pattern-arm-h15        RUNNING   GPU   $0.44/hr
    k8uts5vfewfn98   pattern-arm-7B         RUNNING   GPU   $0.44/hr
    8nfeg9gejlwqbz   duckdb-rocket-sweep    EXITED    GPU   $0.74/hr

Three pods are RUNNING and billing $1.32/hr combined. **The `pattern-arm-*` pods were not
touched** — the account is shared and they are not this project's. Confirm ownership before
acting on them. `duckdb-rocket-sweep` is this project's old GPU pod, EXITED; worth checking
whether its volume is still billing.

## Not done: PLAN.md is stale

`PLAN.md` lines 493-499 has all five Phase 5 boxes unchecked, but eight datasets have run:

| Dataset | Accuracy | Seconds | Test rows |
|---|---|---|---|
| BasicMotions (multivariate) | 1.0000 | 131.5 | 40 |
| Beef | 0.7667 | 129.0 | 30 |
| Coffee | 1.0000 | 128.0 | 28 |
| FaceFour | 0.9773 | 171.9 | 88 |
| GunPoint | 0.9933 | 258.0 | 150 |
| OSULeaf | 0.9711 | 448.5 | 242 |
| SyntheticControl | 0.9867 | 674.2 | 300 |
| Trace | 1.0000 | 260.0 | 100 |

Zero row-alignment failures anywhere. GunPoint is the only one with a Phase 3 comparison:
delta 0.0 and **identical per-row predictions**, which is the end-to-end statement that the C++
transform is interchangeable with the Python oracle.

All eight carry `"caveat": "local Windows timing on a contended box"` — accuracy is valid,
timings are not reportable.

This reconciliation was the committed next step and is **not** done. It was deliberately held
until the last two datasets landed, so the checkboxes could be ticked truthfully in one pass
rather than half-ticked.

## Upstream work: `maxdemarzi/anofox-tabfm` (fork, MIT)

Forked from `DataZooDE/anofox-tabfm`, cloned locally to `C:\Users\maxde\Repositories\anofox-tabfm`
with `upstream` set. Three container-blindness findings came out of Phase 5; one is fixed in our
own pipeline, two are upstream-shaped and written up.

**Branch `container-aware-thread-default` — pushed, NOT built, NOT tested.**
`src/tabfm_settings.cpp` defaulted ONNX intra-op threads to
`std::thread::hardware_concurrency() / 2`, which reads the *host*. On a 64-core cpuset inside a
256-core host that is `anofox_tabfm_threads = 128` per session, and DuckDB builds one session per
concurrent task — 132 threads in one process, load average 143, for a query with `SET threads = 4`.
The branch swaps in `sched_getaffinity`, falling back to `hardware_concurrency()` off Linux and on
failure. Needs a Linux box to:

1. build the extension and confirm it still compiles clean;
2. confirm the default resolves to cores/2 **of the cpuset** (32, not 128, on that pod);
3. confirm no change on a host where the two counts agree;
4. only then decide whether to offer it upstream. No PR has been opened.

**Issues, on the fork** (GitHub disables issues on forks by default; enabled deliberately):

- [#1](https://github.com/maxdemarzi/anofox-tabfm/issues/1) — `tabfm_classify` re-encodes the whole
  train context on every call. 40 encodings are irreducible (one per feature group); the rest are
  repetition. ~78% of ECG5000's model work is 1,400 redundant encodings of one 500-row context.
- [#2](https://github.com/maxdemarzi/anofox-tabfm/issues/2) — classify memory is outside DuckDB's
  buffer manager, so `SET memory_limit` cannot bound it and a cgroup kill leaves no error. 20 GB vs
  8 GB moved the plateau by 0.00 GB.

Nothing has been sent to DataZoo. PLAN.md's etiquette (issue before large PR; no CONTRIBUTING
guide; anofox is a commercial product) is why these sit on the fork.

**Already fixed on our side, needing no upstream change:** `anofox_tabfm_threads` was always
settable, so `scripts/phase5_pipeline.py` now sets it to cores/duckdb_threads (`656c52f`).

## Also open, lower priority

- `description.yml:20` still has `ref: TODO-pin-a-commit-before-submitting`. The community-
  extensions repo builds from that exact revision, so it must be a real commit before submission.
  The file's own header also gates submission on Phase 5 breadth, which is not yet complete.

## Housekeeping done during the restart

- Killed PID 19124, a Claude session hung since 10:37:52 mid-`Edit`. Its work was recovered and
  is in `78d630a` + `b3429fd`; nothing was lost.
- Killed the runaway `duckdb.exe` (PID 29720, 25.7 GB). Free memory went 8.4 GB -> 30.2 GB.
- No duckdb-rocket process is running as of this writing. Three unrelated python jobs were left
  alone: `serve.py`, `f345_load.py`, `evaluate_bird.py`.
