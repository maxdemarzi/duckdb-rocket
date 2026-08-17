# Running this project on a pod

Every timing in [reference/RESULTS.md](../reference/RESULTS.md) came off a rented pod rather than
the workstation, and the reason is in PLAN.md's risk table: local Windows timings mislead, because
WDDM spills to host memory instead of failing, so a run that would OOM on a container merely gets
slow. Accuracy reproduces locally; wall clock does not.

**This file is about `scripts/pod/`. [RUNPOD.md](RUNPOD.md) is not** — that documents the
`black_swan` lane (`scripts/cloud/runpod_launch.py`, `rai-swan`, BIRD, Spider), which this repo's
launchers were ported from rather than share code with. If you are looking for `runpod_cpu.py`, it
is here.

## The account is shared, and that is the first thing to know

`check` lists pods belonging to other people. The `black_swan` project runs `eval-*`,
`pattern-arm-*`, `sft-*`, `harvest-*` and `sqlbase-*` pods on the same account.

**Confirm a pod is yours before stopping or terminating anything.** The names and creation times
are the only evidence available, and a wrong guess destroys someone else's work. Nothing in this
repo terminates a pod it did not create without you naming its id.

```bash
python scripts/pod/runpod_cpu.py check          # read-only: what is billing right now, whose
python scripts/pod/pod_audit.py                 # sharper: which pods bill WITHOUT working
```

`pod_audit.py` exists because every way this project has lost money looks identical from the
outside — `RUNNING`, $0.44/hr, and nothing happening on it. It asks each pod rather than inferring.

## Choosing a pod

```bash
python scripts/pod/runpod_cpu.py plan                        # the exact create body and $/hr, no call
python scripts/pod/runpod_cpu.py create --vcpu 32 --yes-i-will-pay
python scripts/pod/runpod_gpu.py gpus                        # live prices and stock
python scripts/pod/runpod_gpu.py create --gpu "NVIDIA A40" --yes-i-will-pay
```

`create` is the only subcommand that spends money and it refuses to run without
`--yes-i-will-pay`. Everything else is read-only — that split is deliberate: the expensive
decision should be one reviewed command, and the twenty things you want to know first should cost
nothing.

**Prefer CPU.** `anofox_tabfm`'s ONNX Runtime is CPU-only in the community build, so a GPU pod's
card sits idle — that was measured, not assumed, and it cost 140 minutes once. CPU pods bill at
about $0.035/vCPU/hr, so 32 vCPU is roughly $1.12/hr. **Fewer vCPUs is the real lever on cost**,
not CPU-versus-GPU: a 16 vCPU CPU pod is only ~25% under the RTX 6000 Ada this project first used.

**`--vcpu` must be a power of two** — 16, 32, 64 — and asking for a different size is also how you
get placed on a different host. When capacity is exhausted at every size (it happens; all four
`CPU_FLAVORS` are tried per attempt), a GPU pod running `--device cpu` is a valid fallback. Say so
in the write-up, because the environment tuple changes.

**Do not loop over sizes without breaking on success.** Three pods were created at once that way,
all billing, in this repo's history.

## Gate before you bootstrap

```bash
python scripts/pod/runpod_cpu.py gate POD_ID    # read-only; default floor 5 MB/s
```

Placement varies by host and some are unusably slow. Skipping this cost a session: four
consecutive 32 vCPU pods landed on one host at ~0.08 MB/s and the first sat 29 minutes without
cloning a single object. The loop is **create → gate → (PASS: bootstrap) or (FAIL: terminate,
create again)**.

`runpod_gpu.py` additionally refuses placements on a region and machine deny list —
currently region `SE`, whose failures are *transport* (`scp: Connection closed` at the moment
results would come off the machine) rather than compute. `--allow-region` overrides it, but weigh
that against what you are about to transfer: a 69 MB extension upload is exactly the case SE breaks.

## Bootstrapping

```bash
ssh -p PORT root@IP 'bash -s' < scripts/pod/bootstrap.sh
# or, once the repo is on the pod:
ssh -p PORT root@IP 'cd /workspace/duckdb-rocket && bash scripts/pod/bootstrap.sh'
```

Idempotent — every step checks before doing, so re-running after a dropped SSH session costs
seconds rather than another full build.

It finishes by **verifying rather than launching**: `conformance.py` checks the freshly built Linux
extension against the same golden vectors the Windows build is held to, then `doctor.py` records the
environment tuple, then it stops and tells you what to run. The conformance step is the one worth
waiting for — it is what makes a pod timing comparable to a local one instead of merely adjacent to
it, since it proves the two builds compute the same features before either produces a number.

Note that `doctor.py` also reports whether the CPU has native bf16, and on hosts that do, TabPFN's
default `inference_precision="auto"` would run bf16 autocast on CPU. A CPU-versus-GPU agreement
check on such a box is not evidence of fp32 correctness.

`shallow_clone.sh` is why it is quick. A bare `submodule update --init` pulls duckdb's entire
7.8 GB history to build one pinned commit; fetching the commit by SHA takes 11 seconds and 102 MB.
`--depth 1` does not work here — it shallow-fetches the submodule's default branch tip, and the pin
is not the tip.

## The box lies about its size

**Read the cgroup, never `nproc` or `free`.** On one RunPod GPU host `nproc` reported 112 and
`sched_getaffinity` agreed, while `cpu.cfs_quota_us` said **11.9** — the pod was sold as 14 vCPU.
`free` reported 629 GB against a 72.6 GB cgroup limit.

This has now cost this project four separate failures, three of them in one session:

| what read the wrong number | how it failed |
|---|---|
| ONNX intra-op threads from `hardware_concurrency()/2` | 128 threads per session on a 64-core pod, load average 143 |
| four concurrent runs each sizing pools from the visible count | every run died near completion, no error at all |
| `memory_limit` at 70% of the cgroup, per run | at `--jobs 6`, six runs each claimed 44.8 GB of 64 GB; **38 of 160 runs at exit −9** |
| DuckDB sizing itself against RAM it cannot have | slow query becomes dead process, no traceback |

`phase5_pipeline.py` reads the cgroup for its own defaults. **A driver running jobs in parallel
must go further and divide**, because only the driver knows the job count:
`jobs x threads x onnx-threads` is what the box actually carries, and memory has to be split the
same way. `sweep.py` and `resample_power.py` both do this now.

An OOM kill here leaves **exit −9, no DuckDB error and no Python traceback**. It looks exactly like
a hang. `anofox_tabfm_max_memory` turns it into a message you can read, and is off by default
because it is unreleased upstream.

## Two more things concurrency breaks

**Give every parallel run its own `--workdir`.** The default `data/phase5/<dataset>` holds
`raw.parquet`, `predictions.json` and the temp directory, and `predictions.json` is read back to
compute accuracy. Two concurrent runs of one dataset can therefore each report the other's number
with nothing crashing. The loud version — a parquet read mid-write, `No magic bytes found at end of
file` — is the lucky outcome.

**Warm the dataset cache serially first.** The archive downloads and extracts on first use and that
is not safe from several processes at once. Both drivers now do this; the symptom otherwise is one
run dying in ~1 s on a half-written file, and it only ever hits the *first* run of each dataset,
which is exactly what a fresh pod always performs.

## The drivers

```bash
uv run python scripts/pod/sweep.py --seeds 3 --jobs 4          # breadth: datasets x seeds
uv run python scripts/pod/resample_power.py --dry-run          # cost a resample campaign, spend nothing
uv run python scripts/pod/resample_power.py --datasets A,B --resamples 5 --jobs 4
```

Each `(dataset, seed)` or `(dataset, resample, arm)` is a separate `phase5_pipeline.py` process:
ONNX Runtime's API is process-global, a crash in one combination must not take the sweep with it,
and results are written after every run so a pod that dies three hours in keeps what it earned.

`--dry-run` costs a campaign off the archived wall clocks before you spend anything.
`--analyse-only` re-reads a finished run's JSON and recomputes the statistics with no pod at all.

The one-off measurement scripts are `scripts/pod/*_cpu.sh`, each with its question in its header —
`context_cache_phase5_cpu.sh`, `max_memory_cpu.sh`, `perf_levers_cpu.sh` and the rest. They are
kept rather than deleted because the header records what was being asked and what went wrong the
first time.

## Getting artefacts back, and shutting down

`scp` a tarball rather than a glob: a brace-expanded remote glob is expanded by the *local* shell
and fails as one long unmatched path.

```bash
ssh -p PORT root@IP 'cd /workspace && tar czf out.tgz results.json run.log'
scp -P PORT root@IP:/workspace/out.tgz .
python scripts/pod/runpod_cpu.py terminate POD_ID --yes-destroy-the-volume
python scripts/pod/runpod_cpu.py check          # confirm, every time
```

`stop` keeps the volume **and keeps billing it**. `terminate` destroys it, and refuses without
`--yes-destroy-the-volume` so that a reflexive terminate cannot take the results with it.

**`check` before and after every session.** A forgotten pod is the expensive failure mode, and it
is silent.

## The credential

Read at the point of use from `RUNPOD_API_KEY` or `~/.runpod/token.txt` (override with
`RUNPOD_TOKEN_FILE`). It is never printed, logged, written, or interpolated into a command line.
`TABPFN_TOKEN` is injected into the pod environment the same way and belongs in no file here.

Cloudflare rejects RunPod's endpoint with 403/1010 when no `User-Agent` is sent, which reads
exactly like a bad key. The launchers always set one.
