#!/usr/bin/env bash
# The three things the first levers run did not deliver, in the order their measurements need.
#
# **What went wrong the first time, because the order here is the fix.** The kernel sweep put one
# worker per (dataset, kernel size), so six workers reached for the same cold aeon archive at once.
# That corrupted reads -- 44 of 168 fits died on half-written .ts files -- and it hammered Zenodo
# hard enough that the group sweep, starting five minutes later, could not download its datasets at
# all and lost 24 of 28 to a failure that reads as rc=1 with an empty stderr.
#
# Nothing here downloads under concurrency. The cache is warmed once, serially, up front; every
# dataset is then local and no later stage can race for one.
#
#   BUDGET_MIN=240 bash scripts/pod/perf_levers_recover.sh
set -uo pipefail
cd /workspace/duckdb-rocket
export PATH="$HOME/.local/bin:$PATH"

OUT=/workspace/levers
BUDGET_MIN="${BUDGET_MIN:-240}"
TIMEOUT_MIN="${TIMEOUT_MIN:-60}"
# Two shards, not four, and an explicit memory budget each. phase5 sizes memory at 70% of the
# BOX, so four concurrent shards asked for 280% of a 124GB pod and the OOM killer took every one
# of them -- exit -9 in 18 seconds, while the same dataset run alone finished in 598. The limit
# only governs DuckDB (ONNX allocates outside it), so the shard count carries the rest of the
# safety margin.
# SHARDS x THREADS is a count of concurrent tabfm_classify calls, and on a RunPod CPU pod the
# budget for those is **the cgroup limit, not the host RAM**. This box reports 124GB to `free` and
# is capped at 29.8 GiB; memory.max_usage_in_bytes sat exactly on the cap after every kill. One
# call needs 11.75GB for a 50-row train context and 28.7GB for a 500-row one (RESULTS.md, "The
# classify call's memory"), which is the same 29.8GB ceiling ECG5000 hit there.
#
# So the answer is 1. Two concurrent calls cannot fit whatever else is tuned, and datasets with a
# ~600-row train context are expected to fail even alone -- the linear fit through those two
# measured points puts them at ~32GB.
SHARDS="${SHARDS:-1}"
THREADS="${THREADS:-1}"
MEMLIMIT="${MEMLIMIT:-24GB}"
# THREADS is the one that decides whether this survives, and it is not about CPU. DuckDB runs
# `--threads` classify calls at once, and RESULTS.md ("The classify call's memory") measures each
# one at 11.75GB for a 50-row train context and 28.7GB for a 500-row one -- allocated by ONNX,
# OUTSIDE DuckDB's buffer manager, so --memory-limit does not bound it. 2 shards x 4 threads is
# therefore up to 8 concurrent calls, ~160GB on a 124GB box, and the OOM killer took them in the
# order they got big. Concurrency here must be counted in classify calls: SHARDS x THREADS.
# These datasets are single-chunk at --test-chunk 128, so threads buy almost nothing anyway.
ONNX_THREADS="${ONNX_THREADS:-8}"
NPROC=$(nproc)

log() { printf '\n=== %s  [%s]\n' "$*" "$(date -u +%H:%M:%S)"; }

log "repository"
git pull -q --ff-only || true
git log --oneline -1
uv sync -q
mkdir -p "$OUT"

log "warming the dataset cache, serially, before anything runs concurrently"
# This is the step whose absence cost the first run. Serial, retried, and loud about what it cannot
# get -- a dataset missing here is one to drop from the plan, not one to discover inside a worker.
uv run python - <<'PY' 2>&1 | tail -20
import json, sys, time
sys.path.insert(0, ".")
from duckdb_rocket.datasets import load
rows = json.load(open("reference/distill_gate.json"))["rows"]
want = sorted(r["dataset"] for r in rows
              if r.get("students") and max(r["students"].values()) < 0.90)
bad = []
for n in want:
    for attempt in (1, 2, 3):
        try:
            load(n, "train"); load(n, "test"); break
        except Exception as e:
            if attempt == 3:
                bad.append(n); print(f"  UNAVAILABLE {n}: {type(e).__name__}: {e}"[:150])
            else:
                time.sleep(20)
print(f"  warm: {len(want) - len(bad)}/{len(want)} datasets local")
open("/workspace/levers/warm.json", "w").write(json.dumps({"ok": [n for n in want if n not in bad],
                                                          "bad": bad}))
PY

# ---------------------------------------------------------------------------------------------
log "1/4 the timing arm that died after group 1 -- idle box, nothing else started yet"
# SKIP_TIMING=1 once it is known to fail: its escalated arm is already measured and the full-batch
# arm has failed identically twice, so re-running it spends six minutes to reproduce a known bug
# rather than to learn anything.
if [ "${SKIP_TIMING:-0}" = "1" ]; then
  echo "  skipped (SKIP_TIMING=1)"
else
# SemgHandMovementCh2 is the longest series measured here (1500 timepoints) and the one whose
# full-batch arm died after group 1 of 40. It is first because it is the only stage whose answer is
# a wall-clock number.
uv run python scripts/route_serve.py serve --dataset SemgHandMovementCh2 --batch 128 --compare \
    > "$OUT/timing_SemgHandMovementCh2.log" 2>&1
echo "  rc=$?"
sed -n '/batch of/,$p' "$OUT/timing_SemgHandMovementCh2.log" | head -30
fi

# ---------------------------------------------------------------------------------------------
log "2/4 per-group cubes at G=40, ${SHARDS} x ${THREADS} = $((SHARDS * THREADS)) concurrent classify calls"
mapfile -t SHARD_LIST < <(uv run python - "$SHARDS" <<'PY'
import json, sys
shards = int(sys.argv[1])
ok = set(json.load(open("/workspace/levers/warm.json"))["ok"])
rows = [r for r in json.load(open("reference/distill_gate.json"))["rows"]
        if r.get("students") and max(r["students"].values()) < 0.90 and r["dataset"] in ok]
rows.sort(key=lambda r: -r.get("n_test", 0))
buckets = [[] for _ in range(shards)]
for i, r in enumerate(rows):
    buckets[i % shards].append(r["dataset"])
for b in buckets:
    print(" ".join(b))
PY
)
for i in "${!SHARD_LIST[@]}"; do echo "  shard $i: ${SHARD_LIST[$i]}"; done

PIDS=()
for i in "${!SHARD_LIST[@]}"; do
    # shellcheck disable=SC2086
    uv run python scripts/teacher_sweep.py --model tabicl-v2 \
        --datasets ${SHARD_LIST[$i]} \
        --out-dir "$OUT/pergroup" --device cpu --per-group-soft \
        --threads "$THREADS" --onnx-threads "$ONNX_THREADS" --memory-limit "$MEMLIMIT" --test-chunk 128 \
        --budget-min "$BUDGET_MIN" --timeout-min "$TIMEOUT_MIN" \
        > "$OUT/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done
echo "  ${#PIDS[@]} shards running"
for p in "${PIDS[@]}"; do wait "$p"; done
echo "  cubes: $(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l)"

log "3/4 accuracy and routing as a function of G"
uv run python scripts/perf_levers.py --groups --pergroup "$OUT/pergroup" \
    --out "$OUT/perf_groups.json" > "$OUT/groups.log" 2>&1
echo "  rc=$?"
sed -n '/TEACHER GROUPS/,$p' "$OUT/groups.log"

log "4/4 student kernels, re-run with the cache already warm"
uv run python scripts/perf_levers.py --kernels \
    --from-gate reference/distill_gate.json --max-student 0.90 \
    --jobs "$NPROC" --out "$OUT/perf_kernels.json" > "$OUT/kernels.log" 2>&1
echo "  rc=$?"
sed -n '/STUDENT KERNELS/,$p' "$OUT/kernels.log"

log "done"
