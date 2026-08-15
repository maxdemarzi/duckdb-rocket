#!/usr/bin/env bash
# What routing costs, and which of its two knobs are set too high. CPU pod.
#
# **Why a pod rather than this workstation.** Every number here is a timing or is read against one,
# and the workstation is running the user's own jobs at 100% CPU. RESULTS.md already measures ~1.8x
# inflation from background load alone, and a previous "138x" figure had to be retracted because it
# compared a contended 8-core measurement against a 96-core CUDA one. A rented box running nothing
# else is the cheapest way to make the comparison mean something.
#
# Three experiments, in an order chosen so the timed one runs alone:
#
#   1. cost, all three arms at one moment (route_serve.py --compare). Runs FIRST, on an otherwise
#      idle box, because it is the only one whose answer is a wall-clock number.
#   2. the student's kernel count. Accuracy in parallel, then a single-job timing pass.
#   3. the teacher's group count -- the expensive one, saturating the box for hours.
#
#   BUDGET_MIN=240 bash scripts/pod/perf_levers_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/levers
BUDGET_MIN="${BUDGET_MIN:-240}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"
SHARDS="${SHARDS:-4}"
TIMING_SETS="${TIMING_SETS:-Herring ScreenType SemgHandMovementCh2}"

log() { printf '\n=== %s  [%s]\n' "$*" "$(date -u +%H:%M:%S)"; }

log "prerequisites"
for t in git curl unzip; do
    command -v "$t" >/dev/null || {
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl unzip ca-certificates >/dev/null
        break
    }
done
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
NPROC=$(nproc)
echo "  ${NPROC} vCPU"

log "repository"
[ -d "$R/.git" ] || git clone -q --depth 1 https://github.com/maxdemarzi/duckdb-rocket.git "$R"
cd "$R"
git pull -q --ff-only || true
source scripts/pod/shallow_clone.sh
shallow_submodule duckdb || { echo "FATAL: no duckdb submodule"; exit 1; }
git log --oneline -1
uv sync -q
mkdir -p "$OUT"

log "prebuilt rocket shell"
KEY=$(git rev-parse "HEAD:src" "HEAD:CMakeLists.txt" "HEAD:extension_config.cmake" "HEAD:duckdb" \
      | sha256sum | cut -c1-12)
mkdir -p build/release
if curl -fsSL -o build/release/duckdb "https://github.com/maxdemarzi/duckdb-rocket/releases/download/prebuilt/duckdb-rocket-${KEY}-linux_x86_64" 2>/dev/null; then
    chmod +x build/release/duckdb
    build/release/duckdb -c \
      "SELECT len(rocket_transform([1.0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]::DOUBLE[],3,0,0));" \
      >/dev/null 2>&1 && echo "  SHELL CACHE HIT ($KEY)" || { echo "  smoke test failed"; rm -f build/release/duckdb; }
else
    echo "  SHELL CACHE MISS ($KEY)"
fi
[ -x build/release/duckdb ] || { echo "FATAL: no cached shell and this script does not build one"; exit 1; }

log "extension and weights"
# httpfs is REQUIRED and its absence is not obvious: without it tabfm_download cannot reach
# HuggingFace and every dataset dies with "no downloaded weights", which reads as a dataset failure.
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabicl-v2');" >/dev/null 2>&1 \
  && echo "  tabicl-v2 ok" || { echo "FATAL: no weights"; exit 1; }

# ---------------------------------------------------------------------------------------------
# 1. Cost, measured. FIRST, and nothing else running.
# ---------------------------------------------------------------------------------------------
log "experiment 1: three arms, one box, one moment"
echo "  the box must be idle for this; it is, because nothing below has started yet"
for ds in $TIMING_SETS; do
    echo
    echo "  --- $ds"
    uv run python scripts/route_serve.py deploy --dataset "$ds" --target 0.20 \
        > "$OUT/timing_${ds}_deploy.log" 2>&1
    # No pipe into tail. A previous sweep lost its second model because the launching ssh died and
    # the next `| tail` took SIGPIPE; redirect to a file and read it afterwards instead.
    uv run python scripts/route_serve.py serve --dataset "$ds" --batch 128 --compare \
        > "$OUT/timing_${ds}.log" 2>&1
    echo "    rc=$?"
    sed -n '/batch of/,$p' "$OUT/timing_${ds}.log" | head -30
done

# ---------------------------------------------------------------------------------------------
# 2. The student's kernel count.
# ---------------------------------------------------------------------------------------------
log "experiment 2: student kernels, accuracy (parallel)"
uv run python scripts/perf_levers.py --kernels \
    --from-gate reference/distill_gate.json --max-student 0.90 \
    --jobs "$NPROC" --out "$OUT/perf_kernels.json" > "$OUT/kernels.log" 2>&1
echo "  rc=$?"
sed -n '/STUDENT KERNELS/,$p' "$OUT/kernels.log"

log "experiment 2b: student kernels, transform cost (one job, so the ms/row means something)"
# Three datasets rather than all 28: the transform is a fixed number of multiply-accumulates per
# kernel per timepoint, so the SHAPE of this curve is arithmetic and only its slope is measured.
uv run python scripts/perf_levers.py --kernels --datasets $TIMING_SETS \
    --jobs 1 --out "$OUT/perf_kernels_timed.json" > "$OUT/kernels_timed.log" 2>&1
echo "  rc=$?"
sed -n '/STUDENT KERNELS/,$p' "$OUT/kernels_timed.log"

# ---------------------------------------------------------------------------------------------
# 3. The teacher's group count. Long, and it saturates the box.
# ---------------------------------------------------------------------------------------------
log "experiment 3: per-group cubes at G=40, ${SHARDS} shards"
# threads x onnx-threads x shards must land near the vCPU count. phase5's own defaults size
# themselves from the VISIBLE cores, so N concurrent runs each claim the whole box -- which is what
# killed a pod sweep once. Set explicitly here instead.
PER=$(( NPROC / SHARDS )); [ "$PER" -lt 1 ] && PER=1
T=2; O=$(( PER / T )); [ "$O" -lt 1 ] && O=1
echo "  ${SHARDS} shards x ${T} duckdb threads x ${O} onnx threads = $(( SHARDS * T * O )) of ${NPROC} vCPU"

# Sharded round-robin over the cost-sorted subgroup, so no shard gets all the big datasets.
mapfile -t SHARD_LIST < <(uv run python - "$SHARDS" <<'PY'
import json, sys
shards = int(sys.argv[1])
rows = json.load(open("reference/distill_gate.json"))["rows"]
keep = [r for r in rows if r.get("students") and max(r["students"].values()) < 0.90]
keep.sort(key=lambda r: -r.get("n_test", 0))
buckets = [[] for _ in range(shards)]
for i, r in enumerate(keep):
    buckets[i % shards].append(r["dataset"])
for b in buckets:
    print(" ".join(b))
PY
)
for i in "${!SHARD_LIST[@]}"; do
    echo "  shard $i: ${SHARD_LIST[$i]}"
done

PIDS=()
for i in "${!SHARD_LIST[@]}"; do
    # shellcheck disable=SC2086
    uv run python scripts/teacher_sweep.py --model tabicl-v2 \
        --datasets ${SHARD_LIST[$i]} \
        --out-dir "$OUT/pergroup" --device cpu --per-group-soft \
        --threads "$T" --onnx-threads "$O" --test-chunk 128 \
        --budget-min "$BUDGET_MIN" --timeout-min "$TIMEOUT_MIN" \
        > "$OUT/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done
echo "  ${#PIDS[@]} shards running: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "  shards done, $FAIL non-zero"
echo "  cubes: $(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l)"

log "experiment 3: accuracy and routing as a function of G"
uv run python scripts/perf_levers.py --groups --pergroup "$OUT/pergroup" \
    --out "$OUT/perf_groups.json" > "$OUT/groups.log" 2>&1
echo "  rc=$?"
sed -n '/TEACHER GROUPS/,$p' "$OUT/groups.log"

log "what came out"
ls -la "$OUT"/*.json 2>/dev/null
echo "  cubes: $(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l) of 28"
log "done"
