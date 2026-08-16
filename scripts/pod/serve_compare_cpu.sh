#!/usr/bin/env bash
# Does the configuration we now ship actually cost what the analysis said? CPU pod.
#
# `route_serve` defaults to G=10 / 2,500 kernels as of 1d290bd, chosen from an OFFLINE analysis:
# per-group cubes averaged at several prefixes, plus a linear cost model. Nothing has ever served at
# it. The "227.5 s -> ~60 s" in RESULTS.md is arithmetic on a G=40 measurement, not a measurement.
# This runs the three arms at the shipped default so the number is observed.
#
# Two other things fall out of the same run, which is why it is worth a pod:
#
#   * the G=40 arm on the same box at the same moment. Every G=10-vs-G=40 cost claim here is
#     currently a division; this makes it a pair of wall clocks. Run SECOND, so the default gets
#     the quiet box.
#   * SemgHandMovementCh2, which has failed its full-batch arm deterministically at 128 rows and
#     was never re-run after crash.log and the exit code were added. Its retry ladder walks
#     --test-chunk down, which is the lever budget.py names for this and which route_serve only
#     grew a flag for today.
#
#   bash scripts/pod/serve_compare_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/serve_compare
# Herring and ScreenType are the two the archived cost model was fitted on -- 64 and 375 training
# rows, which is what makes the fixed term's dependence on the CONTEXT visible. Semg is the crash.
SETS="${SETS:-Herring ScreenType SemgHandMovementCh2}"
# Walked down only on failure. 128 = one call for a default batch, and is what the archive used.
CHUNKS="${CHUNKS:-128 32 8}"

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
echo "  $(nproc) vCPU"
# The cgroup cap, not `free`. A container on a 124 GB host reported 124 GB and was killed at 29.8:
# that gap is what made an OOM read as a dataset failure for a whole session.
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] && echo "  cgroup memory: $(cat "$f")"
done
free -g | sed -n 2p

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

log "warming the dataset cache, serially"
# Before anything is timed, and before anything forks. Six workers pulling the same cold aeon
# archive at once produced 44 bogus failures in an earlier sweep and got this account rate-limited
# by Zenodo hard enough to cost the NEXT experiment 24 of its 28 datasets.
for ds in $SETS; do
    uv run python -c "
from duckdb_rocket.datasets import load
x, y = load('$ds', 'train'); print('  $ds train', x.shape)" 2>&1 | tail -1
done

# ---------------------------------------------------------------------------------------------
# The shipped default. First, on the quiet box.
# ---------------------------------------------------------------------------------------------
run_one() {  # dataset, groups, tag, chunk
    local ds="$1" g="$2" tag="$3" chunk="$4"
    uv run python scripts/route_serve.py serve --dataset "$ds" --batch 128 --compare \
        --n-groups "$g" --test-chunk "$chunk" > "$OUT/${ds}_${tag}.log" 2>&1
    local rc=$?
    echo "    rc=$rc  (chunk $chunk)"
    sed -n '/batch of/,$p' "$OUT/${ds}_${tag}.log" | head -40
    return $rc
}

for g in 10 40; do
    log "G=$g -- deploy and serve, three arms, one box, one moment"
    [ "$g" = 10 ] && echo "  this is the shipped default; the box is idle because nothing else has run"
    for ds in $SETS; do
        echo
        echo "  --- $ds at G=$g"
        # deploy is what couples n_kernels to n_groups: G=10 deploys 2,500 kernels, G=40 deploys
        # 10,000. Re-deployed per arm rather than served with an override, because an override that
        # changed kernels-per-group is exactly what the guard added in 1d290bd refuses.
        uv run python scripts/route_serve.py deploy --dataset "$ds" --target 0.20 --n-groups "$g" \
            > "$OUT/${ds}_g${g}_deploy.log" 2>&1 || { echo "    DEPLOY FAILED"; tail -5 "$OUT/${ds}_g${g}_deploy.log"; continue; }
        grep -E "threshold|student:|teacher:" "$OUT/${ds}_g${g}_deploy.log" | sed 's/^/    /'
        # The ladder: full batch first, and only step down if it dies. A smaller chunk is not free
        # -- it pays the context pass again per chunk -- so a result at chunk 8 is a different
        # measurement from one at 128 and is labelled as one.
        for chunk in $CHUNKS; do
            run_one "$ds" "$g" "g${g}_c${chunk}" "$chunk" && break
            echo "    (failed at chunk $chunk; crash.log below, then stepping down)"
            head -20 "data/serve/$ds/work/all/crash.log" 2>/dev/null | sed 's/^/      /'
            head -20 "data/serve/$ds/work/crash.log" 2>/dev/null | sed 's/^/      /'
        done
    done
done

log "what came out"
ls -la "$OUT"/*.log 2>/dev/null
# The three lines per run that carry the answer, side by side.
for f in "$OUT"/*_g*_c*.log; do
    [ -e "$f" ] || continue
    echo "--- $(basename "$f")"
    grep -E "^  (routed|student|teacher) |escalating [0-9]|s fixed per group|Per-group spread" "$f"
done
log "done"
