#!/usr/bin/env bash
# Does `anofox_tabfm_max_memory` turn an OOM kill into an error you can read?
#
# `SemgHandMovementCh2` at --test-chunk 128 is killed by the kernel on a 32 GB container: exit -9,
# empty stderr, no DuckDB error, nothing in the log but a group counter that stops. That failure
# cost this project several sessions and one diagnosis that had to be withdrawn -- an 8 GB
# memory_limit was blamed and then exonerated when the same run died at ~87 GB. memory_limit cannot
# help, because the model allocates outside DuckDB's buffer manager.
#
# Upstream #36 adds a setting that reads VmRSS from /proc/self/status and refuses a predict call
# above a ceiling. It is merged and in **no released build** -- the community extension is pinned
# at 2026.08.14 -- so this runs against the CI artifact from main (e5021d9), uploaded separately to
# /workspace/anofox_main/anofox_tabfm.duckdb_extension.
#
# Three arms, in order, on a box small enough to reproduce the failure:
#
#   1. the setting exists and the extension classifies at all
#   2. Semg at chunk 128 with NO ceiling -- must still die, or the test proves nothing
#   3. Semg at chunk 128 WITH a ceiling -- must fail with a message naming it
#
#   bash scripts/pod/max_memory_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/max_memory
EXT="${EXT:-/workspace/anofox_main/anofox_tabfm.duckdb_extension}"
DS="${DS:-SemgHandMovementCh2}"
CEILING="${CEILING:-16GB}"

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
# The cap this test needs to be small enough to hit. On a 64 GB box Semg completes at chunk 128 and
# arm 2 would pass, which would make arm 3 meaningless.
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] && echo "  cgroup memory: $(cat "$f")"
done
[ -f "$EXT" ] || { echo "FATAL: no extension at $EXT -- scp it from the CI artifact first"; exit 1; }
ls -la "$EXT"

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
[ -x build/release/duckdb ] || { echo "FATAL: no cached shell"; exit 1; }

log "weights (through the community build, which shares the cache)"
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabicl-v2');" >/dev/null 2>&1 \
  && echo "  tabicl-v2 ok" || { echo "FATAL: no weights"; exit 1; }

# ---------------------------------------------------------------------------------------------
# 1. The setting exists, and the main build works at all.
# ---------------------------------------------------------------------------------------------
log "arm 1: the setting, and a classification that cannot go wrong"
build/release/duckdb -c "LOAD '${EXT}';
  SELECT name, value, description FROM duckdb_settings() WHERE name LIKE 'anofox%';" \
  2>/dev/null | head -20
echo
build/release/duckdb -c "LOAD '${EXT}'; SET anofox_tabfm_accept_hf_license = true;
  SET anofox_tabfm_max_memory = '${CEILING}';
  CREATE TABLE t AS SELECT * FROM (VALUES
    (1.0,0.1,'a'),(1.1,0.2,'a'),(0.9,0.0,'a'),(1.2,0.1,'a'),
    (5.0,9.1,'b'),(5.1,9.2,'b'),(4.9,9.0,'b'),(5.2,9.1,'b')) v(f1,f2,y);
  CREATE TABLE q AS SELECT * FROM (VALUES (1.05,0.15),(5.05,9.15)) v(f1,f2);
  SELECT string_agg(yhat, ',') AS got FROM tabfm_classify('t','y', test := 'q',
    model := 'tabicl-v2');" 2>&1 | grep -viE "schema error|registered from" | tail -8

log "warming ${DS}"
uv run python -c "
from duckdb_rocket.datasets import load
x, y = load('${DS}', 'train'); xt, _ = load('${DS}', 'test')
print(f'  train {x.shape} test {xt.shape}')" 2>&1 | tail -1

run_arm() {  # label, extra phase5 args
    local label="$1"; shift
    echo
    echo "  --- ${label}"
    # 24 rows escalated is the arm that always worked; the FULL 128-row batch is the one that dies.
    uv run python scripts/phase5_pipeline.py --dataset "$DS" --device cpu --model tabicl-v2 \
        --test-chunk 128 --threads 4 --onnx-threads 4 \
        --anofox-extension "$EXT" \
        --out "$OUT/${label}.json" "$@" > "$OUT/${label}.log" 2>&1
    echo "    rc=$?"
    grep -viE "schema error|registered from" "$OUT/${label}.log" \
      | grep -iE "exit|FAILED|error|refus|ceiling|memory|accuracy" | tail -6 | sed 's/^/      /'
}

# ---------------------------------------------------------------------------------------------
# 2 and 3. The failure, and the failure with a name on it.
# ---------------------------------------------------------------------------------------------
log "arm 2: no ceiling -- this must still die, or arm 3 proves nothing"
run_arm no_ceiling

log "arm 3: with a ${CEILING} ceiling -- this must fail with a message"
run_arm with_ceiling --tabfm-max-memory "$CEILING"

log "what came out"
for f in "$OUT"/no_ceiling.log "$OUT"/with_ceiling.log; do
    [ -e "$f" ] || continue
    echo "--- $(basename "$f")"
    grep -viE "schema error|registered from" "$f" | tail -12 | sed 's/^/    /'
done
log "done"
