#!/usr/bin/env bash
# The five datasets that never produced a per-group cube, and the lever that makes them possible.
#
# The group-count result rests on 24 of the 28 subgroup datasets. The missing five are all
# large-SUPPORT cases -- 450 to 600 labelled rows -- and they are exactly where escalation earns
# most, so leaving them out tilts the sample toward the easy half. They were written off as
# "context too large for the box"; on 2026-08-16 SemgHandMovementCh2's serving arm completed at
# --test-chunk 32 after being killed at 128, which makes all five a chunk-size problem rather than
# an impossible one. Note what does NOT enter: series length. Two of the five are 80 timepoints
# long. The call sees 500 ROCKET features whatever the series was, so the ceiling is rows.
#
# Chunking has to be exact for any of this to be comparable to the archived 24, which were run in
# one call each. That is asserted first, on a dataset that already has a cube, before anything
# expensive runs -- see CONTROL below.
#
#   bash scripts/pod/pergroup_recover_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/pergroup_recover
MISSING="${MISSING:-DistalPhalanxOutlineCorrect MiddlePhalanxOutlineCorrect SemgHandMovementCh2 SemgHandSubjectCh2 EthanolLevel}"
# Tried in order; the first that produces a cube wins. Every step down multiplies the number of
# calls, and each call re-encodes the whole labelled context (0.62 s + 12.0 ms per support row per
# group per call, fitted 2026-08-16), so this is a ladder and not a default.
CHUNKS="${CHUNKS:-128 64 32}"
# 61 test rows, 60 train: one call at the archived 128, four at 16. Cheap, and it is the whole
# basis for merging chunked cubes with unchunked ones.
CONTROL="${CONTROL:-Lightning2}"
CONTROL_CHUNK="${CONTROL_CHUNK:-16}"
TIMEOUT_MIN="${TIMEOUT_MIN:-150}"

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
# The cgroup cap, not `free`. A container on a 124 GB host reported 124 GB and was killed at 29.8;
# that gap is the whole reason these five datasets have no cubes.
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
mkdir -p "$OUT/pergroup" "$OUT/control"

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

log "the archived 24 cubes"
# The tarball's members are already pergroup/*.json, so this lands them in $OUT/pergroup and the
# five new ones are written alongside. The archive is the comparison, so it is unpacked rather
# than re-derived.
tar xzf reference/pergroup_cubes.tar.gz -C "$OUT"
N_ARCHIVED=$(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l)
echo "  ${N_ARCHIVED} archived cubes in place"
[ "$N_ARCHIVED" -eq 24 ] || { echo "FATAL: expected 24 archived cubes, got ${N_ARCHIVED}"; exit 1; }

log "warming the dataset cache, serially"
# Before anything forks or is timed. Six workers pulling the same cold aeon archive at once
# produced 44 bogus failures in an earlier sweep and got this account rate-limited by Zenodo hard
# enough to cost the next experiment 24 of its 28 datasets.
for ds in $CONTROL $MISSING; do
    uv run python -c "
from duckdb_rocket.datasets import load
x, y = load('$ds', 'train'); xt, _ = load('$ds', 'test')
print(f'  $ds train {x.shape} test {xt.shape}')" 2>&1 | tail -1
done

# threads x onnx-threads, one run at a time. These datasets are memory-bound, not CPU-bound: the
# reason they have no cubes is a per-call allocation, and running two at once doubles it.
T=4; O=$(( NPROC / T )); [ "$O" -lt 1 ] && O=1; [ "$O" -gt 8 ] && O=8
echo "  ${T} duckdb threads x ${O} onnx threads, one dataset at a time"

run_sweep() {  # dataset, chunk, out-dir, log
    uv run python scripts/teacher_sweep.py --model tabicl-v2 --datasets "$1" \
        --out-dir "$3" --device cpu --per-group-soft \
        --threads "$T" --onnx-threads "$O" --test-chunk "$2" \
        --budget-min 600 --timeout-min "$TIMEOUT_MIN" > "$4" 2>&1
}

# ---------------------------------------------------------------------------------------------
# CONTROL. Nothing below is comparable to the archive unless this passes.
# ---------------------------------------------------------------------------------------------
log "control: is a chunked cube the same as an unchunked one?"
echo "  $CONTROL at --test-chunk $CONTROL_CHUNK (4 calls per group) vs its archived cube (1 call)"
run_sweep "$CONTROL" "$CONTROL_CHUNK" "$OUT/control" "$OUT/control.log"
echo "  rc=$?"
CTRL_NEW="$OUT/control/phase5_${CONTROL}_gpu_pergroup.json"
CTRL_OLD="$OUT/pergroup/phase5_${CONTROL}_gpu_pergroup.json"
if [ -f "$CTRL_NEW" ] && [ -f "$CTRL_OLD" ]; then
    uv run python - "$CTRL_NEW" "$CTRL_OLD" <<'PY'
import json, sys
import numpy as np
new, old = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
a, b = np.asarray(new["proba"]), np.asarray(old["proba"])
assert new["ids"] == old["ids"], "the two runs are not over the same rows"
assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
d = float(np.max(np.abs(a - b)))
print(f"  max abs probability difference: {d:.3e}")

# NOT "d < 1e-6". The cubes store six decimal places, so 1e-6 is the quantisation unit and that
# test can never pass -- it read 9.0e-06 on the first run and cried wolf. Two boxes at the same
# chunk size were separately measured bit-identical, so the residue is chunking and nothing else;
# what has to be established is that it cannot move a REPORTED number. The analysis averages the
# first G groups and takes an argmax, so the question is whether any decision is closer than the
# difference. Measured: smallest margin 4.4e-3, which is 489x the difference.
ok = True
for G in (1, 5, 10, 20, 40):
    if G > a.shape[0]:
        continue
    pa, pb = a[:G].mean(0), b[:G].mean(0)
    agree = float((pa.argmax(-1) == pb.argmax(-1)).mean())
    margin = float(np.diff(np.sort(pa, axis=-1)[:, -2:], axis=-1).min())
    ok &= agree == 1.0 and margin > 10 * d
    print(f"  G={G:<2d} prefix-averaged predictions agree {agree:.4%}, "
          f"closest decision {margin:.3e} ({margin / d:.0f}x the difference)")
print("  VERDICT:", "chunking cannot move a reported number; the cubes are comparable" if ok
      else "*** CHUNKING REACHES THE DECISIONS -- do not merge these cubes ***")
PY
else
    echo "  *** CONTROL DID NOT PRODUCE A CUBE; the merge below is unvalidated ***"
    tail -20 "$OUT/control.log"
fi

# ---------------------------------------------------------------------------------------------
# The five. Cheapest chunk first; step down only on failure.
# ---------------------------------------------------------------------------------------------
for ds in $MISSING; do
    log "$ds"
    CUBE="$OUT/pergroup/phase5_${ds}_gpu_pergroup.json"
    for chunk in $CHUNKS; do
        echo "  --test-chunk $chunk"
        run_sweep "$ds" "$chunk" "$OUT/pergroup" "$OUT/${ds}_c${chunk}.log"
        if [ -f "$CUBE" ]; then
            echo "  CUBE at chunk $chunk"
            grep -E "FAILED|ok |seconds" "$OUT/${ds}_c${chunk}.log" | tail -3
            echo "$ds $chunk" >> "$OUT/chunk_used.txt"
            break
        fi
        echo "  no cube at chunk $chunk:"
        grep -E "FAILED|exit|Error|error" "$OUT/${ds}_c${chunk}.log" | tail -4
    done
    [ -f "$CUBE" ] || echo "  *** $ds produced no cube at any chunk size ***"
done

log "accuracy and routing as a function of G, now over all cubes"
echo "  cubes: $(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l)"
uv run python scripts/perf_levers.py --groups --pergroup "$OUT/pergroup" \
    --out "$OUT/perf_groups_29.json" > "$OUT/groups.log" 2>&1
echo "  rc=$?"
sed -n '/TEACHER GROUPS/,$p' "$OUT/groups.log"

log "what came out"
cat "$OUT/chunk_used.txt" 2>/dev/null
ls -la "$OUT"/*.json 2>/dev/null
log "done"
