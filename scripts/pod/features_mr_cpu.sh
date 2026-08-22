#!/usr/bin/env bash
# rocket vs multirocket, paired, over the same 29 hard datasets and driver features22 used.
#
# A literature check (2026-08-22) found RocketPFN's own ablation (arXiv 2606.21786 S4.7) rules
# out extractor choice as a source of the 92-dataset reproduction gap (<0.006 at G>=5), but
# TS2TabPFN (arXiv 2608.04174) measured a much bigger Rocket-vs-MultiRocket gap under a different
# TabPFN-family ensembling scheme. Cheap to test directly rather than argue from someone else's
# ablation -- see PLAN.md Phase 8.
#
# Unlike feature_family_cpu.sh's ts/catch22 arms, multirocket is NOT n_groups=1: it is its own
# per-group random extractor, run exactly like rocket_transform's 40 groups, so this uses
# resample_power.py (the paired driver features22 used) rather than teacher_sweep.py.
#
#   bash scripts/pod/features_mr_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/reference/features_mr_r1.json
DATASETS="ACSF1,ArrowHead,Beef,Computers,DistalPhalanxOutlineAgeGroup,DistalPhalanxOutlineCorrect,DistalPhalanxTW,Earthquakes,EthanolLevel,Ham,Haptics,Herring,InlineSkate,LargeKitchenAppliances,Lightning2,Lightning7,MedicalImages,MiddlePhalanxOutlineAgeGroup,MiddlePhalanxOutlineCorrect,MiddlePhalanxTW,ProximalPhalanxOutlineAgeGroup,ProximalPhalanxTW,RefrigerationDevices,ScreenType,SemgHandMovementCh2,SemgHandSubjectCh2,SmallKitchenAppliances,Worms,WormsTwoClass"
JOBS="${JOBS:-2}"
THREADS="${THREADS:-3}"
ONNX_THREADS="${ONNX_THREADS:-3}"

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
echo "  nproc (host, NOT the budget): $(nproc)"
for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; do
    [ -r "$f" ] && echo "  $f: $(cat "$f")"
done

log "repository"
[ -d "$R/.git" ] || git clone -q --depth 1 https://github.com/maxdemarzi/duckdb-rocket.git "$R"
cd "$R"
git pull -q --ff-only
source scripts/pod/shallow_clone.sh
shallow_submodule duckdb || { echo "FATAL: no duckdb submodule"; exit 1; }
git log --oneline -1
uv sync -q
mkdir -p reference/resample

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

log "extensions and weights"
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabicl-v2');" >/dev/null 2>&1 \
  && echo "  tabicl-v2 weights ok" || { echo "FATAL: no tabicl-v2 weights"; exit 1; }

log "warming the dataset cache, serially (avoids concurrent jobs racing the same download)"
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for ds in "${DS_ARR[@]}"; do
    uv run python -c "
from duckdb_rocket.datasets import load
load('$ds', 'train'); load('$ds', 'test')" >/dev/null 2>&1 || echo "  $ds FAILED to load"
done
echo "  done"

log "rocket vs multirocket, ${#DS_ARR[@]} datasets, R=1, --jobs $JOBS --threads $THREADS --onnx-threads $ONNX_THREADS"
uv run python scripts/pod/resample_power.py --arms features_mr \
    --datasets "$DATASETS" --resamples 1 \
    --jobs "$JOBS" --threads "$THREADS" --onnx-threads "$ONNX_THREADS" \
    --test-chunk 128 --out "$OUT" \
    > /workspace/features_mr.log 2>&1
echo "  rc=$?"
tail -60 /workspace/features_mr.log

log "done"
