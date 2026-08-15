#!/usr/bin/env bash
# The other three labellers, over the datasets the gate opened on. CPU pod.
#
# **Why more labellers at all.** Arm B says the teacher's pseudo-labels recover none of the five
# points of headroom those datasets have, because what pseudo-labelling needs is not that the teacher
# beat the student -- it does, by ~3 points -- but that its labels be RIGHT, and at 0.55-0.85 accuracy
# a third of the pool gets the wrong one. The only way out is a labeller that is *more accurate*, and
# an ensemble of architecturally unrelated models is the standard one.
#
# **Why a pod.** Four models x 29 datasets is embarrassingly parallel across datasets and CPU-bound,
# and this workstation has 8 cores that the break-even sweep is already using. Nothing here needs a
# GPU: v2026.08.14 fixed tabicl-v2 on CUDA, but the GPU flavor is still unpublished (#25), so a GPU
# pod would run the CPU EP anyway -- which is exactly the mistake runpod_cpu.py was written after.
#
# **What it produces.** One report and one soft-label sidecar per (model, dataset), which is what
# distill_gate.py needs to score an ensemble labeller against the same students on the same splits.
#
#   BUDGET_MIN=240 bash scripts/pod/ensemble_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/ensemble
BUDGET_MIN="${BUDGET_MIN:-240}"
TIMEOUT_MIN="${TIMEOUT_MIN:-30}"

# **mitra is not in the default list, and the reason is cost rather than capability.** Measured on
# 16 vCPU: Lightning2 -- 60 train, 61 test -- reached group 2 of 40 in about five minutes, so roughly
# 100 minutes for one of the smallest datasets in the subgroup. Every dataset except Beef (30 test
# rows) therefore hit the 30-minute per-dataset timeout, and the first run of this script produced
# exactly one mitra report out of 29.
#
# Why it is slow is the same fact that makes it usable at all. mitra declares max_features = 100
# against tabicl's 512, and the engine covers a 500-feature group by raising the estimator count
# rather than by truncating -- which is why a probe with the only informative feature at index 499
# still classifies it correctly, and why each call costs about five times what the others cost.
#
# So running mitra means either TIMEOUT_MIN=180 and a much longer budget, or fewer groups, and
# either is a different experiment from the one the other three are in. MODELS='mitra' TIMEOUT_MIN=180.
MODELS="${MODELS:-tabpfn-v2 orion-bix}"

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

log "repository"
[ -d "$R/.git" ] || git clone -q --depth 1 https://github.com/maxdemarzi/duckdb-rocket.git "$R"
cd "$R"
git pull -q --ff-only || true
source scripts/pod/shallow_clone.sh
shallow_submodule duckdb || { echo "FATAL: no duckdb submodule"; exit 1; }
git log --oneline -1
uv sync -q

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
# HuggingFace, and every dataset then dies with "no downloaded weights", which reads as a dataset
# failure rather than a setup one. That cost a whole 62-dataset sweep once.
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "SELECT extension_version FROM duckdb_extensions() WHERE extension_name='anofox_tabfm';" 2>&1 | tail -3
for m in tabicl-v2 $MODELS; do
    printf '  downloading %-12s ' "$m"
    build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
      SET anofox_tabfm_accept_hf_license = true;
      FROM tabfm_download('classification', model := '$m');" >/dev/null 2>&1 && echo ok || echo FAILED
done

log "converting the checkpoints the engine cannot read"
# tabpfn-v2's checkpoint uses tensor names the graph was not exported against; orion-bix's pickle
# stream uses an opcode the native reader does not implement. Both are fixed by a sibling
# safetensors, which upstream's converters write. Asserted inside the script.
uv run python -c "import torch, safetensors, huggingface_hub" 2>&1 || {
    echo "FATAL: the converter needs torch, safetensors and huggingface_hub in the project env"; exit 1; }
bash scripts/convert_model_weights.sh "/workspace/anofox-converters" || {
    echo "FATAL: conversion or verification failed; the sweep would produce nothing usable"; exit 1; }

log "seeding reports already computed elsewhere"
mkdir -p "$OUT"
cp reference/phase5_*.json "$OUT/" 2>/dev/null
echo "  seeded $(ls "$OUT"/*.json 2>/dev/null | wc -l) report(s)"

# The subgroup, read from the gate's own output rather than listed here: these are the datasets where
# a label-only student is still below 0.90, which is the only place any of this can show anything.
for m in $MODELS; do
    log "sweep: $m, budget ${BUDGET_MIN} min, per-dataset timeout ${TIMEOUT_MIN} min"
    uv run python scripts/teacher_sweep.py --model "$m" \
        --from-gate reference/distill_gate.json --max-student 0.90 \
        --budget-min "$BUDGET_MIN" --out-dir "$OUT" --device cpu \
        --test-chunk 128 --timeout-min "$TIMEOUT_MIN" 2>&1 | tail -80
    echo "  $m rc=${PIPESTATUS[0]}  reports: $(ls "$OUT"/phase5_*_${m}.json 2>/dev/null | wc -l)"
done

log "what came out"
for m in tabicl-v2 $MODELS; do
    if [ "$m" = tabicl-v2 ]; then n=$(ls "$OUT"/phase5_*_gpu.json 2>/dev/null | wc -l)
    else n=$(ls "$OUT"/phase5_*_${m}.json 2>/dev/null | wc -l); fi
    printf '  %-12s %s report(s)\n' "$m" "$n"
done
echo "  soft sidecars: $(ls "$OUT"/*_soft.json 2>/dev/null | wc -l)"
log "done"
