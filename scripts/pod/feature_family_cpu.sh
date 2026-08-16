#!/usr/bin/env bash
# One backbone, two feature families -- the controlled version of the three-backbone comparison.
#
# `tabicl-v2`, `tabpfn-v2` and `orion-bix` fail together 3.74x, 3.75x and 3.78x more than
# independence: three architectures, three pairs, the same number. The variable that could not be
# changed there is the one thing all three are handed, 500 ROCKET features per call. So change only
# that: the same `tabicl-v2` over the same 17 datasets and the same rows, reading
# `anofox_forecast`'s 116 statistics instead.
#
# Both outcomes settle something:
#
#   * excess well under 3.7x -- the failures follow the representation, an ensemble across feature
#     families has headroom that an ensemble across models does not, and `--features both` becomes
#     the configuration worth measuring properly.
#   * excess still ~3.7x -- these rows are hard whatever they are turned into, the ceiling is the
#     task rather than the features, and this line of inquiry closes.
#
# **The ts arm is expected to be the weaker model**, and that is not the measurement. Over 112
# datasets the 116 statistics lose to 10,000 ROCKET features by about 8 accuracy points under a
# ridge. A weaker model that fails in DIFFERENT places still raises the oracle and justifies a
# selective rule; a weaker model that fails in the same places buys nothing. Overlap is the number.
#
# Cheap, because phase5 refuses n_groups != 1 without a kernel bank to slice: one call per dataset
# against forty.
#
#   bash scripts/pod/feature_family_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/feature_family
FEATURES="${FEATURES:-ts}"
MODEL="${MODEL:-tabicl-v2}"
BUDGET_MIN="${BUDGET_MIN:-240}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"

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
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] && echo "  cgroup memory: $(cat "$f")"
done

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

log "extensions and weights"
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
# anofox_forecast is what supplies ts_features_by. It is BSL 1.1 -- production use permitted,
# offering it to third parties hosted or embedded is not -- so it is an opt-in experiment here and
# never a dependency of the rocket extension.
build/release/duckdb -c "INSTALL anofox_forecast FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD anofox_forecast; SELECT count(*) AS n FROM ts_features_list();" \
  2>/dev/null | tail -3 || { echo "FATAL: anofox_forecast will not load"; exit 1; }
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := '${MODEL}');" >/dev/null 2>&1 \
  && echo "  ${MODEL} weights ok" || { echo "FATAL: no ${MODEL} weights"; exit 1; }

log "which datasets"
# Exactly the rows the three-backbone comparison used, so the new arm drops straight into it.
mapfile -t TARGETS < <(uv run python - <<'PY'
import collections, glob, json, os, re
have = collections.defaultdict(set)
for f in glob.glob("reference/phase5_*_soft.json"):
    b = os.path.basename(f)[: -len("_soft.json")]
    m = re.match(r"phase5_(.+?)_([A-Za-z0-9.\-]+)$", b)
    if not m:
        continue
    try:
        model = json.load(open(f, encoding="utf-8")).get("model")
    except Exception:
        continue
    if model:
        have[m.group(1)].add(model)
for d in sorted(d for d, ms in have.items() if len(ms) >= 2):
    print(d)
PY
)
echo "  ${#TARGETS[@]} datasets: ${TARGETS[*]}"
[ "${#TARGETS[@]}" -gt 0 ] || { echo "FATAL: no target datasets"; exit 1; }

log "warming the dataset cache, serially"
for ds in "${TARGETS[@]}"; do
    uv run python -c "
from duckdb_rocket.datasets import load
load('$ds', 'train'); load('$ds', 'test')" >/dev/null 2>&1 || echo "  $ds FAILED to load"
done
echo "  done"

# ---------------------------------------------------------------------------------------------
# One job at a time. Serial is affordable here precisely because there is one group per dataset,
# and it avoids the concurrency-versus-memory trade that killed two datasets in the orion-bix run.
# ---------------------------------------------------------------------------------------------
log "${MODEL} with --features ${FEATURES}"
NG=1; [ "$FEATURES" = "both" ] && NG=40
# ONE invocation for all of them, not one per dataset. teacher_sweep already runs its list serially
# in a single process, and its candidates() loads every dataset in the UCR archive to count classes
# -- the class cap is not in aeon's metadata tables -- so a per-dataset loop repeats that scan once
# per dataset. Measured here: ~1 minute each after the archive is cached, and the first call has to
# fetch the whole archive from Zenodo, which this project has already been rate-limited by once.
uv run python scripts/teacher_sweep.py --model "$MODEL" --datasets "${TARGETS[@]}" \
    --out-dir "$OUT" --device cpu --per-group-soft \
    --features "$FEATURES" --n-groups "$NG" \
    --threads 4 --onnx-threads $(( NPROC / 4 )) --test-chunk 128 \
    --budget-min "$BUDGET_MIN" --timeout-min "$TIMEOUT_MIN" \
    > "$OUT/sweep.log" 2>&1
echo "  rc=$?"
grep -E "^\[|^accuracy|rc=1|last stderr|clean in" "$OUT/sweep.log" | tail -40 | sed 's/^/  /'

log "the overlap, with the new arm folded in"
cp "$OUT"/phase5_*_soft.json "$OUT"/phase5_*.json reference/ 2>/dev/null
uv run python scripts/error_overlap.py --out "$OUT/error_overlap.json" 2>&1 | tail -30

log "what came out"
echo "  soft labels: $(ls "$OUT"/phase5_*_soft.json 2>/dev/null | wc -l) of ${#TARGETS[@]}"
log "done"
