#!/usr/bin/env bash
# What does the context cache buy the WHOLE pipeline, not one call pair?
#
# `scripts/pod/context_cache_cpu.sh` measured `anofox_tabfm_context_cache` at the extension level:
# 7.2x on a repeated call against the same 375-row context, 0.36x on the first. That is the right
# number for the mechanism and the wrong number for a decision -- a phase-5 run makes 40 group
# calls, and each group's context is DIFFERENT (different kernels, different feature columns), so
# per group the cache starts cold. It only wins inside a group, across `--test-chunk` chunks that
# share one context.
#
# So the whole question is chunks-per-group, and the prediction is arithmetic:
#
#   per group:  1 cold call at ~2.5x cost, then (chunks-1) warm calls at ~1/7 cost
#
# which is a loss at 1 chunk, roughly break-even at 3, and a real win at 9. Two datasets are run
# rather than one, chosen for landing either side of that:
#
#   ItalyPowerDemand   1029 test rows / 128 = 9 chunks per group   <- the case the cache is for
#   OSULeaf             242 test rows / 128 = 2 chunks per group   <- the case it barely helps
#
# If the effect is real it must be LARGER on ItalyPowerDemand. One dataset getting faster proves
# nothing about the mechanism; the two moving apart in the predicted direction does.
#
# And the check that outranks the clock: both arms must predict the same label for every test row.
# The cache is worth nothing if it is a different answer, faster.
#
# If the result comes back NULL -- `on` slower than `off` on both datasets, roughly uniformly --
# read that as the cache never hitting rather than as the cache being worthless, and look here
# first: #40 recognises a repeat context by `memcmp` over the support rows, and the pipeline fills
# `train_cur` once per group and then scans it once per chunk. Byte-identical requires that scan to
# return rows in the same ORDER every time, which a parallel scan at `threads = 4` need not do.
# The content is certainly stable -- archived runs agree row-for-row across chunks -- so ordering
# is the one thing between this design and a hit. `--threads 1` is the cheap way to tell them apart.
#
# Needs, uploaded to the pod first (neither is public):
#   /workspace/ctxext/anofox_tabfm.duckdb_extension   the #40 build (fork CI artifact)
#   /workspace/realsplit/                             the real tabicl-v2 split pair + tensor maps
#
#   bash scripts/pod/context_cache_phase5_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/ctxphase5
EXT="${EXT:-/workspace/ctxext/anofox_tabfm.duckdb_extension}"
SPLIT="${SPLIT:-/workspace/realsplit}"
MODEL_DIR=/workspace/model_split_phase5
DATASETS="${DATASETS:-ItalyPowerDemand OSULeaf}"
CHUNK="${CHUNK:-128}"
THREADS="${THREADS:-4}"
# Sized so the pools multiply out to the cores we actually have: DuckDB runs THREADS of them and
# each ONNX session takes ONNX_THREADS. 16 vCPU gives 4x4, which is what the archived phase-5
# numbers were taken at; a smaller box scales down instead of oversubscribing. CPU capacity was
# exhausted at every size when this was first run, so the box is whatever was available.
ONNX_THREADS="${ONNX_THREADS:-$(( $(nproc) / THREADS > 0 ? $(nproc) / THREADS : 1 ))}"

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
echo "  $(nproc) vCPU, $(free -g | awk '/^Mem:/{print $2}') GB RAM"
[ -f "$EXT" ] || { echo "FATAL: no extension at $EXT -- scp the #40 CI artifact first"; exit 1; }
[ -d "$SPLIT" ] || { echo "FATAL: no split graphs at $SPLIT"; exit 1; }
ls -la "$EXT" "$SPLIT"

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
# The pipeline needs rocket_transform, so the stock CLI will not do here -- unlike the call-pair
# bench, which only needed a working duckdb.
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

log "the extension loads, and carries the setting"
# Without this the run completes with the SET failing, the combined graph running, and a timing
# that reads 1.00x -- a convincing null result produced by nothing having been switched on.
# -csv -noheader so the answer is a bare number: the box output draws with U+2502 and a grep
# written for ASCII pipes fails on a perfectly good load.
build/release/duckdb -unsigned -csv -noheader \
  -c "LOAD '${EXT}'; SELECT count(*) FROM duckdb_settings() WHERE name = 'anofox_tabfm_context_cache';" \
  2>&1 | grep -qx "1" || {
    echo "FATAL: no anofox_tabfm_context_cache in ${EXT}:"
    build/release/duckdb -unsigned -c "LOAD '${EXT}';" 2>&1 | head -5
    exit 1
}
echo "  ok"

log "tabicl-v2 weights (through the community build, which shares the cache)"
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabicl-v2');" >/dev/null 2>&1 \
  && echo "  tabicl-v2 ok" || { echo "FATAL: no weights"; exit 1; }
CKPT=$(find "$HOME/.cache/anofox-tabfm" -name "*.ckpt" | grep -i tabicl | head -1)
[ -n "$CKPT" ] || { echo "FATAL: no tabicl-v2 checkpoint on disk"; exit 1; }
echo "  checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"

log "assemble the model directory"
# The layout IS the configuration: the split pair is discovered by filename beside the combined
# graph, and there is nothing to declare. `model.ckpt` rather than the bench's
# `model_classification.ckpt` because that is the name phase5_pipeline.py registers.
rm -rf "$MODEL_DIR"; mkdir -p "$MODEL_DIR"
cp "$SPLIT"/graph_tabicl_classification.onnx          "$MODEL_DIR/"
cp "$SPLIT"/graph_prepare_tabicl_classification.onnx  "$MODEL_DIR/"
cp "$SPLIT"/graph_query_tabicl_classification.onnx    "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_tabicl_classification.json         "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_prepare_tabicl_classification.json "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_query_tabicl_classification.json   "$MODEL_DIR/"
cp "$CKPT" "$MODEL_DIR/model.ckpt"
ls -la "$MODEL_DIR"

run_arm() {  # dataset, label, extra phase5 args
    local ds="$1" label="$2"; shift 2
    echo "  --- ${ds} ${label}"
    local t0=$(date +%s)
    uv run python scripts/phase5_pipeline.py --dataset "$ds" --device cpu --model tabicl-v2 \
        --test-chunk "$CHUNK" --threads "$THREADS" --onnx-threads "$ONNX_THREADS" \
        --anofox-extension "$EXT" --register-model-dir "$MODEL_DIR" \
        --out "$OUT/${ds}_${label}.json" "$@" > "$OUT/${ds}_${label}.log" 2>&1
    local rc=$?
    echo "    rc=$rc  ($(( $(date +%s) - t0 ))s by the clock outside)"
    # Snapshot the per-row predictions: both arms write into the same workdir, so the second
    # would otherwise overwrite the first and the agreement check would compare a run with itself.
    cp "data/phase5/${ds}/predictions.json" "$OUT/${ds}_${label}_predictions.json" 2>/dev/null \
      || echo "    (no predictions.json -- the arm did not finish)"
    grep -viE "schema error|registered from" "$OUT/${ds}_${label}.log" \
      | grep -iE "accuracy|FAILED|error|exit" | tail -4 | sed 's/^/      /'
}

for ds in $DATASETS; do
    log "$ds: cache OFF, then ON, at --test-chunk $CHUNK"
    run_arm "$ds" off
    run_arm "$ds" on --context-cache
done

log "what came out"
uv run python - "$OUT" "$DATASETS" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for ds in sys.argv[2].split():
    arms = {}
    for label in ("off", "on"):
        f = out / f"{ds}_{label}.json"
        if not f.exists():
            print(f"  {ds} {label}: NO REPORT -- the arm did not finish")
            continue
        arms[label] = json.loads(f.read_text())
    if len(arms) != 2:
        continue

    # The claim that outranks the clock. Compared per row rather than by accuracy: two runs can
    # reach the same accuracy while disagreeing about which rows they got right, and a cache that
    # returns a different answer quickly is not a faster cache.
    preds = {}
    for label in ("off", "on"):
        f = out / f"{ds}_{label}_predictions.json"
        preds[label] = {int(r["id"]): r["yhat"] for r in json.loads(f.read_text())} if f.exists() else {}
    shared = set(preds["off"]) & set(preds["on"])
    disagree = sum(1 for i in shared if preds["off"][i] != preds["on"][i])

    a, b = arms["off"], arms["on"]
    n_test = a["shape"]["n_test"]
    chunks = -(-n_test // b["config"]["test_chunk"])
    rows.append({
        "dataset": ds, "n_test": n_test, "chunks_per_group": chunks,
        "off_classify": a["time_split"]["classify_seconds"],
        "on_classify": b["time_split"]["classify_seconds"],
        "off_total": a["seconds"], "on_total": b["seconds"],
        "off_accuracy": a["accuracy"], "on_accuracy": b["accuracy"],
        "rows_compared": len(shared), "label_disagreements": disagree,
    })

if not rows:
    print("  nothing to report")
    raise SystemExit(1)

print(f"\n  {'dataset':<18} {'test':>5} {'chunks':>7} "
      f"{'off (s)':>9} {'on (s)':>9} {'speedup':>8} {'saved':>9} {'rows':>6} {'disagree':>9}")
for r in rows:
    sp = r["off_classify"] / r["on_classify"] if r["on_classify"] else float("nan")
    print(f"  {r['dataset']:<18} {r['n_test']:>5} {r['chunks_per_group']:>7} "
          f"{r['off_classify']:>9.1f} {r['on_classify']:>9.1f} {sp:>7.2f}x "
          f"{r['off_classify'] - r['on_classify']:>8.1f}s "
          f"{r['rows_compared']:>6} {r['label_disagreements']:>9}")
print("\n  (classify seconds, not total: the cache cannot touch the ROCKET transform, which is"
      "\n   about 6s of a 1000s run and would only dilute the number.)")

for r in rows:
    if r["label_disagreements"]:
        print(f"\n  WARNING: {r['dataset']} disagrees on {r['label_disagreements']} "
              f"of {r['rows_compared']} rows -- the arms are not the same computation")
    if abs(r["off_accuracy"] - r["on_accuracy"]) > 1e-12:
        print(f"\n  WARNING: {r['dataset']} accuracy moved "
              f"{r['off_accuracy']:.4f} -> {r['on_accuracy']:.4f}")

# The mechanism check. One dataset getting faster is consistent with almost anything -- a warmer
# page cache, a quieter box. The cache's whole story is that the win grows with chunks per group,
# so the two datasets must separate in that order or the story is wrong.
if len(rows) >= 2:
    lo, hi = sorted(rows, key=lambda r: r["chunks_per_group"])[0], \
             sorted(rows, key=lambda r: r["chunks_per_group"])[-1]
    s_lo = lo["off_classify"] / lo["on_classify"] if lo["on_classify"] else float("nan")
    s_hi = hi["off_classify"] / hi["on_classify"] if hi["on_classify"] else float("nan")
    print(f"\n  mechanism: {lo['chunks_per_group']} chunks -> {s_lo:.2f}x, "
          f"{hi['chunks_per_group']} chunks -> {s_hi:.2f}x  "
          f"({'as predicted' if s_hi > s_lo else 'NOT as predicted -- the win does not grow with chunks'})")

(out / "summary.json").write_text(json.dumps(rows, indent=2))
print(f"\n  wrote {out / 'summary.json'}")
PY
log "done"
