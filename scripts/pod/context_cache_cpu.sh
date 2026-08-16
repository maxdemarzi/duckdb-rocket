#!/usr/bin/env bash
# What does the upstream context cache actually buy, measured through DuckDB?
#
# `anofox_tabfm_context_cache` (DataZooDE/anofox-tabfm#40, unreleased) encodes the labelled context
# once per support set instead of once per call, for a model that ships the support/query graph pair
# from #38. The 11.2x in the exporter README is a PyTorch prototype, not the extension. This measures
# the extension.
#
# The shape is #37's escalation case: a 375-row labelled context, 500 features, and small query
# batches arriving one after another against the SAME context -- which is what `--test-chunk` does in
# this project's own serving path, and the only pattern the cache can help.
#
# Two arms in ONE duckdb session, so both see byte-identical data:
#
#   1. context_cache = false -- the combined graph, re-encoding the context every call
#   2. context_cache = true  -- prepare once, then query per batch
#
# and then the check that matters more than the clock: do the TEST rows agree? The cache is only
# worth anything if it is the same answer, faster. Context rows are expected to differ -- the query
# half has no label path -- so they are compared and reported separately rather than quietly ignored.
#
# Needs, uploaded to the pod first (neither is public):
#   /workspace/ctxext/anofox_tabfm.duckdb_extension   the #40 build (fork CI artifact)
#   /workspace/realsplit/                             the real tabicl-v2 split pair + tensor maps
#
#   bash scripts/pod/context_cache_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/ctxcache
EXT="${EXT:-/workspace/ctxext/anofox_tabfm.duckdb_extension}"
SPLIT="${SPLIT:-/workspace/realsplit}"
MODEL_DIR=/workspace/model_split
S="${S:-375}"     # labelled context rows
Q="${Q:-22}"      # query rows per call -- the escalation batch size
CALLS="${CALLS:-6}"
H="${H:-500}"     # features

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
[ -f "$EXT" ] || { echo "FATAL: no extension at $EXT"; exit 1; }
[ -d "$SPLIT" ] || { echo "FATAL: no split graphs at $SPLIT"; exit 1; }
ls -la "$EXT" "$SPLIT"

log "repository + prebuilt shell"
[ -d "$R/.git" ] || git clone -q --depth 1 https://github.com/maxdemarzi/duckdb-rocket.git "$R"
cd "$R"
git pull -q --ff-only || true
source scripts/pod/shallow_clone.sh
shallow_submodule duckdb >/dev/null 2>&1 || true
KEY=$(git rev-parse "HEAD:src" "HEAD:CMakeLists.txt" "HEAD:extension_config.cmake" "HEAD:duckdb" \
      | sha256sum | cut -c1-12)
mkdir -p build/release "$OUT"
if curl -fsSL -o build/release/duckdb "https://github.com/maxdemarzi/duckdb-rocket/releases/download/prebuilt/duckdb-rocket-${KEY}-linux_x86_64" 2>/dev/null; then
    chmod +x build/release/duckdb && echo "  SHELL CACHE HIT ($KEY)"
else
    echo "  SHELL CACHE MISS ($KEY) -- falling back to the stock DuckDB CLI"
    curl -fsSL -o /tmp/duckdb_cli.zip https://github.com/duckdb/duckdb/releases/download/v1.5.5/duckdb_cli-linux-amd64.zip
    unzip -o -q /tmp/duckdb_cli.zip -d build/release && chmod +x build/release/duckdb
fi
DUCKDB="$R/build/release/duckdb"
"$DUCKDB" -c "SELECT 1;" >/dev/null || { echo "FATAL: no working duckdb"; exit 1; }

log "tabicl-v2 weights (through the community build)"
"$DUCKDB" -c "INSTALL httpfs; INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
"$DUCKDB" -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabicl-v2');" 2>&1 | tail -2
CKPT=$(find "$HOME/.cache/anofox-tabfm" -name "*.ckpt" | grep -i tabicl | head -1)
[ -n "$CKPT" ] || { echo "FATAL: no tabicl-v2 checkpoint downloaded"; exit 1; }
echo "  checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"

log "assemble the model directory"
# Everything the engine needs for ONE registered model: the combined graph (the
# baseline arm), the split pair (the cache arm), a tensor map per graph, and the
# real checkpoint. The pair is discovered by filename next to the combined graph,
# so the layout IS the configuration -- there is nothing to declare.
rm -rf "$MODEL_DIR"; mkdir -p "$MODEL_DIR"
cp "$SPLIT"/graph_tabicl_classification.onnx          "$MODEL_DIR/"
cp "$SPLIT"/graph_prepare_tabicl_classification.onnx  "$MODEL_DIR/"
cp "$SPLIT"/graph_query_tabicl_classification.onnx    "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_tabicl_classification.json         "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_prepare_tabicl_classification.json "$MODEL_DIR/"
cp "$SPLIT"/tensor_map_query_tabicl_classification.json   "$MODEL_DIR/"
cp "$CKPT" "$MODEL_DIR/model_classification.ckpt"
ls -la "$MODEL_DIR"

log "generate the workload (S=$S context, Q=$Q per call, $CALLS calls, H=$H features)"
python3 - "$MODEL_DIR" "$S" "$Q" "$CALLS" "$H" "$EXT" > "$OUT/bench.sql" <<'PY'
import sys
model_dir, S, Q, CALLS, H, ext = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
cols = ", ".join(f"f{i}" for i in range(H))
# Deterministic features from the row index, so both arms and any rerun see the same table.
feats = ", ".join(f"sin({i + 1} * (i + 1) * 0.017)::DOUBLE AS f{i}" for i in range(H))
p = print
p(".timer on")
p(f"LOAD '{ext}';")
p("SET anofox_tabfm_accept_hf_license = true;")
p("SET anofox_tabfm_threads = 8;")
p(f"""CALL tabfm_register_model(
  id := 'tabicl-split', base_dir := '{model_dir}',
  classification_graph := 'graph_tabicl_classification.onnx',
  classification_weights := 'model_classification.ckpt',
  tensor_map := 'tensor_map_tabicl_classification.json',
  license := 'bsd-3-clause', commercial := true,
  weights_repo := 'local:{model_dir}', weights_revision := 'split-v1',
  preprocessing_profile := 'tabicl_v2_raw',
  max_rows := 4096, max_features := 512, max_classes := 3);""")
p(f"CREATE TABLE ctx AS SELECT i, {feats}, ('c' || (i % 3)) AS label FROM range({S}) t(i);")
p(f"CREATE TABLE qry AS SELECT i, {feats}, NULL::VARCHAR AS label FROM range({S}, {S + Q * CALLS}) t(i);")
for k in range(CALLS):
    lo, hi = S + k * Q, S + (k + 1) * Q
    p(f"CREATE VIEW call{k} AS SELECT * FROM ctx UNION ALL SELECT * FROM qry WHERE i >= {lo} AND i < {hi};")
p("-- the model loads on the first scoring call; warm it OUTSIDE the timed arms so")
p("-- session creation is not charged to whichever arm happens to run first.")
p("SET anofox_tabfm_context_cache = false;")
p("CREATE TABLE warm AS SELECT i, yhat FROM tabfm_classify('call0', 'label', model := 'tabicl-split');")
for arm, flag in (("off", "false"), ("on", "true")):
    p(f"SET anofox_tabfm_context_cache = {flag};")
    for k in range(CALLS):
        p(f"CREATE TABLE {arm}{k} AS SELECT i, yhat, yhat_score, is_training "
          f"FROM tabfm_classify('call{k}', 'label', model := 'tabicl-split');")
p(".timer off")
p("-- Do the two arms give the same answer on the rows that are actually predictions?")
u_off = " UNION ALL ".join(f"SELECT * FROM off{k} WHERE NOT is_training" for k in range(CALLS))
u_on = " UNION ALL ".join(f"SELECT * FROM on{k} WHERE NOT is_training" for k in range(CALLS))
p(f"""SELECT 'TEST rows' AS rows, count(*) AS n,
       count(*) FILTER (WHERE a.yhat <> b.yhat) AS label_disagreements,
       max(abs(a.yhat_score - b.yhat_score)) AS max_score_delta
FROM ({u_off}) a JOIN ({u_on}) b USING (i);""")
p(f"""SELECT 'CONTEXT rows' AS rows, count(*) AS n,
       count(*) FILTER (WHERE a.yhat <> b.yhat) AS label_disagreements,
       max(abs(a.yhat_score - b.yhat_score)) AS max_score_delta
FROM (SELECT * FROM off0 WHERE is_training) a JOIN (SELECT * FROM on0 WHERE is_training) b USING (i);""")
PY
wc -l "$OUT/bench.sql"

log "run both arms"
"$DUCKDB" < "$OUT/bench.sql" > "$OUT/bench.out" 2>&1
echo "  rc=$?"

log "what came out"
grep -viE "schema error|registered from" "$OUT/bench.out" | tail -60

log "per-call wall time"
python3 - "$OUT/bench.out" "$CALLS" <<'PY'
import re, sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
calls = int(sys.argv[2])
times = [float(m.group(1)) for l in lines if (m := re.search(r"Run Time \(s\): real ([0-9.]+)", l))]
# The timed statements after the warm-up are: CALLS off arms, then CALLS on arms.
if len(times) < 2 * calls:
    print(f"  only {len(times)} timings found; see bench.out"); raise SystemExit
off, on = times[-2 * calls:-calls], times[-calls:]
p = print
p(f"  {'call':>6}  {'off (s)':>9}  {'on (s)':>9}  {'speedup':>8}")
for k, (a, b) in enumerate(zip(off, on)):
    p(f"  {k:>6}  {a:>9.3f}  {b:>9.3f}  {a / b if b else 0:>7.2f}x")
p(f"  {'TOTAL':>6}  {sum(off):>9.3f}  {sum(on):>9.3f}  {sum(off) / sum(on) if sum(on) else 0:>7.2f}x")
p(f"\n  steady state (calls 1..{calls - 1}, i.e. excluding each arm's first):")
so, sn = sum(off[1:]), sum(on[1:])
p(f"          off {so:.3f}s   on {sn:.3f}s   {so / sn if sn else 0:.2f}x")
PY
log "done"
