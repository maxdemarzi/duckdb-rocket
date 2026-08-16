#!/usr/bin/env bash
# A second backbone, because every accuracy number in this project comes from one.
#
# `tabicl-v2` was never chosen; it was what loaded. RESULTS.md says so plainly -- "anything
# measured before then is measured on a substitute". The whole cost structure measured here (a
# fixed context pass, 12.0 ms per support row, G=10) is a property of TabICL's architecture, and
# there is no reason to assume it transfers: TabPFN's exported contract does not even take a
# train_size input. This runs the same 29 datasets through `tabpfn-v2` -- Apache-2.0, so unlike
# tabpfn-v2-5 it is shippable -- at the same G=40 and the same 250-kernel groups.
#
# It also produces the per-row predictions the ensemble question needs. RESULTS.md frames that
# measurement as error OVERLAP rather than accuracy, because six models of one icl-transformer
# family scoring one set of ROCKET features can be wrong in the same places; overlap is not
# computable from one model's predictions, which is all that exists today.
#
# Two probes run first and gate the expensive part:
#
#   1. what tabpfn-v2 costs per group. `mitra` declares max_features = 100 against tabicl's 512
#      and the engine covers a 500-feature group by raising the estimator count instead of
#      truncating -- five times the cost, and it timed out on 28 of 29 datasets at a limit that
#      suits the others. If tabpfn-v2 does the same, the sweep below is not affordable and the
#      script says so rather than discovering it four hours in.
#   2. whether `tabpfn-v2-5` -- the paper's actual model -- loads at all now. It did not when
#      every number here was measured; anofox v2026.08.11 claims to fix it and the community
#      build is pinned at 2026.08.14. This does not use it, it just records the answer.
#
#   bash scripts/pod/second_labeller_cpu.sh
set -uo pipefail
cd /workspace

R=/workspace/duckdb-rocket
OUT=/workspace/second_labeller
LABELLER="${LABELLER:-tabpfn-v2}"
# Small, and already measured under tabicl-v2, so the cost probe has something to compare against.
PROBE_SETS="${PROBE_SETS:-Beef Herring}"
# Above this many training rows a run peaked at 39.9 GB under tabicl-v2, so those go one at a
# time. Sharding them would OOM the box, which is what left five datasets without cubes for days.
BIG_TRAIN="${BIG_TRAIN:-400}"
SHARDS="${SHARDS:-4}"
BUDGET_MIN="${BUDGET_MIN:-600}"
TIMEOUT_MIN="${TIMEOUT_MIN:-60}"
# Per group, per call, fitted on tabicl-v2 (RESULTS.md 2026-08-16). The probe is called
# affordable if it lands within this many times that.
COST_CEILING="${COST_CEILING:-2.5}"

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
mkdir -p "$OUT/probe" "$OUT/pergroup"

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
build/release/duckdb -c "INSTALL httpfs;" >/dev/null 2>&1
build/release/duckdb -c "INSTALL anofox_tabfm FROM community;" >/dev/null 2>&1
build/release/duckdb -c "LOAD anofox_tabfm; SELECT extension_version FROM duckdb_extensions()
  WHERE extension_name = 'anofox_tabfm';" 2>/dev/null | tail -3
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := '${LABELLER}');" >/dev/null 2>&1 \
  && echo "  ${LABELLER} weights ok" || { echo "FATAL: no ${LABELLER} weights"; exit 1; }

# ---------------------------------------------------------------------------------------------
# PROBE 2 (cheap, so it goes first): does the paper's model load yet?
# ---------------------------------------------------------------------------------------------
log "probe: does tabpfn-v2-5 load on this build?"
# Recorded, not used. Every accuracy number in this project exists because this returned an error
# when it was asked in Phase 2; anofox v2026.08.11 claims to fix it. The pipeline does not list
# tabpfn-v2-5 as a labeller, so this asks the extension directly.
build/release/duckdb -c "LOAD httpfs; LOAD anofox_tabfm;
  SET anofox_tabfm_accept_hf_license = true;
  FROM tabfm_download('classification', model := 'tabpfn-v2-5');" > "$OUT/tabpfn25_download.log" 2>&1
if grep -qiE "error|fail" "$OUT/tabpfn25_download.log"; then
    echo "  tabpfn-v2-5 STILL DOES NOT DOWNLOAD"; grep -iE "error|fail" "$OUT/tabpfn25_download.log" | head -3
else
    build/release/duckdb -c "LOAD anofox_tabfm;
      CREATE TABLE t AS SELECT * FROM VALUES (1.0,2.0,'a'),(1.1,2.1,'a'),(9.0,8.0,'b'),(9.1,8.1,'b')
        AS v(f0,f1,y);
      CREATE TABLE q AS SELECT * FROM VALUES (1.05,2.05),(9.05,8.05) AS v(f0,f1);
      FROM tabfm_classify('t','y', test := 'q', model := 'tabpfn-v2-5');" \
      > "$OUT/tabpfn25_classify.log" 2>&1
    if grep -qiE "^Error|Exception" "$OUT/tabpfn25_classify.log"; then
        echo "  tabpfn-v2-5 downloads but DOES NOT CLASSIFY:"
        grep -iE "^Error|Exception" "$OUT/tabpfn25_classify.log" | head -3
    else
        echo "  *** tabpfn-v2-5 LOADS AND CLASSIFIES -- the paper's backbone is available ***"
        tail -6 "$OUT/tabpfn25_classify.log"
    fi
fi

# ---------------------------------------------------------------------------------------------
# PROBE 1: what does the second labeller cost per group?
# ---------------------------------------------------------------------------------------------
log "probe: what does ${LABELLER} cost, against tabicl-v2 on the same datasets?"
for ds in $PROBE_SETS; do
    uv run python -c "
from duckdb_rocket.datasets import load
x, y = load('$ds', 'train'); print('  $ds', x.shape)" 2>&1 | tail -1
done
for ds in $PROBE_SETS; do
    for m in "$LABELLER" tabicl-v2; do
        uv run python scripts/teacher_sweep.py --model "$m" --datasets "$ds" \
            --out-dir "$OUT/probe" --device cpu --per-group-soft \
            --threads 4 --onnx-threads 4 --test-chunk 128 \
            --budget-min 30 --timeout-min 20 > "$OUT/probe_${ds}_${m}.log" 2>&1
        echo "  ${ds} / ${m}: rc=$?"
    done
done
uv run python - "$OUT/probe" "$LABELLER" "$COST_CEILING" <<'PY'
import json, pathlib, sys
out, labeller, ceiling = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
def seconds(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    t = d.get("timings") or []
    cl = [float(r["classify_seconds"]) for r in t] if t else []
    return (sum(cl), d.get("accuracy"), len(cl))
ratios = []
for rep in sorted(out.glob("phase5_*.json")):
    if rep.name.endswith(("_soft.json", "_pergroup.json")):
        continue
    print(f"  {rep.name}")
for ds in {p.name.split("_")[1] for p in out.glob("phase5_*.json")}:
    a = out / f"phase5_{ds}_{labeller}.json"
    b = out / f"phase5_{ds}_gpu.json"          # tabicl-v2 is the default and is named _gpu
    if not (a.exists() and b.exists()):
        print(f"  {ds}: missing one arm ({a.exists()=}, {b.exists()=})")
        continue
    sa, acca, na = seconds(a)
    sb, accb, nb = seconds(b)
    if sb <= 0:
        print(f"  {ds}: no baseline timings")
        continue
    ratios.append(sa / sb)
    print(f"  {ds:16s} {labeller} {sa:7.1f}s acc={acca}   tabicl-v2 {sb:7.1f}s acc={accb}   "
          f"{sa / sb:.2f}x")
if ratios:
    worst = max(ratios)
    print(f"\n  cost ratio {worst:.2f}x against a {ceiling}x ceiling: "
          + ("AFFORDABLE, running the sweep" if worst <= ceiling else
             "TOO EXPENSIVE -- this is the mitra failure mode; not running the sweep"))
    pathlib.Path("/workspace/second_labeller/GO").write_text("1" if worst <= ceiling else "0")
else:
    print("\n  no comparable pair; not running the sweep")
    pathlib.Path("/workspace/second_labeller/GO").write_text("0")
PY

[ "$(cat "$OUT/GO" 2>/dev/null)" = "1" ] || { log "stopping before the sweep"; exit 0; }

# ---------------------------------------------------------------------------------------------
# The sweep. Big-context datasets alone, the rest sharded.
# ---------------------------------------------------------------------------------------------
log "the 29 datasets"
mapfile -t BIG < <(uv run python - "$BIG_TRAIN" <<'PY'
import json, sys
from duckdb_rocket.datasets import load
rows = json.load(open("reference/distill_gate.json"))["rows"]
keep = [r["dataset"] for r in rows if r.get("students") and max(r["students"].values()) < 0.90]
for d in sorted(keep):
    try:
        _, y = load(d, "train")
    except Exception:
        continue
    if len(y) >= int(sys.argv[1]):
        print(d)
PY
)
echo "  ${#BIG[@]} large-context datasets run one at a time: ${BIG[*]}"

for ds in "${BIG[@]}"; do
    echo "  --- $ds (serial)"
    uv run python scripts/teacher_sweep.py --model "$LABELLER" --datasets "$ds" \
        --out-dir "$OUT/pergroup" --device cpu --per-group-soft \
        --threads 4 --onnx-threads $(( NPROC / 4 )) --test-chunk 128 \
        --budget-min "$BUDGET_MIN" --timeout-min "$TIMEOUT_MIN" \
        > "$OUT/big_${ds}.log" 2>&1
    echo "    rc=$?  cube=$([ -f "$OUT/pergroup/phase5_${ds}_${LABELLER}_pergroup.json" ] && echo yes || echo NO)"
done

log "the rest, ${SHARDS} shards"
PER=$(( NPROC / SHARDS )); [ "$PER" -lt 1 ] && PER=1
T=2; O=$(( PER / T )); [ "$O" -lt 1 ] && O=1
BIG_CSV=$(IFS=,; echo "${BIG[*]}")
mapfile -t SHARD_LIST < <(uv run python - "$SHARDS" "$BIG_CSV" <<'PY'
import json, sys
shards, big = int(sys.argv[1]), set(filter(None, sys.argv[2].split(",")))
rows = json.load(open("reference/distill_gate.json"))["rows"]
keep = [r for r in rows if r.get("students") and max(r["students"].values()) < 0.90
        and r["dataset"] not in big]
keep.sort(key=lambda r: -r.get("n_test", 0))
buckets = [[] for _ in range(shards)]
for i, r in enumerate(keep):
    buckets[i % shards].append(r["dataset"])
for b in buckets:
    print(" ".join(b))
PY
)
PIDS=()
for i in "${!SHARD_LIST[@]}"; do
    echo "  shard $i: ${SHARD_LIST[$i]}"
    # shellcheck disable=SC2086
    uv run python scripts/teacher_sweep.py --model "$LABELLER" --datasets ${SHARD_LIST[$i]} \
        --out-dir "$OUT/pergroup" --device cpu --per-group-soft \
        --threads "$T" --onnx-threads "$O" --test-chunk 128 \
        --budget-min "$BUDGET_MIN" --timeout-min "$TIMEOUT_MIN" \
        > "$OUT/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "  shards done, $FAIL non-zero"

log "what came out"
echo "  cubes: $(ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | wc -l) of 29"
ls "$OUT"/pergroup/*_pergroup.json 2>/dev/null | sed 's#.*/#    #'
log "done"
