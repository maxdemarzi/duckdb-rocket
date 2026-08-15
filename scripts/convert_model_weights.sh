#!/usr/bin/env bash
# Make the second, third and fourth in-context labellers usable. One-time, per machine.
#
# `anofox_tabfm` advertises seven models. Measured on the 2026.08.14 community build, two run out of
# the box and four do not:
#
#   tabicl-v2    works                      BSD-3-Clause          our teacher for every archived run
#   mitra        works                      Apache-2.0            max_features 100, not 500
#   tabpfn-v2    "checkpoint is missing 127 of the 129 tensors"   Apache-2.0 + attribution
#   orion-bix    "unsupported pickle opcode 0x65"                 MIT, classification only
#   tabpfn-v2-5  "missing 248 of 250 tensors"                     NON-COMMERCIAL
#   tabpfn-v3    "unsupported pickle opcode 0x65"                 NON-COMMERCIAL
#
# The two failures are different faults with the same remedy. The tabpfn checkpoints are published
# under tensor names the exported graph was not built against; `orion-bix` and `tabpfn-v3` are torch
# zips whose pickle stream uses opcode 0x65 (APPENDS), which `src/tabfm_ckpt.cpp` does not implement.
# Either way the engine prefers a sibling `model.safetensors` if one exists, and upstream's
# converters write exactly that. v2026.08.11 made the engine prefer it and made the error message say
# so; it did not make the conversion happen, so `tabfm_download` still fetches a checkpoint the
# engine cannot read.
#
# This script converts only the two that are licensed for commercial use. tabpfn-v2-5 and tabpfn-v3
# are non-commercial and are deliberately left out: they are usable as research instruments to find
# out whether an ensemble of labellers works, and are not shippable in a product, so making them
# effortless to run here would invite exactly the wrong mistake later.
#
# Needs torch, safetensors and huggingface_hub, which `uv sync` already provides, plus ~450 MB of
# transient download. Writes into ~/.cache/anofox-tabfm, where the extension looks.
set -euo pipefail

WORK="${1:-${TMPDIR:-/tmp}/anofox-converters}"
CACHE="${ANOFOX_CACHE:-$HOME/.cache/anofox-tabfm}"
REPO=https://github.com/DataZooDE/anofox-tabfm.git

log() { printf '\n=== %s\n' "$*"; }

log "converter tooling -> $WORK"
# Sparse and blobless: the full repository carries the ONNX graphs and vendored deps. The three
# exporter directories and resources/ are 20 MB.
if [ ! -d "$WORK/.git" ]; then
    git clone -q --depth 1 --filter=blob:none --sparse "$REPO" "$WORK"
    git -C "$WORK" sparse-checkout set tools/export_tabpfn tools/export_orion_bix resources
fi
git -C "$WORK" log --oneline -1

# Each converter is keyed by a tensor map committed in the repository, and reports how many of those
# keys it found in the real checkpoint. A partial match means the graph and the released architecture
# have diverged: injection would leave initializers unbound and the model would fail at inference, or
# worse, run on whatever those buffers happened to hold. So the counts are asserted, not read.
expect_all_tensors() {
    local name="$1" out="$2"
    if ! grep -qE "missing: 0" <<<"$out"; then
        echo "FATAL: $name conversion did not match every tensor-map key:"
        grep -E "tensor-map keys|missing" <<<"$out" || true
        exit 1
    fi
    grep -E "wrote [0-9]+ tensors" <<<"$out"
}

log "tabpfn-v2 (Apache-2.0 + attribution)"
OUT=$(PYTHONPATH="$WORK/tools/export_tabpfn/src" \
      uv run python "$WORK/tools/export_tabpfn/convert_weights.py" classification "$CACHE" 2>&1)
expect_all_tensors tabpfn-v2 "$OUT"

log "orion-bix (MIT, classification only)"
OUT=$(uv run python "$WORK/tools/export_orion_bix/convert_weights.py" "$CACHE" 2>&1)
expect_all_tensors orion-bix "$OUT"

# The conversion writing a file is not the claim. The claim is that the model classifies, and the
# failure this replaces was one where everything up to inference succeeded.
log "each labeller against a two-class problem it cannot get wrong"
SHELL_BIN="${DUCKDB_SHELL:-build/release/duckdb}"
if [ ! -x "$SHELL_BIN" ]; then
    echo "  no shell at $SHELL_BIN; set DUCKDB_SHELL to verify"
    exit 0
fi
rc=0
for m in tabicl-v2 mitra tabpfn-v2 orion-bix; do
    printf '  %-12s ' "$m"
    got=$("$SHELL_BIN" -c "LOAD anofox_tabfm; SET anofox_tabfm_accept_hf_license = true;
      CREATE TABLE t AS SELECT * FROM (VALUES
        (1.0,0.1,'a'),(1.1,0.2,'a'),(0.9,0.0,'a'),(1.2,0.1,'a'),
        (5.0,9.1,'b'),(5.1,9.2,'b'),(4.9,9.0,'b'),(5.2,9.1,'b')) v(f1,f2,y);
      CREATE TABLE q AS SELECT * FROM (VALUES (1.05,0.15),(5.05,9.15)) v(f1,f2);
      SELECT string_agg(yhat, ',') FROM tabfm_classify('t','y', test := 'q', model := '$m');" 2>&1 \
      | tr -d '\r' | grep -oE '\ba,b\b' | head -1)
    if [ "$got" = "a,b" ]; then echo "ok"; else echo "FAILED"; rc=1; fi
done
exit $rc
