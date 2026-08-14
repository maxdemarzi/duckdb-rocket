#!/usr/bin/env bash
# Get a CUDA-capable anofox_tabfm without building it. Sourceable:
#
#   source scripts/pod/anofox_cuda.sh          # sets ANOFOX_EXT and LD_LIBRARY_PATH
#   ANOFOX_FLAVOR=cuda bash scripts/pod/anofox_cuda.sh --publish   # after a local build
#
# Building the cuda flavor takes ~30 minutes and happens on EVERY GPU pod, because no GPU build
# of anofox_tabfm is published for any platform -- ext.anofox.com, the host its own error message
# names, has no DNS record (DataZooDE/anofox-tabfm#25). That was an hour of build time on one
# afternoon across two pods.
#
# What actually has to be cached is only the extension. The ONNX Runtime shared libraries it links
# are Microsoft's own published archive, so they are fetched from the upstream release rather than
# re-hosted -- that keeps our artifact at ~60 MB instead of ~600 MB and means we are not
# redistributing someone else's binaries.
#
# Keyed on the anofox commit, the flavor and the platform: those three fix the binary, and a commit
# that changes nothing relevant simply produces a cache miss and a rebuild, which is the safe
# direction to be wrong in.
set -uo pipefail

ANOFOX_DIR="${ANOFOX_DIR:-/workspace/anofox}"
ANOFOX_FLAVOR="${ANOFOX_FLAVOR:-cuda}"
ORT_VERSION="${ORT_VERSION:-1.23.2}"          # must match cmake/ort.cmake's TABFM_ORT_VERSION
PREBUILT_URL="${PREBUILT_URL:-https://github.com/maxdemarzi/duckdb-rocket/releases/download/prebuilt}"
ORT_DIR="${ORT_DIR:-/workspace/onnxruntime-linux-x64-gpu-${ORT_VERSION}}"

_plat="$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m)"

# The published asset name carries the key; the LOCAL file must not. DuckDB derives an
# extension's entrypoint symbol from its FILENAME, so a file called
# anofox_tabfm-cuda-<key>-linux_x86_64.duckdb_extension makes it look for
# `anofox_tabfm-cuda-<key>-linux_x86_64_duckdb_cpp_init`, which does not exist:
#
#   IO Error: Extension '...' did not contain the expected entrypoint function 'a_duckdb_cpp_init'
#
# So the key lives in the directory and the file keeps its canonical name. The rocket shell cache
# does not have this problem because a shell is a binary, not an extension.
anofox_key() {
    local c
    c="$(git -C "$ANOFOX_DIR" rev-parse --short=12 HEAD 2>/dev/null)" || return 1
    echo "anofox_tabfm-${ANOFOX_FLAVOR}-${c}-${_plat}.duckdb_extension"
}

anofox_local_path() {
    local c
    c="$(git -C "$ANOFOX_DIR" rev-parse --short=12 HEAD 2>/dev/null)" || return 1
    echo "${CACHE_ROOT:-/workspace/cache}/${ANOFOX_FLAVOR}-${c}-${_plat}/anofox_tabfm.duckdb_extension"
}

# ORT ships its own libs; we only ever cache the extension.
fetch_ort() {
    [ -d "$ORT_DIR/lib" ] && return 0
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
    echo "  fetching ONNX Runtime ${ORT_VERSION} from upstream"
    curl -fsSL "$url" | tar -xz --no-same-owner -C "$(dirname "$ORT_DIR")" || return 1
}

# Trust nothing that has not answered a query. A cached extension that loads but cannot see the
# GPU would silently turn a "GPU run" into a CPU run, which is worse than a rebuild.
smoke_test() {
    local shell_bin="$1" ext="$2" n
    n=$("$shell_bin" -unsigned -noheader -list -c \
        "LOAD '$ext'; SELECT count(*) FROM tabfm_devices() WHERE device_id LIKE 'cuda%' AND usable;" \
        2>/dev/null | tail -1)
    [ "${n:-0}" -ge 1 ]
}

anofox_fetch() {
    local shell_bin="$1" key dest
    key="$(anofox_key)" || { echo "  no anofox checkout at $ANOFOX_DIR"; return 1; }
    dest="$(anofox_local_path)"
    mkdir -p "$(dirname "$dest")"
    echo "looking for $key"
    curl -fsSL -o "$dest" "$PREBUILT_URL/$key" 2>/dev/null || { echo "  none published; build required"; return 1; }
    fetch_ort || { echo "  ORT fetch failed; build required"; return 1; }
    export LD_LIBRARY_PATH="$ORT_DIR/lib:${LD_LIBRARY_PATH:-}"
    if smoke_test "$shell_bin" "$dest"; then
        export ANOFOX_EXT="$dest"
        echo "  using the cached extension -- skipping a ~30 minute build"
        return 0
    fi
    echo "  cached extension failed its CUDA smoke test; building instead"
    rm -f "$dest"
    return 1
}

anofox_publish() {
    local built key
    built="$(find "$ANOFOX_DIR/build" -name 'anofox_tabfm.duckdb_extension' | head -1)"
    [ -n "$built" ] || { echo "nothing built under $ANOFOX_DIR/build"; return 1; }
    key="$(anofox_key)" || return 1
    echo "publish with:"
    echo "  gh release upload prebuilt \"$built#$key\" --repo maxdemarzi/duckdb-rocket --clobber"
}

case "${1:-}" in
    --publish) anofox_publish ;;
    --key)     anofox_key ;;
    *)         : ;;   # sourced: call anofox_fetch <shell> yourself
esac
