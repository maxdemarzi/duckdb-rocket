#!/usr/bin/env bash
# Bring a fresh RunPod instance to the point where the breadth sweep can run.
#
#   curl -fsSL <raw-url>/scripts/pod/bootstrap.sh | bash
#   # or, once the repo is cloned:
#   bash scripts/pod/bootstrap.sh 2>&1 | tee bootstrap.log
#
# Idempotent: every step checks before doing. A pod is billed by the hour, so re-running after a
# dropped SSH session should cost seconds, not another full build.
#
# **Never put the Prior Labs token in this file.** It is injected as TABPFN_TOKEN in the pod's
# environment (PLAN.md), and it is only needed if the TabPFN oracle is being run here; the
# DuckDB pipeline uses anofox_tabfm's own weights and its own licence flag.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/maxdemarzi/duckdb-rocket.git}"
WORKDIR="${WORKDIR:-/workspace/duckdb-rocket}"
DUCKDB_VERSION="${DUCKDB_VERSION:-v1.5.5}"

log() { printf '\n=== %s\n' "$*"; }

log "system packages"
# Checked per tool, not on `cmake` alone. RunPod's PyTorch images ship cmake but neither ninja
# nor a compiler, so a single-tool guard skips the install and the failure surfaces much later
# as "CMAKE_CXX_COMPILER not set" during the extension build.
missing=()
for tool in git cmake ninja g++ unzip curl; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "missing: ${missing[*]}"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        git cmake ninja-build build-essential unzip curl ca-certificates
fi
cmake --version | head -1
ninja --version
g++ --version | head -1

log "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "PATH includes $(command -v uv)"

log "repository"
if [ ! -d "$WORKDIR/.git" ]; then
    git clone --recurse-submodules "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git pull --ff-only || true
# The duckdb submodule is pinned to the tag that matches tools/duckdb; without it the extension
# build has nothing to build against.
git submodule update --init --depth 1

log "python environment"
uv sync

log "duckdb CLI ${DUCKDB_VERSION}"
mkdir -p tools
if [ ! -x tools/duckdb ]; then
    curl -fsSL -o /tmp/duckdb_cli.zip \
        "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip"
    unzip -o -q /tmp/duckdb_cli.zip -d tools
    chmod +x tools/duckdb
fi
tools/duckdb --version

log "anofox_tabfm + tabicl-v2 weights"
# Downloading here rather than mid-sweep: the first classify call would otherwise pay for it,
# and a network failure would surface as a dataset failing rather than as a setup problem.
tools/duckdb -c "
INSTALL anofox_tabfm FROM community;
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;
FROM tabfm_download('classification', model := 'tabicl-v2');
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_tabfm';
" 2>/dev/null | tail -20

log "execution providers — does the GPU do anything for anofox?"
# The answer decides whether this pod is buying inference speed or just uncontended cores, and
# it belongs in the run record either way. Locally (Windows) only a CPU provider was offered.
tools/duckdb -c "LOAD anofox_tabfm; FROM tabfm_devices();" 2>/dev/null | tail -12

log "building the rocket extension"
if [ ! -x build/release/duckdb ]; then
    cmake -G Ninja -B build/release -S duckdb \
        -DCMAKE_BUILD_TYPE=Release \
        -DDUCKDB_EXTENSION_CONFIGS="$WORKDIR/extension_config.cmake" \
        -DEXTENSION_STATIC_BUILD=1 \
        -DBUILD_UNITTESTS=0 \
        -DBUILD_SHELL=1
    cmake --build build/release
fi
build/release/duckdb -c "SELECT len(rocket_transform([1.0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]::DOUBLE[], 3, 0, 0)) AS n;"

log "conformance — the Linux build must match the same golden vectors"
uv run python scripts/conformance.py

log "environment tuple"
uv run python scripts/doctor.py || true

log "ready. Run:  uv run python scripts/pod/sweep.py"
