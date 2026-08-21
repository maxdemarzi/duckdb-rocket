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
    git clone -q --depth 1 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git pull --ff-only || true
# The duckdb submodule is pinned to the tag that matches tools/duckdb; without it the extension
# build has nothing to build against.
#
# `--recurse-submodules` on the clone above, and a bare `submodule update --init`, both pull
# duckdb's entire 7.8 GB history to build one pinned commit -- ten minutes of a pod's life, every
# pod. `--depth 1` on the update does not fix it: it shallow-fetches the submodule's default branch
# tip, the pin is not the tip, and it fails. Fetching the commit by SHA takes 11 seconds and 102 MB.
# Located relative to the CLONE, not to this file. Piping the script in -- `ssh 'bash -s' <
# scripts/pod/bootstrap.sh`, which is the line runpod_cpu.py itself prints -- leaves BASH_SOURCE
# unset, and under `set -u` that aborted the whole bootstrap at "/shallow_clone.sh: No such file".
# By this point the repository is cloned and WORKDIR is where it is, so the clone is the reliable
# reference; BASH_SOURCE is used only as a fallback for running the file from somewhere else.
# shellcheck source=scripts/pod/shallow_clone.sh
_here="${BASH_SOURCE[0]:-}"
if [ -f "$WORKDIR/scripts/pod/shallow_clone.sh" ]; then
    source "$WORKDIR/scripts/pod/shallow_clone.sh"
elif [ -n "$_here" ]; then
    source "$(dirname "$_here")/shallow_clone.sh"
else
    echo "FATAL: cannot find shallow_clone.sh next to $WORKDIR or this script"; exit 1
fi
shallow_submodule duckdb || { echo "FATAL: could not obtain the duckdb submodule"; exit 1; }

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

log "prebuilt shell?"
# Building DuckDB from source takes 10-20 minutes and yields a binary that is identical for every
# machine on the same commit and platform. It was rebuilt on four pods in one day before anyone
# counted. The shell is cached rather than the loadable extension because every script here
# already resolves `built_shell()`, so caching it needs no code change anywhere.
#
# Two safeties, both load-bearing:
#   * the key identifies the exact inputs, so a stale binary cannot be silently picked up;
#   * the download is smoke-tested before it is trusted, and a failure falls back to building.
# Skipping either turns "we saved 15 minutes" into "we measured the wrong code".
#
# The key is a hash of what the build actually consumes, NOT the HEAD commit. Keying on HEAD was
# the obvious thing and it was wrong: every docs commit invalidated a byte-identical binary, and
# the artifact had to be republished three times in one afternoon while nothing that compiles had
# changed. These four inputs are the entire build surface -- the sources, the two cmake files and
# the pinned duckdb revision -- so equal key means equal binary, and a change to any of them
# changes the key. `git rev-parse` emits a tree hash for src/, blob hashes for the cmake files and
# the submodule's commit, all content-addressed, so the key is identical on every machine.
build_key() {
    git rev-parse "HEAD:src" "HEAD:CMakeLists.txt" "HEAD:extension_config.cmake" "HEAD:duckdb" \
        | sha256sum | cut -c1-12
}
KEY="$(build_key)"
PREBUILT_URL="${PREBUILT_URL:-https://github.com/maxdemarzi/duckdb-rocket/releases/download/prebuilt}"
ARTIFACT="duckdb-rocket-${KEY}-$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m)"

if [ ! -x build/release/duckdb ]; then
    mkdir -p build/release
    echo "looking for ${ARTIFACT}"
    if curl -fsSL -o build/release/duckdb "${PREBUILT_URL}/${ARTIFACT}" 2>/dev/null; then
        chmod +x build/release/duckdb
        if build/release/duckdb -c \
             "SELECT len(rocket_transform([1.0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]::DOUBLE[], 3, 0, 0));" \
             >/dev/null 2>&1; then
            echo "  using the prebuilt shell for build ${KEY} -- skipping the build"
        else
            echo "  prebuilt shell downloaded but failed its smoke test; building instead"
            rm -f build/release/duckdb
        fi
    else
        echo "  none published for build ${KEY}; building"
    fi
fi

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

# Publish it so the next pod skips the build entirely. Two commands, from a machine with `gh`:
#
#   cp build/release/duckdb "${ARTIFACT}"
#   gh release upload prebuilt "${ARTIFACT}" --repo maxdemarzi/duckdb-rocket --clobber
#
# NOT `"build/release/duckdb#${ARTIFACT}"`. That reads like a rename and isn't one: gh's own
# `--help` says `#text` "define[s] a display label" -- cosmetic, shown in the release's web UI --
# and the uploaded asset keeps the LOCAL file's basename (`duckdb`), which the keyed lookup above
# can never match. Verified empirically on gh 2.96.0: it silently uploaded as `duckdb`, not the
# key, and the wrong-name asset had to be deleted and re-uploaded from a renamed copy. The local
# file must carry the target name before `gh release upload` ever sees it.
#
# Publishing is deliberately manual otherwise: an artifact that uploads itself from whatever
# happens to be checked out is how the wrong binary becomes the cached one.
if [ ! -f /root/.rocket_prebuilt_note ]; then
    echo "  to skip this build next time:"
    echo "      cp build/release/duckdb \"${ARTIFACT}\""
    echo "      gh release upload prebuilt \"${ARTIFACT}\" --repo maxdemarzi/duckdb-rocket --clobber"
    touch /root/.rocket_prebuilt_note 2>/dev/null || true
fi

log "conformance — the Linux build must match the same golden vectors"
uv run python scripts/conformance.py

log "environment tuple"
uv run python scripts/doctor.py || true

log "ready. Run:  uv run python scripts/pod/sweep.py"
