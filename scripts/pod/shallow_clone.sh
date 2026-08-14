#!/usr/bin/env bash
# Get this repo and its pinned duckdb submodule onto a pod without pulling 7.8 GB of history.
#
#   source scripts/pod/shallow_clone.sh
#   clone_repo /workspace/duckdb-rocket https://github.com/maxdemarzi/duckdb-rocket.git
#
# The submodule is the whole problem. `git clone --recurse-submodules` and a bare
# `git submodule update --init` both fetch duckdb's complete history -- 7.8 GB, ten minutes on a
# 930 Mbps pod -- to build one pinned commit. Measured alternative: fetching that single commit
# takes **11 seconds and 102 MB**.
#
# The obvious shortcut does not work. `git submodule update --init --depth 1` shallow-fetches the
# tip of the submodule's default branch, and the pin is not the tip, so it fails with "reference is
# not a tree". That failure is why an earlier attempt at this reverted to the full clone.
#
# What does work is asking for the commit by name: `git fetch --depth 1 origin <sha>`. GitHub
# serves arbitrary SHAs (uploadpack.allowAnySHA1InWant), so this needs no branch, no tag and no
# guess about where the pin sits in history. A host that refuses falls back to a full fetch rather
# than failing the run.
set -uo pipefail

# Fetch exactly the commit a submodule is pinned to, and nothing else.
#
# Deliberately not `git submodule update`: that command's whole model is "resolve the pin from
# history I already have", and the point here is to never download that history.
shallow_submodule() {
    local path="$1" sha url
    sha="$(git rev-parse "HEAD:$path" 2>/dev/null)" || {
        echo "  no submodule pinned at '$path'"; return 1; }
    url="$(git config -f .gitmodules "submodule.$path.url")" || return 1

    # Already at the pin: nothing to do. Checked before the rm, so a warm cache volume is not
    # thrown away and re-fetched on every run.
    if [ -e "$path/.git" ] && [ "$(git -C "$path" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
        echo "  $path already at ${sha:0:12}"
        return 0
    fi

    echo "  fetching $path at ${sha:0:12} (one commit, not the history)"
    rm -rf "$path" && mkdir -p "$path"
    git -C "$path" init -q
    git -C "$path" remote add origin "$url"
    if ! git -C "$path" fetch -q --depth 1 origin "$sha"; then
        # A server with allowAnySHA1InWant disabled cannot serve a bare SHA. Correctness beats
        # speed: take the slow path rather than leaving a half-populated submodule behind.
        echo "  server will not serve that SHA directly; falling back to a full fetch"
        git -C "$path" fetch -q origin || return 1
    fi
    git -C "$path" checkout -q "$sha" || git -C "$path" checkout -q FETCH_HEAD || return 1

    # Register it so the parent does not consider the tree dirty and so `git submodule` commands
    # still work; the gitlink is already correct, this only writes .git/config.
    git submodule init "$path" >/dev/null 2>&1 || true
}

clone_repo() {
    local dir="$1" url="$2"
    # --filter=blob:none would help a repo with large history; this one is small and the parent
    # clone is already seconds. --depth 1 is enough and keeps `git rev-parse HEAD:src` working,
    # which the build cache key depends on.
    [ -d "$dir/.git" ] || git clone -q --depth 1 "$url" "$dir" || return 1
    cd "$dir" || return 1
    shallow_submodule duckdb
}
