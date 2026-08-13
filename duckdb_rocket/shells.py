"""Where the DuckDB binaries live, on whichever platform this is running.

Every script had `tools/duckdb.exe` and `build/release/duckdb.exe` written into it, which is
correct on the development box and wrong everywhere else. That is fine right up until the first
Linux pod, where it fails as "no such shell" and looks like a missing build rather than a
hardcoded suffix.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def pinned_shell() -> Path:
    """The pinned upstream CLI under `tools/` — v1.5.5, matching the `duckdb` submodule.

    This one has no `rocket` extension in it. Use it for anything that only needs
    `anofox_tabfm`, and for checking that a built extension loads against a stock binary.
    """
    return ROOT / "tools" / f"duckdb{EXE_SUFFIX}"


def built_shell() -> Path:
    """The shell built by `scripts/build_extension.bat`, with `rocket` statically linked."""
    return ROOT / "build" / "release" / f"duckdb{EXE_SUFFIX}"


def loadable_extension() -> Path:
    """The standalone `rocket.duckdb_extension`, loadable into a stock CLI with `-unsigned`."""
    return ROOT / "build" / "release" / "extension" / "rocket" / "rocket.duckdb_extension"


def rocket_shell(preferred: Path | None = None) -> tuple[Path, list[str], str]:
    """A shell that can run `rocket_transform`, as (binary, extra argv, SQL to prepend).

    Two ways to get one, and the caller should not have to know which it got:

    * the shell built here, with `rocket` statically linked -- nothing to load;
    * the pinned upstream CLI plus a downloaded `rocket.duckdb_extension`, which needs
      `-unsigned` and an explicit `LOAD`.

    The second exists so a pod does not have to compile DuckDB. Building it takes 10-20 minutes
    and produces a file that is identical for every machine on the same commit and platform --
    it was rebuilt on four pods in one day before anyone counted the cost.
    """
    if preferred is not None:
        return preferred, [], ""
    built = built_shell()
    if built.exists():
        return built, [], ""
    ext = loadable_extension()
    if ext.exists():
        # -unsigned is required for a locally-built extension; the LOAD needs the full path
        # because the extension is not in the CLI's own extension directory.
        return pinned_shell(), ["-unsigned"], f"LOAD '{ext.as_posix()}';\n"
    raise FileNotFoundError(
        f"no shell with `rocket`: neither {built} nor {ext} exists. Build with "
        f"scripts/build_extension.bat, or let scripts/pod/bootstrap.sh fetch a prebuilt one."
    )


def resolve_shell(preferred: Path | None = None) -> Path:
    """Pick a usable shell: an explicit one, else the built one, else the pinned one.

    Falling back to the pinned CLI is deliberate. Several scripts only drive `anofox_tabfm` and
    never call `rocket_transform`, so requiring a local build to run them would make the
    extension a dependency of work that does not use it.
    """
    if preferred is not None:
        return preferred
    built = built_shell()
    return built if built.exists() else pinned_shell()
