"""What this process may actually spend, as opposed to what it can see.

Shared because the same trap caught two scripts. `phase5_pipeline.py` already pinned `threads`
explicitly, with a comment about a many-core pod where four concurrent runs each sized a pool
from the visible core count and all died near completion. Memory is the identical bug one level
down, and `phase3_sql.py` had neither.
"""

from __future__ import annotations

import os
from pathlib import Path


def binding_memory_bytes() -> tuple[int, str]:
    """The memory this process may actually use, and where that number came from.

    Inside a container `free` and `/proc/meminfo` report the *host's* RAM, so DuckDB's default
    limit -- 80% of what it can see -- can land far above the cgroup ceiling. It then allocates
    happily until the kernel kills it, which is indistinguishable from a hang: no DuckDB error,
    no Python traceback, just a dead child. Measured on a RunPod CPU instance that reported
    124 GB through `free` against a 29 GB cgroup limit.
    """
    for path, kind in (
        (Path("/sys/fs/cgroup/memory.max"), "cgroup v2"),                      # unified
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), "cgroup v1"),    # legacy
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":  # v2's "no limit"
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 reports a sentinel near 2^63 when unlimited; anything that large is not a limit.
        if 0 < value < (1 << 62):
            return value, kind

    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "visible RAM"

    import ctypes  # Windows: no cgroups, no sysconf

    class _Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _Status()
    status.dwLength = ctypes.sizeof(_Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullTotalPhys), "visible RAM"


def default_memory_limit() -> str:
    """70% of the binding limit, as a DuckDB size string.

    Not 80%: the budget has to cover the Python parent, the ONNX session's own allocations and
    the OS, none of which are inside DuckDB's accounting. ItalyPowerDemand reached 25.7 GB on a
    box with no limit set at all and took the machine down with it.

    Note what this does NOT do. `tabfm_classify` allocates outside the buffer manager, so this
    setting cannot contain it -- a 6 GB limit died *faster* than a 20 GB one on the same input.
    Bounding that needs a smaller test batch (`--test-chunk`), not a smaller budget.
    """
    total, _ = binding_memory_bytes()
    return f"{max(int(total * 0.70) // (1024 ** 3), 1)}GB"
