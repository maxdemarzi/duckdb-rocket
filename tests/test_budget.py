"""`binding_cpu_count` has two independent blind spots -- a cpuset restriction that
`sched_getaffinity` catches and a CFS quota that it does not -- and this pins the second one.

Found on a real RunPod pod: `sched_getaffinity` returned 112 (the host's full core count) while
`cpu.cfs_quota_us=1190000` / `cpu.cfs_period_us=100000` capped the container at ~11.9 cores. Sizing
`threads`/`onnx_threads` from 112 there would reproduce the 132-thread-in-one-process incident this
function's docstring already records, just from the other detection path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.budget import binding_cpu_count  # noqa: E402


def _fake_read_text(table: dict[str, str]):
    # as_posix(), not str(): budget.py's paths are POSIX-absolute ("/sys/fs/cgroup/..."), and on a
    # Windows test host str(Path(...)) renders them with backslashes, which would never match the
    # table and silently take every lookup down the "file does not exist" branch instead of the
    # one this test means to exercise.
    def fake(self, *a, **kw):
        if self.as_posix() in table:
            return table[self.as_posix()]
        raise OSError(f"no such file: {self}")

    return fake


def test_cfs_quota_wins_when_narrower_than_affinity():
    table = {
        "/sys/fs/cgroup/cpu.max": "max 100000",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "1190000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    }
    with patch("os.sched_getaffinity", return_value=set(range(112)), create=True), \
         patch.object(Path, "read_text", _fake_read_text(table)):
        cores, source = binding_cpu_count()
    assert cores == 11
    assert source == "cgroup v1 cfs_quota"


def test_cgroup_v2_quota_wins_when_narrower_than_affinity():
    table = {"/sys/fs/cgroup/cpu.max": "800000 100000"}
    with patch("os.sched_getaffinity", return_value=set(range(64)), create=True), \
         patch.object(Path, "read_text", _fake_read_text(table)):
        cores, source = binding_cpu_count()
    assert cores == 8
    assert source == "cgroup v2 cpu.max"


def test_unlimited_v2_quota_falls_back_to_affinity():
    table = {"/sys/fs/cgroup/cpu.max": "max 100000"}
    with patch("os.sched_getaffinity", return_value=set(range(16)), create=True), \
         patch.object(Path, "read_text", _fake_read_text(table)):
        cores, source = binding_cpu_count()
    assert cores == 16
    assert source == "sched_getaffinity"


def test_no_quota_files_falls_back_to_affinity():
    with patch("os.sched_getaffinity", return_value=set(range(4)), create=True), \
         patch.object(Path, "read_text", _fake_read_text({})):
        cores, source = binding_cpu_count()
    assert cores == 4
    assert source == "sched_getaffinity"


def test_affinity_wins_when_narrower_than_the_quota():
    # A cpuset restriction tighter than the CFS quota -- sched_getaffinity already caught the
    # real limit, and the quota (a looser, separate ceiling) must not override it.
    table = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "3200000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    }
    with patch("os.sched_getaffinity", return_value=set(range(4)), create=True), \
         patch.object(Path, "read_text", _fake_read_text(table)):
        cores, source = binding_cpu_count()
    assert cores == 4
    assert source == "sched_getaffinity"


def test_negative_v1_quota_means_unset_and_falls_back():
    table = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    }
    with patch("os.sched_getaffinity", return_value=set(range(6)), create=True), \
         patch.object(Path, "read_text", _fake_read_text(table)):
        cores, source = binding_cpu_count()
    assert cores == 6
    assert source == "sched_getaffinity"
