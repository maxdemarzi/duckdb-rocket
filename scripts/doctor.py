"""Record the environment tuple that every reported number must be archived with.

    uv run python scripts/doctor.py                 # human-readable
    uv run python scripts/doctor.py --json out.json # machine-readable, for archiving

Ported in spirit from black_swan's `doctor.json`, which exists because a number without its
environment is not attributable. The additions specific to this project are
`cpu_supports_fast_bf16` and `tabpfn_autocast_default`: TabPFN's `inference_precision="auto"`
resolves per device, so those two fields are what distinguish a genuine fp32 baseline from a
reduced-precision one that will happily agree with another reduced-precision run.

Deliberately dependency-light and failure-tolerant. It runs on a pod that may be in a bad
state, and a diagnostic that raises when something is broken is a diagnostic that never reports
the interesting case -- so every probe degrades to a recorded error string.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _safe(fn, default=None):
    """Run a probe, recording any failure rather than propagating it."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - deliberate: see module docstring
        return f"<error: {type(exc).__name__}: {exc}>"


def collect_platform() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "libc": _safe(lambda: "-".join(p for p in platform.libc_ver() if p) or None),
    }


def collect_torch() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    return info


def collect_precision() -> dict:
    """The fields that decide whether an accuracy number is trustworthy.

    `cpu_supports_fast_bf16` is the one that is easy to forget and expensive to omit: where it
    is True, a CPU run under `inference_precision="auto"` is bf16, not fp32 -- so a CPU-vs-GPU
    agreement check proves nothing, because both sides are reduced precision.
    """
    import torch
    from tabpfn.utils import _cpu_supports_fast_bf16, infer_autocast_inference_mode

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", 0))

    return {
        "cpu_supports_fast_bf16": _safe(lambda: bool(_cpu_supports_fast_bf16())),
        "autocast_default_cpu": _safe(
            lambda: bool(
                infer_autocast_inference_mode(devices=[torch.device("cpu")], enable=None)
            )
        ),
        "autocast_default_cuda": _safe(
            lambda: bool(
                infer_autocast_inference_mode(
                    devices=[torch.device("cuda", 0)], enable=None
                )
            )
            if torch.cuda.is_available()
            else None
        ),
        "project_forces": "inference_precision=torch.float32",
    }


def collect_versions() -> dict:
    from importlib.metadata import version

    out = {}
    for name in ("numpy", "scikit-learn", "pyarrow", "duckdb", "aeon", "tabpfn", "torch"):
        out[name] = _safe(lambda n=name: version(n))
    return out


def collect_duckdb_cli() -> dict:
    exe = Path(__file__).resolve().parent.parent / "tools" / "duckdb.exe"
    if not exe.exists():
        exe_alt = Path(__file__).resolve().parent.parent / "tools" / "duckdb"
        exe = exe_alt if exe_alt.exists() else exe
    if not exe.exists():
        return {"path": None, "version": "<not present>"}
    return {
        "path": str(exe),
        "version": _safe(
            lambda: subprocess.run(
                [str(exe), "--version"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        ),
    }


def collect_git() -> dict:
    root = Path(__file__).resolve().parent.parent

    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    return {
        "commit": _safe(lambda: _git("rev-parse", "HEAD")),
        "dirty": _safe(lambda: bool(_git("status", "--porcelain"))),
    }


def doctor() -> dict:
    return {
        "platform": collect_platform(),
        "torch": _safe(collect_torch, {}),
        "precision": _safe(collect_precision, {}),
        "versions": collect_versions(),
        "duckdb_cli": collect_duckdb_cli(),
        "git": collect_git(),
    }


def _render(report: dict, indent: int = 0) -> None:
    pad = "  " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _render(value, indent + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{pad}{key}:")
            for item in value:
                _render(item, indent + 1)
        else:
            print(f"{pad}{key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the report to this path")
    args = parser.parse_args()

    report = doctor()
    _render(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")

    precision = report.get("precision", {})
    if precision.get("cpu_supports_fast_bf16") is True:
        print(
            "\nNOTE: this CPU has native bf16, so TabPFN's default "
            '`inference_precision="auto"` would run bf16 autocast on CPU here. A CPU-vs-GPU '
            "agreement check is NOT evidence of fp32 correctness on this machine."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
