"""Golden vectors: the conformance fixture the C++ extension is validated against.

The input series are **generated from the same PRNG as the kernels** rather than shipped as
data. That is the point of the design: a C++ conformance test can reconstruct the exact input
from SPEC.md alone, so a mismatch localises to the transform rather than to how someone parsed
a file. The parquet artifacts exist so a failure can be inspected, not because the C++ side
needs them to produce its own inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .prng import SplitMix64
from .rocket import generate_kernels, transform

# Fixed by specification. Changing any of these invalidates the committed vectors, which is
# why they live here as named constants rather than as script arguments with defaults.
GOLDEN_SEED = 20260811
GOLDEN_INPUT_SEED = 0xC0FFEE
GOLDEN_N_SERIES = 8
GOLDEN_N_TIMEPOINTS = 128
GOLDEN_NUM_KERNELS = 64
GOLDEN_FIRST_KERNEL = 0

# A second, offset bank. Kernel indices are global, so this exercises the property the whole
# group design depends on: that group g's kernels are addressable without generating group 0.
# A C++ port that mishandles `first_kernel` passes the first fixture and fails this one.
GOLDEN_OFFSET_FIRST_KERNEL = 9_000
GOLDEN_OFFSET_NUM_KERNELS = 16


def golden_input(
    n_series: int = GOLDEN_N_SERIES,
    n_timepoints: int = GOLDEN_N_TIMEPOINTS,
    seed: int = GOLDEN_INPUT_SEED,
) -> np.ndarray:
    """The fixture input series, reproducible from the spec.

    Row-major draw order: series 0's timepoints in order, then series 1's, and so on, from a
    single stream. Stated explicitly because it is the kind of detail a reimplementation
    silently gets backwards.
    """
    rng = SplitMix64(seed)
    values = [rng.next_normal() for _ in range(n_series * n_timepoints)]
    return np.asarray(values, dtype=np.float64).reshape(n_series, n_timepoints)


def _kernel_table(kernels, first_kernel: int) -> dict:
    return {
        "kernel_index": np.arange(
            first_kernel, first_kernel + kernels.num_kernels, dtype=np.int64
        ),
        "length": kernels.lengths,
        "bias": kernels.biases,
        "dilation": kernels.dilations,
        "padding": kernels.paddings,
        "weights": [
            kernels.weights[kernels.offsets[i] : kernels.offsets[i + 1]].tolist()
            for i in range(kernels.num_kernels)
        ],
    }


def build_golden(
    seed: int = GOLDEN_SEED,
    num_kernels: int = GOLDEN_NUM_KERNELS,
    first_kernel: int = GOLDEN_FIRST_KERNEL,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return `(input_series, kernel_table, features)` for one fixture."""
    x = golden_input()
    kernels = generate_kernels(
        seed, GOLDEN_N_TIMEPOINTS, num_kernels, first_kernel=first_kernel
    )
    return x, _kernel_table(kernels, first_kernel), transform(x, kernels)


def write_golden(out_dir: Path) -> list[Path]:
    """Write every fixture to `out_dir`, returning the paths written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(table: pa.Table, name: str) -> None:
        path = out_dir / name
        # No compression: these files are small, and a conformance fixture that can be read by
        # any parquet reader without codec negotiation is worth more than the saved bytes.
        pq.write_table(table, path, compression="none")
        written.append(path)

    x = golden_input()
    _write(
        pa.table(
            {
                "series_index": np.arange(x.shape[0], dtype=np.int64),
                "values": [row.tolist() for row in x],
            }
        ),
        "input_series.parquet",
    )

    for label, n_kernels, first in (
        ("base", GOLDEN_NUM_KERNELS, GOLDEN_FIRST_KERNEL),
        ("offset", GOLDEN_OFFSET_NUM_KERNELS, GOLDEN_OFFSET_FIRST_KERNEL),
    ):
        _, kernel_table, features = build_golden(GOLDEN_SEED, n_kernels, first)
        _write(pa.table(kernel_table), f"kernels_{label}.parquet")
        _write(
            pa.table(
                {
                    "series_index": np.arange(features.shape[0], dtype=np.int64),
                    "features": [row.tolist() for row in features],
                }
            ),
            f"features_{label}.parquet",
        )

    return written
