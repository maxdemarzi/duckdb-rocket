"""duckdb-rocket — training-free time-series classification in DuckDB."""

from .prng import SplitMix64, kernel_seed
from .rocket import (
    FEATURES_PER_KERNEL,
    KERNEL_LENGTHS,
    Kernels,
    generate_kernel,
    generate_kernels,
    normalize_series,
    transform,
)

__all__ = [
    "FEATURES_PER_KERNEL",
    "KERNEL_LENGTHS",
    "Kernels",
    "SplitMix64",
    "generate_kernel",
    "generate_kernels",
    "kernel_seed",
    "normalize_series",
    "transform",
]
