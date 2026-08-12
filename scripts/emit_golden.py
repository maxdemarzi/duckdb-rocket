"""Write the golden vectors to reference/golden/.

    uv run python scripts/emit_golden.py

Regenerating these changes what the C++ conformance test compares against, so run it
deliberately and review the diff -- a surprising change here means the specification moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckdb_rocket.golden import write_golden  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reference" / "golden"


def main() -> int:
    written = write_golden(OUT_DIR)
    for path in written:
        print(f"{path.relative_to(OUT_DIR.parent.parent)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
