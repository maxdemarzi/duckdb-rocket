"""Phase 2 probe: what can `anofox_tabfm` actually do with ROCKET-shaped input?

PLAN.md Phase 2 lists the open questions. swan pre-answered several against `tabicl-v2`; this
script re-asks them through our own SQL surface and adds the ones swan never had reason to ask
-- above all the 2,000-column question, since the paper's G=10 x 1,000-kernel split is built to
land exactly on TabPFN v2.5's 2,000-feature cap.

Two things learned the hard way and encoded here so they are not rediscovered:

1. **The test view must not contain the target column.** `tabfm_classify` unions train and test
   BY NAME internally, so a target column present in both produces a duplicate-name binder
   error that says nothing about the real cause.
2. **ONNX Runtime prints thousands of "Schema error: Trying to register schema..." lines to
   stderr on every model load.** They are harmless duplicate-registration warnings, but they
   bury the one line that matters. `real_errors()` strips them.

    uv run python scripts/probe_anofox.py
    uv run python scripts/probe_anofox.py --model tabicl-v2 --only feature_cap
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# DuckDB's result tables are drawn with box characters; the default Windows console codec
# cannot encode them and printing one raises UnicodeEncodeError mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DUCKDB = ROOT / "tools" / "duckdb.exe"

# anofox-tabfm is pre-1.0 and tags near-daily; an accuracy number without the extension version
# beside it is not reproducible next week.
EXTENSION_VERSION = "bc6d8af"

# tabpfn-v2-5 is the model this project wants, and it does not load in bc6d8af -- the published
# checkpoint no longer matches anofox's bundled ONNX graph. tabicl-v2 is the working fallback
# and is what swan ships. See reference/PHASE2_FINDINGS.md.
DEFAULT_MODEL = "tabicl-v2"

PREAMBLE = "LOAD anofox_tabfm;\nSET anofox_tabfm_accept_hf_license = true;\n"

# The 500-column ceiling `tabfm_list_models()` advertises is a *configurable guard*, not a model
# limit: anofox raises a Binder Error naming the setting to change. Raising it is what makes the
# paper's 2,000-feature groups reachable at all, so it is set explicitly rather than left to
# whatever the default happens to be.
RAISE_CAP = "SET anofox_tabfm_max_features = 4000;\n"

_ONNX_NOISE = re.compile(r"^(Schema error: Trying to register schema|\s*$)")


def real_errors(stderr: str) -> str:
    """Strip ONNX Runtime's duplicate-schema spam, which is thousands of lines of nothing."""
    return "\n".join(ln for ln in stderr.splitlines() if not _ONNX_NOISE.match(ln)).strip()


def synthetic_data(n_features: int, n_train: int = 60, n_test: int = 40) -> str:
    """A learnable synthetic problem with `n_features` scalar columns.

    Feature 0 carries the signal and the rest are noise -- deliberately the ROCKET situation in
    miniature. If the model cannot find one signal column among many noise columns, the
    column-count question is settled before accuracy enters into it.

    Note the test view omits the target: `tabfm_classify` unions the two views by name.
    """
    cols = []
    for j in range(n_features):
        noise = f"(hash(i * 1000 + {j}) % 1000) / 1000.0"
        cols.append(
            f"CASE WHEN i % 2 = 0 THEN 1.0 ELSE -1.0 END + 0.25 * {noise} AS f{j}"
            if j == 0
            else f"{noise} AS f{j}"
        )
    feature_cols = ",\n        ".join(cols)
    names = ", ".join(f"f{j}" for j in range(n_features))
    return f"""
CREATE OR REPLACE TABLE base AS
SELECT i AS id, (i % 2)::INTEGER AS y,
        {feature_cols}
FROM range({n_train + n_test}) t(i);

CREATE OR REPLACE VIEW train_v AS SELECT id, y, {names} FROM base WHERE id < {n_train};
CREATE OR REPLACE VIEW test_v  AS SELECT id, {names} FROM base WHERE id >= {n_train};
"""


def feature_list(n: int) -> str:
    return "[" + ", ".join(f"'f{j}'" for j in range(n)) + "]"


@dataclass
class Probe:
    name: str
    question: str
    sql: str
    expect: str = ""
    notes: list[str] = field(default_factory=list)


def run_sql(sql: str, timeout: int = 2400, raise_cap: bool = False) -> tuple[bool, str, str]:
    """Run SQL in a fresh DuckDB process. Returns (ok, stdout, filtered stderr).

    A fresh process per probe is deliberate: ONNX Runtime's API is process-global, one probe is
    expected to hard-throw, and a probe that kills the engine must not take the run with it.
    """
    # The SQL goes through a file, not `-c`. A 2,000-column feature list is well past Windows'
    # 32,767-character command-line limit, which surfaces as a bare
    # "FileNotFoundError: [WinError 206] The filename or extension is too long" -- an error that
    # names neither SQL nor length. Phase 3 and 4 inherit this constraint: any generated
    # wide-column SQL must be written to a script.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".sql", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(PREAMBLE + (RAISE_CAP if raise_cap else "") + sql)
        script = handle.name

    try:
        # encoding/errors are load-bearing on Windows: DuckDB draws its result tables with
        # box-drawing characters, which the default cp1252 console codec cannot decode, and the
        # resulting UnicodeDecodeError is raised on a reader *thread* -- so the probe silently
        # reports OK with empty output instead of failing.
        proc = subprocess.run(
            [str(DUCKDB), "-f", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    finally:
        Path(script).unlink(missing_ok=True)
    return proc.returncode == 0, (proc.stdout or "").strip(), real_errors(proc.stderr or "")


def build_probes(model: str, cap_sizes: list[int]) -> list[Probe]:
    f8 = feature_list(8)
    data8 = synthetic_data(8)
    probes = [
        Probe(
            "proba_shape",
            "Does tabfm_classify return `proba`, and in what type?",
            data8
            + f"""
SELECT any_value(typeof(proba)) AS proba_type,
       any_value(proba)         AS example_proba,
       count(*)                                     AS n_rows,
       count(*) FILTER (WHERE is_training)          AS training_rows,
       count(*) FILTER (WHERE NOT is_training)      AS test_rows
FROM tabfm_classify('train_v', 'y', test := 'test_v',
                    model := '{model}', features := {f8});
""",
            "swan: a per-class map. Also checks whether context rows come back too.",
        ),
        Probe(
            "n_estimators",
            "Does opts['n_estimators'] > 1 still hard-throw?",
            data8
            + f"""
SELECT count(*) AS n FROM tabfm_classify('train_v', 'y', test := 'test_v',
       model := '{model}', features := {f8}, opts := {{'n_estimators': 8}});
""",
            "swan: NotImplementedException, gated on anofox milestone M3",
        ),
        Probe(
            "list_valued_features",
            "Can one LIST column stand in for N scalar feature columns?",
            data8
            + f"""
CREATE OR REPLACE VIEW train_l AS
  SELECT id, y, [f0,f1,f2,f3,f4,f5,f6,f7]::DOUBLE[] AS feats FROM base WHERE id < 60;
CREATE OR REPLACE VIEW test_l AS
  SELECT id, [f0,f1,f2,f3,f4,f5,f6,f7]::DOUBLE[] AS feats FROM base WHERE id >= 60;
SELECT count(*) AS n FROM tabfm_classify('train_l', 'y', test := 'test_l',
       model := '{model}', features := ['feats']);
""",
            "swan's evidence pointed to no, but it was never tested directly",
        ),
        Probe(
            "precision_opt",
            "Is there a precision / AMP lever on the ONNX path?",
            data8
            + f"""
SELECT count(*) AS n FROM tabfm_classify('train_v', 'y', test := 'test_v',
       model := '{model}', features := {f8}, opts := {{'precision': 'fp32'}});
""",
            "unknown; 'no such option' is itself a result worth recording",
        ),
        Probe(
            "passthrough_id",
            "Is a non-feature id column echoed back, or silently dropped?",
            data8
            + f"""
SELECT * FROM tabfm_classify('train_v', 'y', test := 'test_v',
       model := '{model}', features := {f8}) LIMIT 1;
""",
            "swan: silently dropped; only target + named features come back",
        ),
        Probe(
            "row_order_stability",
            "Do two identical classify calls agree row-for-row in output order?",
            data8
            + f"""
CREATE OR REPLACE TABLE run_a AS SELECT row_number() OVER () AS pos, yhat, f0
  FROM tabfm_classify('train_v','y',test:='test_v',model:='{model}',features:={f8})
  WHERE NOT is_training;
CREATE OR REPLACE TABLE run_b AS SELECT row_number() OVER () AS pos, yhat, f0
  FROM tabfm_classify('train_v','y',test:='test_v',model:='{model}',features:={f8})
  WHERE NOT is_training;
SELECT count(*) AS n,
       count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) AS yhat_mismatches,
       count(*) FILTER (WHERE a.f0 IS DISTINCT FROM b.f0)     AS order_mismatches
FROM run_a a JOIN run_b b USING (pos);
""",
            "if stable, the G groups can be joined positionally and swan's rowid hack avoided",
        ),
        Probe(
            "row_order_matches_input",
            "Does output order match the test view's own order?",
            data8
            + f"""
WITH scored AS (
  SELECT row_number() OVER () AS pos, f0
  FROM tabfm_classify('train_v','y',test:='test_v',model:='{model}',features:={f8})
  WHERE NOT is_training
), expected AS (
  SELECT row_number() OVER (ORDER BY id) AS pos, f0 FROM test_v
)
SELECT count(*) AS n,
       count(*) FILTER (WHERE s.f0 IS DISTINCT FROM e.f0) AS mismatches
FROM scored s JOIN expected e USING (pos);
""",
            "the cheapest possible row-identity story if it holds",
        ),
    ]

    for n in cap_sizes:
        probes.append(
            Probe(
                f"feature_cap_{n}",
                f"Does a {n}-column classify call succeed?",
                synthetic_data(n)
                + f"""
SELECT count(*) AS n_rows,
       avg(CASE WHEN b.y = c.yhat THEN 1.0 ELSE 0.0 END) AS accuracy
FROM tabfm_classify('train_v', 'y', test := 'test_v',
                    model := '{model}', features := {feature_list(n)}) c
JOIN base b ON b.f0 = c.f0
WHERE NOT c.is_training;
""",
                "tabfm_list_models() advertises max_features=512 for tabicl-v2, 500 for tabpfn-v2-5",
            )
        )
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only", help="comma-separated probe name prefixes")
    parser.add_argument("--cap-sizes", default="100,500,512,513,1000,2000")
    parser.add_argument(
        "--raise-cap",
        action="store_true",
        help="SET anofox_tabfm_max_features high enough for the paper's 2,000-column groups",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "anofox_probe.json")
    args = parser.parse_args()

    cap_sizes = [int(v) for v in args.cap_sizes.split(",") if v.strip()]
    probes = build_probes(args.model, cap_sizes)
    if args.only:
        wanted = tuple(n.strip() for n in args.only.split(","))
        probes = [p for p in probes if p.name.startswith(wanted)]
        if not probes:
            parser.error("no probes matched --only")

    results = []
    for probe in probes:
        print(f"\n=== {probe.name}\n    {probe.question}", flush=True)
        started = time.perf_counter()
        try:
            ok, out, err = run_sql(probe.sql, raise_cap=args.raise_cap)
        except subprocess.TimeoutExpired:
            ok, out, err = False, "", "TIMEOUT"
        elapsed = time.perf_counter() - started

        print(f"    -> {'OK' if ok else 'FAILED'}  ({elapsed:.1f}s)", flush=True)
        for line in (out or err).splitlines()[:14]:
            print(f"    | {line}", flush=True)

        results.append(
            {
                "probe": probe.name,
                "question": probe.question,
                "expectation": probe.expect,
                "ok": ok,
                "seconds": round(elapsed, 1),
                "stdout": out,
                "stderr": err,
            }
        )

    report = {
        "extension_version": EXTENSION_VERSION,
        "model": args.model,
        "duckdb": subprocess.run(
            [str(DUCKDB), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "probes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
