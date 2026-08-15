"""`features := [...]` must name exactly the columns the schema creates.

A name in that list matching no column is **silently dropped**: the macro filters with
`COLUMNS(lambda c: ...)`, which keeps what matches and says nothing about what does not, so by the
time bind could object a typo is indistinguishable from a deliberate omission. The call succeeds,
having trained on fewer features than asked for, and returns predictions that look entirely normal.
DataZooDE/anofox-tabfm#34, fixed upstream 2026-08-15; the community repository serves 2026.08.14, so
an installing user -- including us -- still gets the silent version.

Every accuracy this project has recorded came out of `build_sql`, so this is the check that says
whether any of them were affected. It is deliberately a comparison between the two strings the
generator emits, not a re-derivation of what they ought to be: re-deriving would reproduce whatever
mistake the generator makes.

The `ts` family is where this could plausibly bite. Those names are the extension's own, not ones we
chose; `quantile_0.1` carries a dot that DuckDB reads as a qualifier unless quoted, so the schema
identifier and the string literal genuinely differ in form while having to agree in content.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from duckdb_rocket.pipeline import RocketPFNConfig  # noqa: E402

import phase5_pipeline as p5  # noqa: E402

# Real names, in the shape `ts_feature_names()` probes off DESCRIBE: dotted quantiles included,
# because those are the ones whose identifier form and literal form differ.
TS_NAMES = ["mean", "variance", "quantile_0.1", "quantile_0.9", "fourier_re_1", "acf_1"]


def build(features: str, n_groups: int = 40, num_kernels: int = 10_000) -> str:
    cfg = RocketPFNConfig(num_kernels=num_kernels, n_groups=n_groups, seed=0, n_estimators=1)
    cfg.validate()
    meta = {"raw_parquet": "/tmp/raw.parquet", "n_train": 50, "n_test": 150}
    return p5.build_sql(cfg, meta, Path("/tmp/wd"), 4, "20GB", Path("/tmp/wd"), 128, 16,
                        features=features, ts_names=TS_NAMES if features != "rocket" else None)


def code(sql: str) -> str:
    """The SQL with comment lines removed.

    The generator's comments quote the very constructs these tests search for -- including a warning
    naming the classify call -- so a bare substring search finds the warning about the thing instead
    of the thing. The same reason `test_phase5_sql.py` strips them.
    """
    return "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))


def requested(sql: str) -> list[str]:
    """The names in the feature list handed to the classify call."""
    m = re.search(r"tabfm_classify\(.*?features := \[(.*?)\]\)", code(sql), re.S)
    assert m, "no feature list argument on the classify call in the generated SQL"
    return re.findall(r"'([^']*)'", m.group(1))


def declared(sql: str) -> list[str]:
    """The column names in the CREATE TABLE that train_cur and test_cur share."""
    m = re.search(r"CREATE OR REPLACE TABLE train_cur \((.*?)\);", sql, re.S)
    assert m, "no train_cur declaration in the generated SQL"
    out = []
    for col in m.group(1).split(","):
        tok = col.strip().split()[0]
        out.append(tok[1:-1] if tok.startswith('"') else tok)
    return out


@pytest.mark.parametrize("features", ["rocket", "ts", "both"])
class TestRequestedNamesExist:
    def test_every_requested_name_is_declared(self, features):
        sql = build(features, n_groups=1 if features == "ts" else 40,
                    num_kernels=500 if features == "ts" else 10_000)
        missing = [n for n in requested(sql) if n.lower() not in {d.lower() for d in declared(sql)}]
        assert not missing, f"{features}: silently dropped by the extension: {missing[:5]}"

    def test_the_two_lists_agree_in_order_and_length(self, features):
        # Order matters beyond the drop bug: the id-recovery key is built from the same list, so a
        # permutation between the two forms would join rows to the wrong predictions.
        sql = build(features, n_groups=1 if features == "ts" else 40,
                    num_kernels=500 if features == "ts" else 10_000)
        req = requested(sql)
        dec = [d for d in declared(sql) if d not in ("id", "split", "label", "y", "k")]
        assert req == dec

    def test_the_target_is_not_offered_as_a_feature(self, features):
        sql = build(features, n_groups=1 if features == "ts" else 40,
                    num_kernels=500 if features == "ts" else 10_000)
        assert "y" not in requested(sql)


class TestDottedNamesSurvive:
    def test_a_dotted_ts_name_is_quoted_in_the_schema_and_bare_in_the_list(self):
        # Both forms have to appear, and they have to describe the same column. Quoting the literal
        # or leaving the identifier bare are each a silent drop -- the first matches no column named
        # `"quantile_0.1"` with quotes in it, the second is parsed as table `quantile_0`.
        sql = build("ts", n_groups=1, num_kernels=500)
        assert '"quantile_0.1" DOUBLE' in sql
        assert "'quantile_0.1'" in sql
        assert "quantile_0.1" in requested(sql)


class TestTheGuardIsEmitted:
    def test_the_run_asserts_the_same_thing_against_the_live_schema(self):
        # The static check above covers the generator; this one covers a schema that drifts at
        # runtime for a reason the generator cannot see.
        sql = build("rocket")
        assert "features_check" in sql
        assert "DESCRIBE train_cur" in sql

    def test_the_guard_runs_before_the_first_classify(self):
        sql = code(build("rocket"))
        assert sql.index("features_check") < sql.index("tabfm_classify")
