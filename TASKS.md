# Open tasks — handoff at 2026-08-12 11:50 (machine restart)

Written mid-task because the box ran out of memory. Everything below is verified state, not plans.

## Safe: committed and pushed

`main` is in sync with `origin/main`. Nothing is uncommitted.

- `78d630a` — descriptor + pure-SQL scope docs (recovered from a hung session, see below)
- `b3429fd` — hoisted the reference-length flatten out of the per-row loop in
  `src/rocket_extension.cpp`. Build clean (9/9), 114 python tests, 25 sqllogictest assertions
- `7944d44` — `.gitattributes` now pins the working tree to LF, not just the index. 30 files had
  drifted to CRLF; `build_extension.bat` had drifted the *other* way and was violating its own
  `eol=crlf` rule. All verified with `git check-attr` and a byte-level check

## BLOCKER: Phase 5's last two datasets cannot run on this machine

Phase 5 needs 10 datasets (`UCR_SUBSET` in `duckdb_rocket/datasets.py`). Eight are done and
archived in `reference/phase5_*.json`. The two remaining are **ItalyPowerDemand** and **ECG5000**.

**ItalyPowerDemand was attempted locally and consumed 25.7 GB before being killed.** It died in
step `[3/3] running the whole pipeline in DuckDB` — the DuckDB child process
(`build/release/duckdb.exe -f data/phase5/ItalyPowerDemand/pipeline.sql`), not Python. No
`predictions.json` was produced. Both datasets already have `pipeline.sql` and `raw.parquet`
staged on disk from an earlier session; only execution is missing.

Why this is a wall and not bad luck:

| | test rows | in-context train rows | result |
|---|---|---|---|
| Largest completed (SyntheticControl) | 300 | 300 | 674s, fine |
| ItalyPowerDemand | 1029 | 67 | **25.7 GB, killed** |
| ECG5000 | 4500 | 500 | not attempted |

Every completed dataset is ≤300 test rows. ItalyPowerDemand is the first past that line and it
blew up 3.4× beyond the largest one that worked. ECG5000 is 4.4× ItalyPowerDemand's rows again,
with a 500-row context. **Do not attempt ECG5000 locally.**

The memory goes to the 40 per-group feature tables: 4500 rows x 500 features x 40 groups is
~2.9 GB of doubles before any join or intermediate. The generated SQL for ItalyPowerDemand alone
was 1,133,701 characters.

### Next step — pick one

1. **Run both on a pod.** PLAN.md line 496 already specifies a pod for this
   (`Run the 10-dataset subset on a pod`), and the standing risk table says reported numbers come
   from pods, not the 3060. This is the path PLAN.md already chose. Needs `TABPFN_TOKEN` injected
   (see `token.txt`, gitignored) and a `check` before and after — a forgotten pod bills silently.
2. **Make the pipeline stream groups instead of materializing all 40**, then retry locally. This
   is real work in `scripts/phase5_pipeline.py`, and it changes what is being measured.

Option 1 is what the plan says. Option 2 is only worth it if the pod lane stays blocked.

## Not done: PLAN.md is stale

`PLAN.md` lines 493-499 has all five Phase 5 boxes unchecked, but eight datasets have run:

| Dataset | Accuracy | Seconds | Test rows |
|---|---|---|---|
| BasicMotions (multivariate) | 1.0000 | 131.5 | 40 |
| Beef | 0.7667 | 129.0 | 30 |
| Coffee | 1.0000 | 128.0 | 28 |
| FaceFour | 0.9773 | 171.9 | 88 |
| GunPoint | 0.9933 | 258.0 | 150 |
| OSULeaf | 0.9711 | 448.5 | 242 |
| SyntheticControl | 0.9867 | 674.2 | 300 |
| Trace | 1.0000 | 260.0 | 100 |

Zero row-alignment failures anywhere. GunPoint is the only one with a Phase 3 comparison:
delta 0.0 and **identical per-row predictions**, which is the end-to-end statement that the C++
transform is interchangeable with the Python oracle.

All eight carry `"caveat": "local Windows timing on a contended box"` — accuracy is valid,
timings are not reportable.

This reconciliation was the committed next step and is **not** done. It was deliberately held
until the last two datasets landed, so the checkboxes could be ticked truthfully in one pass
rather than half-ticked.

## Also open, lower priority

- `description.yml:20` still has `ref: TODO-pin-a-commit-before-submitting`. The community-
  extensions repo builds from that exact revision, so it must be a real commit before submission.
  The file's own header also gates submission on Phase 5 breadth, which is not yet complete.

## Housekeeping done during the restart

- Killed PID 19124, a Claude session hung since 10:37:52 mid-`Edit`. Its work was recovered and
  is in `78d630a` + `b3429fd`; nothing was lost.
- Killed the runaway `duckdb.exe` (PID 29720, 25.7 GB). Free memory went 8.4 GB -> 30.2 GB.
- No duckdb-rocket process is running as of this writing. Three unrelated python jobs were left
  alone: `serve.py`, `f345_load.py`, `evaluate_bird.py`.
