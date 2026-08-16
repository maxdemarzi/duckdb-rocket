# Routing: run the cheap model, escalate what it doesn't know

**Status: measured on 28 hard UCR datasets, 2026-08-15. `scripts/distill_gate.py --route`,
`reference/distill_route.json`.**

---

# Part 1 — What it is, and when to use it

## The situation this is for

You have two ways to classify a time series in DuckDB:

| | what it is | accuracy on hard problems | cost per row |
|---|---|---|---|
| **student** | ROCKET features + a ridge classifier, or MultiRocketHydra | baseline | 23-62 ms |
| **teacher** | ROCKET features + `tabfm_classify` (this pipeline) | +2.9 points | **55-69x that** |

Measured with both models on one idle 16-core machine in the same run: on a 64-row Herring batch the
student answered in 23.3 ms/row and the teacher in 1,288 ms/row; on a 128-row ScreenType batch, 30.3
against 2,093. An earlier version of this page said 14x, from two totals divided by their row counts
on *different* hardware.

The gap only exists where the problem is hard. On easy datasets a ridge classifier already scores
0.95+ and the teacher ties it — running the expensive model there buys nothing at all. That is
measured too: over 67 datasets the teacher and the student are level (+0.0085, p=0.69), and the
whole advantage lives in the 29 datasets where a label-only student is still under 0.90.

## What routing does

Run the student on every row. Ask it how sure it was. Send the least sure rows to the teacher, and
keep the student's answer for the rest.

```
    every row ──▶ student ──▶ confident?  ──yes──▶ keep the student's label      (80% of rows, cheap)
                                  │
                                  └────no───────▶ ask the teacher                (20% of rows, dear)
```

**Do not read "20% of the rows" as "20% of the cost".** The teacher charges a large flat fee per
call plus a small amount per row, so escalating a fifth of a batch costs far more than a fifth. How
much more is the first thing in "What it costs" below, and it is the single most important number
on this page.

You choose the escalation fraction. It is a budget dial, not a tuned hyperparameter.

## What you get

At a 20% escalation budget, over the 28 datasets where the teacher has any advantage at all:

| | vs the student alone | vs the teacher on everything |
|---|---|---|
| **ROCKET+ridge student** | **+0.0200** (20/28, p=0.004) | −0.0116 (6/28, p=0.023) |
| **MultiRocketHydra student** | **+0.0145** (19/28, p=0.015) | −0.0060 (11/28, p=0.84) |

Read that honestly: **routing buys most of the teacher's advantage for about a quarter of the extra
spend, and it does not match the teacher.** Against ridge it is measurably behind the teacher; against
MultiRocketHydra it is level. It is a cost trade.

The dial behaves as you would expect, so you can pick a point on it:

| escalate | ridge gain | mr-hydra gain |
|---|---|---|
| 0% | — | — |
| 10% | +0.0095 | +0.0083 |
| 20% | +0.0200 | +0.0145 |
| 30% | +0.0223 | +0.0201 |
| 50% | +0.0283 | +0.0230 |
| 100% (= the teacher) | +0.0316 | +0.0205 |

*(This table used to carry a "cost vs student" column reading 2.3x / 3.6x / 4.9x / 7.5x / 14x. It
was wrong — it assumed the teacher's cost is proportional to the rows you send it. Measured costs
are in the next section.)*

So a 20% budget captures **63%** of the teacher's advantage for ridge and **71%** for
MultiRocketHydra, at 26% of the extra spend. Note the curve is concave: the first 10% of escalated
rows buys a third of the total gain, and the last 50% buys almost none. For MultiRocketHydra the
50% point is *above* the 100% point (+0.0230 against +0.0205) — past a certain budget you start
handing the teacher rows the student was getting right.

## What it costs

Measured with all three arms on one idle 16-core machine in the same run
(`route_serve.py serve --compare`), which is the only arrangement where a cost ratio means anything:

| | Herring (64 rows) | ScreenType (128 rows) |
|---|---|---|
| student, whole batch | 1.5 s | 3.9 s |
| routed, escalating ~20% | 64.7 s | 227.5 s |
| teacher on every row | 82.4 s | 267.9 s |
| **rows escalated** | 22% | 17% |
| **cost of escalating them** | **77%** | **83%** |

Escalating a fifth of the rows costs four fifths of running the teacher on everything. The reason is
the teacher's contract, not this pipeline: it has no trained weights for your task, so **every call
re-encodes your whole training set** before it looks at a single query row. Fitted from the two
batch sizes each run produces:

    Herring      1.40 s per group + 9.0 ms per query row   ->  55.9 s you pay regardless
    ScreenType   5.05 s per group + 9.7 ms per query row   -> 202.1 s you pay regardless

That fixed part is 71% and 80% of the respective full-batch costs, and escalating fewer rows does
not touch it. **What routing saves is calls, not rows.** The saving shows up when escalation drops
the number of `--test-chunk`-sized calls: across the 28 datasets, escalating 20% costs a median 46%
of teacher-everywhere — 77% on test sets under 128 rows, 25% on those over 385.

Three things follow, in the order they are worth doing:

1. **Run the teacher at G=10.** Cost is exactly linear in the group count, and 10 costs −0.0033
   against 40. That is a straight 4x off the expensive path.
2. **Batch escalations.** The fixed fee is per call. One row at a time pays all of it every time.
3. **Do not shrink the training context to save money** — 1.48x for −0.0168, a much worse trade
   than (1). Shrink it only if a run does not fit in memory otherwise.

With (1) and (2), the routed ScreenType batch above goes from 227.5 s to roughly 60 s.

## When it will not help you

* **Your problem is easy.** If a ridge classifier already gets 0.95, there is nothing to escalate to.
  28 of the 67 datasets measured are in this state and routing does nothing on them, because the
  teacher does nothing on them.
* **You cannot run the teacher at inference time.** Routing needs both models live. If the point of
  the exercise was to *stop* needing the expensive model, routing does not do that — and neither
  does distillation, which was tried first and failed (see below).
* **Your student has no usable confidence.** See Part 2; this is a real obstacle and one of the two
  students here needed work to get around it.

## Running it

```bash
uv run python scripts/distill_gate.py --route \
    --from-gate reference/distill_gate.json --max-student 0.90 \
    --jobs 7 --out reference/distill_route.json
```

Student predictions and margins are cached per `(dataset, learner, seed)` under
`data/route_students/`, so re-running the analysis with different escalation fractions costs
nothing already computed. The first run is ~50 minutes on 8 cores for 28 datasets x 2 students.

---

# Part 2 — How it works, and why

## The confidence signal

The rule needs one number per row: how close the student came to changing its mind.

```python
def decision_margin(d):
    if d.ndim == 1:          # binary: RidgeClassifierCV returns signed distance
        return np.abs(d)
    s = np.sort(d, axis=1)   # multiclass: one column per class
    return s[:, -1] - s[:, -2]
```

**The margin, not the top score.** A row scoring 9.0 against 8.9 is a coin flip; a row scoring 2.0
against 0.1 is not. Ranking by the winning score alone would escalate the second and keep the first,
which is backwards.

**`predict_proba` is not an option for MultiRocketHydra.** aeon's classifier wraps a
`RidgeClassifierCV`, and the base class's `predict_proba` falls back to one-hot for a hard-decision
backbone — every row comes back at probability 1.0, which is exactly no information. The margin has
to be recovered from the internal estimator, which means reproducing the private transform pipeline:

```python
m = MultiRocketHydraClassifier(random_state=seed).fit(a, ytr)
xt = np.concatenate((m._scale_hydra.transform(m._transform_hydra.transform(b)),
                     m._scale_multirocket.transform(m._transform_multirocket.transform(b))), axis=1)
pred, direct = m.classifier.predict(xt), m.predict(b)
if not np.array_equal(pred, direct):
    raise RuntimeError("the reconstructed pipeline disagrees with predict(); its internals have "
                       "changed and the margins would be meaningless")
return direct, decision_margin(m.classifier.decision_function(xt))
```

That check is not decoration. A wrong reconstruction — a missed scaler, the two blocks concatenated
in the wrong order — returns perfectly plausible margins and routes the wrong rows, and the result
would be a routing curve that simply looks weak. Comparing against `predict()` on every fit turns a
silent wrong answer into a loud failure.

## The rule

```python
order = np.argsort(sconf, kind="stable")     # least confident first
pred = student_predictions.copy()
k = int(round(f * len(y)))
pred[order[:k]] = teacher_predictions[order[:k]]
```

Two details that are easy to get wrong:

* **`dtype=object`, not a fixed-width string array.** Assigning `"long_label"` into a `<U1` array
  truncates it silently, and the row then counts as wrong for a reason that has nothing to do with
  either model.
* **The fraction is fixed in advance.** Choosing it per dataset from the scored result is an oracle.
  The output does report that oracle as `best`, because it bounds what a tuned rule could reach, but
  it is labelled and must not be quoted as an achievable number. It was quoted that way once in this
  project's own results file before being corrected.

## The control, without which the result is unreadable

Escalating *any* rows to a teacher that is better on average buys something. So a curve that rises
with the escalated fraction is not evidence that the student knows what it does not know. The
control is escalating a **random** fraction of the same size, averaged over 20 draws:

| escalate | routed | random rows | **the signal** | p |
|---|---|---|---|---|
| 10% | +0.0095 | +0.0033 | **+0.0061** | 0.013 |
| 20% | +0.0200 | +0.0065 | **+0.0135** | 0.013 |
| 30% | +0.0223 | +0.0084 | +0.0139 | 0.013 |
| 50% | +0.0283 | +0.0154 | +0.0129 | 0.036 |

(ROCKET+ridge; MultiRocketHydra is the same shape, signal +0.0059 to +0.0146, p ≤ 0.036.)

**The student's own uncertainty picks rows about three times better than chance**, significantly, at
every budget and for both students. That difference is the actual claim; the first column alone
would have supported a much weaker one.

## Why routing works when distillation does not

This was not the first approach tried. Distillation — have the teacher label an unlabelled pool,
train the student on those labels, ship the student alone — was measured first on the same 28
datasets and failed:

| | gain | share of what was available |
|---|---|---|
| pool with **true** labels (the ceiling) | +0.0474 (p=0.0001) | — |
| teacher's full distribution (soft targets) | +0.0119 (p=0.13) | 25% |
| teacher's argmax, most confident half | +0.0064 (p=0.19) | 14% |
| teacher's argmax, whole pool | +0.0011 (p=1.00) | 2% |

The reason is not that the teacher is inaccurate. Corrupting the pool's *true* labels at known rates
shows it tolerates a median **25.6%** label error, and the teacher's median error is **21.6%** — inside
the budget. Yet replacing the teacher's labels with *random* labels of the same error rate is worth
**+0.0516** (10/12, p=0.039).

**A teacher's mistakes are not noise.** They concentrate on the same ambiguous rows and point the
same way, so the student learns a coherent wrong rule; random errors of the same size cancel. The
same property closes the obvious escape route — an ensemble of unrelated labellers. TabICL and
TabPFN-v2 are wrong together 3.87x as often as independence would give, and a soft-vote ensemble of
them gains +0.0001.

Routing is the one approach that never asks a wrong-but-confident prediction to be trusted:

| | what it requires | what breaks it |
|---|---|---|
| distillation | the teacher's labels are **right** | ~22% of them are not, systematically |
| ensembling | the labellers fail in **different places** | they fail in the same places (3.87x) |
| **routing** | the teacher is right where the **student is unsure** | nothing measured so far |

A teacher error costs one row instead of biasing every coefficient in a fit. That is the whole
difference, and it is why the same models that cannot teach can still help.

## Serving it: what actually gets deployed

Nothing about routing changes training. You fit the student on your labelled data exactly as before;
the teacher is not involved. Routing is entirely an inference-time decision.

```
row ──▶ student.predict + margin ──▶ margin >= threshold ──▶ student's label      (~80%, 1x)
                                          │
                                          └── below ──▶ teacher on this row ──▶ teacher's label
```

**The teacher's answers go to the caller and nowhere else.** They are not fed back into the student.
That is distillation, it was measured on these datasets and returned +0.0011, and routing makes it
*worse* rather than better: the rows routing selects are the ones the student found hardest, which
are the same rows the teacher is least accurate on and most systematically wrong about. A training
set built from escalated rows is enriched for exactly the correlated errors that closed distillation.

Three things about deploying the teacher that the accuracy tables do not show:

* **It is an in-context model, so it has no trained weights for your task.** Every call passes your
  labelled training rows as context — literally `tabfm_classify('train_cur', 'y', test := 'test_cur',
  ...)`. Deploying the teacher means deploying the training data with it, and every escalated call
  re-processes that context.
* **Batch the escalated rows.** This follows from the fixed fee and it is the difference between a
  usable system and an unusable one. The context pass is paid once per call, so escalating one row
  at a time pays the whole of it for a single prediction. Accumulate escalated rows and send them
  together, up to whatever `--test-chunk` your memory allows — chunking finer costs 2.18x on
  measured whole-dataset runs, because each chunk re-sends the context.
* **Run the teacher at 10 groups, not 40.** The default of 40 was inherited from the paper's
  configuration and never chosen for cost. Cost is exactly linear in it, and over 24 datasets G=20
  is free (+0.0002 routed) and G=10 costs −0.0033 routed, which no test detects. That is a 4x cut
  on the expensive path and the largest speedup available here.
* **The threshold is one number and it can drift.** It is calibrated against a margin distribution;
  if your inputs move, the realised escalation rate moves with them. The realised rate is worth
  monitoring in production for exactly that reason, since it is observable without labels.

### Calibrating the threshold: it keeps the accuracy, not the budget

`scripts/distill_gate.py --calibrate`, 28 datasets, threshold taken as a quantile of 5-fold
out-of-fold margins on the train split and then applied row by row:

| target | realised (ridge) | spread | gain over the student | vs sorting at the **same realised rate** |
|---|---|---|---|---|
| 10% | 12.6% | 4.1–25.7% | +0.0142 (p=0.015) | −0.0002 |
| 20% | 24.4% | 12.0–48.7% | +0.0219 (p=0.009) | +0.0008 |
| 30% | 35.1% | 15.0–66.9% | +0.0241 (p=0.009) | −0.0003 |

(MultiRocketHydra is the same shape: 13.2%, 25.2%, 34.7% realised, gains +0.0129 to +0.0218.)

**The good news is the last column.** Against sorting the batch *at the rate it actually spent*, a
threshold is worth +0.0008 to −0.0004 and nothing is significant. Not having the batch costs nothing
in accuracy, which is the question this was built to answer.

**The bad news is the second column.** It overspends its budget by 20–25% relative, every time, and
the per-dataset spread is wide enough that a 20% target can cost 49%. The cause is a distribution
difference rather than a bug: test margins are systematically smaller than out-of-fold train margins,
so the model is *less* confident on test rows than on held-out training rows and more of them fall
under any threshold. UCR's train/test splits are predefined and not always exchangeable.

Two consequences for a deployment. **Set the target below the budget you want** — ask for 16% to
spend 20% — knowing the multiplier is dataset-specific and cannot be known in advance for a new one.
And **close the loop on the realised rate**, which is observable without labels: measure what
fraction actually escalates and move the threshold until it matches. That is the only version of
this that survives a distribution shift.

Note what this does *not* say. Comparing the threshold at a 20% target against sorting at 20% makes
the threshold look better by +0.0018 — but only because it spent 24.4%. That comparison was made
here first and it was wrong; the same artifact made a two-dataset smoke test read +0.0160.

## What is not established

* **28 datasets, one seed, one split.** The escalation fractions are fixed rather than tuned, which
  is the honest design, but the per-dataset spread is wide. At a 20% budget, across both students:
  ACSF1 +0.0700 and Lightning2 +0.0656 (MultiRocketHydra), SemgHandMovementCh2 +0.0667 and Worms
  +0.0649 (ridge); against Herring −0.0781 (MultiRocketHydra) and Lightning7 −0.0548 (both). The
  mean is a mean, and eight of 28 datasets go the wrong way for each student.
* **The cost model is fitted on two datasets.** Herring and ScreenType each give two batch sizes,
  which is exactly enough to solve for a fixed and a marginal term and no more. A third,
  SemgHandMovementCh2, failed its full-batch arm twice and is unexplained. The *shape* is
  corroborated across 51 archived runs and the marginal term agrees to within 0.7 ms three ways, but
  the fixed term is two points per dataset.
* **No calibration work has been done.** The margin is used as a raw ranking within a single model,
  which is all the rule needs. Comparing margins *across* models would need calibration, and the one
  place that was tried — picking the more confident of two labellers per row — reached only 6% of
  the available complementary information.
* **`scripts/route_serve.py` runs the whole thing end to end, and is a demonstration rather than a
  server.** It deploys the artifacts, serves a batch through the real extension with a calibrated
  threshold, and reports the split — verified on Herring: 14 of 64 rows escalated (21.9% against a
  20% target), routed 0.6406 against the student's 0.6250, both matching the offline analysis. What
  it is not is a service: no threshold control loop, no batching of escalations across requests, and
  no attempt at concurrency.
* **The context cannot be cached today, and that is where the cost is.** 71-80% of a teacher call
  is re-encoding a training context that did not change since the last call. Three things could be
  cached: the model weights are (`tabfm_load`), the encoded context is not and cannot be from SQL —
  `tabfm_classify(train, y, test := ...)` takes both halves in one call, so there is no
  prepare-then-query split to cache between — and our own call pattern makes it worse by calling
  once per group per chunk. Exposing a reusable context is an upstream change to the exported graph;
  profiling would confirm where the time goes but has no bug to find, since the cost scales cleanly
  with training rows.
* **Sending a smaller context is not the workaround.** Measured over six datasets: half the context
  runs 1.48x faster, not 2x, because the per-query-row term does not shrink — and it costs −0.0168
  accuracy, against −0.0033 for the 4x that cutting groups buys. The damage tracks examples per
  class: MedicalImages at 9.5 per class lost 0.18. It is worth doing only to make an otherwise
  impossible run possible, which it does — two datasets that exceed a CPU pod's 29.8 GiB at full
  context complete at half.

## Where the numbers live

| file | what it holds |
|---|---|
| `reference/distill_route.json` | per dataset and learner: the full 0–100% curve, the random-rows control, the oracle `best` |
| `reference/distill_armb.json` | the distillation arms that failed |
| `reference/distill_breakeven.json` | the label-noise tolerance sweep and the structured-vs-random comparison |
| `reference/distill_gate.json` | the 67-dataset gate that decided which datasets any of this runs on |
| `data/route_students/` | cached student predictions and margins, per `(dataset, learner, seed)` |
