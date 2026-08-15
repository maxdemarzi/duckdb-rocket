# Routing: run the cheap model, escalate what it doesn't know

**Status: measured on 28 hard UCR datasets, 2026-08-15. `scripts/distill_gate.py --route`,
`reference/distill_route.json`.**

---

# Part 1 — What it is, and when to use it

## The situation this is for

You have two ways to classify a time series in DuckDB:

| | what it is | accuracy on hard problems | cost |
|---|---|---|---|
| **student** | ROCKET features + a ridge classifier, or MultiRocketHydra | baseline | 1x |
| **teacher** | ROCKET features + `tabfm_classify` (this pipeline) | +2.9 points | **~14x** |

The 14x is measured, not estimated: 262 seconds against 3,741 seconds over ten datasets, and that is
with a *slow Python* feature extractor on the student's side, so the real ratio favours the student
further.

The gap only exists where the problem is hard. On easy datasets a ridge classifier already scores
0.95+ and the teacher ties it — running the expensive model there buys nothing at all. That is
measured too: over 67 datasets the teacher and the student are level (+0.0085, p=0.69), and the
whole advantage lives in the 29 datasets where a label-only student is still under 0.90.

## What routing does

Run the student on every row. Ask it how sure it was. Send the least sure rows to the teacher, and
keep the student's answer for the rest.

```
    every row ──▶ student ──▶ confident?  ──yes──▶ keep the student's label      (80% of rows, 1x)
                                  │
                                  └────no───────▶ ask the teacher                (20% of rows, 14x)
```

You choose the escalation fraction. It is a budget dial, not a tuned hyperparameter.

## What you get

At a 20% escalation budget, over the 28 datasets where the teacher has any advantage at all:

| | vs the student alone | vs the teacher on everything | cost |
|---|---|---|---|
| **ROCKET+ridge student** | **+0.0200** (20/28, p=0.004) | −0.0116 (6/28, p=0.023) | 3.6x |
| **MultiRocketHydra student** | **+0.0145** (19/28, p=0.015) | −0.0060 (11/28, p=0.84) | 3.6x |

Read that honestly: **routing buys most of the teacher's advantage for about a quarter of the extra
spend, and it does not match the teacher.** Against ridge it is measurably behind the teacher; against
MultiRocketHydra it is level. It is a cost trade.

The dial behaves as you would expect, so you can pick a point on it:

| escalate | ridge gain | mr-hydra gain | cost vs student |
|---|---|---|---|
| 0% | — | — | 1.0x |
| 10% | +0.0095 | +0.0083 | 2.3x |
| 20% | +0.0200 | +0.0145 | 3.6x |
| 30% | +0.0223 | +0.0201 | 4.9x |
| 50% | +0.0283 | +0.0230 | 7.5x |
| 100% (= the teacher) | +0.0316 | +0.0205 | 14x |

So a 20% budget captures **63%** of the teacher's advantage for ridge and **71%** for
MultiRocketHydra, at 26% of the extra spend. Note the curve is concave: the first 10% of escalated
rows buys a third of the total gain, and the last 50% buys almost none. For MultiRocketHydra the
50% point is *above* the 100% point (+0.0230 against +0.0205) — past a certain budget you start
handing the teacher rows the student was getting right.

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

## What is not established

* **28 datasets, one seed, one split.** The escalation fractions are fixed rather than tuned, which
  is the honest design, but the per-dataset spread is wide. At a 20% budget, across both students:
  ACSF1 +0.0700 and Lightning2 +0.0656 (MultiRocketHydra), SemgHandMovementCh2 +0.0667 and Worms
  +0.0649 (ridge); against Herring −0.0781 (MultiRocketHydra) and Lightning7 −0.0548 (both). The
  mean is a mean, and eight of 28 datasets go the wrong way for each student.
* **The 14x cost ratio is from ten datasets** with a Python feature extractor on the student side.
  The direction is certain and the exact multiple is not.
* **No calibration work has been done.** The margin is used as a raw ranking within a single model,
  which is all the rule needs. Comparing margins *across* models would need calibration, and the one
  place that was tried — picking the more confident of two labellers per row — reached only 6% of
  the available complementary information.
* **Nothing here is wired into the extension.** This is an offline analysis over archived teacher
  predictions and cached student margins. A serving implementation would need the student's margin
  computed inline and a threshold rather than a fraction, since a live system does not have the
  whole test set to sort.

## Where the numbers live

| file | what it holds |
|---|---|
| `reference/distill_route.json` | per dataset and learner: the full 0–100% curve, the random-rows control, the oracle `best` |
| `reference/distill_armb.json` | the distillation arms that failed |
| `reference/distill_breakeven.json` | the label-noise tolerance sweep and the structured-vs-random comparison |
| `reference/distill_gate.json` | the 67-dataset gate that decided which datasets any of this runs on |
| `data/route_students/` | cached student predictions and margins, per `(dataset, learner, seed)` |
