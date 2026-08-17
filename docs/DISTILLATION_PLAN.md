# The one-two punch: label with the teacher, serve with a student

**Status: the gate opens, but only on hard datasets. 67 archived teachers, 2026-08-14.**

## The gate, decided (2026-08-14)

`scripts/distill_gate.py --gate`, third design: teacher accuracy `T` read from its archived pipeline
report, each student trained on the train split and scored on the **same full test split**, reported
**per learner** and never as a max over learners. 67 datasets, `reference/distill_gate.json`.

| subgroup | n | vs ROCKET+ridge | vs mr-hydra |
|---|---|---|---|
| all archived teachers | 67 | 30/67, **+0.0085**, p=0.69 | 25/67, **+0.0019**, p=0.68 |
| best student < 0.90 | 29 | 21/29, **+0.0294**, p=0.0125 | 17/29, **+0.0198**, p=0.25 |
| best student < 0.75 | 11 | 11/11, **+0.0572**, p=0.0010 | 8/11, **+0.0326**, p=0.23 |
| best student ≥ 0.95 | 28 | 6/28, −0.0057, p=0.17 | 6/28, −0.0102, p=0.12 |

**Across the whole archive the teacher is level with both students.** Both effects are below the
0.0140 shift n=67 can detect at 80% power, and the median difference is exactly 0.0000 — because 28
of the 67 datasets have a student at 0.95 or better, where neither model has anywhere to go.

**The difficulty split is not post-hoc.** It is the cut this document argued for before any of these
numbers existed, in the section below: a dataset a label-only student already solves has no headroom
to distil into. On the datasets that do have headroom the teacher leads ROCKET+ridge by 3 points, and
on the hardest eleven it wins **11 of 11** at p=0.0010 — which survives correcting for the four
subgroups examined. Against `mr-hydra` the teacher is ahead everywhere too (+0.0198, +0.0326) but
never significantly: only eleven datasets in the whole archive are genuinely hard, so the question
stays underpowered at any archive size.

**This also retracts a six-dataset result of my own.** The same gate over the six hard datasets alone
read 6/6 and +0.0328 at p=0.0312, and was reported as the corrected verdict. At 67 datasets it is
30/67 and +0.0085. The six-dataset number was a *subgroup* effect stated as a general one — the third
time in this project that a six-dataset conclusion did not survive widening, after the feature
shortlist and the "ts beats rocket" claim.

**So distillation is not dead and it is not general.** Its case is confined to unsaturated datasets,
which is exactly where arm B was then run.

## Arm B, and why it fails (2026-08-15)

Run on those 28 datasets, one 50/50 pool/holdout split each, `rocket+ridge`. The pool has **five
points** of headroom with true labels (`C − A` +0.0474, 21/28, p=0.0001) — the premise this document
argued for finally holds, against under one point on the saturated subset that killed the first gate.

The teacher recovers almost none of it: **+0.0011 with hard argmax, 2.4% of the ceiling.** Soft
targets reach +0.0119 (25.1%) and confidence filtering +0.0064 (13.5%); `Bs − B` is +0.0107 at
p=0.0525 paired on identical splits. None of the three clears the 0.0163 this design can detect, so
the only firm statement is the ordering.

**The reason is not the teacher's error rate, and this is the part worth carrying forward.** The
break-even sweep corrupts the pool's true labels at known rates and finds where the gain crosses
zero: the median is **25.6%**, against a median teacher error of **21.6%**. On rate alone the teacher
is good enough on 9 of 13 measurable cases. Yet arm B pays nothing — and swapping the teacher's
labels for *random* labels of the same error rate is worth **+0.0516** (10/12, p=0.0386).

A teacher's mistakes are not noise. They concentrate on the same ambiguous rows and point the same
way, so the student learns a coherent wrong rule; random errors of the same size cancel. Every
prediction this document made about *how much* the teacher must know was aimed at the wrong quantity.

**What that implies for anything built next.** Improving a labeller's accuracy is not the lever —
the rate was never binding. The lever is either decorrelating the error structure (an ensemble of
architecturally unrelated models, judged on error *overlap* rather than accuracy) or not putting
pseudo-labels into a training set at all (routing: run the student, escalate only the rows it is
unsure of, so a teacher error costs one row instead of biasing every coefficient).

Full numbers and method in `reference/RESULTS.md`; raw output in `reference/distill_armb.json` and
`reference/distill_breakeven.json`.

## Why the gate was re-opened

The gate below fired and was honoured, and the measurement stands: on the ten-dataset subset there
is no headroom to distil. But the subset was chosen for *spread*, and nine of its ten datasets sit
at 0.94-1.00, where a ROCKET-family classifier has already extracted everything available. Testing
"does more data help" on saturated problems answers a question nobody asked.

The published UCR results say where the headroom actually is: of 112 datasets, **13 have a best
published accuracy below 0.75 and 7 below 0.60** — Phoneme 0.367, InlineSkate 0.544,
EOGVerticalSignal 0.558, Haptics 0.571, ScreenType 0.595, RefrigerationDevices 0.600. On several,
ROCKET *specifically* trails the field (ScreenType 0.467 vs 0.595; Herring 0.594 vs 0.734). That is
10-20 points of headroom against the <1 point that killed arm B here.

**So the correct reading is "not demonstrated", not "disproved".** The gate should be re-run on the
hard datasets before the student stack -- MultiRocketHydra on CPU, LITE/InceptionTimePlus with
KLDivLoss and augmentation on GPU -- is abandoned.

One hard constraint found while scoping that: **`tabicl-v2` caps at 10 classes** (`max_classes: 10`
in its export report), so Phoneme (39) and EOGVerticalSignal (12) — two of the hardest datasets in
the archive — are out of reach for the teacher entirely. Six of the hard set are reachable: Herring,
MiddlePhalanxTW, ScreenType, RefrigerationDevices, Haptics, InlineSkate.

The prerequisite question is sharper than the gate itself: on those datasets, does the pipeline beat
ROCKET+ridge? Both use the same features. If it does, the in-context model is extracting more from
them and distillation has something to inherit. If both are equally poor, the *features* are the
ceiling and no student can fix that.

## Result on the saturated subset: no headroom there

Arms A and C, `scripts/distill_gate.py`, stratified half/half split of each test set:

| dataset | learner | context | pool | A | C | **C − A** |
|---|---|---|---|---|---|---|
| ItalyPowerDemand | rocket+ridge | 67 | 514 | 0.9650 | 0.9709 | +0.0058 |
| ItalyPowerDemand | mr-hydra | 67 | 514 | 0.9670 | 0.9767 | +0.0097 |
| ECG5000 | rocket+ridge | 500 | 2250 | 0.9471 | 0.9498 | +0.0027 |
| ECG5000 | mr-hydra | 500 | 2250 | 0.9444 | 0.9507 | +0.0062 |
| OSULeaf | rocket+ridge | 200 | 121 | 0.9587 | 0.9669 | +0.0083 |
| OSULeaf | mr-hydra | 200 | 121 | 0.9835 | 0.9835 | +0.0000 |
| SyntheticControl (control) | both | 300 | 150 | 1.0000 | 1.0000 | +0.0000 |

`C − A` is the headroom **using real labels on the pool**. It never reaches one point, and on the
best-shaped candidate — ECG5000, 2,250 pool rows against a 500-row context — it is 0.0027 and
0.0062. Arm B can only ever recover a fraction of that, because pseudo-labels are strictly worse
than the real ones arm C was handed. There is nothing to distil.

The control behaved as predicted (0.0000 at a 0.5× ratio), which is the reason to believe the rest.

**Why, and what it does not say.** These datasets are small and near ceiling, and a ROCKET-family
classifier already extracts nearly everything available from a few hundred labelled examples —
`mr-hydra` reaches 0.9835 on OSULeaf from 200 rows, *better* than the teacher's 0.9711 on the same
data. Adding labels to a saturated problem cannot help, whoever produced them.

So this is not "distillation does not work". It is "UCR cannot demonstrate it". The idea still has
a plausible home: a genuinely hard task, a large unlabelled pool, and a teacher whose pretrained
knowledge exceeds what a linear model can learn from the labels on hand. None of those three hold
here, and the honest move is to say so rather than to build the student and report the number that
would have followed anyway.

**Cost of finding out: about seven minutes of local CPU.** The gate was designed so the two arms
that need no teacher run first, and that ordering is what stopped this from becoming a GPU
distillation pipeline that measured nothing.

---

## The original plan, unedited

## The measurement that motivates it

ROCKET + `RidgeClassifierCV` was run on the same ten datasets, same splits, same 10,000-kernel
bank, differing only in what consumes the features:

| | mean accuracy | wins | total time, 10 datasets |
|---|---|---|---|
| ROCKET + ridge | **0.9636** | 3 | **262 s** (features + fit + predict, Python extractor) |
| this pipeline | 0.9615 | 4 | ~3,741 s |
| ties | | 3 | |

**Accuracy is a coin flip. Cost is not — roughly 14×, and that is with our slow Python feature
extractor rather than the C++ one.** OSULeaf is the one real gap in the pipeline's favour
(+0.0290); Beef swings the other way (+0.0333) but sits inside its own 0.0509 seed noise.

So on any dataset where you already have labels, ridge is the better tool and this project's own
README says so. What ridge cannot do is **train without labels**. That is the entire opening.

## The idea

```
unlabelled pool ─▶ rocket_transform ─▶ tabfm_classify ─▶ soft P(y|x)   [teacher, in DuckDB]
                          │                                   │
                          │                                   ▼
                          └──────── same features ────▶ student fit ─▶ ms-latency serving
```

Use the expensive in-context model **once per example** to manufacture a labelled dataset, then
distil into something that classifies for nothing. It converts this pipeline's defining weakness —
every prediction pays full price, inference is 93.7% of wall clock — into a one-off cost.

*The context cache (anofox-tabfm#40) is not an alternative to this and does not overlap with it. It
reuses one encoded context across calls that share it, measured at 1.85x on the best shape in this
project's archive and a net loss on the most common one. Distillation removes the in-context model
from the serving path entirely, which is a different order of saving; the cache makes manufacturing
the labelled dataset somewhat cheaper, and that is all.*

Two things make this cheaper here than distillation usually is:

1. **Soft labels already exist.** The pipeline materialises
   `all_groups(grp, id, proba MAP(VARCHAR, DOUBLE))` and averages across G=40 kernel groups. That
   is an *ensemble* posterior, not one model's softmax, so it carries genuine uncertainty. Today
   the SQL argmaxes it away; emitting it is a projection change, not new machinery.
2. **The CPU student shares the feature family.** MultiRocket/Hydra is the same convolutional-
   features-plus-linear-classifier shape, so there is no second extraction pipeline to build,
   validate or keep in sync.

## Student lanes

**CPU — `MultiRocketHydraClassifier` (aeon), the default.** Milliseconds on commodity CPUs and
consistently better than vanilla ROCKET. Ridge does not take soft targets natively, but it does not
need to: fit ridge *regression* against the K-column probability matrix and argmax the predictions.
That is one line different from `RidgeClassifier`, which already regresses on ±1 indicator targets
internally.

**GPU — LITE or InceptionTimePlus (tsai), when the pool is large.** Train on the soft posterior
with `KLDivLoss` rather than cross-entropy on hard labels: it preserves the teacher's uncertainty
and stops the student from confidently memorising the teacher's mistakes. Pair with time-series
augmentation — jitter, magnitude scaling, time-warp, random crop — so the student learns invariant
temporal structure instead of the noise in the pseudo-labels.

**Iterative self-training** if the pool is big: label a fraction, fit, use the student's confident
predictions to expand, refit. Only worth attempting after the single-shot version has cleared the
gate below.

## The gate

The teacher is not free of labels — it needs `n_train` rows as in-context examples. So the honest
question is not "does the student work" but:

> Does a student trained on teacher-pseudo-labelled **unlabelled** data beat ridge trained on the
> `n_train` **real** labels you already had?

If it does not, there is nothing here: you would simply train on the real labels and skip the
teacher. Protocol, per dataset:

- **context** = the real train split (`n_train` rows, real labels)
- **pool** = first half of the test split, labels discarded
- **holdout** = second half of the test split, labels kept for scoring only

| arm | trained on | this is |
|---|---|---|
| **A** | ridge on `context` real labels | what you would do without the teacher |
| **B** | student on `pool` teacher soft labels | the proposal |
| **C** | ridge on `context` + `pool` **real** labels | the ceiling distillation is chasing |
| **T** | teacher scoring `holdout` directly | today's pipeline |

All four scored on `holdout`. **B > A is the entire claim.** B ≈ A kills it. C says how much of the
achievable gain was captured, and T says whether the student is worth having at all.

Best-shaped candidates, because pool/context ratio is what should drive the gain:

| dataset | context | pool | holdout | ratio |
|---|---|---|---|---|
| ECG5000 | 500 | 2250 | 2250 | 4.5× |
| ItalyPowerDemand | 67 | 514 | 515 | 7.7× |
| SyntheticControl | 300 | 150 | 150 | 0.5× — expect no gain, include as a control |

## Phases

1. **Emit soft labels.** Project the averaged `proba` map out of `phase5_pipeline.py` alongside
   `yhat`. Assert the rows sum to 1 and that argmax reproduces today's `yhat` exactly — this must
   change nothing about existing results.
2. **Arms A and C.** Pure sklearn/aeon, no teacher, minutes locally. If C ≈ A the dataset has no
   headroom and should be dropped before spending anything on B.
3. **Arm B, CPU lane.** `MultiRocketHydraClassifier` + soft-target ridge. Local.
4. **Gate.** B vs A across the candidates. Stop here if it fails, and record the negative result —
   it is worth as much as the positive one and costs nothing to write down.
5. **GPU lane**, only if the gate passes: LITE/InceptionTimePlus, KL loss, augmentation.
6. **Iterative self-training**, only if step 5 pays.

## What would make this fail, stated in advance

- **Teacher accuracy is the ceiling.** The student cannot exceed the teacher on the pool's
  distribution, and typically lands slightly under. If the teacher is 0.95, expect ≤ 0.95.
- **A is a strong baseline.** Ridge on a few hundred real labels is genuinely good, as the table at
  the top shows. The gap it leaves may simply be small.
- **Small pools.** Below roughly 2× context size there is little for the student to learn that the
  context did not already contain.
- **Cost has to be counted honestly.** Teacher labelling is the pipeline's full price — ECG5000's
  4,500 rows cost 18m39s on an A40. That is amortised only if you will classify many more rows
  later. For a one-off batch, just run the teacher and stop.
- **Calibration is assumed, not measured.** Averaging 40 group posteriors *should* calibrate well,
  but nobody has checked; if it is overconfident, KL against it inherits that.
