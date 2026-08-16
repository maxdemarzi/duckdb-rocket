"""Prototype: compute TabICL's support-derived state once, reuse it across query batches.

The claim being tested is not that caching is a good idea -- it is that the split is EXACT, i.e.
that a two-pass implementation reproduces the single-pass logits bit-for-bit (to float noise), so
adopting it costs no accuracy.

Where the split goes, from TabICL's own source:

    InducedSelfAttentionBlock.induced_attention:
        hidden = multihead_attn1(ind_vectors, src[:train_size], src[:train_size])   support only
        out    = multihead_attn2(src, hidden, hidden)                               per row

`hidden` depends only on the support rows, and every row's output is an independent attention
against it. That holds per block and stacks, because block k's `hidden` is derived from block k-1's
support outputs. So the cacheable state is one `hidden` per ISAB block, plus the support rows'
representations that the ICL stage attends to.

What the prepare pass computes once:      what a query batch then costs:
  * hidden per column-ISAB block            * col-embed the query rows only
  * support reps through row_interactor     * row-interact the query rows only
                                            * ICL against the cached support reps

The support rows never go through attn2 or the row interactor again, which is where 90% of the
time is (col_embedder 48%, row_interactor 42% at S=500/Q=128).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tabicl import TabICLClassifier
from tabicl._model.encoders import InducedSelfAttentionBlock as ISAB

# ---------------------------------------------------------------------------------------------
# A recording/replaying `induced_attention`. Records `hidden` on the prepare pass, replays it on
# the query pass. Deliberately a monkeypatch on the installed package rather than an edit: this is
# a feasibility probe, and the same discipline anofox-tabfm's export tooling uses.
# ---------------------------------------------------------------------------------------------
_ORIGINAL = ISAB.induced_attention
_ORIGINAL_FWD = ISAB.forward
_MODE: dict = {"mode": "off", "store": [], "i": 0}


def _patched(self, src, train_size=None):
    mode = _MODE["mode"]
    if mode == "off":
        return _ORIGINAL(self, src, train_size)

    *batch_shape, _, d_model = src.shape
    if mode == "record":
        ind = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)
        keys = src if train_size is None else src[..., :train_size, :]
        hidden = self.multihead_attn1(ind, keys, keys)
        _MODE["store"].append(hidden)
    else:  # replay
        hidden = _MODE["store"][_MODE["i"]]
        _MODE["i"] += 1
    return self.multihead_attn2(src, hidden, hidden)


def _patched_forward(self, src, train_size=None):
    """In record/replay, bypass the skip-mask and attend over every column group.

    **The skip decision must not be recomputed per pass.** `ISAB.forward` skips a whole column
    group when all of its values equal `skip_value`, and boolean-indexes `src[~skip_mask]`, which
    flattens the batch dims -- so a skipped group changes the tensor SHAPE the cached `hidden` has
    to match. The reserved cls-token columns are padded with exactly `skip_value`, and in the
    combined pass they survive only because the target-aware step adds `y_emb` to the support rows
    and perturbs them. A query-only batch adds no `y_emb`, so those 4 columns look skippable and
    104 groups become 100, against a cache recorded at 104.

    So the split has to inherit the combined pass's decision rather than re-derive it. Here that is
    "never skip", which is what the combined pass does for every real dataset; a production
    implementation would carry the mask across the boundary with the rest of the state.
    """
    if _MODE["mode"] == "off":
        return _ORIGINAL_FWD(self, src, train_size)
    return self.induced_attention(src, train_size)


ISAB.induced_attention = _patched
ISAB.forward = _patched_forward


def prepare(m, xs, y):
    """One pass over the support set. Returns everything a query batch needs."""
    _MODE.update(mode="record", store=[], i=0)
    emb = m.col_embedder(xs, y_train=y, d=None, embed_with_test=False)
    hidden = list(_MODE["store"])
    _MODE.update(mode="off")
    reps = m.row_interactor(emb, d=None)
    return {"hidden": hidden, "support_reps": reps}


def col_embed_query(ce, xq):
    """`ColEmbedding` for query rows only, with the inducing points replayed from the cache.

    Written out rather than routed through `col_embedder(...)` because that path derives
    train_size from y_train.shape[1] and calls `y_train.max()`, which has no meaning for a batch
    that contributes no labels. The steps below are `_train_forward_with_feature_group` and
    `_compute_embeddings` with the target-aware addition dropped -- correct here precisely because
    it only ever applies to `src[..., :train_size, :]`, and a query batch has train_size 0.
    """
    x = ce.feature_grouping(xq)
    if ce.reserve_cls_tokens > 0:
        x = torch.nn.functional.pad(x, (0, 0, ce.reserve_cls_tokens, 0), value=-100.0)
    features = x.transpose(1, 2)
    src = ce.in_linear(features)
    src = ce.tf_col(src, train_size=None)  # replayed: the cache supplies `hidden`
    if ce.affine:
        embeddings = features * ce.ln_w(ce.out_w(src)) + ce.ln_b(ce.out_b(src))
    else:
        embeddings = src
    return embeddings.transpose(1, 2)


def query(m, ctx, xq, y):
    """A query batch, with the support half never recomputed."""
    _MODE.update(mode="replay", store=ctx["hidden"], i=0)
    emb = col_embed_query(m.col_embedder, xq)
    _MODE.update(mode="off")
    reps = m.row_interactor(emb, d=None)
    joined = torch.cat([ctx["support_reps"], reps], dim=1)
    return m.icl_predictor(joined, y, softmax_temperature=0.9, return_logits=True)


def main() -> int:
    rng = np.random.default_rng(0)
    S, Q, H = 500, 128, 100
    xtr = rng.normal(size=(S, H))
    ytr = (xtr[:, 0] + 0.3 * xtr[:, 2] > 0).astype(int)
    xq = rng.normal(size=(Q, H))

    clf = TabICLClassifier(random_state=0, device="cpu").fit(xtr, ytr)
    m = clf.model_
    m.train()  # the path anofox-tabfm exports
    torch.set_num_threads(4)

    xs = torch.tensor(xtr, dtype=torch.float32)[None]
    xq_t = torch.tensor(xq, dtype=torch.float32)[None]
    y = torch.tensor(ytr, dtype=torch.float32)[None]

    with torch.no_grad():
        t0 = time.perf_counter()
        base = m(torch.cat([xs, xq_t], 1), y_train=y, d=None, embed_with_test=False)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        ctx = prepare(m, xs, y)
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        got = query(m, ctx, xq_t, y)
        t_query = time.perf_counter() - t0

    diff = (base - got).abs().max().item()
    print(f"  support {S}, query {Q}, features {H}")
    print(f"  single pass                 {t_base:6.3f} s")
    print(f"  prepare (once per support)  {t_prep:6.3f} s")
    print(f"  query   (per batch)         {t_query:6.3f} s   -> {t_base / t_query:.2f}x per batch")
    print(f"  max |logit difference|      {diff:.3e}")
    print(f"\n  => {'EXACT: the split reproduces the single pass' if diff < 1e-4 else 'MISMATCH'}")
    return 0 if diff < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
