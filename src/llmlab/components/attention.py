"""Attention mechanisms, built up lesson by lesson.

Lesson 01: scaled dot-product attention — the single equation at the heart of
every transformer since 2017:

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V        (Vaswani et al. 2017, eq. 1)

The intuition is a *soft dictionary lookup*. A Python dict maps a key to exactly
one value. Attention relaxes this: a **query** is compared against *every* **key**,
the comparison scores are turned into weights that sum to 1 (softmax), and the
result is the weighted average of *all* the **values**. "Look everything up a
little bit, in proportion to relevance." Because the lookup is a weighted average
instead of a hard choice, it is differentiable — so the model can *learn* what
to look for (queries), what to advertise (keys), and what to hand over (values).

In self-attention Q, K, V are all derived from the same token sequence (each
token asks a question about the others); in cross-attention Q comes from one
sequence and K, V from another (decoder tokens querying the encoder). This one
function serves both — only the caller changes.
"""

import math

import torch
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,   # (..., seq_q, d_k)  one question vector per position
    key: Tensor,     # (..., seq_k, d_k)  one index vector per position
    value: Tensor,   # (..., seq_k, d_v)  one payload vector per position
    mask: Tensor | None = None,  # (..., seq_q, seq_k) bool; True = "may attend"
) -> tuple[Tensor, Tensor]:
    """Return (output, attention_weights).

    output:  (..., seq_q, d_v) — for each query position, a weighted average of
             the value vectors.
    weights: (..., seq_q, seq_k) — the averaging weights; each row sums to 1.
             Returned for teaching/visualization; production code usually
             doesn't materialize them (see FlashAttention, Phase 5).

    Leading `...` dims (batch, heads, ...) are broadcast, because every step
    below is either a batched matmul or elementwise.
    """
    d_k = query.shape[-1]

    # Step 1 — raw similarity scores: every query dotted with every key.
    # (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k).
    # A dot product is large when two vectors point the same way — the model
    # learns to give a token's query and another token's key similar directions
    # exactly when the first should attend to the second.
    scores = query @ key.transpose(-2, -1)

    # Step 2 — the "scaled" part: divide by √d_k.
    # Why: if q and k have roughly unit-variance, uncorrelated components, then
    # q·k = Σᵢ qᵢkᵢ is a sum of d_k such terms, so Var(q·k) ≈ d_k — scores grow
    # like √d_k in magnitude just because the vectors got *longer*, not more
    # similar. Softmax of large-magnitude inputs saturates: one weight ≈ 1, the
    # rest ≈ 0, and the gradient through softmax ≈ 0 — learning stalls at
    # exactly the moment we choose a respectable d_k. Dividing by √d_k brings
    # the variance back to ≈ 1 regardless of d_k. (Verified in the tests.)
    scores = scores / math.sqrt(d_k)

    # Step 3 — masking (optional): where mask is False, overwrite the score
    # with -inf *before* softmax, so the weight there becomes exp(-inf) = 0 and
    # the remaining weights still sum to 1. Setting weights to 0 *after*
    # softmax instead would break the sum-to-1 property. Uses: causal masking
    # ("don't look at the future", lesson 05) and padding masking.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 4 — softmax over the *key* axis (the last one): each query position
    # gets a probability distribution over which positions to read from.
    # Softmax over the wrong axis is the classic silent bug — weights would
    # sum to 1 over queries instead, which means nothing.
    weights = torch.softmax(scores, dim=-1)  # (..., seq_q, seq_k)

    # Step 5 — the lookup itself: weighted average of value vectors.
    # (..., seq_q, seq_k) @ (..., seq_k, d_v) -> (..., seq_q, d_v).
    output = weights @ value

    return output, weights
