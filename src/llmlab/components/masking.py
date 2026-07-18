"""Attention masks — lesson 05.

Lesson 01 built the mechanism (`masked_fill(~mask, -inf)` before softmax) and
left the `mask` argument dangling. This lesson builds the two masks every
transformer actually uses and plugs them in:

**Causal mask** (Vaswani et al. 2017, §3.2.3 — the decoder side). An
autoregressive model generates left-to-right, so at generation time position t
simply has no access to positions t+1... If training lets position t *attend*
to t+1, the model learns to "predict" the next token by reading it — perfect
training loss, useless generator. The causal mask makes training-time
information flow identical to generation-time information flow:

    position q may attend position k   ⟺   k ≤ q

The payoff is enormous: behind this mask, ONE forward pass over an n-token
sequence produces n honest next-token predictions *in parallel* (position 0
predicts token 1, position 1 predicts token 2, ...). An RNN gets the same n
training signals only sequentially. This mask is why decoder transformers
could train on internet-scale data — and it is the one piece of the 2017
decoder that GPT kept when it dropped everything else (Phase 2).

**Padding mask.** Batches are rectangular tensors; sentences have ragged
lengths. Short sequences get a dummy PAD token appended, and the mask stops
*keys* at pad positions from receiving attention weight — pad is storage, not
content. Unlike the causal mask (a fixed function of positions, same for every
sequence), the padding mask is per-example *data*, computed from the token ids.

Convention used throughout this repo (set in lesson 01): masks are **boolean,
True = "may attend"**, broadcastable to (batch, heads, seq_q, seq_k).
"""

import torch
from torch import Tensor


def causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Lower-triangular boolean mask: True where key position ≤ query position.

    Returns (seq_len, seq_len) — no batch/head dims on purpose: the causal rule
    is the same for every sequence and every head, so we build the smallest
    tensor that broadcasts against attention's (batch, heads, seq_q, seq_k).

        causal_mask(4) =  [[ T, F, F, F ],     row = query position q
                           [ T, T, F, F ],     col = key   position k
                           [ T, T, T, F ],     True ⟺ k ≤ q
                           [ T, T, T, T ]]

    Row q reads: "query q may attend keys 0..q" — itself and the past. The
    diagonal is True (a token may attend itself) — without that, row 0 would
    have NO permitted key, and softmax over an all-(−∞) row is NaN, not 0
    (pinned in the tests: e^−∞ terms give 0/0).

    Written as a comparison of position indices rather than `torch.tril` so
    the *rule* (k ≤ q) is the code, not a property the reader must recall
    about triangular matrices. Same result, one broadcast comparison:
    (seq, 1) ≥ (1, seq) → (seq, seq).
    """
    positions = torch.arange(seq_len, device=device)
    return positions.unsqueeze(1) >= positions.unsqueeze(0)  # k ≤ q, i.e. q ≥ k


def padding_mask(token_ids: Tensor, pad_id: int) -> Tensor:
    """Key-padding mask from a batch of token ids: True where the token is real.

    token_ids: (batch, seq_k) integer ids, padded with `pad_id`.
    returns:   (batch, 1, 1, seq_k) bool.

    The two singleton dims are deliberate broadcasting slots — the same mask
    applies to every head (dim 1) and every query position (dim 2); only the
    *key* axis varies. Against attention's (batch, heads, seq_q, seq_k) scores
    this expands for free, no copies.

    Why mask only keys, not queries: masking a key column removes pad tokens
    from everyone's weighted average — that's the correction we need. Masking a
    pad *query's* whole row would leave it zero permitted keys, and softmax
    over an all-(−∞) row is NaN (0/0), which then poisons every downstream
    tensor via matmuls. The convention across implementations: let pad query
    positions compute whatever they compute, and exclude them where it matters
    — in the loss (lesson 06). Garbage in ignored positions is free; NaN
    anywhere is fatal.
    """
    # (batch, seq_k) -> (batch, 1, 1, seq_k): real tokens are True.
    return (token_ids != pad_id)[:, None, None, :]


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """AND together any number of attend-permission masks (None = no constraint).

    A position is attendable only if EVERY mask allows it — permissions
    intersect, so boolean AND is the only correct combinator. Broadcasting
    does the shape work: causal (seq, seq) AND padding (batch, 1, 1, seq_k)
    → (batch, 1, seq, seq_k), exactly what a decoder over padded batches
    needs. Returns None if no constraints were given, matching attention's
    "no mask" fast path.
    """
    result = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else result & mask
    return result
