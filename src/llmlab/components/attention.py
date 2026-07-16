"""Attention mechanisms, built up lesson by lesson.

Lesson 01: scaled dot-product attention — the single equation at the heart of
every transformer since 2017:

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V        (Vaswani et al. 2017, eq. 1)

Lesson 02: multi-head attention — where Q, K, V actually come from (learned
linear projections of the token vectors), and why we run h small attentions
in parallel instead of one big one (Vaswani et al. 2017, §3.2.2).

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
from torch import Tensor, nn


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


class MultiHeadAttention(nn.Module):
    """Multi-head attention (Vaswani et al. 2017, §3.2.2).

        MultiHead(X) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(X W_i^Q, X W_i^K, X W_i^V)

    Two ideas are introduced here, and it pays to see them separately:

    **Idea 1 — learned Q/K/V projections.** In lesson 01, Q, K, V were handed
    to us. In a real transformer all three are *linear projections of the same
    token vectors*: Q = X·W_Q, K = X·W_K, V = X·W_V. Three separate matrices
    because the three roles are different — what a token *asks about* (Q), what
    it *advertises to others* (K), and what it *hands over when read* (V) are
    three different functions of its content. With a single shared projection
    (Q = K), every token would necessarily match itself most strongly — score
    q·q = ‖q‖² is maximal in its own direction — and attention would collapse
    toward a diagonal, unable to learn e.g. "verbs should look at their subjects".

    **Idea 2 — many heads, each in a smaller subspace.** One attention head
    produces ONE weight distribution per query — a token that needs to look at
    two places for two different reasons must average them into a single blur
    ("averaging inhibits this" — the paper's exact complaint). So we run h
    heads in parallel, each with its *own* learned projections, so each can
    learn a different relation. The trick that makes this affordable: each head
    works in a subspace of size d_head = d_model / h, so h heads cost roughly
    the same FLOPs and parameters as one full-width head would. We don't buy h
    relations with h× compute — we buy them by *splitting* the compute we had.

    Afterwards the h outputs (each d_head wide) are concatenated back to
    d_model and mixed by one more linear map W^O. Without W^O, head i's output
    would only ever occupy channels [i·d_head, (i+1)·d_head) of the residual
    stream — heads could never combine what they found. W^O is where
    cross-head mixing happens.

    This module implements the computation twice on purpose:
    `forward(..., naive=True)` loops over heads and calls lesson 01's function
    per head — slow but transparently the definition; the default batched path
    folds heads into a tensor dimension and computes all heads in one matmul.
    A test proves both give identical outputs.
    """

    def __init__(self, d_model: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            # Heads must tile d_model exactly: we *split* the model width into
            # h slices of d_head, we don't add width. 512 = 8 × 64 in the paper.
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / h (paper §3.2.2)

        # One (d_model, d_model) matrix per role, NOT one per head per role.
        # This is the standard fusion: W_Q is all h per-head matrices W_i^Q
        # (each d_model × d_head) laid side by side — projecting once and then
        # slicing columns is exactly equivalent to h separate projections,
        # because each output column depends on one column of the weight.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        # The output mix. Note the parameter bill: 4 d_model² total (+biases),
        # independent of num_heads — heads change how the width is *used*,
        # not how much of it there is.
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x_query: Tensor,             # (batch, seq_q, d_model) — tokens doing the asking
        x_context: Tensor | None = None,  # (batch, seq_k, d_model) — tokens being read
        mask: Tensor | None = None,  # bool, broadcastable to (batch, heads, seq_q, seq_k)
        naive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return (output, attention_weights).

        output:  (batch, seq_q, d_model)
        weights: (batch, num_heads, seq_q, seq_k) — one distribution per head:
                 h different "opinions" about where each query should look.

        Self-attention: call with x_query only (context defaults to it).
        Cross-attention: pass the other sequence as x_context — queries come
        from one sequence, keys AND values from the other (K and V always
        travel together: they are the index and the payload of the same
        entries being looked up).
        """
        if x_context is None:
            x_context = x_query  # self-attention: the sequence reads itself

        batch, seq_q, _ = x_query.shape
        seq_k = x_context.shape[1]

        # The projections. Shapes stay (batch, seq, d_model); heads don't
        # exist yet — they appear only when we *reinterpret* the last axis.
        q = self.w_q(x_query)    # (batch, seq_q, d_model)
        k = self.w_k(x_context)  # (batch, seq_k, d_model)
        v = self.w_v(x_context)  # (batch, seq_k, d_model)

        if naive:
            # The definition, made executable: per head, slice out that head's
            # d_head-wide chunk of q/k/v and run lesson 01's attention on it.
            head_outputs = []
            head_weights = []
            for h in range(self.num_heads):
                lo, hi = h * self.d_head, (h + 1) * self.d_head
                # Give this head its slice of the (possibly 4-D) mask too.
                head_mask = mask
                if head_mask is not None and head_mask.dim() == 4:
                    head_mask = head_mask[:, min(h, head_mask.shape[1] - 1)]
                out_h, w_h = scaled_dot_product_attention(
                    q[..., lo:hi], k[..., lo:hi], v[..., lo:hi], mask=head_mask
                )  # out_h: (batch, seq_q, d_head), w_h: (batch, seq_q, seq_k)
                head_outputs.append(out_h)
                head_weights.append(w_h)
            # Concat(head_1, ..., head_h): undo the split.
            concat = torch.cat(head_outputs, dim=-1)          # (batch, seq_q, d_model)
            weights = torch.stack(head_weights, dim=1)        # (batch, heads, seq_q, seq_k)
        else:
            # The batched path: identical math, zero Python loops.
            # (batch, seq, d_model) -> (batch, seq, heads, d_head): a free
            # reinterpretation of memory — head h's chunk is the same
            # [h·d_head, (h+1)·d_head) slice the naive loop took.
            # Then transpose to (batch, heads, seq, d_head) so that `heads`
            # sits in the broadcast dims of lesson 01's function and every
            # head is computed by the same batched matmul.
            def split_heads(t: Tensor) -> Tensor:
                return t.view(batch, -1, self.num_heads, self.d_head).transpose(1, 2)

            out, weights = scaled_dot_product_attention(
                split_heads(q), split_heads(k), split_heads(v), mask=mask
            )  # out: (batch, heads, seq_q, d_head)

            # Undo: (batch, heads, seq_q, d_head) -> (batch, seq_q, d_model).
            # .contiguous() is needed because transpose only changes strides,
            # and .view() requires the memory to actually be in that order.
            concat = out.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)

        # The cross-head mix — without this, heads live in disjoint channels
        # forever and "concat" would be a partition, not a combination.
        output = self.w_o(concat)  # (batch, seq_q, d_model)

        return output, weights
