"""Masked cross-entropy — the loss that trains every LLM (lesson 07).

The objective is maximum likelihood (papers/training.md). The model factors
the probability of a target sequence by the chain rule,

    p(y | x) = ∏_t p(y_t | y_<t, x),

and we minimize the negative log, which the log turns into a SUM of
per-position terms:

    L = − Σ_t log p(y_t | y_<t, x)  =  Σ_t cross_entropy(logits_t, y_t)

Each position is an independent "score the true next token" problem — exactly
the (batch, seq, vocab) logits layout the causally-masked decoder already
produces in one pass (teacher forcing, lesson 06). This loss, unchanged, is
what GPT-4-class pretraining minimizes; the entire craft is in what x, y and
the model are.

The one non-textbook ingredient is the MASK — lesson 05's promise kept.
Padding made batches rectangular; pad positions hold no data, so they must
contribute nothing to the loss. We include a position iff its LABEL is real:
that keeps position L (label EOS, input y_L — the "learn to stop" signal) and
drops label-PAD positions, whose logits then receive exactly zero gradient
(they multiply by 0 in the masked sum — pinned in the tests).

Why divide by the real-token COUNT (per-token mean), not batch size:

- a per-sequence mean would weight token 3-of-3 more than token 3-of-8 —
  MLE says every observed token is one datum, worth the same;
- the number becomes comparable across any batch shape, and e^loss is always
  "perplexity per token" — the y-axis of every scaling-law plot (Phase 3).
"""

import torch.nn.functional as F
from torch import Tensor

from llmlab.data.toy import PAD


def masked_cross_entropy(logits: Tensor, labels: Tensor, pad_id: int = PAD) -> Tensor:
    """Mean cross-entropy over real (non-pad-label) positions.

    logits: (batch, seq, vocab) — raw scores from the model (no softmax:
            log_softmax happens HERE, fused, because softmax-then-log
            overflows for large scores while logsumexp subtracts the max).
    labels: (batch, seq) int — true next tokens, pad_id where there is none.
    returns: () scalar, differentiable.

    Written out in three lines rather than calling
    F.cross_entropy(ignore_index=...) so the mechanics are visible — a test
    pins that they agree exactly.
    """
    log_probs = F.log_softmax(logits, dim=-1)             # (batch, seq, vocab)
    # Pick out each position's log p(true token): gather needs the index to
    # have the same number of dims, hence unsqueeze/squeeze.
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq)
    real = labels != pad_id                               # (batch, seq) bool
    # Sum over real positions only, divide by their count. (A pad label of 0
    # gathered SOME log-prob — the mask is what makes that value irrelevant
    # and its gradient exactly zero.)
    return (nll * real).sum() / real.sum()
