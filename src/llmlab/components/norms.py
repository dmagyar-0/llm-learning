"""Normalization layers, built up lesson by lesson.

Lesson 03: LayerNorm (Ba, Kiros & Hinton 2016) — normalize each token's
vector across its own features:

    LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β        μ, σ² over the last axis

Why normalize at all: deep nets train best when every layer sees inputs in a
stable range. In a transformer the residual stream is a running *sum* of many
sublayer outputs, so its magnitude would drift layer by layer — and we know
from lesson 01 that attention is scale-sensitive (big inputs → big scores →
saturated softmax → dead gradients). LN resets the stream to a known
operating point.

Why per-token and not per-batch (BatchNorm): BatchNorm normalizes each
feature across the *batch*, which ties examples together, behaves differently
at train vs. test time, and breaks at batch=1 — all wrong for variable-length
sequence models. LN needs only the token's own d_model numbers: same formula
at any batch size, any sequence length, training or inference.

Phase 4 will add RMSNorm here (LLaMA's variant: drop the mean-centering,
keep the rescaling).
"""

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension, written out by hand.

    Functionally identical to `torch.nn.LayerNorm(d_model)` (test-verified);
    we write it ourselves because all four lines carry ideas.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # ε lives inside the √ to keep the division finite when a token's
        # features have (near-)zero variance — e.g. a padding token that is
        # all zeros would otherwise produce 0/0 = NaN and poison the backward
        # pass of the whole batch.
        self.eps = eps
        # γ and β give back what normalization takes away. Forcing every
        # token to exactly mean-0/var-1 would also delete *useful* scale
        # information; with learned per-feature gain and bias, the network
        # can restore any mean/scale it actually wants — normalization
        # removes the accident, not the choice. Initialized to the identity
        # (γ=1, β=0): at the start, LN is pure normalization.
        self.gain = nn.Parameter(torch.ones(d_model))   # γ: (d_model,)
        self.bias = nn.Parameter(torch.zeros(d_model))  # β: (d_model,)

    def forward(self, x: Tensor) -> Tensor:  # x: (..., d_model)
        # Statistics over the FEATURE axis only — each token normalizes
        # itself, independent of its neighbors in the sequence or the batch.
        mean = x.mean(dim=-1, keepdim=True)                    # (..., 1)
        # unbiased=False → divide by d_model, not d_model−1: we're describing
        # this exact vector, not estimating a population (and it matches the
        # paper and torch.nn.LayerNorm).
        var = x.var(dim=-1, keepdim=True, unbiased=False)      # (..., 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)        # mean 0, var 1
        return self.gain * x_hat + self.bias                   # (..., d_model)
