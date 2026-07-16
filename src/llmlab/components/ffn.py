"""Feed-forward network variants, built up lesson by lesson.

Lesson 03: the position-wise FFN (Vaswani et al. 2017, §3.3 / eq. 2):

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂        (d_model → d_ff → d_model)

Attention (lessons 01–02) is the only place tokens *exchange* information —
and notice it never transforms content nonlinearly: outputs are weighted
averages of value vectors. The FFN is the other half of the layer: each token
is processed *by itself*, through a genuine nonlinearity. A useful reading of
the division of labor: attention gathers, the FFN thinks.

"Position-wise" means the same two-layer MLP (same weights) is applied to
every position independently — token 7's output depends only on token 7's
input. Cross-token mixing is attention's job alone; the FFN adds per-token
depth. (Tests verify this independence directly.)

The inner expansion d_ff = 4·d_model (2048 for d_model=512 in the paper) goes
up-then-down: project into a wider space, apply ReLU there, come back. The
width buys capacity — the FFN holds ~2/3 of a transformer layer's parameters
(2·d_model·d_ff vs. attention's 4·d_model²) — and later interpretability work
(Geva et al. 2021) reads these layers as the model's key-value *memories*,
where e.g. factual associations get stored.

Phase 2 swaps ReLU → GELU (GPT-2); Phase 4 swaps the whole shape → SwiGLU
(LLaMA); Phase 6 makes it sparse (Mixture of Experts). All plug in here.
"""

import torch
from torch import Tensor, nn


class PositionwiseFFN(nn.Module):
    """The 2017 paper's FFN: Linear → ReLU → Linear, applied per position."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # the paper's ratio; the default ever since

        self.w1 = nn.Linear(d_model, d_ff)   # expand: (d_model → d_ff)
        self.w2 = nn.Linear(d_ff, d_model)   # contract: (d_ff → d_model)

    def forward(self, x: Tensor) -> Tensor:  # x: (..., seq, d_model)
        # nn.Linear acts on the last axis only, so every (batch, seq) slot is
        # transformed independently with the same weights — that IS the
        # "position-wise" property; no code is needed to enforce it.
        hidden = torch.relu(self.w1(x))      # (..., seq, d_ff)
        # ReLU is the reason the FFN isn't pointless: without a nonlinearity,
        # W₂(W₁x) collapses to a single linear map — no more expressive than
        # one matrix, and the 4× width would buy nothing.
        return self.w2(hidden)               # (..., seq, d_model)
