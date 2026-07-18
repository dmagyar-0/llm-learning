"""Token embeddings — the entry point of the model (lesson 06).

Everything before this lesson operated on ready-made vectors. Real input is
token *ids* — integers indexing a vocabulary. The embedding layer is the
learned table that turns id i into row i of a (vocab_size, d_model) matrix.

Two ways to see the same operation:

1. **Lookup:** `table[ids]` — fancy indexing, which is all `nn.Embedding` does.
2. **Matrix multiply:** one_hot(i) @ table — the id as a one-hot row vector
   times the table. Same numbers, but this view explains the gradient: only
   row i receives gradient from token i's loss (the one-hot zeroes out every
   other row). Rare tokens' vectors therefore train rarely — a fact that
   matters when we build our BPE tokenizer (Phase 2) and choose a vocabulary.

Where the vectors come from: nowhere. They are initialized randomly and
shaped entirely by the training signal — tokens used in similar contexts get
pushed toward similar vectors because that lowers the loss. This is word2vec's
discovery (Mikolov 2013, Phase 0) happening implicitly inside the model.
"""

import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Embedding lookup × √d_model (Vaswani et al. 2017, §3.4).

    The ×√d_model deserves its derivation — it closes a thread from lesson 04.

    The paper ties this table to the pre-softmax output projection (one matrix
    serving as both first and last layer). A linear layer's weights want
    components of scale ~1/√d_model (so outputs stay O(1) — same variance
    argument as lesson 01's √d_k). With that init, an embedding VECTOR has
    norm ≈ √(d_model · 1/d_model) = 1.

    Meanwhile lesson 04's sinusoidal PE row has norm √(d_model/2) — each
    sin/cos pair contributes sin² + cos² = 1, and there are d_model/2 pairs.

    Add them unscaled and position outshouts content by a factor of √(d/2) —
    for d_model = 512, the PE is ~16× louder than the embedding. Multiplying
    embeddings by √d_model lifts their norm to ≈ √d_model, and content speaks
    at (slightly above) position's volume. (Test-verified: the norm ratio.)

    We initialize at std 1/√d_model to make that story true in our code too
    (PyTorch's default N(0,1) would have norm √d_model with no rescue needed —
    but would break the moment we tie weights in Phase 2, so we adopt the
    tied-compatible convention now and keep the paper's forward-pass scale).
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.table = nn.Embedding(vocab_size, d_model)
        # Linear-layer-scale init (see docstring). ~N(0, 1/d_model) per entry.
        nn.init.normal_(self.table.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, ids: Tensor) -> Tensor:
        # ids: (batch, seq) int64 → (batch, seq, d_model) float
        return self.table(ids) * math.sqrt(self.d_model)
