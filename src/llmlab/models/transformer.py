"""The full 2017 Transformer — lesson 06: assembly, no new mechanisms.

Every part of this file was built in lessons 01–05; what's new is only the
wiring (Vaswani et al. 2017, §3.1, Figure 1):

    src ids ─→ embed ×√d ─→ +PE ─→ [EncoderBlock × N] ─→ memory ─┐
                                                                 │ K,V (every block)
    tgt ids ─→ embed ×√d ─→ +PE ─→ [DecoderBlock × N] ←──────────┘
                                        │
                                        └─→ Linear(d_model → tgt_vocab) = logits

The encoder reads the source *bidirectionally* — all-to-all attention, only
padding masked, because the source is given, not generated: nothing to hide.
The decoder generates left-to-right behind lesson 05's causal mask, consulting
the source through cross-attention in every block.

Config-driven per this repo's design principles: the model class reads
everything from a dataclass, and `TransformerConfig.tiny()` runs on CPU in
well under a second. Training it (with the loss masking lesson 05 promised)
is lesson 07.
"""

from dataclasses import dataclass

from torch import Tensor, nn

from llmlab.components.block import DecoderBlock, TransformerBlock
from llmlab.components.embeddings import TokenEmbedding
from llmlab.components.masking import causal_mask, combine_masks, padding_mask
from llmlab.components.positional import SinusoidalPositionalEncoding


@dataclass
class TransformerConfig:
    """Everything that defines a Transformer instance, in one place.

    The paper's base model is d_model=512, num_heads=8, num_layers=6,
    d_ff=2048 (~65M params). We parameterize instead of hard-coding so that
    "the paper's model" vs. "a toy" is a config choice, not a code change —
    the seed of the repo's plug-in architecture ambition.
    """

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6          # N: depth of BOTH stacks (paper uses 6 + 6)
    d_ff: int | None = None      # None → components default to 4·d_model
    max_len: int = 512           # PE buffer capacity (fails loudly beyond)
    dropout: float = 0.0         # paper: 0.1 when training; 0 keeps tests exact
    pad_id: int = 0              # which token id means "padding, not content"

    @classmethod
    def tiny(cls, src_vocab_size: int = 32, tgt_vocab_size: int = 32) -> "TransformerConfig":
        """CPU-in-seconds config for learning and tests (~55k params)."""
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=2,
            max_len=64,
        )


class Transformer(nn.Module):
    """Encoder–decoder transformer, assembled from lessons 01–05.

    The forward pass takes token IDS and builds every mask itself from the
    config's pad_id — callers should not hand-roll masks (that's how the
    "wrong mask on the wrong attention" class of bugs happens). The mask
    recipe, from lesson 05 and papers/attention.md:

        encoder self-attn → src padding
        decoder self-attn → causal ∧ tgt padding
        cross-attn        → src padding (no triangle — source has no future)

    Two vocabularies and two embedding tables because source and target may
    be different languages (the paper translates). GPT (Phase 2) has one
    stream and one vocabulary — that simplification, plus deleting the
    encoder and cross-attention, is the whole architectural diff.

    The final projection to logits is kept UNTIED from the target embedding
    here; the paper actually shares that weight matrix (§3.4). Weight tying
    gets its own lesson in Phase 2, where GPT-2 makes it standard practice.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        c = config

        # Two entry pipelines, one per language. The PE module is shared —
        # it is a pure function of position (lesson 04: zero parameters),
        # so there is nothing language-specific to keep separate.
        self.src_embed = TokenEmbedding(c.src_vocab_size, c.d_model)
        self.tgt_embed = TokenEmbedding(c.tgt_vocab_size, c.d_model)
        self.positional = SinusoidalPositionalEncoding(c.d_model, max_len=c.max_len)
        # §5.4: dropout on (embedding + PE) too, not just inside blocks.
        self.embed_dropout = nn.Dropout(c.dropout)

        self.encoder = nn.ModuleList(
            TransformerBlock(c.d_model, c.num_heads, c.d_ff, c.dropout)
            for _ in range(c.num_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(c.d_model, c.num_heads, c.d_ff, c.dropout)
            for _ in range(c.num_layers)
        )

        # d_model → vocab: one score per candidate next token. Softmax is NOT
        # applied here — training wants raw logits (cross-entropy applies
        # log-softmax itself, more stably), and sampling (Phase 2) wants to
        # temperature-scale logits before normalizing.
        self.lm_head = nn.Linear(c.d_model, c.tgt_vocab_size)

    def encode(self, src_ids: Tensor) -> Tensor:
        """(batch, src_seq) ids → (batch, src_seq, d_model) memory."""
        src_mask = padding_mask(src_ids, self.config.pad_id)
        x = self.embed_dropout(self.positional(self.src_embed(src_ids)))
        for block in self.encoder:
            x = block(x, mask=src_mask)  # same mask every layer: padding
        return x                          # doesn't stop being padding

    def decode(self, tgt_ids: Tensor, memory: Tensor, src_ids: Tensor) -> Tensor:
        """(batch, tgt_seq) ids + memory → (batch, tgt_seq, tgt_vocab) logits.

        src_ids is passed (not a precomputed mask) so the cross-attention
        mask is derived here, next to the self-mask — all mask logic in one
        visible place.
        """
        self_mask = combine_masks(                       # (batch, 1, tgt, tgt)
            causal_mask(tgt_ids.shape[1], device=tgt_ids.device),
            padding_mask(tgt_ids, self.config.pad_id),
        )
        cross_mask = padding_mask(src_ids, self.config.pad_id)  # (batch,1,1,src)

        x = self.embed_dropout(self.positional(self.tgt_embed(tgt_ids)))
        for block in self.decoder:
            x = block(x, memory, self_mask=self_mask, cross_mask=cross_mask)
        return self.lm_head(x)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        """The training-time pass: teacher forcing.

        tgt_ids is the target sequence shifted right (starts with BOS);
        position t's logits are its prediction for target token t+1 — the
        causal mask guarantees the prediction can't peek (lesson 05), so all
        tgt_seq predictions are honest and computed in ONE pass. The loss
        that consumes these logits (ignoring pad positions) is lesson 07.
        """
        memory = self.encode(src_ids)                 # read the source once
        return self.decode(tgt_ids, memory, src_ids)  # generate against it
