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

import torch
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

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        """Autoregressive greedy generation — lesson 08's new mechanism.

        `forward` computed all target predictions in ONE pass because teacher
        forcing handed it the true prefix. At inference there is no true
        prefix: the model must feed its OWN previous outputs back in, one token
        at a time. That loop is what turns a next-token *scorer* into a
        *generator*, and it is the entire difference between training and
        deployment.

        (batch, src_seq) ids  →  (batch, gen_seq) ids, each row starting with
        BOS and (if the model learned to stop) ending at its first EOS. Rows
        that emit EOS are frozen — every later position is forced to PAD — so a
        batch decodes together even though sequences finish at different steps.

        `bos_id`/`eos_id` are passed in rather than read from the model because
        they are a property of the *task's* vocabulary (lesson 07's specials),
        not of the architecture; `pad_id` IS the model's (it shaped every mask).

        "Greedy" = take the argmax token at each step. It is the simplest
        decision rule and the one that most sharply exposes exposure bias
        (below): no sampling to jitter out of a bad prefix, no beam to hedge —
        one committed choice per step, and every choice conditions all the
        rest. Temperature / top-k / top-p sampling and beam search are Phase 2.
        """
        self.eval()  # freeze dropout/BN-like state: generation must be
        #              deterministic, not a different sample each call
        device = src_ids.device
        batch = src_ids.shape[0]

        # The source is fixed, so encode it EXACTLY ONCE and reuse the memory
        # for every step — re-encoding per token would be pure waste (the whole
        # point of the encoder/decoder split: read once, generate many).
        memory = self.encode(src_ids)                       # (batch, src, d)

        # Every sequence is born as just [BOS] — the seed the decoder conditions
        # its first real prediction on (lesson 07: BOS gives position 0 a past).
        gen = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        # Per-row latch: once a row has produced EOS it is "done" and must stop
        # contributing new content, or it would ramble past its own stop signal.
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Re-run the DECODER over the whole prefix so far. This is the naive,
            # honest version: step t redoes the work of steps 0..t−1, so decoding
            # n tokens costs O(n²) decoder passes. Caching the unchanged keys and
            # values to make it O(n) is the KV cache — deliberately deferred to
            # Phase 5 so we first feel the cost it removes.
            logits = self.decode(gen, memory, src_ids)      # (batch, cur, vocab)
            next_token = logits[:, -1].argmax(dim=-1)        # (batch,) greedy pick
            #   only the LAST position's logits matter: earlier positions just
            #   re-predict tokens we already committed to.

            # A finished row emits PAD forever — freezing its content while other
            # rows keep going. (Its logits are computed but thrown away here.)
            next_token = next_token.masked_fill(finished, self.config.pad_id)
            gen = torch.cat([gen, next_token[:, None]], dim=1)  # append column

            finished |= next_token == eos_id
            if bool(finished.all()):
                break  # nothing left to generate; stop early (may end < max_new)

        return gen
