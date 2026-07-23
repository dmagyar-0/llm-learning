"""The decoder-only language model — lesson 09, and the start of Phase 2.

This is where the field's main line begins. Lesson 06 built the full 2017
encoder–decoder; GPT (Radford et al. 2018, 2019 — papers/gpt.md) is what you
get by **deleting half of it**:

    enc–dec (lesson 06)                 GPT (this file)
    ───────────────────                 ───────────────
    src stream ─→ encoder ─┐            (no encoder)
    tgt stream ─→ decoder ←┘ cross      one stream ─→ [self-attn + FFN] × N
       │  self / cross / FFN               │  self / FFN
       └─→ head (tgt vocab)                └─→ head (one vocab)

What actually changes, and nothing else:

1. **One stream, one vocabulary.** No source/target split — there is only the
   text, predicting its own continuation. So one embedding table, not two.
2. **No cross-attention.** With no encoder there is nothing to cross-attend to,
   so the three-sublayer `DecoderBlock` collapses to the two-sublayer
   *self-attention + FFN* block — which is *exactly* lesson 03's
   `TransformerBlock`. We reuse it unchanged and simply feed it a causal mask.
   The encoder block and a GPT block are the same object under different masks;
   that equivalence is the whole point of the plug-in design.
3. **The objective is next-token prediction on the stream itself.** Lesson 07's
   enc–dec toy data pre-split each example into (tgt_in, tgt_out). Here the split
   is trivial and lives at the call site: `input = tokens[:, :-1]`,
   `labels = tokens[:, 1:]`. Same masked cross-entropy, no source.

Deliberately still 2017 here: **post-norm** (lesson 03's block) and **sinusoidal**
positions (lesson 04). This is therefore a *generic* decoder-only LM, not GPT-2
in particular. Specializing it into GPT-2 is the next lessons — pre-norm, learned
positions, BPE, weight tying/init, sampling — each a single component swap on this
same skeleton. Building the skeleton first is what makes those swaps one-liners.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from llmlab.components.block import TransformerBlock
from llmlab.components.embeddings import TokenEmbedding
from llmlab.components.masking import causal_mask, combine_masks, padding_mask
from llmlab.components.norms import LayerNorm
from llmlab.components.positional import build_positional


@dataclass
class GPTConfig:
    """Everything that defines a decoder-only LM instance.

    One `vocab_size` (contrast `TransformerConfig`'s src/tgt pair) — the single
    stream is the whole simplification. GPT-2's real sizes ranged from 124M
    (12 layers, d_model=768) to 1.5B (48 layers, d_model=1600); we parameterize
    so "small GPT-2" vs. "toy" is a config choice, per the repo's design.
    """

    vocab_size: int
    d_model: int = 768           # GPT-2 "small" width
    num_heads: int = 12
    num_layers: int = 12         # N: depth of the single decoder stack
    d_ff: int | None = None      # None → FFN defaults to 4·d_model
    max_len: int = 1024          # GPT-2's context window; PE buffer capacity
    dropout: float = 0.0         # 0 keeps tests exact; >0 when training
    pad_id: int = 0              # which id means "padding, not content"
    norm_placement: str = "pre"  # GPT-2 (2019) moved LN inside the residual
    #                              branch — pre-norm. Set "post" for 2017 style.
    positional: str = "learned"  # GPT-2 uses a LEARNED position table (lesson
    #                              11). Set "sinusoidal" for the 2017 fixed
    #                              formula (lesson 04) — same additive interface.

    @classmethod
    def tiny(cls, vocab_size: int = 32) -> "GPTConfig":
        """CPU-in-seconds config for learning and tests (~30k params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=2,
            max_len=64,
        )


class GPT(nn.Module):
    """A decoder-only, autoregressive language model (Radford et al. 2018/2019).

    The forward pass takes token ids and returns per-position next-token logits.
    Every position t's logits are its prediction for token t+1 — the causal mask
    (lesson 05) guarantees position t cannot see t+1, so all `seq` predictions
    are honest and produced in ONE pass (teacher forcing, lesson 07). That
    parallelism over positions is why decoder LMs train on internet-scale data.

    Masking here is strictly simpler than the enc–dec model: there is exactly
    one attention per block and it always gets the same mask —

        causal ∧ padding

    causal because we generate left-to-right, padding because batches are
    rectangular. No cross-mask, no source — the two most bug-prone masks of
    lesson 06 are simply gone.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        c = config

        # One entry pipeline (vs. the enc–dec model's two). The ×√d_model scale
        # (lesson 06's embedding) is unchanged; the positional component is now a
        # config choice (lesson 11): "learned" gives GPT-2's trained position
        # table, "sinusoidal" the 2017 fixed formula (lesson 04). Both share the
        # same additive (batch, seq, d_model)→same-shape interface, so this line
        # is the only place the choice lives.
        self.embed = TokenEmbedding(c.vocab_size, c.d_model)
        self.positional = build_positional(c.positional, c.d_model, c.max_len)
        self.embed_dropout = nn.Dropout(c.dropout)

        # The stack. `TransformerBlock` is lesson 03's self-attention + FFN block
        # — reused verbatim. A GPT block IS an encoder block; only the mask we
        # pass in makes it autoregressive, and `norm_placement` chooses where its
        # LayerNorms sit (lesson 10: GPT-2 uses "pre").
        self.blocks = nn.ModuleList(
            TransformerBlock(
                c.d_model, c.num_heads, c.d_ff, c.dropout,
                norm_placement=c.norm_placement,
            )
            for _ in range(c.num_layers)
        )

        # Pre-norm never renormalizes the residual highway, so the stream leaves
        # the last block un-normalized (and its magnitude has grown with depth —
        # it's a running sum). GPT-2 closes the stack with one final LayerNorm
        # ("ln_f") before the head; post-norm needs none (its last block already
        # ended in an LN), so we use an identity there.
        self.norm_final = (
            LayerNorm(c.d_model) if c.norm_placement == "pre" else nn.Identity()
        )

        # d_model → vocab: one score per candidate next token. Raw logits (no
        # softmax): cross-entropy applies log-softmax itself, more stably, and
        # sampling (a later lesson) wants to temperature-scale before normalizing.
        # GPT ties this weight to the embedding table (§weight-tying lesson);
        # kept untied for now, exactly as lesson 06 left it.
        self.lm_head = nn.Linear(c.d_model, c.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        """(batch, seq) ids → (batch, seq, vocab) next-token logits.

        One mask, built here so callers never hand-roll it (that is how the
        "forgot the causal mask" class of silent bugs happens):

            causal_mask(seq)              (seq, seq)      k ≤ q
            ∧ padding_mask(input_ids)     (batch,1,1,seq) real tokens
            = (batch, 1, seq, seq)        broadcast-combined
        """
        mask = combine_masks(
            causal_mask(input_ids.shape[1], device=input_ids.device),
            padding_mask(input_ids, self.config.pad_id),
        )
        x = self.embed_dropout(self.positional(self.embed(input_ids)))
        for block in self.blocks:
            x = block(x, mask=mask)  # same causal∧padding mask every layer
        x = self.norm_final(x)       # ln_f (pre-norm) or identity (post-norm)
        return self.lm_head(x)       # (batch, seq, vocab)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> Tensor:
        """Greedily continue a prompt — lesson 08's loop, decoder-only form.

        This is *not* a new mechanism: it is lesson 08's autoregressive loop with
        the encoder removed. There, generation re-ran the decoder against a fixed
        `memory`; here there is no memory, so we just re-run the model over the
        growing token stream and append its argmax each step.

            (batch, prompt) ids → (batch, prompt + up-to-max_new) ids

        Decoder-only makes this the *natural* interface the enc–dec model lacked:
        hand the model any prefix, get its continuation. That "continue the
        prompt" API is precisely what GPT-2 showed subsumes most NLP tasks
        (papers/gpt.md).

        `eos_id` optionally stops a row early; finished rows are frozen with PAD
        so a batch decodes together (same latch as lesson 08). Cost is O(n²) —
        step t reprocesses the whole prefix — because we still have no KV cache
        (Phase 5, deliberately, so we feel the cost first). Only temperature/
        top-k/top-p turn this greedy pick into real *sampling*; that is its own
        lesson — `generate` stays greedy for now.
        """
        self.eval()  # deterministic: freeze dropout so a call is reproducible
        device = prompt_ids.device
        batch = prompt_ids.shape[0]
        gen = prompt_ids
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Re-score the whole prefix; only the LAST position predicts the next
            # token (earlier positions re-predict tokens already committed).
            logits = self.forward(gen)              # (batch, cur, vocab)
            next_token = logits[:, -1].argmax(dim=-1)  # (batch,) greedy pick

            if eos_id is not None:
                # A finished row emits PAD forever, freezing its content while
                # other rows continue (its logits are computed but discarded).
                next_token = next_token.masked_fill(finished, self.config.pad_id)
            gen = torch.cat([gen, next_token[:, None]], dim=1)  # append column

            if eos_id is not None:
                finished |= next_token == eos_id
                if bool(finished.all()):
                    break  # every row has stopped; may end before max_new_tokens

        return gen
