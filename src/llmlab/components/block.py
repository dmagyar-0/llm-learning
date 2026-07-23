"""The transformer block: attention + FFN, glued by residuals and norms.

Lesson 03: the 2017 paper's sublayer formula (§3.1, §5.4 — "post-norm"):

    x = LayerNorm(x + Sublayer(x))        applied twice per block:
                                          Sublayer = multi-head attention,
                                          then Sublayer = position-wise FFN

The residual addition `x + Sublayer(x)` is ResNet's idea (He et al. 2015)
transplanted: a sublayer doesn't *replace* the representation, it computes a
*correction* to add onto it. Two consequences:

1. **Optimization.** ∂(x + F(x))/∂x = I + ∂F/∂x — the gradient always has a
   direct identity path back to earlier layers. Through L blocks the gradient
   contains an unattenuated term instead of being a product of L Jacobians
   that shrinks geometrically. This is the vanishing-gradient disease that
   killed RNNs over *time*, showing up over *depth* — cured the same way:
   give the signal a highway. (Test: healthy gradient through 8 stacked
   blocks — measured with a proper loss; see the amusing trap in the test
   about why `output.sum()` is NOT a proper loss after a LayerNorm.)

2. **The residual-stream picture.** Because every block only *adds*, the
   d_model-wide vector flowing through the network acts like a shared
   workspace: attention writes in what it gathered from other tokens, the FFN
   writes in what it computed from that, and every later layer can read
   everything earlier layers wrote. This framing — blocks as readers/writers
   on a stream — is how interpretability work talks about transformers today.

Residual addition also explains a design constraint from lesson 02: every
sublayer must map d_model → d_model, or `x + Sublayer(x)` wouldn't typecheck.
That's why attention ends in W_O and the FFN contracts back down.

Note the placement: the paper normalizes AFTER the addition ("post-norm"),
so the highway itself passes through LN at every block. This trains — but
deep post-norm stacks famously need learning-rate warmup, because early in
training the LN-scrambled highway isn't yet a clean identity. GPT-2 (Phase 2)
moves the norm inside the branch — `x = x + Sublayer(LayerNorm(x))`,
"pre-norm" — restoring the untouched highway; that one-line change is a big
part of why 100-layer models train. We implement 2017 faithfully first, and
`norm_placement` becomes a config knob when we assemble models.
"""

from torch import Tensor, nn

from llmlab.components.attention import MultiHeadAttention
from llmlab.components.ffn import PositionwiseFFN
from llmlab.components.norms import LayerNorm


class TransformerBlock(nn.Module):
    """One encoder-style block: self-attention + FFN, with a norm-placement knob.

    Same two sublayers as the 2017 paper; what's configurable (lesson 10) is
    WHERE the LayerNorm sits relative to the residual add:

        post-norm (Vaswani 2017):   x = LN(x + Sublayer(x))
        pre-norm  (GPT-2, 2019):    x = x + Sublayer(LN(x))

    Post-norm puts LN *on the highway* — every residual add is renormalized, so
    the identity path of lesson 03 passes through L LayerNorms on its way back.
    Pre-norm puts LN *inside the branch* — the highway is a pure running sum of
    sublayer outputs, and the gradient gets an unattenuated identity path from
    the loss to every layer (the derivation is in lessons/10). That one change
    is a large part of why deep transformers train without heavy LR warmup, so
    every model after 2019 defaults to it; we keep post-norm as the faithful
    2017 default and let assembled models (GPT) choose.

    Note pre-norm leaves the FINAL output un-normalized (the highway is never
    renormed), so a pre-norm *model* must add one closing LayerNorm before its
    head — that lives in the model (GPT.norm_final), not here.

    dropout: applied to each sublayer's output before the residual add
    (§5.4, p_drop=0.1). Default 0.0 — matters only when training; tests want
    determinism.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        norm_placement: str = "post",
    ) -> None:
        super().__init__()
        if norm_placement not in ("post", "pre"):
            raise ValueError(
                f"norm_placement must be 'post' or 'pre', got {norm_placement!r}"
            )
        self.norm_placement = norm_placement
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        # One LayerNorm per sublayer, each with its own γ/β — the stream may
        # need different operating points after "gather" vs. after "think".
        # (Same two norms in both placements; only where we apply them moves.)
        self.norm_attn = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (batch, seq, d_model) → (batch, seq, d_model).

        Shape in == shape out is the whole point: blocks stack like Lego,
        and depth becomes a pure hyperparameter (N=6 in the paper).
        """
        if self.norm_placement == "post":
            # Add THEN norm — the LN sits on the residual highway.
            attn_out, _ = self.attention(x, mask=mask)      # gather
            x = self.norm_attn(x + self.dropout(attn_out))
            ffn_out = self.ffn(x)                           # think
            x = self.norm_ffn(x + self.dropout(ffn_out))
        else:  # "pre": norm THEN sublayer — the LN sits inside the branch, and
            #        the raw x is added back untouched, keeping the highway clean.
            attn_out, _ = self.attention(self.norm_attn(x), mask=mask)  # gather
            x = x + self.dropout(attn_out)
            ffn_out = self.ffn(self.norm_ffn(x))            # think
            x = x + self.dropout(ffn_out)

        return x


class DecoderBlock(nn.Module):
    """One post-norm decoder block (Vaswani et al. 2017, §3.1, Figure 1 right).

    Lesson 06: the decoder block is the encoder block with a THIRD sublayer
    wedged in — **cross-attention** — giving three steps per block:

        1. causal self-attention   "what have I generated so far?"
        2. cross-attention         "what does the source say?"
        3. FFN                     "think about both."

    Cross-attention is where lesson 02's `x_context` argument finally earns
    its keep: queries come from the decoder stream, but keys AND values come
    from the encoder's output (`memory`). It is the ONLY bridge between the
    two stacks — and it is Bahdanau 2014's encoder–decoder attention reborn:
    each target position learns which source positions to consult. Note that
    `memory` is the encoder's FINAL output, fed identically to every decoder
    block — decoder depth means re-*querying* the source repeatedly, not
    processing it further.

    Mask bookkeeping (each attention gets a different one — the classic
    source of silent decoder bugs):

        self-attention  → causal ∧ target-padding   (lesson 05's combination)
        cross-attention → source-padding ONLY — no triangle. The causal mask
          hides the future of the sequence being GENERATED; the source is
          fully known before generation begins, so there is no future to
          hide — only pad keys to skip. (Lesson 05's open question, answered:
          masks encode availability, and the whole source is available.)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        # Three sublayers → three residual+norm wrappers, each with its own
        # γ/β (same reasoning as the encoder block: different operating
        # points after each step).
        self.norm_self = LayerNorm(d_model)
        self.norm_cross = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,                 # (batch, tgt_seq, d_model) decoder stream
        memory: Tensor,            # (batch, src_seq, d_model) encoder output
        self_mask: Tensor | None = None,   # causal ∧ tgt-padding
        cross_mask: Tensor | None = None,  # src-padding (no triangle!)
    ) -> Tensor:
        """(batch, tgt_seq, d_model) → same shape; blocks stack like Lego."""
        # 1 — gather from the past of the target sequence.
        attn_out, _ = self.self_attention(x, mask=self_mask)
        x = self.norm_self(x + self.dropout(attn_out))

        # 2 — gather from the source: Q from x, K/V from memory. Output
        # length follows the QUERIES (tgt_seq) — lesson 01's shapes at work:
        # (tgt_seq, src_seq) weights @ (src_seq, d_v) values → tgt_seq rows.
        cross_out, _ = self.cross_attention(x, x_context=memory, mask=cross_mask)
        x = self.norm_cross(x + self.dropout(cross_out))

        # 3 — think.
        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_out))

        return x
