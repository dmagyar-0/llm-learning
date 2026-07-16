# Positional encodings — giving an order-blind model a sense of order

**Papers:** Attention Is All You Need §3.5 — Vaswani et al. 2017
([1706.03762](https://arxiv.org/abs/1706.03762)) ·
(context) Convolutional Sequence to Sequence Learning — Gehring et al. 2017
([1705.03122](https://arxiv.org/abs/1705.03122)) ·
(forward pointers) Shaw et al. 2018 ([1803.02155](https://arxiv.org/abs/1803.02155)),
RoFormer/RoPE — Su et al. 2021 ([2104.09864](https://arxiv.org/abs/2104.09864))
**Status:** ● studied (Phase 1, lesson 04)

## Why this is a problem at all — and why it's *new* in 2017

RNNs never needed position information: they consume tokens one at a time, so
order is baked into the *computation*. Convolutions know local order through
the layout of their kernel. The transformer deliberately threw both away to
win parallelism — and got **permutation equivariance** in the bargain:
shuffle the input tokens and the outputs shuffle identically, because
attention treats the sequence as a *set* (a softmax-weighted average has no
idea where its inputs sat), and the FFN/LayerNorm act per token. "Dog bites
man" and "man bites dog" are the same input. Some fix is mandatory, and it
must be *injected as data*, since the architecture itself has no slot for
order.

## What Vaswani et al. chose (§3.5)

Add (not concatenate) a position-dependent vector to each token embedding at
the bottom of the stack:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Each pair of channels (2i, 2i+1) is a sin/cos oscillator; wavelengths run in
a geometric progression from 2π (channel pair 0, flips every few tokens) to
10000·2π (last pair, nearly constant over any real sequence). The paper's
stated reasons:

1. **Bounded** — every value in [−1, 1], at any position. Fits lesson 03's
   scale discipline.
2. **Unique** per position (up to astronomically long sequences), and
   **deterministic** — no parameters, no training, defined for *any* length,
   including lengths never seen in training.
3. **Relative positions are linearly accessible** — the property they
   "hypothesized would allow the model to easily learn to attend by relative
   positions": for any fixed offset k, PE(pos+k) is a *linear function* of
   PE(pos). Concretely, each sin/cos pair transforms by a 2×2 **rotation**
   through angle ωᵢ·k — the same rotation regardless of pos (the angle-sum
   identities, nothing more). A learned linear map can therefore express
   "shift attention left by 3" without knowing absolute positions.

The intuition we like best: **it's binary counting made continuous**. Write
positions in binary and bit i flips with period 2^(i+1) — fast bits low,
slow bits high. The sinusoids are the same trick with smooth dials instead
of hard flips (differentiable, and the geometric base is 10000 rather
than 2).

They also tried **learned position embeddings** (as ConvS2S/Gehring et al.
2017 used, and BERT/GPT-2 later chose): a plain trainable lookup table of
max_len × d_model. Result: "nearly identical" quality. They shipped sinusoids
anyway for the extrapolation argument — a table has *no entry* for position
max_len+1, sinusoids are defined everywhere. (Extrapolating in principle ≠
generalizing in practice, as the long-context literature later found — the
model still never *trained* on those positions.)

## Why add instead of concatenate?

Concatenation would keep content and position in separate channels but costs
width (or forces a projection), and everything downstream would have to carry
the split. Addition superimposes position onto content in the same d_model
space and lets the *learned projections* (W_Q, W_K of lesson 02) decide which
mixture of the two any given head cares about — heads that need position can
extract it; heads that don't can ignore it. It works because random
high-dimensional embeddings and the highly structured sinusoids are nearly
orthogonal in practice; the paper also scales embeddings by √d_model (§3.4)
so the trained embeddings aren't drowned by the unit-amplitude sinusoids.

## What we verified in code (tests, lesson 04)

- The whole pre-lesson-04 stack really is permutation-equivariant, and adding
  PE really breaks it (the motivating fact, made executable).
- The rotation property, exactly: one 2×2 matrix per channel pair, built only
  from the offset k, maps PE(pos) → PE(pos+k) for every pos at once.
- PE(p)·PE(p+k) depends only on k, never on p (= Σᵢ cos(ωᵢk)) — a dot-product
  machine like attention can read *distance* straight off these codes…
- …but that dot product is **even in k**: distance yes, *direction* no. The
  direction lives in the phase, which needs the linear/rotation route, not
  the raw dot product.

## Where the field went next (forward pointers)

- **Learned absolute** (GPT-2, Phase 2): simpler, no extrapolation, works.
- **Relative encodings** (Shaw et al. 2018; T5 buckets, Phase 3 reading):
  stop encoding *where a token is*, encode *how far apart a pair is*, right
  where attention compares them. Trains better on long-range structure.
- **RoPE** (Su et al. 2021, Phase 4): the endpoint of the rotation insight —
  don't *add* a code at the bottom and hope W_Q/W_K learn the rotation trick;
  *rotate* q and k by position-dependent angles inside every attention call,
  making q·k depend on relative position by construction. Same sinusoids,
  moved from the data into the mechanism. Nearly every modern open model
  uses it.
