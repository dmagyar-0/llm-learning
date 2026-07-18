# 05 — Causal and padding masks

**Phase:** 1 (The Transformer) · **Paper:** Vaswani 2017 §3.2.3 — see `papers/attention.md`
**Code:** `src/llmlab/components/masking.py` · **Test:** `pytest tests/test_masking.py -v`

## The problem

Two separate problems, one mechanism.

**The decoder problem.** An autoregressive model generates left-to-right: when
producing token t+1, tokens t+2, t+3, ... do not exist yet. But lesson 01's
attention is all-to-all — during *training* (where the whole target sequence
is available at once) position t can happily attend to position t+1. A model
allowed to do that learns the perfect cheat: "predict" the next token by
reading it. Training loss collapses; generation ability is zero, because at
generation time the input it learned to rely on is missing. Training-time
information flow must match generation-time information flow.

**The batching problem.** GPUs want rectangular tensors; sentences have
ragged lengths. So short sequences get a dummy PAD token appended until the
batch is a rectangle. But attention averages over *all* keys — including the
pad positions, which are storage, not content. Unmasked, the amount of junk
mixed into every token's representation would depend on how long the *other*
sequences in its batch happened to be.

## The idea

Both fixes use the hook lesson 01 already built: overwrite forbidden scores
with −∞ *before* softmax, so e^−∞ = 0 kills the weight and softmax
renormalizes the survivors to sum to 1. (Zeroing weights *after* softmax
would break the sum-to-1 property — the freed mass has to go somewhere.)

- **Causal mask:** query q may attend key k ⟺ k ≤ q. Same triangle for every
  sequence and head — a fixed function of positions, shape (seq, seq).
- **Padding mask:** key k is attendable ⟺ token k ≠ PAD. Per-example *data*,
  computed from token ids, shape (batch, 1, 1, seq_k) — the singleton dims
  broadcast over heads and query positions.
- **Both at once:** permissions intersect → boolean AND, and broadcasting
  turns (seq, seq) ∧ (batch, 1, 1, seq) into (batch, 1, seq, seq).

## The math

The paper's whole treatment is one sentence (§3.2.3): mask out "all values in
the input of the softmax which correspond to illegal connections". The
interesting math is what the triangle *buys*:

**One pass = n training examples.** Behind the causal mask, output position t
is a function of inputs 0..t only. So a single forward over an n-token
sequence yields n honest next-token predictions simultaneously — position 0
predicts token 1, position 1 predicts token 2, ... An RNN produces the same n
signals but strictly sequentially. This parallelism-without-cheating is the
reason decoder transformers could train on internet-scale data, and the causal
mask is the entire trick.

**The prefix identity.** Causal attention at position t over the full
sequence ≡ ordinary attention over just the prefix x[0..t] (test-verified,
bit-for-bit): the past's outputs never change when the future arrives. Read
forward, that's why generation can *cache* K and V instead of recomputing
them per token — the KV cache (Phase 5) is this identity, exploited.

## The code

`masking.py` is three small functions — the mask *mechanism* has lived in
`attention.py` since lesson 01; today we finally feed it.

- `causal_mask(seq_len)` — written as `positions[:, None] >= positions[None, :]`
  rather than `torch.tril(...)`, so the rule (k ≤ q) *is* the code instead of
  a property the reader must recall about triangular matrices. Returned at
  (seq, seq) — the smallest shape that broadcasts — because the rule is
  identical for every batch element and head.
- `padding_mask(token_ids, pad_id)` — `(ids != pad_id)[:, None, None, :]`.
  Masks **keys only**. Deliberately: masking a pad *query's* entire row would
  leave it zero permitted keys, and softmax of an all-(−∞) row is 0/0 = NaN —
  which then poisons everything downstream through the matmuls. Convention:
  pad queries compute garbage, and the garbage is excluded where it matters,
  in the loss (lesson 06). Garbage in ignored positions is free; NaN is fatal.
- `combine_masks(*masks)` — boolean AND with None-tolerance (None = no
  constraint, and no masks at all returns None so attention keeps its
  no-mask fast path).

## What breaks without it

- **No causal mask (decoder):** training accuracy soars, generation is
  babble — the model learned to read the answer, not predict it. Test
  `test_future_tokens_cannot_change_past_outputs` shows the leak directly:
  unmasked, editing token 4 changes outputs at positions 0–3.
- **No padding mask:** every token's representation shifts depending on how
  much padding its batch-mates forced — `test_padding_makes_batching_invisible`
  fails, i.e. the same sentence gets different outputs in different batches.
- **Mask after softmax instead of before:** weights no longer sum to 1;
  attention outputs shrink in proportion to how much was masked.
- **Mask the diagonal too / mask pad query rows:** a row with zero permitted
  keys is NaN, not zero (pinned in `test_fully_masked_row_is_nan_not_zero`).
  A NaN loss a few steps into training is very often an over-zealous mask.
- **Gradient leak, the subtle one:** forward causality without backward
  causality would still let the loss at t tune future inputs.
  `test_no_gradient_flows_to_the_future` checks ∂out[t]/∂x[>t] = 0 exactly —
  it holds for free because masking sits before softmax, inside autograd.

## Verified claims (tests)

- Hand-checked 4×4 triangle; future weights exactly 0.0; rows sum to 1.
- Editing a future token leaves earlier outputs bit-for-bit unchanged through
  a full block (and demonstrably changes them without the mask).
- ∂out[t]/∂x[>t] = 0 exactly; past gradients nonzero.
- Full-sequence causal output at t equals plain attention on the prefix
  0..t — the KV-cache identity.
- Pad keys get zero weight everywhere; padded-and-masked outputs match the
  unpadded sequence at real positions (and don't without the mask).
- Causal ∧ padding composes by broadcast to (batch, 1, seq, seq).
- A fully-masked row yields NaN, not zeros (the hazard, pinned).

## Open questions (→ future lessons)

- We now have every component of the 2017 machine. Lesson 06 assembles the
  full encoder–decoder — including the third mask combination: *cross*-
  attention (decoder queries, encoder keys) needs the encoder's padding mask
  but no causal triangle. Why not?
- The loss must ignore pad-query garbage — how loss masking actually works
  (lesson 06, when we first train).
- A causal mask makes attention position-*asymmetric* — position 0 sees 1 key,
  position n−1 sees n. Does that asymmetry alone leak enough order information
  that a decoder could work with *no* positional encoding? (The "NoPE" line of
  work says partly yes — Phase 3/4 reading.)
- The triangle is the simplest mask *shape*. Sliding-window attention
  (Mistral, Phase 5) is a band-shaped causal mask; FlashAttention never
  materializes the (seq, seq) matrix at all yet respects the same triangle.
  Mask shape is where long-context efficiency work lives.
