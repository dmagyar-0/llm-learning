# 04 — Sinusoidal positional encodings

**Phase:** 1 (The Transformer) · **Paper:** Vaswani 2017 §3.5 — see `papers/positional-encodings.md`
**Code:** `src/llmlab/components/positional.py` · **Test:** `pytest tests/test_positional.py -v`

## The problem

Everything from lessons 01–03 is **permutation-equivariant** — and now that's
test-verified, not just claimed: shuffle the input tokens of a whole
transformer block and the outputs shuffle identically,
`block(x[perm]) == block(x)[perm]`. The reasons stack up:

- attention is a softmax-weighted average over a *set* — nothing in Q·Kᵀ
  records where a key sat in the sequence;
- the FFN and LayerNorm act on each token independently.

So the model literally cannot distinguish "dog bites man" from "man bites
dog". RNNs never had this problem — order was baked into their one-token-at-
a-time computation — but that's exactly what the transformer discarded to
win parallelism. The architecture has no slot for order, so order must be
injected **as data**: something position-dependent added to the token
vectors at the bottom of the stack.

## The idea

Stamp each position with a unique, bounded code. The paper's choice is
**binary counting made continuous**: in binary, bit i flips with period
2^(i+1) — fast bits distinguish neighbors, slow bits distinguish distant
regions, and together the bounded digits give every number a unique code.
Replace the hard bit-flips with smooth (sin, cos) oscillators whose
wavelengths grow geometrically, and you get codes that are unique, bounded,
smooth in position, and defined for *any* position — no table to run off
the end of.

## The math

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Channel pair (2i, 2i+1) oscillates at ω_i = 10000^(−2i/d_model): pair 0 has
wavelength 2π (spins fast), the last pair 10000·2π (nearly constant).

**Why pairs?** Because the property the paper actually wanted needs both
coordinates. The angle-sum identities

    sin(ω(p+k)) =  cos(ωk)·sin(ωp) + sin(ωk)·cos(ωp)
    cos(ω(p+k)) = −sin(ωk)·sin(ωp) + cos(ωk)·cos(ωp)

say: PE(pos+k) is obtained from PE(pos) by **rotating each pair through
angle ω_i·k — a matrix built from the offset k alone, the same for every
pos**. So "attend 3 tokens back" is one linear map on these codes,
learnable by lesson 02's W_Q/W_K without knowing absolute positions.
(Test-verified: one 2×2 rotation per pair maps row p to row p+k for all p.)

A second consequence we proved in tests: **PE(p)·PE(p+k) = Σᵢ cos(ωᵢk)** —
the dot product depends only on the offset k, never on p. A dot-product
machine like attention can read *distance* directly off these codes. But
cosine is even, so +k and −k are indistinguishable: raw dot products sense
how far, never *which side*. Direction lives in the phase, reachable only
via the linear/rotation route — the crack that relative encodings (Shaw
2018) and RoPE (Phase 4) later drive a wedge into.

## The code

`sinusoidal_table(seq_len, d_model)` builds the table: frequencies as
`exp(−log(10000)·2i/d)` (numerically friendlier than fractional powers of
10000), an outer product `positions ⊗ frequencies` for the angles, sin into
even channels, cos into odd. Row p never depends on seq_len — "position 5"
means the same thing in every sequence (test-verified prefix property).

`SinusoidalPositionalEncoding` wraps it as a module: precompute up to
`max_len` once, `forward(x) = x + table[:seq]`. Decisions worth remembering:

- **Add, don't concatenate.** Concatenation reserves channels and costs
  width; addition superimposes position onto content and lets each head's
  learned projections extract whichever mixture it needs. Works because
  random high-dim embeddings are nearly orthogonal to the structured
  sinusoids. (The paper also scales embeddings by √d_model so content isn't
  drowned by the unit-amplitude codes — that lands in our embedding lesson.)
- **Buffer, not parameter.** Zero learnable parameters (test: `parameters()`
  is empty, `state_dict()` is empty with `persistent=False`). The optimizer
  never sees it; `.to(device)` still moves it. Phase 2's learned positional
  embeddings differ from this file by exactly one word — `Parameter` — and
  that word is the whole architectural choice.
- **Fail loudly at the edges:** odd d_model (can't rotate half a pair) and
  seq_len > max_len (don't silently wrap positions) both raise.

## What we learned the hard way

A test asserted `(x + PE) − x == PE` with default `allclose` tolerances —
and failed. Not a bug: in float32, adding O(1) noise to a small table entry
and subtracting it back loses ~1e-7 to rounding, and the default
`atol=1e-8` is stricter than machine epsilon at that magnitude. The add is
exact in math, not in bits. Moral: numeric tolerances are claims too —
know what precision the arithmetic can actually deliver before asserting it.

## What breaks without it

- **No positional encoding at all:** the model is a bag-of-tokens machine —
  perfectly fine at "which words appeared", structurally incapable of
  syntax. The permutation test is the proof.
- **Concatenate instead of add:** costs width or a projection; every
  downstream layer must honor the content/position split forever.
- **One channel per frequency (no pairs):** the rotation property dies —
  sin alone can't be phase-shifted linearly; relative offsets stop being
  learnable as linear maps.
- **Linear instead of geometric frequencies:** you'd spend all channels on
  one length-scale; geometric spacing gives every scale of "how far apart"
  its own channel pair, like binary gives every magnitude its own bit.
- **Unbounded codes (e.g. PE = pos itself):** position 3000 has norm 3000×
  position 1 — lesson 03's scale discipline destroyed at the first layer.
- **A learned table (preview, not breakage):** works equally well in-domain
  (the paper measured "nearly identical") but has no entry for position
  max_len+1. GPT-2 chose it anyway — simplicity won. Phase 2 material.

## Verified claims (tests)

- The pre-lesson-04 stack is permutation-equivariant; PE breaks it (the
  motivating fact, executable).
- Formula pinned to hand values: PE(0) = (0,1,0,1,…); pair 0 = (sin p, cos p).
- All entries in [−1,1] at position 2047, yet all 2048 codes distinct.
- Row p independent of table length (prefix property).
- PE(p+k) = per-pair rotation of PE(p), matrix depending on k only.
- PE(p)·PE(p+k) depends only on k — and equally on −k (direction-blind).
- Zero parameters, empty state_dict; gradient through the add is exactly 1;
  odd d_model and over-length sequences raise.

## Open questions (→ future lessons)

- The block exists, positions exist — but attention can still peek at the
  future. Causal + padding masks, then the full encoder–decoder. (L05, L06)
- If sinusoids and learned tables tie in quality, why did GPT-2 pick
  learned? And what actually happens when you feed positions beyond
  training length? (Phase 2, then Phase 5's context-extension reading.)
- The rotation property makes relative offsets *learnable* — RoPE asks why
  the model should have to learn them at all and rotates q, k directly
  inside attention. (Phase 4.)
- Raw PE dot products are direction-blind; how do relative-position methods
  restore direction? (Shaw 2018 / T5 buckets, Phase 3 reading.)
