# 06 — The full encoder–decoder assembly

**Phase:** 1 (The Transformer) · **Paper:** Vaswani 2017 §3.1, §3.4, Figure 1 — see `papers/attention.md`
**Code:** `src/llmlab/components/embeddings.py`, `components/block.py` (DecoderBlock), `models/transformer.py` · **Test:** `pytest tests/test_transformer.py -v`

*(The roadmap's item 6 bundles assembly + toy training; per the small-portions
rule this session is assembly only — training, with the loss masking lesson 05
promised, is lesson 07.)*

## The problem

We have every organ — attention, heads, FFN, norms, residuals, positions,
masks — and no body. Two gaps stand between the components and a working
model:

1. **Real input is token ids, not vectors.** Something must turn integer ids
   into the (batch, seq, d_model) tensors everything so far assumed — and at
   the other end, turn d_model vectors back into scores over a vocabulary.
2. **The 2017 task is translation:** a *given* source sequence in, a
   *generated* target sequence out. Reading and generating have different
   information rules (the source is fully known; the target's future is not),
   so the paper builds two different stacks and one bridge between them.

## The idea

The machine is two stacks and a bridge:

    src ids ─→ embed ×√d ─→ +PE ─→ [EncoderBlock × N] ─→ memory ─┐
                                                                 │ K,V (every block)
    tgt ids ─→ embed ×√d ─→ +PE ─→ [DecoderBlock × N] ←──────────┘
                                        └─→ Linear → logits

The **encoder** reads the source *bidirectionally* — all-to-all attention,
no triangle, because there is nothing to hide in a sequence that is given
rather than generated. The **decoder** is lesson 05's causal machine, plus a
third sublayer in every block: **cross-attention** — queries from the decoder
stream, keys and values from the encoder's output. The decoder block's three
steps read naturally: *what have I said so far?* (causal self-attn), *what
does the source say?* (cross-attn), *think* (FFN). Cross-attention is
Bahdanau 2014's alignment idea reborn — and the only bridge: delete it and
the stacks never communicate (a test proves the bridge is live).

One subtlety: every decoder block gets the encoder's *final* output as K/V.
Decoder depth means re-*querying* the source repeatedly with progressively
refined questions — not processing the source further.

## The math

Nothing new — assembly is bookkeeping, and the bookkeeping that matters is
**which mask goes where** (the paper leaves it implicit; getting it wrong is
the classic silent decoder bug):

| Attention site        | Mask                    | Why |
|-----------------------|-------------------------|-----|
| encoder self-attn     | src padding             | source is given: hide only storage |
| decoder self-attn     | causal ∧ tgt padding    | lesson 05, verbatim |
| cross-attn            | src padding, **no triangle** | the triangle hides the future of the sequence being *generated*; the source has no future — it is fully available before generation begins |

That last row answers lesson 05's open question: masks encode *availability*,
not position — and the whole source is available.

The one derivation is the **embedding scale** (§3.4), closing lesson 04's
open thread. With tied-weight-compatible init (std 1/√d, the same variance
argument as lesson 01's √d_k), an embedding row has norm ≈ 1. A sinusoidal PE
row has norm √(d/2) — each sin/cos pair contributes sin²+cos² = 1, and there
are d/2 pairs. Added unscaled at d=512, position is ~16× louder than content.
Multiplying embeddings by √d_model lifts content to norm ≈ √d — same volume
as position (ratio √2). Test-pinned with the actual norms.

## The code

- `embeddings.py` — `TokenEmbedding`: `nn.Embedding` lookup ×√d_model. The
  docstring carries the two views of an embedding (lookup = one-hot matmul),
  which explains why only used tokens' rows get gradients — a fact that
  returns when we build BPE (Phase 2).
- `block.py` — `DecoderBlock`: three sublayers, each in its own post-norm
  residual wrapper. Cross-attention is just lesson 02's `MultiHeadAttention`
  with `x_context=memory` — the argument finally earns its keep.
- `models/transformer.py` — `TransformerConfig` (dataclass; the paper's
  512/8/6 as defaults, `.tiny()` = 32/2/2, ~55k params, CPU-instant) and
  `Transformer`. Design decisions worth remembering:
  - **The model builds every mask itself** from ids + `pad_id`, inside
    `encode`/`decode` — callers never hand-roll masks, because "wrong mask
    on the wrong attention" is precisely the bug class to design away.
  - **`forward` is teacher forcing:** tgt ids arrive shifted right (BOS
    first); position t's logits predict target token t+1, and the causal
    mask keeps all tgt_seq predictions honest in one pass.
  - **`lm_head` outputs raw logits**, no softmax: cross-entropy wants
    log-softmax internally (stabler), and sampling (Phase 2) wants to
    temperature-scale logits first.
  - **Untied output projection** — the paper actually shares the embedding
    matrix with the pre-softmax linear (§3.4); we defer weight tying to
    Phase 2 where GPT-2 makes it standard.

## What breaks without it

- **Causal mask missing in ANY one of the N decoder blocks:** the end-to-end
  causality test fails — assembly can undo a lesson-05 guarantee with one
  forgotten argument in one layer. That's why the test edits a future token
  and demands bit-for-bit invariance through the *full* model.
- **Triangle on cross-attention** (over-masking): target position 0 could
  only see source position 0 — early generation goes blind to most of the
  source. Translation with word reorder ("dog bites man" → languages where
  the verb comes last) becomes structurally impossible.
- **Triangle on the encoder:** source token 0's representation can't see its
  own sentence's end; the bidirectional-on-purpose test pins the correct
  behavior (and its docstring contrasts it with the decoder test — same edit
  shape, opposite correct answer).
- **No ×√d on embeddings:** content enters at ~1/16 of position's volume;
  early layers must first learn to amplify content out of the positional
  noise floor.
- **A module built but not called in forward** (the classic assembly bug):
  caught by the every-parameter-gets-gradient test — dead wiring shows up as
  a parameter with `grad is None`.

## Verified claims (tests)

- Shapes: (batch, tgt_seq, tgt_vocab) logits; tiny config stays < 100k params.
- Editing a future target token leaves earlier logits bit-for-bit unchanged —
  causality survives assembly.
- Editing one source token moves logits at *every* target position — the
  cross-attention bridge carries signal, and the source has no future.
- Editing the *last* source token changes the memory at source position 0 —
  the encoder is bidirectional on purpose.
- Padding source and target simultaneously leaves real-position logits
  unchanged — all three mask sites honor padding at once.
- One backward pass reaches every parameter with nonzero gradient.
- Norms: raw embedding ≈ 1, PE = √(d/2), scaled embedding ≈ √2 × PE.

## Open questions (→ future lessons)

- The model outputs honest logits but has never seen data. Lesson 07: the
  training loop — cross-entropy with pad positions masked out of the loss
  (lesson 05's promise), teacher forcing in practice, and a toy copy/reverse
  task where we can *watch* it learn on CPU.
- Generation: `forward` needs the whole tgt sequence, but at inference we
  have only what we've generated so far — the loop that feeds the model its
  own output, and why it's O(n²)-per-token naive (→ KV cache, Phase 5).
- If the encoder–decoder split is so principled, why did GPT — a decoder
  minus cross-attention, one stack, one vocabulary — take over? What is
  actually lost when the "given" source just becomes tokens in the same
  stream as the generation? (Phase 2's opening question.)
- Our stacks are 2 layers; the paper's are 6, GPT-3's are 96. Post-norm
  famously needs LR warmup as depth grows — we'll meet this the moment we
  train deeper stacks (pre-norm, Phase 2).
