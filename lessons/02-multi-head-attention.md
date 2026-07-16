# 02 — Multi-head attention

**Phase:** 1 (The Transformer) · **Paper:** Vaswani et al. 2017, §3.2.2 (see `papers/attention.md`)
**Code:** `src/llmlab/components/attention.py` (`MultiHeadAttention`) · **Test:** `pytest tests/test_multi_head_attention.py -v`

## The problem

Lesson 01 left two loose ends.

**Where do Q, K, V come from?** They "fell from the sky" — real transformers
derive all three from the same token vectors. How, exactly, and why three
separate derivations?

**One head = one relation.** Softmax gives each query exactly ONE weight
distribution. But a token often needs to look at several places *for different
reasons* — "it" needs its antecedent (coreference) and its verb (syntax) at
the same time. A single head must average these needs into one blurred lookup.
The paper says it flatly: *"averaging inhibits this."*

## The idea

**Idea 1 — learned projections.** Q = X·W_Q, K = X·W_K, V = X·W_V: three
learned linear maps of the *same* input. Three separate ones because the roles
differ — what a token asks about (Q), what it advertises (K), and what it
hands over when read (V) are different functions of its content. If Q and K
shared a matrix, q·q = ‖q‖² would make every token match *itself* hardest,
and attention would collapse toward the diagonal.

**Idea 2 — split the width into heads.** Run h attentions in parallel, each
with its own projections so each can learn its own relation. The affordability
trick: each head lives in a subspace of size d_head = d_model / h (512/8 = 64
in the paper). We don't pay h× for h relations — we *split* the compute one
full-width head would have used. Heads change how the width is spent, not how
much there is.

**Idea 3 — the output mix W_O.** Concatenating head outputs just stacks them
in disjoint channel ranges; head 3's findings would live in channels 128–191
forever. One final linear map W_O lets heads combine what they found.

    MultiHead(X) = Concat(head_1, ..., head_h) W^O,   head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)

## The math (mostly shapes this time)

The only new math is bookkeeping — which is exactly why it deserves care;
every bug here is a silent one.

    x:        (batch, seq, d_model)
    q, k, v:  (batch, seq, d_model)          after the three projections
    split:    (batch, seq, h, d_head)        view() — free reinterpretation of the last axis
    transpose:(batch, h, seq, d_head)        heads become a broadcast dim
    sdpa:     (batch, h, seq, d_head)        lesson 01's function, unchanged — h rides along
    merge:    (batch, seq, d_model)          transpose back, contiguous(), view()
    W_O:      (batch, seq, d_model)

Lesson 01's function needed zero changes: we wrote it to broadcast over
leading `...` dims, and "heads" is just one more leading dim. The √ scaling is
now √d_head — each head scales by *its* width.

Parameter bill: W_Q, W_K, W_V, W_O are each (d_model × d_model) → **4·d_model²**,
independent of h (test-verified). One (d_model × d_model) W_Q is exactly the h
per-head matrices W_i^Q laid side by side: project once, slice columns.

## The code

`MultiHeadAttention(d_model, num_heads)` computes the block **twice on purpose**:

- `forward(x, naive=True)` — the definition made executable: loop over heads,
  slice out each head's d_head-wide chunk of q/k/v, call lesson 01's function
  per head, concat.
- `forward(x)` — the batched version: `view` + `transpose` fold heads into a
  tensor dim; one call to lesson 01's function computes all heads in one matmul.

`test_naive_loop_equals_batched` proves they're identical (with and without a
causal mask): **the reshape is bookkeeping, not math.** This loop→batch move is
the single most reused pattern in ML engineering; we'll do it again for KV
caching and GQA.

Details worth remembering:

- `.contiguous()` before the merging `.view()`: transpose only changes strides,
  and `view` needs memory to physically be in that order. Deleting it raises an
  error (the friendly kind of bug).
- Self- vs. cross-attention is one argument: `x_context` defaults to `x_query`.
  K and V always come from the same sequence — they're the index and the
  payload of the same entries.
- `d_model % num_heads != 0` raises: we slice the width, we don't pad it.

## What breaks without it

- **One head instead of eight:** −0.9 BLEU in the paper's ablation (Table 3).
  Not catastrophic — just consistently worse, because every layer can express
  only one relation at a time.
- **Too many heads (32 at d_model=512):** also worse! d_head drops to 16 and
  each head is too low-dimensional to compute sharp similarities. Heads are a
  trade-off, not a free lunch.
- **No W_O:** heads can never mix; each writes to its private channel range
  and downstream layers must undo the partition themselves.
- **Shared Q/K projection:** self-matching dominates (q·q is maximal in its
  own direction); attention degenerates toward the identity.
- **Forgetting `.contiguous()`:** immediate RuntimeError — PyTorch protects
  you from this one.

## Verified claims (tests)

- Output keeps the input's shape (ready for the residual stream, lesson 03);
  weights are (batch, heads, seq, seq), every row a distribution.
- Naive per-head loop == batched reshape, exactly, masked and unmasked.
- At random init, heads already produce different attention patterns.
- Parameter count is 4·d_model² + 4·d_model regardless of num_heads.
- Still permutation-equivariant: shuffle input tokens → outputs shuffle along.
  Multi-head fixed "one relation" but not order-blindness (→ lesson 04).
- Cross-attention: output length follows queries, weights span both sequences.
- All four projection matrices receive nonzero gradients in one backward pass.
- With copied weights, matches `torch.nn.MultiheadAttention` to 1e-5.

## Open questions (→ future lessons)

- Attention gathers from *other* tokens, but nothing yet processes each token
  individually — the FFN, plus residuals and LayerNorm to make depth trainable.
  (L03)
- Still order-blind (test-proven). Positional encodings. (L04)
- Do all h heads really need their own K and V? MQA/GQA (Phase 4) answer no —
  queries need diversity; keys/values can be shared. Huge for inference memory.
- What do trained heads actually learn? Induction heads, previous-token heads —
  interpretability rabbit hole for when we train a real model (Phase 2).
