# 01 — Scaled dot-product attention

**Phase:** 1 (The Transformer) · **Paper:** Vaswani et al. 2017, eq. 1 (see `papers/attention.md`)
**Code:** `src/llmlab/components/attention.py` · **Test:** `pytest tests/test_attention.py -v`

## The problem

Before 2017, sequence models were RNNs: to know what word 1000 means, compute
999 states first. Two consequences: training can't parallelize across the
sequence (GPUs idle), and information decays over long distances (each step is
a chance to forget). Attention began (Bahdanau 2014) as a *patch* on RNNs;
the Transformer's move was to ask: what if the patch is the whole mechanism?

## The idea

Attention is a **soft, differentiable dictionary lookup**.

- A Python dict: one key matches, you get exactly its value.
- Attention: *every* key matches a little. Compare the **query** against all
  **keys**, softmax the scores into weights summing to 1, return the weighted
  average of all **values**.

Soft beats hard here for one reason: a weighted average has gradients. The
model can *learn* what questions to ask (Q), what to advertise (K), and what
to hand over (V). Where Q, K, V come from is the next lessons' topic — today
they're just given tensors.

## The math

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Shapes: Q is (seq_q, d_k), K is (seq_k, d_k), V is (seq_k, d_v) →
scores/weights (seq_q, seq_k) → output (seq_q, d_v). Batch/head dims broadcast
in front.

### Deriving the √d_k (the one non-obvious part)

Assume q and k have unit-variance, uncorrelated components (roughly true at
initialization). Their dot product is a sum of d_k terms:

    q·k = Σᵢ qᵢkᵢ  →  Var(q·k) = Σᵢ Var(qᵢkᵢ) = d_k · (1·1) = d_k

So scores have standard deviation √d_k — they grow with head width even when
similarity doesn't. Softmax of inputs with std ≈ 16 (d_k = 256) is essentially
an argmax: one weight ≈ 1, rest ≈ 0, and ∂softmax/∂score ≈ 0 everywhere →
**gradients vanish and learning stalls**, precisely when you pick a decent d_k.
Dividing by √d_k restores Var ≈ 1 for any d_k.
`test_scaling_keeps_score_variance_near_one` verifies this empirically:
raw variance ≈ 256, scaled ≈ 1.

## The code

`scaled_dot_product_attention(query, key, value, mask=None)` — five steps:

1. `scores = q @ kᵀ` — all pairwise similarities in one matmul. This matmul *is*
   the parallelism the RNN lacked: no step depends on another.
2. `scores / √d_k` — as derived above.
3. `masked_fill(~mask, -inf)` — **before** softmax, so forbidden positions get
   weight exp(-inf) = 0 while remaining weights still sum to 1. (Zeroing weights
   *after* softmax would break normalization.)
4. `softmax(dim=-1)` — over the **key** axis. Wrong axis = the classic silent bug:
   everything still runs, shapes still match, nothing means anything.
5. `weights @ v` — the weighted average itself.

We return the weights too (nice for visualization); real implementations don't
even materialize them (FlashAttention, Phase 5).

## What breaks without it

- **No √d_k:** softmax saturates at realistic d_k → vanishing gradients → the
  model trains poorly. (This is a one-character bug that silently costs you
  a few points of loss.)
- **Mask after softmax instead of before:** weights no longer sum to 1;
  downstream magnitudes drift.
- **Softmax over dim=-2:** weights normalize over queries instead of keys;
  outputs are garbage but no error is raised.

## Verified claims (tests)

- Weights are a probability distribution per query; output shapes correct.
- Identical keys → exactly uniform weights → output = mean(V). Attention
  degrades into averaging when there's nothing to distinguish.
- One dominant match → weights ≈ one-hot → output ≈ that value (the dict limit).
- A pencil-and-paper 2-key example matches to 6 decimals.
- Causal mask zeroes the upper triangle, rows still sum to 1.
- `gradcheck` passes in float64 — fully differentiable.
- Agrees with `torch.nn.functional.scaled_dot_product_attention` to 1e-5.

## Open questions (→ future lessons)

- Q, K, V fell from the sky today. In self-attention they're three learned
  linear projections of the *same* token vectors — why three separate ones? (L02)
- One attention does one weighted average — one "relation" per layer. How do we
  get many relations at once? Multi-head. (L02)
- The scores matrix is (seq × seq): O(n²). Fine at n=128, painful at n=128k.
  (Phase 5.)
- Attention never looks at *where* a token is — shuffle the keys/values and
  outputs shuffle along. Position must be injected. (L04)
