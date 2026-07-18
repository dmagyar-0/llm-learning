# Attention Is All You Need — Vaswani et al., 2017

**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762) · **Status:** ● studying (Phase 1)
**Prerequisite context:** Bahdanau et al. 2014 ([1409.0473](https://arxiv.org/abs/1409.0473))

## The problem the paper solved

In 2017 the best translation models were RNN encoder–decoders *with attention
bolted on*. Attention (Bahdanau 2014) already existed — but only as a patch: the
RNN read the sentence word by word, and attention let the decoder peek back at
encoder states instead of relying on one fixed summary vector.

The RNN itself was the bottleneck, in two ways:

1. **No parallelism.** To compute the state at position *t* you need the state at
   *t−1*. A 1000-word document takes 1000 sequential steps — GPUs, which are
   parallel machines, sit mostly idle. This capped how much data you could train on.
2. **Long path lengths.** Information from word 1 must survive ~1000 recurrent
   updates to influence word 1000. Every step is a chance to forget (vanishing
   gradients — LSTMs help but don't solve it).

The paper's radical move: **delete the RNN, keep only attention**. Hence the title.

## The key ideas (and where we'll implement them)

| Idea | One-line intuition | Our lesson |
|------|--------------------|-----------|
| Scaled dot-product attention | soft, differentiable dictionary lookup: Q asks, K indexes, V answers | 01 |
| Multi-head attention | run h small attentions in parallel so different heads can learn different relations | 02 |
| Position-wise FFN + residuals + LayerNorm | per-token processing; the "think" step after the "gather" step | 03 |
| Sinusoidal positional encoding | attention is order-blind, so position must be injected into the input | 04 |
| Causal masking | the decoder must not see the future | 05 |
| Encoder–decoder assembly | the full 2017 machine | 06 |

Consequences of deleting recurrence:

- **Every pair of positions connects in one step** (path length 1 instead of n) —
  long-range dependencies stop being special.
- **Everything is a matrix multiply over the whole sequence at once** — perfectly
  parallel, perfectly GPU-shaped. *This*, more than any accuracy number, is why
  the transformer took over: it made scale cheap.
- The price: cost grows as **O(n²)** in sequence length, and the model becomes
  **order-blind** without positional information. Both problems spawn whole
  research areas we'll meet later (FlashAttention, sliding windows, RoPE...).

## Study notes: multi-head attention (§3.2.2) — for lesson 02

- The paper's one-liner: *"Multi-head attention allows the model to jointly attend
  to information from different representation subspaces at different positions.
  With a single attention head, averaging inhibits this."* The key word is
  **averaging**: one head produces ONE weight distribution per query, so if a token
  needs to look at two different places for two different reasons, a single head
  must blur them into one mixture.
- The fix is *not* to run h copies of full-width attention (that would be h× the
  compute). Instead each head works in a **projected-down subspace**:
  d_k = d_v = d_model / h (512/8 = 64 in the paper). Total cost stays ≈ the cost
  of one full-width head; §3.2.2 says this explicitly.
- The learned projections W_i^Q, W_i^K, W_i^V are what make heads *different* from
  each other — each head gets its own learned "lens" on the same input. Concat +
  W^O then recombines the h answers into one d_model vector.
- Ablation (Table 3, rows (A)): 1 head is 0.9 BLEU worse than 8; but 32 heads is
  also worse than 8 — more heads means narrower heads (d_k = 16), and each head
  becomes too low-dimensional to compute sharp similarities. Heads are a
  trade-off, not a free lunch.
- Aged well: every modern LLM is multi-head. Aged with modification: MQA/GQA
  (Phase 4) later observed that *keys and values* don't need one set per head —
  only queries do — which slashes inference memory.

## Study notes: masking (§3.2.3) — for lesson 05

- The paper spends one sentence on what became the single most consequential
  detail for modern LLMs: *"We need to prevent leftward information flow in the
  decoder to preserve the auto-regressive property. We implement this inside of
  scaled dot-product attention by masking out (setting to −∞) all values in the
  input of the softmax which correspond to illegal connections."*
- **Why the decoder needs it at all:** the model generates left-to-right, one
  token at a time. If, *during training*, position t could attend to position
  t+1, it would learn to predict the next token by *reading* the next token —
  100% training accuracy, zero generative ability. At generation time the future
  doesn't exist, so a model trained with future access is trained on inputs it
  will never have. The mask makes training-time information flow match
  generation-time information flow.
- **Why this is the efficiency jackpot:** with the causal mask, ONE forward pass
  over an n-token sequence yields n valid next-token training signals at once —
  position 0 predicts token 1, position 1 predicts token 2, ... all in parallel.
  An RNN gets the same n signals but must compute them sequentially. Parallel
  training over all positions of every sequence is *the* reason decoder
  transformers could eat the internet. (GPT keeps exactly this part of the 2017
  machine and drops the rest — Phase 2.)
- **−∞ before softmax, not 0 after:** zeroing weights after softmax would break
  the sum-to-1 property (each query's weights are a probability distribution
  over where to look). Putting −∞ into the *scores* makes softmax itself do the
  renormalization: e^−∞ = 0, and the surviving weights sum to 1. Already built
  into our lesson-01 code as the `mask` argument; this lesson finally uses it.
- **Padding masks** are the unglamorous sibling (the paper barely mentions
  them; every implementation needs them): batches are rectangular tensors,
  sentences aren't. Short sequences are padded with a dummy token, and a mask
  keeps *keys* at pad positions from receiving attention weight — pad is
  storage, not content. Unlike the causal mask (same for every sequence),
  the padding mask is per-example data.
- Aged well: every autoregressive LLM since — GPT-n, LLaMA, Claude, all of
  them — trains behind exactly this triangular mask. Aged with modification:
  the mask's *shape* is where later efficiency work lives (sliding-window =
  band mask, Mistral Phase 5; FlashAttention never materializes the matrix
  at all but respects the same triangle).

## Results that made people care

- New SOTA on WMT14 En→De (28.4 BLEU) and En→Fr — with **an order of magnitude
  less training compute** than previous best models (~3.5 days on 8 GPUs).
- Ablations (Table 3) worth remembering: more heads help up to a point (then hurt);
  bigger models help; dropout matters. d_model=512, 6 layers, 8 heads, d_ff=2048
  became the reference config.

## What aged well / what didn't

- **Aged well:** the core block (attention + FFN + residuals + norm) is unchanged
  in GPT-4/LLaMA/Claude-era models. Genuinely one of the most durable designs in ML.
- **Replaced later:** sinusoidal positions (→ learned, → RoPE), post-norm placement
  (→ pre-norm), the encoder–decoder split itself (GPT keeps only the decoder),
  ReLU in the FFN (→ GELU/SwiGLU), and attention's O(n²) is still being attacked.

## Open questions we carry forward

- Why divide by √d_k exactly? (Derived in lesson 01.)
- Why *multiple* heads rather than one big one? (Lesson 02.)
- Why does the decoder-only variant win for generation? (Phase 2.)
