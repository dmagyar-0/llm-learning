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
