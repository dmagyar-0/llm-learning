# Training a sequence model — MLE, teacher forcing, Adam

**Papers:** Adam: A Method for Stochastic Optimization — Kingma & Ba 2014
([1412.6980](https://arxiv.org/abs/1412.6980)) ·
Attention Is All You Need — Vaswani et al. 2017, **§5** (the training regime) ·
(context) Williams & Zipser 1989, where "teacher forcing" got its name (RNN era,
no arXiv — the *idea* is what we take).
**Status:** ● studied (Phase 1, lesson 07)

Lessons 01–06 built a function from token ids to logits. This note collects the
literature for the other half of deep learning: the objective we minimize, the
trick that makes one pass yield a whole sequence of training signals, and the
optimizer every LLM since GPT-1 has used in some variant.

## The objective: maximum likelihood, token by token

There is no "translation loss" or "language loss" in the loss function — only
probability. The model defines a distribution over target sequences by the
chain rule of probability, one token at a time:

    p(y | x) = ∏_t p(y_t | y_<t, x)

Training maximizes the likelihood of the data, i.e. minimizes the negative log:

    L = − Σ_t log p(y_t | y_<t, x)

Two things make this the universal choice:

- **It decomposes.** The log turns the product into a sum of per-position
  terms — each position contributes an independent "score the true next token"
  problem, which is exactly the shape our logits tensor already has.
- **It is cross-entropy.** − log p(true token) averaged over tokens is the
  cross-entropy between the data distribution and the model's — and e^loss is
  **perplexity**, the effective branching factor: loss ln(V) = "no idea,
  uniform over V tokens"; loss 0 = "certain and right". Every scaling-law
  plot we'll meet in Phase 3 has this quantity on the y-axis.

## Teacher forcing: n training signals from one pass

The chain rule conditions position t on the *true* prefix y_<t — not on
whatever the model would have generated. Operationally: feed the ground-truth
target, shifted right by one (BOS first), and read a prediction at every
position. That is teacher forcing (the name is from the RNN literature,
Williams & Zipser 1989; the practice is far older — it is just MLE).

The transformer's causal mask (lesson 05) is what makes this *cheap*: all n
conditionals are computed in one parallel forward pass, each provably unable
to peek at its own answer. An RNN had to walk t steps to get signal t;
the transformer gets all of them for one pass. This — not attention quality —
is the training-throughput reason transformers won (papers/attention.md).

The known cost, deferred: at inference the model conditions on its *own*
output, a distribution it never saw during training. Errors can compound
("exposure bias"). We meet this when we build the generation loop (lesson 08),
and the field's heavyweight answer — training directly on model-generated
sequences with RL — in Phase 8.

## Adam (Kingma & Ba 2014)

**The problem.** Plain SGD uses one global learning rate for every parameter.
In a transformer, gradients differ across parameters by orders of magnitude —
a rare token's embedding row gets a large gradient occasionally; a LayerNorm
gain gets small gradients constantly. One learning rate can't serve both.

**The idea: per-parameter step sizes from running statistics.** Keep two
exponential moving averages per parameter:

    m ← β₁ m + (1−β₁) g          # 1st moment: where gradients point on average
    v ← β₂ v + (1−β₂) g²         # 2nd moment: how big they typically are
    step ∝ − lr · m̂ / (√v̂ + ε)   # m̂, v̂ = bias-corrected m, v

Dividing by √v̂ normalizes each coordinate by its own typical gradient
magnitude — every parameter moves at roughly lr per step regardless of how
loud its gradient is; m is momentum, smoothing the noisy minibatch signal.
The bias correction (m̂ = m/(1−β₁ᵗ), same for v̂) fixes the zero-initialized
averages being underestimates for the first ~1/(1−β) steps — without it, the
early v̂ ≈ 0 would make the very first steps enormous.

**Legacy.** Adam (or AdamW, its weight-decay fix — Phase 2) trains
effectively every LLM in the roadmap. Vaswani et al. use Adam with
β₂ = 0.98 (§5.3); GPT-2, LLaMA, Mixtral: AdamW. It costs 2 extra floats per
parameter — one reason optimizer state dominates training memory (3× the
model), which is why checkpoints that can *resume* must save it.

## Vaswani §5 — what the paper does, what we adopt now

| §5 ingredient | The paper | Us, lesson 07 |
|---|---|---|
| Optimizer | Adam, β=(0.9, 0.98), ε=1e−9 | Adam, PyTorch defaults |
| LR schedule | warmup 4k steps, then ∝ 1/√step | constant — see below |
| Label smoothing | 0.1 (§5.4) | none — pure MLE, so loss ≈ perplexity stays exact |
| Batching | ~25k tokens/batch | fresh random toy batches |
| Regularization | dropout 0.1 | none (toy task, no overfitting possible) |

The warmup is not an incidental detail: **post-norm transformers at depth 6+
diverge without it** (the residual highway passes through LayerNorm — see
papers/residuals-layernorm.md "aged badly"). We can skip it only because our
stacks are 2 layers deep. When we go deeper in Phase 2, warmup + pre-norm is
its own story (Xiong et al. 2020).

## Aged well / aged badly

- **Aged well:** the objective. Next-token cross-entropy under teacher
  forcing is, unchanged, the pretraining loss of GPT-4-class models — the
  whole "LLM revolution" is this loss at scale. Adam too: variants differ,
  the m/√v core survives.
- **Aged badly:** label smoothing (helps BLEU, distorts the probabilities —
  modern LLMs need calibrated logits and dropped it); the 1/√step decay
  (replaced by cosine — Phase 2).
- **Still open:** exposure bias. Teacher forcing trains a next-token predictor
  on gold prefixes and hopes it survives its own prefixes at inference. That
  gap is the seed of both sampling tricks (Phase 2) and RL post-training
  (Phases 7–8). Lesson 08 makes it *measurable* — `teacher_forced_accuracy`
  (gold prefix) vs. `free_running_accuracy` (own prefix, via
  `Transformer.greedy_decode`); on the toy tasks the model is near-perfect so
  the gap is tiny, but the two metrics are now in place to watch it widen with
  scale and sequence length.

## What we take into the implementation

- Loss = per-token cross-entropy over **real** positions only — pad tokens
  are storage, not data (lesson 05's promise): mask them out of the average.
- Per-*token* averaging (not per-sequence), so short and long sequences weight
  each token equally and the number is comparable across batches: e^loss is
  always "perplexity per token".
- Adam with default hyperparameters; no schedule, no clipping, no smoothing —
  each arrives in the lesson whose failure mode demands it.
