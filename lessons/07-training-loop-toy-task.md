# 07 — The training loop: masked loss, teacher forcing, a toy task learned

**Phase:** 1 (The Transformer) · **Papers:** Vaswani 2017 §5 (training regime), Adam — Kingma & Ba 2014 — see `papers/training.md`
**Code:** `src/llmlab/data/toy.py`, `src/llmlab/training/loss.py`, `src/llmlab/training/loop.py` · **Test:** `pytest tests/test_training.py -v` · **Demo:** `python -m llmlab.training.loop` (watch it learn to reverse, ~1 min on CPU)

## The problem

Lesson 06 finished the machine but never turned it on: the model maps token
ids to logits, and the logits are noise. Three gaps stand between "a function
with 55k random parameters" and "a model that does something":

1. **An objective.** What single number should go down? And what do we do
   about lesson 05's IOU — pad positions must be "excluded where it matters:
   in the loss"?
2. **A training signal layout.** `forward` was *described* as teacher forcing
   in lesson 06; now we have to actually build the shifted input/label pair
   and get the off-by-one right.
3. **Data where bugs can't hide.** Real text is statistically murky — a
   half-broken model still gets *somewhere* on English. We want a task whose
   answer key we know, so "it learned" vs. "it's broken" is unambiguous.

## The idea

**The objective is probability, nothing else.** The model assigns each target
sequence a probability via the chain rule — p(y|x) = ∏ₜ p(yₜ | y<ₜ, x) — and
training just maximizes the probability of the data (minimizes −log of it).
The log turns the product into a **sum of per-position cross-entropies**:
every position is an independent "score the true next token" problem. This
exact loss, unchanged, trains every LLM in the roadmap.

**Teacher forcing is a one-token shift.** One example with content y₁..y_L
becomes input/label rows offset by one:

    decoder input  tgt_in  = [BOS, y₁, ..., y_L]     what the model sees
    loss labels    tgt_out = [y₁, ..., y_L, EOS]     what it must predict

Position t sees the *true* prefix through the causal mask and is scored on
token t+1 — n honest next-token problems from ONE parallel pass (this
cheapness is lesson 05's payoff, and why transformers train fast). BOS gives
position 0 something to condition on; EOS is a real vocabulary item the model
must learn to *predict* after the last token — otherwise a generator
(lesson 08) never knows when to stop.

**The toy tasks: copy and reverse.** Random token sequences (length 2–8,
drawn fresh every batch), target = the same sequence copied, or reversed.
Random tokens have no statistics to memorize, so any accuracy is pure
information plumbing through cross-attention. The two tasks differ nicely:
*copy* is pure positional alignment (target position t reads source position
t); *reverse* needs the sequence's **length** — target position 0 must find
source position L−1, and L varies per example, discoverable only through the
padding mask. Same model, same data, measurably harder (demo: copy solved in
~300 steps, reverse ~2× more).

**The loop is five lines** — forward, loss, `zero_grad`, `backward`,
`step` — plus **Adam**, which keeps per-parameter running averages of the
gradient (m) and its square (v) and steps each coordinate by ≈ lr·m̂/√v̂, so
every parameter moves at a sensible rate no matter how loud its gradient is
(embedding rows vs. LayerNorm gains differ by orders of magnitude; one global
SGD rate can't serve both — `papers/training.md`).

## The math

The masked loss, in the three lines the code mirrors:

    log_probs = log_softmax(logits)                  # (batch, seq, vocab)
    nll  = −log_probs[true token at each position]   # (batch, seq)
    loss = Σ(nll · real) / Σ(real)                   # real = (labels ≠ PAD)

Two derivations worth owning:

- **The ln V anchor.** An untrained model's logits are near-uniform noise →
  softmax ≈ uniform → −log(1/V) = ln V. With V=16: ln 16 ≈ 2.77, and the
  demo's step-0 loss is 2.88. This is the cheapest debugging instrument in
  deep learning: step-0 loss far from ln V means something upstream of
  learning is broken (shapes silently broadcasting, labels misaligned).
  e^loss is **perplexity** — the model's effective branching factor — and is
  the y-axis of every scaling-law plot we'll meet in Phase 3.
- **Why divide by token count, not batch size.** MLE says each observed
  token is one datum. A per-sequence mean would make token 3-of-3 worth more
  than token 3-of-8; a per-token mean weights them equally AND makes the
  number comparable across any batch shape — "loss per token" always means
  the same thing.

Masking does the rest: pad-label positions are multiplied by 0 in the sum, so
their logits get **exactly zero gradient** — not small, zero (a test pins
this). Lesson 05's promise, kept: the network may compute garbage over
padding, and none of it can move a single weight.

## The code

- `data/toy.py` — `ToyTaskConfig` + `make_batch`: fresh random batches with
  ragged lengths (so padding is genuinely exercised), specials
  `PAD=0, BOS=1, EOS=2`, and the shift built exactly once. Data generated on
  the fly: nothing to download, nothing repeats — so **train loss IS
  generalization**; memorization is impossible. The fiddly bit is reverse +
  padding: `flip` sends the PAD tail to the front, a stable argsort on
  "is pad" re-left-aligns each row.
- `training/loss.py` — `masked_cross_entropy`, written out (gather the true
  token's log-prob, mask, per-token mean) rather than calling
  `F.cross_entropy(ignore_index=...)` — but a test pins that they agree to
  1e-6, so ours *is* the standard loss, not an approximation. `log_softmax`
  is used (not `softmax().log()`) because it computes via logsumexp and
  can't overflow.
- `training/loop.py` — `train_toy` (the five lines + Adam + loss history),
  `teacher_forced_accuracy`, and a `__main__` demo. Decisions worth
  remembering:
  - **Constant lr=1e-3, no warmup** — legitimate *only* because our post-norm
    stacks are 2 layers deep; the paper's 6-layer model diverges without
    warmup (`papers/residuals-layernorm.md`, "aged badly"). We will hit that
    wall deliberately in Phase 2.
  - **`zero_grad` exists because PyTorch accumulates** gradients on purpose
    (that's how gradient accumulation across micro-batches works — we'll
    want it on the home GPU). Forgetting it sums gradients across steps and
    the loss *still falls at first* — a classic silent bug.
  - **"Teacher-forced accuracy" is named honestly:** position t predicts
    while seeing the true prefix. It measures per-step prediction, not
    free-running generation — that gap (exposure bias) is lesson 08's topic.

Demo output worth staring at: the model predicts EOS at a *pad* position
(`want [..., 2, 0]`, `got [..., 2, 2]`). Not a bug — that position's label is
PAD, its gradient was always zero, so its output is unconstrained garbage.
Free garbage in ignored positions is exactly what lesson 05 said we'd accept.

## What breaks without it

- **Shift off by one** (input = labels): every position can see its own
  answer; loss crashes to ~0 immediately, model is useless. The causal mask
  can't save you from handing the answer to the *input*.
- **No loss masking:** the model is graded on predicting PAD after PAD — with
  our length spread that's harmless-looking but at real-corpus pad ratios the
  model learns "when unsure, say PAD" and pads leak into generation. (It also
  distorts loss values: no more comparing to ln V or across batch shapes.)
- **No EOS in the labels:** teacher-forced accuracy still looks perfect, but
  lesson 08's generator will never emit a stop signal — the failure is
  invisible until you sample.
- **SGD instead of Adam at this lr:** the loss falls ~10× slower; parameters
  with quiet gradients (norms' gains) barely move. Try it — one-line change.
- **Wrong step-0 loss** (≠ ln V): the anchor catches upstream wiring bugs
  before any training time is spent.

## Verified claims (tests)

- The generated batch has the exact teacher-forcing layout: BOS-prefixed
  input, one-token shift, EOS at position L, PAD beyond — and reverse rows
  are the source content backwards, re-left-aligned.
- Uniform logits cost exactly ln V; our three-line loss equals PyTorch's
  `ignore_index` cross-entropy to 1e-6.
- d loss/d logits is *exactly zero* at every pad-label position, nonzero at
  every real one.
- End to end: step-0 loss ≈ ln 16, and 300 Adam steps solve the copy task
  (final loss < 0.15, teacher-forced accuracy > 95%) — on CPU, in seconds,
  on never-repeated data.

## Open questions (→ future lessons)

- **Generation** (lesson 08): the model can predict token t+1 from a true
  prefix — but at inference there is no true prefix, only its own output.
  Build the autoregressive loop (greedy first), watch teacher-forced and
  free-running accuracy diverge, and see why generation is O(n²)-per-token
  naive (→ KV cache, Phase 5).
- **Exposure bias:** training on gold prefixes, generating on own prefixes —
  the mismatch behind many failure modes; the field's answers range from
  sampling tricks (Phase 2) to RL on model-generated sequences (Phase 8).
- **The missing §5 machinery:** LR warmup + decay, label smoothing, gradient
  clipping, and checkpoint/resume — each deferred until the lesson whose
  model actually needs it (warmup arrives with depth, Phase 2; checkpointing
  with real training runs).
- The loss curve occasionally *spikes* (step ~600 of the demo: 0.006 → 0.22
  → recovers). Fresh-batch noise plus Adam's momentum — worth revisiting
  when we meet gradient clipping and loss-spike lore in real runs.
