# 08 — Autoregressive generation: greedy decoding and exposure bias

**Phase:** 1 (The Transformer) · closes Phase 1 · **Papers:** Vaswani 2017 §3.1 (the decoder is autoregressive), and the exposure-bias lineage — Ranzato et al. 2015 "scheduled sampling" — see `papers/training.md`
**Code:** `src/llmlab/models/transformer.py` (`greedy_decode`), `src/llmlab/training/loop.py` (`free_running_accuracy`) · **Test:** `pytest tests/test_generation.py -v` · **Demo:** `python -m llmlab.training.loop` (trains reverse, then *generates* it)

## The problem

Lesson 07 trained the model and measured it with *teacher-forced* accuracy:
at every position we handed the model the **true** previous tokens and asked
only "is the next one right?". That is exactly how training works — one
parallel, causally-masked pass over the gold sequence (lesson 05's payoff).

But no such gold sequence exists at deployment. When you actually *use* the
model there is nothing to copy the prefix from except the model's own previous
outputs. So the one pass has to become a **loop**: predict a token, append it,
feed the longer sequence back in, predict the next. Turning the next-token
*scorer* we trained into a next-token *generator* is the whole of lesson 08 —
and it surfaces a gap that teacher forcing structurally cannot show.

## The idea

**Generation is `forward`, argmax, append — repeated.** No new math. The
decoder already maps a prefix to a next-token distribution; generation just
picks a token from the last position, glues it on, and calls the decoder
again on the now-one-longer prefix:

    gen = [BOS]                                  # the seed (lesson 07)
    repeat:
        logits = decode(gen, memory, src)        # (batch, len, vocab)
        next   = argmax(logits[:, -1])           # greedy: last position only
        gen    = concat(gen, next)               # feed my own output back in
        stop when next == EOS

Two details make it real code:

- **Encode the source once.** The source never changes during generation, so
  `encode(src)` runs a single time and its `memory` is reused every step. This
  is the entire reason the architecture splits into encoder/decoder: read the
  input once, generate against it many times.
- **A per-row "finished" latch.** In a batch, rows hit EOS at different steps.
  Once a row emits EOS we freeze it to PAD forever, so the other rows can keep
  going and the batch finishes together. Without the latch a "done" row rambles
  past its own stop signal.

**Greedy is the simplest decision rule** — always take the argmax. It commits
hard to one token per step with no way to reconsider, which is *why* it is the
right first decoder: it exposes exposure bias most sharply. Temperature, top-k,
top-p sampling and beam search (which hedge across multiple continuations) are
Phase 2, once we have real text where greedy's determinism becomes a liability.

**EOS is now load-bearing.** Lesson 07 put EOS in the labels and it looked like
bookkeeping. Here it is the *only* thing that ends the loop: a model never
taught to predict EOS would generate until it hit `max_new_tokens`, every time.
The demo shows it working — generations end in `2` right after the last real
token.

## Exposure bias — the gap teacher forcing hides

Two accuracies, same trained model:

    teacher-forced : given the TRUE prefix, is the next token right?   (per-token)
    free-running   : left alone on its OWN prefix, is the whole
                     sequence right?                                   (exact-match)

They differ because the model was **only ever trained on gold prefixes**. At
inference it conditions on its own outputs, which — the instant it makes one
mistake — are slightly *out of the distribution it trained on*. That off-
distribution prefix makes the next error a little likelier, which pushes the
prefix further off, and errors **compound**. This is **exposure bias**: the
train/inference mismatch between gold and self-generated context.

Why exact-match (not per-token) for free-running: a generator that gets 7 of 8
tokens right produced the **wrong sequence**. Partial credit is a teacher-
forcing luxury — it can score position 5 correctly because *you* fixed
positions 0–4. Free-running gets no such help, so we grade the whole row.

On our toy copy/reverse tasks the model solves the problem so completely that
both accuracies read ~100% (the demo: teacher-forced *and* free-running both
100% on reverse) — the gap is real but tiny when the model is near-perfect. It
widens exactly where it will bite us later: harder tasks, longer sequences
(more steps to compound over), and under-trained models. Watching the gap is a
Phase-2-onward habit; lesson 08's job is to build the loop that makes the gap
*visible at all*.

## The cost: O(n²), and the fix we're deferring

The naive loop recomputes the decoder over the **entire prefix every step**:
step *t* redoes all the work of steps `0..t−1`. Generating *n* tokens therefore
costs on the order of *n²* decoder passes — quadratic in the output length,
before you even count attention's own per-pass cost. Almost all of that work is
redundant: the keys and values for tokens already generated do not change when
you append a new one. Caching them so each step is O(1) new work — the **KV
cache** — is the single most important inference optimization in production
LLMs, and it gets its own lesson in **Phase 5**. We write the wasteful version
first, on purpose, so we can later measure exactly what the cache buys.

## The code

- `models/transformer.py::greedy_decode(src_ids, bos_id, eos_id, max_new_tokens)`
  — encode once, seed with BOS, loop: decode → argmax last position → mask
  finished rows to PAD → append → break when all rows have emitted EOS.
  `bos_id`/`eos_id` are **arguments**, not model attributes, because they belong
  to the *task's* vocabulary (lesson 07's specials), whereas `pad_id` is the
  model's — it shaped every mask. `@torch.no_grad()` + `.eval()`: generation
  builds no graph and must be deterministic.
- `training/loop.py::free_running_accuracy` — the honest metric: greedy-decode a
  batch, strip the BOS seed, pad hyp and reference to a common width (a *short*
  generation is a *wrong* generation, not a crash), and take whole-row exact
  match. Sits right next to `teacher_forced_accuracy` so the contrast is one
  screen.
- The `__main__` demo now prints **both** accuracies and three real
  generations, so you can watch BOS go in and an EOS-terminated sequence come
  out.

## What breaks without it

- **No EOS in training labels (lesson 07's warning, cashed):** teacher-forced
  accuracy is still perfect, but the generator never emits a stop token and
  runs to `max_new_tokens` every time. The failure is invisible until you
  actually sample — which is *now*.
- **No finished-latch:** a row that emitted EOS keeps predicting content;
  batched generation can never agree on when it's done, and EOS stops meaning
  "stop".
- **Re-encoding the source each step:** correct but wasteful — the encoder is
  the expensive half and its output is constant during decoding.
- **Reporting only teacher-forced accuracy:** you ship a model that scores 99%
  in the notebook and derails on its own third token in production. The whole
  point of `free_running_accuracy` is to refuse that self-deception.

## Verified claims (tests)

- `greedy_decode` seeds every row with BOS and grows by at most
  `max_new_tokens`; rows are frozen to PAD after their first EOS; identical
  inputs give byte-identical outputs (determinism).
- One greedy step equals the teacher-forced argmax on the *same* prefix —
  proving generation adds no new math, only a new *source* for the prefix
  (itself).
- End to end: a model trained to solve copy also **generates** copy correctly
  (free-running exact-match > 0.85 alongside teacher-forced > 0.95), and > 95%
  of generations terminate with EOS within budget.

## Open questions (→ future lessons)

- **Better decoding (Phase 2):** greedy is myopic — locally-best tokens can
  strand you in globally-bad sequences. Temperature/top-k/top-p reintroduce
  controlled randomness; beam search keeps several hypotheses alive. All need
  real text to matter.
- **Closing the exposure-bias gap:** scheduled sampling (train on a mix of gold
  and model tokens), sequence-level objectives, and ultimately **RL on model-
  generated sequences** (Phases 7–8) — the field's escalating answers to the
  same train/inference mismatch we just made visible.
- **Making it fast (Phase 5):** the KV cache turns this O(n²) loop into O(n);
  everything about production inference latency starts here.

---

**Phase 1 is complete.** We have, from scratch and fully explained: scaled
dot-product attention → multi-head attention → FFN/residual/LayerNorm →
sinusoidal positions → causal & padding masks → the full encoder–decoder →
the training loop → **generation**. A tiny transformer now trains on CPU in
seconds and *writes its own output back to itself*. Phase 2 rebuilds this as a
decoder-only GPT-2 on real text — and every mechanism above carries over as a
plug-in component, exactly as the repo set out to prove.
