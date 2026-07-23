# GPT: the decoder-only recipe (Phase 2)

Reading notes for the papers behind Phase 2's architecture. This note covers the
*shape* of the model — GPT-1's decoder-only design and GPT-2's "same thing, bigger"
bet — and why decoder-only won over BERT's encoder-only for generation. The other
Phase 2 ingredients (BPE, weight tying, sampling) get their own notes as we build
them.

## GPT-1 — *Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018)

**The architecture.** Take the 2017 Transformer and throw away half of it: keep only
the **decoder stack**, and inside each decoder block delete the cross-attention
sublayer (there is no encoder to attend to). What remains is the simplest possible
block — *causal self-attention, then FFN* — stacked N times, with a linear head
projecting to the vocabulary. GPT-1 was 12 such blocks, d_model=768 (~117M params).

**The objective.** Plain language modeling: maximize

    L = Σ_t log p(u_t | u_{t−k}, …, u_{t−1}; Θ)

the log-probability of each token given its left context. This is *exactly* lesson
07's masked cross-entropy — but now there is no separate source sequence. The model's
input and its labels are the **same stream, shifted by one position**: predict token
t+1 from tokens ≤ t. The causal mask (lesson 05) is what makes this a valid training
signal — without it, "predict the next token" is trivially solved by reading it.

**The thesis.** Pre-train this LM on a large unlabeled corpus (BooksCorpus), then
*fine-tune* on each downstream task with a small task-specific head. Generative
pre-training learns representations that transfer. This is the "pretrain then adapt"
paradigm that BERT would also ride — and that GPT-2 would then partly abandon.

## GPT-2 — *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019)

**Architecturally, almost nothing new.** GPT-2 is GPT-1's decoder-only stack scaled
up (up to 48 blocks, d_model=1600, ~1.5B params) and trained on a much larger, more
diverse corpus (WebText, ~40GB of scraped links). The only structural tweaks:

- **Pre-norm** — LayerNorm moved to the *input* of each sublayer, plus one extra
  LayerNorm after the final block. This stabilizes training of deep stacks (our next
  lesson).
- A **modified initialization** that scales residual-branch weights by 1/√N to keep
  the residual stream's variance from growing with depth (the weight-tying/init
  lesson).
- A larger vocabulary (50257) via byte-level BPE, and a context window of 1024.

**The bet that mattered: scale → zero-shot.** GPT-1 fine-tuned per task; GPT-2's
claim was that a big enough LM trained on diverse enough text performs tasks it was
*never fine-tuned on*, just by being prompted — translation, summarization, QA — as a
by-product of next-token prediction. "Language models are unsupervised multitask
learners": the internet already contains examples of every task phrased as text, so
an LM good enough to model that text implicitly learns the tasks. This is the seed of
in-context learning that GPT-3 (Phase 3) would make undeniable.

**Why it's our template.** Every modern decoder LLM — GPT-3/4, LLaMA, Mistral,
DeepSeek — is *this* object with components swapped (norm, positions, FFN, attention
variant) and the scale dial turned. Phase 2 builds the GPT-2 recipe piece by piece;
Phases 4–6 are then a sequence of single-component swaps on top of it. That is the
whole reason the repo is organized around interchangeable blocks.

## The road not taken — BERT (Devlin et al., 2018)

Same year as GPT-1, the opposite choice: keep the **encoder** (bidirectional,
all-to-all attention, *no* causal mask) and train with a **masked language model**
objective — blank out ~15% of tokens and predict them from *both* sides of context.
BERT dominated NLP *understanding* benchmarks for years (classification, NER, span QA)
precisely because bidirectionality lets every token see the whole sentence.

But bidirectionality is fatal for **generation**. To generate left-to-right you must
never see the future — that is the causal constraint. A model trained to use both
sides has no natural way to produce text one token at a time. So the field split:

- **encoder-only (BERT):** understanding — read a fixed text, label it.
- **decoder-only (GPT):** generation — extend a text, one token at a time.
- **encoder–decoder (T5, the 2017 original):** transduction — map one text to another.

Decoder-only won the LLM era because *generation subsumes the rest*: once a model can
continue any prompt, classification and QA and translation all become "continue this
prompt appropriately" — no task-specific head required. That realization is GPT-2's
whole point, and it is why we now build the decoder-only path and leave BERT as
📖 reading.

## What we took into the code (lesson 09)

- One vocabulary, one embedding table, one stream (vs. the enc–dec model's two).
- The decoder block reduces to **self-attention + FFN** — which is *already* our
  `TransformerBlock` (lesson 03) once we feed it a causal mask. No new block needed;
  the encoder block and a GPT block are the same object under different masks. That
  equivalence is the lesson.
- The LM objective is lesson 07's masked cross-entropy with `input = tokens[:-1]`,
  `labels = tokens[1:]` — the "shift by one" that replaces the enc–dec toy data's
  pre-split `(tgt_in, tgt_out)`.
- Still 2017's **post-norm** and **sinusoidal** positions here. Turning this generic
  decoder-only LM into *GPT-2 specifically* is the next few lessons: pre-norm →
  learned positions → BPE → weight tying/init → sampling.
