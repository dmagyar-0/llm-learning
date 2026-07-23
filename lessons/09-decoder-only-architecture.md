# 09 — The decoder-only architecture (Phase 2 begins)

**Phase:** 2 (GPT-2 / the decoder-only recipe) · **Paper(s):** GPT-1 (Radford et al. 2018), GPT-2 (Radford et al. 2019); BERT as the road not taken (Devlin et al. 2018) — see `papers/gpt.md`
**Code:** `src/llmlab/models/gpt.py` (new), `src/llmlab/models/__init__.py` · **Test:** `tests/test_gpt.py`

## The problem

Lesson 06 built the full 2017 Transformer: two streams (source + target), an
encoder, a decoder, and cross-attention bridging them. That shape is built for
**transduction** — map one given text to another (translation). But the model that
started the modern LLM line does something simpler and, it turns out, more powerful:
it just **continues text**. No source, no translation — given a prefix, predict what
comes next, forever.

So the question this lesson answers is: *what is the smallest change to lesson 06
that gives us a pure next-token predictor?* And the surprising answer — the reason
this is a small, phone-readable lesson and not a rewrite — is **"delete things."**
GPT is the 2017 Transformer with half of it removed.

## The idea

Take the encoder–decoder and cross out everything that exists to serve a *source*:

- **Delete the encoder.** There is no source text to read.
- **Delete cross-attention.** With no encoder, there is nothing to cross-attend to.
  The decoder block's three sublayers (self-attn → cross-attn → FFN) lose the middle
  one and become just **self-attention → FFN**.
- **Collapse two vocabularies into one.** Source and target were possibly different
  languages, hence two embedding tables. One stream needs one table.

What's left in each block — self-attention + FFN — is *exactly* the block we built
in lesson 03 (`TransformerBlock`, the "encoder" block). The only thing that made
lesson 06's decoder autoregressive was the **causal mask**, and that's a property of
the *mask we pass in*, not of the block. So a GPT block and an encoder block are the
same object; feed it a causal mask and it becomes autoregressive. We reuse the block
verbatim — the plug-in design paying off exactly as promised.

The objective changes shape too, but not in substance. Lesson 07's toy data
pre-split each example into `(tgt_in, tgt_out)` — decoder input and loss labels. With
a single self-predicting stream that split is trivial and lives at the call site:

    input  = tokens[:, :-1]     what the model sees
    labels = tokens[:, 1:]      the next token at each position

Same masked cross-entropy (lesson 07), no source. Predict token *t+1* from tokens
*≤ t*, across the whole stream at once.

## The math

The model factorizes the probability of a sequence by the chain rule and maximizes
its log — identical to lesson 07, minus the conditioning on a source *x*:

    p(u₁ … u_T) = ∏ₜ p(u_t | u_{<t};  Θ)
    L = − Σₜ log p(u_t | u_{<t})        = Σₜ cross_entropy(logits at t−1,  u_t)

The causal mask (lesson 05) is what makes the *t-th* factor depend only on `u_{<t}`.
Without it, "predict `u_t`" is solved by reading `u_t` — zero training loss, a useless
generator. With it, one forward pass over a length-`n` stream produces `n` honest
conditional predictions **in parallel** (position 0 → token 1, position 1 → token 2,
…). That parallelism is why decoder LMs can train on internet-scale corpora; an RNN
gets the same `n` signals only one sequential step at a time.

Shapes, end to end:

    input_ids           (batch, seq)              int64
    embed ×√d + PE      (batch, seq, d_model)
    [self-attn+FFN]×N   (batch, seq, d_model)     mask = causal ∧ padding
    lm_head             (batch, seq, vocab)       one next-token dist per position

## The code

`GPT.forward` is deliberately shorter than the enc–dec `forward`, and the shrinkage
is the lesson:

```python
mask = combine_masks(
    causal_mask(input_ids.shape[1], device=input_ids.device),  # k ≤ q
    padding_mask(input_ids, self.config.pad_id),               # real tokens
)
x = self.embed_dropout(self.positional(self.embed(input_ids)))
for block in self.blocks:
    x = block(x, mask=mask)         # same causal∧padding mask every layer
return self.lm_head(x)
```

- **One mask, one attention per block.** Lesson 06 juggled three mask sites (encoder
  self, decoder self, cross) — the classic home of silent decoder bugs. Here there is
  exactly one, always `causal ∧ padding`. Two of the three bug-prone masks are simply
  gone.
- **`self.blocks` are `TransformerBlock`s** — lesson 03's block, unmodified. A test
  asserts this (`isinstance`), because the equivalence "a GPT block *is* an encoder
  block" is the conceptual payoff, not an implementation accident.
- **Still 2017 inside.** `TokenEmbedding` (×√d, lesson 06), sinusoidal `positional`
  (lesson 04), post-norm block (lesson 03). This is a *generic* decoder-only LM, not
  GPT-2 specifically — see Open questions.

`GPTConfig` carries one `vocab_size` (vs. the enc–dec `src`/`tgt` pair) and defaults
to GPT-2 "small" geometry (d_model=768, 12 heads, 12 layers, ctx 1024). `tiny()` is
~27k params and runs on CPU in seconds.

`generate` is included but is **not a new mechanism** — it's lesson 08's greedy loop
with the encoder removed. There, each step re-ran the decoder against a fixed
`memory`; here there's no memory, so we re-run the whole model over the growing
stream and append the argmax. Cost is still O(n²) (no KV cache until Phase 5, on
purpose). What's genuinely new is the *interface*: hand the model any prefix, get its
continuation — the "continue the prompt" API the enc–dec model never had, and the one
GPT-2 showed subsumes most NLP tasks.

## What breaks without it

- **Drop the causal mask** (pass only padding, or none): the model can attend to
  future tokens, "predicts" them by copying, and training loss collapses to ~0 while
  generation is garbage. The `test_is_causal_end_to_end` test — editing token 4 must
  not move positions 0–3 — fails immediately.
- **Keep bidirectional attention on purpose** and you've rebuilt BERT (encoder-only,
  masked-LM): great at *understanding* a fixed text, structurally unable to *generate*
  left-to-right. That fork (`papers/gpt.md`) is why the field split into encoder-only,
  decoder-only, and encoder–decoder — and why decoder-only won the LLM era: generation
  subsumes the other tasks once the model can continue any prompt.
- **Forget the shift-by-one** (train on `input == labels`): the model learns the
  identity map, not language. The shift is the entire supervision signal.

## Open questions

This is the decoder-only *skeleton*. Turning it into GPT-2 *specifically* is the next
few lessons, each a single component swap on this exact model:

1. **Pre-norm** — move LayerNorm to the input of each sublayer (+ a final norm), so
   the residual highway stays a clean identity through depth. This is the one-line
   `norm_placement` knob lesson 03 foreshadowed, and a big part of why deep stacks
   train without heroic warmup.
2. **Learned positional embeddings** replace the sinusoidal table — the *same* table,
   but now a `nn.Parameter` (lesson 04's own foreshadowing).
3. **BPE tokenizer**, written ourselves — so far `vocab_size` has been a toy integer;
   real GPT-2 uses byte-level BPE with 50257 tokens.
4. **Weight tying + GPT-2 init** — share `lm_head.weight` with the embedding table
   (kept untied here, as lesson 06 left it), and scale residual-branch weights by
   1/√N.
5. **Sampling** — temperature, top-k, top-p — to turn `generate`'s greedy argmax into
   real sampling.
6. Then a real **training run**: char-level TinyShakespeare on CPU, then BPE + small
   GPT-2 on the home GPU.

The bet the design is making: each of those is genuinely *one swap* on the object
built today. If a later lesson needs to fork `gpt.py` instead of adding a config knob
or a registered component, that's a signal the abstraction leaked — worth noticing.
