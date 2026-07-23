# 13 — Weight tying and the GPT-2 init scheme

**Phase:** 2 (GPT-2 recipe) · **Paper(s):** GPT-2 §2.3 (Radford et al. 2019, the init); Press & Wolf 2017 and Inan et al. 2016 (tying the input/output embeddings)
**Code:** `src/llmlab/models/gpt.py` (`tie_weights`, `_init_weights`, `_scale_residual_projections`), `src/llmlab/components/embeddings.py` (`scale_by_sqrt_d_model`) · **Test:** `tests/test_tying_and_init.py`

## The problem

Two loose ends from the last lessons, both about *the parameters themselves* rather
than the architecture:

1. **The model has two big vocab-sized matrices** — the input embedding (lesson 06)
   and the output projection (`lm_head`, lesson 09) — each `(vocab, d_model)`. For
   GPT-2's 50257-vocab, that's ~40M parameters *each*. They do mirror-image jobs
   (id → vector in; vector → id out). Do they need to be different matrices?
2. **We never chose an initialization.** Every `nn.Linear`/`nn.Embedding` has used
   PyTorch's defaults. Lesson 10 left a live warning: pre-norm's residual stream is
   a *sum* over depth, so its scale grows with the number of layers — and nothing
   in our init does anything about that. A deep model would start training from an
   already-inflated stream.

GPT-2 answers both with two small, famous tricks. They're bundled here because they
interact: the init decides the scale of the very matrix that tying makes do double
duty.

## The idea

**Weight tying.** Use *one* matrix as both the input embedding and the output
projection. Row `v` is simultaneously token `v`'s input vector *and* the direction
whose dot-product with the final hidden state scores "next token = `v`". This isn't
just parameter thrift (though folding away a whole `vocab × d_model` matrix is huge
at scale): it says the geometry of "what a token *means* as input" and "what evidence
*predicts* that token as output" should be the same space — which is exactly what you
want, and empirically improves LMs (Press & Wolf 2017).

**The GPT-2 init.** Three rules:

- Every weight (Linear *and* embedding) ~ `N(0, 0.02)`; every bias `0`.
- The two per-block projections that *write into the residual stream* — attention's
  `W_O` and the FFN's second Linear — are additionally scaled by **1/√(2N)**, N =
  number of layers.
- LayerNorm keeps its identity start (γ=1, β=0).

The 1/√(2N) rule is lesson 10's cliffhanger, resolved. And the 0.02 has a beautiful
consequence: a correctly-initialized LM starts with **loss ≈ ln(vocab)**.

## The math

**Why 1/√(2N) exactly.** Pre-norm never renormalizes the highway (lesson 10), so the
stream reaching depth ℓ is `x₀ + Σ (sublayer outputs)`. There are 2N sublayers that
write to it (N blocks × {attention, FFN}). If each writes an output with per-component
variance ≈ σ², and they're roughly independent, the stream's variance after all of
them is ≈ 2N·σ² — it **grows linearly with depth**. Shrink each writing projection's
weights so its output variance is σ²/(2N) instead, i.e. scale the weights by 1/√(2N)
(variance scales with the *square* of the weights). Now 2N outputs sum back to ≈ σ²,
and the stream enters every block at a depth-independent scale. Only the *output*
projections get this — `W_Q/W_K/W_V` and the FFN's input Linear feed the *next
sublayer*, not the residual sum, so they stay at 0.02.

**Why loss ≈ ln(vocab) falls out of tying + small init.** At step 0 a good model knows
nothing, so it should predict ~uniformly: `p ≈ 1/vocab` for every token, giving
cross-entropy `−ln(1/vocab) = ln(vocab)`. That happens iff the logits start *near
zero* (uniform softmax). With tying, `logits = h · Eᵀ` where `E` is the embedding
table (std `s`) and `h` is the final hidden state — and `h` is `ln_f`-normalized, so
its components are ≈ unit variance. Then each logit has variance ≈ `d_model · s²`. At
GPT-2's `s = 0.02`, that's `d_model · 4e-4` — for `d_model=128`, logit std ≈ 0.23,
near-uniform softmax, loss ≈ ln(vocab). The test measures **5.58 vs ln(256)=5.545**.

**A correction worth stating, because it surprised me while building this.** I first
assumed the *√d_model embedding multiply* (lesson 06) was what inflated the starting
loss. It isn't — GPT-2 is pre-norm, so the **first LayerNorm renormalizes the input
embedding before it ever reaches attention**; scaling the input by √d_model is washed
straight out. The lever that actually sets the logit scale is the **embedding table's
init std**, because tying makes that same table the output projection (which sits
*after* `ln_f`, not before a norm). Measured directly:

    embedding table init 0.02   → start loss 5.58   (≈ ln 256) ✓
    embedding table init 0.088  → start loss 9.34   (badly overshoots)

So `scale_by_sqrt_d_model=False` on GPT's embedding is still correct — but for a
*different* reason than I'd have guessed: it balances **content vs. position** at the
input (both tables at 0.02, rather than content ~√d_model louder), a ratio that
LayerNorm *does* preserve. The calibrated loss comes from the small **init**, full
stop.

## The code

**Tying** (`models/gpt.py`), three touches:

```python
self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)   # no bias to tie
self.apply(self._init_weights)          # init runs FIRST (touches lm_head too)
self._scale_residual_projections()
if c.tie_weights:
    self.lm_head.weight = self.embed.table.weight   # now literally ONE Parameter
```

Order matters: init touches every Linear including `lm_head`, then the tie *replaces*
`lm_head.weight` with the embedding table — so whatever init `lm_head` got is
discarded and the head shares the embedding's storage. `nn.Embedding.weight` and
`nn.Linear.weight` are both `(vocab, d_model)`, so the assignment just aliases them:
one tensor, one gradient, updated once per step. (PyTorch de-duplicates shared tensors
in `.parameters()`, so it's not even double-counted.)

**Init** (`models/gpt.py`):

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.02)   # wte and wpe alike

def _scale_residual_projections(self):
    std = 0.02 / math.sqrt(2 * self.config.num_layers)
    for block in self.blocks:
        nn.init.normal_(block.attention.w_o.weight, 0.0, std)  # attn output
        nn.init.normal_(block.ffn.w2.weight,       0.0, std)   # FFN output
```

**Embedding** (`components/embeddings.py`) gains one flag, `scale_by_sqrt_d_model`
(default `True`, so the 2017 encoder–decoder is untouched). GPT builds it `False`.

## What breaks without it

- **No residual scaling, deep model:** the pre-norm stream's variance grows ~2N×, so
  by the last block activations (and the gradients through them) start large; you're
  back to needing warmup/babysitting — the exact thing pre-norm was supposed to buy
  you out of. The `test_deeper_models_scale_..._more` test pins that the scaling
  tracks depth (√2 per doubling).
- **Embedding init too large, tied:** logits start large, softmax starts peaky and
  confidently *wrong*, initial loss overshoots ln(vocab) (9.3 vs 5.5 in the test),
  and the first optimizer steps are wasted just shrinking logits.
- **Tie but keep a head bias:** the bias has no partner in the embedding and quietly
  reintroduces a per-token free parameter GPT-2 doesn't have; we set `bias=False`.
- **Tie before init instead of after:** `_init_weights` would overwrite the shared
  table through the `lm_head` alias, clobbering the embedding init with a second
  0.02 draw. Harmless at 0.02, a real bug at any other head init — so we always tie
  last.

## Open questions

- **Is `1/√(2N)` special or just "some 1/√depth"?** GPT-2 used `1/√N` counting
  residual layers; nanoGPT uses `1/√(2N)` counting residual *paths*. Both are the
  same idea (cancel the depth-sum growth); the constant is empirical. Later recipes
  formalize it — DeepNorm scales residuals *and* init together with derived constants.
- **Tying's downside.** Forcing input and output to share geometry is a constraint;
  at very large scale some models *untie* again (the two roles can specialize). For
  our scale, tied is the right default.
- **We only tie `wte`, never `wpe`.** Position embeddings have no output-side role, so
  they stay independent — consistent with GPT-2.
- **Init and normalization are entangled.** We init LayerNorm to identity and lean on
  it to fix input scale; Phase 4's RMSNorm will change what "identity" even means
  here. Init is never really "done" — it co-evolves with the norm.
- **Next:** with the weights set up GPT-2's way, the model is finally *complete* as an
  architecture. What's left before a real training run is **sampling** (turning greedy
  `generate` into temperature/top-k/top-p) and then pointing lesson 12's BPE at a real
  corpus.
