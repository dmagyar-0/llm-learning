# 10 — Pre-norm: moving LayerNorm inside the residual

**Phase:** 2 (GPT-2 recipe) · **Paper(s):** GPT-2 §2.3 (Radford et al. 2019); *On Layer Normalization in the Transformer Architecture* (Xiong et al. 2020) — the formal why
**Code:** `src/llmlab/components/block.py` (`norm_placement` knob), `src/llmlab/models/gpt.py` (`norm_final`) · **Test:** `tests/test_pre_norm.py`

## The problem

Lesson 09 gave us a decoder-only LM, but still built from lesson 03's **post-norm**
block:

    post-norm:   x = LayerNorm(x + Sublayer(x))

This is faithful to 2017, and it trains — but deep post-norm stacks are famously
finicky. They need a **learning-rate warmup** (start near zero, ramp up over thousands
of steps) or they diverge early in training, and the deeper the stack the worse it
gets. GPT-2 wanted to go deep (up to 48 layers) and train robustly, so it made one
small change that every model since has kept. That change is this lesson.

## The idea

Move the LayerNorm from *after* the residual add to *before* the sublayer:

    post-norm:   x = LayerNorm(x + Sublayer(x))     ← LN on the highway
    pre-norm:    x = x + Sublayer(LayerNorm(x))     ← LN inside the branch

Look at where the raw `x` goes. In pre-norm, the original `x` is added back
**untouched** — the residual highway from lesson 03 is now a pure running sum of
sublayer outputs, never renormalized. LayerNorm still happens, but only on the *copy*
that feeds the sublayer, not on the through-line.

That's the whole code change. Its two consequences are the lesson.

## The math

**Why the highway matters — the gradient.** Write one block's output as a function of
its input. Pre-norm:

    x_{l+1} = x_l + F(LN(x_l))
    ∂x_{l+1}/∂x_l = I + ∂F(LN(x_l))/∂x_l

The **I** is an exact identity path. Compose L blocks and the Jacobian
∂x_L/∂x_l = ∏_k (I + J_k) expands to `I + (∑ J_k) + (higher-order terms)` — it always
contains an **un-attenuated identity term**. Gradient flows from the loss to layer 0
without being multiplied down.

Post-norm:

    x_{l+1} = LN(x_l + F(x_l))
    ∂x_{l+1}/∂x_l = LN'_l · (I + ∂F/∂x_l)

Now a LayerNorm Jacobian `LN'` sits in *front of the whole thing*. Through L layers,
∂x_L/∂x_l = ∏_k LN'_k (I + J_k) — even the "identity" path is multiplied by **L**
LayerNorm Jacobians. `LN'` projects out the mean/scale directions and rescales by
1/σ, so this product distorts and (near the output, at init) inflates the gradient.
Xiong et al. 2020 made this precise: at initialization, post-LN gradients near the
final layer scale like √(model depth), which is exactly what warmup exists to tame;
pre-LN gradients are well-behaved with no warmup. Same reasoning as lesson 03's
residual highway — post-norm partially blocks the highway with an LN at every step;
pre-norm keeps it clear.

**The cost of the clear highway — the stream grows.** Because pre-norm never
renormalizes the through-line, the residual stream is a sum of ~2L sublayer outputs
and its magnitude *grows with depth*. `tests/test_pre_norm.py` measures it directly
through 24 bare blocks:

    post-norm:  ‖stream‖ entry→exit  ≈ 1.0×   (renormalized every block)
    pre-norm:   ‖stream‖ entry→exit  ≈ 2×     (running sum, unbounded)

That unbounded stream would hit the output head at a strange, depth-dependent scale —
so a pre-norm *model* must **renormalize once at the very end**, before the LM head.
GPT-2 calls this final LayerNorm `ln_f`. Post-norm needs none: its last block already
ended in an LN.

## The code

Two touches, both small.

**The block gets a knob** (`components/block.py`). Same two LayerNorms — only *where*
they apply moves:

```python
if self.norm_placement == "post":
    attn_out, _ = self.attention(x, mask=mask)
    x = self.norm_attn(x + self.dropout(attn_out))   # add THEN norm
    ffn_out = self.ffn(x)
    x = self.norm_ffn(x + self.dropout(ffn_out))
else:  # "pre"
    attn_out, _ = self.attention(self.norm_attn(x), mask=mask)  # norm THEN sublayer
    x = x + self.dropout(attn_out)                   # raw x added back, untouched
    ffn_out = self.ffn(self.norm_ffn(x))
    x = x + self.dropout(ffn_out)
```

An unknown string raises at construction — a silent wrong-architecture bug is the
worst kind (it still trains, just worse), so we fail loud.

**The model adds the closing norm** (`models/gpt.py`). One line in `__init__`, one in
`forward`:

```python
self.norm_final = LayerNorm(d_model) if norm_placement == "pre" else nn.Identity()
...
x = self.norm_final(x)      # ln_f (pre) or identity (post)
return self.lm_head(x)
```

Using `nn.Identity` for the post-norm case keeps `forward` branch-free — the
placement decision is made once, at build time. `GPTConfig.norm_placement` now
defaults to `"pre"`: our GPT is one step closer to being GPT-2 specifically. The 2017
encoder–decoder `Transformer` keeps `"post"` — it stays faithful to its paper. Both
live in the *same* `TransformerBlock`; the difference is a config string, not a forked
file. That is the design principle paying rent again.

## What breaks without it

- **Deep post-norm, no warmup:** gradients near the output are too large at init;
  the first steps blow up the weights and the run diverges. Pre-norm removes the need
  for that babysitting — the reason it became universal.
- **Pre-norm but forget `ln_f`:** the head receives a stream whose scale grew with
  depth (the ~2× test). Training still limps along, but you've handed the head a
  moving target; every published pre-norm model includes the final norm.
- **Silent typo in the knob** (`"prenorm"`): without the validation check you'd build
  a post-norm model while believing it's pre-norm, and only notice as a mysteriously
  worse loss curve. Hence the loud `ValueError`.

## Open questions

- **Post-norm sometimes wins the *final* loss.** When it can be trained at all
  (careful warmup, not-too-deep), post-norm occasionally reaches slightly lower loss
  than pre-norm — pre-norm's clean highway can make later layers do less work
  ("representation collapse" in very deep pre-norm nets). Later architectures chase
  both: sandwich-norm, and DeepNorm's residual scaling. We may revisit when we care
  about squeezing a training run.
- **RMSNorm is next-door.** Phase 4's LLaMA keeps pre-norm placement but swaps the
  LayerNorm itself for RMSNorm (drop the mean-centering). Placement (this lesson) and
  the norm *function* (Phase 4) are independent knobs — which is why they're separate
  lessons.
- **Still sinusoidal positions here.** The next specialization toward GPT-2 is
  swapping the fixed sinusoidal table (lesson 04) for a *learned* positional
  embedding — same add, but the table becomes a trained parameter.
