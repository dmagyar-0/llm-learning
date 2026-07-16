# Residual connections & Layer Normalization — the plumbing that makes depth work

**Papers:** Deep Residual Learning — He et al. 2015 ([1512.03385](https://arxiv.org/abs/1512.03385)) ·
Layer Normalization — Ba, Kiros & Hinton 2016 ([1607.06450](https://arxiv.org/abs/1607.06450))
**Status:** ● studied (Phase 1, lesson 03) · Also covers Vaswani et al. 2017 §3.3 (the FFN) and §5.4 (where the norm goes).

Neither paper is about transformers — ResNet is a vision paper, LayerNorm an
RNN paper. But the 2017 transformer is untrainable without both, and they are
in every LLM since, unchanged (LayerNorm) or nearly so (→ RMSNorm, Phase 4).

## Deep Residual Learning (He et al. 2015)

**The problem — degradation, not overfitting.** Before 2015: stacking more
layers past ~20 made networks worse *on the training set*. Not overfitting —
the deeper network couldn't even fit the data as well as the shallow one,
despite strictly greater expressive power. A 56-layer net had higher training
error than a 20-layer one. Something about *optimization* fails with depth.

**The diagnosis.** A deeper network could trivially match a shallower one:
copy the shallow layers, make the extra ones identity. So the deep network's
poor training means SGD **cannot easily learn the identity function** through
a stack of nonlinear layers — the parametrization fights it.

**The fix — reformulate what a layer learns.** Instead of asking layer F to
produce the new representation, ask it to produce the *change*:

    output = x + F(x)        # F learns the residual: "what to add"

Identity is now the easy default — F just outputs ≈ 0 (which small random
init nearly does for free). Layers "earn their keep": each one starts as a
small perturbation of a working signal and learns how to usefully modify it.

**The gradient view (why we care most).** Backprop through `x + F(x)` gives
`∂out/∂x = I + ∂F/∂x` — the gradient always has a direct `I` path. Through L
blocks the gradient is a *sum* containing an unattenuated term, not a *product*
of L Jacobians that shrinks or explodes geometrically. This is the same
disease that killed RNNs over *time* (papers/attention.md), reappearing over
*depth* — and the same medicine: give the signal a highway.

**Legacy in LLMs.** Every transformer layer is two residual additions. The
modern framing: the residual path is the **stream** — a shared workspace
flowing untouched from embeddings to logits — and attention/FFN blocks are
devices that *read from* it and *write (add) into* it. 96-layer GPT-class
models train only because of this.

## Layer Normalization (Ba, Kiros & Hinton 2016)

**The problem.** Deep nets train fastest when each layer's inputs stay in a
stable range; otherwise every layer must constantly re-adapt to the shifting
output distribution of the layer below (plus saturation and LR sensitivity).
BatchNorm (2015) fixed this in vision by normalizing each feature across the
*batch* — but that fails where transformers live: it ties examples in a batch
together, behaves differently train vs. test, and is awkward for variable-
length sequences and small batches.

**The idea — rotate the axis.** Normalize each **token's vector across its
own features** instead:

    LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β      μ, σ² over the d_model axis

Every token is normalized *by itself*: no dependence on batch neighbors, no
train/test mismatch, works at batch = 1, works at any sequence length. The
learned per-feature γ (gain) and β (bias) restore expressiveness: the network
can undo or re-scale the normalization *if it wants to* — LN removes only the
*accident* of scale, not the model's ability to use scale.

**Why transformers need it.** Residual streams are sums of many block
outputs; without normalization the stream's magnitude drifts layer by layer,
and attention is sensitive to input scale (lesson 01: score variance →
softmax saturation → dead gradients). LN before each sublayer's consumer
resets the scale to a known operating point.

## Where they meet: the sublayer formula (Vaswani §3.3, §5.4)

The 2017 paper composes every sublayer (attention or FFN) as:

    x = LayerNorm(x + Sublayer(x))          # "post-norm" — norm AFTER the add

and the FFN itself is two linear maps with a ReLU, applied to each position
independently, with an inner expansion d_ff = 4·d_model:

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂      # (d_model → 4·d_model → d_model)

Attention is the only place tokens exchange information; the FFN is where
each token *processes* what it gathered — per-token, position-wise, weights
shared across positions. It holds ~2/3 of a transformer layer's parameters,
and later interpretability work (Geva et al. 2021: "FFN layers are key-value
memories") reads it as where factual associations are stored.

## Aged well / aged badly

- **Aged well:** residual + norm + FFN as the block skeleton — unchanged from
  2017 to today. The 4× expansion survives too (SwiGLU models use 8/3× to
  keep parameter count equal — Phase 4).
- **Aged badly: the post-norm placement.** Putting LN *after* the addition
  means the residual highway itself passes through LN — the identity path is
  no longer clean, and deep post-norm transformers need a careful learning-
  rate warmup to train at all. GPT-2 (Phase 2) moves the norm *inside* the
  residual branch ("pre-norm"): `x = x + Sublayer(LN(x))` — restoring the
  untouched highway. We implement the 2017 paper faithfully first and will
  feel the difference when we get there.
- ReLU in the FFN → GELU (GPT-2) → SwiGLU (LLaMA); LayerNorm → RMSNorm
  (LLaMA): each is one plug-in swap in Phase 4.

## Open questions we carry forward

- Why exactly does pre-norm stabilize depth while post-norm needs warmup?
  (Lesson: Phase 2; paper: Xiong et al. 2020, "On Layer Normalization in the
  Transformer Architecture".)
- Does the mean-centering in LN even matter, or is the rescaling doing all
  the work? (RMSNorm's bet: drop the mean — Phase 4.)
- If the FFN is 2/3 of the parameters, can we activate only part of it per
  token? (Mixture of Experts — Phase 6.)
