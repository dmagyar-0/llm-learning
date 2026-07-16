# 03 — Position-wise FFN, residual connections, LayerNorm

**Phase:** 1 (The Transformer) · **Papers:** Vaswani 2017 §3.3/§5.4 · He 2015 (ResNet) · Ba 2016 (LayerNorm) — see `papers/residuals-layernorm.md`
**Code:** `src/llmlab/components/{ffn,norms,block}.py` · **Test:** `pytest tests/test_ffn_norms_block.py -v`

## The problem

Lessons 01–02 built the part of the transformer that *moves information
between tokens*. Three things are still missing before blocks can stack into
a deep model:

1. Attention only computes **weighted averages** of value vectors — nothing
   nonlinear ever happens to the content itself. Where does each token
   *process* what it gathered?
2. Deep stacks of layers famously **fail to train** — not overfitting, but
   *worse training error* with more depth (the ResNet "degradation" finding:
   a 56-layer net trained worse than a 20-layer one).
3. Sums of many layer outputs **drift in scale**, and lesson 01 taught us
   attention is scale-sensitive (big inputs → saturated softmax → dead
   gradients). Something must hold the representation at a stable magnitude.

One answer each: the FFN, residual connections, LayerNorm.

## The ideas

**FFN — "attention gathers, the FFN thinks."**

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂        d_model → 4·d_model → d_model

The same 2-layer MLP applied to every position *independently* (weights
shared, no cross-token contact — test-verified: feeding the sequence at once
equals feeding tokens one by one). Expand 4×, ReLU in the wide space, come
back. Without the ReLU, W₂W₁ collapses into one matrix and the width buys
nothing. The FFN holds **2/3 of the layer's parameters** (2·4·d² vs.
attention's 4·d²) — remember that when Phase 6 asks what MoE should sparsify.

**Residuals — layers learn *changes*, not replacements.**

    output = x + F(x)

ResNet's diagnosis: SGD can't easily learn the identity through a stack of
nonlinear layers, so extra depth *hurt*. Reformulate: let F learn what to
*add*. Identity becomes the easy default (F ≈ 0 at init). The gradient view
is the payoff: ∂(x+F(x))/∂x = **I** + ∂F/∂x — backprop always has a direct
path; through L blocks the gradient has an unattenuated term instead of a
product of L Jacobians. Same disease that killed RNNs over *time*, cured
over *depth*, the same way: a highway.

This also births the **residual stream** picture: since blocks only *add*,
the d_model vector is a shared workspace — attention writes what it gathered,
the FFN writes what it computed, later layers read everything. And it explains
a lesson-02 design constraint: sublayers must map d_model → d_model or the
addition wouldn't typecheck (that's why W_O and the FFN's contraction exist).

**LayerNorm — reset each token to a known scale.**

    LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β        stats over the FEATURE axis

Each token normalizes *itself* — no dependence on batch neighbors (that's
BatchNorm, and it's exactly wrong for variable-length, batch-of-1 sequence
work; a test shows token independence directly). ε prevents 0/0 on constant
tokens. Learned γ, β give back what normalization takes: the network keeps
the *choice* of scale while losing the *accident* of scale.

**The block (post-norm, as 2017 shipped it):**

    x = LayerNorm(x + Attention(x))      # gather, add, reset scale
    x = LayerNorm(x + FFN(x))            # think,  add, reset scale

Shape in == shape out → depth becomes a pure hyperparameter (N=6 in the paper).

## What we actually learned the hard way (the best part)

Our first depth test asserted "healthy gradient through 8 blocks" with
`output.sum()` as the loss — and got grad_norm ≈ 1e-7. Looked exactly like
vanishing gradients. It wasn't: **a LayerNorm'd vector's features sum to ~0
by construction** (they're mean-centered), so after post-norm's final LN,
`sum(output)` is a *constant* — zero gradient at depth 1 or depth 100. The
"vanishing gradient" was a degenerate loss. Probing with a fixed random
readout instead shows the residual highway working fine: healthy gradients
through 8 blocks. Both facts are now pinned in one test.

Morals: (1) when a measurement confirms your expectation, check it at a
baseline where the expectation *shouldn't* hold (depth 1 gave it away);
(2) LN's mean-centering annihilates the uniform direction — losses/probes
must not live in it.

## What breaks without it

- **No FFN:** stacked attention is (per-token) just iterated averaging —
  barely nonlinear, most of the model's capacity gone.
- **No ReLU:** the FFN collapses to one linear map; 4× width wasted.
- **No residuals:** the degradation problem returns; gradients become products
  of Jacobians; deep = untrainable.
- **No LayerNorm:** residual-stream scale drifts with depth until softmax
  saturates (lesson 01's failure mode, now at the block level).
- **BatchNorm instead:** examples in a batch contaminate each other;
  train/test mismatch; breaks at batch=1 (our local-inference case!).
- **Post-norm's own flaw:** LN sits *on* the highway, so the identity path
  isn't clean — deep post-norm stacks need LR warmup (Xiong et al. 2020).
  GPT-2's fix, pre-norm — `x = x + Sublayer(LN(x))` — is one line and is
  lesson material for Phase 2.

## Verified claims (tests)

- LN: per-token mean≈0/var≈1 at init, at any input scale; tokens independent;
  matches `torch.nn.LayerNorm`; survives constant input (ε); γ/β restore any
  scale/offset.
- FFN: shape-preserving; d_ff defaults to 4×; position-wise property exact;
  has 2× attention's weight parameters.
- Block: shape in == out; causal mask threads through (future token edits
  don't touch past outputs); every parameter gets gradient; outputs sit at
  the norm's operating point; healthy gradient through 8 blocks — and the
  `sum()`-after-LN degeneracy pinned so nobody "fixes" it back.

## Open questions (→ future lessons)

- Everything so far is permutation-equivariant — the model still can't tell
  "dog bites man" from "man bites dog". Positional encodings. (L04)
- Why exactly does pre-norm train deep stacks without warmup? (Phase 2,
  with Xiong et al. 2020.)
- Is LN's mean-centering even needed, or is rescaling doing all the work?
  (RMSNorm's bet — Phase 4.)
- If the FFN is 2/3 of parameters, activate only part per token? (MoE,
  Phase 6.)
