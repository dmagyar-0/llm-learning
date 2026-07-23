"""Tests for lesson 10 — pre-norm vs post-norm.

The concept is a single knob: where the LayerNorm sits relative to the residual
add. The claims to pin:

    - the knob is validated (typo → loud error, not silent wrong model)
    - both placements produce the same SHAPE and stay causal (it's a norm move,
      not a behavior change to the mask)
    - the DEFINING forward signature: post-norm keeps the residual stream's
      magnitude bounded (it renormalizes every block); pre-norm lets it GROW
      (a running sum, never renormalized) — which is *why* pre-norm needs a
      closing LayerNorm
    - so the pre-norm MODEL adds that final norm (ln_f); the post-norm model
      does not
    - gradients still reach every parameter, the closing norm included
"""

from dataclasses import replace

import torch

from llmlab.components.block import TransformerBlock
from llmlab.models.gpt import GPT, GPTConfig


# ------------------------------------------------- the knob itself

def test_invalid_norm_placement_raises():
    """A typo must fail loudly at construction, not silently build the wrong
    architecture (the worst kind of bug — it trains, just worse)."""
    for bad in ("prenorm", "PRE", "before", ""):
        try:
            TransformerBlock(32, 2, norm_placement=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have raised")


def test_both_placements_same_shape_and_causal():
    """Moving the norm changes the numbers, not the interface: identical output
    shape, and both remain autoregressive (the mask, not the norm, owns that)."""
    for placement in ("post", "pre"):
        torch.manual_seed(0)
        model = GPT(replace(GPTConfig.tiny(), norm_placement=placement))
        ids = torch.randint(1, 32, (1, 6))
        edited = ids.clone()
        edited[0, 4] = (ids[0, 4] + 1) % 31 + 1

        with torch.no_grad():
            a, b = model(ids), model(edited)

        assert a.shape == (1, 6, 32)
        assert torch.equal(a[:, :4], b[:, :4])         # past sealed → causal
        assert not torch.allclose(a[:, 4:], b[:, 4:])   # edit registered


# ------------------------------------------------- the defining signature

def test_post_bounds_the_stream_pre_lets_it_grow():
    """The mechanism, made visible. Run one input through a DEEP stack of bare
    blocks and measure the residual-stream norm at entry vs. exit:

        post-norm → LN sits ON the highway, renormalizing every add, so the
                    magnitude stays put (growth ≈ 1×).
        pre-norm  → LN sits INSIDE the branch; the highway is a pure running
                    sum of sublayer outputs, so the magnitude climbs with depth.

    That growth is exactly why a pre-norm *model* must end with one closing
    LayerNorm before its head (the next test)."""
    depth = 24

    def growth(placement: str) -> float:
        torch.manual_seed(0)
        blocks = [TransformerBlock(32, 2, norm_placement=placement) for _ in range(depth)]
        x = torch.randn(2, 10, 32)
        entry = x.norm(dim=-1).mean().item()
        with torch.no_grad():
            for b in blocks:
                x = b(x)
        return x.norm(dim=-1).mean().item() / entry

    post_growth = growth("post")
    pre_growth = growth("pre")

    assert post_growth < 1.3, f"post-norm should stay bounded, got {post_growth:.2f}×"
    assert pre_growth > 1.4, f"pre-norm should grow with depth, got {pre_growth:.2f}×"
    assert pre_growth > post_growth  # the whole point, in one comparison


# ------------------------------------------------- the model's closing norm

def test_pre_norm_model_has_final_norm_post_does_not():
    """Because pre-norm leaves the stream un-normalized at the top, the GPT
    model closes with ln_f; post-norm's last block already ended in an LN, so
    it needs none (an identity)."""
    pre = GPT(GPTConfig.tiny())                          # default is "pre"
    post = GPT(replace(GPTConfig.tiny(), norm_placement="post"))

    from llmlab.components.norms import LayerNorm
    assert isinstance(pre.norm_final, LayerNorm)          # ln_f present
    assert isinstance(post.norm_final, torch.nn.Identity)  # not needed


def test_pre_norm_gradients_reach_every_parameter():
    """Dead-wiring check including the new closing norm: one backward from a
    plausible loss must touch every parameter, ln_f's γ/β included."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())  # pre-norm
    ids = torch.randint(1, 32, (2, 6))

    model(ids).logsumexp(dim=-1).mean().backward()

    assert model.norm_final.gain.grad is not None  # the new parameter is live
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient at {name}"
