"""Tests for lesson 03 — LayerNorm, position-wise FFN, and the post-norm block.

As always, each test pins one claim made in the docstrings.
"""

import torch

from llmlab.components.block import TransformerBlock
from llmlab.components.ffn import PositionwiseFFN
from llmlab.components.norms import LayerNorm


# ---------------------------------------------------------------- LayerNorm

def test_layernorm_normalizes_each_token():
    """At init (γ=1, β=0), every token's output has mean ≈ 0 and variance ≈ 1
    over its features — regardless of the input's scale."""
    ln = LayerNorm(d_model=64)
    x = torch.randn(2, 5, 64) * 40 + 7   # wild scale and offset on purpose

    y = ln(x)

    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-4)


def test_layernorm_tokens_are_independent():
    """Per-token statistics: changing token 3 must not change any other
    token's output. (This is exactly what BatchNorm would violate across the
    batch — the reason transformers don't use it.)"""
    ln = LayerNorm(d_model=16)
    x = torch.randn(1, 5, 16)
    x_changed = x.clone()
    x_changed[0, 3] = torch.randn(16) * 100

    y, y_changed = ln(x), ln(x_changed)

    others = [0, 1, 2, 4]
    assert torch.allclose(y[0, others], y_changed[0, others])
    assert not torch.allclose(y[0, 3], y_changed[0, 3])


def test_layernorm_matches_pytorch():
    """Our four lines compute the same function as torch.nn.LayerNorm."""
    ours, ref = LayerNorm(32), torch.nn.LayerNorm(32)
    x = torch.randn(2, 7, 32)

    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_layernorm_survives_constant_input():
    """A zero-variance token (e.g. all-zero padding) must yield finite output,
    not 0/0 = NaN — this is what ε is for."""
    ln = LayerNorm(8)
    x = torch.zeros(1, 3, 8)

    assert torch.isfinite(ln(x)).all()


def test_layernorm_gain_bias_restore_expressiveness():
    """γ and β can undo the normalization: with γ=2, β=5 the output is
    2·x̂ + 5 — the network keeps the ability to choose any scale/offset."""
    ln = LayerNorm(16)
    with torch.no_grad():
        ln.gain.fill_(2.0)
        ln.bias.fill_(5.0)
    y = ln(torch.randn(4, 16))

    assert torch.allclose(y.mean(dim=-1), torch.full((4,), 5.0), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.full((4,), 4.0), atol=1e-3)


# ---------------------------------------------------------------- FFN

def test_ffn_shapes_and_default_expansion():
    """d_model in, d_model out; the hidden layer defaults to 4× (the paper's
    2048 for d_model=512)."""
    ffn = PositionwiseFFN(d_model=32)
    x = torch.randn(2, 5, 32)

    assert ffn(x).shape == (2, 5, 32)
    assert ffn.w1.out_features == 128  # 4 × 32


def test_ffn_is_position_wise():
    """The load-bearing claim: every position is processed independently with
    shared weights. Feeding the sequence at once == feeding each token alone."""
    ffn = PositionwiseFFN(d_model=16, d_ff=32)
    x = torch.randn(1, 6, 16)

    together = ffn(x)
    one_by_one = torch.stack([ffn(x[0, t]) for t in range(6)]).unsqueeze(0)

    assert torch.allclose(together, one_by_one, atol=1e-6)


def test_ffn_holds_most_of_the_parameters():
    """With the 4× expansion, the FFN (2·4·d² weights) has twice attention's
    parameters (4·d² weights) — 2/3 of the layer. Worth knowing when Phase 6
    asks 'what should Mixture-of-Experts make sparse?'"""
    d = 64
    count = lambda m: sum(p.numel() for p in m.parameters() if p.dim() > 1)  # weights only

    from llmlab.components.attention import MultiHeadAttention
    assert count(PositionwiseFFN(d)) == 2 * count(MultiHeadAttention(d, 8))


# ---------------------------------------------------------------- Block

def test_block_preserves_shape():
    """Shape in == shape out — the property that lets blocks stack into a
    model of arbitrary depth."""
    block = TransformerBlock(d_model=32, num_heads=4)
    x = torch.randn(2, 7, 32)

    assert block(x).shape == (2, 7, 32)


def test_block_accepts_causal_mask():
    """The mask threads through to the attention sublayer: with a causal mask,
    changing a FUTURE token must not change an earlier token's output."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=2)
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    x = torch.randn(1, 5, 16)
    x_changed = x.clone()
    x_changed[0, 4] = torch.randn(16)      # tamper with the last token only

    y = block(x, mask=causal)
    y_changed = block(x_changed, mask=causal)

    assert torch.allclose(y[0, :4], y_changed[0, :4], atol=1e-6)  # past unaffected
    assert not torch.allclose(y[0, 4], y_changed[0, 4])


def test_gradient_flows_through_deep_stack():
    """The residual highway at work: through 8 stacked blocks the input still
    receives a healthy gradient — no geometric decay toward 0.

    A trap we fell into writing this (kept as a second assertion because it
    teaches): our first version used `output.sum()` as the loss and "measured"
    grad_norm ≈ 1e-7 at EVERY depth, including depth 1. Not vanishing
    gradients — a degenerate loss: a LayerNorm'd vector's features sum to ~0
    *by construction* (they're mean-centered), so after post-norm's final LN,
    sum(output) is a constant and its gradient is exactly zero. Moral: probe
    gradients with a loss that actually depends on the output — here, a fixed
    random readout. (The REAL post-norm depth problem — why it needs LR warmup,
    Xiong et al. 2020 — appears at much larger depth/width than our toy scale;
    we'll meet it properly with pre-norm in Phase 2.)"""
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(*[TransformerBlock(16, 2) for _ in range(8)])
    x = torch.randn(1, 4, 16, requires_grad=True)
    readout = torch.randn(16)  # fixed random projection: a non-degenerate loss

    (blocks(x) * readout).sum().backward()

    assert torch.isfinite(x.grad).all()
    assert x.grad.norm() > 1e-2, "gradient should be healthy through 8 blocks"

    # The degenerate version, pinned down so nobody "fixes" it back:
    x2 = torch.randn(1, 4, 16, requires_grad=True)
    blocks(x2).sum().backward()
    assert x2.grad.norm() < 1e-4  # sum() after LayerNorm is a constant


def test_every_parameter_gets_gradient():
    """One backward pass reaches all four sub-modules (attention, FFN, both
    norms) — nothing is accidentally detached from the graph."""
    block = TransformerBlock(d_model=16, num_heads=2)
    block(torch.randn(2, 5, 16)).sum().backward()

    for name, p in block.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}: no gradient"


def test_output_is_normalized():
    """Post-norm means LayerNorm is the LAST thing a block does — so block
    outputs always sit at the norm's operating point (per-token mean ≈ 0),
    no matter how deep the stack or how wild the input scale."""
    block = TransformerBlock(d_model=32, num_heads=4)
    x = torch.randn(2, 5, 32) * 100

    y = block(x)

    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 5), atol=1e-4)
