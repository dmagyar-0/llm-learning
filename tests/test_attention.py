"""Tests for lesson 01 — scaled dot-product attention.

Each test doubles as a usage example and pins down one claim made in the
docstrings/comments of `attention.py`.
"""

import math

import torch

from llmlab.components.attention import scaled_dot_product_attention


def test_shapes_and_weights_sum_to_one():
    """The basic contract: output/weight shapes, and each query's weights form
    a probability distribution (non-negative, summing to 1)."""
    batch, seq_q, seq_k, d_k, d_v = 2, 5, 7, 16, 8
    q = torch.randn(batch, seq_q, d_k)
    k = torch.randn(batch, seq_k, d_k)
    v = torch.randn(batch, seq_k, d_v)

    out, w = scaled_dot_product_attention(q, k, v)

    assert out.shape == (batch, seq_q, d_v)
    assert w.shape == (batch, seq_q, seq_k)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch, seq_q))


def test_identical_keys_give_uniform_average():
    """If every key is identical, no position is more relevant than another,
    so every weight must be 1/seq_k and the output is the plain mean of the
    values — attention degrades gracefully into averaging."""
    seq_k, d_k, d_v = 4, 8, 3
    q = torch.randn(1, d_k)
    k = torch.ones(seq_k, d_k)          # all keys the same
    v = torch.randn(seq_k, d_v)

    out, w = scaled_dot_product_attention(q, k, v)

    assert torch.allclose(w, torch.full((1, seq_k), 1 / seq_k))
    assert torch.allclose(out, v.mean(dim=0, keepdim=True), atol=1e-6)


def test_strong_match_retrieves_the_matching_value():
    """The dictionary-lookup limit: when one query·key score dominates, the
    output approaches that key's value exactly — a soft lookup becoming hard."""
    # Query points along the first axis; key 0 matches it with a large scale,
    # key 1 is orthogonal (score 0).
    q = torch.tensor([[10.0, 0.0]])
    k = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    v = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    out, w = scaled_dot_product_attention(q, k, v)

    # score_0 = 100/√2 ≈ 70.7, score_1 = 0 → softmax ≈ (1, 0)
    assert w[0, 0] > 0.999
    assert torch.allclose(out, v[0:1], atol=1e-3)


def test_known_values_by_hand():
    """A case small enough to verify with pencil and paper."""
    q = torch.tensor([[1.0, 0.0]])                    # (1, 2)
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])        # (2, 2)
    v = torch.tensor([[10.0], [20.0]])                # (2, 1)

    out, w = scaled_dot_product_attention(q, k, v)

    # scores = [1, 0] / √2 = [0.70711, 0]
    # weights = softmax = [e^0.70711, e^0] / (e^0.70711 + e^0)
    e = math.exp(1 / math.sqrt(2))
    w0 = e / (e + 1)
    assert torch.allclose(w, torch.tensor([[w0, 1 - w0]]), atol=1e-6)
    assert torch.allclose(out, torch.tensor([[10 * w0 + 20 * (1 - w0)]]), atol=1e-5)


def test_mask_zeroes_forbidden_positions():
    """Masked (False) positions get exactly zero weight, and the remaining
    weights still sum to 1 — because we mask *before* the softmax."""
    seq_q, seq_k = 3, 3
    q, k = torch.randn(seq_q, 8), torch.randn(seq_k, 8)
    v = torch.randn(seq_k, 4)
    causal = torch.tril(torch.ones(seq_q, seq_k, dtype=torch.bool))  # lower-triangular

    _, w = scaled_dot_product_attention(q, k, v, mask=causal)

    assert torch.all(w[~causal] == 0)                       # future = zero weight
    assert torch.allclose(w.sum(dim=-1), torch.ones(seq_q))  # rows still normalized


def test_scaling_keeps_score_variance_near_one():
    """The √d_k derivation, verified empirically: with unit-variance inputs,
    raw q·k scores have variance ≈ d_k, and dividing by √d_k restores ≈ 1.
    This is the whole reason 'scaled' is in the name."""
    torch.manual_seed(0)
    d_k = 256
    q = torch.randn(10_000, d_k)
    k = torch.randn(10_000, d_k)

    raw = (q * k).sum(-1)                 # one dot product per row
    scaled = raw / math.sqrt(d_k)

    assert abs(raw.var().item() - d_k) / d_k < 0.1   # Var ≈ d_k (within 10%)
    assert abs(scaled.var().item() - 1.0) < 0.1      # Var ≈ 1


def test_gradients_flow():
    """Attention must be differentiable end-to-end — that is its reason to
    exist (vs. a hard dictionary lookup). gradcheck compares analytic gradients
    against finite differences in float64."""
    q = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v)[0], (q, k, v)
    )


def test_matches_pytorch_reference():
    """Our naive version agrees with PyTorch's optimized built-in — same math,
    different engineering (we'll study the engineering in Phase 5)."""
    q = torch.randn(2, 4, 5, 16)  # (batch, heads, seq, d_k) — broadcasting works
    k = torch.randn(2, 4, 7, 16)
    v = torch.randn(2, 4, 7, 16)

    ours, _ = scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(ours, ref, atol=1e-5)
