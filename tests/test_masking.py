"""Tests for lesson 05 — causal and padding masks.

The headline claims, each made executable below:

1. Behind a causal mask, NOTHING about the future reaches position t — not
   through weights, not through values, not through gradients.
2. Causal attention over a full sequence equals ordinary attention run on each
   prefix — "the past doesn't change when the future arrives" (this identity
   is what makes the KV cache possible, Phase 5).
3. A padding mask makes batch padding invisible: real tokens get the same
   output whether or not junk was appended after them.
4. The known hazard: a row with zero permitted keys yields NaN, not zeros.
"""

import torch

from llmlab.components.attention import MultiHeadAttention, scaled_dot_product_attention
from llmlab.components.block import TransformerBlock
from llmlab.components.masking import causal_mask, combine_masks, padding_mask


# ------------------------------------------------- the causal mask itself

def test_causal_mask_hand_values():
    """Pin the rule to an eyeball-checkable matrix: row q (the query) is True
    exactly at columns 0..q — self and past, never future. Diagonal True:
    a token may always read itself."""
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(causal_mask(4), expected)


def test_causal_weights_are_strictly_lower_triangular():
    """After softmax, future positions get weight EXACTLY 0 (not merely small:
    e^{-inf} = 0 in floating point too), and each row still sums to 1 —
    the surviving past weights absorb the freed mass."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    mha = MultiHeadAttention(d_model=16, num_heads=4)

    _, weights = mha(x, mask=causal_mask(5))  # (batch, heads, seq_q, seq_k)

    future = ~causal_mask(5)  # strictly-upper-triangular positions
    assert (weights[..., future] == 0.0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5))


# ------------------------------------------------- no leak from the future

def test_future_tokens_cannot_change_past_outputs():
    """The operational meaning of 'causal': edit a future token, and outputs
    at all earlier positions are bit-for-bit unchanged — through a FULL block
    (attention is the only cross-token path; FFN and LayerNorm are per-token,
    so masking attention seals the block). Without the mask, the same edit
    changes every position — attention is all-to-all by default."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=2)
    x = torch.randn(1, 6, 16)
    x_edited = x.clone()
    x_edited[0, 4] += 10.0  # rewrite position 4 (a "future" token for 0..3)

    with torch.no_grad():
        masked_a = block(x, mask=causal_mask(6))
        masked_b = block(x_edited, mask=causal_mask(6))
        open_a = block(x)
        open_b = block(x_edited)

    assert torch.equal(masked_a[:, :4], masked_b[:, :4])   # past sealed
    assert not torch.allclose(open_a[:, :4], open_b[:, :4])  # unmasked leaks


def test_no_gradient_flows_to_the_future():
    """Same claim through the backward pass: the loss at position t must not
    push gradients into inputs at positions > t — otherwise training would
    still tune 'the future' to help predict itself, a subtler leak than the
    forward one. ∂out[t]/∂x[>t] = 0, exactly."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=2)
    x = torch.randn(1, 5, 16, requires_grad=True)

    out, _ = mha(x, mask=causal_mask(5))
    out[0, 2].sum().backward()  # loss depends only on position 2

    assert torch.equal(x.grad[0, 3:], torch.zeros(2, 16))  # future untouched
    assert (x.grad[0, :3] != 0).any()                      # past used


def test_causal_attention_equals_attention_over_each_prefix():
    """Position t under a causal mask sees keys 0..t — exactly what plain
    attention sees when fed only x[:, :t+1]. So the full-sequence causal
    pass reproduces every prefix's last-position output at once. This
    identity is (a) why one pass = n training examples, and (b) why
    generation can cache K/V: past outputs never change as tokens are
    appended (Phase 5's KV cache is this test, exploited)."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=2)
    x = torch.randn(1, 6, 16)

    with torch.no_grad():
        full, _ = mha(x, mask=causal_mask(6))
        for t in range(6):
            prefix, _ = mha(x[:, : t + 1])  # no mask needed: future absent
            assert torch.allclose(full[0, t], prefix[0, t], atol=1e-6), f"t={t}"


# ------------------------------------------------- padding masks

def test_padding_mask_shape_and_values():
    """From token ids to a (batch, 1, 1, seq_k) key mask: True on real tokens,
    False on PAD. The singleton dims are broadcast slots for heads and
    queries — only the key axis carries information."""
    ids = torch.tensor([[5, 3, 9, 0, 0], [7, 0, 0, 0, 0]])  # PAD id = 0

    mask = padding_mask(ids, pad_id=0)

    assert mask.shape == (2, 1, 1, 5)
    assert torch.equal(mask[0, 0, 0], torch.tensor([True, True, True, False, False]))
    assert torch.equal(mask[1, 0, 0], torch.tensor([True, False, False, False, False]))


def test_pad_keys_get_zero_weight():
    """No query — at any position, in any head — spends any attention weight
    on a pad key."""
    torch.manual_seed(0)
    ids = torch.tensor([[5, 3, 9, 0, 0]])
    x = torch.randn(1, 5, 16)
    mha = MultiHeadAttention(d_model=16, num_heads=4)

    _, weights = mha(x, mask=padding_mask(ids, pad_id=0))

    assert (weights[..., 3:] == 0.0).all()          # pad columns: zero
    assert torch.allclose(weights.sum(-1), torch.ones(1, 4, 5))


def test_padding_makes_batching_invisible():
    """The point of the exercise: a sequence padded out to a longer rectangle
    yields the SAME outputs at its real positions as the unpadded sequence
    alone. Padding is a storage format, and the mask keeps it from becoming
    model input. (Without the mask this fails: pad vectors enter every
    average.)"""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=2)
    real = torch.randn(1, 3, 16)
    junk = torch.randn(1, 2, 16)  # stands in for whatever embeddings PAD has
    padded = torch.cat([real, junk], dim=1)  # (1, 5, 16)
    ids = torch.tensor([[1, 1, 1, 0, 0]])    # only positions 0..2 are real

    with torch.no_grad():
        out_unpadded, _ = mha(real)
        out_masked, _ = mha(padded, mask=padding_mask(ids, pad_id=0))
        out_unmasked, _ = mha(padded)

    assert torch.allclose(out_masked[:, :3], out_unpadded, atol=1e-6)
    assert not torch.allclose(out_unmasked[:, :3], out_unpadded, atol=1e-6)


# ------------------------------------------------- combining + the hazard

def test_combine_causal_and_padding():
    """A decoder over a padded batch needs both constraints at once; AND +
    broadcasting gives (batch, 1, seq, seq): position q may attend k iff
    k ≤ q AND k is a real token. None-handling: no masks → None (attention's
    fast path), a single mask passes through unchanged."""
    ids = torch.tensor([[4, 2, 0]])  # PAD id = 0: last position is padding
    combined = combine_masks(causal_mask(3), padding_mask(ids, pad_id=0))

    assert combined.shape == (1, 1, 3, 3)
    expected = torch.tensor(
        [
            [True, False, False],   # q=0: itself only
            [True, True, False],    # q=1: the real past
            [True, True, False],    # q=2: causal would allow k=2, padding vetoes
        ]
    )
    assert torch.equal(combined[0, 0], expected)
    assert combine_masks(None, None) is None
    m = causal_mask(3)
    assert torch.equal(combine_masks(None, m), m)


def test_fully_masked_row_is_nan_not_zero():
    """The hazard the docstrings warn about, pinned: a query row with NO
    permitted key puts −inf into every softmax input, and softmax computes
    0/0 = NaN — not a zero vector. This is why padding_mask masks only keys
    and why causal_mask keeps its diagonal. If a model ever emits NaN, an
    over-zealous mask row is one of the first suspects."""
    q = torch.randn(1, 2, 8)
    kv = torch.randn(1, 3, 8)
    mask = torch.tensor([[[True, True, True], [False, False, False]]])

    out, weights = scaled_dot_product_attention(q, kv, kv, mask=mask)

    assert torch.isnan(weights[0, 1]).all()  # the starved row
    assert torch.isnan(out[0, 1]).all()      # ...poisons its output
    assert not torch.isnan(out[0, 0]).any()  # healthy row unaffected
