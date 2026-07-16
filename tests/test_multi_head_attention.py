"""Tests for lesson 02 — multi-head attention.

Each test pins down one claim from the docstrings of
`MultiHeadAttention` — including the big one: the naive per-head loop and the
batched reshape compute *exactly* the same function.
"""

import torch

from llmlab.components.attention import MultiHeadAttention


def test_shapes():
    """The basic contract: output has the input's shape (so it can be added
    back to the residual stream later), weights carry one distribution per
    head per query."""
    batch, seq, d_model, heads = 2, 5, 32, 4
    mha = MultiHeadAttention(d_model, heads)
    x = torch.randn(batch, seq, d_model)

    out, w = mha(x)

    assert out.shape == (batch, seq, d_model)
    assert w.shape == (batch, heads, seq, seq)
    # Every head's every query row is a probability distribution.
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch, heads, seq))


def test_naive_loop_equals_batched():
    """The load-bearing claim of the lesson: looping over heads and calling
    lesson 01's attention per slice computes the SAME function as the
    view/transpose batched version. The reshape is bookkeeping, not math."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=24, num_heads=3)
    x = torch.randn(2, 7, 24)

    out_naive, w_naive = mha(x, naive=True)
    out_batched, w_batched = mha(x, naive=False)

    assert torch.allclose(out_naive, out_batched, atol=1e-6)
    assert torch.allclose(w_naive, w_batched, atol=1e-6)


def test_naive_loop_equals_batched_with_causal_mask():
    """Same equivalence, with a 2-D causal mask broadcast over batch & heads."""
    torch.manual_seed(1)
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))

    out_naive, w_naive = mha(x, mask=causal, naive=True)
    out_batched, w_batched = mha(x, mask=causal, naive=False)

    assert torch.allclose(out_naive, out_batched, atol=1e-6)
    assert torch.allclose(w_naive, w_batched, atol=1e-6)
    # And the mask did its job in every head: no weight on the future.
    assert torch.all(w_batched[..., ~causal] == 0)


def test_heads_learn_different_relations():
    """Heads have independent projections, so (at random init already) they
    produce genuinely different attention patterns over the same input —
    that's the whole point of having more than one."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    x = torch.randn(1, 6, 32)

    _, w = mha(x)  # (1, 4, 6, 6)

    # No two heads share the same weight pattern.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(w[0, i], w[0, j], atol=1e-3)


def test_parameter_count_is_independent_of_heads():
    """Heads split the width, they don't add width: the parameter bill is
    4·d_model² + 4·d_model biases, whether there is 1 head or 8."""
    d_model = 64
    count = lambda m: sum(p.numel() for p in m.parameters())

    expected = 4 * d_model * d_model + 4 * d_model
    assert count(MultiHeadAttention(d_model, num_heads=1)) == expected
    assert count(MultiHeadAttention(d_model, num_heads=8)) == expected


def test_d_model_must_be_divisible_by_heads():
    """d_head = d_model / h must be an integer — we slice, we don't pad."""
    try:
        MultiHeadAttention(d_model=10, num_heads=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_self_attention_is_permutation_equivariant():
    """Multi-head attention still never looks at *positions*: permute the
    input tokens and the output tokens permute identically. This is the
    order-blindness that lesson 04 (positional encodings) must repair."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=2)
    x = torch.randn(1, 6, 16)
    perm = torch.randperm(6)

    out_then_perm = mha(x)[0][:, perm]
    perm_then_out = mha(x[:, perm])[0]

    assert torch.allclose(out_then_perm, perm_then_out, atol=1e-5)


def test_cross_attention_shapes():
    """Queries from one sequence, keys/values from another (the decoder→encoder
    pattern): output length follows the QUERY sequence, weights span both."""
    mha = MultiHeadAttention(d_model=16, num_heads=2)
    decoder_side = torch.randn(2, 3, 16)   # 3 querying tokens
    encoder_side = torch.randn(2, 9, 16)   # 9 tokens being read

    out, w = mha(decoder_side, x_context=encoder_side)

    assert out.shape == (2, 3, 16)
    assert w.shape == (2, 2, 3, 9)


def test_gradients_flow_through_all_parameters():
    """One backward pass touches every parameter: all four projections are in
    the gradient path (if W_O had no gradient, heads would never learn to be
    combined; if W_Q/W_K had none, attention patterns could never train)."""
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    out, _ = mha(x)
    out.sum().backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is exactly zero"


def test_matches_pytorch_reference():
    """Copy our weights into torch.nn.MultiheadAttention and confirm both
    modules compute the same function — same math, PyTorch's engineering."""
    torch.manual_seed(0)
    d_model, heads = 32, 4
    ours = MultiHeadAttention(d_model, heads, bias=True)
    ref = torch.nn.MultiheadAttention(d_model, heads, bias=True, batch_first=True)

    # torch fuses W_Q, W_K, W_V vertically into in_proj_weight (3·d_model, d_model).
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
        )
        ref.in_proj_bias.copy_(
            torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
        )
        ref.out_proj.weight.copy_(ours.w_o.weight)
        ref.out_proj.bias.copy_(ours.w_o.bias)

    x = torch.randn(2, 6, d_model)
    out_ours, w_ours = ours(x)
    out_ref, w_ref = ref(x, x, x, average_attn_weights=False)

    assert torch.allclose(out_ours, out_ref, atol=1e-5)
    assert torch.allclose(w_ours, w_ref, atol=1e-5)
