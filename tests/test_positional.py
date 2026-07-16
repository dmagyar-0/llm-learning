"""Tests for lesson 04 — sinusoidal positional encodings.

As always, each test pins one claim made in the docstrings. The first two are
the reason this lesson exists at all: the stack we built so far genuinely
cannot see order, and adding PE genuinely fixes that.
"""

import math

import torch

from llmlab.components.block import TransformerBlock
from llmlab.components.positional import SinusoidalPositionalEncoding, sinusoidal_table


# ------------------------------------------------- the motivating problem

def test_transformer_block_is_permutation_equivariant():
    """The lesson-01..03 stack treats the sequence as a SET: permuting the
    input tokens permutes the outputs identically — block(x[perm]) equals
    block(x)[perm]. Nothing we built so far can tell 'dog bites man' from
    'man bites dog'."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=2)
    x = torch.randn(1, 6, 16)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])

    assert torch.allclose(block(x[:, perm]), block(x)[:, perm], atol=1e-5)


def test_positional_encoding_breaks_permutation_equivariance():
    """Adding PE before the block destroys the symmetry: each token is now
    stamped with WHERE it sits, so moving it changes its representation, not
    just its slot. This is the entire job of a positional encoding."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=2)
    pe = SinusoidalPositionalEncoding(d_model=16)
    x = torch.randn(1, 6, 16)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])

    assert not torch.allclose(block(pe(x[:, perm])), block(pe(x))[:, perm], atol=1e-5)


# ------------------------------------------------- the table itself

def test_shapes_and_forward_is_a_pure_add():
    """Module preserves (batch, seq, d_model); subtracting the input back out
    leaves the table — forward is x + PE and nothing else. (atol=1e-5, not
    the default 1e-8: in float32, (x + t) − x loses ~1e-7 to rounding when
    x is O(1) — the add is exact in math, not in bits.)"""
    pe = SinusoidalPositionalEncoding(d_model=32, max_len=50)
    x = torch.randn(2, 7, 32)

    y = pe(x)

    assert y.shape == (2, 7, 32)
    assert torch.allclose(y - x, sinusoidal_table(7, 32).expand(2, 7, 32), atol=1e-5)


def test_known_values():
    """Pin the formula to hand-computable numbers: position 0 is
    (sin 0, cos 0, ...) = (0, 1, 0, 1, ...); channel pair 0 has frequency
    ω_0 = 1, so at position p it is exactly (sin p, cos p)."""
    t = sinusoidal_table(seq_len=10, d_model=8)

    assert torch.allclose(t[0], torch.tensor([0.0, 1.0] * 4))
    for p in range(10):
        assert torch.allclose(t[p, 0], torch.tensor(math.sin(p)), atol=1e-6)
        assert torch.allclose(t[p, 1], torch.tensor(math.cos(p)), atol=1e-6)


def test_values_bounded_and_positions_unique():
    """Every entry lies in [−1, 1] regardless of position (lesson 03's scale
    discipline holds even at position 2047), yet every position's code is
    distinct — bounded does not mean ambiguous, exactly like binary digits."""
    t = sinusoidal_table(seq_len=2048, d_model=64)

    assert t.min() >= -1.0 and t.max() <= 1.0
    # All pairwise distances strictly positive → no two rows collide.
    dists = torch.cdist(t, t) + torch.eye(2048) * 1e9  # mask the diagonal
    assert dists.min() > 1e-3


def test_position_code_does_not_depend_on_sequence_length():
    """Row p is a function of p alone: a 100-long table begins with the
    10-long table. 'Position 5' means the same thing in every sequence —
    which is why one precomputed buffer can serve all batch shapes."""
    assert torch.allclose(sinusoidal_table(100, 32)[:10], sinusoidal_table(10, 32))


# ------------------------------------------------- the paper's key property

def test_relative_shift_is_a_rotation():
    """The reason Vaswani et al. picked sinusoids: PE(p+k) is a LINEAR
    function of PE(p). Concretely, for each channel pair with frequency ω,
    the 2×2 rotation by angle ω·k — built from k alone — maps PE(p) to
    PE(p+k) for EVERY p. (Phase 4's RoPE turns exactly this rotation into
    the attention mechanism itself.)"""
    d_model, base, k = 16, 10000.0, 5
    t = sinusoidal_table(seq_len=40, d_model=d_model, base=base)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    freqs = torch.exp(-math.log(base) * i / d_model)   # ω per channel pair

    phi = freqs * k                                     # rotation angle per pair
    for p in range(30):
        pairs = t[p].view(-1, 2)                        # [(sin ωp, cos ωp), ...]
        rotated_sin = pairs[:, 0] * torch.cos(phi) + pairs[:, 1] * torch.sin(phi)
        rotated_cos = -pairs[:, 0] * torch.sin(phi) + pairs[:, 1] * torch.cos(phi)
        rotated = torch.stack([rotated_sin, rotated_cos], dim=1).view(-1)

        assert torch.allclose(rotated, t[p + k], atol=1e-5), f"failed at p={p}"


def test_dot_product_depends_only_on_offset_and_is_direction_blind():
    """PE(p)·PE(p+k) = Σᵢ cos(ωᵢk): the same for every p — so a dot-product
    machine like attention can read DISTANCE straight off these codes. But
    cos is even, so k and −k give the same value: raw dot products sense how
    far, never which side. Direction needs the rotation/phase route (or the
    relative encodings of Shaw 2018 / RoPE — future lessons)."""
    t = sinusoidal_table(seq_len=64, d_model=32)
    k = 7

    d0 = t[3] @ t[3 + k]
    assert torch.allclose(d0, t[20] @ t[20 + k], atol=1e-4)   # p-independent
    assert torch.allclose(d0, t[30] @ t[30 - k], atol=1e-4)   # ±k identical


# ------------------------------------------------- engineering details

def test_no_learnable_parameters():
    """Sinusoidal PE is pure function, not knowledge: nothing for the
    optimizer to touch, nothing in the checkpoint (the buffer is
    non-persistent). Phase 2's learned embeddings differ in exactly this."""
    pe = SinusoidalPositionalEncoding(d_model=32)

    assert list(pe.parameters()) == []
    assert pe.state_dict() == {}


def test_gradient_flows_through_the_add():
    """x + PE must stay differentiable w.r.t. x with gradient exactly 1 —
    the encoding is a constant offset, not a transformation, so it cannot
    distort what the embeddings learn."""
    pe = SinusoidalPositionalEncoding(d_model=8)
    x = torch.randn(1, 4, 8, requires_grad=True)

    pe(x).sum().backward()

    assert torch.allclose(x.grad, torch.ones_like(x))


def test_rejects_sequences_beyond_max_len():
    """The precomputed buffer is a capacity: exceeding it must fail loudly,
    not silently truncate or wrap positions."""
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=4)

    try:
        pe(torch.randn(1, 5, 8))
        assert False, "expected ValueError for seq_len > max_len"
    except ValueError:
        pass


def test_rejects_odd_d_model():
    """Frequencies come in sin/cos PAIRS — half a pair can't rotate, so an
    odd d_model is a construction error, caught at build time."""
    try:
        sinusoidal_table(seq_len=4, d_model=7)
        assert False, "expected ValueError for odd d_model"
    except ValueError:
        pass
