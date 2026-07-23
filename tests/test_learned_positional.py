"""Tests for lesson 11 — learned positional embeddings.

Learned positions add *nothing new* to the forward pass: it is still lesson
04's additive combination (x + a per-position row), still breaks permutation
equivariance, still capped by max_len. What the tests pin is the ONE thing that
changed — the table is now a trained parameter, not a formula — and the two
consequences that flow from it: it lives in the checkpoint, and it cannot
extrapolate past the positions it was trained on.
"""

from dataclasses import replace

import torch

from llmlab.components.block import TransformerBlock
from llmlab.components.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
    build_positional,
)
from llmlab.models.gpt import GPT, GPTConfig


# ------------------------------------------------- same interface as lesson 04

def test_forward_is_a_pure_positional_add():
    """Like the sinusoidal module, forward is exactly x + table[0:seq] and
    nothing else — subtract the input back out and the same (seq, d_model) row
    is left for every batch element (position depends on WHERE, not on which
    example). (atol=1e-6: the add is exact in math, ~1e-7 lossy in float32.)"""
    torch.manual_seed(0)
    pe = LearnedPositionalEmbedding(d_model=16, max_len=32)
    x = torch.randn(3, 7, 16)

    y = pe(x)
    added = y - x  # (3, 7, 16)

    assert y.shape == (3, 7, 16)
    # Row 0 of the batch equals the table's first 7 rows...
    assert torch.allclose(added[0], pe.table.weight[:7], atol=1e-6)
    # ...and every batch element got the identical position rows.
    assert torch.allclose(added[0], added[1], atol=1e-6)
    assert torch.allclose(added[0], added[2], atol=1e-6)


def test_breaks_permutation_equivariance():
    """The whole reason positions exist (lesson 04): stamping each token with
    WHERE it sits means permuting the input no longer just permutes the output.
    A learned table must earn this the same way the fixed one gets it for free —
    with random init it already does, because distinct rows are distinct."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=2)
    pe = LearnedPositionalEmbedding(d_model=16, max_len=16)
    x = torch.randn(1, 6, 16)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])

    assert not torch.allclose(block(pe(x[:, perm])), block(pe(x))[:, perm], atol=1e-5)


# ------------------------------------------------- the one thing that changed

def test_table_is_a_trained_parameter_not_a_buffer():
    """The entire architectural difference from lesson 04: the position table is
    now an nn.Parameter — it has parameters, it is in the state_dict, and the
    optimizer will move it. The sinusoidal module has neither (buffer only)."""
    learned = LearnedPositionalEmbedding(d_model=8, max_len=10)
    fixed = SinusoidalPositionalEncoding(d_model=8, max_len=10)

    assert len(list(learned.parameters())) == 1          # the table
    assert list(learned.parameters())[0].numel() == 10 * 8
    assert "table.weight" in learned.state_dict()         # persists in checkpoint

    assert list(fixed.parameters()) == []                 # lesson 04: nothing learned
    assert fixed.state_dict() == {}                       # non-persistent buffer


def test_only_used_rows_receive_gradient():
    """The extrapolation limit, made concrete. A forward over seq=4 touches only
    rows 0..3, so only those rows get gradient; rows 4..max_len-1 stay at zero.
    Positions never seen in training are never shaped — which is exactly why a
    learned table has nothing to say about positions beyond its trained range."""
    pe = LearnedPositionalEmbedding(d_model=8, max_len=16)
    x = torch.randn(1, 4, 8, requires_grad=False)

    pe(x).sum().backward()

    grad = pe.table.weight.grad            # (16, 8)
    assert grad[:4].abs().sum() > 0        # used positions were shaped
    assert torch.all(grad[4:] == 0)        # unseen positions got nothing


def test_rejects_sequences_beyond_max_len():
    """Sinusoids merely run out of *precomputed* rows (grow max_len and they're
    back). A learned table runs out of *meaning*: rows past max_len were never
    trained. Either way we fail loud rather than wrap or index out of bounds."""
    pe = LearnedPositionalEmbedding(d_model=8, max_len=4)

    try:
        pe(torch.randn(1, 5, 8))
        assert False, "expected ValueError for seq_len > max_len"
    except ValueError:
        pass


# ------------------------------------------------- the config-driven knob

def test_build_positional_selects_and_validates():
    """The assembly knob: a string picks the component, both satisfying one
    interface, and an unknown name fails at build time (a silent wrong-position
    scheme would only show up as a worse loss curve)."""
    assert isinstance(build_positional("learned", 8, 16), LearnedPositionalEmbedding)
    assert isinstance(build_positional("sinusoidal", 8, 16), SinusoidalPositionalEncoding)

    try:
        build_positional("rope", 8, 16)  # a real future component, not yet built
        assert False, "expected ValueError for unknown positional kind"
    except ValueError:
        pass


def test_both_kinds_are_drop_in_interchangeable():
    """Same call, same output shape, same max_len guard — so the model can hold
    either without knowing which. This interchangeability is what makes the swap
    a one-line config change instead of a forked model file."""
    x = torch.randn(2, 5, 8)
    for kind in ("learned", "sinusoidal"):
        pe = build_positional(kind, d_model=8, max_len=6)
        assert pe(x).shape == (2, 5, 8)
        try:
            pe(torch.randn(2, 7, 8))
            assert False, f"{kind} should reject seq > max_len"
        except ValueError:
            pass


# ------------------------------------------------- inside the GPT model

def test_gpt_defaults_to_learned_positions():
    """Phase 2's GPT is now one step closer to GPT-2 specifically: its default
    positional component is the learned table (2017's sinusoid was lesson 09's
    placeholder). The model wires whichever the config names."""
    cfg = GPTConfig.tiny()
    learned_model = GPT(cfg)                                     # default
    sinus_model = GPT(replace(cfg, positional="sinusoidal"))     # one-field swap

    assert isinstance(learned_model.positional, LearnedPositionalEmbedding)
    assert isinstance(sinus_model.positional, SinusoidalPositionalEncoding)

    # The learned model carries exactly max_len × d_model extra parameters — the
    # position table — over the otherwise identical sinusoidal model.
    extra = sum(p.numel() for p in learned_model.parameters()) - sum(
        p.numel() for p in sinus_model.parameters()
    )
    assert extra == cfg.max_len * cfg.d_model


def test_gpt_with_learned_positions_still_causal():
    """Swapping the positional component must not disturb the model's defining
    property: editing a future token still cannot move an earlier position's
    logits (the causal mask, lesson 05, is orthogonal to how positions enter)."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())  # learned positions by default
    ids = torch.randint(1, 32, (1, 6))
    edited = ids.clone()
    edited[0, 4] = (ids[0, 4] + 1) % 31 + 1

    with torch.no_grad():
        a = model(ids)
        b = model(edited)

    assert torch.equal(a[:, :4], b[:, :4])         # past sealed
    assert not torch.allclose(a[:, 4:], b[:, 4:])  # the edit registered
