"""Tests for lesson 06 — the full encoder–decoder assembly.

No new mechanisms were introduced, so these tests are about the WIRING:
does information flow exactly where Figure 1 says — and nowhere else?

    - decoder → still causal, end-to-end through the whole model
    - encoder → actually consulted (cross-attention is live)
    - encoder → bidirectional on purpose (no triangle on the source)
    - padding → invisible on both sides, end-to-end
    - gradients → reach every parameter (nothing is dead wiring)
"""

import torch

from llmlab.components.embeddings import TokenEmbedding
from llmlab.components.positional import sinusoidal_table
from llmlab.models.transformer import Transformer, TransformerConfig

PAD = 0


def make_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(TransformerConfig.tiny())


# ------------------------------------------------- shapes and configs

def test_output_shape_is_per_position_logits():
    """(batch, src_seq) + (batch, tgt_seq) → (batch, tgt_seq, tgt_vocab):
    one next-token distribution per target position — n predictions from
    one pass, exactly the teacher-forcing layout lesson 05 promised."""
    model = make_model()
    src = torch.randint(1, 32, (2, 7))
    tgt = torch.randint(1, 32, (2, 5))

    logits = model(src, tgt)

    assert logits.shape == (2, 5, 32)


def test_tiny_config_is_actually_tiny():
    """The teaching contract demands a CPU-in-seconds config. Pin the order
    of magnitude so a future edit can't silently bloat it."""
    n_params = sum(p.numel() for p in make_model().parameters())
    assert n_params < 100_000, f"tiny config grew to {n_params} params"


# ------------------------------------------------- the wiring claims

def test_decoder_is_causal_end_to_end():
    """Lesson 05 proved causality for one block; assembly can still break it
    (one missing mask in one sublayer of one layer suffices). Editing a
    future target token must leave earlier positions' logits bit-for-bit
    unchanged through the FULL model."""
    model = make_model()
    src = torch.randint(1, 32, (1, 6))
    tgt = torch.randint(1, 32, (1, 5))
    tgt_edited = tgt.clone()
    tgt_edited[0, 3] = (tgt[0, 3] + 1) % 31 + 1  # change token 3, stay non-pad

    with torch.no_grad():
        a = model(src, tgt)
        b = model(src, tgt_edited)

    assert torch.equal(a[:, :3], b[:, :3])       # past sealed
    assert not torch.allclose(a[:, 3:], b[:, 3:])  # edit did register


def test_cross_attention_actually_reads_the_source():
    """The opposite check: the source must MATTER. Change one source token
    and every target position's logits should move — cross-attention is the
    only bridge, so this proves the bridge carries signal."""
    model = make_model()
    src = torch.randint(1, 32, (1, 6))
    src_edited = src.clone()
    src_edited[0, 2] = (src[0, 2] + 1) % 31 + 1
    tgt = torch.randint(1, 32, (1, 4))

    with torch.no_grad():
        a = model(src, tgt)
        b = model(src_edited, tgt)

    # ALL target positions see the change — the source has no "future":
    # even tgt position 0 may consult src position 2.
    assert (~torch.isclose(a, b)).any(dim=-1).all(), "some tgt position ignored the source"


def test_encoder_is_bidirectional_on_purpose():
    """Editing a LATE source token changes the memory at EARLY source
    positions — the encoder has no causal mask, and that is correct: the
    source is given, not generated, so 'future' is meaningless there.
    (Contrast with the decoder test above — same edit shape, opposite
    correct answer.)"""
    model = make_model()
    src = torch.randint(1, 32, (1, 6))
    src_edited = src.clone()
    src_edited[0, 5] = (src[0, 5] + 1) % 31 + 1  # edit the LAST source token

    with torch.no_grad():
        mem_a = model.encode(src)
        mem_b = model.encode(src_edited)

    assert not torch.allclose(mem_a[:, 0], mem_b[:, 0])  # position 0 saw it


def test_padding_is_invisible_end_to_end():
    """Lesson 05's batching claim, now through the whole machine: padding
    the source AND the target must leave logits at real target positions
    unchanged. This exercises all three mask sites at once (encoder self,
    decoder self, cross) — if any one forgot padding, this fails."""
    model = make_model()
    src = torch.randint(1, 32, (1, 5))
    tgt = torch.randint(1, 32, (1, 4))
    src_padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)  # (1, 8)
    tgt_padded = torch.cat([tgt, torch.full((1, 2), PAD)], dim=1)  # (1, 6)

    with torch.no_grad():
        clean = model(src, tgt)
        padded = model(src_padded, tgt_padded)

    assert torch.allclose(clean, padded[:, :4], atol=1e-5)


# ------------------------------------------------- training readiness

def test_gradients_reach_every_parameter():
    """Assembly-level dead-wiring check: one backward pass from a plausible
    loss must touch every parameter — both embeddings, all three attentions
    per decoder layer, the head. A parameter with no gradient is a component
    the wiring silently dropped (a classic assembly bug: e.g. building a
    module but forgetting to call it in forward)."""
    model = make_model()
    src = torch.randint(1, 32, (2, 6))
    tgt = torch.randint(1, 32, (2, 5))

    logits = model(src, tgt)
    # Stand-in for lesson 07's cross-entropy: any loss mixing all positions.
    logits.logsumexp(dim=-1).mean().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient at {name}"


def test_embedding_scale_balances_positional_encoding():
    """The §3.4 detail, pinned (closes lesson 04's open thread): with
    linear-scale init (std 1/√d), a raw embedding row has norm ≈ 1 while a
    PE row has norm √(d/2) — position would drown content ~16× at d=512.
    The forward-pass ×√d_model lifts content to (slightly above) position's
    volume: same order of magnitude, ratio √2."""
    torch.manual_seed(0)
    d = 512
    emb = TokenEmbedding(vocab_size=1000, d_model=d)
    ids = torch.arange(100)

    scaled_norm = emb(ids).norm(dim=-1).mean()               # ≈ √d
    raw_norm = emb.table(ids).norm(dim=-1).mean()            # ≈ 1
    pe_norm = sinusoidal_table(100, d).norm(dim=-1).mean()   # = √(d/2)

    assert abs(raw_norm - 1.0) < 0.2                  # the problem
    assert abs(pe_norm - (d / 2) ** 0.5) < 1e-3       # what it must match
    assert 1.0 < scaled_norm / pe_norm < 2.0          # the fix: same volume
