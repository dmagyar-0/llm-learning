"""Tests for lesson 07 — the toy task, the masked loss, and the training loop.

The claims under test, in dependency order:

    - the data generator emits the exact teacher-forcing layout (shift, BOS,
      EOS placement, PAD tails) — everything downstream trusts this
    - masked cross-entropy: known value on uniform logits, exact agreement
      with PyTorch's ignore_index, and ZERO gradient from pad positions
    - the loop: an untrained model starts at ≈ ln(V) ("knows nothing") and
      300 steps later solves the copy task — the end-to-end proof that
      model + masks + shift + loss + optimizer are wired right together
"""

import math

import torch
import torch.nn.functional as F

from llmlab.data.toy import BOS, EOS, PAD, ToyTaskConfig, make_batch
from llmlab.models.transformer import Transformer, TransformerConfig
from llmlab.training.loop import teacher_forced_accuracy, train_toy
from llmlab.training.loss import masked_cross_entropy


def batch(task="copy", n=8, seed=0, **kw):
    cfg = ToyTaskConfig(task=task, **kw)
    gen = torch.Generator().manual_seed(seed)
    return cfg, *make_batch(cfg, n, generator=gen)


# ------------------------------------------------- the data layout

def test_teacher_forcing_shift():
    """tgt_in and tgt_out must be the SAME sequence offset by one: input
    [BOS, y...], labels [y..., EOS]. Get this off by one and the model is
    asked to predict the token it can already see — trivially perfect loss,
    useless model (the bug lesson 05's causal mask exists to prevent)."""
    _, src, tgt_in, tgt_out = batch()
    assert (tgt_in[:, 0] == BOS).all()
    # Wherever the label row holds real content, input token t+1 must equal
    # label token t (the shift); EOS/PAD tails differ by design. Content can
    # only occupy label columns 0..max_len−1 (column max_len is EOS or PAD),
    # so compare over those columns — matching tgt_in's columns 1..max_len.
    content = (tgt_out[:, :-1] != PAD) & (tgt_out[:, :-1] != EOS)
    assert torch.equal(tgt_in[:, 1:][content], tgt_out[:, :-1][content])


def test_eos_sits_after_the_last_real_token():
    """Each row must teach exactly one 'stop here': EOS at position L (the
    first position after the content), PAD everywhere beyond."""
    _, src, _, tgt_out = batch()
    lengths = (src != PAD).sum(dim=1)
    for row, L in zip(tgt_out, lengths):
        assert row[L] == EOS
        assert (row[L + 1:] == PAD).all()
        assert (row[:L] != PAD).all() and (row[:L] != EOS).all()


def test_reverse_task_reverses():
    """Row-by-row: labels' content must be the source content backwards,
    left-aligned (the flip-and-realign in make_batch is the fiddly part)."""
    _, src, _, tgt_out = batch(task="reverse")
    for s, y in zip(src, tgt_out):
        L = int((s != PAD).sum())
        assert y[:L].tolist() == s[:L].flip(0).tolist()


# ------------------------------------------------- the masked loss

def test_uniform_logits_cost_ln_vocab():
    """The 'knows nothing' anchor, exactly: all-equal logits → softmax is
    uniform over V → −log(1/V) = ln V, whatever the true label is."""
    V = 16
    logits = torch.zeros(2, 5, V)
    labels = torch.randint(3, V, (2, 5))
    assert math.isclose(masked_cross_entropy(logits, labels).item(),
                        math.log(V), rel_tol=1e-6)


def test_agrees_with_pytorch_ignore_index():
    """Our three visible lines vs. F.cross_entropy(ignore_index=PAD) on
    ragged labels: same number. The reference implements the same mean-over-
    real-tokens semantics — pinning equality proves our mask+mean IS the
    standard loss, not an approximation of it."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 16)
    _, _, _, labels = batch(n=4, seed=3)
    labels = labels[:, :7]
    ours = masked_cross_entropy(logits, labels)
    ref = F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1),
                          ignore_index=PAD)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_pad_positions_contribute_zero_gradient():
    """Lesson 05 promised pad positions would be excluded 'where it matters —
    in the loss'. Here is the receipt: d loss / d logits is EXACTLY zero at
    every pad-label position (they multiply by 0 in the masked sum), and
    nonzero at every real one. Whatever garbage the network computes over
    padding, none of it can move a single weight."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 16, requires_grad=True)
    _, _, _, labels = batch(n=4, seed=3)
    labels = labels[:, :7]

    masked_cross_entropy(logits, labels).backward()

    real = labels != PAD                                  # (batch, seq)
    grad_magnitude = logits.grad.abs().sum(dim=-1)        # (batch, seq)
    assert (grad_magnitude[~real] == 0).all()
    assert (grad_magnitude[real] > 0).all()


# ------------------------------------------------- the loop, end to end

def test_transformer_learns_to_copy():
    """The lesson's headline claim: ~300 Adam steps of fresh random batches
    take the tiny transformer from clueless (loss ≈ ln 16 ≈ 2.77) to solving
    the copy task — on CPU, in seconds. Because every batch is fresh, the
    final numbers are generalization, not memorization. Any wiring bug —
    wrong mask, wrong shift, unmasked loss, dead component — shows up here
    as a loss curve that stalls."""
    torch.manual_seed(0)
    cfg = ToyTaskConfig(task="copy")
    model = Transformer(TransformerConfig.tiny(cfg.vocab_size, cfg.vocab_size))

    log = train_toy(model, cfg, steps=300, batch_size=64, lr=1e-3)

    assert abs(log.losses[0] - math.log(cfg.vocab_size)) < 0.5   # sanity anchor
    assert log.losses[-1] < 0.15, f"did not learn: final loss {log.losses[-1]}"
    acc = teacher_forced_accuracy(model, cfg)
    assert acc > 0.95, f"teacher-forced accuracy only {acc:.3f}"
