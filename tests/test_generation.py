"""Tests for lesson 08 — autoregressive generation (greedy decoding).

Lesson 07 proved the model can score the next token given the true prefix.
Lesson 08 makes it run on its OWN prefix. The claims under test:

    - greedy_decode's mechanics: it starts every row with BOS, appends one
      token per step, stops rows at their first EOS, freezes finished rows to
      PAD, and reuses a single source encoding
    - greedy IS argmax teacher forcing when you feed a prefix the model would
      itself have produced — i.e. the loop just re-selects the same tokens
    - end to end: a model trained to solve copy also GENERATES copy correctly
      (free-running exact-match is high), the honest inference-time proof that
      teacher-forced accuracy was not hiding exposure bias on this easy task
"""

import torch

from llmlab.data.toy import BOS, EOS, PAD, ToyTaskConfig, make_batch
from llmlab.models.transformer import Transformer, TransformerConfig
from llmlab.training.loop import (
    free_running_accuracy,
    teacher_forced_accuracy,
    train_toy,
)


def tiny_model(vocab=16, seed=0):
    torch.manual_seed(seed)
    return Transformer(TransformerConfig.tiny(vocab, vocab))


# ------------------------------------------------- the decoding mechanics

def test_greedy_decode_starts_with_bos_and_has_expected_shape():
    """Every generated row is seeded with BOS (the decoder's given prefix) and
    grows by at most `max_new_tokens` columns on top of it."""
    model = tiny_model()
    src = torch.randint(3, 16, (4, 6))
    out = model.greedy_decode(src, bos_id=BOS, eos_id=EOS, max_new_tokens=10)
    assert (out[:, 0] == BOS).all()
    assert out.shape[0] == 4
    assert 1 < out.shape[1] <= 1 + 10   # BOS + up to max_new_tokens


def test_finished_rows_are_frozen_to_pad_after_eos():
    """Once a row emits EOS it must stop producing content — every later
    position is PAD. Without this latch a row would ramble past its own stop
    signal and the batch could never agree on when it is done."""
    model = tiny_model(seed=3)
    src = torch.randint(3, 16, (16, 5))
    out = model.greedy_decode(src, bos_id=BOS, eos_id=EOS, max_new_tokens=12)
    for row in out:
        eos_positions = (row == EOS).nonzero().flatten()
        if len(eos_positions):
            first_eos = int(eos_positions[0])
            assert (row[first_eos + 1:] == PAD).all()


def test_greedy_is_deterministic():
    """Greedy decoding takes the argmax — no sampling — so the same input must
    give byte-identical output every call (eval mode, no dropout)."""
    model = tiny_model(seed=1)
    src = torch.randint(3, 16, (3, 7))
    a = model.greedy_decode(src, bos_id=BOS, eos_id=EOS, max_new_tokens=9)
    b = model.greedy_decode(src, bos_id=BOS, eos_id=EOS, max_new_tokens=9)
    assert torch.equal(a, b)


def test_greedy_step_matches_teacher_forced_argmax_on_same_prefix():
    """The generation loop adds no new math: feeding a prefix through the full
    teacher-forcing pass and taking the last position's argmax gives the SAME
    token greedy_decode picks at that step. Generation is 'forward, take argmax,
    append' — the only new thing is *where the prefix comes from* (itself)."""
    model = tiny_model(seed=2)
    src = torch.randint(3, 16, (5, 6))

    out = model.greedy_decode(src, bos_id=BOS, eos_id=EOS, max_new_tokens=1)
    generated_first = out[:, 1]                       # the one token it produced

    prefix = torch.full((5, 1), BOS)                  # the same [BOS] prefix
    logits = model(src, prefix)                       # teacher-forced pass
    tf_first = logits[:, -1].argmax(dim=-1)           # argmax at last position
    assert torch.equal(generated_first, tf_first)


# ------------------------------------------------- end to end, after training

def test_trained_model_generates_copy_correctly():
    """The headline: a model trained on copy doesn't just SCORE next tokens
    well (teacher-forced) — set loose on its own output it GENERATES the right
    sequences (free-running exact-match). On this easy task the two accuracies
    stay close, i.e. exposure bias is small here; the gap is what widens on the
    harder tasks and deeper models of later phases."""
    cfg = ToyTaskConfig(task="copy")
    model = tiny_model(vocab=cfg.vocab_size)

    train_toy(model, cfg, steps=400, batch_size=64, lr=1e-3)

    tf = teacher_forced_accuracy(model, cfg)
    fr = free_running_accuracy(model, cfg)
    assert tf > 0.95, f"teacher-forced accuracy only {tf:.3f}"
    assert fr > 0.85, f"free-running exact-match only {fr:.3f}"


def test_generated_sequences_terminate_with_eos():
    """A trained generator must learn to STOP: on the copy task nearly every
    row should emit EOS within the budget rather than running to the cap. This
    is why lesson 07 put EOS in the labels — a model never taught to predict it
    would generate forever."""
    cfg = ToyTaskConfig(task="copy")
    model = tiny_model(vocab=cfg.vocab_size)
    train_toy(model, cfg, steps=400, batch_size=64, lr=1e-3)

    gen = torch.Generator().manual_seed(99)
    src, _, tgt_out = make_batch(cfg, 128, generator=gen)
    out = model.greedy_decode(src, bos_id=BOS, eos_id=EOS,
                              max_new_tokens=tgt_out.shape[1])
    emitted_eos = (out == EOS).any(dim=1)
    assert emitted_eos.float().mean() > 0.95
