"""Toy sequence-to-sequence tasks — lesson 07's training data.

Before real text we train on a task whose *answer key we know*: given a random
sequence of tokens, output it copied (or reversed). This is the classic first
test of a seq2seq model — if the architecture and training loop are wired
correctly, a tiny transformer nails it on CPU in seconds; if any mask, shift,
or loss detail is wrong, it visibly can't. Random tokens carry no linguistic
statistics, so the ONLY way to score well is to actually move information
from the source through cross-attention — there is nothing to memorize.

Why the two tasks teach different things:

- **copy**: target position t must find source position t. A pure positional
  alignment — the easiest possible use of cross-attention (attend to your own
  position's key, read its value).
- **reverse**: target position t must find source position L−1−t, where L is
  *this sequence's* length — which varies per example. Position alone is not
  enough; the model must infer where the sequence ends (only the padding mask
  reveals it) and count backwards. Same data, same model, noticeably harder.

Special tokens (a convention we keep for the rest of the repo):

    PAD=0  fills ragged batches to rectangles; masked everywhere (lesson 05)
    BOS=1  seeds the decoder — the "given prefix" for predicting token 0
    EOS=2  what the model must PREDICT after the last real token, so that at
           generation time (lesson 08) it can say "I'm done" — without EOS a
           generator never learns to stop
    3..vocab_size−1  the actual content alphabet

The teacher-forcing layout (Vaswani §3.1 "shifted right"; papers/training.md):
one example with content y₁..y_L becomes an (input, label) pair *offset by one*:

    decoder input  tgt_in  = [BOS, y₁, ... , y_L]      what the model SEES
    loss labels    tgt_out = [y₁, ... , y_L, EOS]      what it must PREDICT

so position t sees prefix [BOS, y₁..y_t] and is scored on y_{t+1} — every
position is a genuine next-token problem, all solved in one causally-masked
pass. Both are padded with PAD to the batch's longest L+1; the loss masks
those label positions out (training/loss.py).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

PAD, BOS, EOS = 0, 1, 2
NUM_SPECIALS = 3


@dataclass
class ToyTaskConfig:
    """A family of random copy/reverse problems.

    vocab_size counts specials + content: content tokens are drawn uniformly
    from [3, vocab_size). Lengths are drawn uniformly from [min_len, max_len]
    — the point of *variable* lengths is to force real padding into every
    batch, so the pad-handling we built (masks) and build here (loss masking)
    is actually exercised, not just present.
    """

    vocab_size: int = 16
    min_len: int = 2
    max_len: int = 8
    task: str = "copy"  # or "reverse"

    def __post_init__(self) -> None:
        if self.task not in ("copy", "reverse"):
            raise ValueError(f"unknown task {self.task!r}")
        if self.vocab_size <= NUM_SPECIALS:
            raise ValueError("vocab_size must leave room for content tokens")


def make_batch(
    cfg: ToyTaskConfig,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a fresh batch: (src, tgt_in, tgt_out).

    src:     (batch, max_len)      content tokens, PAD after each row's length
    tgt_in:  (batch, max_len + 1)  BOS + answer, PAD-filled — the decoder input
    tgt_out: (batch, max_len + 1)  answer + EOS, PAD-filled — the loss labels

    The +1 is EOS's seat: an L-token answer yields L+1 predictions (L tokens,
    then "stop"). Note tgt_in and tgt_out are the SAME content at a one-token
    offset — teacher forcing is this shift, nothing more.

    We generate data on the fly instead of storing a dataset: the task
    distribution is tiny code, every batch is fresh (no example repeats, so
    train loss IS generalization — nothing to overfit), and there is nothing
    to download. Real-text lessons (Phase 2) will need actual datasets.

    An explicit torch.Generator (rather than the global seed) keeps data
    randomness independent of model-init randomness — reproducible batches
    regardless of what else has consumed the global RNG.
    """
    lengths = torch.randint(
        cfg.min_len, cfg.max_len + 1, (batch_size,), generator=generator
    )
    content = torch.randint(
        NUM_SPECIALS, cfg.vocab_size, (batch_size, cfg.max_len), generator=generator
    )
    # (batch, max_len) bool: True at positions < this row's length. The same
    # comparison-against-arange trick as lesson 05's causal mask.
    real = torch.arange(cfg.max_len) < lengths[:, None]

    src = content * real  # PAD (=0) beyond each row's length, content before

    answer = src.flip(dims=[1]) if cfg.task == "reverse" else src
    if cfg.task == "reverse":
        # flip() reversed the whole rectangle, sending the PAD tail to the
        # front ([a b c 0 0] → [0 0 c b a]). Re-left-align each row by
        # sorting its "is padding" flags (stable sort keeps content order):
        # False (real) sorts before True (pad) → [c b a 0 0].
        order = (answer == PAD).to(torch.int8).argsort(dim=1, stable=True)
        answer = answer.gather(1, order)

    tgt_in = torch.full((batch_size, cfg.max_len + 1), PAD)
    tgt_out = torch.full((batch_size, cfg.max_len + 1), PAD)
    tgt_in[:, 0] = BOS
    tgt_in[:, 1:] = answer                            # [BOS, y₁ ... y_L, PAD...]
    tgt_out[:, : cfg.max_len] = answer
    tgt_out[torch.arange(batch_size), lengths] = EOS  # [y₁ ... y_L, EOS, PAD...]

    return src, tgt_in, tgt_out
