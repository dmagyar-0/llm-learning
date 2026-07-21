"""The training loop, written plainly — lesson 07.

No Trainer framework, no callbacks: the loop that trains a transformer is
five lines, and this file's job is to make sure we always know which five.
Everything else (schedules, clipping, checkpointing, mixed precision) will be
added in later lessons AS its failure mode appears — per papers/training.md,
we take from Vaswani §5 only what a 2-layer model on a toy task needs: Adam.

The five lines, annotated:

    logits = model(src, tgt_in)          # forward: teacher forcing (lesson 06)
    loss = masked_cross_entropy(...)     # scalar objective (training/loss.py)
    optimizer.zero_grad()                # else .backward() ADDS to old grads
    loss.backward()                      # autograd: d loss / d every parameter
    optimizer.step()                     # Adam: params -= lr · m̂/√v̂ per coord

Why zero_grad at all: PyTorch *accumulates* gradients into .grad on purpose
(it enables gradient accumulation across micro-batches — how big models fake
big batches, a trick we'll want on the home GPU). The price of that feature
is that forgetting to zero silently sums gradients across steps — the loss
still falls at first, which is what makes it a classic hard-to-spot bug.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from llmlab.data.toy import BOS, EOS, ToyTaskConfig, make_batch
from llmlab.models.transformer import Transformer, TransformerConfig
from llmlab.training.loss import masked_cross_entropy


@dataclass
class TrainLog:
    """What we keep from a run: enough to plot and to assert on.

    losses[k] is the loss at step k — on a FRESH batch, so this curve is
    honest generalization (nothing repeats, nothing can be memorized).
    """

    losses: list[float] = field(default_factory=list)


@torch.no_grad()
def teacher_forced_accuracy(
    model: Transformer, cfg: ToyTaskConfig, batch_size: int = 256, seed: int = 1234
) -> float:
    """Fraction of real target positions where argmax(logits) is the label.

    "Teacher-forced" is a real caveat: position t predicts while SEEING the
    true tokens up to t, so this measures per-step prediction, not the
    model's ability to run free on its own output — that loop (and how the
    two can differ) is lesson 08. For a toy task, ≈100% here means the
    information plumbing works end to end.
    """
    model.eval()  # disables dropout; harmless now (p=0), a bug-in-waiting later
    gen = torch.Generator().manual_seed(seed)
    src, tgt_in, tgt_out = make_batch(cfg, batch_size, generator=gen)
    pred = model(src, tgt_in).argmax(dim=-1)      # (batch, seq): best token id
    real = tgt_out != model.config.pad_id         # score only real labels
    return ((pred == tgt_out) & real).sum().item() / real.sum().item()


@torch.no_grad()
def free_running_accuracy(
    model: Transformer, cfg: ToyTaskConfig, batch_size: int = 256, seed: int = 1234
) -> float:
    """Fraction of examples the model generates *entirely* correctly on its own.

    The honest counterpart to `teacher_forced_accuracy`, and the whole reason
    lesson 08 exists. There the model saw the TRUE prefix at every step; here it
    sees only its OWN previous outputs (`greedy_decode`), exactly as at
    deployment. The two metrics answer different questions:

        teacher-forced : "given the right prefix, is the next token right?"
                         — per-token, forgiving: one slip costs one token.
        free-running   : "left alone, does it produce the right sequence?"
                         — whole-sequence, unforgiving: one early slip feeds
                           itself and can derail everything after it.

    That second effect is **exposure bias** — the model was only ever trained on
    gold prefixes, so its own imperfect prefixes are mildly out-of-distribution,
    and errors compound. On a fully-solved toy task the gap is small; it widens
    on harder tasks, deeper into a sequence, and on under-trained models. We
    score exact-match (the generated answer, BOS-stripped, must equal the
    labels) because a generator that gets 7 of 8 tokens produced the WRONG
    sequence — partial credit is a teacher-forcing luxury.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    src, _, tgt_out = make_batch(cfg, batch_size, generator=gen)

    # Give it room for the whole answer plus EOS (labels are answer+EOS already).
    produced = model.greedy_decode(
        src, bos_id=BOS, eos_id=EOS, max_new_tokens=tgt_out.shape[1]
    )
    hyp = produced[:, 1:]                       # drop the BOS seed → compare to labels
    pad = model.config.pad_id

    # Pad hyp and labels to a common width so a length mismatch counts as a
    # mismatch (a short generation is a wrong generation), not a crash.
    width = max(hyp.shape[1], tgt_out.shape[1])
    hyp = torch.nn.functional.pad(hyp, (0, width - hyp.shape[1]), value=pad)
    ref = torch.nn.functional.pad(tgt_out, (0, width - tgt_out.shape[1]), value=pad)

    exact = (hyp == ref).all(dim=1)            # (batch,) whole-row correctness
    return exact.float().mean().item()


def train_toy(
    model: Transformer,
    cfg: ToyTaskConfig,
    steps: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int | None = None,
) -> TrainLog:
    """Train `model` on fresh toy batches; return the loss history.

    lr=1e-3 is Adam's classic default and fine here; it is NOT fine at depth
    — the paper's 6-layer post-norm stack needs warmup from lr≈0 or it
    diverges (papers/residuals-layernorm.md). We will hit that wall on
    purpose in Phase 2.

    The first loss should be ≈ ln(vocab_size): an untrained model knows
    nothing, softmax over random logits is near-uniform, and uniform over V
    choices costs ln V nats. If step-0 loss is far from ln V something is
    broken *before* training (wrong shapes silently broadcast, wrong labels);
    this "loss sanity anchor" is the cheapest debugging tool in deep learning.
    """
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = TrainLog()

    model.train()
    for step in range(steps):
        src, tgt_in, tgt_out = make_batch(cfg, batch_size, generator=generator)

        logits = model(src, tgt_in)                    # (batch, tgt_seq, vocab)
        loss = masked_cross_entropy(logits, tgt_out, model.config.pad_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:4d}  loss {loss.item():.4f}")
    return log


def _demo() -> None:
    """`python -m llmlab.training.loop` — watch a transformer learn to reverse.

    Reverse (not copy) because it is the satisfying one: the model must find
    the end of a variable-length sequence — knowable only through the padding
    mask — and read the source backwards from there. Takes ~a minute on CPU.
    """
    torch.manual_seed(0)
    cfg = ToyTaskConfig(task="reverse")
    model = Transformer(TransformerConfig.tiny(cfg.vocab_size, cfg.vocab_size))

    print(f"task: {cfg.task}, expect initial loss ≈ ln({cfg.vocab_size}) = "
          f"{torch.tensor(float(cfg.vocab_size)).log():.3f}")
    train_toy(model, cfg, steps=2000, log_every=100)
    # Two accuracies, side by side — the point of lesson 08. Teacher-forced
    # sees gold prefixes (per-token); free-running feeds itself (exact-match).
    print(f"teacher-forced accuracy: {teacher_forced_accuracy(model, cfg):.1%}")
    print(f"free-running  accuracy: {free_running_accuracy(model, cfg):.1%}")

    # Watch it actually generate: BOS in, tokens out, one step at a time, until
    # it decides (by emitting EOS) that it is done.
    gen = torch.Generator().manual_seed(7)
    src, _, tgt_out = make_batch(cfg, batch_size=3, generator=gen)
    produced = model.greedy_decode(src, bos_id=BOS, eos_id=EOS,
                                   max_new_tokens=tgt_out.shape[1])
    for i in range(3):
        print(f"src {src[i].tolist()}  want {tgt_out[i].tolist()}  "
              f"generated {produced[i].tolist()}")


if __name__ == "__main__":
    _demo()
