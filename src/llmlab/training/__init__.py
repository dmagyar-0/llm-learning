"""Training infrastructure.

Here now (lesson 07): `loss.py` — masked cross-entropy, the LLM objective;
`loop.py` — the plain five-line training loop with Adam. Lesson 08 adds
`free_running_accuracy` to `loop.py`, the honest inference-time metric that
pairs with `teacher_forced_accuracy` (the generation loop itself lives on the
model as `Transformer.greedy_decode`). `python -m llmlab.training.loop` trains
the reverse task and then *generates* it.

Planned: AdamW + warmup/cosine schedule (as used by GPT-2/LLaMA), gradient
clipping, and checkpoint/resume — required because our remote sessions are
ephemeral and the home GPU runs are short.
"""
