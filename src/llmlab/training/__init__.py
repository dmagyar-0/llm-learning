"""Training infrastructure.

Here now (lesson 07): `loss.py` — masked cross-entropy, the LLM objective;
`loop.py` — the plain five-line training loop with Adam
(`python -m llmlab.training.loop` runs the reverse-task demo).

Planned: AdamW + warmup/cosine schedule (as used by GPT-2/LLaMA), gradient
clipping, and checkpoint/resume — required because our remote sessions are
ephemeral and the home GPU runs are short.
"""
