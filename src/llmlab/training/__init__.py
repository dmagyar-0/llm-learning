"""Training infrastructure.

Planned: a plain training loop we fully understand (no Trainer frameworks),
AdamW + warmup/cosine schedule (as used by GPT-2/LLaMA), gradient clipping,
and checkpoint/resume — required because our remote sessions are ephemeral
and the home GPU runs are short.
"""
