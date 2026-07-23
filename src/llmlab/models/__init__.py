"""Architectures assembled from llmlab.components.

Built: transformer.py (Vaswani 2017 encoder–decoder), gpt.py (decoder-only LM).
Planned: llama.py, mixtral-style MoE variant. Each model is a config dataclass +
an assembly of registered components; two models should differ in config, not in
code shape. Every model ships a `tiny` config (CPU, seconds) and a `small` config
(home GPU).
"""

from llmlab.models.gpt import GPT, GPTConfig
from llmlab.models.transformer import Transformer, TransformerConfig

__all__ = ["GPT", "GPTConfig", "Transformer", "TransformerConfig"]
