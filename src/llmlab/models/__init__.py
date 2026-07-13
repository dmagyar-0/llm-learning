"""Architectures assembled from llmlab.components.

Planned: transformer.py (Vaswani 2017 encoder-decoder), gpt2.py, llama.py,
mixtral-style MoE variant. Each model is a config dataclass + an assembly of
registered components; two models should differ in config, not in code shape.
Every model ships a `tiny` config (CPU, seconds) and a `small` config (home GPU).
"""
