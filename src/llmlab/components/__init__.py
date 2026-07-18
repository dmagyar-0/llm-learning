"""Reusable transformer building blocks.

Planned inhabitants (added one lesson at a time — see docs/ROADMAP.md):

    attention.py    scaled dot-product → multi-head → MQA/GQA → sliding window
    masking.py      causal + padding masks (what makes attention decoder-safe)
    positional.py   sinusoidal → learned → RoPE
    norms.py        LayerNorm (post- and pre-norm placement) → RMSNorm
    ffn.py          GELU MLP (GPT-2) → SwiGLU (LLaMA) → MoE (Mixtral)
    embeddings.py   token embeddings, weight tying

Every class documents the paper it comes from and annotates tensor shapes.
"""
