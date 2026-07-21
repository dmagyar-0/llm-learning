"""Datasets and tokenizers.

Here now: `toy.py` — on-the-fly random copy/reverse tasks (lesson 07), where
the answer key is known so training bugs have nowhere to hide.

Planned: character-level tokenizer + TinyShakespeare (zero dependencies,
trains on CPU), then our own from-scratch BPE (Sennrich 2015, as used by
GPT-2), then TinyStories-scale data for the home GPU.
"""
