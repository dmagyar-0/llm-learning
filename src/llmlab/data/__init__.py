"""Datasets and tokenizers.

Here now:
  - `toy.py` — on-the-fly random copy/reverse tasks (lesson 07), where the
    answer key is known so training bugs have nowhere to hide.
  - `bpe.py` — our from-scratch byte-level Byte-Pair Encoding tokenizer
    (lesson 12; Sennrich 2015 + GPT-2's byte base). Turns real text ↔ token ids
    with no <unk>, ever.

Planned: TinyShakespeare / TinyStories-scale corpora to actually train the
GPT-2 recipe (BPE → small GPT-2 on the home GPU).
"""
