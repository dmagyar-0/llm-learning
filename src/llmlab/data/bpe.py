"""Byte-Pair Encoding — our own tokenizer (lesson 12, Phase 2).

Everything so far consumed integer token *ids* and never asked where they came
from (lesson 07's toy task invented them; lesson 06's embedding just looked them
up). Real text is a string. Something must turn "hello world" into a list of ids
the model can embed, and turn ids back into text. That something is the
**tokenizer**, and it is *not* part of the neural network — it is a fixed,
learned-once preprocessing table that the model is then trained against. Get it
wrong and every downstream id means something different; the tokenizer is as much
"the model" as the weights are.

Three ways one could split text into a fixed vocabulary, and why BPE wins:

- **Word-level.** Vocab = the words in your corpus. Two fatal problems: the vocab
  is huge and open-ended (every name, typo, hashtag is a new word), and any word
  unseen at training time is an **<unk>** — information destroyed before the model
  even sees it. "unhappiness" and "unhappy" are unrelated ids.
- **Character-level.** Vocab = the alphabet (what lesson 07's toy data effectively
  was). No <unk> within a known alphabet, tiny vocab — but sequences get very long
  (one id per character) and each id carries almost no meaning, so the model spends
  its capacity relearning spelling. And a truly unseen character (some emoji) is
  still <unk>.
- **Subword (BPE).** Meet in the middle: start from a small base alphabet and
  *learn* which adjacent pairs to glue into new symbols, keeping the frequent
  chunks ("the", "ing", " world") as single ids while rare words fall back to
  their pieces. Common text → few tokens; rare text → more tokens, but never
  <unk>. This is the tokenizer GPT-2 (and essentially every LLM since) uses.

**Why bytes as the base alphabet (the GPT-2 choice we copy).** Sennrich's original
BPE (2015) started from characters. GPT-2 starts from the **256 possible byte
values** of UTF-8. That one decision kills <unk> *forever*: any string whatsoever —
any language, emoji, control character, corrupted input — is a sequence of bytes,
and every byte 0..255 is already in the vocabulary. There is literally no input the
tokenizer cannot represent. The cost is that a non-ASCII character is several bytes
(so several base tokens) until the merges glue it back together, which training
happily does for anything frequent.

The algorithm itself is three moves, and this whole file is just those three:

1. **Count** every adjacent pair of ids in the corpus (`get_stats`).
2. **Merge** the single most frequent pair everywhere, minting it a new id
   (`merge`). Record the merge.
3. Repeat until the vocab reaches the target size. `train` is this loop; `encode`
   replays the learned merges on new text; `decode` inverts the id→bytes table.

We follow the structure of Karpathy's `minbpe` "basic" tokenizer: byte-level BPE
with no regex pre-tokenizer yet (that refinement — stopping merges from spanning
word boundaries — is its own later step; see the lesson's open questions).
"""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs — step 1 of BPE.

    For ids [7, 7, 9, 7] the pairs are (7,7), (7,9), (9,7), each counted once
    per occurrence. This is the frequency table BPE greedily maximizes over:
    the pair with the highest count is the next merge, because merging it buys
    the most compression per new vocabulary slot.
    """
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):  # every overlapping adjacent pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id` — step 2 of BPE.

    [7, 7, 9, 7], pair=(7,9), new_id=256  →  [7, 256, 7].

    Left-to-right and non-overlapping: on a match we jump two positions, so in
    [5, 5, 5] merging (5,5) yields [256, 5], not [256] and not an ambiguous
    overlap. That greedy left bias is a real convention (encode below reuses the
    same `merge`, so training and inference glue identically — the only way the
    ids stay consistent).
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        # Match only if a full pair starts here (guard against i being the last
        # index, where there is no ids[i+1]).
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # consume BOTH members of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level Byte-Pair Encoding (Sennrich 2015 + GPT-2's byte base).

    State is two small tables, both built by `train`:

        merges: dict[(int, int) -> int]   the learned merges, in creation order
                                          (dict insertion order == priority)
        vocab:  dict[int -> bytes]        every id's byte string, for decoding

    An untrained tokenizer is already valid: with no merges it is a plain **byte
    tokenizer** (encode = the raw UTF-8 bytes, decode = UTF-8 decode). Training
    only *adds* higher ids on top of the 256 byte ids — it never removes the base,
    which is why there is never an <unk>.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        # Base vocabulary: id i is literally the one-byte string bytes([i]).
        # This is the whole 0..255 alphabet, present before any training.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ train

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text` — step 3, the loop.

        `vocab_size` is the FINAL vocabulary: 256 base bytes plus the merges we
        mint, so it must be ≥ 256. GPT-2 used 50257 (50000 merges + 256 bytes +
        1 special <|endoftext|>); our tiny configs use a few hundred.

        Each iteration greedily commits the most frequent adjacent pair. Crucially
        we merge on the *already-merged* id stream, so merges compound: once
        (t,h)→256 exists, a later pass can merge (256, e) to make "the" — new
        symbols are built out of old ones, which is how BPE reaches multi-char
        chunks from single bytes.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be ≥ 256 (the byte base), got {vocab_size}")
        num_merges = vocab_size - 256

        # Start from raw bytes: the corpus as a list of base ids in [0, 256).
        ids = list(text.encode("utf-8"))

        # Fresh tables each train() call (idempotent — retraining replaces).
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # corpus collapsed to a single token; nothing left to pair
            # Most frequent pair. Ties break by first appearance (dict order),
            # so training is deterministic for a given corpus.
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            # The new symbol's bytes = its two parts concatenated. Recursive by
            # construction: parts may themselves be merged symbols.
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}) had {stats[pair]} occurrences")

    # ------------------------------------------------------------------ use

    def encode(self, text: str) -> list[int]:
        """Text → token ids, by replaying the learned merges in priority order.

        The subtle part: we must apply merges in the SAME order training created
        them, because later merges assume earlier ones already happened (the "the"
        example above). So each round we find the eligible pair with the smallest
        merge id — i.e. the earliest-learned merge — and apply just that one, then
        recount. Applying a late merge before an early one would produce ids that
        never occurred in training.
        """
        ids = list(text.encode("utf-8"))  # start from raw bytes, like training
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Among pairs present, pick the one learned earliest. Unmerged pairs
            # get +inf so they are never chosen; when the min is +inf, nothing
            # here is mergeable and we stop.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids → text, by looking up each id's bytes and UTF-8 decoding.

        Concatenate the byte strings, then decode the whole blob at once — NOT
        per token, because one character can span several tokens (a multi-byte
        char split across merges), and its bytes only form valid UTF-8 when
        joined. `errors="replace"` guards the rare case of an id sequence whose
        bytes aren't valid UTF-8 (e.g. a hand-built id list), emitting the U+FFFD
        replacement char instead of crashing.
        """
        blob = b"".join(self.vocab[i] for i in ids)
        return blob.decode("utf-8", errors="replace")
