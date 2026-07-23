# 12 — Byte-Pair Encoding: our own tokenizer

**Phase:** 2 (GPT-2 recipe) · **Paper(s):** Sennrich, Haddow & Birch 2015 (BPE for NMT — [notes](../papers/bpe.md)); GPT-2 §2.2 (Radford et al. 2019, byte-level BPE)
**Code:** `src/llmlab/data/bpe.py` (`BPETokenizer`, `get_stats`, `merge`) · **Test:** `tests/test_bpe.py`

## The problem

Every lesson so far took token *ids* as given — lesson 07's toy task invented random
ids, lesson 06's embedding just looked them up. But real input is a **string**.
Something has to turn `"hello world"` into ids and back, and that something — the
**tokenizer** — is a fixed table learned once and then frozen. It is not part of the
neural net, but it is part of *the model*: change it and every id means something
different, so all the weights become meaningless. Before we can train on real text
(the next lessons), we need it.

How do you cut text into a **fixed vocabulary**? The output softmax is a distribution
over that vocab, so it can't be open-ended. Three answers:

- **Word-level.** Vocab = words. The vocab is huge and still open-ended (every name,
  typo, emoji is a new word), and anything unseen becomes `<unk>` — information
  destroyed before the model sees it. `"unhappiness"` and `"unhappy"` are unrelated.
- **Character-level.** Vocab = alphabet (what lesson 07 effectively was). No `<unk>`
  within a known alphabet and a tiny vocab — but sequences get very long (one id per
  char) and each id means almost nothing, so the model burns capacity on spelling.
- **Subword (BPE).** The middle path: start small and *learn* which adjacent pairs to
  glue into new symbols. Frequent chunks (`"the"`, `"ing"`, `" world"`) become single
  ids; rare words fall back to pieces. Common text → few tokens, rare text → more
  tokens, but **never `<unk>`.** This is what GPT-2 and essentially every LLM uses.

## The idea

BPE is a 1994 *compression* algorithm Sennrich repurposed to *build a vocabulary*.
Three moves, and the whole file is just these:

1. **Count** every adjacent pair of ids in the corpus.
2. **Merge** the single most frequent pair everywhere, minting it a new id. Record it.
3. **Repeat** until the vocab hits the target size. The number of merges *is* the
   vocab-size dial — 0 merges = character-level, many = word-level.

The twist we copy from **GPT-2: the base alphabet is the 256 UTF-8 byte values**, not
characters. This is the move that kills `<unk>` *forever, for anything* — every
possible string is a byte sequence, and every byte 0–255 is already in the vocab.
Unseen words, unseen scripts, emoji, corrupted input: all representable. The price is
that a non-ASCII character is 2–4 bytes (several base tokens) until merges glue it
back — which training does for anything frequent.

## The math

There isn't equation-math here, there's algorithm-math — three functions.

**Count** (`get_stats`). For ids `[7, 7, 9, 7]` the adjacent pairs are `(7,7), (7,9),
(9,7)`. BPE greedily maximizes this frequency table: the highest-count pair is the
next merge, because merging it removes the most symbols per new vocab slot.

**Merge** (`merge`). Replace a pair with a new id, **left-to-right, non-overlapping**:
`[5,5,5]` merging `(5,5)` → `[256,5]` (jump two on a match). That greedy-left rule is
a *convention* — what matters is that training and `encode` use the *same* `merge`, so
they segment identically.

**The train loop.** Merges **compound**, and that's the crux: we merge on the
already-merged stream, so once `(t,h)→256` exists, a later pass can merge `(256,e)` to
form `"the"`. Multi-character symbols are built recursively out of single bytes:

    ids = list(text.encode("utf-8"))          # base ids in [0, 256)
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair  = max(stats, key=stats.get)      # most frequent adjacent pair
        new   = 256 + i
        ids   = merge(ids, pair, new)          # apply everywhere
        merges[pair]   = new                   # dict order == learned priority
        vocab[new]     = vocab[pair[0]] + vocab[pair[1]]   # bytes concatenated

So `vocab_size = 256 + (number of merges)`; GPT-2 used `50257 = 256 + 50000 + 1`
special token.

**The subtle bit — `encode` must replay merges in learned order.** Later merges assume
earlier ones already happened, so each round we apply the **earliest-learned** eligible
merge (smallest merge id), then recount:

    while len(ids) >= 2:
        stats = get_stats(ids)
        pair  = min(stats, key=lambda p: merges.get(p, inf))  # earliest merge wins
        if pair not in merges: break                          # nothing left to glue
        ids = merge(ids, pair, merges[pair])

Applying a *late* merge before an early one would produce id sequences that never
occurred during training — the model would see gibberish segmentation.

**Decode** joins each id's bytes and UTF-8-decodes the **whole blob at once** (not per
token: one character can span several tokens, and its bytes are only valid UTF-8 when
joined), with `errors="replace"` to survive a hand-built invalid id list.

## The code

The three functions above are the entire algorithm; `BPETokenizer` just holds the two
tables they fill:

    merges: dict[(int,int) -> int]   learned merges, insertion order = priority
    vocab:  dict[int -> bytes]       every id's byte string, for decoding

A nice property that fell out of the design: an **untrained tokenizer is already a
valid byte tokenizer** — with no merges, `encode` returns the raw UTF-8 bytes and
`decode` inverts them (test: `test_untrained_is_a_plain_byte_tokenizer`). Training only
*adds* higher ids on top of the 256-byte base; it never removes the base, which is the
mechanical reason there is never an `<unk>`.

## What breaks without it

- **`encode` ignores merge order** (e.g. greedily merges the *most frequent* present
  pair instead of the *earliest-learned*): you get a different segmentation than
  training produced, so the model sees id patterns it never trained on. Round-trip
  still works, but the tokens are wrong. Hence `min(..., merges.get(p, inf))`.
- **Decode per-token instead of joining first:** a multi-byte character split across
  two tokens has each half be invalid UTF-8 alone — you'd get replacement chars or a
  crash. Join all bytes, decode once.
- **Character base instead of byte base:** you're back to `<unk>` for any unseen
  character (test `test_never_emits_unknown_on_unseen_input` would fail on the emoji /
  Japanese). The byte base is the whole robustness story.
- **`merge` overlapping or right-to-left:** inconsistent segmentation between examples;
  `[5,5,5]` could ambiguously become `[5,256]` vs `[256,5]`. Pin one convention and
  share it between train and encode.

## Open questions

- **No regex pre-tokenizer yet.** GPT-2 first splits text into chunks (words, numbers,
  punctuation runs, leading-space+word) and only merges *within* a chunk — so a merge
  can't glue a period onto every word, while `" the"` becomes a natural unit. Ours can
  merge across word boundaries. That regex split is the clean next refinement of this
  file (Karpathy's `minbpe` calls ours the "basic" tokenizer, that one the "regex").
- **Special tokens.** `<|endoftext|>` (document separator) and, for Phase 7, chat-format
  tokens are reserved ids *outside* the merge process — needed before a real training
  run.
- **Save/load.** A trained tokenizer must persist alongside the weights; right now it
  lives only in memory. A tiny `save`/`load` of the merges is a small future add.
- **Where it plugs in.** Next we point this at a real corpus (TinyShakespeare/Stories),
  `encode` the text to an id stream, and feed lesson 09's GPT the shift-by-one of that
  stream — the first time our model sees actual language instead of random ids.
