# BPE: subword tokenization (Phase 2)

Reading note for **Neural Machine Translation of Rare Words with Subword Units**
(Sennrich, Haddow & Birch, 2015 — arXiv 1508.07909), plus the byte-level twist
GPT-2 (Radford et al. 2019) added and that we actually implement (`lesson 12`,
`src/llmlab/data/bpe.py`).

## The problem it solved

2015 neural MT had a **fixed, smallish vocabulary** (say 30–50k words) for a hard
reason: the output softmax is a distribution over the vocab, so its cost and
parameter count scale with vocab size. Anything outside that vocab — rare words,
names, compounds, morphological variants, typos — collapsed to a single `<unk>`
token. That is catastrophic for translation: you literally cannot emit a word you
have no token for, and languages with rich morphology (German compounds, Turkish
agglutination) generate novel words endlessly. Word-level vocab and open-ended
language are fundamentally incompatible.

The two escapes both hurt: grow the vocab (softmax gets expensive, rare words still
barely train) or go character-level (no `<unk>`, but sequences become very long and
the model wastes capacity relearning spelling).

## The idea — learn a subword vocabulary by merging

Sennrich borrowed Byte-Pair Encoding, a 1994 **data-compression** algorithm, and
repurposed it for vocabulary construction:

1. Start with the base alphabet (characters), each word a sequence of symbols.
2. Count every adjacent symbol pair across the corpus.
3. Merge the **most frequent** pair into a new single symbol; record the merge.
4. Repeat for a fixed number of merges — that count *is* the vocab-size knob.

Frequent sequences ("the", "ing", "tion") get merged into whole units; rare words
are left as a handful of subword pieces. So common text costs few tokens, rare
text costs more tokens **but is never `<unk>`** — it always decomposes into pieces,
in the limit down to single characters. One dial (number of merges) slides you
between character-level (0 merges) and word-level (many merges).

The encoder is a greedy replay: take new text, apply the learned merges **in the
order they were learned** (earlier, more-frequent merges first), because later
merges are defined on top of earlier ones. Decoding is trivial — concatenate the
subword strings.

## GPT-2's twist: bytes, not characters

GPT-2 kept the algorithm but changed the **base alphabet** from characters to the
**256 byte values** of UTF-8. This is the version we implement, and it matters:

- **No `<unk>`, ever — for anything.** Every possible string is a byte sequence and
  every byte is in the base vocab. Not just unseen *words* but unseen *scripts*,
  emoji, control bytes, corrupted input — all representable. Character-BPE still
  `<unk>`s a character it never saw; byte-BPE cannot.
- **Language-agnostic and preprocessing-free.** No unicode normalization, no
  language-specific segmentation. The cost: a non-ASCII character is 2–4 bytes, so
  several base tokens until merges glue it back — which training does for anything
  frequent.
- GPT-2's final vocab was **50257** = 256 byte tokens + 50000 learned merges + 1
  special `<|endoftext|>` document separator. Context window 1024 (lesson 11).

GPT-2 also adds a **regex pre-tokenizer** that first splits text into chunks
(words, numbers, punctuation runs, leading-space+word) and only ever merges
*within* a chunk — so a merge can't span "dog." → glue the period onto every word,
and " the" (space+word) becomes a natural unit while cross-word noise is prevented.
We implement the core byte-BPE **without** this regex first (Karpathy's `minbpe`
"basic" tokenizer); the regex split is a clean follow-on refinement.

## What we took into the code (lesson 12)

- Three functions are the whole algorithm: `get_stats` (count pairs), `merge`
  (replace a pair with a new id), and the `train` loop (repeat: count → merge most
  frequent → record). `encode` replays merges by learned priority; `decode` joins
  each id's bytes and UTF-8-decodes the blob.
- Base vocab is `{i: bytes([i]) for i in range(256)}`; `vocab_size = 256 + merges`.
- `encode` picks the eligible pair with the smallest merge index each round — the
  detail that makes inference reproduce training's segmentation.
- Losslessness (`decode(encode(s)) == s`) and no-`<unk>` are the tested invariants;
  compression (fewer tokens than bytes) is the payoff.

## Why it's still the standard

Every major LLM tokenizes with BPE or a close cousin (WordPiece in BERT;
Unigram/SentencePiece in T5 and LLaMA — LLaMA uses SentencePiece BPE over bytes).
The vocabulary is learned **once** on a corpus and then frozen; it is part of the
model's identity — swap the tokenizer and every id means something else. That is why
we build it ourselves before training any real-text model: the tokenizer is not a
detail bolted on, it is the interface between language and the network.

## Open threads (future lessons / reading)

- **Regex pre-tokenization** (GPT-2/GPT-4 split patterns) — prevent merges across
  word/punctuation boundaries; the next refinement of this file.
- **Special tokens** (`<|endoftext|>` and, later, chat-format tokens) — reserved ids
  outside the merge process; needed before a real training run and for Phase 7.
- **Save/load** the merges+vocab — a trained tokenizer must persist with the model.
- **Unigram LM tokenization** (Kudo 2018, SentencePiece) — the probabilistic
  alternative to greedy BPE that LLaMA-family models often use.
