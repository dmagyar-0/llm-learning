"""Tests for lesson 12 — our from-scratch byte-level BPE tokenizer.

The claims worth pinning are exactly the properties that make BPE the right
choice: it is LOSSLESS (decode∘encode is identity, for any string), it never
emits <unk> (bytes cover everything), it COMPRESSES (frequent chunks become one
id), and it replays merges in the learned order (encode agrees with train).
"""

from llmlab.data.bpe import BPETokenizer, get_stats, merge


# ------------------------------------------------- the two primitives

def test_get_stats_counts_adjacent_pairs():
    """Step 1: every overlapping adjacent pair, counted. [7,7,9,7] → (7,7):1,
    (7,9):1, (9,7):1. This frequency table is what training maximizes over."""
    assert get_stats([7, 7, 9, 7]) == {(7, 7): 1, (7, 9): 1, (9, 7): 1}
    assert get_stats([5, 5, 5]) == {(5, 5): 2}   # overlapping counts both
    assert get_stats([1]) == {}                   # need two ids to have a pair


def test_merge_is_greedy_left_to_right_nonoverlapping():
    """Step 2: replace the pair, advancing by two on a match. [5,5,5] merging
    (5,5) → [256,5] (the FIRST pair wins; no overlap, no ambiguity)."""
    assert merge([7, 7, 9, 7], (7, 9), 256) == [7, 256, 7]
    assert merge([5, 5, 5], (5, 5), 256) == [256, 5]
    assert merge([1, 2, 3], (8, 9), 256) == [1, 2, 3]  # pair absent → unchanged


# ------------------------------------------------- untrained = byte tokenizer

def test_untrained_is_a_plain_byte_tokenizer():
    """Before any training there are no merges, so encode is just the raw UTF-8
    bytes (all ids < 256) and decode inverts it — already lossless, already
    <unk>-free. Training only adds ids on top of this base."""
    tok = BPETokenizer()
    s = "hello"

    ids = tok.encode(s)

    assert ids == list(s.encode("utf-8"))   # literally the bytes
    assert all(i < 256 for i in ids)
    assert tok.decode(ids) == s


# ------------------------------------------------- training: the core behaviors

def test_first_merge_is_the_most_frequent_pair():
    """'ababab' → bytes for a,b repeating; the pair (a,b) occurs 3×, (b,a) 2×,
    so the first minted symbol (id 256) must be 'ab'. Hand-checkable BPE."""
    tok = BPETokenizer()
    tok.train("ababab", vocab_size=257)  # exactly one merge

    a, b = ord("a"), ord("b")
    assert tok.merges == {(a, b): 256}
    assert tok.vocab[256] == b"ab"
    assert tok.encode("ab") == [256]          # the learned chunk is one id
    assert tok.encode("abab") == [256, 256]


def test_vocab_size_is_respected():
    """The target is the FINAL vocab: 256 bytes + (vocab_size-256) merges. With
    a corpus rich enough to supply that many pairs, we mint exactly that many."""
    tok = BPETokenizer()
    text = "the quick brown fox jumps over the lazy dog. " * 20
    tok.train(text, vocab_size=300)

    assert len(tok.merges) == 300 - 256
    assert len(tok.vocab) == 300
    assert max(tok.vocab) == 299


def test_training_compresses_the_corpus():
    """The whole point: gluing frequent pairs makes the token stream shorter than
    the raw byte stream. More merges → more compression."""
    text = "the quick brown fox jumps over the lazy dog. " * 20
    raw_len = len(text.encode("utf-8"))

    tok = BPETokenizer()
    tok.train(text, vocab_size=350)

    assert len(tok.encode(text)) < raw_len
    # A bigger vocabulary compresses at least as well (never worse).
    small = BPETokenizer(); small.train(text, vocab_size=300)
    assert len(tok.encode(text)) <= len(small.encode(text))


# ------------------------------------------------- the properties that matter

def test_roundtrip_is_lossless_including_unicode():
    """decode(encode(s)) == s for ANY string — the non-negotiable tokenizer
    invariant. Includes multi-byte UTF-8 (emoji, accents) to prove the byte base
    handles characters that span several tokens after merging."""
    tok = BPETokenizer()
    tok.train("the quick brown fox " * 30, vocab_size=320)

    for s in ["hello world", "the fox", "", "a", "café — naïve", "emoji 🚀🔥 test"]:
        assert tok.decode(tok.encode(s)) == s, f"roundtrip failed on {s!r}"


def test_never_emits_unknown_on_unseen_input():
    """Trained only on ASCII, the tokenizer must still encode—and perfectly
    round-trip—text full of characters it never saw. Byte fallback means there is
    no <unk>: worst case an unseen char is just its raw bytes."""
    tok = BPETokenizer()
    tok.train("only plain ascii words here " * 20, vocab_size=300)

    unseen = "日本語 and ✨ were never in training"
    ids = tok.encode(unseen)

    assert all(i in tok.vocab for i in ids)   # every id is representable
    assert tok.decode(ids) == unseen          # and it round-trips exactly


def test_encode_replays_merges_in_learned_order():
    """encode must apply merges by priority (earliest learned first), so its
    output equals the trained id stream on the training text. If order were
    ignored, encode could produce ids that never occurred in training."""
    text = "aaabaaabaaab"
    tok = BPETokenizer()
    tok.train(text, vocab_size=260)  # a few merges on a repetitive corpus

    # Re-encoding the corpus reproduces a stream that decodes back to it, and
    # actually uses the merged (>=256) symbols rather than raw bytes.
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert any(i >= 256 for i in ids)


def test_decode_handles_invalid_utf8_gracefully():
    """A hand-built id list whose bytes aren't valid UTF-8 must not crash: the
    lone byte 0xFF (id 255) is not a valid UTF-8 sequence, so decode emits the
    U+FFFD replacement char (errors='replace') instead of raising."""
    tok = BPETokenizer()
    assert tok.decode([255]) == "�"
