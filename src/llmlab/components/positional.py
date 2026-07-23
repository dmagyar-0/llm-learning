"""Positional encodings, built up lesson by lesson.

Lesson 04: sinusoidal positional encodings (Vaswani et al. 2017, §3.5):

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why anything like this must exist: everything we built in lessons 01–03 is
**permutation-equivariant**. Attention is a softmax-weighted average over a
*set* of key/value vectors — nothing in Q·Kᵀ knows which position a vector
came from — and the FFN and LayerNorm act on each token independently. So
shuffling the input tokens shuffles the outputs identically (test-verified):
to the whole stack, "dog bites man" and "man bites dog" are the same input.
RNNs got order for free from processing tokens one at a time; the transformer
traded that away for parallelism, so order has to be handed back *as data* —
a position-dependent vector added to each token embedding at the bottom of
the stack.

Why sinusoids specifically — think "binary counting, made continuous". In
binary, bit i flips with period 2^(i+1): low bits spin fast, high bits crawl.
That gives every number a unique code out of a fixed set of bounded digits.
Sinusoids do the same with smooth dials: channel pair (2i, 2i+1) is a
(sin, cos) oscillator whose wavelength grows geometrically from 2π to
10000·2π across the pairs — fast pairs distinguish neighbors, slow pairs
distinguish paragraph-scale positions. The payoff over hard bits: the code
is smooth in pos, and every value stays in [−1, 1] at any position (the
scale discipline of lesson 03 — the residual stream starts life at O(1)).

The property the paper actually chose them for: for any fixed offset k,
PE(pos+k) is a *linear* function of PE(pos) — the angle-sum identities

    sin(ω(p+k)) =  cos(ωk)·sin(ωp) + sin(ωk)·cos(ωp)
    cos(ω(p+k)) = −sin(ωk)·sin(ωp) + cos(ωk)·cos(ωp)

say each (sin, cos) pair transforms by a 2×2 **rotation** through angle ω·k,
the *same* matrix for every p. So "attend 3 tokens to the left" is a single
linear map on these codes — learnable by the W_Q/W_K projections of lesson 02
without ever knowing absolute positions. Phase 4's RoPE is this insight
promoted from hope to mechanism: rotate q and k inside attention instead of
adding codes at the bottom.

Lesson 11: learned absolute embeddings (GPT-2, Phase 2) — same additive combo,
but the table becomes a trained parameter instead of a formula.

Still in this file's future (per the roadmap): RoPE (LLaMA, Phase 4).
"""

import math

import torch
from torch import Tensor, nn


def sinusoidal_table(seq_len: int, d_model: int, base: float = 10000.0) -> Tensor:
    """Build the (seq_len, d_model) table of positional codes.

    Row `pos` is that position's code; nothing is learned. Row p never
    depends on seq_len (test-verified: a longer table starts with a shorter
    one) — position 5 means the same thing in every sequence, which is what
    makes the codes reusable across sequence lengths.
    """
    if d_model % 2 != 0:
        # The construction is inherently pairwise — every frequency needs both
        # its sin and its cos channel, or the rotation property above dies
        # (you can't rotate half a coordinate). Real models use even d_model.
        raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")

    # One frequency per PAIR of channels: ω_i = base^(−2i/d_model),
    # i = 0 .. d_model/2 − 1. Computed as exp(−log(base)·2i/d) — same number,
    # but exp/log of moderate values is numerically friendlier than raising
    # 10000 to tiny fractional powers.
    # ω_0 = 1 (wavelength 2π: flips within a couple of tokens);
    # ω_last ≈ 1/base (wavelength 10000·2π: essentially DC over any real
    # sequence). Geometric spacing means each scale of "how far apart" gets
    # a channel pair — the continuous analogue of binary's bit periods.
    i = torch.arange(0, d_model, 2, dtype=torch.float32)          # (d_model/2,)
    freqs = torch.exp(-math.log(base) * i / d_model)              # (d_model/2,)

    pos = torch.arange(seq_len, dtype=torch.float32)              # (seq_len,)
    # Outer product: every position at every frequency.
    angles = pos[:, None] * freqs[None, :]                        # (seq_len, d_model/2)

    table = torch.empty(seq_len, d_model)
    table[:, 0::2] = torch.sin(angles)  # even channels: sin — so PE(0) = (0, 1, 0, 1, ...)
    table[:, 1::2] = torch.cos(angles)  # odd channels:  cos
    return table  # (seq_len, d_model), every entry in [−1, 1]


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal position codes to a batch of embeddings (§3.5).

    Why ADD instead of concatenate: concatenation would reserve channels for
    position, costing width and forcing every downstream layer to honor the
    split. Addition superimposes position onto content in the same d_model
    space, and lesson 02's learned projections choose per-head how much of
    each to extract — heads that need position take it, heads that don't
    ignore it. It works because random high-dimensional embeddings are nearly
    orthogonal to these structured sinusoids; the paper additionally scales
    embeddings by √d_model (§3.4 — that lands in our embedding lesson) so
    trained content isn't drowned by the unit-amplitude position signal.

    This module has ZERO parameters (test-verified). The table is a buffer:
    part of the module (moves with .to(device)), not part of the state_dict
    (persistent=False — it's cheap to rebuild and not learned), never touched
    by the optimizer. Contrast with Phase 2's learned embeddings, where the
    same table IS a parameter — that single difference is the whole
    architectural choice.
    """

    def __init__(self, d_model: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.max_len = max_len
        # Precompute once up to max_len; forward just slices. max_len is a
        # capacity, not a promise — sinusoids are DEFINED for any position
        # (the paper's argument for them over learned tables), we simply
        # don't spend memory on positions we never expect.
        self.register_buffer(
            "table", sinusoidal_table(max_len, d_model, base), persistent=False
        )  # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:  # x: (batch, seq, d_model)
        """Return x + PE[0:seq] — same shape, now order-aware.

        The add broadcasts the (seq, d_model) table over the batch dimension:
        position codes depend on *where*, never on *what* or on which example.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds precomputed max_len "
                f"{self.max_len} — construct with a larger max_len"
            )
        return x + self.table[:seq_len]  # (batch, seq, d_model)


class LearnedPositionalEmbedding(nn.Module):
    """Add a LEARNED absolute position vector to each token (GPT-1/2, Phase 2).

    Look back at `SinusoidalPositionalEncoding`: its whole job is to add a
    (seq, d_model) table to the embeddings, one row per position. This module
    does *exactly the same thing* — same additive combination, same shape, same
    "breaks permutation-equivariance" effect — and differs in ONE place: where
    the table comes from.

        sinusoidal:  table = fixed formula (a buffer, no gradient)
        learned:     table = nn.Parameter, filled in by gradient descent

    That single swap, buffer → parameter, is the entire architectural decision
    GPT-2 made here (§2.3, and the same `wpe` as GPT-1). Everything else about
    how positions enter the model is unchanged — which is the point of keeping
    these as interchangeable components: the model file does not care which one
    it holds.

    Mechanically it is a SECOND embedding table sitting beside the token table
    (lesson 06's `TokenEmbedding`). The token table is indexed by *what* (token
    id); this one is indexed by *where* (position 0, 1, 2, ...). GPT-2 looks up
    both and adds them: `h = wte[ids] + wpe[positions]`. Same `nn.Embedding`
    lookup, different question asked of it.

    Two consequences follow from "it's learned, not a formula", and they are the
    whole lesson (see the note):

    1. **Hard length cap, no extrapolation.** Row `p` only exists (and only ever
       receives gradient) for `p < max_len`. Position 1024 in a model trained to
       1023 is a row that was never learned — undefined, not merely unseen.
       Sinusoids are *defined* at every real position; a learned table is not.
       This is exactly why GPT-2's context is a hard 1024, and a big reason the
       field later moved to RoPE/ALiBi (Phase 4) to get length flexibility back.
    2. **No built-in relative structure.** Lesson 04's sinusoids handed the model
       the "shift = rotation" and "dot product = f(distance)" gifts for free.
       A learned table starts as noise and must discover any relative-position
       behavior from data. At scale it does, well enough — but nothing is given.

    Unlike token embeddings we do NOT scale by √d_model here. That √d_model
    (embeddings lesson) existed to lift fixed unit-amplitude signals to a chosen
    volume; a learned table has no fixed amplitude to fight — gradient descent
    sets its magnitude relative to the content itself. So it self-calibrates,
    and we just initialize it small (GPT-2 used ~N(0, 0.02)) and let training do
    the rest. Full init/tying discipline is its own later lesson.
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        self.max_len = max_len
        # One learnable row per position. This IS a parameter (contrast the
        # sinusoidal buffer): it lives in the state_dict, moves with .to(device),
        # and the optimizer updates it every step.
        self.table = nn.Embedding(max_len, d_model)
        # Small Gaussian init, GPT-2 style. Position rows start ~silent and grow
        # only as far as the loss rewards — nothing is baked in, so we do not
        # want them shouting over content at step 0.
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:  # x: (batch, seq, d_model)
        """Return x + wpe[0:seq] — same shape, now order-aware.

        `positions` is [0, 1, ..., seq-1]: absolute indices, the same for every
        row in the batch (position depends on *where*, never on *what* or on
        which example — identical to the sinusoidal add, just a learned lookup).
        The (seq, d_model) result broadcasts over the batch dimension.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            # Not a capacity we could cheaply grow (as with sinusoids): rows
            # beyond max_len were never trained and carry no meaning. Fail loud
            # rather than index out of the table or wrap silently.
            raise ValueError(
                f"sequence length {seq_len} exceeds learned max_len "
                f"{self.max_len} — a learned table cannot extrapolate; it has "
                f"no row for positions it was never trained on"
            )
        positions = torch.arange(seq_len, device=x.device)  # (seq,)
        return x + self.table(positions)  # (batch, seq, d_model)


def build_positional(kind: str, d_model: int, max_len: int) -> nn.Module:
    """Pick a positional component by name — the config-driven assembly knob.

    Both returned modules share one contract: call with (batch, seq, d_model),
    get the same shape back with position information added, and a loud error if
    `seq > max_len`. Because the interface is identical, the model swaps
    "sinusoidal" for "learned" by changing a config string — no other code moves.
    That interchangeability is the repo's whole design thesis (a paper's idea =
    a new registered component + a config entry, never a forked model file).
    """
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len=max_len)
    if kind == "learned":
        return LearnedPositionalEmbedding(d_model, max_len=max_len)
    raise ValueError(
        f"unknown positional kind {kind!r}; expected 'sinusoidal' or 'learned'"
    )
