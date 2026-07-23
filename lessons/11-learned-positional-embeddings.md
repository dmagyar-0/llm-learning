# 11 — Learned positional embeddings

**Phase:** 2 (GPT-2 recipe) · **Paper(s):** GPT-1 §3 (Radford et al. 2018) and GPT-2 §2.3 (Radford et al. 2019) — both use a learned position table (`wpe`)
**Code:** `src/llmlab/components/positional.py` (`LearnedPositionalEmbedding`, `build_positional`), `src/llmlab/models/gpt.py` (`positional` knob) · **Test:** `tests/test_learned_positional.py`

## The problem

Lesson 04 gave the order-blind transformer a sense of position with a **fixed
sinusoidal formula**. It was elegant and free — zero parameters, defined at every
position, with the built-in "shift = rotation" gift. But lesson 09/10's GPT still
carried it only as a *placeholder*: the real GPT-1 and GPT-2 do not use sinusoids.
They learn the position table the same way they learn the token table.

Why would you throw away a free, mathematically-principled encoding for a bag of
learnable weights? The honest answer GPT gave in 2018–19 was pragmatic: *let the
model decide what position should mean, instead of us guessing a formula.* This
lesson makes that swap — and the swap is deliberately tiny, because that is the
whole point of building positions as an interchangeable component.

## The idea

Look at what the sinusoidal module actually does: it adds a `(seq, d_model)` table
to the embeddings, one row per position. The learned version does **exactly the
same add** — same shape, same "breaks permutation-equivariance" effect, same
`max_len` cap. One thing moves:

    sinusoidal:  table = a fixed formula        →  register_buffer (no gradient)
    learned:     table = an nn.Parameter        →  filled in by gradient descent

That's it. Buffer → parameter. Mechanically it is just a **second embedding table**
sitting next to the token table from lesson 06:

    token embedding  wte[ids]        answers "WHAT is this token?"
    position table   wpe[positions]  answers "WHERE is it?"
    GPT input        h = wte[ids] + wpe[positions]

Same `nn.Embedding` lookup, asked a different question — one indexed by token id,
the other by the integers `0, 1, 2, …`. GPT adds them and feeds the sum to the stack.

## The math

There is barely any — that's the surprise. Forward is

    LearnedPos(x)[b, p, :] = x[b, p, :] + W_pos[p, :]        # W_pos: (max_len, d_model)

with `W_pos` a plain parameter matrix. Compare lesson 04, where that same row was
`[sin(ω₀p), cos(ω₀p), sin(ω₁p), …]`. The forward pass is identical; only the *origin*
of `W_pos[p]` differs.

The interesting math is in what we **lost**, and it's worth stating precisely because
it drives Phase 4:

- **No relative structure at init.** Sinusoids satisfy `PE(p+k) = R(k)·PE(p)` for a
  rotation `R(k)` independent of `p` (lesson 04's key property), and
  `PE(p)·PE(p+k) = Σᵢ cos(ωᵢk)` depends only on the offset `k`. A learned `W_pos`
  starts as noise: `W_pos[p+k]` has **no** fixed relationship to `W_pos[p]`. The model
  can *learn* approximate relative behavior from data, but nothing is handed to it.
- **A hard length wall.** `W_pos` has exactly `max_len` rows. Row `p` only receives
  gradient when a training sequence is at least `p+1` long, so positions past the
  longest training sequence are **never shaped**. Asking for position 1024 in a model
  trained to 1023 indexes a row that means nothing. Sinusoids are *defined* for every
  real number; a learned table is a finite lookup. This is why GPT-2's context is a
  hard 1024 tokens, full stop.

`tests/test_learned_positional.py::test_only_used_rows_receive_gradient` makes the
wall visible: run one forward over `seq=4`, and rows `0..3` get gradient while rows
`4..max_len-1` stay exactly zero.

**One thing we *don't* do:** scale by √d_model. Token embeddings got that factor
(embeddings lesson) to lift a fixed unit-amplitude signal to a chosen volume so it
wasn't drowned by the sinusoid. A learned table has no fixed amplitude to defend —
gradient descent sets its magnitude relative to the content on its own. So it
**self-calibrates**; we just init small (GPT-2 used ~N(0, 0.02)) so positions start
quiet and grow only as far as the loss rewards.

## The code

**A new component, same interface** (`components/positional.py`). The class is short
because it's a thin wrapper over `nn.Embedding`:

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.table = nn.Embedding(max_len, d_model)          # THIS is a parameter
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, x):                                     # x: (batch, seq, d_model)
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError("a learned table cannot extrapolate...")   # fail loud
        positions = torch.arange(seq_len, device=x.device)   # [0, 1, ..., seq-1]
        return x + self.table(positions)                     # broadcast over batch
```

**A factory that makes the choice a string** (`components/positional.py`):

```python
def build_positional(kind, d_model, max_len):
    if kind == "sinusoidal": return SinusoidalPositionalEncoding(d_model, max_len=max_len)
    if kind == "learned":    return LearnedPositionalEmbedding(d_model, max_len=max_len)
    raise ValueError(f"unknown positional kind {kind!r}")
```

Both modules honor one contract — `(batch, seq, d_model) → same shape`, loud error
past `max_len` — so the model can hold either without knowing which.

**The model gains a knob** (`models/gpt.py`). `GPTConfig.positional` defaults to
`"learned"` (GPT-2 specifically), and the one construction line changed:

```python
self.positional = build_positional(c.positional, c.d_model, c.max_len)
```

No other line in the model moved. Swapping GPT's positions between the 2017 formula
and GPT-2's learned table is now a single config field — `dataclasses.replace(cfg,
positional="sinusoidal")` — exactly like `norm_placement` in lesson 10. Two
architectural eras, one model file. That is the design thesis paying rent again.

## What breaks without it

- **Keep sinusoids, believe you have GPT-2:** you'd have a subtly different model.
  It would still train, but it isn't the GPT-2 recipe, and comparisons to published
  numbers would quietly drift. Architecture bugs that still train are the worst kind
  (lesson 10's refrain) — so the choice is explicit and defaulted, not implicit.
- **Forget the `max_len` guard:** `torch.arange(seq)` would index past the table and
  either throw a cryptic CUDA/CPU indexing error or, worse, wrap — silently reusing
  position 0's vector for position `max_len`. We fail loud with a message that names
  the real cause (no row was ever trained there).
- **Scale the learned table by √d_model too:** you'd start positions ~28× louder than
  content (for d_model=768), stomping the token signal for the first many steps until
  the optimizer claws the magnitude back down. Learned tables self-calibrate; the
  √d_model crutch is for *fixed* signals only.

## Open questions

- **Was giving up extrapolation worth it?** GPT-2 said yes for simplicity; the field
  later said no. Absolute learned positions can't handle sequences longer than
  training, and even within range they don't generalize offsets cleanly. Phase 4's
  **RoPE** (and ALiBi) win back relative structure and length flexibility by rotating
  q/k *inside* attention instead of adding a table at the bottom — the "shift =
  rotation" property of lesson 04, promoted from a happy accident of sinusoids into
  the mechanism itself. This lesson is the foil that makes RoPE feel necessary.
- **Two tables or one?** GPT keeps `wte` (token) and `wpe` (position) separate and
  adds them. Weight tying (next lessons) ties `wte` to the output head — but never
  `wpe`. Why position embeddings stay untied, and how init interacts with tying, is
  the next specialization toward GPT-2.
- **Does position 0 ever get special?** Because row 0 is used by *every* sequence, it
  trains far more than high positions — a mild imbalance the sinusoid never had. Worth
  remembering when we later stare at a learned `wpe` heatmap and see structure emerge.
