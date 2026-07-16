# CLAUDE.md — LLM Learning Lab

## Vision

This repository is a **personal, hands-on curriculum for understanding how modern
large language models work** — by walking the actual historical path the field took.

For every milestone architecture we:

1. **Collect the literature** — the papers that introduced or influenced it, with
   summaries written in our own words (`papers/`).
2. **Implement it from scratch in Python/PyTorch** — no `transformers` library,
   no copy-paste. Every module is built up in *small, fully-explained increments*
   (`src/llmlab/`).
3. **Write the lesson down** — each coding session produces a lesson note so the
   explanations survive outside the chat and can be re-read on mobile (`lessons/`).
4. **Keep the map of history current** — a timeline of both open- and closed-source
   models so we always know where a given technique sits relative to today's
   frontier (`docs/TIMELINE.md`).

The end state: a single codebase where architectures are assembled from
**interchangeable components** (attention variants, positional encodings, norms,
FFN variants, MoE routers...) so that "GPT-2 vs. LLaMA vs. Mixtral" becomes a
question of *which blocks you plug together*, not separate codebases.

## Teaching contract (how Claude must work in this repo)

These rules override default coding behavior:

- **Small portions per session.** Implement ONE concept at a time (e.g. "scaled
  dot-product attention", not "the whole transformer"). A session's diff should be
  small enough to read on a phone.
- **Explain everything, like a teacher.** For every non-trivial line: *why* it is
  there, what breaks without it, what the shapes are, and where the idea came from
  (cite the paper). Prefer deriving things (e.g. why divide by √d_k) over asserting them.
- **Docstrings and comments carry the teaching.** Code should be readable as a
  textbook. Tensor shapes are annotated everywhere, e.g. `# (batch, seq, d_model)`.
- **Every lesson ends with a lesson note** in `lessons/` summarizing what was built,
  the intuition, and open questions — written for re-reading later, not as a chat log.
- **Test as we go.** Each component gets a small `pytest` test that doubles as a
  usage example (shape checks, known-value checks, gradient checks).
- **History first.** Before implementing a new concept, add/update its entry in
  `papers/` and `docs/TIMELINE.md` so the "why did the field move here" context exists
  before the code does.
- **No magic dependencies.** PyTorch, numpy, pytest, and (for data/tokenizers when
  we get there) minimal extras. We implement the interesting parts ourselves —
  including BPE tokenization when we reach GPT-2.

## Hardware & environment constraints

- The user works **mostly from mobile via remote Claude sessions** — the remote
  container is **CPU-only** and ephemeral. Everything must run (slowly) on CPU:
  tiny configs, shape tests, character-level toy training.
- A **home PC with a single consumer NVIDIA GPU** exists for occasional small runs
  (think: minutes-to-hours, ~10M–124M parameter models on TinyShakespeare /
  TinyStories-scale data). Never design experiments that need more than that.
- Therefore: every model exposes a `tiny` config (runs on CPU in seconds, for
  learning/tests) and a `small` config (for the home GPU). Training scripts must
  checkpoint/resume, since sessions are ephemeral.
- Commit and push at the end of every session — the remote container does not persist.

## Repository layout

```
CLAUDE.md            ← this file: vision + working agreement
docs/
  TIMELINE.md        ← history of LLMs, open + closed source, kept current
  ROADMAP.md         ← the phased curriculum: what we study, in what order, with which papers
papers/
  README.md          ← index of all collected literature with our summaries
  fetch_papers.sh    ← downloads the PDFs from arXiv into papers/pdfs/ (gitignored)
  <topic>.md         ← per-topic reading notes as we study them
lessons/
  README.md          ← lesson-note format
  NN-<topic>.md      ← one note per implementation session
src/llmlab/          ← the from-scratch library
  components/        ← reusable blocks: attention, norms, ffn, positional, embeddings
  models/            ← architectures assembled from components (gpt2.py, llama.py, ...)
  training/          ← loop, optimizer setup, lr schedules, checkpointing
  data/              ← datasets + tokenizers (char-level first, then our own BPE)
tests/               ← pytest tests; each is also a usage example
```

## Design principles for `src/llmlab`

- **Config-driven assembly.** A model is a dataclass config naming its blocks
  (`attention="mha"`, `norm="layernorm"`, `norm_placement="pre"`,
  `positional="learned"`, `ffn="gelu_mlp"`...). GPT-2 and LLaMA differ only in config.
- **Components are registered, not hard-wired**, so a new paper's idea is a new
  registered component + a config entry, never a fork of the model file.
- **Readability beats performance.** We write the naive version first, understand it,
  and only then (as its own lesson) the optimized version — e.g. attention first as
  explicit loops over heads, then batched, then with KV cache.

## Current status

- [x] Project scaffolding, vision, roadmap, timeline
- [x] Lesson 01: scaled dot-product attention (`lessons/01-*.md`)
- [x] Lesson 02: multi-head attention (`lessons/02-*.md`)
- [x] Lesson 03: position-wise FFN, residuals, LayerNorm (`lessons/03-*.md`)
- [x] Lesson 04: sinusoidal positional encodings (`lessons/04-*.md`)
- [ ] Lesson 05: causal masking (decoder side) and padding masks
- See `docs/ROADMAP.md` for the full plan and `lessons/` for progress.

## Conventions

- Python ≥ 3.10, PyTorch. Format-light: clear code over tooling ceremony.
- Lesson notes numbered `NN-topic.md` in study order.
- Commits: `phase-N: <what was learned/built>`.
