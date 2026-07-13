# llm-learning

A hands-on curriculum for understanding LLMs by re-walking the field's history:
collect the papers, implement each milestone architecture from scratch in
PyTorch, and write every lesson down.

- **Start here:** [`CLAUDE.md`](CLAUDE.md) — the vision and working agreement
- **The plan:** [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased curriculum, from
  *Attention Is All You Need* and GPT-2 up through LLaMA, Mistral, MoE, RLHF/DPO,
  and reasoning models
- **The map:** [`docs/TIMELINE.md`](docs/TIMELINE.md) — history of open and
  closed models, 2013 → today
- **The literature:** [`papers/README.md`](papers/README.md) — every paper,
  indexed by phase (`bash papers/fetch_papers.sh` downloads the PDFs)
- **The lessons:** [`lessons/`](lessons/) — one note per implementation session
- **The code:** [`src/llmlab/`](src/llmlab/) — architectures as plug-in components
