# Roadmap — the curriculum

The plan follows history: each phase exists because the previous one hit a limit.
Phases 1–2 are the foundation and are implemented completely. Later phases each
introduce a small number of *new* concepts on top of what we already have — that's
the payoff of the plug-in component design.

Legend: 📄 = paper we collect and summarize, 🔧 = thing we implement, 📖 = reading-only
(too big to train, but we study the ideas).

---

## Phase 0 — Pre-transformer context (short, reading-heavy)

*Why did attention appear at all?* Just enough history to make the Transformer feel
inevitable rather than arbitrary.

- 📄 Word2vec — Mikolov et al. 2013 (embeddings: words as vectors)
- 📄 Seq2seq — Sutskever et al. 2014 (encoder–decoder, the bottleneck problem)
- 📄 Bahdanau attention 2014 (attention as a fix for the bottleneck)
- 📖 Concepts: distributed representations, the RNN sequential bottleneck, why
  parallelism across the sequence matters for training on GPUs.
- 🔧 Tiny warm-up: an embedding layer + the "attention as soft dictionary lookup"
  intuition in ~30 lines of numpy.

## Phase 1 — The Transformer (2017): *Attention Is All You Need*

The core of everything that follows. Built piece by piece, one session each:

1. 🔧 Scaled dot-product attention (and *why* √d_k)
2. 🔧 Multi-head attention (naive per-head loop → batched)
3. 🔧 Position-wise FFN, residual connections, LayerNorm (post-norm, as in the paper)
4. 🔧 Sinusoidal positional encodings
5. 🔧 Causal masking (decoder side) and padding masks
6. 🔧 Full encoder–decoder assembly; train on a toy copy/reverse task on CPU

- 📄 Attention Is All You Need — Vaswani et al. 2017
- 📄 Layer Normalization — Ba et al. 2016
- 📄 (context) Residual learning — He et al. 2015

## Phase 2 — GPT-2 (2019): the decoder-only recipe

The template every modern LLM still follows. New concepts vs. Phase 1:

1. 🔧 Decoder-only architecture (drop the encoder; language modeling objective)
2. 🔧 **Pre-norm** (LayerNorm moved inside the residual — why this stabilizes depth)
3. 🔧 Learned positional embeddings
4. 🔧 Byte-Pair Encoding tokenizer, written ourselves
5. 🔧 Weight tying (embedding = output projection), GPT-2 init scheme
6. 🔧 Sampling: temperature, top-k, top-p
7. 🔧 Training run: char-level TinyShakespeare on CPU → BPE + small GPT-2 on the home GPU

- 📄 GPT-1 — Radford et al. 2018 (generative pretraining + finetuning)
- 📄 GPT-2 — Radford et al. 2019 (zero-shot, scale, the WebText bet)
- 📄 BPE for NMT — Sennrich et al. 2015
- 📖 BERT — Devlin et al. 2018 (the road not taken: encoder-only, masked LM — why
  decoder-only won for generation)

## Phase 3 — Scaling era (2020–2022): 📖 reading phase

Nothing here is implementable on our hardware — but it explains *why* models look
the way they do today.

- 📄 GPT-3 — Brown et al. 2020 (in-context learning emerges)
- 📄 Scaling Laws — Kaplan et al. 2020 (loss is predictable from N, D, C)
- 📄 Chinchilla — Hoffmann et al. 2022 (compute-optimal: ~20 tokens per parameter;
  why everyone was undertraining)
- 📄 T5 — Raffel et al. 2020 (everything-is-text; also introduced relative positions)
- 📄 Emergent abilities — Wei et al. 2022 (+ the "mirage" rebuttal, Schaeffer 2023)
- 🔧 Small empirical exercise: fit a mini scaling law across 3–4 tiny model sizes we
  can actually train, and see the power law appear on our own loss curves.

## Phase 4 — LLaMA (2023): the modern open recipe

LLaMA 1/2 distilled the post-GPT-3 learnings into the architecture that nearly every
open model now copies. Each change is one plug-in component swap:

1. 🔧 **RMSNorm** replaces LayerNorm (📄 Zhang & Sennrich 2019)
2. 🔧 **RoPE** rotary positional embeddings replace learned positions (📄 RoFormer, Su et al. 2021)
3. 🔧 **SwiGLU** FFN replaces GELU MLP (📄 Shazeer 2020, "GLU Variants Improve Transformer")
4. 🔧 No biases, and the LLaMA-2 addition: **Grouped-Query Attention**
   (📄 GQA, Ainslie et al. 2023; 📄 MQA, Shazeer 2019)
5. 🔧 Train "TinyLLaMA" on the same data as our GPT-2 → compare loss curves directly

- 📄 LLaMA — Touvron et al. 2023; 📄 LLaMA 2 — Touvron et al. 2023

## Phase 5 — Inference & long context

Why serving LLMs is its own science. Mistral 7B is the case study.

1. 🔧 **KV cache** — the single most important inference idea; measure the speedup
2. 🔧 **Sliding-window attention** (📄 Mistral 7B, Jiang et al. 2023; 📄 Longformer 2020)
3. 📖 **FlashAttention** — Dao et al. 2022 (IO-aware exact attention; we study the
   tiling idea, use PyTorch's built-in `scaled_dot_product_attention` to feel it)
4. 📖 Context extension: position interpolation, YaRN, NTK scaling
5. 📖 Quantization basics (why 4-bit works; GPTQ/AWQ at a conceptual level)

## Phase 6 — Mixture of Experts: more parameters ≠ more compute

1. 🔧 **Top-k gating / expert routing** + load-balancing loss
2. 🔧 A tiny MoE FFN block plugged into our LLaMA-style model
3. 📄 Sparsely-Gated MoE — Shazeer et al. 2017
4. 📄 Switch Transformer — Fedus et al. 2021
5. 📄 Mixtral of Experts — Jiang et al. 2024
6. 📖 DeepSeek-V2/V3 — Multi-head Latent Attention (MLA), fine-grained experts,
   shared experts, multi-token prediction; the 2024 efficiency masterclass

## Phase 7 — Post-training: from LM to assistant

Why raw GPT-3 was unusable and ChatGPT wasn't. Architecture stops changing here;
*training* becomes the story.

1. 📄 InstructGPT — Ouyang et al. 2022 (SFT → reward model → PPO: the RLHF pipeline)
2. 📄 Constitutional AI — Bai et al. 2022 (Anthropic's RLAIF)
3. 📄 **DPO** — Rafailov et al. 2023 (preference tuning without RL)
4. 🔧 SFT: fine-tune our small model on an instruction dataset
5. 🔧 DPO on toy preference pairs (fits on the home GPU; PPO does not — 📖 only)
6. 📄 LoRA — Hu et al. 2021 🔧 (implement low-rank adapters — this is what makes
   fine-tuning possible on a single consumer GPU)

## Phase 8 — Reasoning models (2024–now)

1. 📄 Chain-of-Thought prompting — Wei et al. 2022
2. 📄 Self-consistency — Wang et al. 2022; 📄 STaR — Zelikman et al. 2022
3. 📖 OpenAI o1 (2024) — test-time compute as a new scaling axis
4. 📄 **DeepSeek-R1** — 2025 (GRPO; RL directly on verifiable rewards; the open
   replication of reasoning training)
5. 🔧 Toy experiment: GRPO-style RL on arithmetic tasks with our small model —
   watch chain-of-thought length grow during training

## Phase 9 — Beyond the transformer (outlook)

- 📄 Mamba — Gu & Dao 2023 (selective state-space models, linear-time)
- 📖 Hybrids: Jamba, Zamba, Griffin — attention + SSM layers
- 📖 Multimodality: how vision plugs in (CLIP, LLaVA-style adapters)
- 🔧 Optional: a minimal SSM block plugged into our stack, if appetite remains

---

## Suggested pace

Each numbered 🔧 item ≈ one session (readable on a phone). Phases 1–2 are the long
ones (~12–15 sessions); later phases are shorter because the scaffolding already
exists. Reading phases (3, and the 📖 parts) can be interleaved with coding whenever
a change of pace is welcome.
