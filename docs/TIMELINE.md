# LLM Timeline — how we got here

A running map of the field: the models that mattered, what was new about each, and
whether we can read their code/weights (open) or only their papers/behavior (closed).
Updated as we progress and as the field moves. ★ marks items we implement or study
deeply in the roadmap.

> Closed-source caveat: for closed models "what's inside" is often inferred from
> technical reports, leaks, and community analysis. Confidence is noted where it matters.

## Pre-history (2013–2016) — learning to represent language

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2013 | word2vec (Google) | open | Words as vectors; meaning as geometry ★ |
| 2014 | seq2seq (Google) | paper | Encoder–decoder for translation; fixed-vector bottleneck ★ |
| 2014 | Bahdanau attention | paper | Attention invented — as a *patch* for the bottleneck ★ |
| 2015 | ResNet (MSR) | open | Residual connections — quietly load-bearing in every transformer ★ |
| 2016 | LayerNorm | paper | The normalization transformers would adopt ★ |

## The transformer arrives (2017–2019)

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2017 | **Transformer** (Google) | paper+code | Attention *replaces* recurrence; full parallelism across the sequence ★ |
| 2018 | GPT-1 (OpenAI) | weights | Generative pretraining works: pretrain on text, finetune per task ★ |
| 2018 | BERT (Google) | open | Encoder-only, masked LM; dominated NLP benchmarks for years ★ |
| 2019 | **GPT-2** (OpenAI) | weights (staged release) | Scale → zero-shot ability; "too dangerous to release" moment; the decoder-only template ★ |
| 2019 | T5 (Google) | open | Everything as text-to-text; careful ablation of the design space |

## The scaling era (2020–2022)

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2020 | Scaling Laws (OpenAI) | paper | Loss predictable from parameters/data/compute — scale became a strategy ★ |
| 2020 | **GPT-3** 175B (OpenAI) | closed (API) | In-context learning emerges; the API business model ★ |
| 2021 | Codex (OpenAI) | closed | LLMs write code → Copilot |
| 2021 | Gopher, Megatron-Turing | closed | The parameter race (280B, 530B) — mostly a dead end |
| 2022 | **Chinchilla** (DeepMind) | paper | Everyone was undertraining: ~20 tokens/param; data becomes the constraint ★ |
| 2022 | PaLM 540B (Google) | closed | Peak "just make it bigger"; also where SwiGLU/parallel blocks were validated at scale |
| 2022 | **InstructGPT** (OpenAI) | paper | RLHF: the missing step between LM and assistant ★ |
| 2022 | FlashAttention (Stanford) | open | Exact attention, memory-efficient — now in every training stack ★ |
| 2022 | BLOOM, OPT | open | First serious open GPT-3-class attempts; showed how hard replication was |
| 2022 | **ChatGPT** (OpenAI, Nov 30) | closed | GPT-3.5 + RLHF + a chat box → the world notices |

## Open models catch up (2023)

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2023 | **GPT-4** (OpenAI, Mar) | closed | Large capability jump; multimodal input; architecture undisclosed (widely believed MoE, unconfirmed) |
| 2023 | **LLaMA 1** (Meta, Feb) | weights (leaked→open) | The modern open recipe: RoPE + RMSNorm + SwiGLU; Chinchilla-inspired ★ |
| 2023 | Alpaca, Vicuna (Stanford, etc.) | open | Cheap instruction-tuning on LLaMA; the open finetuning explosion |
| 2023 | Claude 1/2 (Anthropic) | closed | Constitutional AI (RLAIF) in production; long context (100k) ★ paper |
| 2023 | **LLaMA 2** (Meta, Jul) | open | Properly licensed; adds GQA; open RLHF details ★ |
| 2023 | **Mistral 7B** (Sep) | open | Small model, big punch: sliding-window attention, GQA; efficiency focus ★ |
| 2023 | Gemini 1 (Google, Dec) | closed | Google's unified multimodal answer to GPT-4 |
| 2023 | Mamba | open | First credible attention alternative (selective SSM) ★ |
| 2023 | Phi-1/2 (Microsoft) | open | "Textbooks are all you need": data quality vs. scale |

## Efficiency, MoE, and multimodal (2024)

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2024 | **Mixtral 8x7B** (Jan) | open | MoE goes mainstream open-source: 47B params, ~13B active ★ |
| 2024 | Claude 3 family (Mar) | closed | Opus/Sonnet/Haiku tiering; vision; first credible GPT-4 rival |
| 2024 | LLaMA 3 / 3.1 (Apr/Jul) | open | 15T tokens (way past Chinchilla — optimizing *inference* cost); 405B open flagship; 128k ctx |
| 2024 | GPT-4o (May) | closed | Native multimodal (audio in/out); cheap + fast frontier |
| 2024 | Gemini 1.5 (Feb) | closed | 1M+ token context (likely MoE + architectural tricks, unconfirmed) |
| 2024 | Qwen 2 / 2.5 (Alibaba) | open | Consistently strong open family across sizes; multilingual |
| 2024 | DeepSeek-V2 (May) | open | **MLA** (multi-head latent attention) + fine-grained MoE: radical KV-cache savings ★ |
| 2024 | **o1** (OpenAI, Sep) | closed | Test-time compute: hidden chain-of-thought RL; new scaling axis ★ concept |
| 2024 | **DeepSeek-V3** (Dec) | open | 671B MoE (37B active), trained for ~$5.5M reported compute cost; FP8, multi-token prediction ★ |

## Reasoning goes open (2025 → today)

| Year | What | Open? | Why it mattered |
|------|------|-------|-----------------|
| 2025 | **DeepSeek-R1** (Jan) | open | o1-class reasoning, openly published: GRPO, RL on verifiable rewards, distilled small models ★ |
| 2025 | Claude 3.7 → 4 → 4.5 (Anthropic) | closed | Hybrid reasoning (same model thinks or answers directly); agentic coding focus |
| 2025 | Gemini 2.0 / 2.5 (Google) | closed | Reasoning + massive context + native multimodal converge |
| 2025 | o3, GPT-5 (OpenAI) | closed | Test-time compute scaling continues; routing across effort levels |
| 2025 | LLaMA 4 (Meta) | open | Meta's MoE + multimodal generation; mixed reception |
| 2025 | Qwen 3, Kimi K2, GLM-4.5+ | open | Open frontier increasingly Chinese-led; K2: 1T-param open MoE |
| 2025 | gpt-oss (OpenAI) | open | OpenAI returns to open weights (first since GPT-2) |
| 2026 | Claude 5 family (Anthropic) | closed | Current frontier tier |

## The through-line

1. **2013–2016:** learn representations → hit the RNN sequential bottleneck.
2. **2017–2019:** attention replaces recurrence → training parallelizes → scale becomes possible.
3. **2020–2022:** scale predictably (scaling laws) → correct the recipe (Chinchilla) → make it usable (RLHF).
4. **2023:** the recipe escapes into open source (LLaMA) → efficiency race (Mistral, MoE).
5. **2024–2025:** spend compute at *inference* instead of just training (o1, R1) → reasoning.
6. **Open vs. closed today:** closed leads at the frontier by months, not years; open leads
   on efficiency innovation (MLA, cheap training) because it *has* to.

Each numbered step is roughly one phase of our roadmap.
