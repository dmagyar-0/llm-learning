# Papers — the literature collection

Every paper the curriculum touches, indexed by phase. As we *study* a paper (not
just list it), it gets its own reading-note file in this directory
(`<topic>.md`) with a summary in our own words, the key equations, and what we
took into our implementation.

**PDFs:** most arXiv papers grant arXiv a non-exclusive license that does not
allow redistribution, so we don't commit PDFs to git. Instead run:

```bash
cd papers && bash fetch_papers.sh          # downloads everything into papers/pdfs/ (gitignored)
bash fetch_papers.sh attention gpt2        # or just specific topics
```

Status: ☐ listed → ◐ skimmed → ● studied (has a reading note)

## Phase 0 — Pre-transformer

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Efficient Estimation of Word Representations (word2vec) — Mikolov et al. | 2013 | [1301.3781](https://arxiv.org/abs/1301.3781) |
| ☐ | Sequence to Sequence Learning with Neural Networks — Sutskever et al. | 2014 | [1409.3215](https://arxiv.org/abs/1409.3215) |
| ☐ | Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. | 2014 | [1409.0473](https://arxiv.org/abs/1409.0473) |
| ● | Deep Residual Learning for Image Recognition — He et al. ([notes](residuals-layernorm.md)) | 2015 | [1512.03385](https://arxiv.org/abs/1512.03385) |
| ● | Layer Normalization — Ba, Kiros, Hinton ([notes](residuals-layernorm.md)) | 2016 | [1607.06450](https://arxiv.org/abs/1607.06450) |

## Phase 1 — The Transformer

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ● | **Attention Is All You Need** — Vaswani et al. ([notes](attention.md)) | 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) |

## Phase 2 — GPT-2 / decoder-only

| St | Paper | Year | Link |
|----|-------|------|------|
| ☐ | Improving Language Understanding by Generative Pre-Training (GPT-1) — Radford et al. | 2018 | [OpenAI PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| ☐ | **Language Models are Unsupervised Multitask Learners (GPT-2)** — Radford et al. | 2019 | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| ☐ | Neural Machine Translation of Rare Words with Subword Units (BPE) — Sennrich et al. | 2015 | [1508.07909](https://arxiv.org/abs/1508.07909) |
| ☐ | BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al. | 2018 | [1810.04805](https://arxiv.org/abs/1810.04805) |
| ☐ | GELU activation — Hendrycks & Gimpel | 2016 | [1606.08415](https://arxiv.org/abs/1606.08415) |

## Phase 3 — Scaling

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Language Models are Few-Shot Learners (GPT-3) — Brown et al. | 2020 | [2005.14165](https://arxiv.org/abs/2005.14165) |
| ☐ | Scaling Laws for Neural Language Models — Kaplan et al. | 2020 | [2001.08361](https://arxiv.org/abs/2001.08361) |
| ☐ | Training Compute-Optimal LLMs (Chinchilla) — Hoffmann et al. | 2022 | [2203.15556](https://arxiv.org/abs/2203.15556) |
| ☐ | Exploring the Limits of Transfer Learning (T5) — Raffel et al. | 2019 | [1910.10683](https://arxiv.org/abs/1910.10683) |
| ☐ | Emergent Abilities of Large Language Models — Wei et al. | 2022 | [2206.07682](https://arxiv.org/abs/2206.07682) |
| ☐ | Are Emergent Abilities a Mirage? — Schaeffer et al. | 2023 | [2304.15004](https://arxiv.org/abs/2304.15004) |

## Phase 4 — LLaMA recipe

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | LLaMA: Open and Efficient Foundation LMs — Touvron et al. | 2023 | [2302.13971](https://arxiv.org/abs/2302.13971) |
| ☐ | Llama 2: Open Foundation and Fine-Tuned Chat Models — Touvron et al. | 2023 | [2307.09288](https://arxiv.org/abs/2307.09288) |
| ☐ | Root Mean Square Layer Normalization — Zhang & Sennrich | 2019 | [1910.07467](https://arxiv.org/abs/1910.07467) |
| ☐ | RoFormer: Rotary Position Embedding (RoPE) — Su et al. | 2021 | [2104.09864](https://arxiv.org/abs/2104.09864) |
| ☐ | GLU Variants Improve Transformer (SwiGLU) — Shazeer | 2020 | [2002.05202](https://arxiv.org/abs/2002.05202) |
| ☐ | Fast Transformer Decoding: One Write-Head (MQA) — Shazeer | 2019 | [1911.02150](https://arxiv.org/abs/1911.02150) |
| ☐ | GQA: Generalized Multi-Query Attention — Ainslie et al. | 2023 | [2305.13245](https://arxiv.org/abs/2305.13245) |

## Phase 5 — Inference & long context

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Mistral 7B — Jiang et al. | 2023 | [2310.06825](https://arxiv.org/abs/2310.06825) |
| ☐ | FlashAttention — Dao et al. | 2022 | [2205.14135](https://arxiv.org/abs/2205.14135) |
| ☐ | Longformer: The Long-Document Transformer — Beltagy et al. | 2020 | [2004.05150](https://arxiv.org/abs/2004.05150) |
| ☐ | YaRN: Efficient Context Window Extension — Peng et al. | 2023 | [2309.00071](https://arxiv.org/abs/2309.00071) |

## Phase 6 — Mixture of Experts

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Outrageously Large Neural Networks (Sparsely-Gated MoE) — Shazeer et al. | 2017 | [1701.06538](https://arxiv.org/abs/1701.06538) |
| ☐ | Switch Transformers — Fedus et al. | 2021 | [2101.03961](https://arxiv.org/abs/2101.03961) |
| ☐ | Mixtral of Experts — Jiang et al. | 2024 | [2401.04088](https://arxiv.org/abs/2401.04088) |
| ☐ | DeepSeek-V2 (MLA) — DeepSeek-AI | 2024 | [2405.04434](https://arxiv.org/abs/2405.04434) |
| ☐ | DeepSeek-V3 — DeepSeek-AI | 2024 | [2412.19437](https://arxiv.org/abs/2412.19437) |

## Phase 7 — Post-training

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Training LMs to Follow Instructions (InstructGPT) — Ouyang et al. | 2022 | [2203.02155](https://arxiv.org/abs/2203.02155) |
| ☐ | Constitutional AI — Bai et al. | 2022 | [2212.08073](https://arxiv.org/abs/2212.08073) |
| ☐ | Direct Preference Optimization (DPO) — Rafailov et al. | 2023 | [2305.18290](https://arxiv.org/abs/2305.18290) |
| ☐ | LoRA: Low-Rank Adaptation — Hu et al. | 2021 | [2106.09685](https://arxiv.org/abs/2106.09685) |

## Phase 8 — Reasoning

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Chain-of-Thought Prompting — Wei et al. | 2022 | [2201.11903](https://arxiv.org/abs/2201.11903) |
| ☐ | Self-Consistency — Wang et al. | 2022 | [2203.11171](https://arxiv.org/abs/2203.11171) |
| ☐ | STaR: Self-Taught Reasoner — Zelikman et al. | 2022 | [2203.14465](https://arxiv.org/abs/2203.14465) |
| ☐ | DeepSeekMath (GRPO introduced) — Shao et al. | 2024 | [2402.03300](https://arxiv.org/abs/2402.03300) |
| ☐ | DeepSeek-R1 — DeepSeek-AI | 2025 | [2501.12948](https://arxiv.org/abs/2501.12948) |

## Phase 9 — Beyond transformers

| St | Paper | Year | arXiv |
|----|-------|------|-------|
| ☐ | Mamba: Linear-Time Sequence Modeling — Gu & Dao | 2023 | [2312.00752](https://arxiv.org/abs/2312.00752) |
| ☐ | Jamba: Hybrid Transformer-Mamba — Lieber et al. | 2024 | [2403.19887](https://arxiv.org/abs/2403.19887) |
