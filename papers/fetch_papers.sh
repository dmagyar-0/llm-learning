#!/usr/bin/env bash
# Download the curriculum's papers into papers/pdfs/ (gitignored — see README.md
# for why we don't commit PDFs). Usage:
#   bash fetch_papers.sh              # everything
#   bash fetch_papers.sh attention gpt2 rope   # only these topics
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdfs

# topic|url pairs. arXiv PDF URLs are https://arxiv.org/pdf/<id>
PAPERS="
word2vec|https://arxiv.org/pdf/1301.3781
seq2seq|https://arxiv.org/pdf/1409.3215
bahdanau-attention|https://arxiv.org/pdf/1409.0473
resnet|https://arxiv.org/pdf/1512.03385
layernorm|https://arxiv.org/pdf/1607.06450
attention|https://arxiv.org/pdf/1706.03762
convs2s|https://arxiv.org/pdf/1705.03122
relative-positions|https://arxiv.org/pdf/1803.02155
gpt1|https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
gpt2|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
bpe|https://arxiv.org/pdf/1508.07909
bert|https://arxiv.org/pdf/1810.04805
gelu|https://arxiv.org/pdf/1606.08415
gpt3|https://arxiv.org/pdf/2005.14165
scaling-laws|https://arxiv.org/pdf/2001.08361
chinchilla|https://arxiv.org/pdf/2203.15556
t5|https://arxiv.org/pdf/1910.10683
emergent-abilities|https://arxiv.org/pdf/2206.07682
emergent-mirage|https://arxiv.org/pdf/2304.15004
llama1|https://arxiv.org/pdf/2302.13971
llama2|https://arxiv.org/pdf/2307.09288
rmsnorm|https://arxiv.org/pdf/1910.07467
rope|https://arxiv.org/pdf/2104.09864
swiglu|https://arxiv.org/pdf/2002.05202
mqa|https://arxiv.org/pdf/1911.02150
gqa|https://arxiv.org/pdf/2305.13245
mistral|https://arxiv.org/pdf/2310.06825
flash-attention|https://arxiv.org/pdf/2205.14135
longformer|https://arxiv.org/pdf/2004.05150
yarn|https://arxiv.org/pdf/2309.00071
sparse-moe|https://arxiv.org/pdf/1701.06538
switch-transformer|https://arxiv.org/pdf/2101.03961
mixtral|https://arxiv.org/pdf/2401.04088
deepseek-v2|https://arxiv.org/pdf/2405.04434
deepseek-v3|https://arxiv.org/pdf/2412.19437
instructgpt|https://arxiv.org/pdf/2203.02155
constitutional-ai|https://arxiv.org/pdf/2212.08073
dpo|https://arxiv.org/pdf/2305.18290
lora|https://arxiv.org/pdf/2106.09685
chain-of-thought|https://arxiv.org/pdf/2201.11903
self-consistency|https://arxiv.org/pdf/2203.11171
star|https://arxiv.org/pdf/2203.14465
deepseek-math-grpo|https://arxiv.org/pdf/2402.03300
deepseek-r1|https://arxiv.org/pdf/2501.12948
mamba|https://arxiv.org/pdf/2312.00752
jamba|https://arxiv.org/pdf/2403.19887
"

fetch() {
  local topic="$1" url="$2" out="pdfs/$1.pdf"
  if [ -s "$out" ]; then
    echo "have    $topic"
  else
    echo "fetch   $topic"
    curl -fsSL --retry 3 -o "$out" "$url" || { echo "FAILED  $topic ($url)" >&2; rm -f "$out"; }
    sleep 1  # be polite to arXiv
  fi
}

while IFS='|' read -r topic url; do
  [ -z "$topic" ] && continue
  if [ "$#" -gt 0 ]; then
    for want in "$@"; do
      [ "$topic" = "$want" ] && fetch "$topic" "$url"
    done
  else
    fetch "$topic" "$url"
  fi
done <<< "$PAPERS"

echo "done — PDFs in papers/pdfs/"
