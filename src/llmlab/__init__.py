"""llmlab — LLM architectures implemented from scratch, for learning.

The library is organized so that an "architecture" is just a configuration that
plugs reusable components together:

    components/   Reusable building blocks (attention variants, norms, FFNs,
                  positional encodings). Each block exists because some paper
                  introduced it; the docstring says which one.
    models/       Architectures assembled from components. gpt2.py and llama.py
                  should differ in *configuration*, not in structure.
    training/     Training loop, LR schedules, checkpointing (must survive
                  ephemeral remote sessions).
    data/         Datasets and tokenizers (char-level first, then our own BPE).

Design rule: readability beats performance. We write the naive, obvious version
first and optimize only as its own explicit lesson.
"""

__version__ = "0.0.1"
