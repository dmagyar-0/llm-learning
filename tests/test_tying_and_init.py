"""Tests for lesson 13 — weight tying and the GPT-2 initialization scheme.

Two coupled claims. **Tying**: the input embedding and the output projection are
literally one matrix (fewer params, one gradient). **Init**: GPT-2's recipe —
0.02 everywhere, biases zero, and the two per-block residual-writing projections
shrunk by 1/√(2N) — which together make an untrained model's loss start at the
calibrated ln(vocab). That last number is the headline sanity check for "is my
init correct?", so it gets its own test.
"""

import math
from dataclasses import replace

import torch

from llmlab.data.toy import PAD
from llmlab.models.gpt import GPT, GPTConfig
from llmlab.training.loss import masked_cross_entropy


def _wide_cfg(**kw) -> GPTConfig:
    """A model wide/deep enough that std estimates are statistically meaningful
    (tiny's 32-wide matrices are too small to measure a std against)."""
    base = dict(vocab_size=256, d_model=128, num_heads=4, num_layers=6, max_len=64)
    base.update(kw)
    return GPTConfig(**base)


# ------------------------------------------------- weight tying

def test_head_and_embedding_are_the_same_parameter():
    """Not merely equal — the SAME object. Row v is token v's input vector and
    the output direction that scores 'next token = v'; one tensor serves both,
    so a single gradient updates it once per step."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())

    assert model.lm_head.weight is model.embed.table.weight
    # Editing one is editing the other — proof it's shared storage, not a copy.
    with torch.no_grad():
        model.embed.table.weight[0, 0] = 123.0
    assert model.lm_head.weight[0, 0].item() == 123.0


def test_tying_removes_a_whole_vocab_by_dmodel_matrix():
    """The saving is exactly one (vocab × d_model) matrix — for GPT-2's 50257
    vocab that's ~40M params folded away. The untied model carries it twice."""
    cfg = _wide_cfg()
    tied = sum(p.numel() for p in GPT(cfg).parameters())
    untied = sum(p.numel() for p in GPT(replace(cfg, tie_weights=False)).parameters())

    assert untied - tied == cfg.vocab_size * cfg.d_model


def test_untied_model_has_two_independent_matrices():
    """With the knob off, head and embedding are distinct objects again — the
    escape hatch exists, but GPT-2's default (and ours) is tied."""
    model = GPT(_wide_cfg(tie_weights=False))
    assert model.lm_head.weight is not model.embed.table.weight


def test_head_has_no_bias():
    """GPT-2's output projection is pure: a bias would be a per-token constant
    with no counterpart in the embedding it's tied to."""
    assert GPT(GPTConfig.tiny()).lm_head.bias is None


def test_tied_weight_gets_gradient_from_both_roles():
    """The shared matrix is used at the bottom (embedding lookup) and the top
    (logits). A backward from the LM loss must therefore reach it — and it
    appears just once in named_parameters (PyTorch de-dupes shared tensors)."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())
    ids = torch.randint(1, 32, (4, 9))
    masked_cross_entropy(model(ids[:, :-1]), ids[:, 1:], pad_id=PAD).backward()

    shared = model.embed.table.weight
    assert shared.grad is not None and shared.grad.abs().sum() > 0
    names = [n for n, _ in model.named_parameters()]
    assert names.count("embed.table.weight") == 1
    assert "lm_head.weight" not in names   # de-duplicated: it's the same tensor


# ------------------------------------------------- the init scheme

def test_standard_layers_init_at_gpt2_std():
    """0.02 for Linear weights and both embedding tables (wte, wpe); biases zero.
    This is GPT-2's single init constant."""
    torch.manual_seed(0)
    model = GPT(_wide_cfg())
    b0 = model.blocks[0]

    for w in [b0.attention.w_q.weight, b0.attention.w_k.weight,
              b0.ffn.w1.weight, model.embed.table.weight,
              model.positional.table.weight]:
        assert abs(w.std().item() - 0.02) < 0.004, w.std().item()
    assert torch.all(b0.attention.w_q.bias == 0)  # biases start at zero


def test_residual_projections_are_scaled_by_depth():
    """The one non-obvious rule (lesson 10's payoff): the layers that WRITE to
    the residual stream — attention W_O and the FFN's 2nd Linear — are shrunk by
    1/√(2N) so that summing 2N of them keeps the stream's scale stable with
    depth. Their std is √(2N)× smaller than an ordinary layer's."""
    N = 6
    torch.manual_seed(0)
    model = GPT(_wide_cfg(num_layers=N))
    expected = 0.02 / math.sqrt(2 * N)

    for block in model.blocks:
        assert abs(block.attention.w_o.weight.std().item() - expected) < 0.001
        assert abs(block.ffn.w2.weight.std().item() - expected) < 0.001
    # ...and NON-residual projections are NOT scaled (still ~0.02).
    assert model.blocks[0].attention.w_q.weight.std().item() > 0.015


def test_deeper_models_scale_the_residual_projections_more():
    """The scaling tracks depth: double the layers and the residual-projection
    std shrinks by √2. This is what keeps the pre-norm stream's variance ~constant
    across model sizes."""
    torch.manual_seed(0)
    shallow = GPT(_wide_cfg(num_layers=4))
    deep = GPT(_wide_cfg(num_layers=16))

    s = shallow.blocks[0].ffn.w2.weight.std().item()
    d = deep.blocks[0].ffn.w2.weight.std().item()
    assert abs((s / d) - math.sqrt(16 / 4)) < 0.1   # ratio ≈ √(16/4) = 2


def test_initial_loss_is_calibrated_to_ln_vocab():
    """The headline check for correct init. Before any training, a good LM should
    be maximally uncertain — assign ~uniform probability over the vocab — so its
    cross-entropy starts at ln(vocab_size). GPT-2's small 0.02 tied init produces
    exactly this: tiny embedding rows → tiny logits → near-uniform softmax. A
    too-large embedding init would start meaningfully higher (next test). This is
    *the* reason to care about init: a wrong start wastes the first steps just
    deflating logits instead of learning."""
    torch.manual_seed(0)
    cfg = _wide_cfg(vocab_size=256, num_layers=6)
    model = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (16, 33))

    loss = masked_cross_entropy(model(ids[:, :-1]), ids[:, 1:], pad_id=PAD).item()

    assert abs(loss - math.log(cfg.vocab_size)) < 0.25, loss


def test_large_embedding_init_breaks_the_calibration():
    """Contrast, to pin the *actual* mechanism: because the head is tied, the
    embedding table's std sets the logit scale directly. Re-init it large (the
    old 1/√d_model default ≈ 0.088 for d=128) and the logits inflate, so the
    starting loss overshoots ln(vocab) badly. It is the small 0.02 *init* that is
    load-bearing here — not the √d_model forward flag, which pre-norm normalizes
    away (that flag only balances content vs. position)."""
    torch.manual_seed(0)
    cfg = _wide_cfg(vocab_size=256, num_layers=6)
    model = GPT(cfg)
    # Blow up only the (tied) embedding table; leave everything else GPT-2-correct.
    torch.nn.init.normal_(model.embed.table.weight, mean=0.0, std=1.0 / math.sqrt(cfg.d_model))

    ids = torch.randint(0, 256, (16, 33))
    loss = masked_cross_entropy(model(ids[:, :-1]), ids[:, 1:], pad_id=PAD).item()

    assert loss > math.log(256) + 1.0   # markedly worse than the calibrated start
