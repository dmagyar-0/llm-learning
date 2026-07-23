"""Tests for lesson 09 — the decoder-only language model.

The model introduces no new *mechanism* (it reuses lesson 03's block behind
lesson 05's causal mask); the claims to pin are therefore about the WIRING and
the OBJECTIVE that the decoder-only collapse creates:

    - shape → one next-token distribution per position
    - causal → editing a future token never moves an earlier position's logits
    - padding → invisible, end-to-end
    - the block reused IS an encoder block (equivalence, not a copy)
    - the LM objective → shift-by-one on ONE stream trains cleanly
    - generation → lesson 08's loop, decoder-only, is deterministic and stops
"""

import torch

from llmlab.components.block import TransformerBlock
from llmlab.data.toy import EOS, PAD
from llmlab.models.gpt import GPT, GPTConfig
from llmlab.training.loss import masked_cross_entropy


def make_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig.tiny())


# ------------------------------------------------- shapes and configs

def test_output_shape_is_per_position_logits():
    """(batch, seq) → (batch, seq, vocab): every position predicts the next
    token, so one pass over an n-token stream yields n training signals."""
    model = make_model()
    ids = torch.randint(1, 32, (2, 7))

    logits = model(ids)

    assert logits.shape == (2, 7, 32)


def test_tiny_config_is_actually_tiny():
    """The teaching contract demands a CPU-in-seconds config. Decoder-only is
    LEANER than the enc–dec toy (one stream, no cross-attention), so pin that it
    stays well under the same bound."""
    n_params = sum(p.numel() for p in make_model().parameters())
    assert n_params < 100_000, f"tiny config grew to {n_params} params"


# ------------------------------------------------- the wiring claims

def test_is_causal_end_to_end():
    """The defining property. Change a token at position 4 and every EARLIER
    position's logits must be bit-for-bit unchanged through the full stack —
    a single missing causal mask anywhere would leak the future and break this."""
    model = make_model()
    ids = torch.randint(1, 32, (1, 6))
    edited = ids.clone()
    edited[0, 4] = (ids[0, 4] + 1) % 31 + 1  # change token 4, stay non-pad

    with torch.no_grad():
        a = model(ids)
        b = model(edited)

    assert torch.equal(a[:, :4], b[:, :4])        # past sealed
    assert not torch.allclose(a[:, 4:], b[:, 4:])  # the edit did register


def test_padding_is_invisible():
    """Appending PAD to the stream must not change logits at the real
    positions — the padding mask makes pad keys unattendable, end-to-end."""
    model = make_model()
    ids = torch.randint(1, 32, (1, 5))
    padded = torch.cat([ids, torch.full((1, 3), PAD)], dim=1)  # (1, 8)

    with torch.no_grad():
        clean = model(ids)
        out = model(padded)

    assert torch.allclose(clean, out[:, :5], atol=1e-5)


def test_gpt_block_is_the_encoder_block():
    """The lesson's equivalence claim, made executable: the block GPT stacks is
    LITERALLY lesson 03's `TransformerBlock`. A GPT block and an encoder block
    are the same object — only the mask fed in differs."""
    model = make_model()
    assert all(isinstance(b, TransformerBlock) for b in model.blocks)


# ------------------------------------------------- the LM objective

def test_next_token_objective_trains():
    """The Phase-2 objective: one stream predicting itself, shifted by one.
    `input = tokens[:, :-1]`, `labels = tokens[:, 1:]` — no source. A few steps
    of SGD on a single fixed batch must drive the loss down, proving the whole
    forward→shift→masked-loss→backward path is wired and differentiable."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())
    tokens = torch.randint(1, 32, (4, 9))          # one stream per row
    inputs, labels = tokens[:, :-1], tokens[:, 1:]  # the shift-by-one

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(20):
        loss = masked_cross_entropy(model(inputs), labels, pad_id=PAD)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < first  # it learned to memorize the fixed batch


def test_gradients_reach_every_parameter():
    """Dead-wiring check: one backward from a plausible loss must touch every
    parameter — the embedding, all blocks, the head. A grad-less parameter is a
    component the forward pass forgot to call."""
    model = make_model()
    ids = torch.randint(1, 32, (2, 6))

    model(ids).logsumexp(dim=-1).mean().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient at {name}"


# ------------------------------------------------- generation

def test_generate_extends_the_prompt_deterministically():
    """`generate` is lesson 08's loop, decoder-only. Greedy → deterministic
    (same prompt, same continuation), and it appends exactly max_new_tokens
    when no EOS is requested."""
    model = make_model()
    prompt = torch.randint(1, 32, (2, 3))

    out_a = model.generate(prompt, max_new_tokens=5)
    out_b = model.generate(prompt, max_new_tokens=5)

    assert out_a.shape == (2, 3 + 5)             # prompt preserved + grown
    assert torch.equal(out_a[:, :3], prompt)     # prompt is a true prefix
    assert torch.equal(out_a, out_b)             # greedy is deterministic


def test_generate_stops_and_freezes_at_eos():
    """With an eos_id, a row that emits EOS is frozen with PAD forever — so the
    batch can decode together even when rows finish at different steps."""
    torch.manual_seed(0)
    model = GPT(GPTConfig.tiny())
    prompt = torch.randint(3, 32, (1, 2))  # avoid specials in the prompt

    out = model.generate(prompt, max_new_tokens=6, eos_id=EOS)

    # Once EOS appears, every subsequent generated position must be PAD.
    row = out[0].tolist()
    if EOS in row:
        first_eos = row.index(EOS)
        assert all(t == PAD for t in row[first_eos + 1:])
