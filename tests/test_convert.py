# AI Generated Test for converting olmo model -> llama
"""Checks for src/convert.py on a tiny random model in OLMo's layout (no download needed).

What these catch: missing or extra weights, wrong shapes, and the split order of the two packed
matrices. Whether the real weights produce a sensible model is checked separately by the
pretraining-loss check on DCLM.
"""

import torch

from src.convert import convert_state_dict, llama_config_from_olmo
from transformers import LlamaForCausalLM

# Same architecture flags as the real checkpoints, but tiny sizes.
TINY_CFG = {
    "d_model": 16, "n_heads": 2, "n_layers": 2, "mlp_ratio": 8, "mlp_hidden_size": None,
    "embedding_size": 64, "max_sequence_length": 32, "layer_norm_eps": 1e-6, "rope_theta": 10000,
    "eos_token_id": 0, "pad_token_id": 1,
    "activation_type": "swiglu", "layer_norm_type": "rms", "layer_norm_with_affine": True,
    "include_bias": False, "bias_for_layer_norm": False, "rope": True, "alibi": False,
    "attention_layer_norm": False, "norm_after": False, "block_type": "sequential",
    "weight_tying": False, "clip_qkv": None, "n_kv_heads": None, "multi_query_attention": None,
    "scale_logits": False,
}


def tiny_olmo_weights(cfg):
    """Random weights with OLMo's names and shapes (matching what inspect_checkpoint.py printed)."""
    d, vocab = cfg["d_model"], cfg["embedding_size"]
    ff = cfg["mlp_ratio"] * d  # ff_proj output; half of it is the MLP hidden width
    w = {
        "transformer.wte.weight": torch.randn(vocab, d),
        "transformer.ln_f.weight": torch.randn(d),
        "transformer.ff_out.weight": torch.randn(vocab, d),
    }
    for i in range(cfg["n_layers"]):
        p = f"transformer.blocks.{i}."
        w[p + "att_proj.weight"] = torch.randn(3 * d, d)
        w[p + "attn_out.weight"] = torch.randn(d, d)
        w[p + "ff_proj.weight"] = torch.randn(ff, d)
        w[p + "ff_out.weight"] = torch.randn(d, ff // 2)
        w[p + "attn_norm.weight"] = torch.randn(d)
        w[p + "ff_norm.weight"] = torch.randn(d)
    return w


def test_every_weight_fits_llama():
    """strict=True raises if any Llama weight is missing, any converted weight is unused, or a shape is wrong."""
    model = LlamaForCausalLM(llama_config_from_olmo(TINY_CFG))
    model.load_state_dict(convert_state_dict(tiny_olmo_weights(TINY_CFG), TINY_CFG), strict=True)
    logits = model(torch.tensor([[2, 3, 4]])).logits
    assert logits.shape == (1, 3, TINY_CFG["embedding_size"])


def test_packed_matrices_split_in_the_right_order():
    """q, k, v come out in that order; the first half of ff_proj is 'up', the second is 'gate'."""
    d = TINY_CFG["d_model"]
    olmo = tiny_olmo_weights(TINY_CFG)
    llama = convert_state_dict(olmo, TINY_CFG)
    att, ff = olmo["transformer.blocks.0.att_proj.weight"], olmo["transformer.blocks.0.ff_proj.weight"]
    half = ff.shape[0] // 2
    assert torch.equal(llama["model.layers.0.self_attn.q_proj.weight"], att[:d])
    assert torch.equal(llama["model.layers.0.self_attn.k_proj.weight"], att[d:2 * d])
    assert torch.equal(llama["model.layers.0.self_attn.v_proj.weight"], att[2 * d:])
    assert torch.equal(llama["model.layers.0.mlp.up_proj.weight"], ff[:half])
    assert torch.equal(llama["model.layers.0.mlp.gate_proj.weight"], ff[half:])
