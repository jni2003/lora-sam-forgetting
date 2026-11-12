# AI Generated Script to convert olmo cp to llama (for LoRA ffinetuning)
"""Load Watts et al.'s raw OLMo checkpoints as a Hugging Face `LlamaForCausalLM`.

A checkpoint folder holds `config.yaml` (the architecture) and `model.pt` (the trained weights,
stored under OLMo's own names). The model math matches Llama exactly (pre-norm RMSNorm, SwiGLU,
RoPE, no biases), so we build an empty Llama with the same settings and copy the weights in.
Most weights only need renaming; two are packed in OLMo and must be split:

    OLMo name                    Llama name                                  Note
    wte                          model.embed_tokens                          input embeddings
    blocks.N.attn_norm           model.layers.N.input_layernorm
    blocks.N.att_proj            model.layers.N.self_attn.{q,k,v}_proj       split in 3: q, k, v
    blocks.N.attn_out            model.layers.N.self_attn.o_proj
    blocks.N.ff_norm             model.layers.N.post_attention_layernorm
    blocks.N.ff_proj             model.layers.N.mlp.{up,gate}_proj           split in 2: up first, then gate
    blocks.N.ff_out              model.layers.N.mlp.down_proj
    ln_f                         model.norm                                  final norm
    ff_out (top level)           lm_head                                     output layer

The split orders come from the authors' `olmo/model.py`:
    q, k, v = qkv.split(...)                      -> q first, then k, then v
    x, gate = x.chunk(2); silu(gate) * x          -> first half "up" (x), second half "gate"

Usage:
    from src.convert import load_olmo_as_llama
    model = load_olmo_as_llama("checkpoints/olmo-60m-sam")
"""

from pathlib import Path

import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM


def read_olmo_config(ckpt_dir) -> dict:
    """Return the `model` section of the checkpoint's config.yaml."""
    config = yaml.safe_load((Path(ckpt_dir) / "config.yaml").read_text())
    return config["model"]


def check_architecture(cfg: dict) -> None:
    """Fail loudly if the checkpoint isn't the Llama-compatible variant this converter assumes.

    The 20M, 60M and 150M checkpoints all share these settings; anything else would need
    a different mapping, so we refuse instead of silently producing a wrong model.
    """
    expected = {
        "activation_type": "swiglu",
        "layer_norm_type": "rms",
        "layer_norm_with_affine": True,   # norms have learned scales
        "include_bias": False,
        "bias_for_layer_norm": False,
        "rope": True,
        "alibi": False,
        "attention_layer_norm": False,    # no QK-norm
        "norm_after": False,              # pre-norm, like Llama
        "block_type": "sequential",
        "weight_tying": False,            # separate input and output embeddings
        "clip_qkv": None,
        "n_kv_heads": None,               # plain multi-head attention
        "multi_query_attention": None,
        "scale_logits": False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported OLMo config: {key}={cfg.get(key)!r}, expected {value!r}")


def llama_config_from_olmo(cfg: dict) -> LlamaConfig:
    """Translate OLMo's architecture settings into the equivalent LlamaConfig."""
    check_architecture(cfg)
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    # OLMo's ff_proj outputs mlp_ratio * d_model numbers, which SwiGLU splits in half,
    # so the MLP's hidden width (Llama's intermediate_size) is half of that.
    ff_proj_out = cfg["mlp_hidden_size"] or cfg["mlp_ratio"] * d_model
    return LlamaConfig(
        vocab_size=cfg["embedding_size"],          # 100,352: the padded size the weights were trained with
        hidden_size=d_model,
        intermediate_size=ff_proj_out // 2,
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=d_model // n_heads,
        hidden_act="silu",                          # SwiGLU's gate activation
        max_position_embeddings=cfg["max_sequence_length"],
        rms_norm_eps=cfg["layer_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": float(cfg["rope_theta"])},
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        eos_token_id=cfg["eos_token_id"],
        pad_token_id=cfg["pad_token_id"],
        bos_token_id=None,                          # OLMo uses no beginning-of-text token
    )


def convert_state_dict(olmo: dict, cfg: dict) -> dict:
    """Rename and split OLMo weights into Llama's names. Returns a new dict of tensors."""
    d_model = cfg["d_model"]
    llama = {
        "model.embed_tokens.weight": olmo["transformer.wte.weight"],
        "model.norm.weight": olmo["transformer.ln_f.weight"],
        "lm_head.weight": olmo["transformer.ff_out.weight"],
    }
    for i in range(cfg["n_layers"]):
        old = f"transformer.blocks.{i}."
        new = f"model.layers.{i}."

        # Attention: att_proj stacks q, k, v along its rows (3 * d_model rows).
        q, k, v = olmo[old + "att_proj.weight"].split(d_model, dim=0)
        llama[new + "self_attn.q_proj.weight"] = q
        llama[new + "self_attn.k_proj.weight"] = k
        llama[new + "self_attn.v_proj.weight"] = v
        llama[new + "self_attn.o_proj.weight"] = olmo[old + "attn_out.weight"]

        # MLP: ff_proj stacks [up; gate] along its rows. Swapping these is the classic silent bug.
        up, gate = olmo[old + "ff_proj.weight"].chunk(2, dim=0)
        llama[new + "mlp.up_proj.weight"] = up
        llama[new + "mlp.gate_proj.weight"] = gate
        llama[new + "mlp.down_proj.weight"] = olmo[old + "ff_out.weight"]

        llama[new + "input_layernorm.weight"] = olmo[old + "attn_norm.weight"]
        llama[new + "post_attention_layernorm.weight"] = olmo[old + "ff_norm.weight"]

    # Every OLMo weight must be used exactly once: 3 top-level + 6 per block.
    expected_count = 3 + 6 * cfg["n_layers"]
    if len(olmo) != expected_count:
        raise ValueError(f"Expected {expected_count} OLMo weights, found {len(olmo)}: unmapped weights exist")
    return llama


def load_olmo_as_llama(ckpt_dir) -> LlamaForCausalLM:
    """Build a Llama model from config.yaml and fill it with the weights from model.pt."""
    cfg = read_olmo_config(ckpt_dir)
    model = LlamaForCausalLM(llama_config_from_olmo(cfg))  # correct structure, random weights
    # weights_only=True loads only tensors, never code stored in the file (safe for downloads).
    olmo_weights = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)
    # strict=True: error if any Llama weight is missing or any converted weight is unused or misshapen.
    model.load_state_dict(convert_state_dict(olmo_weights, cfg), strict=True)
    model.eval()  # inference mode (no dropout; these models have none, but it's the convention)
    return model
