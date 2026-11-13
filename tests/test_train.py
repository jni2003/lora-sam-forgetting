# AI generated test for training loop and LoRA
"""Checks for src/train.py on a tiny random Llama model (no download, runs in seconds on CPU)."""

import copy

import numpy as np
import pytest
import torch

from src.train import add_lora, lr_multiplier, train
from tests.test_evaluate import VOCAB, tiny_model


def pattern_rows(numRows, seqLen=16):
    """Every row is the same counting pattern 0, 1, 2, ...: easy to learn, so the loss must drop."""
    return np.tile(np.arange(seqLen) % VOCAB, (numRows, 1)).astype(np.uint32)


def test_lr_schedule_warms_up_then_decays_to_ten_percent():
    total = 100  # warmup = first 10 steps
    assert lr_multiplier(0, total) == pytest.approx(0.1)   # starts small
    assert lr_multiplier(9, total) == pytest.approx(1.0)   # peak at the end of warmup
    assert lr_multiplier(total, total) == pytest.approx(0.1)  # ends at 10% of the peak
    decay = [lr_multiplier(s, total) for s in range(10, total)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))   # only goes down after warmup


def test_full_fine_tuning_lowers_the_loss():
    model = tiny_model()
    losses = train(model, pattern_rows(40), lr=1e-2, batchSize=2, microBatchSize=1, device="cpu", seed=0)
    assert len(losses) == 20
    assert losses[-1] < 0.5 * losses[0]


def test_gradient_accumulation_matches_one_big_batch():
    # One step with a batch of 4 in one piece vs. split into micro-batches of 2 should give the
    # same weights. Catches forgetting to divide the loss by the number of micro-batches.
    rows = np.random.default_rng(0).integers(0, VOCAB, size=(4, 16)).astype(np.uint32)
    whole, split = tiny_model(), tiny_model()
    train(whole, rows, lr=1e-3, batchSize=4, microBatchSize=4, device="cpu", seed=0)
    train(split, rows, lr=1e-3, batchSize=4, microBatchSize=2, device="cpu", seed=0)
    for a, b in zip(whole.parameters(), split.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_lora_trains_adapters_and_leaves_base_weights_untouched():
    base = tiny_model()
    before = copy.deepcopy(base.state_dict())
    model = add_lora(base, rank=4, alpha=8)
    losses = train(model, pattern_rows(40), lr=1e-2, batchSize=2, microBatchSize=1, device="cpu", seed=0)
    assert losses[-1] < losses[0]

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)  # only adapters train

    # The original weights inside the LoRA-wrapped model are unchanged...
    for name, value in model.state_dict().items():
        if "lora_" in name:
            continue
        originalName = name.replace("base_model.model.", "").replace("base_layer.", "")
        assert torch.equal(value, before[originalName]), name

    # ...and merging folds the learned update into them, so the merged model differs from the base.
    merged = model.merge_and_unload()
    assert not torch.equal(merged.model.layers[0].self_attn.q_proj.weight, before["model.layers.0.self_attn.q_proj.weight"])
