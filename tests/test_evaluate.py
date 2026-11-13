# AI Generated test for the loss computation on a tiny model
"""Checks for src/evaluate.py on a tiny random Llama model (no download, runs in seconds on CPU)."""

import math

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.evaluate import row_losses, summarize

VOCAB = 64


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=32, tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def random_rows(numRows=5, seqLen=16):
    return np.random.default_rng(0).integers(0, VOCAB, size=(numRows, seqLen)).astype(np.uint32)


def test_uniform_model_has_loss_log_vocab():
    # With a zeroed output layer every token gets the same score, so each true token has
    # probability 1/VOCAB and the loss is exactly ln(VOCAB). Catches a wrong loss formula.
    model = tiny_model()
    torch.nn.init.zeros_(model.lm_head.weight)
    losses = row_losses(model, random_rows(), batchSize=2, device="cpu")
    assert np.allclose(losses, math.log(VOCAB), atol=1e-5)


def test_matches_transformers_own_loss_for_any_batch_size():
    # transformers computes the same next-token loss when given labels. Agreeing with it checks
    # that we line up predictions with the right (next) tokens. Batch sizes 1 and 3 (with a
    # smaller last batch of 2) check that splitting into batches doesn't change the result.
    model = tiny_model()
    rows = random_rows()
    ids = torch.from_numpy(rows.astype(np.int64))
    with torch.no_grad():
        expected = model(input_ids=ids, labels=ids).loss.item()
    for batchSize in [1, 3]:
        assert abs(row_losses(model, rows, batchSize, "cpu").mean() - expected) < 1e-5


def test_summary_per_file():
    summary = summarize(np.array([1.0, 2.0, 3.0, 4.0]), fileIds=np.array([0, 1, 0, 1]))
    assert summary["loss"] == 2.5
    assert summary["lossPerFile"] == {0: 2.0, 1: 3.0}
