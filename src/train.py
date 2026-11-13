# AI Generated helper for full and LoRA finetuning 
"""Fine-tune a model on packed token rows: full fine-tuning or LoRA.

Recipe follows Watts et al.'s fine-tuning code (sft_small.py / artifacts.py), except weight decay,
which their paper tuned to 0 (Appendix C.2, Table 7):
    AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0, gradient clipping at norm 1.0,
    learning rate: linear warmup over the first 10% of steps, then cosine decay to 10% of the peak,
    batch of 64 rows x 1024 tokens, one pass over the data (10M tokens -> about 152 steps).

Full fine-tuning updates every weight. LoRA (Hu et al. 2022) freezes the pretrained weights and
learns a low-rank update for each weight matrix W: W + (alpha / rank) * B @ A, where A is
(rank x in) and B is (out x rank). B starts at zero, so training starts from the exact base model.
"""

import math
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

# Every weight matrix inside the transformer blocks (attention and MLP). The embeddings and the
# output layer stay frozen, as in standard LoRA setups.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_lora(model, rank, alpha):
    """Freeze the model and attach LoRA adapters of the given rank to every block weight matrix."""
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS)
    return get_peft_model(model, config)


def lr_multiplier(step, totalSteps, warmupFrac=0.1, finalFrac=0.1):
    """Learning rate at `step` as a fraction of the peak: linear warmup, then cosine to finalFrac."""
    warmupSteps = max(1, round(warmupFrac * totalSteps))
    if step < warmupSteps:
        return (step + 1) / warmupSteps  # reaches 1.0 at the end of warmup
    progress = (step - warmupSteps) / max(1, totalSteps - warmupSteps)  # 0 -> 1 over the decay
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
    return finalFrac + (1 - finalFrac) * cosine


def train(model, rows, lr, batchSize, microBatchSize, device, seed, logEvery=10):
    """Train for one pass over `rows` (shuffled). Returns the list of per-step training losses.

    A batch of 64 x 1024 tokens doesn't fit through the model at once (the output scores alone
    would be 64 x 1024 x 100,352 numbers), so each batch is split into micro-batches whose
    gradients add up before the update ("gradient accumulation"). Same math as one big batch.
    """
    assert batchSize % microBatchSize == 0, "batch size must be a multiple of the micro-batch size"
    accumSteps = batchSize // microBatchSize
    totalSteps = len(rows) // batchSize  # any leftover rows (less than one batch) are skipped

    torch.manual_seed(seed)
    order = np.random.default_rng(seed).permutation(len(rows))

    # Only trainable weights go to the optimizer (with LoRA: just the adapters).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, totalSteps))

    # On the GPU, run the forward pass in bfloat16 (as in pretraining) for speed; weights stay fp32.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))

    model.train()
    losses = []
    startTime = time.time()
    for step in range(totalSteps):
        stepLoss = 0.0
        for micro in range(accumSteps):
            start = step * batchSize + micro * microBatchSize
            batchRows = rows[order[start : start + microBatchSize]]
            inputIds = torch.from_numpy(batchRows.astype(np.int64)).to(device)
            with autocast:
                # labels=input_ids: transformers shifts them by one internally (next-token loss).
                loss = model(input_ids=inputIds, labels=inputIds).loss
            (loss / accumSteps).backward()  # divide so the summed gradients equal the batch average
            stepLoss += loss.item() / accumSteps
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        stepLr = scheduler.get_last_lr()[0]  # the learning rate this step uses
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(stepLoss)
        if step % logEvery == 0 or step == totalSteps - 1:
            print(f"step {step + 1}/{totalSteps}  loss {stepLoss:.4f}  "
                  f"lr {stepLr:.2e}  {time.time() - startTime:.0f}s", flush=True)
    model.eval()
    return losses
