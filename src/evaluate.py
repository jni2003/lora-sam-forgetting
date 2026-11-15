# AI generated helper to compute per-row loss, standard error, and per-file loss
"""Language-modeling loss of a model on packed token rows (see src/data.py).

Used for both metrics:
- forgetting: loss on the held-out DCLM slice, compared with the base model's loss;
- fine-tuning loss: loss on the StarCoder validation rows.

The loss is the average next-token cross-entropy, in nats: for each position, how surprised the
model was by the true next token (-log of the probability it gave that token). Every row has the
same length, so the loss over all tokens equals the average of the per-row losses.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()  # evaluation only: no gradients, which saves memory
def token_losses(model, rows, batchSize, device):
    """Return a NumPy array of shape (numRows, seqLen - 1): the loss on every predicted token.

    rows: uint32 array of shape (numRows, seqLen) of token IDs.
    Each row predicts its own tokens 2..seqLen from the tokens before them, so a row of 1024
    tokens contributes 1023 predictions (the first token has nothing before it).
    Column t is the loss of predicting token t+1 from tokens 0..t.
    """
    model.eval()
    losses = []
    for start in range(0, len(rows), batchSize):
        inputIds = torch.from_numpy(rows[start : start + batchSize].astype(np.int64)).to(device)
        logits = model(input_ids=inputIds).logits  # (batch, seqLen, vocab): scores for the next token
        # Position t predicts token t+1: drop the last position's scores and the first token.
        tokenLosses = F.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(),  # (batch * (seqLen-1), vocab)
            inputIds[:, 1:].flatten(),             # (batch * (seqLen-1),)
            reduction="none",
        ).view(inputIds.shape[0], -1)
        losses.append(tokenLosses.cpu())
    return torch.cat(losses).numpy()


def row_losses(model, rows, batchSize, device):
    """Return a NumPy array with the average next-token loss of each row."""
    return token_losses(model, rows, batchSize, device).mean(axis=1)


def summarize(rowLoss, fileIds=None):
    """Mean loss, its standard error, and (if fileIds is given) the mean loss per source file.

    Standard error = std across rows / sqrt(numRows): how much the mean would move with a different
    random sample of rows. Rows from the same long document are a bit correlated, so this slightly
    understates the true uncertainty; good enough to tell whether the slice is big enough.
    """
    summary = {
        "loss": float(rowLoss.mean()),
        "stdErr": float(rowLoss.std(ddof=1) / math.sqrt(len(rowLoss))) if len(rowLoss) > 1 else None,
        "numRows": int(len(rowLoss)),
    }
    if fileIds is not None:
        summary["lossPerFile"] = {int(f): float(rowLoss[fileIds == f].mean()) for f in np.unique(fileIds)}
    return summary
