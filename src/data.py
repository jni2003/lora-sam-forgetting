# AI generated helper for joining documents with EOS and cutting -> 1024-token rows
"""Shared helpers for preparing token data.

Both data scripts (DCLM held-out slice, StarCoder fine-tuning data) do the same thing:
tokenize documents, glue them end to end with an EOS token after each one, and cut the
long stream into fixed-length rows of `seqLen` tokens. That's how the OLMo models were
pretrained and how Watts et al. prepared their fine-tuning data ("packing"): no padding,
and a single row can contain the end of one document and the start of the next.

Rows are saved as a uint32 NumPy array of shape (numRows, seqLen). uint32 is needed because
token IDs go up to 100,277, which doesn't fit in 16 bits.
"""

import json
from pathlib import Path

import numpy as np


def pack_documents(docTokenLists, eosId, seqLen, maxTokens):
    """Concatenate documents (each followed by EOS) and cut the stream into rows of seqLen tokens.

    docTokenLists: any iterable of token-ID lists, one per document (can be a lazy generator,
        so we only tokenize as many documents as we need).
    Stops once maxTokens tokens are collected and drops the incomplete last row.
    Returns (rows, numDocs): a uint32 array of shape (numRows, seqLen) and the number of documents used.
    """
    stream = []
    numDocs = 0
    for docTokens in docTokenLists:
        stream.extend(docTokens)
        stream.append(eosId)  # marks where one document ends and the next begins
        numDocs += 1
        if len(stream) >= maxTokens:
            break
    numRows = min(len(stream), maxTokens) // seqLen
    rows = np.array(stream[: numRows * seqLen], dtype=np.uint32).reshape(numRows, seqLen)
    return rows, numDocs


def save_rows(rows, outPath, info):
    """Save packed rows to outPath (.npy) and a small JSON description next to it (.json)."""
    outPath = Path(outPath)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    np.save(outPath, rows)
    info = {**info, "numRows": int(rows.shape[0]), "seqLen": int(rows.shape[1]), "numTokens": int(rows.size)}
    outPath.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"saved {outPath}: {rows.shape[0]} rows x {rows.shape[1]} tokens = {rows.size:,} tokens")
