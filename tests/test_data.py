# AI generated test for joining documents with EOS and cutting -> rows
"""Checks for the packing helper in src/data.py (no downloads needed)."""

import numpy as np

from src.data import pack_documents


def test_documents_are_joined_with_eos_and_cut_into_rows():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]  # stream: 1 2 3 E 4 5 E 6 7 8 9 E  (E = EOS = 0)
    rows, numDocs = pack_documents(docs, eosId=0, seqLen=4, maxTokens=100)
    expected = np.array([[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]], dtype=np.uint32)
    assert np.array_equal(rows, expected)
    assert numDocs == 3


def test_stops_at_max_tokens_and_drops_partial_row():
    docs = ([10, 11, 12] for _ in range(1000))  # a lazy generator, like streaming from Hugging Face
    rows, numDocs = pack_documents(docs, eosId=0, seqLen=4, maxTokens=10)
    assert rows.shape == (2, 4)    # 10 tokens -> two full rows of 4; the leftover 2 tokens are dropped
    assert numDocs == 3            # stopped reading after 3 docs (12 tokens >= 10), not all 1000
    assert rows.dtype == np.uint32  # token IDs up to 100,277 need 32 bits
