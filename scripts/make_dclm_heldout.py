# AI Generated Script to build the DCLM forgetting evaluation set
"""Build the held-out DCLM slice used to measure forgetting (pretraining loss) for every model.

Picks `--files` random files out of DCLM-baseline's 27,838 (fixed seed, so it's reproducible),
takes an equal share of tokens from the start of each, tokenizes with the models' own tokenizer
(allenai/dolma2-tokenizer), and packs into 1024-token rows (see src/data.py).

Size: 20M tokens from 100 files by default, matching Watts et al.'s 20M-token DCLM validation set.
The rows are shuffled (fixed seed) before saving, so any prefix (e.g. the first 1M tokens = first
~977 rows, used as a fast subset for repeated evals) is a sample spread over all files, not just
the first few. A second array `<out>_fileids.npy` records which file each row came from, so the
eval can report per-file loss.

Spreading over many files avoids depending on one file's quirks (a single file gave noticeably
different losses in our first quick check). "Held-out": the models saw 192B of DCLM's ~4T tokens,
so a random document has only about a 5% chance of having been in training; good enough.

Usage (run from the repo root):
    python -m scripts.make_dclm_heldout                          # full slice: 20M tokens -> data/dclm_heldout.npy (run on the cluster CPU partition)
    python -m scripts.make_dclm_heldout --tokens 20000 --files 2 --out data/dclm_tiny.npy   # quick local test
"""

import argparse
import random

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from src.data import pack_documents, save_rows
import numpy as np

REPO = "mlfoundations/dclm-baseline-1.0"

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=20_000_000, help="total tokens in the slice")
parser.add_argument("--files", type=int, default=100, help="how many random DCLM files to draw from")
parser.add_argument("--seed", type=int, default=0, help="which random files get picked")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--out", default="data/dclm_heldout.npy")
args = parser.parse_args()

# model_max_length only silences a harmless warning about documents longer than the model's context.
tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", model_max_length=10**9)

allFiles = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".jsonl.zst"))
chosenFiles = random.Random(args.seed).sample(allFiles, args.files)
tokensPerFile = args.tokens // args.files

allRows = []
allFileIds = []  # for each row, the index (into chosenFiles) of the file it came from
for fileIndex, fileName in enumerate(chosenFiles):
    # streaming=True reads the file as we go, so we only download the part we use.
    docs = load_dataset(REPO, data_files=[fileName], split="train", streaming=True)
    docTokenLists = (tok(doc["text"], add_special_tokens=False)["input_ids"] for doc in docs)
    rows, numDocs = pack_documents(docTokenLists, tok.eos_token_id, args.seq_len, tokensPerFile)
    print(f"{fileName}: {numDocs} documents -> {rows.shape[0]} rows")
    allRows.append(rows)
    allFileIds.append(np.full(rows.shape[0], fileIndex, dtype=np.uint16))

# Shuffle rows (same permutation for rows and file IDs) so any prefix mixes all files.
allRows = np.concatenate(allRows)
allFileIds = np.concatenate(allFileIds)
order = np.random.default_rng(args.seed).permutation(len(allRows))
allRows, allFileIds = allRows[order], allFileIds[order]

save_rows(
    allRows,
    args.out,
    info={"source": REPO, "files": chosenFiles, "seed": args.seed, "tokenizer": "allenai/dolma2-tokenizer"},
)
# After save_rows, which creates the output folder if needed.
np.save(args.out.replace(".npy", "_fileids.npy"), allFileIds)
