# AI generated Script to build the StarCoder fien-tuning data
"""Build the StarCoder fine-tuning data: Python code, packed into 1024-token rows.

Matches Watts et al.'s recipe (their new_utils/tokenize_data.py): the Python subset of
bigcode/starcoderdata, each file's `content` tokenized and followed by EOS, packed end to end,
loss on all code tokens.

Train and validation come from different parquet files (the first and the last of the 59),
so no code file appears in both. Validation loss is the "how well did it learn code" axis.

Usage (run from the repo root; needs the StarCoder terms accepted on Hugging Face):
    python -m scripts.make_starcoder                                        # 10M train + 1M val tokens
    python -m scripts.make_starcoder --train-tokens 20000 --val-tokens 5000 --out-dir data/tiny   # quick local test
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data import pack_documents, save_rows

REPO = "bigcode/starcoderdata"
TRAIN_FILE = "python/train-00000-of-00059.parquet"
VAL_FILE = "python/train-00058-of-00059.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--train-tokens", type=int, default=10_000_000)
parser.add_argument("--val-tokens", type=int, default=1_000_000)
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--out-dir", default="data")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", model_max_length=10**9)

for splitName, fileName, maxTokens in [("train", TRAIN_FILE, args.train_tokens), ("val", VAL_FILE, args.val_tokens)]:
    # streaming=True reads the file as we go, so we only download the part we use.
    codeFiles = load_dataset(REPO, data_files=[fileName], split="train", streaming=True)
    docTokenLists = (tok(f["content"], add_special_tokens=False)["input_ids"] for f in codeFiles)
    rows, numDocs = pack_documents(docTokenLists, tok.eos_token_id, args.seq_len, maxTokens)
    # We stopped partway through the file. Close the stream now so the parquet reader cleans up
    # while Python is still running; otherwise the script hangs at exit (pyarrow issue 45214).
    docTokenLists.close()
    print(f"{splitName}: {numDocs} code files from {fileName}")
    save_rows(
        rows,
        f"{args.out_dir}/starcoder_{splitName}.npy",
        info={"source": REPO, "file": fileName, "split": splitName, "tokenizer": "allenai/dolma2-tokenizer"},
    )
