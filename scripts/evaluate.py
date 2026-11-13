# AI generated script to evaluate a model's loss on pre-packed datasets
"""Evaluate one model's loss on one or more packed datasets and save the numbers as JSON.

The model can be a raw OLMo checkpoint folder (config.yaml + model.pt, converted on load) or a
folder saved by transformers' save_pretrained (has config.json), e.g. a fine-tuned model.

Usage (run from the repo root):
    python -m scripts.evaluate --model checkpoints/olmo-60m-sam \
        --data data/dclm_heldout.npy data/starcoder_val.npy --out results/base_eval_sam.json
    # fast subset: only the first 977 rows (~1M tokens) of each dataset
    python -m scripts.evaluate --model checkpoints/olmo-60m-sam --data data/dclm_heldout.npy --max-rows 977 --out ...

If a dataset has a matching `<name>_fileids.npy` (the DCLM slice does), the loss per source file is
reported too, to check that no single file dominates.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

from src.convert import load_olmo_as_llama
from src.evaluate import row_losses, summarize

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="checkpoint folder")
parser.add_argument("--data", nargs="+", required=True, help="one or more .npy files of packed rows")
parser.add_argument("--max-rows", type=int, default=None, help="use only the first N rows of each dataset")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--out", required=True, help="where to write the JSON results")
args = parser.parse_args()

# GPU if there is one (the cluster), else CPU. Never Apple's "mps": it froze the 8 GB Mac.
device = "cuda" if torch.cuda.is_available() else "cpu"

if (Path(args.model) / "config.json").exists():
    model = LlamaForCausalLM.from_pretrained(args.model)
else:
    model = load_olmo_as_llama(args.model)
model.to(device)

results = {"model": args.model, "device": device, "datasets": {}}
for dataPath in args.data:
    rows = np.load(dataPath)[: args.max_rows]  # [:None] keeps everything
    fileIdsPath = Path(dataPath.replace(".npy", "_fileids.npy"))
    fileIds = np.load(fileIdsPath)[: args.max_rows] if fileIdsPath.exists() else None

    startTime = time.time()
    rowLoss = row_losses(model, rows, args.batch_size, device)
    summary = summarize(rowLoss, fileIds)
    summary["seconds"] = round(time.time() - startTime, 1)
    results["datasets"][Path(dataPath).stem] = summary
    print(f"{Path(dataPath).stem}: loss {summary['loss']:.4f} ± {summary['stdErr']:.4f} "
          f"({summary['numRows']} rows, {summary['seconds']}s)")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=2))
print(f"saved {args.out}")
