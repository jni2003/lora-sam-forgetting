# AI Generated Script to break down loss by token position and document boundaries responding to LoRA's unexpected DCLM loss decrease
"""Diagnostic: where in the text does a model's loss differ? (Built to explain why LoRA lowered DCLM loss.)

For each model, computes the loss on every token of the first --max-rows rows, then averages it
over groups of tokens:
  - by position in the 1024-token row (early positions have little context);
  - the token right after an end-of-text (EOS) token, i.e. the first token of a new document;
  - predicting the EOS token itself (the end of a document);
  - all other tokens.
If a fine-tuned model's lower loss is spread evenly over all groups, the change is global (e.g. the
base model was miscalibrated); if it is concentrated in one group, it is about formatting.

Usage (on the cluster, free on the CPU partition or quick on a GPU):
    python -m scripts.position_loss --data data/dclm_heldout.npy --max-rows 977 \
        --models checkpoints/olmo-60m-sam checkpoints/finetuned/sam_lora8_lr5e-4 checkpoints/finetuned/sam_full_lr3e-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

from src.convert import load_olmo_as_llama
from src.evaluate import token_losses

EOS_ID = 100257  # allenai/dolma2-tokenizer end-of-text

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--data", default="data/dclm_heldout.npy")
parser.add_argument("--max-rows", type=int, default=977, help="977 rows = the 1M-token fast subset")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--out", default=None, help="optional JSON output")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))
rows = np.load(args.data)[: args.max_rows]
targets = rows[:, 1:]                       # the token each loss column predicts
previous = rows[:, :-1]                     # the token just before it
positions = np.broadcast_to(np.arange(1, rows.shape[1]), targets.shape)  # position of the predicted token

groups = {
    "pos 1-15": positions < 16,
    "pos 16-127": (positions >= 16) & (positions < 128),
    "pos 128-511": (positions >= 128) & (positions < 512),
    "pos 512-1023": positions >= 512,
    "first token of a doc (after EOS)": previous == EOS_ID,
    "predicting EOS": targets == EOS_ID,
}
groups["all other tokens"] = ~(groups["first token of a doc (after EOS)"] | groups["predicting EOS"])

results = {}
for modelPath in args.models:
    if (Path(modelPath) / "config.json").exists():
        model = LlamaForCausalLM.from_pretrained(modelPath).to(device)
    else:
        model = load_olmo_as_llama(modelPath).to(device)
    losses = token_losses(model, rows, args.batch_size, device)
    results[modelPath] = {"all": float(losses.mean())} | {g: float(losses[m].mean()) for g, m in groups.items()}
    del model

counts = {g: int(m.sum()) for g, m in groups.items()}
names = list(results)
print(f"{'group':36s} {'tokens':>8s} " + " ".join(f"{Path(n).name[:20]:>20s}" for n in names))
for g in ["all"] + list(groups):
    count = targets.size if g == "all" else counts[g]
    print(f"{g:36s} {count:>8d} " + " ".join(f"{results[n][g]:>20.4f}" for n in names))
if len(names) > 1:
    print("\nchange vs the first model:")
    for n in names[1:]:
        print(f"  {Path(n).name}: " + ", ".join(f"{g} {results[n][g] - results[names[0]][g]:+.3f}" for g in ["all"] + list(groups)))
if args.out:
    Path(args.out).write_text(json.dumps({"device": device, "counts": counts, "results": results}, indent=2))
