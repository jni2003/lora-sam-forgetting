# AI generated script to finetune, evaluate, and save a model
"""Fine-tune one base model on StarCoder (full or LoRA), evaluate it, and save results + weights.

Steps: load the base (raw OLMo checkpoint, converted on load) -> optionally add LoRA adapters ->
train one pass over the training rows (src/train.py) -> merge LoRA into the weights -> evaluate
loss on each --eval-data file (src/evaluate.py) -> save results JSON and the fine-tuned model.

Forgetting = this run's DCLM loss minus the base model's DCLM loss (from scripts/evaluate.py).

Usage (run from the repo root):
    # full fine-tuning, SAM base, learning rate 5e-4
    python -m scripts.train --model checkpoints/olmo-60m-sam --lr 5e-4 --run-name sam_full_lr5e-4 \
        --eval-data data/dclm_heldout.npy data/starcoder_val.npy
    # LoRA rank 8 (alpha defaults to 2 x rank)
    python -m scripts.train --model checkpoints/olmo-60m-sam --lr 5e-4 --lora-rank 8 --run-name sam_lora8_lr5e-4 ...

Outputs: results/<run-name>.json (small, commit it) and checkpoints/finetuned/<run-name>/ (weights,
gitignored).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.convert import load_olmo_as_llama
from src.evaluate import row_losses, summarize
from src.train import add_lora, train

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="base checkpoint folder (raw OLMo)")
parser.add_argument("--train-data", default="data/starcoder_train.npy")
parser.add_argument("--eval-data", nargs="+", required=True, help=".npy files to evaluate after training")
parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
parser.add_argument("--lora-rank", type=int, default=None, help="leave out for full fine-tuning")
parser.add_argument("--lora-alpha", type=float, default=None, help="default: 2 x rank")
parser.add_argument("--batch-size", type=int, default=64, help="rows per optimizer step")
parser.add_argument("--micro-batch-size", type=int, default=8, help="rows per forward pass (memory)")
parser.add_argument("--seed", type=int, default=6198, help="data order; 6198 as in Watts et al.'s code")
parser.add_argument("--max-train-rows", type=int, default=None, help="use only the first N rows (quick tests)")
parser.add_argument("--max-eval-rows", type=int, default=None, help="evaluate only the first N rows of each file")
parser.add_argument("--eval-batch-size", type=int, default=8)
parser.add_argument("--run-name", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"  # never "mps": it froze the 8 GB Mac
startTime = time.time()

model = load_olmo_as_llama(args.model).to(device)
if args.lora_rank is not None:
    alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
    model = add_lora(model, args.lora_rank, alpha)
numTrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{args.run_name}: {numTrainable:,} trainable parameters on {device}")

trainRows = np.load(args.train_data)[: args.max_train_rows]
trainLosses = train(model, trainRows, args.lr, args.batch_size, args.micro_batch_size, device, args.seed)
trainSeconds = time.time() - startTime

if args.lora_rank is not None:
    model = model.merge_and_unload()  # fold B @ A into W, giving a plain Llama model again

evalResults = {}
for dataPath in args.eval_data:
    rows = np.load(dataPath)[: args.max_eval_rows]
    fileIdsPath = Path(dataPath.replace(".npy", "_fileids.npy"))
    fileIds = np.load(fileIdsPath)[: args.max_eval_rows] if fileIdsPath.exists() else None
    evalResults[Path(dataPath).stem] = summarize(row_losses(model, rows, args.eval_batch_size, device), fileIds)
    print(f"{Path(dataPath).stem}: loss {evalResults[Path(dataPath).stem]['loss']:.4f}")

results = {
    "args": vars(args),
    "device": device,
    "numTrainableParams": numTrainable,
    "trainLosses": trainLosses,
    "eval": evalResults,
    "trainSeconds": round(trainSeconds, 1),
    "totalSeconds": round(time.time() - startTime, 1),
    "peakGpuMemoryGB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
}
Path("results").mkdir(exist_ok=True)
Path(f"results/{args.run_name}.json").write_text(json.dumps(results, indent=2))
model.save_pretrained(f"checkpoints/finetuned/{args.run_name}")
print(f"saved results/{args.run_name}.json and checkpoints/finetuned/{args.run_name}/")
