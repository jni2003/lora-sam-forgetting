# AI Generated Script to plot forgetting vs finetuning loss
"""Plot forgetting vs. fine-tuning loss (Watts et al.-style), from the results JSON files.

Each fine-tuning run is one point:
    x = fine-tuning loss    (loss on StarCoder validation rows; lower = learned the task better)
    y = forgetting          (run's DCLM loss minus its base model's DCLM loss; lower = forgot less)
Colour = base model (SAM vs AdamW), marker = method (full fine-tuning vs LoRA rank), and points of
the same base + method are joined in order of learning rate. Compare models at the same x
("matched fine-tuning loss"): the curve that sits lower forgets less for the same learning.

Reads results/base_eval_{sam,adamw}.json (scripts/slurm/eval_base.sbatch) and every other
results/*.json written by scripts/train.py. Also prints the numbers as a table.

Usage:  python -m scripts.plot_forgetting            # -> results/forgetting_vs_finetune_loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file; no window needed (works on the cluster too)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results")
parser.add_argument("--out", default="results/forgetting_vs_finetune_loss.png")
parser.add_argument("--no-lora", action="store_true", help="plot full fine-tuning only")
args = parser.parse_args()
resultsDir = Path(args.results_dir)

baseDclm = {
    base: json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"]["dclm_heldout"]["loss"]
    for base in ["sam", "adamw"]
}

points = []  # (base, method, lr, fineTuneLoss, forgetting)
for path in sorted(resultsDir.glob("*.json")):
    run = json.loads(path.read_text())
    if "trainLosses" not in run:
        continue  # not a fine-tuning run (e.g. a base eval)
    if "starcoder" not in run["args"]["train_data"]:
        continue  # a control trained on other data (e.g. the DCLM cooldown control), not a StarCoder point
    base = "sam" if "sam" in run["args"]["model"] else "adamw"
    rank = run["args"]["lora_rank"]
    if args.no_lora and rank is not None:
        continue
    method = "full" if rank is None else f"LoRA r{rank}"
    fineTuneLoss = run["eval"]["starcoder_val"]["loss"]
    forgetting = run["eval"]["dclm_heldout"]["loss"] - baseDclm[base]
    points.append((base, method, run["args"]["lr"], fineTuneLoss, forgetting))

print(f"base DCLM loss: SAM {baseDclm['sam']:.4f}, AdamW {baseDclm['adamw']:.4f}")
print(f"{'base':6} {'method':9} {'lr':>8} {'ft loss':>8} {'forgetting':>10}")
for p in sorted(points):
    print(f"{p[0]:6} {p[1]:9} {p[2]:>8.1e} {p[3]:>8.4f} {p[4]:>10.4f}")

colors = {"sam": "tab:blue", "adamw": "tab:orange"}
markers = {"full": "o"}  # LoRA ranks get "s" (square)
fig, ax = plt.subplots(figsize=(6, 4.5))
for base, method in sorted({(p[0], p[1]) for p in points}):
    group = sorted((p for p in points if p[0] == base and p[1] == method), key=lambda p: p[2])  # join in order of learning rate
    ax.plot([p[3] for p in group], [p[4] for p in group], color=colors[base],
            marker=markers.get(method, "s"), linestyle="-" if method == "full" else "--",
            label=f"{'SAM' if base == 'sam' else 'AdamW'} base, {method}")
    for p in group:
        ax.annotate(f"{p[2]:.0e}", (p[3], p[4]), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set_xlabel("fine-tuning loss (StarCoder val; lower = learned more)")
ax.set_ylabel("forgetting (Δ DCLM loss; lower = forgot less)")
ax.set_title("Forgetting vs. fine-tuning loss (60M, StarCoder)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"saved {args.out}")
