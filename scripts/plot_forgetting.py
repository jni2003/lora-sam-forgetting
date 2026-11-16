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
parser.add_argument("--panels", action="store_true", help="two panels: full fine-tuning (left), LoRA (right); give --out")
args = parser.parse_args()
resultsDir = Path(args.results_dir)

baseDclm = {
    base: json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"]["dclm_heldout"]["loss"]
    for base in ["sam", "adamw"]
}
baseStarcoder = {  # base models' StarCoder val loss, to spot diverged runs
    base: json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"]["starcoder_val"]["loss"]
    for base in ["sam", "adamw"]
}

points = []  # (base, method, lr, fineTuneLoss, forgetting)
divergedRuns = []  # (base, method, lr) of runs that blew up: marked with an arrow in the --panels figure
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
    if fineTuneLoss > baseStarcoder[base]:
        print(f"skipping {path.name}: diverged (StarCoder loss {fineTuneLoss:.2f} > base {baseStarcoder[base]:.2f})")
        divergedRuns.append((base, "full" if rank is None else f"LoRA r{rank}", run["args"]["lr"]))
        continue  # worse on StarCoder than before fine-tuning = training blew up, not a point on the curve
    forgetting = run["eval"]["dclm_heldout"]["loss"] - baseDclm[base]
    points.append((base, method, run["args"]["lr"], fineTuneLoss, forgetting))

# Runs repeated with another seed: average them into one point (else two points at one lr make the line zigzag)
seedRuns = {}  # (base, method, lr) -> list of (fineTuneLoss, forgetting)
for base, method, lr, fineTuneLoss, forgetting in points:
    seedRuns.setdefault((base, method, lr), []).append((fineTuneLoss, forgetting))
points = [(base, method, lr, sum(r[0] for r in runs) / len(runs), sum(r[1] for r in runs) / len(runs), len(runs))
          for (base, method, lr), runs in seedRuns.items()]  # now (base, method, lr, fineTuneLoss, forgetting, numSeeds)

print(f"base DCLM loss: SAM {baseDclm['sam']:.4f}, AdamW {baseDclm['adamw']:.4f}")
print(f"{'base':6} {'method':9} {'lr':>8} {'ft loss':>8} {'forgetting':>10} {'seeds':>5}")
for p in sorted(points):
    print(f"{p[0]:6} {p[1]:9} {p[2]:>8.1e} {p[3]:>8.4f} {p[4]:>10.4f} {p[5]:>5}")

# Two-panel figure for the write-up: each panel gets its own axes, so the LoRA points aren't squeezed next to full FT
def drawPanel(ax, panelPoints, title, labelOffsets):
    baseColors = {"sam": "#2a78d6", "adamw": "#eb6834"}  # SAM blue, AdamW orange (passed the colorblind/contrast checks)
    methodStyles = {"full": ("o", "-"), "LoRA r8": ("s", "--"), "LoRA r64": ("^", "-")}  # marker, line per method
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)  # forgetting = 0 (no change from base)
    for base, method in sorted({(p[0], p[1]) for p in panelPoints}):
        group = sorted((p for p in panelPoints if p[0] == base and p[1] == method), key=lambda p: p[2])  # join by lr
        marker, line = methodStyles[method]
        ax.plot([p[3] for p in group], [p[4] for p in group], color=baseColors[base], marker=marker, linestyle=line,
                linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=1,
                label=f"{'SAM' if base == 'sam' else 'AdamW'}, {method}")
        for p in group:  # small gray lr labels next to each point
            ax.annotate(f"{p[2]:.0e}", (p[3], p[4]), textcoords="offset points", xytext=labelOffsets.get((base, method, p[2]), (5, 4)),
                        fontsize=7, color="#555555")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("fine-tuning loss (StarCoder val, lower = learned more)", fontsize=9)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.margins(x=0.12, y=0.1)  # room for the labels at the edges


# Diverged runs are far off the axes (fine-tuning loss ~6-7), so mark them instead: an arrow from the curve's
# last stable point, pointing out of the plot, labelled with the learning rate(s) that diverged
def markDiverged(ax, panelPoints, panelDiverged):
    baseColors = {"sam": "#2a78d6", "adamw": "#eb6834"}
    for base, method in sorted({(d[0], d[1]) for d in panelDiverged}):
        badLrs = sorted(d[2] for d in panelDiverged if d[0] == base and d[1] == method)
        stable = [p for p in panelPoints if p[0] == base and p[1] == method and p[2] < badLrs[0]]
        last = max(stable, key=lambda p: p[2])  # highest learning rate that still trained normally
        direction = (-40, 32) if base == "sam" else (30, 32)
        ax.annotate(f"diverged at {', '.join(f'{lr:.0e}' for lr in badLrs)}", xy=(last[3], last[4]),
                    xytext=direction, textcoords="offset points", fontsize=7, color=baseColors[base],
                    ha="center", arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=4, color=baseColors[base], linestyle="--", linewidth=1))


if args.panels:
    fig, (axFull, axLora) = plt.subplots(1, 2, figsize=(11, 4.5))
    # label nudges where two lr labels would overlap: (base, method, lr) -> (dx, dy) in points
    # label placement: SAM labels below-left of the point, AdamW above-right, so the two bases' labels don't collide
    labelOffsets = {}
    for base, method, lr, *_ in points:
        labelOffsets[(base, method, lr)] = (-30, -11) if base == "sam" else (5, 5)
    drawPanel(axFull, [p for p in points if p[1] == "full"], "Full fine-tuning", labelOffsets)
    drawPanel(axLora, [p for p in points if p[1] != "full"], "LoRA (rank 8 and rank 64)", labelOffsets)
    yBottom, yTop = axLora.get_ylim()
    axLora.set_ylim(yBottom, yTop + 0.35 * (yTop - yBottom))  # headroom for the "diverged" labels
    markDiverged(axLora, [p for p in points if p[1] != "full"], [d for d in divergedRuns if d[1] != "full"])
    axFull.set_ylabel("forgetting (change in DCLM loss, lower = forgot less)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")
else:
    colors = {"sam": "tab:blue", "adamw": "tab:orange"}
    markers = {"full": "o", "LoRA r8": "s", "LoRA r64": "^"}  # other ranks get "s"
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
