#AI Generated script to plot intruder count vs forgetting
"""Plot intruder count vs. forgetting: one point per fine-tuning run.

x = number of intruders (out of 560 checked vectors) at one threshold eps
y = forgetting (run's DCLM loss minus its base model's DCLM loss)
Colour = base model, marker = method. Shows whether intruder count tracks forgetting
within a method, and whether it does across methods.

Reads results/intruders.json (scripts/countIntruders.py) and the per-run results JSONs.
Skips diverged runs and controls (same rules as scripts/plot_forgetting.py); seed reruns are averaged.

Usage:  python -m scripts.plotIntruders --eps 0.6 --out results/intruders_vs_forgetting.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results")
parser.add_argument("--eps", type=float, default=0.6, help="intruder threshold")
parser.add_argument("--out", default="results/intruders_vs_forgetting.png")
args = parser.parse_args()
resultsDir = Path(args.results_dir)

intruderRuns = json.loads((resultsDir / "intruders.json").read_text())["runs"]
baseEval = {base: json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"] for base in ["sam", "adamw"]}

# Collect (base, method, lr) -> list of (intruders, forgetting), one entry per seed
runs = {}
for runName, intruderData in intruderRuns.items():
    run = json.loads((resultsDir / f"{runName}.json").read_text())
    if "starcoder" not in run["args"]["train_data"]:
        continue  # DCLM control, not a StarCoder run
    base = intruderData["base"]
    if run["eval"]["starcoder_val"]["loss"] > baseEval[base]["starcoder_val"]["loss"]:
        continue  # diverged
    rank = run["args"]["lora_rank"]
    method = "full" if rank is None else f"LoRA r{rank}"
    maxSims = [s for matrix in intruderData["matrices"].values() for s in matrix["maxSims"]]
    intruders = sum(s < args.eps for s in maxSims)
    forgetting = run["eval"]["dclm_heldout"]["loss"] - baseEval[base]["dclm_heldout"]["loss"]
    runs.setdefault((base, method, run["args"]["lr"]), []).append((intruders, forgetting))

# Average seed reruns into one point
points = [(base, method, lr, sum(r[0] for r in rs) / len(rs), sum(r[1] for r in rs) / len(rs))
          for (base, method, lr), rs in runs.items()]

baseColors = {"sam": "#2a78d6", "adamw": "#eb6834"}  # same colours as the forgetting figure
methodMarkers = {"full": "o", "LoRA r8": "s", "LoRA r64": "^"}

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)  # no forgetting
for base in ["sam", "adamw"]:
    for method in ["full", "LoRA r8", "LoRA r64"]:
        group = sorted((p for p in points if p[0] == base and p[1] == method), key=lambda p: p[2])
        ax.plot([p[3] for p in group], [p[4] for p in group], color=baseColors[base], linewidth=1,
                alpha=0.4, linestyle="-" if method == "full" else "--", zorder=1)  # thin line in lr order, to see each curve
        ax.scatter([p[3] for p in group], [p[4] for p in group], color=baseColors[base], marker=methodMarkers[method],
                   s=50, edgecolor="white", linewidth=1, label=f"{'SAM' if base == 'sam' else 'AdamW'}, {method}")
ax.set_xlabel(f"intruder dimensions (of 560, ε = {args.eps})", fontsize=9)
ax.set_ylabel("forgetting (change in DCLM loss, lower = forgot less)", fontsize=9)
ax.grid(color="#e5e5e5", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"saved {args.out}")

for p in sorted(points):
    print(f"{p[0]:6} {p[1]:9} {p[2]:>8.1e} intruders {p[3]:6.1f}  forgetting {p[4]:+.3f}")
