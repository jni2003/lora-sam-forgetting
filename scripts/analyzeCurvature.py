# Ai generated script to summarize curvature along intruders, other top directions, and random directions
"""Summarize results/curvature.json: is the pretraining loss flatter along SAM's intruders?

For each run, at each threshold ε (Shuttleworth et al. Algorithm 1: intruder if maxSim < ε):
  - κ (Watts et al. Eq. 4) of the intruders, of the other top-k directions, and of random
    directions in the same matrices of the base model;
  - a rough forgetting share of the intruders, ½ ‖uᵀΔW‖² κ summed over intruders (Watts et al.
    Eq. 3 restricted to each intruder's piece of the update), next to the run's measured forgetting.

Medians are reported next to means because κ can be negative and a few directions can dominate a mean.

Usage:  python -m scripts.analyzeCurvature --curvature results/curvature.json --eps 0.5 0.6 0.7
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--curvature", default="results/curvature.json")
parser.add_argument("--results-dir", default="results", help="holds <run>.json and base_eval_<base>.json")
parser.add_argument("--eps", type=float, nargs="+", default=[0.5, 0.6, 0.7])
args = parser.parse_args()
resultsDir = Path(args.results_dir)


def measuredForgetting(runName, base): # DCLM loss after fine-tuning minus the base model's, from the eval JSONs
    runLoss = json.loads((resultsDir / f"{runName}.json").read_text())["eval"]["dclm_heldout"]["loss"]
    baseLoss = json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"]["dclm_heldout"]["loss"]
    forgetting = runLoss - baseLoss
    return forgetting


def splitByThreshold(runData, eps): # (intruder entries, other top-k entries) across all matrices
    entries = [entry for matrixEntries in runData["matrices"].values() for entry in matrixEntries]
    intruders = [entry for entry in entries if entry["maxSim"] < eps]
    others = [entry for entry in entries if entry["maxSim"] >= eps]
    return intruders, others


def describeKappas(kappas): # "n=.., median .., mean .." for a list of κ, or n=0
    if not kappas:
        return "n=0"
    description = f"n={len(kappas)}, median {np.median(kappas):.3g}, mean {np.mean(kappas):.3g}"
    return description


def intruderForgettingShare(intruders): # Σ ½ ‖uᵀΔW‖² κ: Watts et al. Eq. 3 applied to each intruder's piece of the update
    # Approximate: κ is measured along u vᵀ, while the update's piece is u (uᵀΔW); the two match
    # when the intruder is nearly new (Δᵀu ≈ σ v). Cross terms between intruders are ignored.
    share = sum(0.5 * entry["updateNormAlongU"] ** 2 * entry["kappa"] for entry in intruders)
    return share


curvature = json.loads(Path(args.curvature).read_text())
print(f"settings: {curvature['settings']}\n")

# Random baseline per base model: curvature along arbitrary directions, for scale
for base, matrices in curvature["randomBaselines"].items():
    randomKappas = [kappa for kappas in matrices.values() for kappa in kappas]
    print(f"random directions, {base} base: {describeKappas(randomKappas)}")

# Per run: intruders vs other top directions, then the intruders' rough share of forgetting
for runName, runData in sorted(curvature["runs"].items()):
    print(f"\n{runName}   measured forgetting {measuredForgetting(runName, runData['base']):+.3f}")
    for eps in args.eps:
        intruders, others = splitByThreshold(runData, eps)
        print(f"  ε={eps}: intruders {describeKappas([e['kappa'] for e in intruders])}"
              f" | other top {describeKappas([e['kappa'] for e in others])}"
              f" | intruder share ≈ {intruderForgettingShare(intruders):+.3g}")

# Spot checks: the HVP's κ should match finite differences at the larger steps
for check in curvature["finiteDifferenceChecks"]:
    steps = ", ".join(f"{eps}: {value:.3g}" for eps, value in check["finiteDifference"].items())
    print(f"\nspot check {check['run']} {check['matrix']}: HVP κ {check['kappaHvp']:.3g} | finite differences {steps}")
