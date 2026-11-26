# AI generated script testing Watts et al.'s forgetting formula along each run's whole finetuning update
"""Watts et al.'s forgetting formula along each run's whole fine-tuning update ΔW.

Eq. 3: L(θ+Δ) - L(θ) ≈ gᵀΔ + ½ ΔᵀHΔ, with g ≈ 0 at a pretrained minimum, so forgetting ≈ ½ ΔᵀHΔ
= ½ ‖Δ‖² κ(Δ; H) (Eq. 4). H and g are of the DCLM loss at the base weights, on a fixed batch of rows.

For every stable run, two directions: Δ over every changed parameter (the whole update), and Δ
restricted to the 56 attention/MLP matrices (where intruders live, for comparison with their share).
Saved per direction: ‖Δ‖, κ, gᵀΔ, ½ΔᵀHΔ, the predicted change gᵀΔ + ½ΔᵀHΔ, the exact change
L(θ+Δ) - L(θ) on the same rows, and finite differences of ΔᵀHΔ as a check of the HVP.
Measured forgetting (20M-token DCLM eval) is copied from results/<run>.json.

Per parameter group (embeddings, output layer, norms, attention/MLP): gᵀΔ and the exact change from
moving only that group. Tests whether the offset (g ≠ 0 on our DCLM data) is absorbed mostly in
parameters LoRA can't touch, which would explain full fine-tuning's lower forgetting at low learning rates.

Usage (cluster):  python -u -m scripts.curvatureAlongUpdate --runs checkpoints/finetuned/* --out results/curvatureUpdate.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

from src.convert import load_olmo_as_llama
from src.curvature import (directionalCurvature, directionalDerivative, finiteDifferenceCurvature, lossChange,
                           makeChunkedLanguageModelLosses, meanOverChunks)

MATRIX_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # same matrices as the intruders
PARAMETER_GROUPS = {"embeddings": ["embed_tokens"], "outputLayer": ["lm_head"], "norms": ["norm"],
                    "attentionAndMlp": MATRIX_TYPES}  # every weight matches exactly one group's keys
FINITE_DIFFERENCE_STEPS = [0.1, 0.03]  # fractions of Δ; the whole step (1.0) is the exact loss change, reported separately

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True, help="fine-tuned checkpoint folders; diverged runs are skipped")
parser.add_argument("--base-dir", default="checkpoints", help="folder holding olmo-60m-{sam,adamw}")
parser.add_argument("--results-dir", default="results", help="holds <run>.json and base_eval_<base>.json")
parser.add_argument("--data", default="data/dclm_heldout.npy", help="DCLM rows for the pretraining loss")
parser.add_argument("--rows", type=int, default=8, help="rows of 1024 tokens; same batch as the intruder curvature run")
parser.add_argument("--rows-per-chunk", type=int, default=1, help="rows processed at once; results are averaged (exact)")
parser.add_argument("--out", default="results/curvatureUpdate.json")
args = parser.parse_args()
resultsDir = Path(args.results_dir)


def loadBaseModel(base, device): # base model ready for second derivatives: eager attention, float32, on device
    model = load_olmo_as_llama(f"{args.base_dir}/olmo-60m-{base}")
    model.set_attn_implementation("eager")  # fused attention kernels may not support the second backward pass
    return model.to(device)


def runMetadata(runName, base): # (measured forgetting, is the run diverged?) from the eval JSONs
    run = json.loads((resultsDir / f"{runName}.json").read_text())
    baseEval = json.loads((resultsDir / f"base_eval_{base}.json").read_text())["datasets"]
    forgetting = run["eval"]["dclm_heldout"]["loss"] - baseEval["dclm_heldout"]["loss"]
    # Diverged = worse on StarCoder than before fine-tuning (same rule as the plots). Only for StarCoder
    # runs: the DCLM checks never trained on code, so a worse StarCoder loss is expected there, not divergence.
    isTrainedOnStarcoder = "starcoder" in run["args"]["train_data"]
    isDiverged = isTrainedOnStarcoder and run["eval"]["starcoder_val"]["loss"] > baseEval["starcoder_val"]["loss"]
    return forgetting, isDiverged


def updateDirection(model, runDir): # {name: ΔW} for every parameter fine-tuning changed, on the model's device
    tunedWeights = LlamaForCausalLM.from_pretrained(runDir).state_dict()
    update = {}
    for name, baseWeight in model.named_parameters():
        change = tunedWeights[name].to(baseWeight.device) - baseWeight.detach()
        if change.abs().max() > 0:  # LoRA leaves embeddings, norms and the output layer untouched
            update[name] = change
    return update


def measureDirection(model, lossChunks, direction): # Eq. 3's terms along one direction, averaged over row chunks (exact)
    meanOf = lambda measure: meanOverChunks(measure, lossChunks)
    dHd = meanOf(lambda computeLoss: directionalCurvature(computeLoss, model, direction)[0])
    gradientTerm = meanOf(lambda computeLoss: directionalDerivative(computeLoss, model, direction))
    squaredNorm = sum((part ** 2).sum().item() for part in direction.values())
    summary = {
        "updateNorm": squaredNorm ** 0.5,
        "kappa": dHd / squaredNorm,
        "gradientTerm": gradientTerm,
        "halfDHd": 0.5 * dHd,  # Watts et al.'s forgetting estimate (g ≈ 0)
        "predictedChange": gradientTerm + 0.5 * dHd,
        "exactChangeOnRows": meanOf(lambda computeLoss: lossChange(computeLoss, model, direction)),
        "finiteDifferenceDHd": {str(eps): meanOf(lambda computeLoss: finiteDifferenceCurvature(computeLoss, model, direction, eps))
                                for eps in FINITE_DIFFERENCE_STEPS},
        "dHd": dHd,
    }
    return summary


def groupOf(name): # the one parameter group a weight belongs to, e.g. 'norms' for '...input_layernorm.weight'
    matches = [group for group, keys in PARAMETER_GROUPS.items() if any(key in name for key in keys)]
    if len(matches) != 1:
        raise ValueError(f"{name} matches groups {matches}; PARAMETER_GROUPS needs exactly one")
    return matches[0]


def measureGroups(model, lossChunks, update, wholeGradientTerm): # {group: ‖Δ‖, gᵀΔ, exact change moving only that group}
    meanOf = lambda measure: meanOverChunks(measure, lossChunks)
    groups = {}
    for group in PARAMETER_GROUPS:
        groupUpdate = {name: part for name, part in update.items() if groupOf(name) == group}
        if not groupUpdate:
            continue  # e.g. LoRA never changes the embeddings
        groups[group] = {
            "updateNorm": sum((part ** 2).sum().item() for part in groupUpdate.values()) ** 0.5,
            "gradientTerm": meanOf(lambda computeLoss: directionalDerivative(computeLoss, model, groupUpdate)),
            "exactChangeOnRows": meanOf(lambda computeLoss: lossChange(computeLoss, model, groupUpdate)),
        }

    # gᵀΔ is linear in Δ, so the groups' terms must add up to the whole update's: a check that no weight was dropped
    groupTotal = sum(summary["gradientTerm"] for summary in groups.values())
    if abs(groupTotal - wholeGradientTerm) > 1e-3 * abs(wholeGradientTerm) + 1e-6:
        raise ValueError(f"group gradient terms add to {groupTotal}, whole update has {wholeGradientTerm}")
    return groups


device = "cuda" if torch.cuda.is_available() else "cpu"
inputIds = torch.from_numpy(np.load(args.data)[: args.rows].astype(np.int64)).to(device)
lossChunks = makeChunkedLanguageModelLosses(inputIds, args.rows_per_chunk)  # pretraining (DCLM) loss: H is its Hessian

runsByBase = {}  # base -> run folders, so each base model is loaded once
for runDir in sorted(args.runs):
    runsByBase.setdefault(Path(runDir).name.split("_")[0], []).append(runDir)  # "sam" or "adamw" prefix

startTime = time.time()
results = {"settings": {"rows": args.rows, "rowsPerChunk": args.rows_per_chunk, "data": args.data}, "runs": {}, "skipped": []}
for base, runDirs in runsByBase.items():
    model = loadBaseModel(base, device)
    for runDir in runDirs:
        runName = Path(runDir).name
        forgetting, isDiverged = runMetadata(runName, base)
        if isDiverged:
            results["skipped"].append(runName)  # a broken model is far outside the quadratic regime
            continue

        # Whole update, then only the attention/MLP matrices (for comparison with the intruders' share)
        print(f"run {runName}: whole update and attention/MLP part ...")
        update = updateDirection(model, runDir)
        blockUpdate = {name: part for name, part in update.items() if any(f".{t}.weight" in name for t in MATRIX_TYPES)}
        whole = measureDirection(model, lossChunks, update)
        blocks = measureDirection(model, lossChunks, blockUpdate)
        byGroup = measureGroups(model, lossChunks, update, whole["gradientTerm"])
        results["runs"][runName] = {"base": base, "measuredForgetting": forgetting, "numChangedTensors": len(update),
                                    "wholeUpdate": whole, "attentionAndMlpOnly": blocks, "byGroup": byGroup}
        print(f"  ‖Δ‖ {whole['updateNorm']:.3g}, κ {whole['kappa']:.3g}, ½ΔᵀHΔ {whole['halfDHd']:+.3g}, "
              f"gᵀΔ {whole['gradientTerm']:+.3g}, predicted {whole['predictedChange']:+.3g}, "
              f"exact on rows {whole['exactChangeOnRows']:+.3g}, measured forgetting {forgetting:+.3g}")
        print("  by group (gᵀΔ / exact change moving only that group): " +
              ", ".join(f"{group} {s['gradientTerm']:+.3g} / {s['exactChangeOnRows']:+.3g}" for group, s in byGroup.items()))
    del model

results["settings"]["minutes"] = (time.time() - startTime) / 60
results["settings"]["peakGpuMemoryGB"] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
Path(args.out).write_text(json.dumps(results, indent=1))
print(f"done in {results['settings']['minutes']:.1f} min; skipped (diverged): {results['skipped']}; saved {args.out}")
